"""Generate offline Shields endpoints; see ../BADGES.md for the input contract."""

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from xml.etree import ElementTree

from check_junit import check_report
from databricks_compat import RUNTIMES, TESTS

SCOPE = "equivalent-cpu-dependencies-local-spark"
IDENTIFIERS = ("commit_sha", "actions_run_id", "actions_run_attempt")


def mapping(value: object) -> dict[str, object]:
    """Narrow JSON objects at the decoder boundary (stdlib returns Any)."""
    if not isinstance(value, dict):
        raise TypeError("Expected a JSON object")
    # JSON object keys are strings; callers also use this for nested JSON objects.
    return cast(dict[str, object], value)


def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def invalid_constant(value: str) -> object:
    raise ValueError(f"Non-finite JSON number: {value}")


def read_json(path: Path) -> dict[str, object]:
    return mapping(
        json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=invalid_constant,
        )
    )


def integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Expected a nonnegative integer, got {value!r}")
    return value


def percentage(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Coverage percentage must be numeric")
    if not 0 <= value <= 100 or not math.isfinite(value):
        raise ValueError("Coverage percentage must be finite and between 0 and 100")
    return float(value)


def timestamp(value: object) -> None:
    if not isinstance(value, str) or datetime.fromisoformat(value).utcoffset() is None:
        raise ValueError("Expected an ISO 8601 timestamp with timezone")


def validate_metadata(metadata: dict[str, object]) -> None:
    for key in IDENTIFIERS:
        value = metadata.get(key)
        pattern = r"[0-9a-f]{40}" if key == "commit_sha" else r"[1-9][0-9]*"
        if value is not None and (
            not isinstance(value, str) or re.fullmatch(pattern, value) is None
        ):
            raise ValueError(f"Invalid {key}")
    if metadata.get("consolidated_at") is not None:
        timestamp(metadata["consolidated_at"])
    url = metadata.get("actions_run_url")
    if url is not None:
        if not isinstance(url, str):
            raise ValueError("Invalid Actions URL")
        match = re.fullmatch(
            r"https://github\.com/[^/]+/[^/]+/actions/runs/([1-9][0-9]*)", url
        )
        if match is None or match[1] != metadata.get("actions_run_id"):
            raise ValueError("Actions URL must match actions_run_id")


def match_metadata(report: dict[str, object], seen: dict[str, object]) -> None:
    validate_metadata(report)
    for key in IDENTIFIERS:
        value = report.get(key)
        if value is not None:
            if seen.get(key) is not None and seen[key] != value:
                raise ValueError(f"Inconsistent execution metadata: {key}")
            seen[key] = value


def coverage_value(report: dict[str, object]) -> float:
    meta = mapping(report.get("meta"))
    if meta.get("branch_coverage") is not True or meta.get("format") != 3:
        raise ValueError("Expected coverage.py JSON format 3 with branch coverage")
    files = mapping(report.get("files"))
    if not files or any(
        not name.replace("\\", "/").startswith("src/tyrannis/")
        or ".." in name.replace("\\", "/").split("/")
        for name in files
    ):
        raise ValueError("Coverage must contain only src/tyrannis files")
    totals = mapping(report.get("totals"))
    measured = percentage(totals.get("percent_covered"))
    lines = integer(totals.get("num_statements"))
    branches = integer(totals.get("num_branches"))
    covered_lines = integer(totals.get("covered_lines"))
    covered_branches = integer(totals.get("covered_branches"))
    if (
        covered_lines + integer(totals.get("missing_lines")) != lines
        or covered_branches + integer(totals.get("missing_branches")) != branches
        or integer(totals.get("num_partial_branches"))
        > integer(totals.get("missing_branches"))
    ):
        raise ValueError("Inconsistent coverage counters")
    # Validate aggregate consistency only; the displayed metric remains the supplied total.
    expected = (
        100 * (covered_lines + covered_branches) / (lines + branches)
        if lines + branches
        else 100
    )
    if not math.isclose(measured, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Coverage percentage contradicts aggregate counters")
    for key, covered, total in (
        ("percent_statements_covered", covered_lines, lines),
        ("percent_branches_covered", covered_branches, branches),
    ):
        if key in totals and not math.isclose(
            percentage(totals[key]),
            100 * covered / total if total else 100,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Inconsistent {key}")
    return measured


def coverage_evidence(
    path: Path, metadata: dict[str, object], seen: dict[str, object]
) -> dict[str, object]:
    result: dict[str, object] = {"source": str(path), "state": "unverified"}
    if not path.exists():
        result["reason"] = "Coverage report missing"
        return result
    report = read_json(path)
    match_metadata(report, seen)
    result["percent_covered"] = coverage_value(report)
    scope = metadata.get("coverage_scope")
    result["scope"] = scope
    result["scope_basis"] = "caller-declared; coverage.py does not record OS or suite"
    if scope != {"python": "3.12", "os": "ubuntu-24.04", "suite": "not optional"}:
        result["reason"] = "Coverage scope absent or outside the Task 1 scope"
    else:
        result["state"] = "approved"
    return result


def runtime_evidence(
    runtime: str, path: Path, junit: Path, seen: dict[str, object]
) -> dict[str, object]:
    result: dict[str, object] = {
        "runtime": f"{runtime} LTS ML",
        "source": str(path),
        "junit": str(junit),
        "state": "unverified",
    }
    if not path.exists():
        result["reason"] = "Compatibility report missing"
        return result
    report = read_json(path)
    match_metadata(report, seen)
    if (
        report.get("runtime") != result["runtime"]
        or type(report.get("schema_version")) is not int
        or report["schema_version"] != 1
    ):
        raise ValueError(f"Invalid runtime identity/schema: {path}")
    result["evidence"] = report
    if report.get("result") not in {"passed", "failed"}:
        raise ValueError(f"Invalid runtime result: {path}")
    if "validated_at" in report:
        timestamp(report["validated_at"])
    required = (
        "validation_scope",
        "expected_python",
        "python_version",
        "expected_dependencies",
        "installed_dependencies",
        "dependency_check",
        "test_targets",
        "validated_at",
    )
    if any(report.get(key) is None for key in required):
        result["reason"] = "Required compatibility evidence missing"
        return result
    if (
        report["validation_scope"] != SCOPE
        or report["expected_python"] != "3.12"
        or report["expected_dependencies"] != RUNTIMES[runtime]
        or report["test_targets"] != list(TESTS)
    ):
        raise ValueError(
            f"Compatibility evidence contradicts the tested matrix: {path}"
        )
    python = report["python_version"]
    if not isinstance(python, str) or not re.fullmatch(
        r"[0-9]+\.[0-9]+\.[0-9]+", python
    ):
        raise ValueError("Invalid installed Python version")
    installed = mapping(report["installed_dependencies"])
    if set(installed) != set(RUNTIMES[runtime]):
        result["reason"] = "Dependency evidence incomplete"
        return result
    if report["dependency_check"] not in {"passed", "not_passed"}:
        raise ValueError("Invalid dependency_check")
    dependencies_ok = installed == RUNTIMES[runtime] and python.startswith("3.12.")
    if not dependencies_ok or report["dependency_check"] != "passed":
        if report["result"] == "passed" or report["dependency_check"] == "passed":
            raise ValueError("Success contradicts installed dependencies")
        result.update(state="failed", reason="Dependency validation failed")
        return result
    if not junit.exists():
        result["reason"] = "JUnit report missing"
        return result
    root = ElementTree.parse(junit).getroot()
    cases = list(root.iter("testcase"))
    skipped = sum(case.find("skipped") is not None for case in cases)
    if (
        report.get("tests_executed") is None
        or report.get("tests_skipped") is None
        or report.get("pytest_exit_code") is None
    ):
        result["reason"] = "Test execution evidence incomplete"
        return result
    executed = integer(report["tests_executed"])
    if executed != len(cases) - skipped or integer(report["tests_skipped"]) != skipped:
        raise ValueError("JSON and JUnit counts disagree")
    code = integer(report["pytest_exit_code"])
    try:
        _ = check_report(junit)
    except ValueError as exc:
        if report["result"] == "passed":
            raise ValueError(f"Success contradicts JUnit: {exc}") from exc
        result["state"] = (
            "failed"
            if any(node.tag in {"failure", "error", "skipped"} for node in root.iter())
            else "unverified"
        )
        result["reason"] = str(exc)
        return result
    classes = {case.get("classname", "") for case in cases}
    if not all(
        any(
            name == target[:-3].replace("/", ".")
            or name.startswith(target[:-3].replace("/", ".") + ".")
            for name in classes
        )
        for target in TESTS
    ):
        result["reason"] = "JUnit does not demonstrate every required Spark test module"
        return result
    if report["result"] == "passed" and (code != 0 or report.get("error")):
        raise ValueError("Success contradicts exit status/error")
    result["state"] = (
        "approved" if code == 0 and report["result"] == "passed" else "failed"
    )
    return result


def endpoint(label: str, message: str, color: str) -> dict[str, object]:
    return {"schemaVersion": 1, "label": label, "message": message, "color": color}


def generate(
    coverage: Path, reports: list[tuple[str, Path, Path]], metadata: dict[str, object]
) -> tuple[dict[str, object], int]:
    """Validate all inputs before returning any serializable output."""
    validate_metadata(metadata)
    seen = {key: metadata.get(key) for key in IDENTIFIERS}
    indexed: dict[str, tuple[Path, Path]] = {}
    for runtime, report, junit in reports:
        if runtime not in RUNTIMES or runtime in indexed:
            raise ValueError(f"Unknown or duplicate runtime: {runtime}")
        indexed[runtime] = (report, junit)
    cov = coverage_evidence(coverage, metadata, seen)
    runtimes = [
        runtime_evidence(runtime, *indexed[runtime], seen)
        if runtime in indexed
        else {
            "runtime": f"{runtime} LTS ML",
            "state": "unverified",
            "reason": "No report supplied",
        }
        for runtime in sorted(
            RUNTIMES, key=lambda name: tuple(map(int, name.split(".")))
        )
    ]
    if cov["state"] == "approved":
        value = percentage(cov["percent_covered"])
        color = next(
            color
            for minimum, color in (
                (90, "brightgreen"),
                (80, "green"),
                (70, "yellowgreen"),
                (60, "yellow"),
                (0, "red"),
            )
            if value >= minimum
        )
        coverage_badge = endpoint("coverage", f"{value:.0f}%", color)
    else:
        coverage_badge = endpoint("coverage", "not verified", "lightgrey")
    approved = [
        str(item["runtime"]).removesuffix(" LTS ML")
        for item in runtimes
        if item["state"] == "approved"
    ]
    states = [cov["state"], *(item["state"] for item in runtimes)]
    overall = (
        "failed"
        if "failed" in states
        else "unverified"
        if "unverified" in states
        else "approved"
    )
    badge = endpoint(
        "Databricks LTS ML",
        " | ".join(approved)
        if approved
        else "failed"
        if any(item["state"] == "failed" for item in runtimes)
        else "not verified",
        "green"
        if approved
        else "red"
        if any(item["state"] == "failed" for item in runtimes)
        else "lightgrey",
    )
    missing = [
        key
        for key in (*IDENTIFIERS, "actions_run_url", "consolidated_at")
        if metadata.get(key) is None
    ]
    unbound = [
        str(item["runtime"])
        for item in runtimes
        if "evidence" in item
        and any(mapping(item["evidence"]).get(key) is None for key in IDENTIFIERS)
    ]
    provenance: dict[str, object] = {
        "schema_version": 1,
        "metadata": metadata,
        "observed_identifiers": seen,
        "coverage": cov,
        "databricks": runtimes,
        "evidence_state": overall,
        "publication": {
            "missing_metadata": missing,
            "reports_without_execution_identity": unbound,
            "requires_workflow_verification": True,
        },
    }
    return {
        "coverage.json": coverage_badge,
        "databricks-lts-ml.json": badge,
        "provenance.json": provenance,
    }, int(overall != "approved" or bool(missing) or bool(unbound))


def write_outputs(
    output: Path, inputs: list[Path], documents: dict[str, object]
) -> None:
    """Stage the entire bundle, then rename it; never overwrite an earlier run."""
    destination = output.resolve()
    if output.exists() or any(
        destination.is_relative_to(path.resolve().parent)
        or path.resolve().is_relative_to(destination)
        for path in inputs
    ):
        raise ValueError("Output must be new and independent of all input directories")
    serialized = {
        name: json.dumps(
            value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
        )
        + "\n"
        for name, value in documents.items()
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".badges-", dir=output.parent) as temporary:
        bundle = Path(temporary) / "bundle"
        bundle.mkdir()
        for name, content in serialized.items():
            _ = (bundle / name).write_text(content, encoding="utf-8")
        bundle.rename(output)


class Options(argparse.Namespace):
    coverage: Path
    metadata: Path
    output: Path
    databricks: list[list[str]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--coverage", type=Path, required=True)
    _ = parser.add_argument("--metadata", type=Path, required=True)
    _ = parser.add_argument("--output", type=Path, required=True)
    _ = parser.add_argument(
        "--databricks",
        nargs=3,
        action="append",
        default=[],
        metavar=("RUNTIME", "JSON", "JUNIT"),
    )
    args = parser.parse_args(namespace=Options())
    reports = [
        (runtime, Path(report), Path(junit))
        for runtime, report, junit in args.databricks
    ]
    try:
        documents, code = generate(args.coverage, reports, read_json(args.metadata))
        write_outputs(
            args.output,
            [
                args.coverage,
                args.metadata,
                *(path for _, report, junit in reports for path in (report, junit)),
            ],
            documents,
        )
    except (OSError, TypeError, ValueError, ElementTree.ParseError) as exc:
        print(f"Invalid badge inputs: {exc}", file=sys.stderr)
        return 2
    if code:
        print(
            "Non-affirmative or publication-incomplete evidence; see provenance.json",
            file=sys.stderr,
        )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
