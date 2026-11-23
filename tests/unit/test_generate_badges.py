"""Offline CLI contracts using synthetic evidence, never runtime certification."""

import json
import subprocess
import sys
from pathlib import Path
from typing import cast
from xml.etree import ElementTree

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / ".github/scripts/generate_badges.py"
TARGETS = [
    "tests/unit/backend/test_spark_parallel.py",
    "tests/unit/backend/test_spark_distributed.py",
    "tests/unit/backend/test_spark_communication.py",
    "tests/integration/backend/test_spark_parallel.py",
    "tests/integration/backend/test_spark_distributed.py",
]


def read(path: Path) -> dict[str, object]:
    # stdlib JSON has no typed decoding API; these files are objects we construct.
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def write(path: Path, data: dict[str, object]) -> None:
    _ = path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def evidence(tmp_path: Path) -> Path:
    """CI-specific documents have no equivalent in the optimizer test helpers."""
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    metadata: dict[str, object] = {
        "commit_sha": "a" * 40,
        "actions_run_id": "123",
        "actions_run_attempt": "1",
        "actions_run_url": "https://github.com/example/project/actions/runs/123",
        "consolidated_at": "2026-11-23T12:00:00+00:00",
        "coverage_scope": {
            "python": "3.12",
            "os": "ubuntu-24.04",
            "suite": "not optional",
        },
    }
    write(inputs / "metadata.json", metadata)
    write(
        inputs / "coverage.json",
        {
            "meta": {"format": 3, "branch_coverage": True},
            "files": {"src/tyrannis/__init__.py": {}},
            "totals": {
                "num_statements": 9,
                "covered_lines": 8,
                "missing_lines": 1,
                "num_branches": 3,
                "covered_branches": 2,
                "missing_branches": 1,
                "num_partial_branches": 1,
                "percent_covered": 100 * 10 / 12,
            },
        },
    )
    for runtime in ("17.3", "18"):
        dependencies = {
            "pyspark": "4.0.0" if runtime == "17.3" else "4.1.0",
            "numpy": "2.1.3",
            "scipy": "1.15.1" if runtime == "17.3" else "1.15.3",
            "cloudpickle": "3.0.0",
            "joblib": "1.4.2",
            "pandas": "2.2.3",
            "pyarrow": "19.0.1" if runtime == "17.3" else "21.0.0",
        }
        write(
            inputs / f"{runtime}.json",
            {
                "schema_version": 1,
                "runtime": f"{runtime} LTS ML",
                "validation_scope": "equivalent-cpu-dependencies-local-spark",
                "expected_python": "3.12",
                "python_version": "3.12.13",
                "expected_dependencies": dependencies,
                "installed_dependencies": dependencies,
                "dependency_check": "passed",
                "result": "passed",
                "pytest_exit_code": 0,
                "tests_executed": 5,
                "tests_skipped": 0,
                "test_targets": TARGETS,
                "tyrannis_version": "1.0.0",
                "validated_at": "2026-11-23T11:00:00+00:00",
                **{
                    key: metadata[key]
                    for key in ("commit_sha", "actions_run_id", "actions_run_attempt")
                },
            },
        )
        suite = ElementTree.Element(
            "testsuite", tests="5", skipped="0", failures="0", errors="0"
        )
        for target in TARGETS:
            _ = ElementTree.SubElement(
                suite,
                "testcase",
                classname=target[:-3].replace("/", "."),
                name="test_synthetic",
            )
        ElementTree.ElementTree(suite).write(
            inputs / f"{runtime}.xml", encoding="utf-8"
        )
    return inputs


def run(
    inputs: Path, output: Path | None = None, extra: tuple[str, ...] = ()
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--coverage",
            str(inputs / "coverage.json"),
            "--metadata",
            str(inputs / "metadata.json"),
            "--output",
            str(output or inputs.parent / "output"),
            "--databricks",
            "17.3",
            str(inputs / "17.3.json"),
            str(inputs / "17.3.xml"),
            "--databricks",
            "18",
            str(inputs / "18.json"),
            str(inputs / "18.xml"),
            *extra,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_generates_precise_traceable_endpoints_deterministically(
    evidence: Path,
) -> None:
    result = run(evidence)
    assert result.returncode == 0, result.stderr
    output = evidence.parent / "output"
    assert read(output / "coverage.json") == {
        "schemaVersion": 1,
        "label": "coverage",
        "message": "83%",
        "color": "green",
    }
    assert read(output / "databricks-lts-ml.json") == {
        "schemaVersion": 1,
        "label": "Databricks LTS ML",
        "message": "17.3 | 18",
        "color": "green",
    }
    provenance = read(output / "provenance.json")
    assert provenance["metadata"] == read(evidence / "metadata.json")
    assert (
        cast(dict[str, object], provenance["coverage"])["percent_covered"]
        == 100 * 10 / 12
    )
    assert provenance["evidence_state"] == "approved"
    second = evidence.parent / "second"
    assert run(evidence, second).returncode == 0
    assert {path.name: path.read_bytes() for path in output.iterdir()} == {
        path.name: path.read_bytes() for path in second.iterdir()
    }


@pytest.mark.parametrize("value", ["85", True, float("nan"), -1, 101, 75])
def test_rejects_invalid_or_inconsistent_global_metric(
    evidence: Path, value: object
) -> None:
    path = evidence / "coverage.json"
    report = read(path)
    cast(dict[str, object], report["totals"])["percent_covered"] = value
    write(path, report)
    assert run(evidence).returncode == 2
    assert not (evidence.parent / "output").exists()


@pytest.mark.parametrize("outcome", ["failure", "skipped", "dependencies", "empty"])
def test_excludes_failed_or_unexecuted_runtime(evidence: Path, outcome: str) -> None:
    path = evidence / "18.json"
    report = read(path)
    report["result"] = "failed"
    xml = evidence / "18.xml"
    tree = ElementTree.parse(xml)
    root = tree.getroot()
    if outcome == "dependencies":
        report["installed_dependencies"] = {
            **cast(dict[str, object], report["installed_dependencies"]),
            "numpy": "1.0",
        }
        report["dependency_check"] = "not_passed"
        xml.unlink()
    elif outcome == "empty":
        root.clear()
        root.attrib.update(tests="0", skipped="0", failures="0", errors="0")
        report.update(tests_executed=0, pytest_exit_code=5)
    else:
        _ = ElementTree.SubElement(root[0], outcome)
        root.set("failures" if outcome == "failure" else "skipped", "1")
        report["pytest_exit_code"] = 1 if outcome == "failure" else 0
        if outcome == "skipped":
            report.update(tests_executed=4, tests_skipped=1)
    if outcome != "dependencies":
        tree.write(xml, encoding="utf-8")
    write(path, report)
    assert run(evidence).returncode == 1
    output = evidence.parent / "output"
    assert read(output / "databricks-lts-ml.json")["message"] == "17.3"
    assert read(output / "provenance.json")["evidence_state"] == (
        "unverified" if outcome == "empty" else "failed"
    )


@pytest.mark.parametrize("missing", ["coverage.json", "18.json", "18.xml"])
def test_missing_evidence_is_not_zero_or_approval(evidence: Path, missing: str) -> None:
    (evidence / missing).unlink()
    assert run(evidence).returncode == 1
    output = evidence.parent / "output"
    badge = "coverage.json" if missing == "coverage.json" else "databricks-lts-ml.json"
    assert read(output / badge)["message"] == (
        "not verified" if missing == "coverage.json" else "17.3"
    )


@pytest.mark.parametrize(
    "change", ["incomplete", "counts", "identity", "exit", "targets"]
)
def test_success_requires_complete_consistent_evidence(
    evidence: Path, change: str
) -> None:
    path = evidence / "18.json"
    report = read(path)
    if change == "incomplete":
        del report["dependency_check"]
    else:
        changes: dict[str, tuple[str, object]] = {
            "counts": ("tests_executed", 6),
            "identity": ("commit_sha", "b" * 40),
            "exit": ("pytest_exit_code", 1),
            "targets": ("test_targets", []),
        }
        key, value = changes[change]
        report[key] = value
    write(path, report)
    assert run(evidence).returncode == (1 if change == "incomplete" else 2)


def test_missing_publication_metadata_is_reported(evidence: Path) -> None:
    write(evidence / "metadata.json", {})
    assert run(evidence).returncode == 1
    provenance = read(evidence.parent / "output/provenance.json")
    publication = cast(dict[str, object], provenance["publication"])
    assert "commit_sha" in cast(list[str], publication["missing_metadata"])
    assert read(evidence.parent / "output/coverage.json")["message"] == "not verified"


@pytest.mark.parametrize("filename", ["coverage.json", "18.json", "18.xml"])
def test_malformed_input_does_not_leave_output(evidence: Path, filename: str) -> None:
    _ = (evidence / filename).write_text("malformed", encoding="utf-8")
    result = run(evidence)
    assert result.returncode == 2
    assert result.stderr
    assert not (evidence.parent / "output").exists()


def test_duplicate_runtime_is_rejected(evidence: Path) -> None:
    assert (
        run(
            evidence,
            extra=(
                "--databricks",
                "18",
                str(evidence / "18.json"),
                str(evidence / "18.xml"),
            ),
        ).returncode
        == 2
    )


def test_output_must_be_independent_and_new(evidence: Path) -> None:
    assert run(evidence, evidence / "badges").returncode == 2
    assert run(evidence).returncode == 0
    output = evidence.parent / "output"
    original = {path.name: path.read_bytes() for path in output.iterdir()}
    assert run(evidence).returncode == 2
    assert original == {path.name: path.read_bytes() for path in output.iterdir()}


@pytest.mark.parametrize("field", ["actions_run_id", "actions_run_attempt"])
def test_rejects_mixed_execution_identifiers(evidence: Path, field: str) -> None:
    path = evidence / "18.json"
    report = read(path)
    report[field] = "999"
    write(path, report)
    assert run(evidence).returncode == 2


def test_rejects_url_with_another_run_id(evidence: Path) -> None:
    path = evidence / "metadata.json"
    report = read(path)
    report["actions_run_url"] = "https://github.com/example/project/actions/runs/999"
    write(path, report)
    assert run(evidence).returncode == 2


def test_unrelated_junit_cannot_certify_spark(evidence: Path) -> None:
    path = evidence / "18.xml"
    tree = ElementTree.parse(path)
    for case in tree.iter("testcase"):
        case.set("classname", "unrelated.tests")
    tree.write(path, encoding="utf-8")
    assert run(evidence).returncode == 1
    assert read(evidence.parent / "output/databricks-lts-ml.json")["message"] == "17.3"


def test_success_cannot_override_junit_failure(evidence: Path) -> None:
    path = evidence / "18.xml"
    tree = ElementTree.parse(path)
    _ = ElementTree.SubElement(tree.getroot()[0], "failure")
    tree.getroot().set("failures", "1")
    tree.write(path, encoding="utf-8")
    assert run(evidence).returncode == 2


@pytest.mark.parametrize("state", ["failed", "unverified"])
def test_no_approved_runtime_produces_nonaffirmative_badge(
    evidence: Path, state: str
) -> None:
    for runtime in ("17.3", "18"):
        path = evidence / f"{runtime}.json"
        if state == "unverified":
            path.unlink()
        else:
            report = read(path)
            report.update(result="failed", dependency_check="not_passed")
            write(path, report)
    assert run(evidence).returncode == 1
    assert read(evidence.parent / "output/databricks-lts-ml.json") == {
        "schemaVersion": 1,
        "label": "Databricks LTS ML",
        "message": "failed" if state == "failed" else "not verified",
        "color": "red" if state == "failed" else "lightgrey",
    }


def test_wrong_coverage_scope_is_not_announced(evidence: Path) -> None:
    path = evidence / "metadata.json"
    metadata = read(path)
    metadata["coverage_scope"] = {
        "python": "3.12",
        "os": "windows",
        "suite": "not optional",
    }
    write(path, metadata)
    assert run(evidence).returncode == 1
    assert read(evidence.parent / "output/coverage.json")["message"] == "not verified"


@pytest.mark.parametrize(
    "covered,color",
    [
        (0, "red"),
        (60, "yellow"),
        (70, "yellowgreen"),
        (80, "green"),
        (90, "brightgreen"),
        (100, "brightgreen"),
    ],
)
def test_coverage_scale_has_no_passing_threshold(
    evidence: Path, covered: int, color: str
) -> None:
    path = evidence / "coverage.json"
    report = read(path)
    report["totals"] = {
        "num_statements": 100,
        "covered_lines": covered,
        "missing_lines": 100 - covered,
        "num_branches": 100,
        "covered_branches": covered,
        "missing_branches": 100 - covered,
        "num_partial_branches": 0,
        "percent_covered": covered,
    }
    write(path, report)
    assert run(evidence).returncode == 0
    badge = read(evidence.parent / "output/coverage.json")
    assert badge["message"] == f"{covered}%"
    assert badge["color"] == color
