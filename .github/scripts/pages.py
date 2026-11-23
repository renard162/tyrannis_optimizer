"""Bind CI evidence to one attempt, compose the complete site, and guard deploys."""

import hashlib
import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen
from xml.etree.ElementTree import ParseError

from databricks_compat import RUNTIMES
from generate_badges import (
    IDENTIFIERS,
    endpoint,
    generate,
    mapping,
    read_json,
    validate_metadata,
    write_outputs,
)

COVERAGE = "coverage-core-python312"
JOBS = {
    COVERAGE: "Core Tests (ubuntu-24.04, Python 3.12)",
    **{
        f"databricks-{runtime}-lts-ml": (
            f"Databricks {runtime} LTS ML (equivalent dependencies)"
        )
        for runtime in RUNTIMES
    },
}
FILES = {
    COVERAGE: ("coverage.json", "coverage.xml"),
    **{
        f"databricks-{runtime}-lts-ml": ("compatibility.json", "spark.xml")
        for runtime in RUNTIMES
    },
}


def identity() -> dict[str, object]:
    metadata: dict[str, object] = {
        "commit_sha": os.environ["GITHUB_SHA"],
        "actions_run_id": os.environ["GITHUB_RUN_ID"],
        "actions_run_attempt": os.environ["GITHUB_RUN_ATTEMPT"],
        "actions_run_url": (
            f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}"
            f"/actions/runs/{os.environ['GITHUB_RUN_ID']}"
        ),
        "consolidated_at": datetime.now(UTC).isoformat(),
    }
    validate_metadata(metadata)
    return metadata


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(directory: Path, name: str, metadata: dict[str, object]) -> None:
    """A sidecar binds file contents and the artifact's logical name to the producer."""
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        **metadata,
        "artifact": name,
        "files": {
            filename: digest(directory / filename)
            for filename in FILES[name]
            if (directory / filename).is_file()
        },
    }
    _ = (directory / "execution.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def api(path: str) -> dict[str, object]:
    request = Request(
        f"{os.environ['GITHUB_API_URL']}/repos/{os.environ['GITHUB_REPOSITORY']}/{path}",
        headers={
            "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urlopen(request, timeout=30) as response:
        return mapping(json.load(response))


def objects(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise TypeError("Expected an API array")
    return [mapping(item) for item in value]


def pages(path: str, key: str) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    page = 1
    while True:
        batch = objects(api(f"{path}?per_page=100&page={page}")[key])
        result.extend(batch)
        if len(batch) < 100:
            return result
        page += 1


def job_result(
    jobs: list[dict[str, object]], name: str, metadata: dict[str, object]
) -> str:
    matches = [job for job in jobs if job.get("name") == name]
    if len(matches) != 1:
        return "unverified"
    job = matches[0]
    if (
        job.get("head_sha") != metadata["commit_sha"]
        or str(job.get("run_id")) != metadata["actions_run_id"]
        or job.get("status") != "completed"
    ):
        return "unverified"
    return str(job.get("conclusion"))


def verified_directory(
    root: Path,
    name: str,
    metadata: dict[str, object],
    artifacts: list[dict[str, object]],
) -> Path:
    artifact_name = f"{name}-attempt-{metadata['actions_run_attempt']}"
    matches = [item for item in artifacts if item.get("name") == artifact_name]
    if len(matches) != 1:
        raise ValueError(f"Missing or duplicate artifact: {artifact_name}")
    artifact = matches[0]
    owner = mapping(artifact.get("workflow_run"))
    if (
        artifact.get("expired") is not False
        or owner.get("head_sha") != metadata["commit_sha"]
        or str(owner.get("id")) != metadata["actions_run_id"]
    ):
        raise ValueError(f"Artifact belongs to a different execution: {artifact_name}")
    directory = root / artifact_name
    manifest = read_json(directory / "execution.json")
    if manifest.get("artifact") != name or any(
        manifest.get(key) != metadata[key] for key in IDENTIFIERS
    ):
        raise ValueError(f"Invalid producer identity: {artifact_name}")
    hashes = mapping(manifest.get("files"))
    if set(hashes) - set(FILES[name]):
        raise ValueError(f"Unexpected evidence files: {artifact_name}")
    if {path.name for path in directory.iterdir()} != set(hashes) | {"execution.json"}:
        raise ValueError(f"Artifact contents differ from its manifest: {artifact_name}")
    for filename in FILES[name]:
        path = directory / filename
        if path.exists() != (filename in hashes) or (
            path.exists() and (path.is_symlink() or digest(path) != hashes[filename])
        ):
            raise ValueError(f"Invalid evidence digest: {path}")
    if name != COVERAGE and (directory / "compatibility.json").exists():
        report = read_json(directory / "compatibility.json")
        if any(report.get(key) != metadata[key] for key in IDENTIFIERS):
            raise ValueError(f"Unbound compatibility report: {name}")
    return directory


def compose(
    root: Path,
    site: Path,
    metadata: dict[str, object],
    artifacts: list[dict[str, object]],
    jobs: list[dict[str, object]],
    gate: str,
) -> None:
    """Use Task 3A unchanged; apply workflow approval policy before writing its bundle."""
    if not (site / "index.html").is_file() or (site / "badges").exists():
        raise ValueError("Build the complete site first; badges/ is reserved")
    missing = root / "missing"
    if missing.exists():
        raise ValueError("The missing-evidence sentinel must not exist")
    accepted: dict[str, Path] = {}
    rejected: dict[str, str] = {}
    results = {name: job_result(jobs, job, metadata) for name, job in JOBS.items()}
    metadata = {
        **metadata,
        "coverage_scope": {
            "python": "3.12",
            "os": "ubuntu-24.04",
            "suite": "not optional",
        },
        "ci_gate": gate,
        "job_results": results,
    }
    # Validate sources independently so one corrupt report cannot retain old greens.
    for name in JOBS:
        try:
            directory = verified_directory(root, name, metadata, artifacts)
            runtime = name.removeprefix("databricks-").removesuffix("-lts-ml")
            _ = generate(
                directory / "coverage.json" if name == COVERAGE else missing,
                []
                if name == COVERAGE
                else [
                    (runtime, directory / "compatibility.json", directory / "spark.xml")
                ],
                metadata,
            )
            accepted[name] = directory
        except (OSError, TypeError, ValueError, ParseError) as exc:
            rejected[name] = str(exc)
            print(f"::warning::{name}: {exc}")
    coverage = accepted.get(COVERAGE, missing) / "coverage.json"
    if results[COVERAGE] != "success":
        coverage = missing
    reports = [
        (runtime, directory / "compatibility.json", directory / "spark.xml")
        for runtime in RUNTIMES
        if (directory := accepted.get(f"databricks-{runtime}-lts-ml")) is not None
    ]
    documents, code = generate(coverage, reports, metadata)
    provenance = mapping(documents["provenance.json"])
    evidence = {
        COVERAGE: mapping(provenance["coverage"]),
        **{
            f"databricks-{str(item['runtime']).removesuffix(' LTS ML')}-lts-ml": item
            for item in objects(provenance["databricks"])
        },
    }
    for name, item in evidence.items():
        if results[name] != "success":
            item["report_state"] = item["state"]
            item["state"] = "failed" if results[name] == "failure" else "unverified"
            item["reason"] = f"Producer job result: {results[name]}"
    # CI Gate stays authoritative. Unrelated failures conservatively revoke greens too.
    blocked = gate != "success" or any(value != "success" for value in results.values())
    if blocked:
        failed = gate == "failure" or "failure" in results.values()
        message, color = ("failed", "red") if failed else ("not verified", "lightgrey")
        documents["coverage.json"] = endpoint("coverage", message, color)
        documents["databricks-lts-ml.json"] = endpoint(
            "Databricks LTS ML", message, color
        )
    states = [item["state"] for item in evidence.values()]
    provenance["evidence_state"] = (
        "failed"
        if gate == "failure" or "failed" in states
        else "unverified"
        if blocked or "unverified" in states
        else "approved"
    )
    publication = mapping(provenance["publication"])
    publication.update(
        workflow_verified=True,
        rejected_inputs=rejected,
        generator_exit_code=code,
        artifacts=[
            item
            for item in artifacts
            if item.get("name")
            in {f"{name}-attempt-{metadata['actions_run_attempt']}" for name in JOBS}
        ],
    )
    write_outputs(site / "badges", [root / "inputs"], documents)


def deploy_allowed(
    metadata: dict[str, object],
    run: dict[str, object],
    head: str,
    latest: dict[str, object],
    event: str,
    ref: str,
) -> bool:
    return (
        event == "push"
        and ref == "refs/heads/master"
        and run.get("event") == "push"
        and run.get("head_branch") == "master"
        and head == run.get("head_sha") == metadata["commit_sha"]
        and str(run.get("id")) == metadata["actions_run_id"]
        and str(run.get("run_attempt")) == metadata["actions_run_attempt"]
        and latest.get("id") == run.get("id")
        and run.get("conclusion") != "cancelled"
    )


def guard(metadata: dict[str, object]) -> bool:
    run = api(f"actions/runs/{metadata['actions_run_id']}")
    ref = api("git/ref/heads/master")
    runs = objects(
        api(
            f"actions/workflows/{run['workflow_id']}/runs?branch=master&event=push&per_page=1"
        )["workflow_runs"]
    )
    return bool(runs) and deploy_allowed(
        metadata,
        run,
        str(mapping(ref["object"])["sha"]),
        runs[0],
        os.environ["GITHUB_EVENT_NAME"],
        os.environ["GITHUB_REF"],
    )


def main() -> None:
    metadata = identity()
    command = sys.argv[1]
    if command == "record":
        record(Path(sys.argv[2]), sys.argv[3], metadata)
    elif command == "compose":
        run_id, attempt = metadata["actions_run_id"], metadata["actions_run_attempt"]
        artifacts = pages(f"actions/runs/{run_id}/artifacts", "artifacts")
        jobs = pages(f"actions/runs/{run_id}/attempts/{attempt}/jobs", "jobs")
        gate = job_result(jobs, "CI Gate", metadata)
        if gate != os.environ["CI_GATE_RESULT"]:
            raise ValueError("API and needs disagree about CI Gate")
        compose(Path("badge-inputs"), Path("site"), metadata, artifacts, jobs, gate)
    elif command == "prepare":
        # Replace this source with the future documentation build, in the same job.
        _ = shutil.copytree(".github/pages", "site")
    elif command == "guard":
        allowed = guard(metadata)
        with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
            _ = output.write(f"allowed={str(allowed).lower()}\n")
        print(
            "Deploy authorized" if allowed else "Obsolete/cancelled run: deploy skipped"
        )
    else:
        raise ValueError(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
