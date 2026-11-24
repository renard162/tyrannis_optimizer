"""Official-release gates and cross-run promotion; shared checks live in release.py."""

import argparse
import json
import os
import re
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import release

ROOT = Path("release")
EVIDENCE = ROOT / "evidence"
DIST = ROOT / "dist"
CD1_JOBS = {
    "Validate source and distributions",
    "Publish exact validated bytes to TestPyPI",
    "Verify published TestPyPI files and installation",
}


def read(path: Path) -> dict[str, object]:
    return release.mapping(json.loads(path.read_text("utf-8")))


def number(value: object) -> str:
    text = str(value)
    if re.fullmatch(r"[1-9][0-9]*", text) is None:
        raise ValueError("Expected a positive GitHub identifier")
    return text


def tag() -> str:
    current = release.version()
    # Official stable releases only, with canonical (non-normalized) spelling.
    if (
        re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", current)
        is None
    ):
        raise ValueError("Official releases require a stable X.Y.Z version")
    return f"v{current}"


def guard() -> None:
    """Recheck the live release, tag, ancestry and CI after environment approval."""
    expected = {
        "GITHUB_REPOSITORY": release.REPOSITORY,
        "GITHUB_EVENT_NAME": "release",
        "GITHUB_REF": f"refs/tags/{tag()}",
    }
    for name, value in expected.items():
        if os.environ.get(name) != value:
            raise ValueError(f"Official publication requires {name}={value}")
    sha = os.environ["GITHUB_SHA"]
    if (
        re.fullmatch(r"[0-9a-f]{40}", sha) is None
        or release.git("rev-parse", "HEAD") != sha
    ):
        raise ValueError("Checkout differs from the release commit")
    event = read(Path(os.environ["GITHUB_EVENT_PATH"]))
    repository = release.mapping(event["repository"])
    live_repository = release.api("")
    if (
        event.get("action") != "published"
        or repository.get("full_name") != release.REPOSITORY
        or repository.get("fork") is not False
        or live_repository.get("fork") is not False
        or repository.get("id") != live_repository["id"]
    ):
        raise ValueError("Release event has an unauthorized action or repository")
    received = release.mapping(event["release"])
    live = release.api(f"releases/{number(received['id'])}")
    for item in (received, live):
        if (
            item.get("id") != received["id"]
            or item.get("tag_name") != tag()
            or item.get("draft") is not False
            or item.get("prerelease") is not False
            or not item.get("published_at")
            or item.get("target_commitish") not in {"master", sha}
            or item.get("url")
            != f"https://api.github.com/repos/{release.REPOSITORY}/releases/{received['id']}"
        ):
            raise ValueError("Release identity, tag or official status is inconsistent")
    reference = release.api(f"git/ref/tags/{tag()}")
    if reference.get("ref") != f"refs/tags/{tag()}":
        raise ValueError("GitHub returned a different tag reference")
    obj = release.mapping(reference["object"])
    # Support lightweight and annotated tags, including nested annotated tags.
    for _ in range(10):
        if obj.get("type") != "tag":
            break
        object_sha = str(obj["sha"])
        if re.fullmatch(r"[0-9a-f]{40}", object_sha) is None:
            raise ValueError("Invalid tag object SHA")
        obj = release.mapping(release.api(f"git/tags/{object_sha}")["object"])
    if obj.get("type") != "commit" or obj.get("sha") != sha:
        raise ValueError(
            "Tag moved or does not resolve to the checked-out release commit"
        )
    comparison = release.api(f"compare/{sha}...master")
    if (
        comparison.get("status") not in {"ahead", "identical"}
        or release.mapping(comparison["base_commit"]).get("sha") != sha
        or release.mapping(comparison["merge_base_commit"]).get("sha") != sha
    ):
        raise ValueError(
            "Release commit is not contained in the current master history"
        )
    release.require_ci(sha)


def index_release(index: str) -> dict[str, object] | None:
    host = {"testpypi": "test.pypi.org", "pypi": "pypi.org"}[index]
    try:
        with urlopen(
            f"https://{host}/pypi/tyrannis/{release.version()}/json", timeout=30
        ) as response:
            result = release.mapping(json.load(response))
    except HTTPError as error:
        if error.code == 404:
            return None
        raise
    info = release.mapping(result["info"])
    if info.get("name") != "tyrannis" or info.get("version") != release.version():
        raise ValueError(f"{index} returned inconsistent project identity")
    return result


def absent() -> None:
    if index_release("pypi") is not None:
        raise ValueError(
            "Version already exists on PyPI; no overwrite or skip-existing allowed"
        )


def source(artifact: dict[str, object]) -> dict[str, str] | None:
    """Authenticate the original CD-1 attempt, never substitute the current run."""
    pattern = rf"tyrannis-{re.escape(release.version())}-{release.git('rev-parse', 'HEAD')}-([1-9][0-9]*)-([1-9][0-9]*)"
    match = re.fullmatch(pattern, str(artifact["name"]))
    if match is None or artifact.get("expired") is not False:
        return None
    run_id, attempt = match.groups()
    run = release.api(f"actions/runs/{run_id}/attempts/{attempt}")
    workflow = release.api("actions/workflows/release.yml")
    sha = release.git("rev-parse", "HEAD")
    if (
        str(run.get("id")) != run_id
        or str(run.get("run_attempt")) != attempt
        or run.get("workflow_id") != workflow["id"]
        or run.get("path") != ".github/workflows/release.yml"
        or run.get("event") != "workflow_dispatch"
        or run.get("head_branch") != "master"
        or run.get("head_sha") != sha
        or run.get("status") != "completed"
        or run.get("conclusion") != "success"
        or release.mapping(run["repository"]).get("full_name") != release.REPOSITORY
        or release.mapping(run["head_repository"]).get("full_name")
        != release.REPOSITORY
    ):
        return None
    origin = release.mapping(artifact["workflow_run"])
    if (
        str(origin.get("id")) != run_id
        or origin.get("head_sha") != sha
        or origin.get("head_branch") != "master"
        or origin.get("repository_id") != release.mapping(run["repository"])["id"]
        or origin.get("head_repository_id") != origin.get("repository_id")
    ):
        raise ValueError("Artifact origin differs from the original CD-1 run")
    jobs = release.api(f"actions/runs/{run_id}/attempts/{attempt}/jobs?per_page=100")
    entries = release.objects(jobs["jobs"])
    if jobs["total_count"] != len(entries):
        raise ValueError("Incomplete original job evidence")
    successful = {
        job["name"]
        for job in entries
        if job.get("status") == "completed" and job.get("conclusion") == "success"
    }
    if not CD1_JOBS <= successful:
        return None
    archive_digest = str(artifact.get("digest"))
    if re.fullmatch(r"sha256:[0-9a-f]{64}", archive_digest) is None:
        raise ValueError("Original artifact has no SHA-256 archive evidence")
    return {
        "artifact_id": number(artifact["id"]),
        "archive_digest": archive_digest,
        "run_id": run_id,
        "run_attempt": attempt,
    }


def select() -> None:
    """Only a genuine 404 permits a new build; ambiguous promotion fails closed."""
    if ROOT.exists():
        raise ValueError("Release directory must be fresh")
    absent()
    if index_release("testpypi") is None:
        release.output("mode", "build")
        return
    candidates: list[dict[str, str]] = []
    for page in range(1, 11):
        response = release.api(f"actions/artifacts?per_page=100&page={page}")
        artifacts = release.objects(response["artifacts"])
        for artifact in artifacts:
            evidence = source(artifact)
            if evidence is not None:
                candidates.append(evidence)
        if len(artifacts) < 100:
            break
    else:
        raise ValueError("Artifact search limit exceeded; cannot prove unique origin")
    if len(candidates) != 1:
        raise ValueError(
            "Version exists on TestPyPI but no unique successful CD-1 artifact "
            "matches this commit (or evidence expired). Promotion blocked; no rebuild."
        )
    release.output("mode", "promote")
    for name, value in candidates[0].items():
        release.output(name, value)


def promoted(artifact_id: str) -> dict[str, str]:
    origin = source(release.api(f"actions/artifacts/{number(artifact_id)}"))
    if origin is None:
        raise ValueError("Original TestPyPI publication is not approved")
    manifest = read(EVIDENCE / "release.json")
    expected = {
        "repository": release.REPOSITORY,
        "sha": release.git("rev-parse", "HEAD"),
        "run_id": origin["run_id"],
        "run_attempt": origin["run_attempt"],
        "name": "tyrannis",
        "version": release.version(),
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Original manifest differs from authorized promotion origin")
    ci = release.api(
        f"actions/runs/{number(manifest['ci_run_id'])}"
        f"/attempts/{number(manifest['ci_run_attempt'])}"
    )
    workflow = release.api("actions/workflows/ci.yml")
    if (
        not release.approved_ci(ci, workflow["id"], expected["sha"])
        or str(ci["id"]) != str(manifest["ci_run_id"])
        or str(ci["run_attempt"]) != str(manifest["ci_run_attempt"])
        or ci["html_url"] != manifest.get("ci_run_url")
    ):
        raise ValueError("Original manifest CI approval cannot be verified")
    if manifest.get("files") != release.inspect(DIST):
        raise ValueError("Promoted hashes/content differ from the original manifest")
    published = index_release("testpypi")
    if published is None:
        raise ValueError("Original TestPyPI version disappeared")
    files = release.objects(published["urls"])
    hashes = {
        str(item["filename"]): release.mapping(item["digests"])["sha256"]
        for item in files
    }
    if len(files) != 2 or hashes != manifest["files"]:
        raise ValueError("TestPyPI hashes differ from the original approved artifacts")
    return origin


def seal(mode: str, artifact_id: str) -> None:
    origin = promoted(artifact_id) if mode == "promote" else None
    if mode == "build":
        if index_release("testpypi") is not None:
            raise ValueError(
                "TestPyPI version appeared during build; promotion required"
            )
        release.verify(DIST, EVIDENCE, os.environ["MANIFEST_SHA256"])
    if mode == "promote":
        # Download and inspect the exact index bytes as additional promotion evidence.
        release.fetch(EVIDENCE, Path("promoted-index"))
    envelope = {
        **release.identity(),
        "tag": tag(),
        "release_id": release.mapping(
            read(Path(os.environ["GITHUB_EVENT_PATH"]))["release"]
        )["id"],
        "mode": mode,
        "origin": origin,
        "manifest_sha256": release.digest(EVIDENCE / "release.json"),
    }
    path = EVIDENCE / "official.json"
    _ = path.write_text(json.dumps(envelope, indent=2) + "\n", encoding="utf-8")
    release.output("evidence_sha256", release.digest(path))
    print(json.dumps(envelope, indent=2))


def verify(expected: str) -> None:
    path = EVIDENCE / "official.json"
    if not expected or release.digest(path) != expected:
        raise ValueError("Official evidence differs from the validation job output")
    envelope = read(path)
    if any(envelope.get(key) != value for key, value in release.identity().items()):
        raise ValueError("Official artifacts belong to another checkout/run/attempt")
    received = release.mapping(read(Path(os.environ["GITHUB_EVENT_PATH"]))["release"])
    if envelope.get("tag") != tag() or envelope.get("release_id") != received["id"]:
        raise ValueError("Official artifact tag/release identity differs")
    manifest_digest = str(envelope["manifest_sha256"])
    if release.digest(EVIDENCE / "release.json") != manifest_digest:
        raise ValueError("Original manifest was modified")
    if envelope.get("mode") == "promote":
        origin = release.mapping(envelope["origin"])
        if promoted(str(origin["artifact_id"])) != origin:
            raise ValueError("Original promotion evidence changed")
    elif envelope.get("mode") == "build":
        release.verify(DIST, EVIDENCE, manifest_digest)
        if index_release("testpypi") is not None:
            raise ValueError(
                "TestPyPI version now exists; rebuild cannot replace promotion"
            )
    else:
        raise ValueError("Unknown official artifact strategy")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument(
        "command", choices=("guard", "select", "seal", "verify", "absent")
    )
    _ = parser.add_argument("--mode", choices=("build", "promote"), default="build")
    _ = parser.add_argument("--artifact-id", default="")
    _ = parser.add_argument("--evidence-sha256", default="")
    args = parser.parse_args()
    if args.command == "guard":
        guard()
    elif args.command == "select":
        select()
    elif args.command == "seal":
        seal(args.mode, args.artifact_id)
    elif args.command == "verify":
        verify(args.evidence_sha256)
    else:
        absent()


if __name__ == "__main__":
    main()
