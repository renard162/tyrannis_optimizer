"""Official-release gates and artifact integrity; shared checks live in release.py."""

import argparse
import json
import os
import re
from pathlib import Path

import release

ROOT = Path("release")
EVIDENCE = ROOT / "evidence"
DIST = ROOT / "dist"


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
    return release.index_release(index)


def absent() -> None:
    if index_release("pypi") is not None:
        raise ValueError(
            "Version already exists on PyPI; no overwrite or skip-existing allowed"
        )


def seal() -> None:
    release.verify(DIST, EVIDENCE, os.environ["MANIFEST_SHA256"])
    envelope = {
        **release.identity(),
        "tag": tag(),
        "release_id": release.mapping(
            read(Path(os.environ["GITHUB_EVENT_PATH"]))["release"]
        )["id"],
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
    release.verify(DIST, EVIDENCE, manifest_digest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("command", choices=("guard", "seal", "verify", "absent"))
    _ = parser.add_argument("--evidence-sha256", default="")
    args = parser.parse_args()
    if args.command == "guard":
        guard()
    elif args.command == "seal":
        seal()
    elif args.command == "verify":
        verify(args.evidence_sha256)
    else:
        absent()


if __name__ == "__main__":
    main()
