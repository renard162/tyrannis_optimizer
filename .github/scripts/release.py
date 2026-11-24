"""Validate package contents once, then verify release identity and exact bytes."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import tarfile
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import cast
from urllib.request import Request, urlopen

REPOSITORY = "renard162/tyrannis_optimizer"


def mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("Expected a JSON/TOML object")
    # Decoder boundary: JSON/TOML object keys are strings, values are validated below.
    return cast(dict[str, object], value)


def objects(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise TypeError("Expected an API array")
    return [mapping(item) for item in cast(list[object], value)]


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def project() -> dict[str, object]:
    return mapping(tomllib.loads(Path("pyproject.toml").read_text("utf-8"))["project"])


def version() -> str:
    data = project()
    result = data["version"]
    if data["name"] != "tyrannis" or not isinstance(result, str):
        raise ValueError("Expected the tyrannis project with a static version")
    if re.fullmatch(r"[0-9][A-Za-z0-9.!+]*", result) is None:
        raise ValueError("Unsafe version in pyproject.toml")
    return result


def output(name: str, value: object) -> None:
    print(f"{name}={value}")
    if "GITHUB_OUTPUT" in os.environ:
        with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as stream:
            _ = stream.write(f"{name}={value}\n")


def api(path: str) -> dict[str, object]:
    request = Request(
        f"https://api.github.com/repos/{REPOSITORY}" + (f"/{path}" if path else ""),
        headers={
            "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urlopen(request, timeout=30) as response:
        return mapping(json.load(response))


def guard() -> None:
    """Used before building and again after environment approval, before upload."""
    expected = {
        "GITHUB_REPOSITORY": REPOSITORY,
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_REF": "refs/heads/master",
    }
    for name, value in expected.items():
        if os.environ.get(name) != value:
            raise ValueError(f"Release requires {name}={value}")
    sha = os.environ["GITHUB_SHA"]
    if git("rev-parse", "HEAD") != sha:
        raise ValueError("Checkout does not match the dispatched SHA")
    current = version()
    enabled = os.environ.get("PUBLISH_TESTPYPI")
    if enabled not in {"true", "false"}:
        raise ValueError("Publication input must be an explicit boolean")
    if enabled == "true" and os.environ.get("CONFIRM_VERSION") != current:
        raise ValueError("Confirmed version must exactly match pyproject.toml")
    try:
        if mapping(api("git/ref/heads/master")["object"])["sha"] != sha:
            raise ValueError("Dispatched SHA is no longer the current master HEAD")
    except Exception as error:
        raise RuntimeError(f"Cannot verify current master: {error}") from error


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distributions(directory: Path) -> list[Path]:
    wheels = list(directory.glob("*.whl"))
    sdists = list(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("Expected exactly one wheel and one sdist")
    files = wheels + sdists
    if set(directory.iterdir()) != set(files):
        raise ValueError("Unexpected files in distribution directory")
    if any(path.is_symlink() or not path.is_file() for path in files):
        raise ValueError("Distributions must be regular files")
    return files


def archive(path: Path) -> dict[str, bytes]:
    result: dict[str, bytes] = {}

    def add(name: str, data: bytes) -> None:
        parts = PurePosixPath(name).parts
        if name in result or name.startswith("/") or ".." in parts or "\\" in name:
            raise ValueError(f"Unsafe or duplicate archive path: {name}")
        result[name] = data

    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as wheel:
            for member in wheel.infolist():
                if not member.is_dir():
                    add(member.filename, wheel.read(member))
    else:
        with tarfile.open(path, "r:gz") as sdist:
            for member in sdist:
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError(f"Non-regular sdist member: {member.name}")
                stream = sdist.extractfile(member)
                if stream is None:
                    raise ValueError(f"Unreadable sdist member: {member.name}")
                add(member.name, stream.read())
    return result


def inspect(directory: Path) -> dict[str, str]:
    """Allow only tracked package files and known build-backend metadata."""
    current = version()
    wheel_path, sdist_path = distributions(directory)
    if (
        wheel_path.name != f"tyrannis-{current}-py3-none-any.whl"
        or sdist_path.name != f"tyrannis-{current}.tar.gz"
    ):
        raise ValueError("Distribution filenames do not match project identity")
    sources = git("ls-files", "src/tyrannis").splitlines()
    if not sources or git(
        "diff",
        "HEAD",
        "--",
        "src/tyrannis",
        "pyproject.toml",
        "uv.lock",
        "README.md",
        "LICENSE",
    ):
        raise ValueError("Package inputs differ from the checked-out commit")
    for name in sources:
        path = PurePosixPath(name)
        if any(
            part.startswith(".") or part == "__pycache__" for part in path.parts
        ) or path.suffix in {".pyc", ".pyo", ".tmp", ".bak", ".log"}:
            raise ValueError(f"Development residue in tracked sources: {name}")
    wheel = archive(wheel_path)
    sdist = archive(sdist_path)
    prefix = f"tyrannis-{current}"
    metadata_dir = f"{prefix}.dist-info"
    wheel_generated = {
        f"{metadata_dir}/{name}"
        for name in (
            "METADATA",
            "WHEEL",
            "RECORD",
            "entry_points.txt",
            "licenses/LICENSE",
        )
    }
    expected_wheel = {name.removeprefix("src/") for name in sources} | wheel_generated
    expected_sdist = {
        f"{prefix}/{name}"
        for name in sources
        + ["PKG-INFO", "pyproject.toml", "pyproject.toml.orig", "README.md", "LICENSE"]
    }
    if set(wheel) != expected_wheel or set(sdist) != expected_sdist:
        raise ValueError(
            f"Unexpected/missing archive content: wheel={set(wheel) ^ expected_wheel}, "
            f"sdist={set(sdist) ^ expected_sdist}"
        )
    for name in sources:
        source = Path(name).read_bytes()
        if (
            wheel[name.removeprefix("src/")] != source
            or sdist[f"{prefix}/{name}"] != source
        ):
            raise ValueError(f"Archive source differs from validated checkout: {name}")
    for name in ("README.md", "LICENSE"):
        if sdist[f"{prefix}/{name}"] != Path(name).read_bytes():
            raise ValueError(f"sdist {name} differs from checkout")
    if wheel[f"{metadata_dir}/licenses/LICENSE"] != Path("LICENSE").read_bytes():
        raise ValueError("Wheel license differs from checkout")
    # uv_build retains the original and rewrites the effective TOML for the sdist.
    if sdist[f"{prefix}/pyproject.toml.orig"] != Path("pyproject.toml").read_bytes():
        raise ValueError("sdist original pyproject differs from checkout")
    original = tomllib.loads(Path("pyproject.toml").read_text("utf-8"))
    if tomllib.loads(sdist[f"{prefix}/pyproject.toml"].decode()) != original:
        raise ValueError("sdist effective pyproject differs from checkout")
    raw = wheel[f"{metadata_dir}/METADATA"]
    if raw != sdist[f"{prefix}/PKG-INFO"]:
        raise ValueError("Wheel and sdist metadata differ")
    metadata = BytesParser().parsebytes(raw)
    data = project()
    for field, expected in {
        "Name": "tyrannis",
        "Version": current,
        "Summary": data["description"],
        "License-Expression": data["license"],
    }.items():
        if metadata.get(field) != expected:
            raise ValueError(f"Metadata mismatch: {field}")
    if str(metadata.get("Requires-Python")).replace(" ", "") != str(
        data["requires-python"]
    ).replace(" ", ""):
        raise ValueError("Python compatibility metadata differs")
    if metadata.get_all("Classifier") != data["classifiers"]:
        raise ValueError("Classifiers differ from pyproject.toml")
    urls = [f"{key}, {value}" for key, value in mapping(data["urls"]).items()]
    if sorted(metadata.get_all("Project-URL", [])) != sorted(urls):
        raise ValueError("Project URLs differ from pyproject.toml")
    if set(metadata.get_all("Provides-Extra", [])) != set(
        mapping(data["optional-dependencies"])
    ):
        raise ValueError("Optional dependency groups differ from pyproject.toml")
    if metadata.get_all("License-File") != data["license-files"]:
        raise ValueError("License file metadata differs from pyproject.toml")
    payload = metadata.get_payload(decode=True)
    if not isinstance(payload, bytes):
        raise TypeError("Expected a UTF-8 description payload")
    body = payload.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    if body != Path("README.md").read_text("utf-8"):
        raise ValueError("Distribution description differs from README.md")
    return {path.name: digest(path) for path in (wheel_path, sdist_path)}


def identity() -> dict[str, str]:
    return {
        "repository": os.environ.get("GITHUB_REPOSITORY", REPOSITORY),
        "sha": git("rev-parse", "HEAD"),
        "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", "local"),
    }


def record(directory: Path, evidence: Path) -> None:
    hashes = inspect(directory)
    evidence.mkdir(parents=True, exist_ok=True)
    manifest = {
        **identity(),
        "name": "tyrannis",
        "version": version(),
        "files": hashes,
    }
    path = evidence / "release.json"
    _ = path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    output("manifest_sha256", digest(path))
    output("producer_attempt", manifest["run_attempt"])
    for name, value in hashes.items():
        print(f"{value}  {name}")


def verify(
    directory: Path, evidence: Path, expected: str, producer_attempt: str
) -> None:
    path = evidence / "release.json"
    if not expected or digest(path) != expected:
        raise ValueError("Manifest digest differs from the build job output")
    manifest = mapping(json.loads(path.read_text("utf-8")))
    # A failed-jobs rerun can consume a successful build from an earlier attempt.
    # Trust the producer's output, never the consumer's current run_attempt.
    if re.fullmatch(r"[1-9][0-9]*", producer_attempt) is None:
        raise ValueError("Missing or invalid producer attempt")
    expected_identity = {**identity(), "run_attempt": producer_attempt}
    if expected_identity["sha"] != os.environ.get(
        "GITHUB_SHA", expected_identity["sha"]
    ) or any(manifest.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("Artifacts belong to another checkout/run/attempt")
    if manifest.get("name") != "tyrannis" or manifest.get("version") != version():
        raise ValueError("Artifact version differs from checkout")
    files = distributions(directory)
    # The pinned manifest records the filenames already inspected by Package Build.
    if manifest.get("files") != {path.name: digest(path) for path in files}:
        raise ValueError("Artifact SHA-256 differs from the validated build")


def publication_identity(index: str) -> dict[str, str]:
    event = mapping(
        json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text("utf-8"))
    )
    return {
        **{key: value for key, value in identity().items() if key != "run_attempt"},
        "index": index,
        "version": version(),
        "ref": os.environ["GITHUB_REF"],
        "release_id": str(mapping(event["release"])["id"])
        if os.environ["GITHUB_EVENT_NAME"] == "release"
        else "manual",
    }


def artifact(identifier: str, name: str) -> None:
    if re.fullmatch(r"[1-9][0-9]*", identifier) is None:
        raise ValueError("Missing original artifact ID; recovery blocked")
    try:
        item = api(f"actions/artifacts/{identifier}")
    except Exception as error:
        raise RuntimeError(
            "Original artifact unavailable; restore access and rerun the same job. "
            "Expired/deleted artifacts cannot be rebuilt for recovery."
        ) from error
    run = mapping(item["workflow_run"])
    if (
        str(item.get("id")) != identifier
        or item.get("name") != name
        or item.get("expired") is not False
        or str(run.get("id")) != os.environ["GITHUB_RUN_ID"]
        or run.get("head_sha") != os.environ["GITHUB_SHA"]
    ):
        raise ValueError("Original artifact expired or belongs to another source/run")


def locate(index: str) -> None:
    """Look only in this authorized run, never search by package version."""
    name = f"publication-{index}-{os.environ['GITHUB_RUN_ID']}"
    result = api(f"actions/runs/{os.environ['GITHUB_RUN_ID']}/artifacts?per_page=100")
    items = objects(result["artifacts"])
    if result.get("total_count") != len(items):
        raise ValueError("Artifact listing incomplete; recovery blocked")
    matches = [item for item in items if item.get("name") == name]
    if len(matches) > 1:
        raise ValueError("Ambiguous publication binding; recovery blocked")
    if matches:
        identifier = str(matches[0]["id"])
        artifact(identifier, name)
        output("binding_id", identifier)
    elif os.environ["GITHUB_RUN_ATTEMPT"] != "1":
        raise ValueError(
            "Original publication binding unavailable on rerun; upload blocked. "
            "Investigate the original run/artifact retention; do not substitute a rebuild."
        )
    else:
        output("binding_id", "")


def bind(index: str, path: Path) -> None:
    """Pin CI outputs once, before any upload, in a native immutable artifact."""
    expected = publication_identity(index)
    if path.exists():
        data = mapping(json.loads(path.read_text("utf-8")))
        if any(data.get(key) != value for key, value in expected.items()):
            raise ValueError("Publication binding belongs to another release/run/index")
    else:
        if os.environ["GITHUB_RUN_ATTEMPT"] != "1":
            raise ValueError(
                "Missing original publication binding; no rebuild fallback"
            )
        data = {
            **expected,
            "artifact_id": os.environ["CI_ARTIFACT_ID"],
            "manifest_sha256": os.environ["CI_MANIFEST_SHA256"],
            "producer_attempt": os.environ["CI_PRODUCER_ATTEMPT"],
        }
    attempt = str(data["producer_attempt"])
    if (
        re.fullmatch(r"[1-9][0-9]*", attempt) is None
        or int(attempt) > int(os.environ["GITHUB_RUN_ATTEMPT"])
        or re.fullmatch(r"[0-9a-f]{64}", str(data["manifest_sha256"])) is None
    ):
        raise ValueError("Invalid original CI evidence")
    artifact(
        str(data["artifact_id"]),
        f"tyrannis-{expected['sha']}-{expected['run_id']}-{attempt}",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        _ = path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    for key in ("artifact_id", "manifest_sha256", "producer_attempt"):
        output(key, data[key])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument(
        "command", choices=("guard", "record", "verify", "locate", "bind")
    )
    _ = parser.add_argument(
        "--binding", type=Path, default=Path("binding/publication.json")
    )
    _ = parser.add_argument("--dist", type=Path, default=Path("release/dist"))
    _ = parser.add_argument("--evidence", type=Path, default=Path("release/evidence"))
    _ = parser.add_argument("--manifest-sha256", default="")
    _ = parser.add_argument("--producer-attempt", default="")
    _ = parser.add_argument("--index", choices=("testpypi", "pypi"), default="testpypi")
    args = parser.parse_args()
    if args.command == "guard":
        guard()
    elif args.command == "record":
        record(args.dist, args.evidence)
    elif args.command == "verify":
        verify(args.dist, args.evidence, args.manifest_sha256, args.producer_attempt)
    elif args.command == "locate":
        locate(args.index)
    else:
        bind(args.index, args.binding)


if __name__ == "__main__":
    main()
