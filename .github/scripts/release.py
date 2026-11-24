"""Fail closed on release provenance and audit the exact distribution bytes."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import tarfile
import time
import tomllib
import zipfile
from email.parser import BytesParser
from http.client import HTTPException
from pathlib import Path, PurePosixPath
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
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
    enabled = os.environ.get("PUBLISH_TESTPYPI")
    if enabled not in {"true", "false"}:
        raise ValueError("Publication input must be an explicit boolean")
    if enabled == "true" and os.environ.get("CONFIRM_VERSION") != version():
        raise ValueError("Confirmed version must exactly match pyproject.toml")
    try:
        if mapping(api("git/ref/heads/master")["object"])["sha"] != sha:
            raise ValueError("Dispatched SHA is no longer the current master HEAD")
    except Exception as error:
        raise RuntimeError(f"Cannot verify current master: {error}") from error
    output("version", version())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distributions(directory: Path) -> list[Path]:
    wheels = list(directory.glob("*.whl"))
    sdists = list(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("Expected exactly one wheel and one sdist")
    files = wheels + sdists
    # uv writes this ignore file in its output directory; nothing else is allowed.
    if {p.name for p in directory.iterdir()} - {p.name for p in files} - {".gitignore"}:
        raise ValueError("Unexpected files in distribution directory")
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
    _ = (evidence / "SHA256SUMS").write_text(
        "".join(f"{value}  {name}\n" for name, value in hashes.items()),
        encoding="utf-8",
    )
    output("manifest_sha256", digest(path))
    for name, value in hashes.items():
        print(f"{value}  {name}")


def verify(directory: Path, evidence: Path, expected: str) -> None:
    path = evidence / "release.json"
    if not expected or digest(path) != expected:
        raise ValueError("Manifest digest differs from the build job output")
    manifest = mapping(json.loads(path.read_text("utf-8")))
    if any(manifest.get(key) != value for key, value in identity().items()):
        raise ValueError("Artifacts belong to another checkout/run/attempt")
    if manifest.get("name") != "tyrannis" or manifest.get("version") != version():
        raise ValueError("Artifact version differs from checkout")
    if manifest.get("files") != inspect(directory):
        raise ValueError("Artifact SHA-256 differs from the validated build")


def index_release(
    index: str, *, wait_for_publication: bool = False
) -> dict[str, object] | None:
    """Return version-scoped info/urls, confirmed absence, or fail closed.

    A project response describes the latest version in info; only its releases
    key identifies the requested version. Even an empty file list is existence
    evidence and must never authorize a new build/upload.
    """
    hosts = {"testpypi": "test.pypi.org", "pypi": "pypi.org"}
    if index not in hosts:
        raise ValueError("Unknown package index")
    current = version()
    base = f"https://{hosts[index]}/pypi/tyrannis"
    deadline = time.monotonic() + 180
    reason = "metadata unavailable"
    for attempt, delay in enumerate((0, 2, 4, 8, 16, 30), start=1):
        if time.monotonic() + delay >= deadline:
            break
        if delay:
            print(f"{index} {current}: {reason}; retry {attempt}/6 in {delay}s")
            time.sleep(delay)
        missing = 0
        for project_endpoint, url in (
            (False, f"{base}/{current}/json"),
            (True, f"{base}/json"),
        ):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                with urlopen(url, timeout=min(30, remaining)) as response:
                    result = mapping(json.load(response))
            except HTTPError as error:
                reason = f"{url}: HTTP {error.code}"
                if error.code == 404:
                    missing += 1
                elif error.code not in {408, 429} and not 500 <= error.code <= 599:
                    raise
                continue
            except (URLError, TimeoutError, ConnectionError, HTTPException) as error:
                reason = f"{url}: {error}"
                continue
            # Invalid JSON, identity or schema is not a transient absence.
            info = mapping(result["info"])
            if info.get("name") != "tyrannis" or not isinstance(
                info.get("version"), str
            ):
                raise ValueError(f"{index} returned inconsistent project identity")
            if project_endpoint:
                releases = mapping(result["releases"])
                if current not in releases:
                    if info["version"] == current:
                        raise ValueError(
                            f"{index} returned inconsistent release listing"
                        )
                    missing += 1
                    reason = f"{url}: version {current} not listed"
                    continue
                files = objects(releases[current])
            else:
                if info["version"] != current:
                    raise ValueError(f"{index} returned inconsistent release identity")
                files = objects(result["urls"])
            if time.monotonic() >= deadline:
                break
            # Do not carry latest-version metadata into a version-scoped result.
            return {"info": {"name": "tyrannis", "version": current}, "urls": files}
        if time.monotonic() >= deadline:
            break
        if missing == 2 and not wait_for_publication:
            return None
    raise RuntimeError(
        f"Cannot obtain {index} metadata for tyrannis {current} "
        f"within 6 attempts / 180s: {reason}"
    )


def fetch(evidence: Path, destination: Path, index: str = "testpypi") -> None:
    """Download exact files from one fixed index; never resolve across indexes."""
    file_host = (
        "test-files.pythonhosted.org"
        if index == "testpypi"
        else "files.pythonhosted.org"
    )
    if index not in {"testpypi", "pypi"}:
        raise ValueError("Unknown package index")
    manifest = mapping(json.loads((evidence / "release.json").read_text("utf-8")))
    if manifest.get("version") != version() or manifest.get("name") != "tyrannis":
        raise ValueError(f"{index} verification requires the original checkout")
    release = index_release(index, wait_for_publication=True)
    if release is None:
        raise RuntimeError(f"{index} published version is unavailable")
    expected = mapping(manifest["files"])
    published = objects(release["urls"])
    if (
        len(published) != 2
        or {item["filename"] for item in published} != set(expected)
        or set(expected)
        != {f"tyrannis-{version()}-py3-none-any.whl", f"tyrannis-{version()}.tar.gz"}
    ):
        raise ValueError(f"{index} file set differs from the validated build")
    for item in published:
        name, url = str(item["filename"]), str(item["url"])
        parsed = urlparse(url)
        if (
            parsed.scheme != "https"
            or parsed.hostname != file_host
            or PurePosixPath(name).name != name
        ):
            raise ValueError(f"Unexpected {index} download location")
        if mapping(item["digests"])["sha256"] != expected[name]:
            raise ValueError(f"{index} reports a different hash for {name}")
    destination.mkdir(parents=True, exist_ok=False)
    for item in published:
        name, url = str(item["filename"]), str(item["url"])
        with urlopen(url, timeout=60) as response:
            _ = (destination / name).write_bytes(response.read())
    if inspect(destination) != expected:
        raise ValueError(f"Downloaded {index} bytes differ from validated artifacts")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("command", choices=("guard", "record", "verify", "fetch"))
    _ = parser.add_argument("--dist", type=Path, default=Path("release/dist"))
    _ = parser.add_argument("--evidence", type=Path, default=Path("release/evidence"))
    _ = parser.add_argument("--manifest-sha256", default="")
    _ = parser.add_argument("--index", choices=("testpypi", "pypi"), default="testpypi")
    args = parser.parse_args()
    if args.command == "guard":
        guard()
    elif args.command == "record":
        record(args.dist, args.evidence)
    elif args.command == "verify":
        verify(args.dist, args.evidence, args.manifest_sha256)
    else:
        fetch(args.evidence, args.dist, args.index)


if __name__ == "__main__":
    main()
