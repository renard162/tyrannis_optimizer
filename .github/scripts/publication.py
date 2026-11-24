"""Bounded publication preparation and verification on the Linux release runners."""

import argparse
import errno
import io
import json
import os
import re
import shutil
import signal
import socket
import ssl
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from email.message import Message
from email.utils import parsedate_to_datetime
from functools import partial
from http.client import HTTPException, IncompleteRead, RemoteDisconnected
from pathlib import Path
from types import FrameType
from typing import BinaryIO, Never, TypeVar
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

import release

WINDOW = 600
MAX_ATTEMPTS = 24
INITIAL_DELAY = 2
MAX_DELAY = 30
NETWORK_TIMEOUT = 30
# Reconciliation shares the CD-05 retry policy, with a smaller bounded window.
ABSENCE_WINDOW = 180
ABSENCE_ATTEMPTS = 6
INDEX_HOSTS = {"pypi": "pypi.org", "testpypi": "test.pypi.org"}
FILE_HOSTS = {
    "pypi": "files.pythonhosted.org",
    "testpypi": "test-files.pythonhosted.org",
}
T = TypeVar("T")


class Unconfirmed(Exception):
    """Verification exhausted its budget; this says nothing about upload success."""


class Inconsistent(Exception):
    """Positive evidence contradicts the approved identity or bytes."""


class PartiallyPublished(Exception):
    """A valid final observation still lacks approved distributions."""


class Transient(Exception):
    def __init__(self, reason: str, retry_after: float = 0) -> None:
        super().__init__(reason)
        self.retry_after: float = retry_after


@dataclass
class Retry:
    deadline: float
    attempts: int = MAX_ATTEMPTS

    def remaining(self) -> float:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise Unconfirmed("Global verification deadline exhausted")
        return remaining

    def run(self, operation: str, action: Callable[[], T]) -> T:
        for attempt in range(1, self.attempts + 1):
            print(
                f"{operation}: attempt {attempt}/{self.attempts}; "
                f"remaining {self.remaining():.1f}s",
                flush=True,
            )
            try:
                result = action()
                _ = self.remaining()
                return result
            except Transient as error:
                remaining = self.remaining()
                delay = max(
                    min(INITIAL_DELAY * 2 ** (attempt - 1), MAX_DELAY),
                    error.retry_after,
                )
                if attempt == self.attempts or delay >= remaining:
                    raise Unconfirmed(
                        f"{operation}: {error}; attempt {attempt}/{self.attempts}; "
                        f"no retry fits the remaining limits ({remaining:.1f}s)"
                    ) from error
                print(
                    f"{operation}: attempt {attempt}/{self.attempts}: {error}; "
                    f"retry in {delay:.1f}s; remaining {remaining:.1f}s",
                    flush=True,
                )
                time.sleep(delay)
        raise Unconfirmed(f"{operation}: attempt limit exhausted")


def expired(_signum: int, _frame: FrameType | None) -> None:
    raise Unconfirmed("Global verification deadline exhausted")


@contextmanager
def budget(seconds: float = WINDOW, attempts: int = MAX_ATTEMPTS) -> Iterator[Retry]:
    # urllib socket timeouts do not bound DNS or a peer trickling response bytes.
    # SIGALRM also interrupts those blocking calls on the Ubuntu release runners.
    if os.name != "posix":
        raise RuntimeError("Publication verification requires a POSIX runner")
    retry = Retry(time.monotonic() + seconds, attempts)
    previous = signal.signal(signal.SIGALRM, expired)
    _ = signal.setitimer(signal.ITIMER_REAL, retry.remaining())
    try:
        yield retry
    finally:
        _ = signal.setitimer(signal.ITIMER_REAL, 0)
        _ = signal.signal(signal.SIGALRM, previous)


def retry_after(value: str | None) -> float:
    if value is None:
        return 0
    if value.strip().isdigit():
        return float(value)
    try:
        return max(0, parsedate_to_datetime(value).timestamp() - time.time())
    except (ValueError, TypeError, OverflowError):
        return 0


def location(url: str, host: str) -> None:
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != host
        or parsed.port not in {None, 443}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise Inconsistent("Unexpected index/download location or redirect")


class SameOriginRedirect(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: BinaryIO,
        code: int,
        msg: str,
        headers: Message,
        newurl: str,
    ) -> Request | None:
        location(newurl, str(urlparse(req.full_url).hostname))
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def network_failure(error: OSError) -> Never:
    if isinstance(error, socket.gaierror):
        transient = error.errno == socket.EAI_AGAIN
    else:
        transient = isinstance(
            error, (TimeoutError, ConnectionError)
        ) or error.errno in {
            errno.ETIMEDOUT,
            errno.ECONNRESET,
            errno.ECONNREFUSED,
            errno.ECONNABORTED,
            errno.ENETUNREACH,
            errno.EHOSTUNREACH,
        }
    if (transient and not isinstance(error, ssl.SSLError)) or isinstance(
        error, ssl.SSLEOFError
    ):
        raise Transient(
            f"Temporary connection/read failure ({type(error).__name__})"
        ) from error
    raise RuntimeError(
        f"Network failure of undetermined nature ({type(error).__name__}); not retried"
    ) from error


def transfer(url: str, host: str, target: BinaryIO, retry: Retry) -> None:
    location(url, host)
    try:
        with build_opener(SameOriginRedirect()).open(
            url, timeout=min(NETWORK_TIMEOUT, retry.remaining())
        ) as response:
            location(response.geturl(), host)
            length = response.headers.get("Content-Length")
            received = 0
            while chunk := response.read1(64 * 1024):
                _ = retry.remaining()
                received += len(chunk)
                _ = target.write(chunk)
            _ = retry.remaining()
            if length is not None and received != int(length):
                raise Transient("Interrupted download (Content-Length mismatch)")
    except HTTPError as error:
        delay = retry_after(error.headers.get("Retry-After"))
        error.close()
        if error.code in {404, 408, 429} or 500 <= error.code <= 599:
            raise Transient(f"HTTP {error.code}", delay) from error
        raise RuntimeError(f"Non-retryable HTTP {error.code}") from error
    except (IncompleteRead, RemoteDisconnected) as error:
        raise Transient(f"Interrupted response ({type(error).__name__})") from error
    except URLError as error:
        if isinstance(error.reason, OSError):
            network_failure(error.reason)
        raise RuntimeError(
            "Network failure of undetermined nature; not retried"
        ) from error
    except OSError as error:
        network_failure(error)


def index_once(index: str, current: str, retry: Retry) -> dict[str, object] | None:
    """Fallback to the project endpoint only when the version endpoint is unavailable."""
    base = f"https://{INDEX_HOSTS[index]}/pypi/tyrannis"
    missing = 0
    reason = Transient("Version metadata unavailable")
    for project_endpoint, url in (
        (False, f"{base}/{current}/json"),
        (True, f"{base}/json"),
    ):
        try:
            body = io.BytesIO()
            transfer(url, INDEX_HOSTS[index], body, retry)
        except Transient as error:
            # Do not bypass a server's Retry-After by using its other endpoint.
            if error.retry_after:
                raise
            reason = error
            if str(error) == "HTTP 404":
                missing += 1
            continue
        try:
            result = release.mapping(json.loads(body.getvalue()))
            info = release.mapping(result["info"])
            if info.get("name") != "tyrannis" or not isinstance(
                info.get("version"), str
            ):
                raise Inconsistent("Index returned inconsistent package identity")
            if project_endpoint:
                releases = release.mapping(result["releases"])
                if current not in releases:
                    if info["version"] == current:
                        raise Inconsistent(
                            "Index returned inconsistent release listing"
                        )
                    missing += 1
                    reason = Transient("Version not listed yet")
                    continue
                files = release.objects(releases[current])
            else:
                if info["version"] != current:
                    raise Inconsistent("Index returned a different version")
                files = release.objects(result["urls"])
        except (ValueError, TypeError, KeyError) as error:
            raise Inconsistent("Invalid index metadata/schema") from error
        return {"info": {"name": "tyrannis", "version": current}, "urls": files}
    if missing == 2:
        return None
    raise reason


def published_files(
    index: str,
    current: str,
    expected: dict[str, object],
    retry: Retry,
) -> dict[str, str]:
    result = index_once(index, current, retry)
    if result is None:
        return {}
    urls: dict[str, str] = {}
    try:
        for item in release.objects(result["urls"]):
            name, url = item["filename"], item["url"]
            if not isinstance(name, str) or name not in expected or name in urls:
                raise Inconsistent("Unexpected or duplicate distribution")
            if release.mapping(item["digests"])["sha256"] != expected[name]:
                raise Inconsistent(f"Index SHA-256 differs for {name}")
            if not isinstance(url, str):
                raise Inconsistent(f"Invalid download URL for {name}")
            location(url, FILE_HOSTS[index])
            urls[name] = url
    except (TypeError, KeyError, ValueError) as error:
        raise Inconsistent("Invalid distribution metadata/schema") from error
    return urls


def reconcile(index: str, expected: dict[str, object], retry: Retry) -> dict[str, str]:
    """Wait for propagation; never turn a failed query into evidence of absence."""
    latest: dict[str, str] | None = None
    seen: set[str] = set()

    def observe() -> dict[str, str]:
        nonlocal latest
        latest = None  # A later transport failure invalidates an earlier snapshot.
        latest = published_files(index, release.version(), expected, retry)
        if not seen.issubset(latest):
            latest = None
            raise Unconfirmed("Previously observed files disappeared; upload blocked")
        seen.update(latest)
        if set(latest) == set(expected):
            return latest
        raise Transient("Waiting for distribution propagation")

    try:
        urls = retry.run(f"{index} reconciliation", observe)
    except Unconfirmed:
        _ = retry.remaining()
        if not latest:
            raise
        urls = latest
    if set(urls) != set(expected):
        raise PartiallyPublished(
            f"Missing: {', '.join(sorted(set(expected) - set(urls)))}. "
            "Only verification may be retried; index absence cannot authorize resending"
        )
    return urls


def prepare(
    directory: Path,
    evidence: Path,
    destination: Path,
    index: str,
    *,
    prior_upload: bool,
) -> None:
    """Authorize an initial upload only; absence cannot prove a safe resend."""
    expected = release.mapping(
        release.mapping(json.loads((evidence / "release.json").read_text("utf-8")))[
            "files"
        ]
    )
    # A rerun may follow a killed uploader, even when no outcome was recorded.
    # A partial index snapshot likewise does not prove the other file was rejected.
    # In either case only the verifier may proceed, using the original artifact.
    if prior_upload:
        print("Previous attempt may have uploaded; verification only")
        release.output("upload", "false")
        return
    with budget(ABSENCE_WINDOW, ABSENCE_ATTEMPTS) as retry:
        fresh = retry.run(
            "Pre-upload reconciliation",
            lambda: published_files(index, release.version(), expected, retry),
        )
    if fresh:
        print("Distributions already visible; verification only, no resend")
        release.output("upload", "false")
        return
    if index == "pypi":
        import release_official

        release_official.guard()
    else:
        release.guard()
        if os.environ.get("PUBLISH_TESTPYPI") != "true":
            raise ValueError("Validation-only dispatch cannot authorize uploads")
    destination.mkdir(parents=True, exist_ok=False)
    for name in sorted(expected):
        _ = shutil.copyfile(directory / name, destination / name)
        if release.digest(destination / name) != expected[name]:
            raise Inconsistent(f"Staged SHA-256 differs for {name}")
    release.output("upload", "true")
    print("Prepared original distributions: " + ", ".join(sorted(expected)))


def download(url: str, host: str, path: Path, expected: object, retry: Retry) -> None:
    partial = path.with_name(path.name + ".part")
    try:
        with partial.open("wb") as stream:
            transfer(url, host, stream, retry)
        if release.digest(partial) != expected:
            raise Inconsistent(f"Downloaded SHA-256 differs for {path.name}")
        _ = retry.remaining()
        _ = partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def command(args: list[str], cwd: Path, retry: Retry, *, install: bool = False) -> None:
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    env.update(
        UV_HTTP_RETRIES="0", UV_HTTP_TIMEOUT=str(NETWORK_TIMEOUT), UV_NO_PROGRESS="1"
    )
    # A new process group lets deadline cancellation kill sdist build children too.
    with subprocess.Popen(
        args,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            log, _ = process.communicate(timeout=retry.remaining())
        except subprocess.TimeoutExpired as error:
            raise Unconfirmed(f"Deadline exhausted during {args[0]}") from error
        finally:
            # Also kill descendants if their parent exited before cancellation.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        _ = retry.remaining()
        if process.returncode:
            # Fail closed: only explicit transport evidence in a fetch failure
            # permits a fresh installation. Do not log raw URLs/credentials.
            deterministic = re.search(
                r"conflicting dependencies|no solution found|invalid metadata|"
                r"not a supported wheel|requires-python|syntaxerror|"
                r"build backend returned an error|failed to build|"
                r"certificate verify failed|401 unauthorized|403 forbidden",
                log,
                re.IGNORECASE,
            )
            transport = re.search(
                r"connection reset|connection refused|connection timed out|"
                r"operation timed out|network timeout|temporary failure in name resolution|"
                r"incomplete message|unexpected eof",
                log,
                re.IGNORECASE,
            )
            fetch = re.search(
                r"failed to (?:download|fetch)|error sending request",
                log,
                re.IGNORECASE,
            )
            server = re.search(
                r"\b(?:408|429|5\d\d)\s+(?:request timeout|too many requests|"
                r"internal server error|bad gateway|service unavailable|gateway timeout)",
                log,
                re.IGNORECASE,
            )
            if install and fetch and server and not deterministic:
                # uv's failure log does not expose response headers. Retrying here
                # could violate Retry-After; leave this verification unconfirmed.
                raise Unconfirmed(
                    f"Installer HTTP {server.group(0)}; Retry-After unavailable. "
                    "Retry verification later; no automatic installer retry."
                )
            if install and fetch and transport and not deterministic:
                raise Transient(f"Installer transport failure: {transport.group(0)}")
            raise RuntimeError(
                f"{args[0]} {args[1]} failed (exit {process.returncode}); "
                + (
                    f"deterministic failure ({deterministic.group(0)})"
                    if deterministic
                    else "functional/configuration failure or undetermined cause; not retried"
                )
            )


def installations(paths: list[Path], retry: Retry) -> None:
    checkout = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="tyrannis-published-") as directory:
        root = Path(directory)
        _ = shutil.copyfile(
            checkout / ".github/scripts/smoke_package.py", root / "smoke_package.py"
        )
        dependencies = root / "dependencies.txt"
        export = ["uv", "export", "--locked", "--offline", "--no-dev"]
        export += ["--no-emit-project", "--no-hashes", "--format", "requirements-txt"]
        command(export + ["--output-file", str(dependencies)], checkout, retry)
        for path in paths:
            # Each retry gets a new environment; approved distributions and prior
            # successful installations are outside this retry loop.
            def install(path: Path = path) -> None:
                with tempfile.TemporaryDirectory(
                    prefix="install-", dir=root
                ) as attempt:
                    environment = Path(attempt) / "venv"
                    python = str(environment / "bin/python")
                    command(
                        ["uv", "venv", "--python", "3.12", str(environment)],
                        root,
                        retry,
                    )
                    pip = ["uv", "pip", "install", "--python", python]
                    pip += ["--default-index", "https://pypi.org/simple"]
                    for requirements in (
                        ["-r", str(dependencies)],
                        ["--no-deps", str(path)],
                    ):
                        command(pip + requirements, root, retry, install=True)
                    command(["uv", "pip", "check", "--python", python], root, retry)
                    command([python, "-I", str(root / "smoke_package.py")], root, retry)

            retry.run(f"Install {path.name}", install)
            print(f"Installation and smoke passed: {path.name}", flush=True)


def verify_published(
    evidence: Path, destination: Path, index: str, retry: Retry
) -> None:
    current = release.version()
    manifest = release.mapping(
        json.loads((evidence / "release.json").read_text("utf-8"))
    )
    expected = release.mapping(manifest["files"])
    if (
        manifest.get("name") != "tyrannis"
        or manifest.get("version") != current
        or set(expected)
        != {f"tyrannis-{current}-py3-none-any.whl", f"tyrannis-{current}.tar.gz"}
        or any(
            not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in expected.values()
        )
    ):
        raise Inconsistent(
            "Approved manifest identity, filenames or hashes are invalid"
        )
    urls = reconcile(index, expected, retry)
    destination.mkdir(parents=True, exist_ok=False)
    paths: list[Path] = []
    for name, url in urls.items():
        path = (destination / name).resolve()
        retry.run(
            f"Download {name}",
            partial(download, url, FILE_HOSTS[index], path, expected[name], retry),
        )
        paths.append(path)
    installations(paths, retry)
    _ = retry.remaining()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--index", choices=tuple(INDEX_HOSTS), required=True)
    _ = parser.add_argument("--dist", type=Path, default=Path("release/dist"))
    _ = parser.add_argument("--evidence", type=Path, default=Path("release/evidence"))
    _ = parser.add_argument("--destination", type=Path, default=Path("published"))
    _ = parser.add_argument("--manifest-sha256", required=True)
    _ = parser.add_argument("--producer-attempt", required=True)
    _ = parser.add_argument("--mode", choices=("prepare", "verify"), default="verify")
    _ = parser.add_argument(
        "--upload-outcome", choices=("success", "failure", "skipped", ""), default=""
    )
    args = parser.parse_args()
    status, code = "Published and verified", 0
    print(
        f"Original upload action outcome: {args.upload_outcome or 'not attempted'}",
        flush=True,
    )
    try:
        try:
            release.verify(
                args.dist, args.evidence, args.manifest_sha256, args.producer_attempt
            )
        except OSError as error:
            raise Unconfirmed(
                "Original CI artifacts unavailable; restore access to the original "
                "artifact and rerun this job. Expired/deleted files cannot be rebuilt."
            ) from error
        except (ValueError, TypeError, KeyError) as error:
            raise Inconsistent(
                f"Original CI artifacts unavailable or invalid: {error}"
            ) from error
        if args.mode != "verify":
            prepare(
                args.dist,
                args.evidence,
                args.destination,
                args.index,
                prior_upload=os.environ["GITHUB_RUN_ATTEMPT"] != "1",
            )
            return
        with budget() as retry:
            verify_published(args.evidence, args.destination, args.index, retry)
    except PartiallyPublished as error:
        status, code = f"Partially published: {error}", 4
    except Unconfirmed as error:
        status, code = f"Publication unconfirmed: {error}", 2
    except Inconsistent as error:
        status, code = f"Publication inconsistent: {error}", 3
    except (
        RuntimeError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
        HTTPException,
        subprocess.SubprocessError,
    ) as error:
        # Boundary reporting only: unknown exceptions are never retried.
        reason = (
            str(error)
            if isinstance(error, (RuntimeError, ValueError))
            else f"{type(error).__name__}; configuration error or undetermined cause; not retried"
        )
        status, code = f"Verification failed: {reason}", 1
    print(status, flush=True)
    if "GITHUB_STEP_SUMMARY" in os.environ:
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open(
            "a", encoding="utf-8"
        ) as stream:
            _ = stream.write(f"{status}\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
