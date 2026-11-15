"""Subprocess support for real MPI integration tests."""

import os
import subprocess
from contextlib import suppress
from tempfile import TemporaryFile

__all__ = ["_run_mpi"]


def _run_mpi(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    # A file keeps diagnostics available even if a rank outlives the launcher.
    with (
        TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as log,
        subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=os.name == "posix",
        ) as process,
    ):
        try:
            status = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            _ = log.seek(0)
            raise TimeoutError(
                f"{command[0]} timed out after {timeout}s:\n{log.read()}"
            ) from exc
        finally:
            if process.poll() is None and os.name == "posix":
                # Give the launcher time to finalize the ranks itself.
                process.terminate()
                with suppress(subprocess.TimeoutExpired):
                    _ = process.wait(timeout=10)
            try:
                if os.name == "posix" and process.returncode != 0:
                    # Ranks can have separate process groups in this private session.
                    cleanup = subprocess.run(
                        ["pkill", "-KILL", "-s", str(process.pid)],
                        check=False,
                        timeout=5,
                    )
                    if (
                        cleanup.returncode != 1
                    ):  # 1 means the session was already empty.
                        cleanup.check_returncode()
                elif process.poll() is None:
                    _ = subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                        check=True,
                        timeout=10,
                    )
            finally:
                if process.poll() is None:
                    process.kill()
        _ = log.seek(0)
        return subprocess.CompletedProcess(command, status, stdout=log.read())
