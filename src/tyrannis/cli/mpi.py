import os
import sys
from importlib.util import find_spec
from typing import Literal, TypeAlias

if find_spec("mpi4py") is None:
    raise ImportError(
        "MPI support is not installed. "
        "Install Tyrannis with MPI support using 'pip install \"tyrannis[mpi]\"'."
    )

MPIMode: TypeAlias = Literal["parallel", "distributed"]

_MPI_MODE_ENV = "TYRANNIS_MPI_MODE"


def _usage() -> str:
    return "Usage: tyrannis-mpi <parallel|distributed> <python-script> [arguments...]"


def _parse_mode() -> MPIMode:
    if len(sys.argv) < 2:
        raise SystemExit(_usage())

    mode = sys.argv[1]

    if mode in {"-h", "--help"}:
        print(_usage())
        raise SystemExit(0)

    if mode == "parallel":
        return "parallel"

    if mode == "distributed":
        return "distributed"

    raise SystemExit(f"Invalid MPI execution mode {mode!r}.\n{_usage()}")


def main() -> None:
    """Run a Python program using the Tyrannis MPI execution launcher."""
    mode = _parse_mode()

    if len(sys.argv) < 3:
        raise SystemExit(_usage())

    os.environ[_MPI_MODE_ENV] = mode

    arguments = [sys.executable, "-m", "mpi4py.futures", *sys.argv[2:]]

    os.execv(sys.executable, arguments)
