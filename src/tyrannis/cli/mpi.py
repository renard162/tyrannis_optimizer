import os
import sys
from importlib.util import find_spec

if find_spec("mpi4py") is None:
    raise ImportError(
        "MPI support is not installed. "
        "Install Tyrannis with MPI support using 'pip install \"tyrannis[mpi]\"'."
    )


def main() -> None:
    """Run a Python program using the mpi4py.futures MPI launcher."""
    arguments = [
        sys.executable,
        "-m",
        "mpi4py.futures",
        *sys.argv[1:],
    ]

    os.execv(sys.executable, arguments)
