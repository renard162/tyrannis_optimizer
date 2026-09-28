import os
import sys
from importlib.util import find_spec
from typing import Literal, TypeAlias

MPIMode: TypeAlias = Literal["parallel", "distributed"]

_MPI_MODE_ENV = "TYRANNIS_MPI_MODE"

_HELP = """\
Tyrannis MPI compatibility launcher

Usage:
  tyrannis-mpi <parallel|distributed> <python-script> [arguments...]
  tyrannis-mpi --help

Requirements:
  Tyrannis must be installed with MPI support:

    pip install "tyrannis[mpi]"

  An MPI runtime such as OpenMPI or MPICH must also be available in the
  execution environment.

Modes:
  parallel
    Executes a script configured with the MPIParallel backend.

    MPIParallel keeps the optimization loop and global population on rank 0
    and distributes particle processing among the remaining MPI ranks.

    Parallelism is determined by the number and placement of MPI ranks
    configured by the MPI launcher. A typical configuration uses one rank per
    available CPU/core.

    Example:

      mpiexec --hostfile hostfile --map-by core -n <total-ranks> tyrannis-mpi parallel optimizer.py

    When the hostfile explicitly defines one slot per available CPU, the
    equivalent slot-based configuration can be used:

      mpiexec --hostfile hostfile --map-by slot -n <total-slots> tyrannis-mpi parallel optimizer.py

    Rank 0 is reserved for the Tyrannis driver. All remaining ranks process
    particles.

  distributed
    Executes a script configured with the MPIDistributed backend.

    MPIDistributed reserves rank 0 for the driver and assigns one complete
    optimization island to each remaining MPI rank. Each island executes its
    own Processor and optimization loop, including migration handling.

    The recommended topology is one MPI rank per machine/node:

      mpiexec --hostfile hostfile --map-by ppr:1:node -n <nodes> tyrannis-mpi distributed optimizer.py

    The number of islands is therefore:

      islands = MPI ranks - 1

    For example, four MPI ranks distributed as one rank per node produce one
    driver and three optimization islands.

    Parallelism inside each island is controlled by its Tyrannis Processor,
    independently of the number of MPI ranks.

The Python script must use the backend corresponding to the selected mode:
  parallel     -> MPIParallel
  distributed  -> MPIDistributed

Infrastructure configuration such as hosts, rank count, CPU binding, mapping,
and resource allocation remains the responsibility of the MPI launcher or
cluster resource manager.
"""


def _print_help() -> None:
    print(_HELP)


def _parse_mode() -> MPIMode:
    if len(sys.argv) < 2:
        _print_help()
        raise SystemExit(1)

    mode = sys.argv[1]

    if mode in {"-h", "--help"}:
        _print_help()
        raise SystemExit(0)

    if mode == "parallel":
        return "parallel"

    if mode == "distributed":
        return "distributed"

    print(f"Invalid MPI execution mode: {mode!r}\n", file=sys.stderr)
    _print_help()
    raise SystemExit(1)


def _check_mpi_dependency() -> None:
    if find_spec("mpi4py") is None:
        raise ImportError(
            "MPI support is not installed. "
            "Install Tyrannis with MPI support using "
            "'pip install \"tyrannis[mpi]\"'."
        )


def main() -> None:
    """Run a Python program using the Tyrannis MPI compatibility launcher."""
    mode = _parse_mode()

    if len(sys.argv) < 3:
        print("A Python script must be provided.\n", file=sys.stderr)
        _print_help()
        raise SystemExit(1)

    if sys.argv[2] in {"-h", "--help"}:
        _print_help()
        raise SystemExit(0)

    _check_mpi_dependency()

    os.environ[_MPI_MODE_ENV] = mode

    arguments = [sys.executable, "-m", "mpi4py.futures", *sys.argv[2:]]

    os.execv(sys.executable, arguments)
