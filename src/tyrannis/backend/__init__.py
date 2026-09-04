import importlib
import importlib.util
from typing import TYPE_CHECKING

# Fix linter checking
if TYPE_CHECKING:
    from .distributed.spark_distributed import SparkDistributed  # noqa: F401
    from .parallel.spark_parallel import SparkParallel  # noqa: F401
    # from .distributed.mpi_distributed import MPIDistributed
    # from .parallel.mpi_parallel import MPIParallel

"""Lazy import to permit module import without install all dependencies"""

_MODULES = {
    "SparkParallel": (".parallel.spark_parallel", "pyspark"),
    "SparkDistributed": (".distributed.spark_distributed", "pyspark"),
    # "MPIParallel": (".parallel.mpi_parallel", "mpi4py"),
    # "MPIDistributed": (".distributed.mpi_distributed", "mpi4py"),
}


def __getattr__(name: str):
    try:
        module_name, _ = _MODULES[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    module = importlib.import_module(module_name, __name__)
    return getattr(module, name)


__all__ = [  # pyright: ignore[reportUnsupportedDunderAll]
    name
    for name, (_, dependency) in _MODULES.items()
    if importlib.util.find_spec(dependency) is not None
]
