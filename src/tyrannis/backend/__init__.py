from .distributed.spark_distributed import SparkDistributed
from .local import Local
from .parallel.spark_parallel import SparkParallel

__all__ = [
    "Local",
    "SparkDistributed",
    "SparkParallel",
]
