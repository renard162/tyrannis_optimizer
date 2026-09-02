import json
from typing import Any, cast

from pyspark.sql import SparkSession

from ...algorithm.base import CostFunctionWrapperBase
from ..processor.base import ProcessorBase
from .base import DistributedBackendBase


class SparkDistributedCostFunctionWrapper(CostFunctionWrapperBase):
    """Spark distributed cost-function wrapper."""


class SparkDistributed(DistributedBackendBase):
    """Spark backend for distributed processor execution."""

    def __init__(
        self,
        spark: SparkSession,
        n_executors: int | None = None,
    ) -> None:
        if spark is None:
            raise ValueError("Spark session cannot be None.")

        self._spark = spark

        if n_executors is None:
            self._n_executors = self._get_n_executors()
        else:
            self._n_executors = n_executors

        self._identifier = "SparkDistributed"
        self._cost_function_wrapper = SparkDistributedCostFunctionWrapper

        self._local_bests = {}

    def execute(self) -> None:
        if self._processor is None:
            raise RuntimeError("Processor cannot be None.")

        self._processor.create_processors_pool(self._n_executors)
        spark_context = self._spark.sparkContext
        processors = spark_context.parallelize(
            list(self._processor.processors_pool.values()),
            numSlices=self._n_executors,
        )

        results = processors.map(
            _run_processor,
        ).collect()

        self._local_bests = dict(results)
        self.update_result()

    def _get_n_executors(self) -> int:
        jsc = cast(Any, self._spark.sparkContext._jsc)
        return jsc.sc().getExecutorMemoryStatus().size


def _run_processor(
    processor: ProcessorBase,
) -> tuple[str, dict]:
    """
    Execute a processor inside a Spark executor.

    The processor enters its execution context only after Spark has
    deserialized it. Runtime resources are therefore created inside
    the executor and removed after execution.
    """
    processor.initialize_execution_context()

    try:
        processor.run()

        local_best = processor.local_best

        if not local_best:
            result: dict = {}
        else:
            result = json.loads(local_best)

        return processor.identifier, result

    finally:
        processor.finalize_execution_context()
