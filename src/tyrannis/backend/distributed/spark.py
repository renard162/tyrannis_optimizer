from typing import Any, cast

from pyspark.sql import SparkSession

from ..processor.base import ProcessorBase


class SparkDistributed:
    """Spark backend for distributed processor execution."""

    def __init__(
        self,
        spark: SparkSession,
        processor: ProcessorBase,
        n_executors: int | None = None,
    ) -> None:
        if spark is None:
            raise ValueError("Spark session cannot be None.")

        self._spark = spark
        self._processor = processor

        self._n_executors = n_executors
        if self._n_executors is None:
            self._n_executors = self._get_n_executors()

        self._processor.create_processors_pool(
            self._n_executors,
        )

        if len(self._processor.processors_pool) != self._n_executors:
            raise RuntimeError(
                "The number of processors in the processor pool must be "
                "identical to the number of Spark executors."
            )

        self._local_bests: dict[str, dict] = {}

    @property
    def processor(self) -> ProcessorBase:
        return self._processor

    @property
    def processors_pool(self) -> dict[str, ProcessorBase]:
        return self._processor.processors_pool

    @property
    def local_bests(self) -> dict[str, dict]:
        return self._local_bests.copy()

    @property
    def n_executors(self) -> int | None:
        return self._n_executors

    def execute(self) -> None:
        """
        Execute one processor on each Spark partition.

        Each processor in the processor pool represents one isolated
        optimization island. The processor is serialized by Spark,
        deserialized inside the executor, initialized for execution,
        executed, and finally returned to a serializable state.
        """
        spark_context = self._spark.sparkContext

        processors = spark_context.parallelize(
            list(self._processor.processors_pool.values()),
            numSlices=self._n_executors,
        )

        results = processors.map(
            _run_processor,
        ).collect()

        self._local_bests = dict(results)

    def _get_n_executors(self) -> int:
        jsc = cast(Any, self._spark.sparkContext._jsc)

        return jsc.sc().getExecutorMemoryStatus().size()


def _run_processor(
    processor: ProcessorBase,
) -> tuple[str, dict]:
    """
    Execute a processor inside a Spark executor.

    The processor must enter its execution context only after Spark
    has deserialized it. This is required because execution-context
    resources such as Events and thread/process pools must not be
    serialized or replicated.
    """
    processor.initialize_execution_context()

    try:
        processor.run()

        local_best = processor.local_best

        if not local_best:
            result: dict = {}
        else:
            import json

            result = json.loads(local_best)

        return processor.identifier, result

    finally:
        processor.finalize_execution_context()
