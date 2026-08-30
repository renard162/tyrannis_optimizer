import json

from pyspark.sql import SparkSession

from .processor.base import ProcessorBase


class Spark:
    """
    Spark backend for distributed processor execution.

    A processor pool is created on the Spark driver with one processor for
    each Spark worker. During execution, each Spark worker receives one
    processor from this pool.

    Parameters
    ----------
    spark:
        Spark session used to execute the processors.
    processor:
        Configured processor instance used as the template for the processor
        pool.
    """

    def __init__(
        self,
        spark: SparkSession,
        processor: ProcessorBase,
    ) -> None:
        if spark is None:
            raise ValueError("Spark session cannot be None")

        self._spark = spark
        self._processor = processor

        self._n_workers = self._get_n_workers()

        self._processor.create_processors_pool(self._n_workers)

        if len(self._processor.processors_pool) != self._n_workers:
            raise RuntimeError(
                "The number of processors in the processor pool must be "
                "identical to the number of Spark workers."
            )

        self._local_bests: dict[str, dict] = {}

        self._job_group = f"tyrannis-spark-{id(self)}"

    @property
    def processor(self) -> ProcessorBase:
        return self._processor

    @property
    def processors_pool(self) -> list[ProcessorBase]:
        return self._processor.processors_pool

    @property
    def local_bests(self) -> dict[str, dict]:
        return self._local_bests.copy()

    @property
    def n_workers(self) -> int:
        return self._n_workers

    def run(self) -> None:
        spark_context = self._spark.sparkContext

        processors = spark_context.parallelize(
            self._processor.processors_pool,
            numSlices=self._n_workers,
        )

        results = processors.map(
            _run_processor,
        ).collect()

        self._local_bests = dict(results)

    def stop(self) -> None:
        self._spark.sparkContext.cancelJobGroup(self._job_group)

    def _get_n_workers(self) -> int:
        executor_infos = self._spark.sparkContext._jsc.sc().getExecutorMemoryStatus()

        return executor_infos.size()


def _run_processor(processor: ProcessorBase) -> tuple[str, dict]:
    processor.initialize_execution_context()

    try:
        processor.run()

        local_best = json.loads(processor.local_best)
        identifier = processor.identifier
    finally:
        processor.finalize_execution_context()

    return identifier, local_best
