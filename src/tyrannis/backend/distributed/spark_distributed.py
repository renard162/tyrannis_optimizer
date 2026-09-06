import json
from pathlib import Path
from typing import Any, cast

from pyspark.sql import SparkSession

from ...core.algorithm import CostFunctionWrapperBase
from ...core.backend_distributed import DistributedBackendBase
from ...core.backend_migration import MigrationDriverBase
from ...core.processor import ProcessorBase
from ...core.signals import LocalEvent
from .communication.spark_communication import (
    SparkCommunicationDriver,
    SparkCommunicationProcessor,
)


class SparkDistributedCostFunctionWrapper(CostFunctionWrapperBase):
    """Spark distributed cost-function wrapper."""


class SparkDistributed(DistributedBackendBase):
    """Spark backend for distributed processor execution."""

    def __init__(
        self,
        spark: SparkSession,
        n_executors: int | None = None,
        code_archive: str | Path | None = None,
        communication_port: int = 6062,
    ) -> None:
        if spark is None:
            raise ValueError("Spark session cannot be None.")

        if not 1 <= communication_port <= 65535:
            raise ValueError("communication_port must be between 1 and 65535.")

        self._spark = spark
        self._code_archive = Path(code_archive) if code_archive is not None else None
        self._communication_port = communication_port

        if n_executors is None:
            self._n_executors = self._get_n_executors()
        else:
            self._n_executors = n_executors

        self._identifier = "SparkDistributed"
        self._cost_function_wrapper = SparkDistributedCostFunctionWrapper

        self._local_bests = {}

    def initialize_context(
        self,
        algorithm,
        n_iter: int,
        n_particles: int,
        migration: MigrationDriverBase,
        processor: ProcessorBase | None = None,
        fitness_failure_strategy: str = "invalidate",
        seed: int | None = None,
    ) -> None:
        super().initialize_context(
            algorithm=algorithm,
            n_iter=n_iter,
            n_particles=n_particles,
            migration=migration,
            processor=processor,
            fitness_failure_strategy=fitness_failure_strategy,
            seed=seed,
        )

        island_ids = [f"island:{idx}" for idx in range(self._n_executors)]

        communication_stop_signal = LocalEvent()

        communication_driver = SparkCommunicationDriver(
            island_ids=island_ids,
            port=self._communication_port,
            stop_signal=communication_stop_signal,
        )

        driver_ip = self._spark.conf.get(
            "spark.driver.host",
        )

        migration.initialize_context(
            communication_driver=communication_driver,
            communication_processor_class=SparkCommunicationProcessor,
            communication_processor_kargs={
                "driver_ip": driver_ip,
                "port": self._communication_port,
            },
        )

    def execute(self) -> None:
        if self._processor is None:
            raise RuntimeError("Processor cannot be None.")

        if self._code_archive is not None:
            if not self._code_archive.is_file():
                raise FileNotFoundError(
                    f"Spark code archive not found: {self._code_archive}"
                )

            self._spark.sparkContext.addPyFile(
                str(self._code_archive),
            )

        self._processor.create_processors_pool(self._n_executors)

        self._migration.start()

        try:
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

        finally:
            self._migration.stop()

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
