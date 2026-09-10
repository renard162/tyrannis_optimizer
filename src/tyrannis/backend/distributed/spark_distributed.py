import warnings
from pathlib import Path

from pyspark.sql import SparkSession

from ...core.algorithm import CostFunctionWrapperBase
from ...core.backend_distributed import DistributedBackendBase
from ...core.backend_migration import MigrationDriverBase
from ...core.processor import ProcessorBase
from ...core.results import HistoryConfig, ProcessorResult
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
        n_executors: int,
        communication_port: int | None = None,
        code_archive: str | Path | None = None,
    ) -> None:
        if spark is None:
            raise ValueError("Spark session cannot be None.")

        if (communication_port is not None) and (not 1 <= communication_port <= 65535):
            raise ValueError("communication_port must be between 1 and 65535.")

        self._spark = spark
        self._code_archive = Path(code_archive) if code_archive is not None else None
        self._n_executors = n_executors

        if communication_port is None:
            self._communication_port = 18081
            warnings.warn(
                (
                    f"WARNING: The default communication port ({self._communication_port}) is being used. "
                    "This port may conflict with another service or process running on the "
                    "cluster, which can prevent SparkDistributed from establishing the required "
                    "communication channel. Specify communication_port explicitly if this port "
                    "is already in use."
                ),
                UserWarning,
                stacklevel=2,
            )
        else:
            self._communication_port = communication_port

        self._identifier = "SparkDistributed"
        self._cost_function_wrapper = SparkDistributedCostFunctionWrapper

        self._local_bests = {}

    def initialize_context(
        self,
        algorithm,
        n_iter: int,
        n_particles: int,
        migration: MigrationDriverBase,
        processor: ProcessorBase | None,
        fitness_failure_strategy: str,
        history_config: HistoryConfig,
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
            history_config=history_config,
        )

        island_ids = [f"island:{idx}" for idx in range(self._n_executors)]

        communication_driver = SparkCommunicationDriver(
            island_ids=island_ids,
            port=self._communication_port,
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
            history_config=self._history_config,
        )

        if self._history_config.history_enabled:
            warnings.warn(
                (
                    "\n"
                    "============================================================\n"
                    "CRITICAL WARNING — SPARK HISTORY ENABLED\n"
                    "============================================================\n"
                    "Enabling optimization history with SparkDistributed may "
                    "cause the Spark task to FAIL.\n\n"
                    "Large histories can cause the result returned by a Spark "
                    "task to exceed the maximum response/message size "
                    "configured for Spark. If this limit is exceeded, the "
                    "Spark task may fail and the optimization run will not "
                    "complete successfully.\n\n"
                    "DO NOT ENABLE HISTORY FOR LARGE OPTIMIZATION RUNS "
                    "WITHOUT VERIFYING THAT THE EXPECTED RESULT SIZE IS "
                    "WITHIN THE LIMITS OF YOUR SPARK CONFIGURATION.\n"
                    "============================================================"
                ),
                UserWarning,
                stacklevel=3,
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

        if len(self._processor.processors_pool) == 0:
            self._processor.create_processors_pool(
                self._n_executors,
            )

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


def _run_processor(processor: ProcessorBase) -> tuple[str, ProcessorResult]:
    """
    Execute a processor inside a Spark executor.

    The processor enters its execution context only after Spark has
    deserialized it. Runtime resources are therefore created inside
    the executor and removed after execution.

    The complete `ProcessorResult` produced by the processor is returned
    to the driver. Result consolidation, including history transfer and
    selection of the globally best result, is performed by the distributed
    backend after all processors have completed.
    """
    processor.initialize_execution_context()

    try:
        processor.run()

        return processor.identifier, processor.result

    finally:
        processor.finalize_execution_context()
