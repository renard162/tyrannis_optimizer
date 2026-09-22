import warnings
from pathlib import Path
from typing import Any

from pyspark.sql import SparkSession

from ...core.algorithm import AlgorithmBase, CostFunctionWrapperBase
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
        """
        Spark backend for distributed island-based optimization.

        `SparkDistributed` distributes the complete population among
        independent optimization islands. One processor and algorithm instance
        is created for each island, and the islands are submitted to Spark as
        independent partitions. Spark is responsible for scheduling those
        partitions on the executors available in the cluster.

        By default, the particles within each island are processed serially.
        A different `ProcessorBase` implementation can be supplied through the
        backend configuration, in which case the corresponding processor is used
        independently within each Spark island, allowing additional parallel
        processing of particles inside the executor running that island.

        Parameters
        ----------
        spark:
            Active Spark session used to distribute the optimization islands
            across the Spark cluster. The session must already be configured and
            available when the backend is created.

        n_executors:
            Number of optimization islands created by the backend. The same
            value is used as the number of Spark partitions when the island
            processors are submitted for execution.

            Despite the parameter name, this value does not configure or
            guarantee the number of physical Spark executors. Spark determines
            dynamically which executor runs each partition, so multiple islands
            may execute on the same executor and executors may process different
            islands over the lifetime of the Spark job.

            The value of `n_particles` supplied to the optimization is applied
            independently to every island, resulting in an initial distributed
            population of `n_particles * n_executors` particles. For efficient
            resource utilization, this value should normally be chosen according
            to the parallel execution capacity available in the Spark cluster.

        communication_port:
            TCP port used by the Spark communication layer for communication
            between the driver and the optimization islands during migration.
            It must be an integer between 1 and 65535. If `None`, port `18081`
            is used. The default may conflict with another service or process
            running on the cluster, in which case an explicit port should be
            provided.

        code_archive:
            Path to a Python archive containing the project code and any modules
            required by the Spark executors. If provided, the archive is
            distributed to Spark workers through the Spark context before the
            optimization starts. If `None`, no additional Python archive is
            distributed.

        Notes
        -----
        `SparkDistributed` is a general distributed execution model for
        population-based optimization. Instead of distributing individual
        particle evaluations while keeping a single optimization state on the
        driver, the complete optimization algorithm is replicated across
        independent islands. Each island owns its own population, algorithm
        state, processor, migration processor, and local optimization result.

        The driver coordinates the islands through the configured migration
        strategy. When migration is enabled, the migration driver establishes
        the communication infrastructure and applies the migration rules
        between islands. This allows the backend to implement island-based
        optimization while using Spark as the distributed execution
        environment.

        The default processor executes the particles of each island serially.
        This avoids introducing a second layer of parallelization inside the
        Spark task executing that island and is generally appropriate when the
        Spark cluster already provides the required degree of parallelism. If a
        different `ProcessorBase` is configured, that processor is replicated
        independently for each island and controls the particle-level
        processing within its corresponding Spark task. Consequently,
        `SparkDistributed` can also be combined with local parallel processors
        when additional parallelism inside each island is appropriate.

        A key characteristic of this architecture is that the optimization
        state is not transmitted between the driver and executors after every
        particle operation. Each island performs its optimization iterations
        locally, and the relevant population or state is transmitted between
        the distributed components primarily at iteration boundaries and at
        migration events. This drastically reduces communication and
        serialization overhead compared with architectures that repeatedly
        transfer particle data between the driver and workers during every
        iteration.

        The reduction in communication overhead comes at the cost of additional
        algorithmic complexity. The population is no longer represented as a
        single homogeneous collection: particles are associated with specific
        islands, and the optimization must account for their relationships
        through the configured migration strategy. The choice of migration
        topology, frequency, selection rules, and exchanged particles can
        therefore influence both the optimization behavior and the effective
        communication cost of the distributed execution.

        Because each island maintains an independent optimization state, the
        algorithm and processor are serialized and submitted as Spark partition
        data. The algorithm, processor, cost function, and all objects required
        for their execution must therefore be compatible with Spark's
        serialization mechanism and available in the executor environment. If
        required project modules are not already installed on the workers,
        `code_archive` can be used to distribute the corresponding Python code.

        The history logging system should be used with extreme care with
        `SparkDistributed`. Processor histories are returned from the Spark
        executors to the driver as part of the distributed results. Spark
        imposes limits on the size of data that can be transmitted between
        executors and the driver, and a sufficiently large optimization history
        can exceed these limits. In that situation, the Spark task may fail
        while returning its result, causing the complete optimization run to
        fail even if the optimization itself has otherwise completed
        successfully. Large populations, many iterations, or detailed history
        events can cause this limit to be reached particularly quickly.

        `SparkDistributed` is therefore particularly suitable when the cost of
        maintaining independent optimization islands is outweighed by the
        computational cost of the optimization itself. By keeping most
        computation local to each island and communicating only at iteration
        boundaries or migration events, the backend can scale population-based
        optimization across distributed resources while substantially reducing
        the communication overhead associated with fine-grained particle-level
        distribution.
        """
        if spark is None:
            raise ValueError("Spark session cannot be None.")

        if not isinstance(n_executors, int) or isinstance(n_executors, bool):
            raise TypeError("n_executors must be an integer.")

        if n_executors <= 0:
            raise ValueError("n_executors must be greater than zero.")

        if communication_port is not None:
            if not isinstance(communication_port, int) or isinstance(
                communication_port, bool
            ):
                raise TypeError("communication_port must be an integer or None.")

            if not 1 <= communication_port <= 65535:
                raise ValueError("communication_port must be between 1 and 65535.")

        self._spark = spark
        self._code_archive = Path(code_archive) if code_archive is not None else None
        self._n_executors = n_executors

        if communication_port is None:
            self._communication_port = 18081
            warnings.warn(
                message=(
                    f"WARNING: The default communication port "
                    f"({self._communication_port}) is being used. "
                    "This port may conflict with another service or process running "
                    "on the cluster, which can prevent SparkDistributed from "
                    "establishing the required communication channel. Specify "
                    "communication_port explicitly if this port is already in use."
                ),
                category=UserWarning,
                stacklevel=2,
            )
        else:
            self._communication_port = communication_port

        self._identifier = "SparkDistributed"
        self._cost_function_wrapper = SparkDistributedCostFunctionWrapper

        self._local_bests = {}

    def initialize_context(
        self,
        algorithm: AlgorithmBase,
        n_iter: int,
        n_particles: int,
        migration: MigrationDriverBase,
        processor: ProcessorBase[Any] | None,
        fitness_failure_strategy: str,
        history_config: HistoryConfig,
        seed: int | None,
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
            island_ids=island_ids, port=self._communication_port
        )

        driver_ip = self._spark.conf.get("spark.driver.host")

        self._migration.initialize_context(
            communication_driver=communication_driver,
            communication_processor_class=SparkCommunicationProcessor,
            communication_processor_kargs={
                "driver_ip": driver_ip,
                "port": self._communication_port,
            },
            history_config=self._history_config,
            seed=self._seed,
        )

        if self._history_config.history_enabled:
            warnings.warn(
                message=(
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
                category=UserWarning,
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

            self._spark.sparkContext.addPyFile(str(self._code_archive))

        if not self._processor.processors_pool:
            self.init_processors()

        self._migration.start()

        try:
            spark_context = self._spark.sparkContext

            processors = spark_context.parallelize(
                c=list(self._processor.processors_pool.values()),
                numSlices=self._n_executors,
            )

            results = processors.map(_run_processor).collect()

            self._local_bests = dict(results)

            self.update_result()

        finally:
            self._migration.stop()


def _run_processor(processor: ProcessorBase[Any]) -> tuple[str, ProcessorResult]:
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
