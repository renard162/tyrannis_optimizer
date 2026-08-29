from collections.abc import Callable

from pyspark import TaskContext
from pyspark.sql import SparkSession

from ..algorithm.base import AlgorithmBase
from .processor.base import ProcessorBase


class Spark:
    """
    Spark backend for distributed processor execution.

    Each Spark task creates exactly one Processor instance and executes it
    independently. The processor identifier is unique within the backend and
    is also used as the prefix for the identifiers of its particles.

    The algorithm and its fitness function must be serializable because the
    algorithm is transferred from the Spark driver to the executors.

    Parameters
    ----------
    spark:
        Spark session used to execute the processors.
    processor:
        Processor class to instantiate in each Spark task.
    algorithm:
        Algorithm instance used as the template for the processors.
    n_workers:
        Number of processors to execute.
    n_iter:
        Number of iterations executed by each processor.
    n_particles:
        Number of particles created by each processor.
    fitness_failure_strategy:
        Strategy used when the fitness function raises an exception.
    """

    def __init__(
        self,
        spark: SparkSession,
        processor: type[ProcessorBase],
        algorithm: AlgorithmBase,
        n_workers: int,
        n_iter: int,
        n_particles: int,
        fitness_failure_strategy: str = "invalidate",
    ) -> None:
        if n_workers < 1:
            raise ValueError("n_workers must be greater than or equal to 1.")

        if n_iter < 0:
            raise ValueError("n_iter must be greater than or equal to 0.")

        if n_particles < 1:
            raise ValueError("n_particles must be greater than or equal to 1.")

        self._spark = spark
        self._processor = processor
        self._algorithm = algorithm
        self._n_workers = n_workers
        self._n_iter = n_iter
        self._n_particles = n_particles
        self._fitness_failure_strategy = fitness_failure_strategy

        self._local_bests: dict[str, str] = {}

        self._job_group = f"tyrannis-spark-{id(self)}"

    @property
    def local_bests(self) -> dict[str, str]:
        return self._local_bests.copy()

    def run(self) -> dict[str, str]:
        spark_context = self._spark.sparkContext

        spark_context.setJobGroup(
            self._job_group,
            "Tyrannis processor execution",
            interruptOnCancel=True,
        )

        try:
            workers = spark_context.parallelize(
                range(self._n_workers),
                numSlices=self._n_workers,
            )

            results = workers.mapPartitions(
                lambda partition: self._run_processor(
                    partition,
                    self._processor,
                    self._algorithm,
                    self._n_iter,
                    self._n_particles,
                    self._fitness_failure_strategy,
                ),
            ).collect()

        finally:
            spark_context.clearJobGroup()

        self._local_bests = {
            processor_id: local_best for processor_id, local_best in results
        }

        return self.local_bests

    def stop(self) -> None:
        self._spark.sparkContext.cancelJobGroup(self._job_group)

    def _run_processor(
        partition,
        processor_class,
        algorithm,
        n_iter,
        n_particles,
        fitness_failure_strategy,
    ):
        worker_id = next(iter(partition))

        processor = processor_class(
            identifier=str(worker_id),
            algorithm=algorithm,
            n_iter=n_iter,
            n_particles=n_particles,
            fitness_failure_strategy=fitness_failure_strategy,
        )

        processor.run()

        yield str(worker_id), processor.local_best
