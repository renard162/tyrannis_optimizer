from __future__ import annotations

import pickle
import time
from collections.abc import Iterator
from functools import partial

import pyarrow as pa
from pyspark.sql import SparkSession
from pyspark.sql.types import BinaryType, StructField, StructType

from ..algorithm.base import AlgorithmBase, ParticleBase


class SparkParallel:
    """
    Execute an optimization algorithm using Spark task parallelism.

    The processor operates on a single population and does not implement
    processor pools, isolated populations, synchronization between islands,
    or particle migration.

    The algorithm instance maintained by the driver is the authoritative
    state of the optimization. For each parallelized stage, a serialized
    copy of the algorithm is made available to the Spark tasks. Updated
    particles are returned to the driver as serialized batches.

    Spark's DataFrame API and ``mapInArrow`` are used instead of RDDs or
    direct SparkContext access, allowing the processor to operate in Spark
    environments with restricted RDD support, such as Databricks Serverless.

    The execution lifecycle is:

        init_particles

        for actual_iter in range(n_iter + 1):
            pre_iteration

            initialize new particles  [parallelized]
            update_population

            update particles           [parallelized]
            update_population

            post_iteration

            update_status
    """

    _particle_schema = StructType(
        [
            StructField(
                "particles",
                BinaryType(),
                nullable=False,
            ),
        ]
    )

    def __init__(self, spark: SparkSession) -> None:
        if spark is None:
            raise ValueError("Spark session cannot be None.")

        self._spark = spark

        self._algorithm: AlgorithmBase | None = None
        self._n_iter = 0
        self._n_particles = 0
        self._fitness_failure_strategy = "invalidate"

        self._actual_iter = -1

    @property
    def algorithm(self) -> AlgorithmBase:
        if self._algorithm is None:
            raise RuntimeError(
                "The processor execution context has not been initialized."
            )

        return self._algorithm

    @property
    def actual_iter(self) -> int:
        return self._actual_iter

    @property
    def local_best(self) -> str:
        if self.algorithm.local_best is None:
            return ""

        return self.algorithm.local_best.dump()

    def initialize_context(
        self,
        algorithm: AlgorithmBase,
        n_iter: int,
        n_particles: int,
        fitness_failure_strategy: str = "invalidate",
        seed: int | None = None,
    ) -> None:
        """
        Initialize the processor execution context.

        Parameters
        ----------
        algorithm:
            Optimization algorithm to execute.
        n_iter:
            Number of optimization iterations. Iteration zero is reserved
            for population initialization.
        n_particles:
            Number of particles in the initial population.
        fitness_failure_strategy:
            Strategy used when particle evaluation fails. ``"invalidate"``
            assigns infinite fitness to the candidate particle, while
            ``"raise"`` propagates the exception.
        seed:
            Optional random seed used to configure the algorithm.
        """
        if n_iter < 0:
            raise ValueError("n_iter must be greater than or equal to zero.")

        if n_particles <= 0:
            raise ValueError("n_particles must be greater than zero.")

        if fitness_failure_strategy not in (
            "invalidate",
            "raise",
        ):
            raise ValueError(
                f"Invalid fitness failure strategy "
                f"{fitness_failure_strategy!r}. "
                "The strategy must be either 'invalidate' or 'raise'."
            )

        self._algorithm = algorithm
        self._n_iter = n_iter
        self._n_particles = n_particles
        self._fitness_failure_strategy = fitness_failure_strategy

        if seed is not None:
            self._algorithm.configure(
                identifier="SparkParallel|algorithm",
                seed=seed,
            )

    def run(self) -> None:
        """
        Execute the complete optimization lifecycle.

        The algorithm state remains exclusively on the driver between
        parallelized stages. Spark tasks operate on serialized copies of
        the algorithm and return updated particles to the driver in
        serialized batches.
        """
        total_start = time.perf_counter()

        self.init_particles()

        for actual_iter in range(self._n_iter + 1):
            self._actual_iter = actual_iter

            print(f"\n{'=' * 72}\nITERATION {actual_iter}\n{'=' * 72}")

            self.algorithm.pre_iteration(actual_iter)

            new_particles_ids = self.algorithm.new_particles_id

            if new_particles_ids:
                print(
                    f"[ITERATION {actual_iter}] "
                    f"INITIALIZE PARTICLES: "
                    f"{len(new_particles_ids)}"
                )

                initialized_particles = self._parallel_initialize_particles(
                    new_particles_ids
                )

                self.algorithm.update_population(initialized_particles)

            if actual_iter > 0:
                particle_ids = self.algorithm.population

                print(
                    f"[ITERATION {actual_iter}] UPDATE PARTICLES: {len(particle_ids)}"
                )

                updated_particles = self._parallel_update_particles(particle_ids)

                self.algorithm.update_population(updated_particles)

            self.algorithm.post_iteration(actual_iter)

            self.update_status()

        total_elapsed = time.perf_counter() - total_start

        print(f"\n{'=' * 72}\nTOTAL TIME: {total_elapsed:.6f}s\n{'=' * 72}")

    def init_particles(self) -> None:
        """
        Create the initial population on the driver.
        """
        if self.algorithm.population:
            return

        for p_idx in range(self._n_particles):
            self.algorithm.create_particle(
                identifier=f"SparkParallel|particle:{p_idx}",
            )

    def update_status(self) -> None:
        """
        Update driver-side execution status.

        Population-level state is maintained by the driver and therefore
        does not need to be synchronized from Spark tasks.
        """
        return

    def _parallel_initialize_particles(
        self,
        particle_ids: list[str],
    ) -> list[ParticleBase]:
        return self._parallel_process_particles(
            particle_ids=particle_ids,
            initialize_particle=True,
        )

    def _parallel_update_particles(
        self,
        particle_ids: list[str] | dict[str, ParticleBase],
    ) -> list[ParticleBase]:
        if isinstance(particle_ids, dict):
            particle_ids = list(particle_ids)

        return self._parallel_process_particles(
            particle_ids=particle_ids,
            initialize_particle=False,
        )

    def _parallel_process_particles(
        self,
        particle_ids: list[str],
        initialize_particle: bool,
    ) -> list[ParticleBase]:
        if not particle_ids:
            return []

        operation = "INITIALIZE" if initialize_particle else "UPDATE"

        print(f"\n[DRIVER] {operation} START")

        # --------------------------------------------------------------
        # Algorithm serialization
        # --------------------------------------------------------------
        algorithm_start = time.perf_counter()

        serialized_algorithm = pickle.dumps(
            self.algorithm,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

        algorithm_serialization_elapsed = time.perf_counter() - algorithm_start

        algorithm_size_mb = len(serialized_algorithm) / 1024 / 1024

        print(
            f"[DRIVER] Algorithm serialization: "
            f"{algorithm_serialization_elapsed:.6f}s | "
            f"size={algorithm_size_mb:.3f} MB"
        )

        # --------------------------------------------------------------
        # DataFrame creation
        # --------------------------------------------------------------
        start = time.perf_counter()

        particles_df = self._spark.createDataFrame(
            [(particle_id,) for particle_id in particle_ids],
            ["particle_id"],
        )

        create_dataframe_elapsed = time.perf_counter() - start

        print(f"[DRIVER] createDataFrame: {create_dataframe_elapsed:.6f}s")

        # --------------------------------------------------------------
        # Worker function
        #
        # Only the serialized algorithm is captured by the worker.
        # The SparkParallel instance and SparkSession are never sent
        # to the executors.
        # --------------------------------------------------------------
        worker = partial(
            _process_particle_batches,
            serialized_algorithm=serialized_algorithm,
            initialize_particle=initialize_particle,
            fitness_failure_strategy=self._fitness_failure_strategy,
        )

        # --------------------------------------------------------------
        # mapInArrow construction
        # --------------------------------------------------------------
        start = time.perf_counter()

        result_df = particles_df.mapInArrow(
            worker,
            schema=self._particle_schema,
        )

        map_elapsed = time.perf_counter() - start

        print(f"[DRIVER] mapInArrow construction: {map_elapsed:.6f}s")

        # --------------------------------------------------------------
        # Actual Spark execution
        # --------------------------------------------------------------
        start = time.perf_counter()

        rows = result_df.collect()

        collect_elapsed = time.perf_counter() - start

        print(f"[DRIVER] collect: {collect_elapsed:.6f}s")

        # --------------------------------------------------------------
        # Batch deserialization
        # --------------------------------------------------------------
        start = time.perf_counter()

        particles = []

        for row in rows:
            particles.extend(pickle.loads(row["particles"]))

        particle_deserialization_elapsed = time.perf_counter() - start

        print(
            f"[DRIVER] batch deserialization: {particle_deserialization_elapsed:.6f}s"
        )

        print(
            f"[DRIVER] {operation} END | "
            f"particles={len(particles)} | "
            f"batches={len(rows)} | "
            f"collect={collect_elapsed:.6f}s"
        )

        return particles


def _process_particle_batches(
    batches: Iterator[pa.RecordBatch],
    serialized_algorithm: bytes,
    initialize_particle: bool,
    fitness_failure_strategy: str,
) -> Iterator[pa.RecordBatch]:

    algorithm = pickle.loads(serialized_algorithm)

    particles = []

    for batch in batches:
        particle_ids = batch.column(batch.schema.get_field_index("particle_id"))

        for particle_id in particle_ids:
            particle_id = particle_id.as_py()

            try:
                if initialize_particle:
                    particle = algorithm.initialize_particle(particle_id)
                else:
                    particle = algorithm.update_particle(particle_id)

            except Exception:
                if fitness_failure_strategy == "raise":
                    raise

                particle = algorithm.get_unmodified_particle(particle_id)

                particle.candidate_fitness = float("inf")

            particles.append(particle)

    serialized_particles = pickle.dumps(
        particles,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

    yield pa.RecordBatch.from_arrays(
        [
            pa.array(
                [serialized_particles],
                type=pa.binary(),
            )
        ],
        names=["particles"],
    )
