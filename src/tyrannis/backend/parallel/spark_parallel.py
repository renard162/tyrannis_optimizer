from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from joblib.parallel import BACKENDS
from pyspark import cloudpickle
from pyspark.sql import SparkSession
from pyspark.sql.types import BinaryType, StructField, StructType

from ...core.algorithm import FITNESS_UNDEFINED, CostFunctionWrapperBase, ParticleBase
from ...core.backend_parallel import ParallelBackendBase


class SparkParallelCostFunctionWrapper(CostFunctionWrapperBase):
    """Spark parallel cost-function wrapper."""


class SparkParallel(ParallelBackendBase):
    _particle_schema = StructType(
        [StructField("particles", BinaryType(), nullable=False)]
    )

    def __init__(
        self,
        spark: SparkSession,
        spark_code_archive: str | Path | None = None,
        n_aux_jobs: int = -1,
        aux_backend: str = "sequential",
        aux_batch_size: int | str = "auto",
        aux_pre_dispatch: int | str = "2 * n_jobs",
    ) -> None:
        """
        Spark-based backend for fully parallel particle optimization.

        `SparkParallel` executes the optimization loop as a single fully
        parallelized process, without distinguishing or partitioning the
        population into optimization islands. Particle initialization and
        updates are distributed across Spark workers, while the complete
        optimization state and iteration lifecycle remain coordinated by the
        backend.

        Parameters
        ----------
        spark:
            Active Spark session used to distribute particle initialization and
            update operations across Spark workers. The session must already be
            configured and available when the backend is created.

        spark_code_archive:
            Path to a Python archive containing the project code and any modules
            required by the Spark workers. If provided, the archive is
            distributed to Spark workers through the Spark context before the
            optimization starts. If `None`, no additional Python archive is
            distributed.

        n_aux_jobs:
            Number of jobs used by the auxiliary Joblib parallel processing
            performed locally by the driver. Positive values specify the exact
            number of jobs, while negative values specify the number of jobs
            relative to the available logical CPUs. For example, `-1` uses all
            available logical CPUs, `-2` uses all CPUs except one, and `-3` uses
            all CPUs except two. The default is `-1`.

        aux_backend:
            Joblib backend used for auxiliary parallel processing. The default
            is `"sequential"`. Other available options are `"threading"`, which
            uses threads and is suitable for I/O-bound or GIL-releasing tasks,
            `"loky"`, which uses separate processes and is suitable for
            CPU-bound tasks, and `"multiprocessing"`, which also uses separate
            processes through Python's multiprocessing backend.

        aux_batch_size:
            Number of tasks grouped into each batch for auxiliary Joblib
            processing. If `"auto"`, Joblib determines the batch size
            automatically based on the observed execution time of the tasks.
            The default is `"auto"`.

        aux_pre_dispatch:
            Number of auxiliary Joblib batches that may be dispatched ahead of
            execution. An integer specifies the number of batches, while a
            string expression such as `"2 * n_jobs"` is evaluated by Joblib
            relative to the configured number of jobs. The default is
            `"2 * n_jobs"`.

        Notes
        -----
        `SparkParallel` executes the optimization loop in a fully parallelized
        process without distinction between islands. The complete population
        therefore belongs to a single optimization instance, and particle
        initialization and updates are distributed directly across Spark
        workers rather than being assigned to independent optimization
        processes or migration islands.

        This execution model is particularly sensitive to the communication
        overhead introduced by transferring the optimization state between the
        driver and Spark workers at each iteration. The algorithm is serialized
        and distributed to the workers, while the resulting particles are
        serialized and collected back by the driver after each parallel
        operation. Consequently, the cost of transmitting and serializing data
        can represent a substantial fraction of the total execution time,
        especially when particle evaluations are inexpensive.

        For this reason, `SparkParallel` is generally more appropriate when the
        optimization uses a large number of particles and a relatively small
        number of iterations. A large population provides enough parallel work
        to amortize the per-iteration communication overhead, while a smaller
        number of iterations limits how frequently the population and algorithm
        state must be transmitted between the driver and the Spark workers.
        Problems with few particles, many iterations, or very inexpensive cost
        functions may obtain little benefit from this backend because the
        communication and serialization overhead can dominate the computation.

        The auxiliary Joblib processing controlled by `n_aux_jobs`,
        `aux_backend`, `aux_batch_size`, and `aux_pre_dispatch` is independent
        of the main Spark parallelization. It is used for auxiliary processing
        associated with the iterations, particularly operations with lower
        computational cost that are not suitable for distribution through the
        main Spark particle-processing stage. Although these operations are
        individually inexpensive, they may occur for a very large number of
        particles, making their cumulative computational cost significant.

        Using a parallel Joblib backend for the auxiliary processing can reduce
        this cumulative cost when the number of auxiliary operations is large
        enough to justify the additional scheduling overhead. Conversely,
        `"sequential"` avoids creating additional local parallel workers and can
        be preferable when the auxiliary operations are sufficiently inexpensive
        or infrequent.

        The optimization algorithm is serialized with `cloudpickle` before
        particle processing is submitted to Spark. The algorithm and all objects
        required to initialize or update particles must therefore be compatible
        with `cloudpickle` serialization. Objects required by the cost function
        should likewise be serializable and available to the Spark workers. If
        such dependencies are defined in project modules that are not already
        available in the worker environment, `spark_code_archive` can be used
        to distribute the required Python code.

        The cost function and particle-processing operations execute on Spark
        workers rather than on the driver. They should therefore avoid relying
        on mutable driver-side state, local resources, or assumptions that
        execution occurs in the driver's process. Any state required by the
        evaluation should be explicitly serializable and available to the
        worker executing the particle.

        Unlike `ProcessPool`, `SparkParallel` does not maintain a pool of
        multiprocessing workers belonging to the optimization process.
        Scheduling and distribution of the primary particle workload are
        delegated to Spark. The Spark cluster configuration, worker
        availability, task scheduling, serialization, and data movement
        therefore have a direct influence on the performance of the backend.
        """
        if spark is None:
            raise ValueError("Spark session cannot be None.")
        if not isinstance(n_aux_jobs, int) or isinstance(n_aux_jobs, bool):
            raise TypeError("n_aux_jobs must be an integer.")

        if n_aux_jobs == 0:
            raise ValueError("n_jobs cannot be zero.")

        if not isinstance(aux_backend, str):
            raise TypeError("aux_backend must be a string.")

        if aux_backend not in BACKENDS:
            available_backends = ", ".join(sorted(BACKENDS))
            raise ValueError(
                f"Invalid Joblib backend {aux_backend!r}. "
                f"Available backends are: {available_backends}."
            )

        if isinstance(aux_batch_size, bool):
            raise TypeError("aux_batch_size must be a positive integer or 'auto'.")

        if isinstance(aux_batch_size, int):
            if aux_batch_size <= 0:
                raise ValueError("aux_batch_size must be greater than zero.")
        elif aux_batch_size != "auto":
            raise ValueError("aux_batch_size must be a positive integer or 'auto'.")

        if isinstance(aux_pre_dispatch, bool):
            raise TypeError("pre_dispatch must be a positive integer or a string.")

        if isinstance(aux_pre_dispatch, int):
            if aux_pre_dispatch <= 0:
                raise ValueError("aux_pre_dispatch must be greater than zero.")
        elif not isinstance(aux_pre_dispatch, str):
            raise TypeError("aux_pre_dispatch must be a positive integer or a string.")

        self._n_process = n_aux_jobs
        self._joblib_backend = aux_backend
        self._batch_size = aux_batch_size
        self._pre_dispatch = aux_pre_dispatch

        self._spark = spark
        self._code_archive = (
            Path(spark_code_archive) if spark_code_archive is not None else None
        )

        self._identifier = "SparkParallel"
        self._cost_function_wrapper = SparkParallelCostFunctionWrapper

    def execute(self) -> None:
        if self._code_archive is not None:
            if not self._code_archive.is_file():
                raise FileNotFoundError(
                    f"Spark code archive not found: {self._code_archive}"
                )

            self._spark.sparkContext.addPyFile(str(self._code_archive))

        self.init_particles()

        with Parallel(
            n_jobs=self._n_process,
            backend=self._joblib_backend,
            batch_size=cast(str, self._batch_size),
            pre_dispatch=cast(str, self._pre_dispatch),
            return_as="list",
        ) as parallel:
            for actual_iter in range(self._n_iter + 1):
                self._algorithm.pre_iteration(actual_iter)
                self.pre_iteration_log(actual_iter)

                new_particles_ids = self._algorithm.new_particles_id

                if new_particles_ids:
                    self._algorithm.create_random_cache(
                        particle_ids=new_particles_ids, initialize=True
                    )
                    new_particles = self._parallel_initialize_particles(
                        new_particles_ids
                    )
                    self.error_log(
                        actual_iter=actual_iter, updated_particles=new_particles
                    )
                    new_particles = parallel(
                        delayed(self._algorithm.consolidate_new_particles)(particle)
                        for particle in new_particles
                    )
                    self.new_particle_log(
                        actual_iter=actual_iter,
                        new_particles=cast(Iterable[ParticleBase], new_particles),
                    )
                    self._algorithm.update_population(
                        cast(Iterable[ParticleBase], new_particles)
                    )

                if actual_iter > 0:
                    self._algorithm.create_random_cache(
                        particle_ids=list(self._algorithm.population), initialize=False
                    )
                    processed_particles = self._parallel_update_particles(
                        self._algorithm.population
                    )
                    self.error_log(
                        actual_iter=actual_iter, updated_particles=processed_particles
                    )
                    self._algorithm.update_population(processed_particles)

                    if self._algorithm.double_particle_check:
                        self._algorithm.inter_iteration(actual_iter)
                        self._algorithm.create_random_cache(
                            particle_ids=list(self._algorithm.population),
                            initialize=False,
                        )
                        processed_particles = self._parallel_update_particles(
                            self._algorithm.population, second_update=True
                        )
                        self.error_log(
                            actual_iter=actual_iter,
                            updated_particles=processed_particles,
                        )
                        self._algorithm.update_population(processed_particles)

                self._algorithm.post_iteration(actual_iter)
                self.iteration_log(actual_iter)

                self.update_result()
                self.best_log(actual_iter)

    def _parallel_initialize_particles(
        self, particle_ids: list[str]
    ) -> list[ParticleBase]:
        return self._parallel_process_particles(
            particle_ids=particle_ids, initialize_particle=True
        )

    def _parallel_update_particles(
        self,
        particle_ids: list[str] | dict[str, ParticleBase],
        second_update: bool = False,
    ) -> list[ParticleBase]:
        if isinstance(particle_ids, dict):
            particle_ids = list(particle_ids)

        return self._parallel_process_particles(
            particle_ids=particle_ids,
            initialize_particle=False,
            second_update=second_update,
        )

    def _parallel_process_particles(
        self,
        particle_ids: list[str],
        initialize_particle: bool,
        second_update: bool = False,
    ) -> list[ParticleBase]:
        if not particle_ids:
            return []

        serialized_algorithm = cloudpickle.dumps(self._algorithm)
        particles_df = self._spark.createDataFrame(
            [(particle_id,) for particle_id in particle_ids], ["particle_id"]
        )

        fitness_failure_strategy = self._fitness_failure_strategy

        def worker(batches: Iterable[pd.DataFrame]) -> Iterator[pd.DataFrame]:
            return _process_particle_batches(
                batches=iter(batches),
                serialized_algorithm=serialized_algorithm,
                initialize_particle=initialize_particle,
                second_update=second_update,
                fitness_failure_strategy=fitness_failure_strategy,
            )

        result_df = particles_df.mapInPandas(worker, schema=self._particle_schema)

        rows = result_df.collect()
        particles: list[ParticleBase] = []

        for row in rows:
            particles.extend(cloudpickle.loads(row["particles"]))

        return particles


def _process_particle_batches(
    batches: Iterator[pd.DataFrame],
    serialized_algorithm: bytes,
    initialize_particle: bool,
    second_update: bool,
    fitness_failure_strategy: str,
) -> Iterator[pd.DataFrame]:
    algorithm = cloudpickle.loads(serialized_algorithm)

    particles: list[ParticleBase] = []

    for batch in batches:
        for particle_id in batch["particle_id"]:
            try:
                if initialize_particle:
                    particle = algorithm.initialize_particle(particle_id)
                elif second_update:
                    particle = algorithm.second_update_particle(particle_id)
                else:
                    particle = algorithm.update_particle(particle_id)

                candidate_fitness = particle.candidate_fitness

                if (candidate_fitness is not None) and np.isnan(candidate_fitness):
                    algorithm.population[particle_id].error_fitness = candidate_fitness
                    raise ValueError(
                        f"Cost function returned NaN for particle '{particle_id}'. "
                        "NaN is an invalid cost function result."
                    )

            except Exception:
                if fitness_failure_strategy == "raise":
                    raise

                particle = algorithm.population[particle_id]
                particle.candidate_fitness = np.inf
                if particle.error_fitness is None:
                    particle.error_fitness = FITNESS_UNDEFINED

            particles.append(particle)

    serialized_particles = cloudpickle.dumps(particles)

    yield pd.DataFrame({"particles": [serialized_particles]})
