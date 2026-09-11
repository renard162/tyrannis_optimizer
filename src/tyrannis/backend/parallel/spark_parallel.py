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
        [
            StructField(
                "particles",
                BinaryType(),
                nullable=False,
            ),
        ]
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

            self._spark.sparkContext.addPyFile(
                str(self._code_archive),
            )

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
                        particle_ids=new_particles_ids,
                        initialize=True,
                    )
                    new_particles = self._parallel_initialize_particles(
                        new_particles_ids,
                    )
                    self.error_log(
                        actual_iter=actual_iter,
                        updated_particles=new_particles,
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
                        particle_ids=list(self._algorithm.population),
                        initialize=False,
                    )
                    processed_particles = self._parallel_update_particles(
                        self._algorithm.population,
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

        serialized_algorithm = cloudpickle.dumps(self._algorithm)
        particles_df = self._spark.createDataFrame(
            [(particle_id,) for particle_id in particle_ids],
            ["particle_id"],
        )

        fitness_failure_strategy = self._fitness_failure_strategy

        def worker(
            batches: Iterable[pd.DataFrame],
        ) -> Iterator[pd.DataFrame]:
            return _process_particle_batches(
                batches=iter(batches),
                serialized_algorithm=serialized_algorithm,
                initialize_particle=initialize_particle,
                fitness_failure_strategy=fitness_failure_strategy,
            )

        result_df = particles_df.mapInPandas(
            worker,
            schema=self._particle_schema,
        )

        rows = result_df.collect()
        particles: list[ParticleBase] = []

        for row in rows:
            particles.extend(cloudpickle.loads(row["particles"]))

        return particles


def _process_particle_batches(
    batches: Iterator[pd.DataFrame],
    serialized_algorithm: bytes,
    initialize_particle: bool,
    fitness_failure_strategy: str,
) -> Iterator[pd.DataFrame]:
    algorithm = cloudpickle.loads(
        serialized_algorithm,
    )

    particles: list[ParticleBase] = []

    for batch in batches:
        for particle_id in batch["particle_id"]:
            try:
                if initialize_particle:
                    particle = algorithm.initialize_particle(particle_id)
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

    yield pd.DataFrame(
        {
            "particles": [serialized_particles],
        }
    )
