from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark import cloudpickle
from pyspark.sql import SparkSession
from pyspark.sql.types import BinaryType, StructField, StructType

from ...core.algorithm import CostFunctionWrapperBase, ParticleBase
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
        code_archive: str | Path | None = None,
    ) -> None:
        if spark is None:
            raise ValueError("Spark session cannot be None.")

        self._spark = spark
        self._code_archive = Path(code_archive) if code_archive is not None else None

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
                new_particles = [
                    self._algorithm.consolidate_new_particles(particle)
                    for particle in new_particles
                ]
                self.new_particle_log(
                    actual_iter=actual_iter,
                    new_particles=new_particles,
                )
                self._algorithm.update_population(new_particles)

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

            except Exception:
                if fitness_failure_strategy == "raise":
                    raise

                particle = algorithm.population[particle_id]
                # particle.candidate_variables = particle.variables
                particle.candidate_fitness = np.inf

            particles.append(particle)

    serialized_particles = cloudpickle.dumps(particles)

    yield pd.DataFrame(
        {
            "particles": [serialized_particles],
        }
    )
