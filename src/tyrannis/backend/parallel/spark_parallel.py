from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
import pandas as pd
from pyspark import cloudpickle
from pyspark.sql import SparkSession
from pyspark.sql.types import BinaryType, StructField, StructType

from ...algorithm.base import CostFunctionWrapperBase, ParticleBase
from .base import ParallelBackendBase


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

    def __init__(self, spark: SparkSession) -> None:
        if spark is None:
            raise ValueError("Spark session cannot be None.")

        self._spark = spark
        self._actual_iter = -1

        self._identifier = "SparkParallel"
        self._cost_function_wrapper = SparkParallelCostFunctionWrapper

    @property
    def actual_iter(self) -> int:
        return self._actual_iter

    @property
    def local_best(self) -> str:
        if self._algorithm.local_best is None:
            return ""

        return self._algorithm.local_best.dump()

    def execute(self) -> None:
        self.init_particles()

        for actual_iter in range(self._n_iter + 1):
            self._actual_iter = actual_iter

            self._algorithm.pre_iteration(actual_iter)

            new_particles_ids = self._algorithm.new_particles_id
            if new_particles_ids:
                initialized_particles = self._parallel_initialize_particles(
                    new_particles_ids
                )
                self._algorithm.update_population(initialized_particles)

            if actual_iter > 0:
                updated_particles = self._parallel_update_particles(
                    self._algorithm.population
                )
                self._algorithm.update_population(updated_particles)

            self._algorithm.post_iteration(actual_iter)

        self.update_result()

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

                particle = algorithm.get_unmodified_particle(particle_id)
                particle.candidate_fitness = np.inf

            particles.append(particle)

    serialized_particles = cloudpickle.dumps(particles)

    yield pd.DataFrame(
        {
            "particles": [serialized_particles],
        }
    )
