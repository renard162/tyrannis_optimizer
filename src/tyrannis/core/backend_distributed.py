from abc import abstractmethod

import numpy as np

from .algorithm import AlgorithmBase
from .backend import BackendBase
from .backend_migration import MigrationDriverBase
from .processor import ProcessorBase


class DistributedBackendBase(BackendBase):
    """Base class for distributed optimization backends."""

    _local_bests: dict[str, dict[str, float | dict[str, float]] | None]
    _n_executors: int
    _migration_driver: MigrationDriverBase

    @abstractmethod
    def initialize_context(
        self,
        algorithm: AlgorithmBase,
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
            processor=processor,
            migration=migration,
            fitness_failure_strategy=fitness_failure_strategy,
            seed=seed,
        )
        """
        Call super of this method and call initialize context of
        _migration_driver setting up the communication module.
        
        If migration is None, set the default migration module.
        """

    def init_processors(self) -> None:
        if self._processor is None:
            raise RuntimeError("Processor is not initialized.")

        self._processor.create_processors_pool(
            self._n_executors,
        )

    def update_result(self) -> None:
        best_candidate = None
        best_fitness = np.inf

        for candidate in self._local_bests.values():
            if candidate is None:
                continue

            fitness = candidate["fitness"]

            if not isinstance(fitness, float):
                continue

            if fitness < best_fitness:
                best_fitness = fitness
                best_candidate = candidate

        if best_candidate is None:
            return

        self._result = {
            key: value
            for key, value in best_candidate.items()
            if key in self._result_keys
        }
