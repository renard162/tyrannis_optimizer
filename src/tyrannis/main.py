from __future__ import annotations

from typing import Any

from .backend.local import Local
from .backend.migration.island_isolation import IslandIsolation
from .backend.processor.serial import Serial
from .core.algorithm import AlgorithmBase
from .core.backend import BackendBase
from .core.backend_migration import MigrationDriverBase
from .core.processor import ProcessorBase
from .core.space import SpaceBase


class Optimizer:
    """Orchestrate the optimization execution."""

    def __init__(
        self,
        space: SpaceBase,
        algorithm: AlgorithmBase,
        n_iterations: int,
        n_particles: int,
        backend: BackendBase | None = None,
        processor: ProcessorBase | None = None,
        migration: MigrationDriverBase | None = None,
        seed: int | None = None,
        fitness_failure_strategy: str = "invalidate",
    ) -> None:
        if not isinstance(n_iterations, int) or n_iterations < 1:
            raise ValueError(
                "n_iterations must be an integer greater than or equal to 1."
            )

        if not isinstance(n_particles, int) or n_particles < 1:
            raise ValueError(
                "n_particles must be an integer greater than or equal to 1."
            )

        self._space = space
        self._algorithm = algorithm
        self._n_iterations = n_iterations
        self._n_particles = n_particles
        self._backend = backend if backend is not None else Local()
        self._processor = processor if processor is not None else Serial()
        self._migration = migration if migration is not None else IslandIsolation()
        self._seed = seed
        self._fitness_failure_strategy = fitness_failure_strategy
        self._result = None

        self._space.initialize_context(seed=self._seed)

        self._algorithm.initialize_context(
            fitness_function=self._space,
            boundaries=self._space.encoded_boundaries,
        )

        self._processor.initialize_context(
            algorithm=self._algorithm,
            n_iter=self._n_iterations,
            n_particles=self._n_particles,
            migration_driver=self._migration,
            seed=self._seed,
            fitness_failure_strategy=self._fitness_failure_strategy,
        )

        self._backend.initialize_context(
            algorithm=self._algorithm,
            n_iter=self._n_iterations,
            n_particles=self._n_particles,
            migration=self._migration,
            processor=self._processor,
            seed=self._seed,
            fitness_failure_strategy=self._fitness_failure_strategy,
        )

    def fit(self) -> tuple[float, Any] | None:
        """Execute the optimization and return the fitness and decoded result."""

        self._backend.execute()

        backend_result = self._backend.result

        if backend_result is None:
            self._result = None
            return None

        if not isinstance(backend_result, dict):
            raise TypeError("Backend result must be a dictionary.")

        self._result = backend_result
        variables = self._result["variables"]

        if not isinstance(variables, dict):
            raise TypeError("Optimization result variables must be a dictionary.")

        fitness = self._result["fitness"]

        if not isinstance(fitness, float):
            raise TypeError("Optimization result fitness must be a float.")

        result_output = self._space.decode(variables)

        return fitness, result_output
