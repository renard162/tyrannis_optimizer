from __future__ import annotations

from typing import Any

from .backend.local import Local
from .core.algorithm import AlgorithmBase
from .core.backend import BackendBase
from .core.backend_migration import MigrationDriverBase
from .core.processor import ProcessorBase
from .core.results import HistoryConfig
from .core.space import SpaceBase
from .migration.island_isolation import IslandIsolation
from .processor.serial import Serial


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
        history: str | list[str] | None = None,
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
        self._history_config = self._configure_particle_history(history)

        self._space.initialize_context(
            seed=self._seed,
        )

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
            history_config=self._history_config,
        )

        self._backend.initialize_context(
            algorithm=self._algorithm,
            n_iter=self._n_iterations,
            n_particles=self._n_particles,
            migration=self._migration,
            processor=self._processor,
            seed=self._seed,
            fitness_failure_strategy=self._fitness_failure_strategy,
            history_config=self._history_config,
        )

    @staticmethod
    def _configure_particle_history(history: str | list[str] | None) -> HistoryConfig:
        history_config = {
            "migration": False,
            "pre_iteration": False,
            "new_particle": False,
            "error": False,
            "iteration": False,
            "best": False,
        }

        if history is None:
            return HistoryConfig(**history_config)

        if isinstance(history, str):
            if history == "all":
                history_config = dict.fromkeys(history_config, True)

            elif history in history_config:
                history_config[history] = True

            else:
                raise ValueError(f"Invalid history event: {history!r}.")

            return HistoryConfig(**history_config)

        if isinstance(history, list):
            if "all" in history:
                raise ValueError(
                    "'all' cannot be used inside a history list. "
                    "Use history='all' instead."
                )

            for event in history:
                if event not in history_config:
                    raise ValueError(f"Invalid history event: {event!r}.")

                history_config[event] = True

            return HistoryConfig(**history_config)

        raise TypeError("history must be None, a string, or a list of strings.")

    @property
    def best_solution(self) -> Any:
        """Return the best solution found by the optimization."""

        if (self._result is None) or (self._result.result is None):
            raise AttributeError(
                "best_solution is not available before fit() is called."
            )

        variables = self._result.result["variables"]

        if not isinstance(variables, dict):
            raise TypeError("Optimization result variables must be a dictionary.")

        return self._space.decode(variables)

    @property
    def best_fitness(self) -> float:
        """Return the fitness of the best solution found by the optimization."""

        if (self._result is None) or (self._result.result is None):
            raise AttributeError(
                "best_fitness is not available before fit() is called."
            )

        fitness = self._result.result["fitness"]

        if not isinstance(fitness, float):
            raise TypeError("Optimization result fitness must be a float.")

        return fitness

    @property
    def result_(self) -> dict[str, Any] | None:
        """Return the best particle found by the optimization."""

        if (self._result is None) or (self._result.result is None):
            return None

        variables = self._result.result["variables"]

        if not isinstance(variables, dict):
            raise TypeError("Optimization result variables must be a dictionary.")

        return {
            "identifier": self._result.result["identifier"],
            "variables": self._space.decode(variables),
            "fitness": self._result.result["fitness"],
        }

    def fit(self) -> Optimizer:
        """Execute the optimization and return the optimizer instance."""

        self._backend.execute()

        backend_result = self._backend.result

        if backend_result is None:
            self._result = None
            return self

        self._result = backend_result

        return self
