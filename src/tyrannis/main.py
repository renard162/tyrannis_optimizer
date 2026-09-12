from __future__ import annotations

import json
from collections.abc import Iterator
from csv import writer
from pathlib import Path
from typing import Any

import numpy as np

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
        if backend is not None:
            self._migration = migration if migration is not None else IslandIsolation()
        else:
            self._migration = IslandIsolation()
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
            n_iter=self._n_iterations,
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
            "status": False,
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

    @property
    def history_columns(self) -> list[str]:
        """Return the column names generated by ``generate_history``."""

        return [
            "event",
            "iteration",
            "identifier",
            "origin",
            "destination",
            "fitness",
            *self._space.encoded_boundaries,
            "internal_state",
        ]

    @property
    def internal_history_generator_(self) -> Iterator[list[Any]]:
        """Generate optimization history as rows of tabular data.

        Only history records whose event is registered in
        ``self._history_config.registered_events`` are processed.
        """
        variable_names = list(self._space.encoded_boundaries)

        if self._result is None:
            return

        for history_entry in self._result.history:
            data = json.loads(history_entry)

            event = data.get("event")

            if event not in self._history_config.registered_events:
                continue

            particle = data.get("particle") or {}
            variables = particle.get("variables") or {}

            internal_state = {
                key: value
                for key, value in particle.items()
                if key not in {"identifier", "variables", "fitness"}
            }

            yield [
                event,
                data.get("iteration", None),
                particle.get("identifier", None),
                data.get("origin", None),
                data.get("destination", None),
                particle.get("fitness", None),
                *(variables.get(variable, None) for variable in variable_names),
                json.dumps(internal_state) if internal_state else None,
            ]

    @property
    def history_generator(self) -> Iterator[list[Any]]:
        """Generate optimization history as rows of tabular data.

        Only history records whose event is registered in
        ``self._history_config.registered_events`` are processed.
        """
        if self._result is None:
            return

        for history_entry in self._result.history:
            data = json.loads(history_entry)
            event = data.get("event")

            if event not in self._history_config.registered_events:
                continue

            particle = data.get("particle", {})
            variables = particle.get("variables", {})
            decoded_variables = self._space.decode(variables)

            if isinstance(decoded_variables, dict):
                decoded_variables = decoded_variables.values()

            yield [
                event,
                data.get("iteration", None),
                particle.get("identifier", None),
                data.get("origin", None),
                data.get("destination", None),
                particle.get("fitness", None),
                *decoded_variables,
            ]

    def save_history_csv(
        self,
        file_location: str | Path,
        encoding: str = "utf-8",
        sep: str = ",",
        internal_state: bool = False,
    ) -> None:
        """Save the registered optimization history to a CSV file."""

        file_location = Path(file_location)

        if file_location.suffix.lower() != ".csv":
            raise ValueError(
                f"History file must have a '.csv' extension: {file_location}"
            )

        if file_location.exists() and not file_location.is_file():
            raise ValueError(f"History file location is not a file: {file_location}")

        if not file_location.parent.exists():
            raise FileNotFoundError(
                f"Parent directory does not exist: {file_location.parent}"
            )

        if len(sep) != 1:
            raise ValueError(f"CSV separator must be a single character: {sep!r}")
        columns = self.history_columns
        columns = columns if internal_state else columns[:-1]
        fitness_index = columns.index("fitness")
        error_event = self._history_config.get_event("error")
        row_generator = (
            self.internal_history_generator_
            if internal_state
            else self.history_generator
        )

        with file_location.open("w", newline="", encoding=encoding) as file:
            csv_writer = writer(file, delimiter=sep)
            csv_writer.writerow(columns)

            for row in row_generator:
                fitness = row[fitness_index]

                if np.isinf(fitness) and (row[0] == error_event):
                    row[fitness_index] = "Raise"
                elif np.isinf(fitness):
                    row[fitness_index] = "Infinity"
                elif np.isnan(fitness):
                    row[fitness_index] = "NaN"
                else:
                    row[fitness_index] = str(fitness)

                csv_writer.writerow(row)
