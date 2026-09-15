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
        """
        Configure and execute a cost-function minimization problem.

        `Optimizer` is the main interface for configuring and executing an
        optimization problem in Tyrannis. The framework formulates optimization
        as the minimization of a cost function over a defined search space,
        combining a search space, an optimization algorithm, and an execution
        backend with the desired population size, number of iterations, and
        execution options.

        In optimization literature, optimization may refer to either minimization
        or maximization. Tyrannis, however, is designed specifically for
        minimization: the objective of every optimization is to find the candidate
        solution that minimizes the defined cost function.

        Parameters
        ----------
        space:
            Search space defining the domain over which the optimization is
            performed. In optimization literature, the search space is the set of
            candidate solutions considered by the optimization algorithm. It
            contains the `boundaries` or available options for the variables of
            the problem, defining the domain of the cost function, as well as the
            cost function itself, which is minimized by the optimization
            algorithm.

        algorithm:
            Optimization algorithm used to search for the best solution.

        n_iterations:
            Number of iterations performed by the optimization. Must be an integer
            greater than or equal to 1.

        n_particles:
            Number of particles in the population. Must be an integer greater than
            or equal to 1. For distributed backends, this value specifies the
            number of particles assigned to each island.

        backend:
            Backend defining how the optimization process is distributed and
            executed. It determines how the population, iterations, and associated
            optimization workload are organized across the available execution
            resources. Different backends provide different execution models,
            including local execution in a single process and distributed
            execution across multiple workers or islands. If ``None``, the
            optimization is executed locally.

        processor:
            Processor defining how the iterative optimization process is executed
            on each machine involved in the optimization. It determines whether
            the process is executed serially or with parallel processing within
            each machine. If ``None``, the iterations are executed serially. When
            parallel processing is desired, :class:`~tyrannis.processor.joblib.Joblib`
            is the recommended processor, as it is specifically designed to
            distribute the processing of the particles across the available
            execution resources of each machine.

        migration:
            Migration strategy defining how solutions are exchanged between
            islands during the optimization when using a backend that supports
            island-based optimization. If ``None`` and the selected backend uses
            a migration-based algorithm, the islands operate independently from
            one another, without exchanging solutions, and the final result is the
            best solution found among all islands.

        seed:
            Optional random seed used to make the optimization reproducible.

        fitness_failure_strategy:
            Strategy used when the cost function raises an exception while
            evaluating a particle. If ``"raise"``, the exception is propagated
            immediately and the optimization is interrupted. If ``"invalidate"``,
            the failed evaluation is suppressed and the particle is assigned a
            fitness of ``inf``, causing it to be treated as an invalid solution
            while allowing the optimization to continue. The default is
            ``"raise"``.

        history:
            History events to record during the optimization. If ``None``, no
            history is recorded. A single event name can be supplied to record
            that event, a list of event names can be supplied to record multiple
            events, or ``"all"`` can be used to record all available events.

            Available events are ``"migration"``, ``"pre_iteration"``,
            ``"new"``, ``"error"``, ``"iteration"``, ``"status"``, and ``"best"``.

        Methods
        -------
        fit()
            Execute the optimization and return the optimizer instance.

        save_history_csv(file_location, encoding="utf-8", sep=",", internal_state=False)
            Save the recorded optimization history to a CSV file. When
            ``internal_state=False``, the history contains the decoded variables
            in the representation defined by the search space. When
            ``internal_state=True``, the CSV also contains the internal particle
            state and the encoded variables used by the optimization algorithm.

        Attributes
        ----------
        best_solution:
            Best solution found by the optimization, decoded into the
            representation expected by the cost function. The solution is
            returned as a list when the cost function receives positional
            arguments and as a dictionary mapping argument names to values when
            it receives keyword arguments.

        best_fitness:
            Fitness value of the best solution found by the optimization.

        result_:
            Dictionary containing the best result found by the optimization. It
            contains the particle `identifier`, its `fitness`, and its decoded
            `variables`. The `variables` value is returned as a list when the
            cost function receives positional arguments and as a dictionary
            mapping argument names to values when the cost function receives
            keyword arguments. The remaining keys and their meaning are unchanged
            regardless of the cost function signature.

        history_columns:
            Column names used by the history generators and by
            :meth:`save_history_csv`.

        history_generator:
            Iterator that generates the registered optimization history as rows of
            tabular data, with variables represented in their decoded form.

        internal_history_generator_:
            Iterator that generates the registered optimization history as rows of
            tabular data, including the encoded variables and the particle's
            internal state.

        Notes
        -----
        The default configuration uses the local backend and serial processor,
        providing a simple sequential execution model. Computationally
        expensive cost functions or large populations can benefit from using
        a distributed backend together with an appropriate processor.

        The returned `best_solution` always uses the representation defined by
        the search space. The encoded representation used internally by the
        optimization algorithm is not exposed through this property.

        When ``fitness_failure_strategy="invalidate"`` is used, exceptions
        raised by the cost function are suppressed during fitness evaluation.
        Failed evaluations are assigned a fitness of ``inf`` and therefore
        cannot become the best solution, allowing the optimization to continue.
        This strategy should be used with caution, as errors in the cost
        function may remain unnoticed while the optimization completes
        normally. In such cases, the problem may only become apparent when the
        resulting fitness values are inspected.

        History recording is disabled by default. Enabling it can substantially
        increase memory usage, particularly for large populations or long
        optimizations, since multiple entries can be generated for each
        particle and iteration. As an example, an optimization with only
        50 particles and 300 iterations can produce more than 31_000 entries
        in the history table when all events are recorded.

        Examples
        --------
        >>> from tyrannis import Optimizer
        >>> from tyrannis.algorithm import PSO
        >>> from tyrannis.space import Continuous
        >>>
        >>> def cost_function(x, y):
        ...     return x**2 + y**2
        >>>
        >>> space = Continuous(
        ...     boundaries={"x": (-10, 10), "y": (-10, 10)},
        ...     cost_function=cost_function
        ... )
        >>> algorithm = PSO()
        >>> optimizer = Optimizer(
        ...     space=space,
        ...     algorithm=algorithm,
        ...     n_iterations=100,
        ...     n_particles=50
        ... )
        >>> optimizer.fit()
        Optimizer(...)
        >>> optimizer.best_solution
        {"x": 0.0, "y": 0.0}
        >>> optimizer.best_fitness
        0.0

        When the cost function uses keyword arguments, the categorical choices can
        be defined with a dictionary. In this case, the `variables` value in
        `result_` is also returned as a dictionary, with each key corresponding to
        an argument of the cost function:

        >>> def cost_function(color, size):
        ...     return color_factor(color) + size_factor(size)
        >>>
        >>> space = Categorical(
        ...     choices={
        ...         "color": ["red", "green", "blue"],
        ...         "size": ["small", "medium", "large"]
        ...     },
        ...     cost_function=cost_function
        ... )
        >>> optimizer = Optimizer(
        ...     space=space,
        ...     algorithm=algorithm,
        ...     n_iterations=100,
        ...     n_particles=50
        ... )
        >>> optimizer.fit()
        Optimizer(...)
        >>> optimizer.result_
        {
            "identifier": 17,
            "variables": {"color": "blue", "size": "medium"},
            "fitness": 0.0
        }

        In contrast, when the categorical choices are defined positionally, the
        same result is returned with `variables` represented as a list:

        >>> def cost_function(*data):
        ...     return color_factor(data[0]) + size_factor(data[1])
        >>>
        >>> space = Categorical(
        ...     choices=[
        ...         ["red", "green", "blue"],
        ...         ["small", "medium", "large"]
        ...     ],
        ...     cost_function=cost_function
        ... )
        >>> optimizer.fit()
        Optimizer(...)
        >>> optimizer.result_
        {
            "identifier": 17,
            "variables": ["blue", "medium"],
            "fitness": 0.0
        }
        """
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
