from __future__ import annotations

from typing import Any, Literal

from .core.algorithm import AlgorithmBase
from .core.backend import BackendBase
from .core.backend_distributed import DistributedBackendBase
from .core.backend_migration import MigrationDriverBase
from .core.orchestrator import OrchestratorBase
from .core.processor import ProcessorBase
from .core.space import SpaceBase


class Optimizer(OrchestratorBase):
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
        fitness_failure_strategy: str = "raise",
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
            Number of optimization iterations per call to `fit`. Must be an integer
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

            Local execution always uses isolated islands, including when an
            explicit ``Local()`` backend is supplied.

        seed:
            Optional random seed used to make the optimization reproducible.

        fitness_failure_strategy:
            Strategy used when the cost function raises an exception while evaluating
            a particle. Supported strategies include:

            - ``"raise"``: Propagates the exception and interrupts the optimization.
            (default)
            - ``"invalidate"``: Assigns ``inf`` fitness to the failed particle and
            continues the optimization.

        history:
            History events to record during the optimization. Supported options include:

            - ``None``: Disables history recording. (default)
            - ``"all"``: Records all available events. Cannot be used in a list.
            - ``"migration"``: Logs all particles that undergo migration.
            - ``"pre_iteration"``: Logs all particles before each iteration.
            - ``"new_particle"``: Logs newly created particles immediately after their
            fitness is calculated.
            - ``"error"``: Records fitness evaluation errors.
            - ``"iteration"``: Logs all particles at each iteration.
            - ``"status"``: Logs the final state of each iteration, including its worst
            and best particle and the historical best particle.
            - ``"best"``: Logs the historical best particle at the end of each
            iteration.

            A list of event names can also be supplied to record multiple events.

        Methods
        -------
        fit()
            Execute the optimization and return the optimizer instance.

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

        history:
            Independent optimization history provided by `OrchestratorBase`.
            ``history.columns`` and ``history.generator`` expose decoded rows;
            ``history.internal_columns_`` and ``history.internal_generator_``
            expose encoded variables and internal particle state. The object
            remains available before fitting and when recording is disabled,
            with configuration and columns but no rows. CSV export is provided
            by the history object through ``history.save_csv()``.

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

        The same ``history`` object is updated with the backend's accumulated
        records after each fit, including partial results published following
        an exception. Consecutive supported fits preserve earlier records.
        History reads use an independent decoding snapshot and do not modify
        the optimization's random state. With stochastic decoders, decoded
        values can differ between reads and from the original evaluated inputs.
        The history remains usable after the optimizer is released.

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
            "identifier": "island:0|particle:0",
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
            "identifier": "island:0|particle:0",
            "variables": ["blue", "medium"],
            "fitness": 0.0
        }
        """
        self._validate_common_arguments(n_iterations, n_particles)
        self._result = None
        self._fit_state: Literal["new", "unfinished", "completed"] = "new"
        self._configure_components(
            space=space,
            algorithm=algorithm,
            n_iterations=n_iterations,
            n_particles=n_particles,
            backend=backend,
            processor=processor,
            migration=migration,
            seed=seed,
            fitness_failure_strategy=fitness_failure_strategy,
            history=history
        )
        self._initialize_context()
        self._configure_history()

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
        """Execute the optimization and return the optimizer instance.

        Local and parallel executions continue from the last completed iteration.
        Iteration zero prepares the population only on the first call. A force
        stop remains active on later calls. Distributed execution supports one
        successful call. An execution interrupted by an exception requires a
        new optimizer.

        The existing history object is synchronized with the backend's complete
        published records on both successful and exceptional exits. Its decoding
        state is preserved, and repeated fits do not append duplicate records.
        """
        if self._fit_state == "unfinished":
            raise RuntimeError(
                "Cannot resume an interrupted fit. Create a new Optimizer."
            )
        if self._fit_state == "completed" and isinstance(
            self._backend, DistributedBackendBase
        ):
            raise RuntimeError(
                "Repeated fit() calls are supported only for local and parallel backends."
            )

        self._fit_state = "unfinished"
        try:
            self._backend.execute()
        finally:
            self._result = self._backend.result
            self._update_history(self._result.history)
        self._fit_state = "completed"

        return self
