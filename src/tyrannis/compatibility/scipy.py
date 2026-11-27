from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence, Sized
from copy import deepcopy
from dataclasses import dataclass, fields, replace
from decimal import Decimal
from inspect import signature
from math import isfinite
from numbers import Real
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import Bounds, OptimizeResult, OptimizeWarning

from ..backend.local import Local
from ..core.algorithm import AlgorithmBase
from ..core.backend import BackendBase
from ..core.backend_distributed import DistributedBackendBase
from ..core.backend_migration import MigrationDriverBase
from ..core.backend_parallel import ParallelBackendBase
from ..core.orchestrator import OrchestratorBase
from ..core.processor import ProcessorBase
from ..space.continuous import Boundary, Continuous


@dataclass(frozen=True, slots=True)
class _MinimizerConfig:
    """Constructor settings or resolved settings for a single problem."""

    algorithm: AlgorithmBase
    n_iterations: int
    n_particles: int
    backend: BackendBase | None
    processor: ProcessorBase | None
    migration: MigrationDriverBase | None
    seed: int | None
    fitness_failure_strategy: str
    history: str | list[str] | None
    use_cache: bool
    cache_type: str
    cache_size: int
    maxiter: int | None
    tol: float | None
    target_fitness: float | None

    def __post_init__(self) -> None:
        """Give each configuration its own history selection."""
        if isinstance(self.history, list):
            object.__setattr__(self, "history", self.history.copy())


@dataclass(frozen=True, slots=True)
class _VariableLayout:
    """Original positions of free variables and indexed values of fixed ones."""

    free_indices: tuple[int, ...]
    fixed_values: tuple[tuple[int, Real | Decimal], ...]
    vector_dtype: type[np.float64 | np.object_]

    def reconstruct(
        self, free_values: Sequence[float]
    ) -> NDArray[np.float64 | np.object_]:
        """Restore one full vector in its original order in fresh storage."""
        size = len(self.free_indices) + len(self.fixed_values)
        vector = np.empty(size, dtype=self.vector_dtype)
        for index, value in zip(self.free_indices, free_values, strict=True):
            vector[index] = value
        for index, value in self.fixed_values:
            vector[index] = value
        return vector


@dataclass(frozen=True, slots=True)
class _Domain:
    """Positional intervals for Continuous and the layout for reconstruction."""

    free_bounds: list[Boundary]
    layout: _VariableLayout


type _StopReason = Literal[
    "callback", "saturation", "target_fitness", "force_stop", "tol", "budget"
]


@dataclass(slots=True)
class _ExecutionState:
    """Batch progress and variable layout for the current backend result and history.

    A batch counts once execute returns normally, even if force stop shortened it.
    The native stop flag is recorded independently of the final stopping reason.
    The reason remains unset if execution or the callback raises an unhandled error.
    """

    layout: _VariableLayout
    max_batches: int
    batches_completed: int = 0
    stop_reason: _StopReason | None = None
    force_stopped: bool = False


@dataclass(frozen=True, slots=True)
class _ScipyObjective:
    """Adapt a SciPy scalar objective to the positional inputs of Continuous."""

    fun: Callable[..., float]
    args: tuple[object, ...]
    layout: _VariableLayout

    def __call__(self, *free_values: float) -> float:
        return self.fun(self.layout.reconstruct(free_values), *self.args)


def _finite_real_array(values: ArrayLike, message: str) -> NDArray[np.object_]:
    """Validate finiteness and retain the real entries for interval comparisons."""
    array = np.array(values, dtype=object)
    for index, value in enumerate(array.flat):
        if isinstance(value, np.generic):
            value = value.item()
            array.flat[index] = value
        if not isinstance(value, (Real, Decimal)):
            raise TypeError(message)
        try:
            finite = isfinite(value)
        except (OverflowError, ValueError) as error:
            raise ValueError(message) from error
        if not finite:
            raise ValueError(message)
    return array


class ScipyMinimizer(OrchestratorBase):
    """
    Minimize a bounded continuous cost function with `scipy.optimize.minimize`.

    Pass a `ScipyMinimizer` instance as the `method` argument to `minimize`.
    The constructor configures the algorithm and execution settings. Each call
    to `minimize` supplies the problem, builds its continuous search space
    internally and returns a :class:`~scipy.optimize.OptimizeResult`.

    Parameters
    ----------
    algorithm:
        Optimization algorithm used to search for the best solution.

    n_iterations:
        Number of optimization iterations per batch. Must be an integer
        greater than or equal to 1. This value remains constant throughout a
        call to `minimize`, which can execute multiple batches.

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
        Defaults to ``None``.

    fitness_failure_strategy:
        Strategy used when the cost function raises an exception while evaluating
        a particle. Supported strategies include:

        - ``"raise"``: Propagates the exception and interrupts the optimization.
        (default)
        - ``"invalidate"``: Assigns ``inf`` fitness to the failed particle and
        continues the optimization.

        On a fully fixed domain the same policy applies to the single
        evaluation. An invalidated evaluation returns an unsuccessful result.

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

    use_cache:
        Whether cost-function evaluations should be cached. When
        ``False``, no cache is created or used. The default is ``False``.
        When enabled, repeated inputs can reuse a cached fitness.

    cache_type:
        Cache strategy to use when caching is enabled. Supported strategies are:

        - ``"lru"``: Least Recently Used cache. (default)
        - ``"disk"``: Temporary disk-backed runtime cache. Its contents are
        local to the current runtime and are not preserved through serialization.
        - ``"lfu"``: Least Frequently Used cache. Requires the optional
        dependencies for advanced caching.
        - ``"fifo"``: First In, First Out cache. Requires the optional dependencies
        for advanced caching.
        - ``"rr"``: Random Replacement cache. Requires the optional dependencies
        for advanced caching.

        Install the ``tyrannis[cache]`` extra for the advanced strategies.

    cache_size:
        Maximum cache size for in-memory caches, expressed as the maximum
        number of cached records. This parameter has no effect when
        ``cache_type="disk"``. The default is ``100_000``.

    maxiter:
        Global iteration budget for a population run. When supplied, it must
        be an integer greater than or equal to `n_iterations`. The budget
        schedules only whole batches, up to ``maxiter // n_iterations``. Any
        remainder is unused. Early termination can reduce the actual number of
        iterations. The default is ``None``, which schedules one batch if both
        additional stopping criteria are disabled. Enabling either criterion
        without specifying `maxiter` schedules ten batches and emits an
        ``OptimizeWarning``. Distributed backends require ``maxiter=None`` or
        ``maxiter=n_iterations``.

    tol:
        Absolute tolerance on the change in best fitness between consecutive
        batches. Starting with the second batch, stop if both best values are
        finite and their absolute difference is less than or equal to `tol`.
        Supply a finite, nonnegative value. Zero requires equal fitness values.
        The default, ``None``, disables this comparison.

    target_fitness:
        Stop at the end of any batch whose best fitness is less than or equal
        to this target. Supply a finite value or negative infinity. Positive
        infinity and NaN are rejected. The default, ``None``, disables the
        explicit target. A best fitness of negative infinity always terminates
        the search. When both `tol` and `target_fitness` are enabled, meeting
        either criterion ends the optimization.

    Attributes
    ----------
    history:
        Independent optimization history provided by `OrchestratorBase`.
        The property becomes available after a call has prepared its history.
        Access raises ``AttributeError`` immediately after construction or if a
        new call reaches the adapter but fails before preparing a new history.
        An earlier result keeps its own history in either case.

        ``history.columns`` and ``history.generator`` expose decoded rows.
        ``history.internal_columns_`` and ``history.internal_generator_``
        expose encoded variables and internal particle state. The history object
        provides CSV export and Pickle serialization through ``history.export()``.
        Load only trusted Pickle files, since deserialization can execute code.

    Arguments supplied by scipy.optimize.minimize
    ---------------------------------------------
    fun:
        Scalar cost function to minimize. It receives a fresh one-dimensional
        NumPy array containing all variables in their original order, followed
        by the positional arguments in `args`. Return a real scalar fitness.

    x0:
        Nonempty one-dimensional vector of finite real values within the
        inclusive bounds. It determines the original problem dimension for
        validating and broadcasting the bounds. The algorithm initializes its
        own population and gives no preference to `x0`.

    args:
        Additional positional arguments passed as ``fun(x, *args)``.
        The default is the empty tuple.

    bounds:
        Required lower and upper bounds for every variable. Supply a sequence
        of one pair per entry of `x0`, or a :class:`~scipy.optimize.Bounds`
        object whose lower and upper bounds broadcast to the shape of `x0`.
        Both endpoints must be defined, finite and real. Reversed intervals
        and values of `x0` outside their intervals are rejected. The
        `keep_feasible` flag of a `Bounds` object is silently ignored.

        Equal endpoints fix the variable at that value. Only originally free
        intervals enter the `Continuous` space. A free interval whose endpoints
        coincide after conversion to float64 is rejected. Objective inputs,
        callback vectors and result positions normally use float64. When needed
        to retain a higher-precision fixed value, reconstruction uses an array
        with ``dtype=object``. Free variables still use the precision of the
        continuous space.

    constraints:
        General constraints are ignored. A forwarded constraint object emits
        a ``RuntimeWarning`` if it has no length or its length is nonzero.
        ``None`` and empty collections produce no warning.

    jac, hess, hessp:
        Gradient, Hessian and Hessian-vector product information is unused.
        The adapter emits a separate ``RuntimeWarning`` for a truthy forwarded
        `jac` and for each of `hess` and `hessp` that differs from ``None``.
        SciPy forwards a callable gradient as supplied. With ``jac=True``,
        SciPy expects the objective to return a value and gradient and forwards
        a scalar objective wrapper and a callable derivative. SciPy can
        normalize a finite-difference gradient selection to ``None`` before
        dispatch, in which case the adapter issues no gradient warning.

    callback:
        Optional callable invoked by the orchestration process once per
        completed batch with a published solution, including the last batch.
        Native SciPy methods commonly invoke callbacks after each iteration.
        Here a batch may contain many iterations.

        If the signature has a single parameter named `intermediate_result`,
        the adapter calls ``callback(intermediate_result=result)``, where
        `result` is an `OptimizeResult` with the full best position `x` and its
        fitness `fun`. The parameter must accept a keyword argument. Otherwise,
        ``callback(xk)`` receives the full best position positionally. Each
        call receives an independent array including the fixed variables.
        Return values are ignored. Raise ``StopIteration`` to request a
        controlled stop. Other exceptions propagate to the caller.

    options:
        Any public constructor parameter can be overridden for the current
        optimization through the `options` dictionary of `minimize`. SciPy
        expands its entries into individual keyword arguments to the method.
        Options take precedence over constructor arguments, which take
        precedence over defaults. An explicit ``None`` overrides an earlier
        value wherever that parameter permits it. If `minimize` receives a
        `tol` value other than ``None``, SciPy supplies that option only if
        the dictionary omits it.
        Unknown solver options are ignored with an ``OptimizeWarning``. Each
        call validates its effective settings and preserves the base configuration.

    OptimizeResult
    --------------
    Every result includes `x`, `fun`, `success`, `status`, `message` and
    `history`. The availability of `nit` and `nfev` depends on the execution.

    x:
        Full position corresponding to the published best fitness, including
        fixed variables. Without a published solution, `x` is ``None``.
        On a fully fixed domain, `x` contains the only feasible position.

    fun:
        Fitness corresponding to `x`, or positive infinity when no solution
        was published. Finite values and negative infinity are valid. Positive
        infinity and NaN cannot yield a successful result.

    success:
        ``True`` exactly when `status` is zero.

    status:
        Integer termination code. A missing or invalid fitness takes precedence
        over every stopping reason. With a valid fitness, the batch checks, in
        order, are callback interruption, negative infinity, target attainment,
        algorithm force stop, tolerance and budget exhaustion.

        - ``0``: Valid fitness on a fully fixed domain, saturation at negative
          infinity, a reached target, a satisfied tolerance, or normal completion
          of the planned batches with both additional criteria disabled.
        - ``1``: The planned batches completed with valid fitness. At least
          one additional criterion was enabled and none was met.
        - ``2``: The algorithm requested a detectable force stop.
        - ``3``: No valid solution was found.
        - ``99``: The callback requested termination through ``StopIteration``.

    message:
        Human-readable description of the termination condition.

    history:
        The `OptimizationHistory` instance of this run, also accessible through
        the minimizer's `history` property. Even with recording disabled, it
        retains its configuration and column metadata and contains zero records.
        Variable columns represent only the free dimensions in their original
        relative order. The result position includes all original dimensions.

    nit:
        Conditional iteration count, excluding initialization at iteration zero.
        Available for known complete batches in local and parallel execution,
        including stops at a batch boundary. It is omitted when a force stop
        may have shortened a batch or no exact global count is available, as
        in distributed execution. A fully fixed domain returns ``nit=0``.

    nfev:
        Conditional objective evaluation count. A fully fixed domain returns
        ``nfev=1``. Population runs omit this field. The result provides no
        derivative or constraint fields.

    Notes
    -----
    The default ``None`` settings for `backend`, `processor` and `migration`
    select local execution, serial processing and isolated islands.

    Each independent population optimization starts with a new positional
    `Continuous` space, runtime cache configuration, history and seeded execution
    context. The algorithm and supplied processor are deep-copied before
    initialization and must support this operation. Supplied backends and
    migration drivers are retained by reference and reinitialized according
    to their execution model. External resources such as a Spark session
    remain attached to the supplied backend. Use the same minimizer or shared
    mutable backend and migration components only in sequential calls.
    Concurrent and reentrant calls with shared state are unsupported. When
    evaluations cross process boundaries, the objective and its arguments must
    support the execution model's serialization mechanism.

    Consecutive batches continue the same population, algorithm state and
    random streams. Runtime caches are local to the evaluating process and
    their contents are excluded from serialization. After each backend call,
    the same history object publishes the accumulated records, including any
    partial results the backend made available before an exception. Publishing
    the cumulative records preserves earlier entries without duplicating them.
    A subsequent optimization creates a distinct history. Earlier histories
    retain their decoding metadata and remain usable after the minimizer is
    released.

    Distributed backends support one batch and require both `tol` and
    `target_fitness` to be ``None``. Each island can stop independently.
    Local execution observes force stop on the executing processor's algorithm.
    Parallel execution observes the backend algorithm. A custom backend
    without a global stop signal may complete a call without exposing an
    algorithm force stop to the minimizer.

    When all variables are fixed, the adapter evaluates the only feasible
    position once, without initializing the algorithm, processor, backend or
    migration driver. It invokes no callback. The history uses the requested
    event selection and contains zero records. A valid fitness completes the
    optimization successfully, regardless of the additional stopping criteria.

    When ``fitness_failure_strategy="invalidate"`` is used, exceptions
    raised by the cost function are suppressed during fitness evaluation.
    Failed evaluations are assigned a fitness of ``inf`` and therefore
    cannot yield a successful solution, allowing population execution to continue.
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
    >>> from scipy.optimize import minimize
    >>> from tyrannis.algorithm import PSO
    >>> from tyrannis.compatibility.scipy import ScipyMinimizer
    >>>
    >>> def cost_function(x):
    ...     return x[0]**2 + x[1]**2
    >>>
    >>> method = ScipyMinimizer(algorithm=PSO(), n_iterations=20, n_particles=50)
    >>> x0 = [0.0, 0.0]
    >>> bounds = [(-10, 10), (-10, 10)]
    >>> result = minimize(cost_function, x0=x0, bounds=bounds, method=method)
    >>> best_position, best_fitness = result.x, result.fun

    With 20 iterations per batch, a budget of 65 schedules at most three
    batches. For this call, stop early if the fitness tolerance is met and
    record the best particle at each iteration.

    >>> result = minimize(
    ...     cost_function, x0=x0, bounds=bounds, method=method,
    ...     options={"maxiter": 65, "tol": 1e-6, "history": "best"}
    ... )
    >>> from tyrannis.core.results import OptimizationHistory
    >>> history = result.history
    >>> columns = history.columns
    >>> first_row = next(history.generator, None)
    >>> history.export("history.csv")
    >>> history.export("history.pkl", format="pickle")
    >>> restored = OptimizationHistory.load("history.pkl")

    Fix the second coordinate at 2 and pass the desired offset through
    `args`. The callback receives the full best position after the first
    batch and requests termination before the next batch.

    >>> def shifted_cost(x, offset):
    ...     return (x[0] - offset)**2 + x[1]**2
    >>>
    >>> def stop_after_batch(intermediate_result):
    ...     raise StopIteration
    >>>
    >>> result = minimize(
    ...     shifted_cost, x0=[0, 2], args=(1.5,), bounds=[(-10, 10), (2, 2)],
    ...     method=method, callback=stop_after_batch, options={"maxiter": 60}
    ... )
    >>> result.x.shape
    (2,)
    >>> float(result.x[1])
    2.0
    >>> result.status
    99
    """

    def __init__(
        self,
        algorithm: AlgorithmBase,
        n_iterations: int,
        n_particles: int,
        backend: BackendBase | None = None,
        processor: ProcessorBase | None = None,
        migration: MigrationDriverBase | None = None,
        seed: int | None = None,
        fitness_failure_strategy: str = "raise",
        history: str | list[str] | None = None,
        use_cache: bool = False,
        cache_type: str = "lru",
        cache_size: int = 100_000,
        maxiter: int | None = None,
        tol: float | None = None,
        target_fitness: float | None = None,
    ) -> None:
        self._base_config = _MinimizerConfig(
            algorithm=algorithm,
            n_iterations=n_iterations,
            n_particles=n_particles,
            backend=backend,
            processor=processor,
            migration=migration,
            seed=seed,
            fitness_failure_strategy=fitness_failure_strategy,
            history=history,
            use_cache=use_cache,
            cache_type=cache_type,
            cache_size=cache_size,
            maxiter=maxiter,
            tol=tol,
            target_fitness=target_fitness,
        )
        self._execution: _ExecutionState | None = None
        self._result = None

    def _resolve_options(self, **options: object) -> _MinimizerConfig:
        """Merge expanded SciPy options and validate the resulting configuration."""
        names = {field.name for field in fields(_MinimizerConfig)}
        unknown = [name for name in options if name not in names]
        if unknown:
            message = f"Unknown solver options: {', '.join(unknown)}"
            warnings.warn(message, OptimizeWarning, stacklevel=3)
        overrides = {name: value for name, value in options.items() if name in names}
        config = replace(self._base_config, **overrides)
        self._validate_config(config)
        self._validate_backend(config)
        return config

    @staticmethod
    def _validate_config(config: _MinimizerConfig) -> None:
        """Check iteration and population counts after applying all overrides."""
        OrchestratorBase._validate_common_arguments(
            config.n_iterations, config.n_particles
        )
        if config.maxiter is not None:
            if isinstance(config.maxiter, bool) or not isinstance(config.maxiter, int):
                raise ValueError("maxiter must be an integer.")
            if config.maxiter < config.n_iterations:
                raise ValueError(
                    "maxiter must be greater than or equal to n_iterations."
                )

    @staticmethod
    def _validate_backend(config: _MinimizerConfig) -> None:
        """Reject iteration and stopping settings incompatible with island execution."""
        if isinstance(config.backend, DistributedBackendBase):
            if config.maxiter is not None and config.maxiter != config.n_iterations:
                raise ValueError(
                    "Distributed backends require maxiter == n_iterations."
                )
            if config.tol is not None or config.target_fitness is not None:
                raise ValueError(
                    "Distributed backends require tol and target_fitness to be None."
                )

    @staticmethod
    def _warn_unsupported_arguments(
        constraints: object, jac: object, hess: object, hessp: object
    ) -> None:
        """Warn about nonempty constraints and derivative information sent by SciPy."""
        has_constraints = constraints is not None and (
            not isinstance(constraints, Sized) or len(constraints) > 0
        )
        for supplied, message in (
            (has_constraints, "cannot handle constraints."),
            (bool(jac), "does not use gradient information (jac)."),
            (hess is not None, "does not use Hessian information (hess)."),
            (
                hessp is not None,
                "does not use Hessian-vector product information (hessp).",
            ),
        ):
            if supplied:
                message = f"Method ScipyMinimizer {message}"
                warnings.warn(message, RuntimeWarning, stacklevel=3)

    @staticmethod
    def _prepare_domain(x0: ArrayLike, bounds: Bounds | ArrayLike | None) -> _Domain:
        """Validate x0 and finite intervals and separate fixed from free variables."""
        point_message = (
            "x0 must be a nonempty one-dimensional vector of finite real values."
        )
        point = _finite_real_array(x0, point_message)
        if point.ndim != 1 or point.size == 0:
            raise ValueError(point_message)

        bounds_message = (
            "Every variable requires explicitly defined, finite real bounds."
        )
        if bounds is None:
            raise ValueError(bounds_message)
        if isinstance(bounds, Bounds):
            lower = np.broadcast_to(
                _finite_real_array(bounds.lb, bounds_message), point.shape
            )
            upper = np.broadcast_to(
                _finite_real_array(bounds.ub, bounds_message), point.shape
            )
        else:
            intervals = _finite_real_array(bounds, bounds_message)
            if intervals.shape != (point.size, 2):
                raise ValueError(
                    "Provide exactly one pair of bounds for each variable in x0."
                )
            lower, upper = intervals.T

        if np.any(lower > upper):
            raise ValueError(
                "An upper bound is less than the corresponding lower bound."
            )
        if np.any((point < lower) | (point > upper)):
            raise ValueError("x0 must lie within the bounds for every variable.")

        free_indices: list[int] = []
        free_bounds: list[Boundary] = []
        fixed_values: list[tuple[int, Real | Decimal]] = []
        vector_dtype: type[np.float64 | np.object_] = np.float64
        for index, (lb, ub) in enumerate(zip(lower, upper, strict=True)):
            if lb == ub:
                if float(lb) != lb:
                    vector_dtype = np.object_
                fixed_values.append((index, lb))
            else:
                lower_float, upper_float = float(lb), float(ub)
                if lower_float == upper_float:
                    raise ValueError("A free interval collapses at float64 precision.")
                free_indices.append(index)
                free_bounds.append((lower_float, upper_float))
        layout = _VariableLayout(tuple(free_indices), tuple(fixed_values), vector_dtype)
        return _Domain(free_bounds, layout)

    @staticmethod
    def _validate_stopping_criteria(config: _MinimizerConfig) -> float:
        """Validate both thresholds and return the target used for comparisons."""
        tol = config.tol
        if tol is not None and (
            not isinstance(tol, (Real, Decimal)) or not 0 <= tol < np.inf
        ):
            raise ValueError("tol must be a finite, nonnegative real number.")

        target = config.target_fitness
        if target is None:
            return -np.inf
        if not isinstance(target, (Real, Decimal)) or not -np.inf <= target < np.inf:
            raise ValueError("target_fitness must be finite or negative infinity.")
        return target

    @staticmethod
    def _max_batches(config: _MinimizerConfig) -> int:
        """Resolve the whole-batch budget from the original effective options."""
        if config.maxiter is not None:
            return config.maxiter // config.n_iterations
        if config.tol is None and config.target_fitness is None:
            return 1
        message = (
            "A stopping criterion was supplied without maxiter. "
            f"Using maxiter={10 * config.n_iterations} (10 batches)."
        )
        warnings.warn(message, OptimizeWarning, stacklevel=3)
        return 10

    def __call__(
        self,
        fun: Callable[..., float],
        x0: ArrayLike,
        args: tuple[object, ...] = (),
        *,
        jac: object = None,
        hess: object = None,
        hessp: object = None,
        bounds: Bounds | ArrayLike | None = None,
        constraints: object = (),
        callback: Callable[..., object] | None = None,
        **options: object,
    ) -> OptimizeResult:
        """
        Solve the bounded problem supplied by `scipy.optimize.minimize`.

        Parameters
        ----------
        fun:
            Scalar cost function called with the full variable vector and `args`.
        x0:
            Nonempty, finite, one-dimensional real vector within the bounds.
            Its length determines the original dimensionality.
        args:
            Extra positional arguments for `fun`. Defaults to an empty tuple.
        jac, hess, hessp:
            Forwarded derivative information. A truthy `jac` or a supplied
            Hessian or Hessian-vector product emits a ``RuntimeWarning``.
        bounds:
            Required finite intervals, given as one pair per variable or a
            broadcastable `Bounds` object. Equal endpoints fix a variable.
        constraints:
            Ignored general constraints. A nonempty collection or a constraint
            without a length emits a ``RuntimeWarning``.
        callback:
            Optional batch callback accepting the full position or a sole
            keyword-compatible parameter named `intermediate_result`.
            Raise ``StopIteration`` to stop.
        **options:
            Individual solver options expanded by SciPy. Recognized arguments
            override the constructor settings for this call.

        Returns
        -------
        OptimizeResult
            Full best position `x`, fitness `fun`, termination fields `success`,
            `status` and `message`, and the run's `history`. Without a published
            solution, `x` is ``None`` and `fun` is positive infinity. The
            optional `nit` counts optimization iterations after initialization.
            Only fully fixed domains return `nfev`, with one evaluation and
            zero iterations. See the class documentation for termination codes
            and the availability of iteration counts.
        """
        self._result = None
        self._execution = None
        if hasattr(self, "_history"):
            del self._history

        config = self._resolve_options(**options)
        self._warn_unsupported_arguments(constraints, jac, hess, hessp)
        domain = self._prepare_domain(x0, bounds)
        target = self._validate_stopping_criteria(config)
        if not domain.free_bounds:
            return self._fixed_result(fun, args, domain, config)

        objective = _ScipyObjective(fun, args, domain.layout)
        state = self._run(objective, domain, config, target, callback)
        return self._population_result(state, config)

    def _run(
        self,
        objective: _ScipyObjective,
        domain: _Domain,
        config: _MinimizerConfig,
        target: float,
        callback: Callable[..., object] | None,
    ) -> _ExecutionState:
        """Initialize a fresh population context and continue it across batches."""
        parameters = signature(callback).parameters if callback is not None else ()
        intermediate = set(parameters) == {"intermediate_result"}
        space = Continuous(
            domain.free_bounds,
            objective,
            use_cache=config.use_cache,
            cache_type=config.cache_type,
            cache_size=config.cache_size,
        )
        self._configure_components(
            space=space,
            algorithm=deepcopy(config.algorithm),
            n_iterations=config.n_iterations,
            n_particles=config.n_particles,
            backend=config.backend,
            processor=deepcopy(config.processor),
            migration=config.migration,
            seed=config.seed,
            fitness_failure_strategy=config.fitness_failure_strategy,
            history=config.history,
        )
        self._initialize_context()
        self._configure_history()
        state = _ExecutionState(domain.layout, self._max_batches(config))
        self._execution = state
        self._execute_batches(state, config.tol, target, callback, intermediate)
        return state

    def _fixed_result(
        self,
        fun: Callable[..., float],
        args: tuple[object, ...],
        domain: _Domain,
        config: _MinimizerConfig,
    ) -> OptimizeResult:
        """Prepare empty history, then evaluate the sole feasible point on a copy."""
        if config.fitness_failure_strategy not in ("raise", "invalidate"):
            raise ValueError(
                "fitness_failure_strategy must be 'raise' or 'invalidate'."
            )
        self._history_config = self._configure_particle_history(config.history)
        self._space = Continuous(domain.free_bounds)
        self._space.initialize_context(config.seed)
        self._configure_history()

        x = domain.layout.reconstruct(())
        evaluation_point = x.copy()
        try:
            scalar = np.asarray(fun(evaluation_point, *args)).item()
            if not isinstance(scalar, (Real, Decimal)):
                raise TypeError("Objective fitness must be a real scalar.")
            fitness = float(scalar)
            if np.isnan(fitness):
                raise ValueError("Objective fitness cannot be NaN.")
        except Exception:
            if config.fitness_failure_strategy == "raise":
                raise
            fitness = np.inf

        result = self._build_result(x, fitness, "fixed", config)
        result["nit"] = 0
        result["nfev"] = 1
        return result

    def _population_result(
        self, state: _ExecutionState, config: _MinimizerConfig
    ) -> OptimizeResult:
        """Decode the published best and attach iteration metadata for known executors."""
        if self._result is None or state.stop_reason is None:
            raise RuntimeError("Population execution has not completed.")
        best = self._result.result
        x = None
        fitness = np.inf
        if best is not None:
            x = self._decode_solution(best["variables"], state.layout)
            fitness = self._published_fitness(best["fitness"])
        result = self._build_result(x, fitness, state.stop_reason, config)
        if not state.force_stopped and (
            isinstance(self._backend, ParallelBackendBase)
            or (isinstance(self._backend, Local) and self._processor.processors_pool)
        ):
            result["nit"] = state.batches_completed * config.n_iterations
        return result

    def _build_result(
        self,
        x: NDArray[np.float64 | np.object_] | None,
        fitness: float,
        reason: _StopReason | Literal["fixed"],
        config: _MinimizerConfig,
    ) -> OptimizeResult:
        """Translate the recorded outcome and attach this call's published history."""
        if fitness == np.inf or np.isnan(fitness):
            status, message = 3, "No valid solution was found."
        else:
            match reason:
                case "callback":
                    status, message = 99, "Optimization terminated by the callback."
                case "saturation":
                    status, message = 0, "Negative infinity fitness reached."
                case "target_fitness":
                    status, message = 0, "Target fitness reached."
                case "force_stop":
                    status = 2
                    message = "Optimization terminated by the algorithm's force stop."
                case "tol":
                    status, message = 0, "Fitness tolerance satisfied."
                case "fixed":
                    status, message = 0, "All variables are fixed."
                case "budget" if (
                    config.tol is not None or config.target_fitness is not None
                ):
                    status = 1
                    message = "Iteration budget exhausted before meeting the criteria."
                case "budget":
                    status = 0
                    message = "Optimization completed within the iteration budget."

        return OptimizeResult(
            x=x,
            fun=fitness,
            success=status == 0,
            status=status,
            message=message,
            history=self.history,
        )

    @staticmethod
    def _published_fitness(value: object) -> float:
        """Read a scalar fitness according to the backend's best-result contract."""
        if not isinstance(value, float):
            raise TypeError("Optimization result fitness must be a float.")
        return float(value)

    def _execute_batches(
        self,
        state: _ExecutionState,
        tol: float | None,
        target: float,
        callback: Callable[..., object] | None,
        intermediate: bool,
    ) -> None:
        """Continue one initialized context, publishing even on exceptional exits."""
        previous_best: float | None = None
        for _ in range(state.max_batches):
            try:
                self._backend.execute()
            finally:
                self._result = self._backend.result
                self._update_history(self._result.history)

            state.batches_completed += 1
            state.force_stopped = self._force_stopped()
            best = self._result.result
            current_best: float | None = None
            if best is not None:
                current_best = self._published_fitness(best["fitness"])
                if callback is not None:
                    x = self._decode_solution(best["variables"], state.layout)
                    if self._callback_stopped(callback, intermediate, x, current_best):
                        state.stop_reason = "callback"
                        return

            if current_best == -np.inf:
                state.stop_reason = "saturation"
            elif current_best is not None and current_best <= target:
                state.stop_reason = "target_fitness"
            elif state.force_stopped:
                state.stop_reason = "force_stop"
            elif (
                tol is not None
                and current_best is not None
                and previous_best is not None
                and isfinite(current_best)
                and isfinite(previous_best)
                and abs(current_best - previous_best) <= tol
            ):
                state.stop_reason = "tol"
            if state.stop_reason is not None:
                return
            previous_best = current_best
        state.stop_reason = "budget"

    def _force_stopped(self) -> bool:
        """Read the native stop flag where a backend exposes its global executor."""
        if isinstance(self._backend, Local):
            executor = next(iter(self._processor.processors_pool.values()), None)
            return executor is not None and executor.stopped_after_iteration
        if isinstance(self._backend, ParallelBackendBase):
            return self._algorithm.force_stop
        return False

    def _decode_solution(
        self, variables: object, layout: _VariableLayout
    ) -> NDArray[np.float64 | np.object_]:
        """Decode a published particle and restore its full vector in fresh storage."""
        if not isinstance(variables, dict):
            raise TypeError("Optimization result variables must be a dictionary.")
        free_values = self._space.decode(variables)
        if not isinstance(free_values, list):
            raise TypeError("Positional Continuous decoding must return a list.")
        return layout.reconstruct(free_values)

    @staticmethod
    def _callback_stopped(
        callback: Callable[..., object],
        intermediate: bool,
        x: NDArray[np.float64 | np.object_],
        fitness: float,
    ) -> bool:
        """Invoke the selected callback form and catch only its stop request."""
        result = OptimizeResult(x=x, fun=fitness) if intermediate else None
        try:
            if result is None:
                callback(x)
            else:
                callback(intermediate_result=result)
        except StopIteration:
            return True
        return False
