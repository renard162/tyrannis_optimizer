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
    """Keep original numeric precision for interval and feasibility comparisons."""
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
    """Minimize bounded objectives as a callable method of scipy.optimize.minimize.

    Construction stores the component templates and settings for subsequent
    sequential calls. Pass the instance as ``method`` to run Tyrannis and
    receive a SciPy OptimizeResult.

    Parameters
    ----------
    algorithm:
        Tyrannis algorithm for minimizing the scalar objective.
    n_iterations:
        Positive integer iteration count per batch.
    n_particles:
        Positive integer population size, per island for distributed backends.
    backend:
        Execution backend. The default, ``None``, selects local execution.
    processor:
        Processor template. The default, ``None``, selects serial processing.
    migration:
        Island migration driver. The default, ``None``, selects isolation.
        Local execution always uses isolation.
    seed:
        Random seed for the components. Defaults to ``None``.
    fitness_failure_strategy:
        Evaluation failure policy. The default, ``"raise"``, propagates errors.
        With ``"invalidate"``, Tyrannis assigns infinite fitness on failure.
    history:
        History event name, list of event names, or ``"all"``. The default,
        ``None``, disables recording.
    use_cache:
        Enable the Continuous evaluation cache. Defaults to ``False``.
    cache_type:
        Continuous cache strategy. Defaults to ``"lru"``.
    cache_size:
        Maximum number of in-memory cache entries. Defaults to 100,000.
        The disk cache ignores this setting.
    maxiter:
        Optional total iteration limit. When supplied, it must be an integer
        at least as large as ``n_iterations``. Only whole batches fit within
        this limit. For population runs the default is one batch, or ten batches
        with an ``OptimizeWarning`` if either stopping criterion is enabled.
    tol:
        Finite, nonnegative absolute tolerance between the best fitness values
        of two consecutive completed batches. Zero requires exact stagnation.
        The default, ``None``, disables the comparison.
    target_fitness:
        Stop when the published best fitness is at or below this value.
        Finite values and negative infinity are accepted. The default,
        ``None``, leaves only saturation at negative infinity active.

    Notes
    -----
    Each resolution starts from the constructor settings, independently of any
    execution state. SciPy passes options as individual keywords. Each supplied
    option overrides its constructor setting, including an explicit ``None`` for
    optional parameters. Omitted options retain the constructor value or its
    default. Unknown names are ignored with one ``OptimizeWarning``. Each
    configuration owns its history selection and retains component references.

    Each population run creates a positional Continuous space with its own
    objective and cache configuration. It deep-copies the selected algorithm
    and any supplied processor. The backend manages its execution replicas.
    Supplied backends and migration drivers retain their external resources
    and receive fresh contexts according to their execution model. Sharing
    mutable drivers or backends is supported only across sequential runs.

    Within a run, each backend call continues the same population, adaptive
    state, random streams and runtime caches according to the backend's own
    execution model. The iteration count per batch stays constant. A new
    history object is created for every run, including when recording is
    disabled. After each backend call, even if it raises, the adapter publishes
    the backend's cumulative records on that run's history object.
    The result exposes that same object. Its variable columns represent only
    free dimensions. Result and callback vectors include all fixed dimensions.

    The domain requires explicit finite bounds for every variable. The finite
    one-dimensional ``x0`` supplies the dimension for validation and broadcasting
    and must lie within the inclusive bounds. It does not seed the population.
    Equal endpoints fix a variable. Only free intervals belong to the continuous
    space, and an empty list of free intervals represents a fully fixed problem.
    The ``keep_feasible`` flag of SciPy Bounds is silently ignored. Free intervals
    must retain distinct endpoints at the precision of the continuous space.
    Reconstruction normally yields a float64 array. When a fixed value requires
    higher precision, an object array preserves the original real scalar.
    When every variable is fixed, the objective is evaluated once at the only
    feasible position. No batch runs, the callback is never invoked, and the
    history has the requested event configuration and zero records.

    Nonempty general constraints and supplied derivative information are ignored
    with separate runtime warnings. Objective adaptation calls the scalar
    function supplied by SciPy with a fresh full vector and its extra arguments.
    The function and arguments must support the serialization mechanism of the
    selected processor and backend when evaluation crosses process boundaries.

    Distributed backends require ``maxiter`` to be absent or equal to
    ``n_iterations``, with both ``tol`` and ``target_fitness`` set to ``None``.
    They execute one batch and force stop applies separately to each island.
    Local execution observes the executing processor's stop request. Parallel
    execution observes the algorithm directly. Other backend models have no
    common interface for reporting force stop.

    Native SciPy methods typically invoke callbacks after each iteration. Here
    the orchestration process invokes the callback once per completed batch
    with a published solution, including the last batch. A sole parameter named
    ``intermediate_result`` receives an OptimizeResult by keyword with ``x``
    and ``fun``. Other signatures receive the full vector positionally. Both
    forms receive independent solution storage. Return values are ignored.
    StopIteration from the callback ends the run with its published result.

    At each batch boundary, stopping reasons in priority order are callback
    interruption, saturation at negative infinity, reaching the fitness target,
    native force stop, absolute fitness tolerance, and batch budget exhaustion.
    Only finite fitness values participate in the tolerance comparison.
    """

    # Keep the last parameter and its corresponding argument free of trailing commas.
    # fmt: off
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
        target_fitness: float | None = None
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
            target_fitness=target_fitness
        )
        self._execution: _ExecutionState | None = None
        self._result = None
    # fmt: on

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
        **options: object
    ) -> OptimizeResult:  # fmt: skip
        """Solve one bounded problem with independent execution state and history.

        SciPy supplies the scalar objective ``fun``, initial vector ``x0``, extra
        positional objective arguments ``args``, bounds, and an optional callback.
        It forwards solver options as individual keyword arguments. Supplied
        derivatives (``jac``, ``hess``, ``hessp``) and nonempty ``constraints``
        are ignored with warnings.

        Returns
        -------
        OptimizeResult
            ``x`` is the full best position and ``fun`` its scalar fitness. If
            no solution was published, they are None and positive infinity.
            Finite values and negative infinity are valid fitness values.
            ``history`` is this call's OptimizationHistory, also available when
            recording is disabled. ``message`` describes the stopping condition.
            ``success`` is true only for status 0, which requires valid fitness
            and a completed fixed horizon, a reached target, satisfied tolerance,
            negative infinity, or a fully fixed domain. Status 1 indicates budget
            exhaustion with unmet criteria, 2 a force stop, 3 a missing or
            invalid solution, and 99 a callback interruption.

            ``nit`` counts optimization iterations after initialization. It is
            omitted when a batch may have been shortened or the backend has no
            global iteration count, including distributed execution. Population
            runs omit ``nfev`` because their evaluation count is unavailable.
            Fully fixed problems return ``nit=0`` and ``nfev=1``.
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
        callback: Callable[..., object] | None
    ) -> _ExecutionState:  # fmt: skip
        """Initialize and execute batches for a prepared domain with free variables."""
        parameters = signature(callback).parameters if callback is not None else ()
        intermediate = set(parameters) == {"intermediate_result"}
        space = Continuous(
            domain.free_bounds,
            objective,
            use_cache=config.use_cache,
            cache_type=config.cache_type,
            cache_size=config.cache_size
        )  # fmt: skip
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
            history=config.history
        )  # fmt: skip
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
        config: _MinimizerConfig
    ) -> OptimizeResult:  # fmt: skip
        """Evaluate the only feasible point and prepare its empty native history."""
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
        """Decode the official best and report iterations only for complete batches."""
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
        config: _MinimizerConfig
    ) -> OptimizeResult:  # fmt: skip
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
            history=self.history
        )  # fmt: skip

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
        intermediate: bool
    ) -> None:  # fmt: skip
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
        fitness: float
    ) -> bool:  # fmt: skip
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
