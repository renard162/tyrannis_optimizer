from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence, Sized
from dataclasses import dataclass, fields, replace
from decimal import Decimal
from math import isfinite
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import Bounds, OptimizeWarning

from ..core.algorithm import AlgorithmBase
from ..core.backend import BackendBase
from ..core.backend_distributed import DistributedBackendBase
from ..core.backend_migration import MigrationDriverBase
from ..core.orchestrator import OrchestratorBase
from ..core.processor import ProcessorBase
from ..space.continuous import Boundary


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
    """Configure Tyrannis components and prepare bounded SciPy minimization inputs.

    Construction stores settings only. Run contexts, history, and an execution
    method are not provided during input preparation.

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
        at least as large as ``n_iterations``. Defaults to ``None``.
    tol:
        Optional stopping tolerance, preserved as supplied. Defaults to ``None``.
    target_fitness:
        Optional target objective value, preserved as supplied. Defaults to ``None``.

    Notes
    -----
    Each resolution starts from the constructor settings, independently of any
    execution state. SciPy passes options as individual keywords. Each supplied
    option overrides its constructor setting, including an explicit ``None`` for
    optional parameters. Omitted options retain the constructor value or its
    default. Unknown names are ignored with one ``OptimizeWarning``. Each
    configuration owns its history selection and retains component references.

    The domain requires explicit finite bounds for every variable. The finite
    one-dimensional ``x0`` supplies the dimension for validation and broadcasting
    and must lie within the inclusive bounds. It does not seed the population.
    Equal endpoints fix a variable. Only free intervals belong to the continuous
    space, and an empty list of free intervals represents a fully fixed problem.
    The ``keep_feasible`` flag of SciPy Bounds is silently ignored. Free intervals
    must retain distinct endpoints at the precision of the continuous space.
    Reconstruction normally yields a float64 array. When a fixed value requires
    higher precision, an object array preserves the original real scalar.

    Nonempty general constraints and supplied derivative information are ignored
    with separate runtime warnings. Objective adaptation calls the scalar
    function supplied by SciPy with a fresh full vector and its extra arguments.
    The function and arguments must support the serialization mechanism of the
    selected processor and backend when evaluation crosses process boundaries.

    Distributed backends require ``maxiter`` to be absent or equal to
    ``n_iterations``, with both ``tol`` and ``target_fitness`` set to ``None``.
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
