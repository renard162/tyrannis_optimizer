"""Direct custom-method contracts, before SciPy can normalize the caller's inputs."""

import warnings
from collections.abc import Callable
from decimal import Decimal
from multiprocessing.managers import SyncManager
from typing import Never, TypedDict, cast
from unittest.mock import Mock

import numpy as np
import pytest
from _support.distributed_backend import TestDistributedBackend
from _support.integration_probe import IntegrationProbeAlgorithm
from _support.numerics import BASE_SEED
from _support.objectives import constant_objective
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import Bounds, OptimizeWarning

from tyrannis import Optimizer
from tyrannis.backend.local import Local
from tyrannis.compatibility.scipy import ScipyMinimizer
from tyrannis.core.orchestrator import OrchestratorBase
from tyrannis.core.results import HistoryConfig, OptimizationHistory
from tyrannis.migration.island_isolation import IslandIsolation
from tyrannis.processor.serial import Serial

# SciPy's unannotated Bounds constructor infers scalar defaults even though
# its documented public interface accepts broadcastable arrays.
array_bounds = cast(Callable[[ArrayLike, ArrayLike], Bounds], Bounds)


class SolverOptions(TypedDict, total=False):
    n_iterations: int
    n_particles: int
    maxiter: float
    tol: float
    target_fitness: float
    use_cache: bool
    cache_type: str
    cache_size: int


@pytest.fixture
def method() -> ScipyMinimizer:
    return ScipyMinimizer(IntegrationProbeAlgorithm(), 1, 2, seed=BASE_SEED)


def forbidden(*args: object, **kwargs: object) -> Never:
    pytest.fail("The adapter invoked an unused component")


def test_public_orchestrator(method: ScipyMinimizer) -> None:
    assert isinstance(method, OrchestratorBase)
    assert not isinstance(method, Optimizer)
    assert not hasattr(method, "history")


def test_options_are_resolved_for_one_call_only() -> None:
    events = ["iteration"]
    method = ScipyMinimizer(
        IntegrationProbeAlgorithm(), 2, 3, history=events, seed=BASE_SEED, maxiter=2
    )
    events.clear()
    with pytest.warns(OptimizeWarning) as caught:
        result = method(
            constant_objective,
            [0],
            bounds=[(0, 1)],
            history=None,
            maxiter=None,
            n_iterations=1,
            extra_a=True,
            extra_b=4,
        )
    assert len(caught) == 1
    assert all(key in str(caught[0].message) for key in ("extra_a", "extra_b"))
    assert result.nit == 1
    assert not method.history.config.history_enabled
    result = method(constant_objective, [0], bounds=[(0, 1)])
    assert result.nit == 2
    assert method.history.config == HistoryConfig(iteration=True)
    assert len(method.history.records) == 3 * (1 + 2)


@pytest.mark.parametrize(
    "options",
    [
        {"n_iterations": 2, "maxiter": 1},
        {"maxiter": 1.5},
        {"maxiter": True},
        {"tol": -1},
        {"tol": np.inf},
        {"tol": np.nan},
        {"target_fitness": np.inf},
        {"target_fitness": np.nan},
        {"n_particles": 0},
        {"use_cache": True, "cache_size": 0},
        {"use_cache": True, "cache_type": "unsupported"},
    ],
)
def test_invalid_effective_options(
    method: ScipyMinimizer, options: SolverOptions
) -> None:
    with pytest.raises(Exception):  # noqa: B017
        method(forbidden, [0], bounds=[(0, 1)], **options)


@pytest.mark.parametrize("constraint", [None, ()])
def test_absent_or_empty_constraints_are_silent(
    method: ScipyMinimizer, constraint: object
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        method(
            constant_objective, [0], bounds=[(0, 0)], constraints=constraint, jac=None
        )


@pytest.mark.parametrize(
    "constraint", [forbidden, [{"type": "ineq", "fun": forbidden}]]
)
def test_unsupported_arguments_are_individually_reported_and_unused(
    method: ScipyMinimizer, constraint: object
) -> None:
    with pytest.warns(RuntimeWarning) as caught:
        method(
            constant_objective,
            [0],
            bounds=[(-1, 1)],
            constraints=constraint,
            jac=forbidden,
            hess=forbidden,
            hessp=forbidden,
        )
    messages = [str(item.message) for item in caught]
    assert len(messages) == 4
    for argument in ("constraints", "(jac)", "(hess)", "(hessp)"):
        assert sum(argument in message for message in messages) == 1


@pytest.mark.parametrize(
    ("x0", "bounds"),
    [
        ([0], None),
        ([0], [(None, 1)]),
        ([0], [(-np.inf, 1)]),
        ([0], [(0, np.nan)]),
        ([0], [(1, -1)]),
        ([0, 0], [(0, 1)]),
        ([0], [(0, 1, 2)]),
        ([], []),
        ([[0]], [(0, 1)]),
        ([1j], [(0, 1)]),
        ([np.nan], [(0, 1)]),
        ([-1], [(0, 1)]),
        ([0, 0], array_bounds([0, 0, 0], 1)),
    ],
    ids=[
        "missing",
        "undefined",
        "infinite",
        "nan-bound",
        "reversed",
        "count",
        "arity",
        "empty",
        "matrix",
        "complex",
        "nan-point",
        "outside",
        "broadcast",
    ],
)
def test_invalid_domain(
    method: ScipyMinimizer, x0: ArrayLike, bounds: Bounds | ArrayLike | None
) -> None:
    with pytest.raises(Exception):  # noqa: B017
        method(forbidden, x0, bounds=bounds)


def test_bounds_broadcast_and_inclusive_endpoints(method: ScipyMinimizer) -> None:
    bounds = Bounds(0, 2, keep_feasible=True)
    point = np.array([0.0, 2.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = method(constant_objective, point, bounds=bounds)
    np.testing.assert_array_equal(
        result.x, [1, 1]
    )  # Probe initializes at the midpoint.
    np.testing.assert_array_equal(point, [0, 2])
    np.testing.assert_array_equal(bounds.lb, [0])
    np.testing.assert_array_equal(bounds.ub, [2])
    assert bounds.keep_feasible.all()


def test_reconstruction_preserves_interleaved_coordinates_and_exact_fixed_values(
    method: ScipyMinimizer,
) -> None:
    fixed = Decimal("0.1")
    bounds = [(fixed, fixed), (-4, -2), (7, 7), (2, 4)]
    original = list(bounds)
    received: list[NDArray[np.object_]] = []

    def objective(x: NDArray[np.object_], label: str, bias: int) -> float:
        assert x.dtype == object and x.shape == (4,)
        assert x[0] == fixed and x[2] == 7
        assert -4 <= x[1] <= -2 and 2 <= x[3] <= 4
        assert label == "cost"
        fitness = float(x[1] + bias * x[3])
        received.append(x)
        x[:] = 100
        return fitness

    result = method(
        objective,
        np.array([fixed, -4, 7, 4], dtype=object),
        args=("cost", 2),
        bounds=bounds,
        history="iteration",
    )
    assert len(received) == 4 and len({id(x) for x in received}) == 4
    assert bounds == original
    assert result.x.tolist()[::2] == [fixed, 7]
    assert result.fun == result.x[1] + 2 * result.x[3]
    assert result.x.dtype == object
    assert [c for c in method.history.columns if c.isdecimal()] == ["0", "1"]


def test_originally_free_interval_cannot_collapse(method: ScipyMinimizer) -> None:
    lower = 2**53
    with pytest.raises(Exception):  # noqa: B017
        method(forbidden, [lower], bounds=[(lower, lower + 1)])


def test_float64_conversion_keeps_evaluations_inside_original_interval(
    method: ScipyMinimizer,
) -> None:
    # The interval contains two distinct float64 values even after inward rounding.
    lower, upper = 2**53 + 1, 2**53 + 4

    def feasible(x: NDArray[np.float64]) -> float:
        # Python float/int comparison retains the exact integer endpoint.
        assert lower <= float(x[0]) <= upper
        return 0.0

    method(feasible, [upper], bounds=[(lower, upper)])


@pytest.mark.parametrize(
    "options", [{"maxiter": 2}, {"tol": 0}, {"target_fitness": -np.inf}]
)
def test_distributed_limits_precede_initialization(
    method: ScipyMinimizer, monkeypatch: pytest.MonkeyPatch, options: SolverOptions
) -> None:
    backend = TestDistributedBackend(SyncManager())  # Deliberately unstarted.
    monkeypatch.setattr(IntegrationProbeAlgorithm, "initialize_context", forbidden)
    monkeypatch.setattr(backend, "initialize_context", forbidden)
    with pytest.raises(Exception):  # noqa: B017
        method(forbidden, [0], bounds=[(0, 1)], backend=backend, **options)


@pytest.mark.parametrize("maxiter", [None, 1])
def test_compatible_distributed_overrides(maxiter: int | None) -> None:
    method = ScipyMinimizer(
        IntegrationProbeAlgorithm(),
        1,
        1,
        backend=TestDistributedBackend(SyncManager()),
        maxiter=3,
        tol=1,
        target_fitness=1,
    )
    result = method(
        constant_objective,
        [0],
        bounds=[(0, 0)],
        maxiter=maxiter,
        tol=None,
        target_fitness=None,
    )
    assert result.success


@pytest.mark.parametrize(("fitness", "status"), [(1.5, 0), (-np.inf, 0), (np.inf, 3)])
def test_fully_fixed_domain_evaluates_once_without_execution(
    method: ScipyMinimizer, monkeypatch: pytest.MonkeyPatch, fitness: float, status: int
) -> None:
    for component in (
        IntegrationProbeAlgorithm,
        Serial,
        Local,
        TestDistributedBackend,
        IslandIsolation,
    ):
        monkeypatch.setattr(component, "initialize_context", forbidden)
    monkeypatch.setattr(TestDistributedBackend, "execute", forbidden)

    def objective(x: NDArray[np.float64], offset: float) -> np.float64:
        np.testing.assert_array_equal(x, [1.5, 3])
        assert offset == 4
        x[:] = -1
        return np.float64(fitness)

    evaluate = Mock(wraps=objective)
    # A distributed backend permits no extra criteria. Use the local default
    # for the stopping-criterion case, and the supplied components otherwise.
    backend = None if status == 0 else TestDistributedBackend(SyncManager())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = method(
            evaluate,
            [1.5, 3],
            args=(4,),
            bounds=[(1.5, 1.5), (3, 3)],
            target_fitness=-np.inf if backend is None else None,
            history="best",
            callback=forbidden,
            backend=backend,
            processor=Serial(),
            migration=IslandIsolation(),
        )
    np.testing.assert_array_equal(result.x, [1.5, 3])
    assert result.fun == fitness
    assert result.status == status and result.success == (status == 0)
    assert result.message.strip()
    assert (evaluate.call_count, result.nfev, result.nit) == (1, 1, 0)
    assert (
        isinstance(result.history, OptimizationHistory)
        and result.history is method.history
    )
    assert method.history.config == HistoryConfig(best=True)
    assert list(method.history.records) == []


@pytest.mark.parametrize(
    "outcome",
    [RuntimeError("evaluation"), [1.0, 2.0], np.nan],
    ids=["exception", "vector", "nan"],
)
def test_fixed_evaluation_failure_policy(
    method: ScipyMinimizer, outcome: object
) -> None:
    objective = (
        Mock(side_effect=outcome)
        if isinstance(outcome, Exception)
        else Mock(return_value=outcome)
    )
    with pytest.raises(Exception):  # noqa: B017
        method(objective, [3], bounds=[(3, 3)])
    assert objective.call_count == 1
    result = method(
        objective, [3], bounds=[(3, 3)], fitness_failure_strategy="invalidate"
    )
    assert objective.call_count == 2
    assert (result.fun, result.status, result.success) == (np.inf, 3, False)
    np.testing.assert_array_equal(result.x, [3])
    assert (result.nit, result.nfev) == (0, 1)


def test_empty_backend_publication_has_no_position_or_callback(
    method: ScipyMinimizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = Local()
    # A normally returning backend may leave its public ProcessorResult empty.
    monkeypatch.setattr(backend, "execute", lambda: None)
    result = method(
        forbidden, [0], bounds=[(-1, 1)], backend=backend, callback=forbidden
    )
    assert result.x is None and result.fun == np.inf
    assert (result.status, result.success) == (3, False)
    assert "valid" in result.message.lower() and "nfev" not in result
    assert result.history is method.history and not method.history.records
