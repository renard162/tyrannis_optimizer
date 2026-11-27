"""SciPy custom-method dispatch and Tyrannis execution through the public API."""

import json
import warnings
from multiprocessing import get_context
from os import getpid
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from _support.distributed_backend import TestDistributedBackend
from _support.integration_probe import IntegrationProbeAlgorithm
from _support.numerics import BASE_SEED, seed_for
from _support.objectives import (
    constant_objective,
    shifted_sphere,
    sphere,
    worker_shifted_sphere,
)
from numpy.typing import NDArray
from scipy.optimize import Bounds, OptimizeResult, OptimizeWarning, minimize

from tyrannis.algorithm import PSO
from tyrannis.backend.local import Local
from tyrannis.compatibility.scipy import ScipyMinimizer
from tyrannis.core.results import OptimizationHistory
from tyrannis.migration.island_isolation import IslandIsolation
from tyrannis.processor.joblib import Joblib
from tyrannis.processor.serial import Serial


@pytest.fixture
def method() -> ScipyMinimizer:
    return ScipyMinimizer(IntegrationProbeAlgorithm(), 2, 2, seed=BASE_SEED)


def test_pso_default_stack_returns_a_consistent_scipy_result() -> None:
    method = ScipyMinimizer(PSO(), 2, 3, seed=BASE_SEED)
    result = minimize(
        shifted_sphere, [0, 2], args=(3,), method=method, bounds=[(-1, 1), (2, 2)]
    )
    assert isinstance(result, OptimizeResult)
    assert (
        isinstance(result.history, OptimizationHistory)
        and result.history is method.history
    )
    assert result.x.shape == (2,) and result.x.dtype == np.float64
    assert -1 <= result.x[0] <= 1 and result.x[1] == 2
    assert result.fun == shifted_sphere(result.x, 3)
    assert (result.status, result.success, result.nit) == (0, True, 2)
    assert "budget" in result.message.lower()
    assert {
        "nfev",
        "jac",
        "hess",
        "hess_inv",
        "njev",
        "nhev",
        "maxcv",
        "constr",
    }.isdisjoint(result)
    assert not method.history.config.history_enabled
    assert method.history.columns and method.history.internal_columns_
    assert not method.history.records
    assert (
        list(method.history.generator) == list(method.history.internal_generator_) == []
    )


def test_scipy_dispatches_the_scalar_part_of_a_joint_objective(
    method: ScipyMinimizer,
) -> None:
    calls: list[float] = []

    def value_and_gradient(
        x: NDArray[np.float64], bias: float
    ) -> tuple[float, NDArray[np.float64]]:
        calls.append(float(x[0]))
        return shifted_sphere(x, bias), 2 * x

    with pytest.warns(RuntimeWarning, match="jac"):
        result = minimize(
            value_and_gradient,
            [0.5],
            args=(4,),
            bounds=Bounds(0, 1),
            jac=True,
            method=method,
        )
    assert result.fun == shifted_sphere(result.x, 4)
    assert result.success and float(result.x[0]) in calls


@pytest.mark.parametrize("jac", ["2-point", "3-point", "cs"])
def test_scipy_normalizes_finite_difference_selection(
    method: ScipyMinimizer, jac: str
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = minimize(
            constant_objective, [0], bounds=[(0, 1)], method=method, jac=jac
        )
    assert result.success and result.fun == 1


def test_direct_tol_and_explicit_none_in_options(method: ScipyMinimizer) -> None:
    for options, iterations in [({"maxiter": 6}, 4), ({"maxiter": 6, "tol": None}, 6)]:
        result = minimize(
            constant_objective,
            [0],
            bounds=[(0, 1)],
            method=method,
            tol=0,
            options=options,
        )
        assert (result.nit, result.status, result.success) == (iterations, 0, True)


def test_batch_continuation_preserves_particles_history_rng_and_cache(
    method: ScipyMinimizer,
) -> None:
    histories: list[OptimizationHistory] = []
    prefixes: list[list[str]] = []
    calls: list[float] = []

    def objective(x: NDArray[np.float64]) -> float:
        calls.append(float(x[0]))
        return float(x[0])

    def observe(intermediate_result: OptimizeResult) -> None:
        history = method.history
        histories.append(history)
        prefixes.append(list(history.records))
        assert intermediate_result.fun == min(calls)
        assert intermediate_result.x.tolist() == [min(calls)]

    # Only two representable positions: a fresh per-batch cache would repeat calls.
    bounds = [(1.0, float(np.nextafter(1.0, 2.0)))]
    result = minimize(
        objective,
        [1],
        bounds=bounds,
        method=method,
        callback=observe,
        options={"maxiter": 5, "history": "iteration", "use_cache": True},
    )
    assert result.nit == 4 and result.status == 0
    assert len(prefixes) == 2 and all(h is result.history for h in histories)
    assert prefixes[1][: len(prefixes[0])] == prefixes[0]
    assert [len(rows) for rows in prefixes] == [6, 10]
    records = [json.loads(row) for row in prefixes[-1]]
    positions = {r["particle"]["variables"]["0"] for r in records}
    assert sorted(calls) == sorted(positions)
    ids = {r["particle"]["identifier"] for r in records}
    assert len(ids) == 2
    for identifier in ids:
        snapshots = [r for r in records if r["particle"]["identifier"] == identifier]
        assert [r["iteration"] for r in snapshots] == list(range(5))
        trace = snapshots[-1]["particle"]["trace"]
        assert [e["phase"] for e in trace] == ["initialize"] + ["update"] * 4
        assert [e["iteration"] for e in trace] == list(range(5))
        for step, record in enumerate(snapshots):
            assert record["particle"]["trace"] == trace[: step + 1]
            assert record["particle"]["evaluation_count"] == step + 1

    # The probe's transition ignores its horizon. A single long batch provides
    # an independent dispatch schedule for comparing the advancing RNG streams.
    reference = minimize(
        objective,
        [1],
        bounds=bounds,
        method=method,
        options={"n_iterations": 4, "history": "iteration", "use_cache": True},
    )

    def draws(
        history: OptimizationHistory,
    ) -> list[tuple[int, float, tuple[float, ...]]]:
        samples = []
        for row in history.records:
            record = json.loads(row)
            event = record["particle"]["trace"][-1]
            samples.append(
                (record["iteration"], event["scalar"], tuple(event["draws"]))
            )
        return sorted(samples)

    assert draws(result.history) == draws(reference.history)


@pytest.mark.parametrize(
    ("target", "batches", "status"), [(None, 1, 0), (-np.inf, 10, 1)]
)
def test_automatic_budget_with_a_callback(
    method: ScipyMinimizer, target: float | None, batches: int, status: int
) -> None:
    positions: list[NDArray[np.float64]] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = minimize(
            constant_objective,
            [0],
            bounds=[(-1, 1)],
            method=method,
            callback=positions.append,
            options={"target_fitness": target},
        )
    assert len(caught) == int(target is not None)
    if caught:
        assert caught[0].category is OptimizeWarning and "10" in str(caught[0].message)
    assert len(positions) == batches and result.nit == 2 * batches
    assert result.status == status and result.success == (status == 0)
    assert "budget" in result.message.lower()


@pytest.mark.parametrize(
    ("fitness", "options", "nit", "status", "reason"),
    [
        (1.0, {"target_fitness": 1.0, "tol": 0}, 2, 0, "target"),
        (1.0, {"target_fitness": 0.0, "tol": 0}, 4, 0, "tolerance"),
        (-np.inf, {}, 2, 0, "infinity"),
        (-np.inf, {"target_fitness": -np.inf}, 2, 0, "infinity"),
        (np.inf, {"tol": 0.0}, 6, 3, "valid"),
    ],
)
def test_boundary_stop_reasons(
    method: ScipyMinimizer,
    fitness: float,
    options: dict[str, float],
    nit: int,
    status: int,
    reason: str,
) -> None:
    result = minimize(
        lambda x: fitness,
        [0],
        bounds=[(-1, 1)],
        method=method,
        options={"maxiter": 6, **options},
    )
    assert (result.fun, result.nit, result.status, result.success) == (
        fitness,
        nit,
        status,
        status == 0,
    )
    assert reason in result.message.lower()
    assert result.x.shape == (1,) and -1 <= result.x[0] <= 1


def test_positive_tolerance_is_inclusive() -> None:
    # With one particle, the batch bests are 3 and 2. A strict comparison would miss.
    values = iter([4.0, 3.0, 2.0])
    method = ScipyMinimizer(IntegrationProbeAlgorithm(), 1, 1, seed=BASE_SEED)
    result = minimize(
        lambda x: next(values),
        [0],
        bounds=[(-1, 1)],
        method=method,
        options={"tol": 1.0, "maxiter": 3},
    )
    assert (result.fun, result.nit, result.status, result.success) == (2.0, 2, 0, True)
    assert "tolerance" in result.message.lower()


def test_adaptive_horizon_extends_across_batches() -> None:
    method = ScipyMinimizer(IntegrationProbeAlgorithm(), 1, 1, seed=BASE_SEED)
    minimize(
        constant_objective,
        [0],
        bounds=[(0, 1)],
        method=method,
        options={"maxiter": 2, "history": "iteration"},
    )
    final = json.loads(method.history.records[-1])
    assert [event["max_iterations"] for event in final["particle"]["trace"]] == [
        1,
        1,
        2,
    ]


@pytest.mark.parametrize(
    ("stop_at", "target", "status"), [(1, None, 2), (3, None, 2), (1, 1.0, 0)]
)
def test_force_stop_on_executing_replica(
    stop_at: int, target: float | None, status: int
) -> None:
    method = ScipyMinimizer(
        IntegrationProbeAlgorithm(), 2, 1, seed=BASE_SEED, history="iteration"
    )
    batches: list[int] = []

    def observe(x: NDArray[np.float64]) -> None:
        batches.append(len(method.history.records))

    with pytest.warns(UserWarning, match="probe requested early stop"):
        result = minimize(
            constant_objective,
            [0],
            bounds=[(0, 1)],
            method=method,
            callback=observe,
            options={
                "algorithm": IntegrationProbeAlgorithm(force_stop_at_iteration=stop_at),
                "maxiter": 6,
                "target_fitness": target,
            },
        )
    assert (result.status, result.success, result.fun) == (status, status == 0, 1.0)
    assert ("target" if target is not None else "force stop") in result.message.lower()
    assert "nit" not in result
    assert batches == ([2] if stop_at == 1 else [3, 4])
    assert [json.loads(row)["iteration"] for row in method.history.records] == list(
        range(stop_at + 1)
    )
    recovered = minimize(constant_objective, [0], bounds=[(0, 1)], method=method)
    assert recovered.success and recovered.nit == 2
    assert recovered.history is not result.history


@pytest.mark.parametrize(
    "intermediate", [False, True], ids=["positional", "intermediate-result"]
)
def test_callback_gets_an_independent_full_best_after_every_batch(
    method: ScipyMinimizer, intermediate: bool
) -> None:
    delivered: list[NDArray[np.float64]] = []
    values: list[float] = []

    def positional(x: NDArray[np.float64]) -> bool:
        assert x.shape == (2,) and x[1] == 3 and -1 <= x[0] <= 1
        delivered.append(x)
        values.append(shifted_sphere(x, 4))
        x[:] = np.nan
        return True

    def keyword(*, intermediate_result: OptimizeResult) -> bool:
        assert isinstance(intermediate_result, OptimizeResult)
        assert intermediate_result.fun == shifted_sphere(intermediate_result.x, 4)
        return positional(intermediate_result.x)

    result = minimize(
        shifted_sphere,
        [-1, 3],
        args=(4,),
        bounds=[(-1, 1), (3, 3)],
        method=method,
        callback=keyword if intermediate else positional,
        options={"maxiter": 4},
    )
    assert result.success and result.nit == 4
    assert len(delivered) == 2
    assert not np.shares_memory(delivered[0], delivered[1])
    assert all(not np.shares_memory(x, result.x) for x in delivered)
    assert result.fun == shifted_sphere(result.x, 4) == values[-1] <= values[0]


@pytest.mark.parametrize(("fitness", "status"), [(-np.inf, 99), (np.inf, 3)])
def test_callback_interruption_precedence_and_reuse(
    method: ScipyMinimizer, fitness: float, status: int
) -> None:
    calls = 0

    def stop(x: NDArray[np.float64]) -> None:
        nonlocal calls
        calls += 1
        raise StopIteration

    result = minimize(
        lambda x: fitness,
        [0],
        bounds=[(0, 1)],
        method=method,
        callback=stop,
        options={"maxiter": 4},
    )
    assert calls == 1 and result.nit == 2
    assert (result.status, result.success, result.fun) == (status, False, fitness)
    assert ("callback" if status == 99 else "valid") in result.message.lower()
    recovered = minimize(constant_objective, [0], bounds=[(0, 1)], method=method)
    assert (recovered.status, recovered.nit, recovered.fun) == (0, 2, 1.0)


def test_reusing_adaptive_pso_restarts_the_entire_trajectory_independently_of_x0() -> (
    None
):
    def build() -> ScipyMinimizer:
        return ScipyMinimizer(
            PSO(inertia=(0.4, 0.9)),
            2,
            3,
            seed=BASE_SEED,
            backend=Local(),
            processor=Serial(),
            history="iteration",
        )

    def solve(method: ScipyMinimizer, x0: float) -> OptimizeResult:
        return minimize(
            shifted_sphere, [x0], args=(2,), bounds=[(-3, 3)], method=method
        )

    def states(history: OptimizationHistory) -> list[tuple[int, str]]:
        snapshots = []
        for row in history.records:
            record = json.loads(row)
            particle = record["particle"]
            # Compare positions, fitnesses, velocities and personal bests across runs.
            del particle["identifier"]
            snapshots.append(
                (record["iteration"], json.dumps(particle, sort_keys=True))
            )
        return sorted(snapshots)

    method = build()
    reference = solve(build(), -3)
    expected = states(reference.history)
    assert [step for step, _ in expected] == [0] * 3 + [1] * 3 + [2] * 3
    for x0 in (-3.0, -3.0, 3.0):
        result = solve(method, x0)
        assert states(result.history) == expected
        np.testing.assert_array_equal(result.x, reference.x)
        assert (result.fun, result.nit, result.status) == (reference.fun, 2, 0)


def test_overrides_apply_only_to_the_selected_problem(method: ScipyMinimizer) -> None:
    settings = {
        "n_iterations": 1,
        "maxiter": 3,
        "history": "iteration",
        "seed": seed_for(1),
        "target_fitness": 1.0,
        "n_particles": 1,
    }

    def solve(adapter: ScipyMinimizer, options: dict[str, object]) -> OptimizeResult:
        return minimize(
            constant_objective, [0], bounds=[(-1, 1)], method=adapter, options=options
        )

    overridden = solve(method, settings)
    reference = solve(
        ScipyMinimizer(
            IntegrationProbeAlgorithm(),
            1,
            1,
            seed=seed_for(1),
            history="iteration",
            target_fitness=1.0,
            maxiter=3,
        ),
        {},
    )

    def coordinates(history: OptimizationHistory) -> list[object]:
        return [row[history.columns.index("0")] for row in history.generator]

    assert coordinates(overridden.history) == coordinates(reference.history)
    assert len(coordinates(overridden.history)) == 2
    assert overridden.nit == 1 and "target" in overridden.message.lower()
    seen: list[tuple[float, ...]] = []

    def record(x: NDArray[np.float64]) -> float:
        seen.append(tuple(x))
        return constant_objective(x)

    restored = minimize(record, [0], bounds=[(-1, 1)], method=method)
    trajectory = seen.copy()
    seen.clear()
    minimize(
        record,
        [0],
        bounds=[(-1, 1)],
        method=ScipyMinimizer(IntegrationProbeAlgorithm(), 2, 2, seed=BASE_SEED),
    )
    assert seen == trajectory and len(seen) == 6
    assert restored.nit == 2 and not restored.history.config.history_enabled
    assert "budget" in restored.message.lower()


def test_cache_is_private_to_each_objective_and_layout(method: ScipyMinimizer) -> None:
    first_inputs: list[tuple[float, ...]] = []
    second_inputs: list[tuple[float, ...]] = []

    def first(x: NDArray[np.float64], offset: int) -> float:
        first_inputs.append(tuple(x))
        return shifted_sphere(x, offset)

    def second(x: NDArray[np.float64], offset: int) -> float:
        second_inputs.append(tuple(x))
        return offset - sphere(*x)

    a = minimize(
        first,
        [0, 2],
        args=(1,),
        bounds=[(-1, 1), (2, 2)],
        method=method,
        options={"use_cache": True},
    )
    b = minimize(
        second,
        [7, -1, 4],
        args=(100,),
        bounds=[(7, 7), (-1, 1), (4, 4)],
        method=method,
        options={"use_cache": True},
    )
    # All free-coordinate keys collide across runs, including the shared midpoint
    # of both initialized particles. Each objective must have its own cache.
    assert [x for x, _ in first_inputs] == [x for _, x, _ in second_inputs]
    assert len(first_inputs) == len(set(first_inputs)) == 5
    assert all(f == 2 for _, f in first_inputs)
    assert all((f, g) == (7, 4) for f, _, g in second_inputs)
    assert a.fun == shifted_sphere(a.x, 1)
    assert b.fun == 100 - sphere(*b.x)


@pytest.mark.parametrize("failure", ["objective", "callback", "bounds"])
def test_recovery_from_exceptions_publishes_only_this_runs_history(
    method: ScipyMinimizer, failure: str
) -> None:
    options = {"history": "iteration", "n_particles": 1, "maxiter": 4}
    baseline = minimize(
        constant_objective, [0], bounds=[(0, 1)], method=method, options=options
    )
    saved = list(baseline.history.records)
    calls = 0

    def objective(x: NDArray[np.float64]) -> float:
        nonlocal calls
        calls += 1
        if failure == "objective" and calls == 3:
            raise StopIteration("from objective")
        return 1.0

    def callback(x: NDArray[np.float64]) -> None:
        if failure == "callback":
            raise RuntimeError("from callback")

    with pytest.raises(Exception):  # noqa: B017
        minimize(
            objective,
            [0],
            bounds=[(1, 2) if failure == "bounds" else (0, 1)],
            method=method,
            callback=callback,
            options=options,
        )
    partial = None
    records: list[str] = []
    if failure == "bounds":
        assert not hasattr(method, "history")
        assert calls == 0
    else:
        partial = method.history
        records = list(partial.records)
        expected_steps = [0, 1] if failure == "objective" else [0, 1, 2]
        assert [json.loads(row)["iteration"] for row in records] == expected_steps
        assert calls == 3
    recovered = minimize(
        constant_objective, [0], bounds=[(0, 1)], method=method, options=options
    )
    assert (recovered.nit, recovered.status, recovered.fun) == (4, 0, 1.0)
    assert (
        recovered.history is not partial and recovered.history is not baseline.history
    )
    index = method.history.columns.index("0")
    assert [r[index] for r in method.history.generator] == [
        r[index] for r in baseline.history.generator
    ]
    assert list(baseline.history.records) == saved
    if partial is not None:
        assert list(partial.records) == records


def test_history_and_result_survive_free_fixed_free_sequence(
    method: ScipyMinimizer, tmp_path: Path
) -> None:
    first = minimize(
        shifted_sphere,
        [0, 4, -1],
        args=(2,),
        bounds=[(0, 1), (4, 4), (-2, -1)],
        method=method,
        options={"history": "iteration"},
    )
    history = method.history
    assert first.history is history
    position, fitness = first.x.copy(), first.fun
    records = list(history.records)
    columns, internal_columns = history.columns.copy(), history.internal_columns_.copy()
    decoded, encoded = list(history.generator), list(history.internal_generator_)
    assert [c for c in columns if c.isdecimal()] == ["0", "1"]
    assert [c for c in internal_columns if c.isdecimal()] == ["0", "1"]
    for row in decoded:
        left, right = row[columns.index("0")], row[columns.index("1")]
        assert isinstance(left, float) and 0 <= left <= 1
        assert isinstance(right, float) and -2 <= right <= -1
        assert row[columns.index("fitness")] == sphere(left, 4, right) + 2

    fixed = minimize(constant_objective, [7], bounds=[(7, 7)], method=method)
    assert fixed.history is method.history and fixed.history is not history
    assert not fixed.history.records and not fixed.history.config.history_enabled
    assert fixed.nit == 0 and fixed.nfev == 1 and fixed.x.tolist() == [7]
    last = minimize(shifted_sphere, [8], args=(1,), bounds=[(8, 9)], method=method)
    assert (
        last.history is method.history
        and last.history is not fixed.history
        and last.history is not history
    )
    assert (
        last.nit == 2 and "nfev" not in last and last.fun == shifted_sphere(last.x, 1)
    )
    np.testing.assert_array_equal(first.x, position)
    assert first.fun == fitness and first.x[1] == 4
    assert history.columns == columns and history.internal_columns_ == internal_columns
    assert list(history.records) == records
    assert (
        list(history.generator) == decoded
        and list(history.internal_generator_) == encoded
    )
    # Export the retained history after both a schema change and a new population.
    archive = tmp_path / "scipy-history.pkl"
    history.export(archive, format="pickle")
    restored = OptimizationHistory.load(archive)
    assert restored.columns == columns and list(restored.generator) == decoded
    first.x[:] = 0
    assert list(history.generator) == decoded and fixed.x.tolist() == [7]


@pytest.mark.multiprocess
def test_loky_evaluates_reconstructed_vectors_in_workers(
    method: ScipyMinimizer,
) -> None:
    result = minimize(
        worker_shifted_sphere,
        [0, 3],
        args=(7, getpid()),
        bounds=[(-1, 1), (3, 3)],
        method=method,
        options={"processor": Joblib(n_jobs=2, joblib_backend="loky")},
    )
    assert (result.status, result.success, result.nit) == (0, True, 2)
    assert result.x.shape == (2,) and result.x[1] == 3
    assert result.fun == shifted_sphere(result.x, 7)


@pytest.mark.multiprocess
def test_distributed_one_island_stops_and_driver_publishes_global_best(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with get_context("spawn").Manager() as manager:
        backend = TestDistributedBackend(manager)
        method = ScipyMinimizer(
            IntegrationProbeAlgorithm(
                force_stop_at_iteration=1, force_stop_on_island="island:0"
            ),
            2,
            2,
            seed=BASE_SEED,
            backend=backend,
            history="iteration",
        )
        migration = IslandIsolation()
        start = Mock(wraps=migration.start)
        monkeypatch.setattr(migration, "start", start)
        driver = getpid()
        callbacks: list[NDArray[np.float64]] = []

        def observe(x: NDArray[np.float64]) -> None:
            assert getpid() == driver
            assert len(backend.worker_pids) == 2 and method.history.records
            callbacks.append(x)

        result = minimize(
            worker_shifted_sphere,
            [0, 3],
            args=(7, driver),
            bounds=[(-1, 1), (3, 3)],
            method=method,
            callback=observe,
            options={"migration": migration},
        )
        start.assert_called_once_with()
        assert result.history is method.history
        assert (result.status, result.success) == (0, True)
        assert "nit" not in result and "nfev" not in result
        records = [json.loads(row) for row in method.history.records]
        assert {
            island: sorted(r["iteration"] for r in records if r["origin"] == island)
            for island in backend.local_bests
        } == {"island:0": [0, 0, 1, 1], "island:1": [0, 0, 1, 1, 2, 2]}
        bests = [island.result for island in backend.local_bests.values()]
        assert len(bests) == 2 and all(best is not None for best in bests)
        assert result.fun == min(best["fitness"] for best in bests if best is not None)
        assert result.fun == shifted_sphere(result.x, 7) and result.x[1] == 3
        assert len(callbacks) == 1
        np.testing.assert_array_equal(callbacks[0], result.x)
