"""Optimizer-owned propagation contracts through the default execution stack."""

from collections import Counter
from math import inf

import pytest
from _support.assertions import assert_optimizer_result_consistent
from _support.integration_probe import IntegrationProbeAlgorithm
from _support.numerics import BASE_SEED
from _support.objectives import fails_for_positive, sphere

from tyrannis import Optimizer
from tyrannis.core.results import HistoryConfig
from tyrannis.space.continuous import Continuous


def test_default_execution_exposes_consistent_result_and_requested_history() -> None:
    # Unlike component smokes, this checks combined history selection and both
    # public generators against the result of the same default-stack execution.
    # Construction is explicit because configuration propagation is the contract.
    optimizer = Optimizer(
        space=Continuous(boundaries={"x": (1.0, 3.0)}, cost_function=sphere),
        algorithm=IntegrationProbeAlgorithm(
            double_eval=False, force_stop_at_iteration=None
        ),
        n_particles=4,
        n_iterations=1,
        seed=BASE_SEED,
        history=["iteration", "best"],
    )

    assert optimizer.fit() is optimizer
    assert_optimizer_result_consistent(optimizer)

    rows: list[dict[str, object]] = [
        dict(zip(optimizer.history_columns, values, strict=True))
        for values in optimizer.history_generator
    ]
    internal_rows: list[dict[str, object]] = [
        dict(zip(optimizer.internal_history_columns_, values, strict=True))
        for values in optimizer.internal_history_generator_
    ]
    iteration_event = HistoryConfig.get_event("iteration")
    best_event = HistoryConfig.get_event("local_best")
    assert Counter((row["event"], row["iteration"]) for row in rows) == {
        (iteration_event, 0): 4,
        (iteration_event, 1): 4,
        (best_event, 0): 1,
        (best_event, 1): 1,
    }
    # Continuous has identical encoded/decoded values. Compare by content,
    # without imposing physical particle ordering on either generator.
    assert Counter(tuple(row.values()) for row in rows) == Counter(
        tuple(row[column] for column in optimizer.history_columns)
        for row in internal_rows
    )
    identifiers = {row["identifier"] for row in rows if row["event"] == iteration_event}
    assert len(identifiers) == 4
    assert all(isinstance(identifier, str) and identifier for identifier in identifiers)
    fitnesses: list[float] = []
    for row in rows:
        assert row["identifier"] in identifiers
        x, fitness = row["x"], row["fitness"]
        assert isinstance(x, float)
        assert isinstance(fitness, float)
        assert sphere(x=x) == fitness
        if row["event"] == iteration_event:
            fitnesses.append(fitness)
        if row["event"] == best_event and row["iteration"] == 1:
            assert optimizer.result_ == {
                "identifier": row["identifier"],
                "variables": {"x": x},
                "fitness": fitness,
            }
            assert sphere(x=x) == optimizer.best_fitness
    assert optimizer.best_fitness == min(fitnesses)


def test_raise_strategy_propagates_real_objective_failure() -> None:
    # Component smokes do not propagate a failing objective through Optimizer.
    optimizer = Optimizer(
        space=Continuous(
            boundaries={"x": (1.0, 3.0)}, cost_function=fails_for_positive
        ),
        algorithm=IntegrationProbeAlgorithm(),
        n_particles=4,
        n_iterations=1,
        seed=BASE_SEED,
        fitness_failure_strategy="raise",
    )

    with pytest.raises(Exception):  # noqa: B017
        _ = optimizer.fit()


def test_invalidate_strategy_completes_with_invalid_public_result() -> None:
    # Unlike an objective returning inf, this exercises exception handling all
    # the way from the configured strategy to actual evaluation and reporting.
    optimizer = Optimizer(
        space=Continuous(
            boundaries={"x": (1.0, 3.0)}, cost_function=fails_for_positive
        ),
        algorithm=IntegrationProbeAlgorithm(),
        n_particles=4,
        n_iterations=1,
        seed=BASE_SEED,
        fitness_failure_strategy="invalidate",
        history="error",
    )

    assert optimizer.fit() is optimizer
    assert_optimizer_result_consistent(optimizer)
    assert optimizer.best_fitness == inf
    rows: list[dict[str, object]] = [
        dict(zip(optimizer.history_columns, values, strict=True))
        for values in optimizer.history_generator
    ]
    # Errors from both initialization and the real update reach the public API.
    assert {row["iteration"] for row in rows} == {0, 1}
    assert {row["event"] for row in rows} == {HistoryConfig.get_event("error")}
