"""Unit contracts owned by the Optimizer orchestrator."""

from __future__ import annotations

import json
from typing import final, override

import pytest
from _support.assertions import assert_optimizer_result_consistent
from _support.factories import make_optimizer
from _support.objectives import sphere

from tyrannis import Optimizer
from tyrannis.algorithm.pso import PSO
from tyrannis.backend.local import Local
from tyrannis.core.results import HistoryConfig, OptimizationHistory, ProcessorResult
from tyrannis.space.continuous import Continuous
from tyrannis.space.integer import Integer


@final
class _BackendProbe(Local):
    """Record execution while providing a controlled backend result."""

    def __init__(self) -> None:
        super().__init__()
        self.executions: int = 0

    def publish(self, result: ProcessorResult) -> None:
        self._result = result

    @override
    def execute(self) -> None:
        self.executions += 1


ConfiguredOptimizer = tuple[Optimizer, _BackendProbe]


@pytest.fixture
def setup_optimizer() -> ConfiguredOptimizer:
    backend = _BackendProbe()
    optimizer = make_optimizer(
        Integer({"x": (-5, 5)}, sphere), PSO(), backend=backend, history=None
    )
    return optimizer, backend


def test_result_properties_before_fit(setup_optimizer: ConfiguredOptimizer) -> None:
    optimizer, _ = setup_optimizer

    assert optimizer.result_ is None
    assert isinstance(optimizer.history, OptimizationHistory)
    assert optimizer.history.config == HistoryConfig()
    assert (
        "x" in optimizer.history.columns and "x" in optimizer.history.internal_columns_
    )
    assert (
        optimizer.history.records
        == list(optimizer.history.generator)
        == list(optimizer.history.internal_generator_)
        == []
    )
    removed = {
        "history_columns",
        "history_generator",
        "internal_history_columns_",
        "internal_history_generator_",
        "save_history_csv",
    }
    assert removed.isdisjoint(dir(optimizer))
    with pytest.raises(Exception):  # noqa: B017
        _ = optimizer.best_solution
    with pytest.raises(Exception):  # noqa: B017
        _ = optimizer.best_fitness


def test_fit_adopts_backend_result(setup_optimizer: ConfiguredOptimizer) -> None:
    optimizer, backend = setup_optimizer
    backend.publish(
        ProcessorResult(
            result={
                "identifier": "island:0|particle:1",
                "variables": {"x": 2.3},
                "fitness": 4.0,
            }
        )
    )

    assert optimizer.fit() is optimizer
    assert backend.executions == 1
    assert optimizer.result_ == {
        "identifier": "island:0|particle:1",
        "variables": {"x": 2},
        "fitness": 4.0,
    }
    assert_optimizer_result_consistent(optimizer)


def test_empty_backend_result(setup_optimizer: ConfiguredOptimizer) -> None:
    optimizer, backend = setup_optimizer
    backend.publish(ProcessorResult(result=None))

    assert optimizer.fit() is optimizer
    assert backend.executions == 1
    assert optimizer.result_ is None
    with pytest.raises(Exception):  # noqa: B017
        _ = optimizer.best_solution


def test_result_and_history_decode_positional_variables() -> None:
    backend = _BackendProbe()
    optimizer = make_optimizer(
        Continuous([(0.0, 5.0)], sphere), PSO(), backend=backend, history="iteration"
    )
    event = HistoryConfig.get_event("iteration")
    particle = {"identifier": "particle:1", "variables": {"0": 2.0}, "fitness": 4.0}
    record = json.dumps(
        {"event": event, "particle": {"variables": {"0": 2.0}, "fitness": 4.0}}
    )
    backend.publish(ProcessorResult(result=particle, history=[record]))

    _ = optimizer.fit()

    assert optimizer.best_solution == [2.0]
    assert optimizer.result_ == {
        "identifier": "particle:1",
        "variables": [2.0],
        "fitness": 4.0,
    }
    assert list(optimizer.history.generator) == [
        [event, None, None, None, None, 4.0, 2.0]
    ]
