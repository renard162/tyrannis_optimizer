"""Shared configuration and history lifecycle of an abstract orchestrator."""

from inspect import isabstract
from typing import final
from unittest.mock import patch

import pytest
from _support.numerics import BASE_SEED
from _support.objectives import sphere

from tyrannis import Optimizer
from tyrannis.algorithm.pso import PSO
from tyrannis.backend.local import Local
from tyrannis.core.orchestrator import OrchestratorBase
from tyrannis.core.results import HistoryConfig
from tyrannis.migration.global_best import GlobalBest
from tyrannis.migration.island_isolation import IslandIsolation
from tyrannis.processor.serial import Serial
from tyrannis.space.continuous import Continuous


@final
class _Orchestrator(OrchestratorBase):
    """Expose the shared history lifecycle without an execution method."""

    def __init__(self, space: Continuous, config: HistoryConfig) -> None:
        self._result = None
        self._space = space
        self._history_config = config
        self.reset_history()

    def reset_history(self) -> None:
        self._configure_history()

    def publish(self, records: list[str]) -> None:
        self._update_history(records)


@pytest.mark.parametrize("argument", ["n_iterations", "n_particles"])
@pytest.mark.parametrize("invalid", [0, 1.5], ids=["below-minimum", "non-integer"])
def test_invalid_counts(argument: str, invalid: float) -> None:
    with pytest.raises(Exception):  # noqa: B017
        counts = {"n_iterations": 1, "n_particles": 1, argument: invalid}
        vars(OrchestratorBase)["_validate_common_arguments"](**counts)


def test_default_components_receive_the_shared_configuration() -> None:
    space = Continuous({"x": (-5.0, 5.0)}, sphere)
    algorithm = PSO()
    with (
        patch.object(Local, "initialize_context", autospec=True) as initialize,
        patch.object(Serial, "initialize_context", autospec=True) as prepare,
    ):
        _ = Optimizer(space, algorithm, n_iterations=2, n_particles=3, seed=BASE_SEED)
    options = initialize.call_args.kwargs
    assert isinstance(initialize.call_args.args[0], Local)
    assert isinstance(options["migration"], IslandIsolation)
    assert options["algorithm"] is algorithm
    assert options["n_iter"] == 2
    assert options["n_particles"] == 3
    assert options["seed"] == BASE_SEED
    assert space.encoded_boundaries == {"x": (-5.0, 5.0)}
    assert vars(algorithm)["_boundaries"] == space.encoded_boundaries
    assert vars(algorithm)["_max_iterations"] == 2
    assert vars(algorithm)["_n_particles"] == 3
    assert isinstance(options["processor"], Serial)
    assert prepare.call_args.kwargs["seed"] == BASE_SEED


EVENT_FLAGS = (
    "migration",
    "pre_iteration",
    "new_particle",
    "error",
    "iteration",
    "status",
    "best",
)
ALL_EVENTS = HistoryConfig(**dict.fromkeys(EVENT_FLAGS, True))

EVENT_SELECTIONS = [
    pytest.param(None, HistoryConfig(), id="none"),
    pytest.param("error", HistoryConfig(error=True), id="one-event"),
    pytest.param("all", ALL_EVENTS, id="all"),
    pytest.param(["error", "best"], HistoryConfig(error=True, best=True), id="list"),
]


@pytest.mark.parametrize(("history", "expected"), EVENT_SELECTIONS)
def test_history_configuration_accepts_supported_forms(
    history: str | list[str] | None, expected: HistoryConfig
) -> None:
    optimizer = Optimizer(Continuous((0.0, 1.0), sphere), PSO(), 1, 1, history=history)
    assert optimizer.history.config == expected


INVALID_EVENT_SELECTIONS = [
    pytest.param("unknown", id="invalid-event"),
    pytest.param(["error", "unknown"], id="invalid-list-event"),
    pytest.param(["all"], id="all-in-list"),
    pytest.param(42, id="invalid-type"),
]


@pytest.mark.parametrize("history", INVALID_EVENT_SELECTIONS)
def test_history_configuration_rejects_invalid_forms(history: object) -> None:
    with pytest.raises(Exception):  # noqa: B017
        vars(OrchestratorBase)["_configure_particle_history"](history)


def test_published_history_lifecycle() -> None:
    assert isabstract(OrchestratorBase)
    space = Continuous({"x": (-1.0, 1.0)}, sphere)
    space.initialize_context(seed=BASE_SEED)
    orchestrator = _Orchestrator(space, HistoryConfig(iteration=True))
    history = orchestrator.history
    records = history.records
    assert records == []
    assert history.config == HistoryConfig(iteration=True)
    assert "x" in history.columns
    first, second = '{"iteration": 1}', '{"iteration": 2}'
    orchestrator.publish([first])
    assert records == [first]
    orchestrator.publish([first, second])
    assert orchestrator.history is history
    assert history.records is records
    assert records == [first, second]
    orchestrator.reset_history()
    assert orchestrator.history is not history
    assert orchestrator.history.records == []
    assert history.records == [first, second]


def test_explicit_local_components_receive_the_selected_history() -> None:
    backend, processor, algorithm = Local(), Serial(), PSO()
    space = Continuous({"x": (-1.0, 1.0)}, sphere)
    with patch.object(backend, "initialize_context") as initialize:
        _ = Optimizer(
            space, algorithm, 1, 1, backend, processor, GlobalBest(), history="best"
        )
    options = initialize.call_args.kwargs
    assert options["algorithm"] is algorithm
    assert options["processor"] is processor
    assert isinstance(options["migration"], IslandIsolation)
    assert options["history_config"] == HistoryConfig(best=True)
