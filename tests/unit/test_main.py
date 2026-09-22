"""Unit contracts owned by the Optimizer orchestrator."""

from __future__ import annotations

import csv
import json
from math import inf, nan
from pathlib import Path
from typing import final, override

import pytest

from tests._support.assertions import assert_optimizer_result_consistent
from tests._support.factories import make_optimizer
from tests._support.numerics import BASE_SEED
from tests._support.objectives import sphere
from tyrannis import Optimizer
from tyrannis.algorithm.pso import PSO
from tyrannis.backend.local import Local
from tyrannis.core.results import HistoryConfig, ProcessorResult
from tyrannis.migration.island_isolation import IslandIsolation
from tyrannis.processor.serial import Serial
from tyrannis.space.continuous import Continuous
from tyrannis.space.integer import Integer


@final
class _BackendProbe(Local):
    """Record execution while providing a controlled backend result."""

    def __init__(self) -> None:
        super().__init__()
        self.executions: int = 0

    @override
    def execute(self) -> None:
        self.executions += 1


@pytest.fixture
def configured_optimizer() -> tuple[Optimizer, _BackendProbe]:
    backend = _BackendProbe()
    optimizer = make_optimizer(
        Integer({"x": (-5, 5)}, sphere),
        PSO(),
        backend=backend,
        history=["error", "iteration"],
    )
    return optimizer, backend


@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(1.5, id="float"),
        pytest.param("2", id="string"),
    ],
)
def test_constructor_rejects_invalid_iterations(invalid: object) -> None:
    with pytest.raises(ValueError, match="n_iterations"):
        vars(Optimizer)["__init__"](
            Optimizer.__new__(Optimizer),
            Continuous((0.0, 1.0), sphere),
            PSO(),
            invalid,
            1,
        )


@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(1.5, id="float"),
        pytest.param("2", id="string"),
    ],
)
def test_constructor_rejects_invalid_particles(invalid: object) -> None:
    with pytest.raises(ValueError, match="n_particles"):
        vars(Optimizer)["__init__"](
            Optimizer.__new__(Optimizer),
            Continuous((0.0, 1.0), sphere),
            PSO(),
            1,
            invalid,
        )


def test_constructor_configures_default_components() -> None:
    space = Continuous({"x": (-5.0, 5.0)}, sphere)
    algorithm = PSO()
    optimizer = make_optimizer(
        space, algorithm, n_iterations=2, n_particles=3, seed=BASE_SEED
    )

    assert isinstance(optimizer._backend, Local)
    assert isinstance(optimizer._processor, Serial)
    assert isinstance(optimizer._migration, IslandIsolation)
    assert space.encoded_boundaries == {"x": (-5.0, 5.0)}
    assert algorithm._boundaries == space.encoded_boundaries
    assert algorithm._max_iterations == 2
    assert algorithm._n_particles == 3
    assert optimizer._processor._seed_sequence.entropy == BASE_SEED
    assert optimizer._backend._seed == BASE_SEED


@pytest.mark.parametrize(
    ("history", "expected"),
    [
        pytest.param(None, HistoryConfig(), id="none"),
        pytest.param("error", HistoryConfig(error=True), id="one-event"),
        pytest.param(
            "all",
            HistoryConfig(
                migration=True,
                pre_iteration=True,
                new_particle=True,
                error=True,
                iteration=True,
                status=True,
                best=True,
            ),
            id="all",
        ),
        pytest.param(
            ["error", "best"], HistoryConfig(error=True, best=True), id="list"
        ),
    ],
)
def test_history_configuration_accepts_supported_forms(
    history: str | list[str] | None, expected: HistoryConfig
) -> None:
    assert Optimizer._configure_particle_history(history) == expected


@pytest.mark.parametrize(
    ("history", "error", "message"),
    [
        pytest.param(
            "unknown", ValueError, "Invalid history event", id="invalid-event"
        ),
        pytest.param(
            ["error", "unknown"],
            ValueError,
            "Invalid history event",
            id="invalid-list-event",
        ),
        pytest.param(["all"], ValueError, "cannot be used", id="all-in-list"),
        pytest.param(42, TypeError, "history must be", id="invalid-type"),
    ],
)
def test_history_configuration_rejects_invalid_forms(
    history: object, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        vars(Optimizer)["_configure_particle_history"](history)


def test_result_properties_before_fit(
    configured_optimizer: tuple[Optimizer, _BackendProbe],
) -> None:
    optimizer, _ = configured_optimizer

    assert optimizer.result_ is None
    assert list(optimizer.history_generator) == []
    assert list(optimizer.internal_history_generator_) == []
    with pytest.raises(AttributeError, match="before fit"):
        _ = optimizer.best_solution
    with pytest.raises(AttributeError, match="before fit"):
        _ = optimizer.best_fitness


def test_fit_adopts_backend_result_and_decodes_solution(
    configured_optimizer: tuple[Optimizer, _BackendProbe],
) -> None:
    optimizer, backend = configured_optimizer
    backend._result = ProcessorResult(
        result={
            "identifier": "island:0|particle:1",
            "variables": {"x": 2.3},
            "fitness": 4.0,
        }
    )

    assert optimizer.fit() is optimizer
    assert backend.executions == 1
    assert optimizer.result_ == {
        "identifier": "island:0|particle:1",
        "variables": {"x": 2},
        "fitness": 4.0,
    }
    assert_optimizer_result_consistent(optimizer)


def test_fit_with_missing_backend_result_leaves_public_result_unavailable(
    configured_optimizer: tuple[Optimizer, _BackendProbe],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer, backend = configured_optimizer
    monkeypatch.setattr(backend, "_result", None)

    assert optimizer.fit() is optimizer
    assert backend.executions == 1
    assert optimizer.result_ is None
    with pytest.raises(AttributeError, match="before fit"):
        _ = optimizer.best_solution


def test_result_and_history_decode_positional_variables() -> None:
    backend = _BackendProbe()
    optimizer = make_optimizer(
        Continuous([(0.0, 5.0)], sphere), PSO(), backend=backend, history="iteration"
    )
    event = HistoryConfig.get_event("iteration")
    backend._result = ProcessorResult(
        result={"identifier": "particle:1", "variables": {"0": 2.0}, "fitness": 4.0},
        history=[
            json.dumps(
                {"event": event, "particle": {"variables": {"0": 2.0}, "fitness": 4.0}}
            )
        ],
    )

    _ = optimizer.fit()

    assert optimizer.best_solution == [2.0]
    assert optimizer.result_ == {
        "identifier": "particle:1",
        "variables": [2.0],
        "fitness": 4.0,
    }
    assert list(optimizer.history_generator) == [
        [event, None, None, None, None, 4.0, 2.0]
    ]


def test_history_columns_include_public_and_encoded_variables(
    configured_optimizer: tuple[Optimizer, _BackendProbe],
) -> None:
    optimizer, _ = configured_optimizer

    assert optimizer.history_columns == [
        "event",
        "iteration",
        "identifier",
        "origin",
        "destination",
        "fitness",
        "x",
    ]
    assert optimizer.internal_history_columns_ == [
        *optimizer.history_columns,
        "internal_state",
    ]


def test_history_generators_filter_events_and_preserve_internal_state(
    configured_optimizer: tuple[Optimizer, _BackendProbe],
) -> None:
    optimizer, backend = configured_optimizer
    event = HistoryConfig.get_event("iteration")
    particle = {
        "identifier": "particle:1",
        "variables": {"x": 2.3},
        "fitness": 4.0,
        "velocity": {"x": 0.5},
    }
    backend._result = ProcessorResult(
        history=[
            json.dumps({"event": "unregistered", "particle": particle}),
            json.dumps(
                {
                    "event": event,
                    "iteration": 3,
                    "origin": "a",
                    "destination": "b",
                    "particle": particle,
                }
            ),
        ]
    )
    _ = optimizer.fit()

    assert list(optimizer.history_generator) == [
        [event, 3, "particle:1", "a", "b", 4.0, 2]
    ]
    internal_rows = list(optimizer.internal_history_generator_)
    assert len(internal_rows) == 1
    assert internal_rows[0][:-1] == [event, 3, "particle:1", "a", "b", 4.0, 2.3]
    internal_state: object = internal_rows[0][-1]
    assert isinstance(internal_state, str)
    assert json.loads(internal_state) == {"velocity": {"x": 0.5}}


@pytest.mark.parametrize(
    ("name", "separator", "error"),
    [
        pytest.param("history.txt", ",", ValueError, id="extension"),
        pytest.param("missing/history.csv", ",", FileNotFoundError, id="parent"),
        pytest.param("history.csv", "::", ValueError, id="separator"),
    ],
)
def test_save_history_csv_validates_destination(
    configured_optimizer: tuple[Optimizer, _BackendProbe],
    tmp_path: Path,
    name: str,
    separator: str,
    error: type[Exception],
) -> None:
    optimizer, _ = configured_optimizer

    with pytest.raises(error):
        optimizer.save_history_csv(tmp_path / name, sep=separator)


@pytest.mark.parametrize(
    ("fitness", "event_type", "rendered"),
    [
        pytest.param(4.0, "iteration", "4.0", id="finite"),
        pytest.param(inf, "iteration", "Infinity", id="infinity"),
        pytest.param(nan, "iteration", "NaN", id="nan"),
        pytest.param(inf, "error", "Raise", id="error-infinity"),
    ],
)
def test_save_history_csv_writes_header_rows_and_fitness(
    configured_optimizer: tuple[Optimizer, _BackendProbe],
    tmp_path: Path,
    fitness: float,
    event_type: str,
    rendered: str,
) -> None:
    optimizer, backend = configured_optimizer
    event = HistoryConfig.get_event(event_type)
    backend._result = ProcessorResult(
        history=[
            json.dumps(
                {
                    "event": event,
                    "iteration": 2,
                    "particle": {
                        "identifier": "particle:1",
                        "variables": {"x": 2.3},
                        "fitness": fitness,
                    },
                }
            )
        ]
    )
    _ = optimizer.fit()

    destination = tmp_path / "history.csv"
    optimizer.save_history_csv(destination, sep=";")

    with destination.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream, delimiter=";"))
    assert rows == [
        optimizer.history_columns,
        [event, "2", "particle:1", "", "", rendered, "2"],
    ]


def test_save_history_csv_uses_internal_rows(
    configured_optimizer: tuple[Optimizer, _BackendProbe], tmp_path: Path
) -> None:
    optimizer, backend = configured_optimizer
    event = HistoryConfig.get_event("iteration")
    backend._result = ProcessorResult(
        history=[
            json.dumps(
                {
                    "event": event,
                    "particle": {
                        "identifier": "particle:1",
                        "variables": {"x": 2.3},
                        "fitness": 4.0,
                        "velocity": {"x": 0.5},
                    },
                }
            )
        ]
    )
    _ = optimizer.fit()

    destination = tmp_path / "internal.csv"
    optimizer.save_history_csv(destination, internal_state=True)

    with destination.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == optimizer.internal_history_columns_
    assert rows[1][:-1] == [event, "", "particle:1", "", "", "4.0", "2.3"]
    assert json.loads(rows[1][-1]) == {"velocity": {"x": 0.5}}
