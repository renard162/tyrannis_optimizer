"""Small public contracts for processor results and history settings."""

import csv
import json
import pickle
from math import inf, nan
from pathlib import Path
from unittest.mock import patch

import pytest
from _support.numerics import BASE_SEED

from tyrannis.core.results import HistoryConfig, OptimizationHistory, ProcessorResult
from tyrannis.space.binary import Binary
from tyrannis.space.integer import Integer

HISTORY_FLAGS = [
    "migration",
    "pre_iteration",
    "new_particle",
    "error",
    "iteration",
    "status",
    "best",
]


@pytest.mark.parametrize("flag", HISTORY_FLAGS, ids=HISTORY_FLAGS)
def test_any_history_flag_enables_history(flag: str) -> None:
    assert not HistoryConfig().history_enabled
    assert HistoryConfig(**{flag: True}).history_enabled


def test_history_event_names_match_registered_mapping() -> None:
    assert {
        name: HistoryConfig.get_event(name) for name in HistoryConfig.EVENT
    } == HistoryConfig.EVENT
    assert HistoryConfig().registered_events == list(HistoryConfig.EVENT.values())


def test_history_event_mapping_is_read_only() -> None:
    events = HistoryConfig.EVENT
    before = list(events.items())
    # Exercise mutations that the public Mapping type also rejects statically.
    with pytest.raises(Exception):  # noqa: B017
        events["new"] = "new event"  # pyright: ignore[reportIndexIssue]
    with pytest.raises(Exception):  # noqa: B017
        events["iteration"] = "changed"  # pyright: ignore[reportIndexIssue]
    with pytest.raises(Exception):  # noqa: B017
        del events["iteration"]  # pyright: ignore[reportIndexIssue]
    assert list(HistoryConfig.EVENT.items()) == before


def test_processor_results_start_with_independent_empty_histories() -> None:
    first = ProcessorResult()
    second = ProcessorResult()

    assert first.result is None
    assert second.result is None
    assert first.history == second.history == []
    first.history.append("event")
    assert second.history == []


def _make_history(records: list[str]) -> OptimizationHistory:
    space = Integer({"x": (-5, 5)})
    space.initialize_context(seed=BASE_SEED)
    return OptimizationHistory(records, HistoryConfig(), space)


@pytest.fixture
def history() -> OptimizationHistory:
    return _make_history([])


def test_decoded_and_encoded_columns(history: OptimizationHistory) -> None:
    assert history.columns == [
        "event",
        "iteration",
        "identifier",
        "origin",
        "destination",
        "fitness",
        "x",
    ]
    assert history.internal_columns_ == [
        *history.columns,
        "internal_state",
    ]


def test_filtered_rows_keep_particle_state() -> None:
    event = HistoryConfig.get_event("iteration")
    particle = {
        "identifier": "particle:1",
        "variables": {"x": 2.3},
        "fitness": 4.0,
        "velocity": {"x": 0.5},
    }
    history = _make_history(
        [
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

    assert list(history.generator) == [[event, 3, "particle:1", "a", "b", 4.0, 2]]
    internal_rows = list(history.internal_generator_)
    assert len(internal_rows) == 1
    assert internal_rows[0][:-1] == [event, 3, "particle:1", "a", "b", 4.0, 2.3]
    internal_state: object = internal_rows[0][-1]
    assert isinstance(internal_state, str)
    assert json.loads(internal_state) == {"velocity": {"x": 0.5}}


INVALID_CSV_DESTINATIONS = [
    pytest.param("history.txt", ",", id="extension"),
    pytest.param("missing/history.csv", ",", id="parent"),
    pytest.param("history.csv", "::", id="separator"),
]


@pytest.mark.parametrize(("name", "separator"), INVALID_CSV_DESTINATIONS)
def test_export_validates_destination(
    history: OptimizationHistory, tmp_path: Path, name: str, separator: str
) -> None:
    with pytest.raises(Exception):  # noqa: B017
        history.export(tmp_path / name, sep=separator)


CSV_FITNESS_CASES = [
    pytest.param(4.0, "iteration", "4.0", id="finite"),
    pytest.param(inf, "iteration", "Infinity", id="infinity"),
    pytest.param(nan, "iteration", "NaN", id="nan"),
    pytest.param(inf, "error", "Raise", id="error-infinity"),
    pytest.param(-inf, "iteration", "-Infinity", id="negative-infinity"),
    pytest.param(-inf, "error", "-Infinity", id="error-negative-infinity"),
]


@pytest.mark.parametrize(("fitness", "kind", "text"), CSV_FITNESS_CASES)
@pytest.mark.parametrize("internal_state", [False, True], ids=["decoded", "internal"])
def test_csv_fitness(
    tmp_path: Path, fitness: float, kind: str, text: str, internal_state: bool
) -> None:
    event = HistoryConfig.get_event(kind)
    history = _make_history(
        [
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

    destination = tmp_path / "history.csv"
    history.export(
        destination, sep=";", encoding="utf-16", internal_state=internal_state
    )

    with destination.open(newline="", encoding="utf-16") as stream:
        rows = list(csv.reader(stream, delimiter=";"))
    variables = ["2.3", ""] if internal_state else ["2"]
    assert rows == [
        history.internal_columns_ if internal_state else history.columns,
        [event, "2", "particle:1", "", "", text, *variables],
    ]


def test_export_uses_internal_rows(tmp_path: Path) -> None:
    event = HistoryConfig.get_event("iteration")
    history = _make_history(
        [
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

    destination = tmp_path / "internal.csv"
    history.export(destination, format="csv", internal_state=True)

    with destination.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == history.internal_columns_
    assert rows[1][:-1] == [event, "", "particle:1", "", "", "4.0", "2.3"]
    assert json.loads(rows[1][-1]) == {"velocity": {"x": 0.5}}


def test_constructor_snapshots_records_and_interprets_one_at_a_time() -> None:
    event = HistoryConfig.get_event("iteration")
    records = [
        json.dumps({"event": event, "particle": {"variables": {"x": 1.3}}}),
        "unparsed trailing record" * 100,
    ]
    history = _make_history(records)
    records.clear()
    assert "unparsed trailing record" not in repr(history)
    for rows, values in [
        (history.generator, [1]),
        (history.internal_generator_, [1.3, None]),
    ]:
        assert next(rows) == [event, None, None, None, None, None, *values]
        with pytest.raises(Exception):  # noqa: B017
            _ = next(rows)


def _assert_records_are_read_only(history: OptimizationHistory) -> None:
    records = history.records
    assert {"append", "extend", "clear", "pop", "remove", "insert"}.isdisjoint(
        dir(records)
    )
    # Try the forbidden writes on a populated view so index zero is valid.
    with pytest.raises(Exception):  # noqa: B017
        records[0] = "changed"  # pyright: ignore[reportIndexIssue]
    with pytest.raises(Exception):  # noqa: B017
        del records[0]  # pyright: ignore[reportIndexIssue]
    with pytest.raises(Exception):  # noqa: B017
        history.records = []  # pyright: ignore[reportAttributeAccessIssue]


def test_records_are_read_only() -> None:
    history = _make_history(["first"])
    _assert_records_are_read_only(history)
    assert list(history.records) == ["first"]


def test_load_legacy_pickle_with_list_in_records_slot(tmp_path: Path) -> None:
    event = HistoryConfig.get_event("iteration")
    record = json.dumps({"event": event, "particle": {"variables": {"x": 1.3}}})
    history = _make_history([record])
    path = tmp_path / "legacy.pkl"
    # Reproduce the former slot layout in the actual Pickle stream.
    with patch.object(history, "_records", [record]):
        history.export(path, format="pickle")
    restored = OptimizationHistory.load(path)
    _assert_records_are_read_only(restored)
    assert list(restored.records) == [record]
    assert list(restored.generator) == [[event, None, None, None, None, None, 1]]


def test_pickle_preserves_independent_decoder_state(tmp_path: Path) -> None:
    # A local objective cannot be pickled. Only its space's decoder can travel.
    def objective(bit: bool) -> float:
        return float(bit)

    space = Binary(["bit"], objective, decoder="s-shape")
    reference = Binary(["bit"], decoder="s-shape")
    for decoder in (space, reference):
        decoder.initialize_context(seed=BASE_SEED)
    inputs = {"bit": 0.0}
    event = HistoryConfig.get_event("iteration")
    record = json.dumps(
        {"event": event, "particle": {"variables": inputs, "fitness": 1.0}}
    )
    history = OptimizationHistory([record] * 12, HistoryConfig(iteration=True), space)

    def decode_bit(decoder: Binary) -> bool:
        decoded = decoder.decode(inputs)
        assert isinstance(decoded, dict)
        return decoded["bit"]

    first = list(history.generator)
    assert len(first) == len(history.records) == 12
    assert [row[-1] for row in first] == [decode_bit(reference) for _ in first]
    assert [decode_bit(space) for _ in first] == [row[-1] for row in first]
    del space

    path = tmp_path / "history.PKL"
    history.export(
        path, format="pickle", sep="ignored", encoding="ignored", internal_state=True
    )
    restored = OptimizationHistory.load(path)
    assert restored.config == history.config == HistoryConfig(iteration=True)
    assert restored.columns == history.columns
    assert restored.internal_columns_ == history.internal_columns_
    assert restored.records == history.records
    assert restored.records is not history.records
    _assert_records_are_read_only(restored)
    expected_internal = [event, None, None, None, None, 1.0, 0.0, None]
    assert list(restored.internal_generator_) == [expected_internal] * len(first)
    expected = [
        [event, None, None, None, None, 1.0, decode_bit(reference)] for _ in first
    ]
    assert list(history.generator) == expected
    assert list(restored.generator) == expected
    second = tmp_path / "restored.pickle"
    restored.export(second, format="pickle")
    assert list(OptimizationHistory.load(second).generator) == list(restored.generator)


def test_pickle_validation(history: OptimizationHistory, tmp_path: Path) -> None:
    with pytest.raises(Exception):  # noqa: B017
        vars(OptimizationHistory)["export"](
            history, tmp_path / "history.csv", format="other"
        )
    with pytest.raises(Exception):  # noqa: B017
        history.export(tmp_path / "history.csv", format="pickle")
    with pytest.raises(Exception):  # noqa: B017
        history.export(tmp_path / "missing" / "history.pkl", format="pickle")

    wrong_extension = tmp_path / "history.csv"
    _ = wrong_extension.write_bytes(pickle.dumps(history))
    with pytest.raises(Exception):  # noqa: B017
        _ = OptimizationHistory.load(wrong_extension)
    wrong_type = tmp_path / "other.pkl"
    _ = wrong_type.write_bytes(pickle.dumps(history.records))
    with pytest.raises(Exception):  # noqa: B017
        _ = OptimizationHistory.load(wrong_type)
