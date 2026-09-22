"""Small public contracts for processor results and history settings."""

import pytest

from tyrannis.core.results import HistoryConfig, ProcessorResult

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


def test_processor_results_start_with_independent_empty_histories() -> None:
    first = ProcessorResult()
    second = ProcessorResult()

    assert first.result is None
    assert second.result is None
    assert first.history == second.history == []
    first.history.append("event")
    assert second.history == []
