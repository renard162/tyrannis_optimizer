"""Contracts of the global-best migration policy on both sides of an island."""

from __future__ import annotations

import json
from unittest.mock import Mock, patch

import numpy as np
import pytest

from tyrannis.backend.distributed.communication.no_communication import (
    NoCommunicationDriver,
    NoCommunicationProcessor,
)
from tyrannis.core.results import HistoryConfig
from tyrannis.core.signals import LocalEvent
from tyrannis.migration.global_best import GlobalBest, GlobalBestProcessor


class _ObservableGlobalBest(GlobalBest):
    """Expose driver policy without involving its background router."""

    def process_pending(self) -> None:
        self._process_incoming()

    @property
    def best_state(self) -> tuple[float | None, dict[str, object] | None]:
        return self._global_best_fitness, self._global_best_particle


def _processor(
    *, synchronous: bool = False
) -> tuple[GlobalBestProcessor, NoCommunicationProcessor]:
    communication = NoCommunicationProcessor("island:a")
    communication.start()
    processor = GlobalBestProcessor(2, 2, synchronous, communication, None)
    processor.initialize_loop_context(LocalEvent())
    return processor, communication


def _driver(
    *, synchronous: bool = False
) -> tuple[_ObservableGlobalBest, NoCommunicationDriver]:
    communication = NoCommunicationDriver(["island:a", "island:b"])
    driver = _ObservableGlobalBest(
        initial_iter=2, check_interval=2, synchronous=synchronous
    )
    driver.initialize_context(
        communication, NoCommunicationProcessor, {}, HistoryConfig(), None
    )
    return driver, communication


def test_driver_rejects_initial_iteration_below_one() -> None:
    with pytest.raises(ValueError):
        _ = GlobalBest(initial_iter=0)


def test_driver_rejects_non_positive_check_interval() -> None:
    with pytest.raises(ValueError):
        _ = GlobalBest(check_interval=0)


def test_processor_lifecycle_connects_signal_and_delegates_communication() -> None:
    communication = NoCommunicationProcessor("island:a")
    processor = GlobalBestProcessor(2, 2, False, communication, None)
    signal = LocalEvent()

    with (
        patch.object(communication, "set_message_signal") as set_signal,
        patch.object(communication, "start") as start,
        patch.object(communication, "stop") as stop,
    ):
        processor.start()
        processor.initialize_loop_context(signal)
        processor.finalize_loop_context()
        processor.stop()

    start.assert_called_once_with()
    stop.assert_called_once_with()
    assert set_signal.call_args_list == [((signal,),), ((None,),)]


def test_async_processor_publishes_only_strict_improvements_at_checkpoints() -> None:
    processor, communication = _processor()
    population: dict[str, np.float64] = {}
    insert = Mock()
    depart = Mock()

    processor.migration_control(
        1, population, '{"identifier":"p","fitness":4}', insert, depart
    )
    assert communication.outgoing_queue.empty()

    processor.migration_control(
        2, population, '{"identifier":"p","fitness":4}', insert, depart
    )
    assert json.loads(communication.outgoing_queue.get_nowait()) == {
        "type": GlobalBestProcessor.MESSAGE_TYPE,
        "particle": {"identifier": "p", "fitness": 4},
        "actual_iter": 2,
    }

    processor.migration_control(
        3, population, '{"identifier":"p","fitness":2}', insert, depart
    )
    processor.migration_control(
        4, population, '{"identifier":"p","fitness":4}', insert, depart
    )
    processor.migration_control(
        6, population, '{"identifier":"p","fitness":5}', insert, depart
    )
    assert communication.outgoing_queue.empty()

    processor.migration_control(
        8, population, '{"identifier":"p","fitness":2}', insert, depart
    )
    assert json.loads(communication.outgoing_queue.get_nowait()) == {
        "type": GlobalBestProcessor.MESSAGE_TYPE,
        "particle": {"identifier": "p", "fitness": 2},
        "actual_iter": 8,
    }
    assert communication.outgoing_queue.empty()


@pytest.mark.parametrize(
    "candidate",
    ["not-json", "[]", '{"identifier":"p","fitness": "bad"}', '{"fitness": NaN}'],
    ids=["undecodable", "non-dict", "invalid-fitness", "non-finite"],
)
def test_processor_ignores_invalid_publication_candidates(candidate: str) -> None:
    processor, communication = _processor()
    processor.migration_control(2, {}, candidate, Mock(), Mock())
    assert communication.outgoing_queue.empty()


def test_processor_replaces_worst_valid_particle_through_callbacks() -> None:
    processor, communication = _processor()
    population = {
        "good": np.float64(2),
        "worst": np.float64(9),
        "undefined": np.float64(np.nan),
    }
    departures: list[str] = []
    arrivals: list[dict[str, object]] = []

    def depart(identifier: str) -> dict[str, object] | None:
        departures.append(identifier)
        del population[identifier]
        return {"identifier": identifier}

    def insert(particle: dict[str, object]) -> None:
        arrivals.append(particle)
        identifier = particle["identifier"]
        fitness = particle["fitness"]
        assert isinstance(identifier, str)
        assert isinstance(fitness, (int, float))
        population[identifier] = np.float64(fitness)

    communication.messages.put(
        json.dumps(
            {
                "type": GlobalBestProcessor.MESSAGE_TYPE,
                "particle": {
                    "identifier": "remote",
                    "fitness": 1,
                    "variables": {"x": 3},
                },
            }
        )
    )
    processor.migration_control(1, population, None, insert, depart)

    assert departures == ["worst"]
    assert arrivals == [{"identifier": "worst", "fitness": 1, "variables": {"x": 3}}]
    assert set(population) == {"good", "worst", "undefined"}
    assert population["worst"] == 1


def test_processor_leaves_population_without_valid_replacement_unchanged() -> None:
    processor, communication = _processor()
    population = {"undefined": np.float64(np.nan)}
    insert = Mock()
    depart = Mock()
    communication.messages.put(
        json.dumps(
            {
                "type": GlobalBestProcessor.MESSAGE_TYPE,
                "particle": {"identifier": "remote", "fitness": 1},
            }
        )
    )

    processor.migration_control(1, population, None, insert, depart)

    insert.assert_not_called()
    depart.assert_not_called()
    assert list(population) == ["undefined"]


@pytest.mark.parametrize(
    "message",
    [
        "not-json",
        "[]",
        '{"type":"global_best_update","particle":{"fitness":1}}',
        '{"type":"global_best_update","particle":{"identifier":"p","fitness":"bad"}}',
    ],
    ids=["undecodable", "non-dict", "missing-identifier", "invalid-fitness"],
)
def test_processor_ignores_invalid_arrivals(message: str) -> None:
    processor, communication = _processor()
    insert = Mock()
    depart = Mock()
    communication.messages.put(message)
    processor.migration_control(1, {"local": np.float64(3)}, None, insert, depart)
    insert.assert_not_called()
    depart.assert_not_called()


def test_sync_processor_pauses_and_resumes_at_checkpoint() -> None:
    processor, communication = _processor(synchronous=True)
    population: dict[str, np.float64] = {}
    insert = Mock()
    depart = Mock()
    candidate = '{"identifier":"p","fitness":3}'

    processor.migration_control(1, population, candidate, insert, depart)
    processor.synchronization_control(1, insert, depart)
    assert communication.outgoing_queue.empty()

    processor.migration_control(2, population, candidate, insert, depart)
    communication.messages.put(
        json.dumps(
            {"type": GlobalBestProcessor.SYNCHRONIZATION_RELEASE, "actual_iter": 2}
        )
    )
    processor.synchronization_control(2, insert, depart)
    assert json.loads(communication.outgoing_queue.get_nowait()) == {
        "type": GlobalBestProcessor.MESSAGE_TYPE,
        "particle": {"identifier": "p", "fitness": 3},
        "actual_iter": 2,
    }
    assert json.loads(communication.outgoing_queue.get_nowait()) == {
        "type": GlobalBestProcessor.SYNCHRONIZATION_PAUSE,
        "actual_iter": 2,
    }

    processor.migration_control(
        3, population, '{"identifier":"p","fitness":1}', insert, depart
    )
    assert communication.outgoing_queue.empty()
    processor.migration_control(
        4, population, '{"identifier":"p","fitness":1}', insert, depart
    )
    assert (
        json.loads(communication.outgoing_queue.get_nowait())["particle"]["fitness"]
        == 1
    )


def test_async_driver_broadcasts_only_strict_global_improvements_to_other_islands() -> (
    None
):
    driver, communication = _driver()
    first = {"identifier": "a", "fitness": 5}
    better = {"identifier": "b", "fitness": 2}

    communication.incoming_queues["island:a"].put(
        json.dumps(
            {
                "type": GlobalBestProcessor.MESSAGE_TYPE,
                "particle": first,
                "actual_iter": 2,
            }
        )
    )
    driver.process_pending()
    assert driver.best_state == (5, first)
    assert communication.outgoing_queues["island:a"].empty()
    assert json.loads(communication.outgoing_queues["island:b"].get_nowait()) == {
        "type": GlobalBestProcessor.MESSAGE_TYPE,
        "particle": first,
    }

    for particle in (first, {"identifier": "a", "fitness": 6}):
        communication.incoming_queues["island:a"].put(
            json.dumps(
                {
                    "type": GlobalBestProcessor.MESSAGE_TYPE,
                    "particle": particle,
                    "actual_iter": 4,
                }
            )
        )
    driver.process_pending()
    assert all(queue.empty() for queue in communication.outgoing_queues.values())

    communication.incoming_queues["island:b"].put(
        json.dumps(
            {
                "type": GlobalBestProcessor.MESSAGE_TYPE,
                "particle": better,
                "actual_iter": 4,
            }
        )
    )
    driver.process_pending()
    assert driver.best_state == (2, better)
    assert json.loads(communication.outgoing_queues["island:a"].get_nowait()) == {
        "type": GlobalBestProcessor.MESSAGE_TYPE,
        "particle": better,
    }
    assert communication.outgoing_queues["island:b"].empty()


@pytest.mark.parametrize(
    "message",
    [
        "not-json",
        "[]",
        '{"type":"global_best_update","particle":{"fitness":1},"actual_iter":"bad"}',
        '{"type":"global_best_update","particle":{"fitness":"bad"},"actual_iter":2}',
    ],
    ids=["undecodable", "non-dict", "invalid-iteration", "invalid-fitness"],
)
def test_driver_ignores_invalid_candidates(message: str) -> None:
    driver, communication = _driver()
    communication.incoming_queues["island:a"].put(message)
    driver.process_pending()
    assert driver.best_state == (None, None)
    assert all(queue.empty() for queue in communication.outgoing_queues.values())


def test_sync_driver_waits_for_all_islands_then_broadcasts_and_releases() -> None:
    driver, communication = _driver(synchronous=True)
    first = {"identifier": "a", "fitness": 5}
    best = {"identifier": "b", "fitness": 2}

    communication.incoming_queues["island:a"].put(
        json.dumps(
            {
                "type": GlobalBestProcessor.MESSAGE_TYPE,
                "particle": first,
                "actual_iter": 2,
            }
        )
    )
    communication.incoming_queues["island:a"].put(
        json.dumps(
            {"type": GlobalBestProcessor.SYNCHRONIZATION_PAUSE, "actual_iter": 2}
        )
    )
    driver.process_pending()
    assert driver.best_state == (None, None)
    assert all(queue.empty() for queue in communication.outgoing_queues.values())

    communication.incoming_queues["island:b"].put(
        json.dumps(
            {
                "type": GlobalBestProcessor.MESSAGE_TYPE,
                "particle": best,
                "actual_iter": 2,
            }
        )
    )
    communication.incoming_queues["island:b"].put(
        json.dumps(
            {"type": GlobalBestProcessor.SYNCHRONIZATION_PAUSE, "actual_iter": 2}
        )
    )
    driver.process_pending()

    assert driver.best_state == (2, best)
    assert json.loads(communication.outgoing_queues["island:a"].get_nowait()) == {
        "type": GlobalBestProcessor.MESSAGE_TYPE,
        "particle": best,
    }
    release = {
        "type": GlobalBestProcessor.SYNCHRONIZATION_RELEASE,
        "actual_iter": 2,
    }
    assert json.loads(communication.outgoing_queues["island:a"].get_nowait()) == release
    assert json.loads(communication.outgoing_queues["island:b"].get_nowait()) == release


def test_driver_start_and_stop_manage_communication_lifecycle() -> None:
    driver, communication = _driver()
    with (
        patch.object(communication, "start") as start,
        patch.object(communication, "stop") as stop,
    ):
        driver.start()
        driver.stop()

    start.assert_called_once_with()
    stop.assert_called_once_with()
