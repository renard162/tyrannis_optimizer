"""Policy contracts of island migration on the processor and driver sides."""

from __future__ import annotations

import json
from collections.abc import Callable
from unittest.mock import Mock, patch

import numpy as np
import pytest

from tests._support.numerics import BASE_SEED
from tyrannis.backend.distributed.communication.no_communication import (
    NoCommunicationDriver,
    NoCommunicationProcessor,
)
from tyrannis.core.results import HistoryConfig
from tyrannis.core.signals import LocalEvent
from tyrannis.migration.island_migration import (
    IslandMigration,
    IslandMigrationProcessor,
)


class _ObservableIslandMigration(IslandMigration):
    """Expose migration decisions without running the background router."""

    def process_pending(self) -> None:
        self._process_incoming()

    def choose_receiver(self, donor: str, candidates: list[str]) -> str:
        return self._choose_receiver(donor, candidates)

    def choose_donor(self, candidates: list[str]) -> str:
        return self._choose_donor(candidates)


def _driver(
    migration: _ObservableIslandMigration, islands: tuple[str, ...] = ("a", "b")
) -> NoCommunicationDriver:
    communication = NoCommunicationDriver(list(islands))
    migration.initialize_context(
        communication,
        NoCommunicationProcessor,
        {},
        HistoryConfig(migration=True),
        BASE_SEED,
    )
    return communication


def _state(
    communication: NoCommunicationDriver,
    island: str,
    population: dict[str, int],
    actual_iter: int = 1,
) -> None:
    communication.incoming_queues[island].put(
        json.dumps(
            {
                "type": IslandMigrationProcessor.STATE,
                "actual_iter": actual_iter,
                "population": population,
            }
        )
    )


def _processor(
    *, selection: str = "random", trigger: str = "n_iter"
) -> tuple[IslandMigrationProcessor, NoCommunicationProcessor]:
    communication = NoCommunicationProcessor("a")
    communication.start()
    processor = IslandMigrationProcessor(
        initial_iter=2,
        selection=selection,
        migration_size=1,
        min_interval=2,
        movement_strategy="random",
        trigger=trigger,
        communication_processor=communication,
        seed=BASE_SEED,
    )
    processor.initialize_loop_context(LocalEvent())
    return processor, communication


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: IslandMigration(initial_iter=0), "initial_iter"),
        (lambda: IslandMigration(movement_strategy="unknown"), "movement_strategy"),
        (lambda: IslandMigration(particle_selection="unknown"), "selection"),
        (lambda: IslandMigration(trigger="unknown"), "trigger"),
        (lambda: IslandMigration(movement_strategy="ring"), "requires"),
        (lambda: IslandMigration(migration_size=0), "migration_size"),
        (lambda: IslandMigration(min_population=-1), "min_population"),
        (lambda: IslandMigration(min_interval=0), "min_interval"),
        (lambda: IslandMigration(migration_probability=-0.1), "migration_probability"),
        (lambda: IslandMigration(migration_probability=1.1), "migration_probability"),
    ],
    ids=[
        "initial-iteration",
        "movement",
        "selection",
        "trigger",
        "ring-sync",
        "size",
        "population",
        "interval",
        "probability-low",
        "probability-high",
    ],
)
def test_configuration_rejects_invalid_policy(
    factory: Callable[[], IslandMigration], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _ = factory()


def test_processor_lifecycle_connects_signal_and_communication() -> None:
    communication = NoCommunicationProcessor("a")
    processor = IslandMigrationProcessor(
        2, "random", 1, 2, "random", "n_iter", communication, BASE_SEED
    )
    signal = LocalEvent()

    with (
        patch.object(communication, "start") as start,
        patch.object(communication, "stop") as stop,
        patch.object(communication, "set_message_signal") as set_signal,
    ):
        processor.start()
        processor.initialize_loop_context(signal)
        processor.finalize_loop_context()
        processor.stop()

    start.assert_called_once_with()
    stop.assert_called_once_with()
    assert set_signal.call_args_list == [((signal,),), ((None,),)]


def test_processor_publishes_population_from_initial_iteration() -> None:
    processor, communication = _processor()
    population = {"first": np.float64(1.5), "second": np.float64(4)}
    insert = Mock()
    depart = Mock()

    processor.migration_control(1, population, None, insert, depart)
    assert communication.outgoing_queue.empty()

    processor.migration_control(2, population, "first", insert, depart)
    assert json.loads(communication.outgoing_queue.get_nowait()) == {
        "type": IslandMigrationProcessor.STATE,
        "actual_iter": 2,
        "population": {"first": 1.5, "second": 4},
    }
    insert.assert_not_called()
    depart.assert_not_called()


@pytest.mark.parametrize(
    ("selection", "expected"),
    [("best", "good"), ("worst", "bad"), ("random", None)],
    ids=["best", "worst", "random"],
)
def test_processor_selects_and_sends_departing_particle(
    selection: str, expected: str | None
) -> None:
    processor, communication = _processor(selection=selection)
    population = {
        "good": np.float64(1),
        "middle": np.float64(5),
        "bad": np.float64(9),
    }
    departures: list[str] = []

    def depart(identifier: str) -> dict[str, object]:
        departures.append(identifier)
        return {"identifier": identifier, "fitness": 5}

    communication.messages.put(
        json.dumps(
            {
                "type": IslandMigrationProcessor.MIGRATION_REQUEST,
                "request_id": "r1",
                "actual_iter": 2,
            }
        )
    )

    processor.migration_control(2, population, None, Mock(), depart)

    assert len(departures) == 1
    selected = departures[0]
    assert selected in population
    if expected is not None:
        assert selected == expected
    assert json.loads(communication.outgoing_queue.get_nowait()) == {
        "type": IslandMigrationProcessor.PARTICLE,
        "request_id": "r1",
        "actual_iter": 2,
        "particle": {"identifier": selected, "fitness": 5},
    }
    assert json.loads(communication.outgoing_queue.get_nowait())["type"] == (
        IslandMigrationProcessor.STATE
    )


def test_processor_inserts_valid_arrival_and_ignores_invalid_messages() -> None:
    processor, communication = _processor()
    insert = Mock()
    depart = Mock()
    arrival = {"identifier": "incoming", "fitness": 3}
    for message in (
        "not-json",
        "[]",
        json.dumps({"type": IslandMigrationProcessor.PARTICLE, "particle": None}),
        json.dumps(
            {"type": IslandMigrationProcessor.MIGRATION_REQUEST, "request_id": 1}
        ),
    ):
        communication.messages.put(message)
    communication.messages.put(
        json.dumps({"type": IslandMigrationProcessor.PARTICLE, "particle": arrival})
    )

    processor.migration_control(1, {}, None, insert, depart)

    insert.assert_called_once_with(arrival)
    depart.assert_not_called()
    assert communication.outgoing_queue.empty()


def test_synchronous_processor_pauses_once_and_resumes_on_release() -> None:
    processor, communication = _processor(trigger="synchronous")
    insert = Mock()
    depart = Mock()
    processor.migration_control(2, {}, None, insert, depart)
    assert json.loads(communication.outgoing_queue.get_nowait())["type"] == (
        IslandMigrationProcessor.STATE
    )
    communication.messages.put(
        json.dumps(
            {"type": IslandMigrationProcessor.SYNCHRONIZATION_RELEASE, "actual_iter": 2}
        )
    )

    processor.synchronization_control(2, insert, depart)

    assert json.loads(communication.outgoing_queue.get_nowait()) == {
        "type": IslandMigrationProcessor.SYNCHRONIZATION_PAUSE,
        "actual_iter": 2,
    }
    processor.synchronization_control(3, insert, depart)
    assert communication.outgoing_queue.empty()


@pytest.mark.parametrize(
    ("strategy", "destinations"),
    [
        ("random", {"a", "c", "d"}),
        ("random_neighbor", {"a", "c"}),
        ("greedy", {"c"}),
        ("greedy_neighbor", {"c"}),
        ("greedy_random", {"c"}),
        ("star", {"a"}),
        ("ring", {"c"}),
        ("ring_sequential", {"c"}),
    ],
    ids=[
        "random",
        "random-neighbor",
        "greedy",
        "greedy-neighbor",
        "greedy-random",
        "star",
        "ring",
        "ring-sequential",
    ],
)
def test_movement_strategy_selects_its_destination(
    strategy: str, destinations: set[str]
) -> None:
    migration = _ObservableIslandMigration(
        initial_iter=5,
        movement_strategy=strategy,
        trigger="synchronous" if strategy == "ring" else "n_iter",
    )
    communication = _driver(migration, ("d", "b", "a", "c"))
    for island, fitness in (("a", 9), ("b", 5), ("c", 0), ("d", 1)):
        _state(communication, island, {f"{island}-particle": fitness})
    migration.process_pending()

    observed = {migration.choose_receiver("b", ["a", "c", "d"]) for _ in range(50)}
    assert observed == destinations


def test_star_hub_and_ring_wrap_follow_sorted_island_order() -> None:
    star = _ObservableIslandMigration(movement_strategy="star")
    ring = _ObservableIslandMigration(movement_strategy="ring", trigger="synchronous")
    _ = _driver(star, ("c", "a", "b"))
    _ = _driver(ring, ("c", "a", "b"))

    assert star.choose_receiver("a", ["b", "c"]) in {"b", "c"}
    assert ring.choose_receiver("c", ["a", "b"]) == "a"


@pytest.mark.parametrize(
    "strategy", ["greedy", "greedy_random"], ids=["greedy", "greedy-random"]
)
def test_fitness_based_movement_falls_back_when_no_fitness_exists(
    strategy: str,
) -> None:
    migration = _ObservableIslandMigration(initial_iter=5, movement_strategy=strategy)
    communication = _driver(migration, ("a", "b", "c"))
    _state(communication, "a", {"a-0": 1})
    _state(communication, "b", {})
    _state(communication, "c", {})
    migration.process_pending()

    observed = {migration.choose_receiver("a", ["b", "c"]) for _ in range(50)}

    assert observed == {"b", "c"}


def test_population_balance_favors_the_larger_eligible_donor() -> None:
    balanced = _ObservableIslandMigration(initial_iter=5, balance_population=True)
    unbalanced = _ObservableIslandMigration(initial_iter=5, balance_population=False)
    for migration in (balanced, unbalanced):
        communication = _driver(migration)
        _state(communication, "a", {f"a-{index}": 1 for index in range(5)})
        _state(communication, "b", {"b-0": 1})
        migration.process_pending()

    balanced_large = sum(balanced.choose_donor(["a", "b"]) == "a" for _ in range(200))
    unbalanced_large = sum(
        unbalanced.choose_donor(["a", "b"]) == "a" for _ in range(200)
    )

    assert balanced_large > unbalanced_large


def test_async_flow_reserves_departures_forwards_particles_and_updates_state() -> None:
    migration = _ObservableIslandMigration(
        initial_iter=2,
        min_interval=2,
        min_population=1,
        migration_size=3,
        movement_strategy="ring_sequential",
    )
    communication = _driver(migration)
    donor_population = {"a-0": 1, "a-1": 2, "a-2": 3}
    receiver_population = {"b-0": 4}
    _state(communication, "a", donor_population)
    _state(communication, "b", receiver_population)
    migration.process_pending()
    assert all(queue.empty() for queue in communication.outgoing_queues.values())

    _state(communication, "a", donor_population, 2)
    _state(communication, "b", receiver_population, 2)
    migration.process_pending()
    requests: list[dict[str, object]] = [
        json.loads(communication.outgoing_queues["a"].get_nowait()) for _ in range(2)
    ]
    assert all(
        request["type"] == IslandMigrationProcessor.MIGRATION_REQUEST
        and request["actual_iter"] == 2
        for request in requests
    )
    assert len({request["request_id"] for request in requests}) == 2
    assert communication.outgoing_queues["a"].empty()
    assert communication.outgoing_queues["b"].empty()

    for request, identifier in zip(requests, ("a-0", "a-1"), strict=True):
        communication.incoming_queues["a"].put(
            json.dumps(
                {
                    "type": IslandMigrationProcessor.PARTICLE,
                    "request_id": request["request_id"],
                    "particle": {
                        "identifier": identifier,
                        "fitness": donor_population[identifier],
                    },
                }
            )
        )
    migration.process_pending()

    forwarded: list[dict[str, object]] = [
        json.loads(communication.outgoing_queues["b"].get_nowait()) for _ in range(2)
    ]
    assert forwarded == [
        {
            "type": IslandMigrationProcessor.PARTICLE,
            "actual_iter": 2,
            "particle": {
                "identifier": identifier,
                "fitness": donor_population[identifier],
            },
        }
        for identifier in ("a-0", "a-1")
    ]
    history: list[dict[str, object]] = [
        json.loads(entry) for entry in migration.consume_migration_history()
    ]
    assert history == [
        {
            "event": HistoryConfig.get_event("migration"),
            "iteration": 2,
            "origin": "a",
            "destination": "b",
            "particle": {
                "identifier": identifier,
                "fitness": donor_population[identifier],
            },
        }
        for identifier in ("a-0", "a-1")
    ]

    # The old processor snapshots must not undo confirmed transfers.
    _state(communication, "a", donor_population, 4)
    _state(communication, "b", receiver_population, 4)
    migration.process_pending()
    assert communication.outgoing_queues["a"].empty()
    assert json.loads(communication.outgoing_queues["b"].get_nowait())["type"] == (
        IslandMigrationProcessor.MIGRATION_REQUEST
    )


def test_migration_size_caps_requests_when_population_is_available() -> None:
    migration = _ObservableIslandMigration(
        initial_iter=2, migration_size=2, movement_strategy="ring_sequential"
    )
    communication = _driver(migration)
    _state(communication, "a", {f"a-{index}": index for index in range(4)}, 2)
    _state(communication, "b", {}, 2)

    migration.process_pending()

    assert communication.outgoing_queues["a"].qsize() == 2
    assert communication.outgoing_queues["b"].empty()


@pytest.mark.parametrize(
    ("probability", "requests"), [(0.0, 0), (1.0, 1)], ids=["never", "always"]
)
def test_random_trigger_respects_probability_extremes(
    probability: float, requests: int
) -> None:
    migration = _ObservableIslandMigration(
        initial_iter=2, trigger="random", migration_probability=probability
    )
    communication = _driver(migration)
    _state(communication, "a", {"a-0": 1}, 2)
    _state(communication, "b", {"b-0": 2}, 2)
    migration.process_pending()

    assert (
        sum(queue.qsize() for queue in communication.outgoing_queues.values())
        == requests
    )


def test_synchronous_flow_waits_for_states_and_releases_after_particle() -> None:
    migration = _ObservableIslandMigration(
        initial_iter=2,
        min_interval=2,
        min_population=1,
        trigger="synchronous",
        movement_strategy="ring",
    )
    communication = _driver(migration)
    _state(communication, "a", {"a-0": 1, "a-1": 2}, 2)
    communication.incoming_queues["a"].put(
        json.dumps(
            {"type": IslandMigrationProcessor.SYNCHRONIZATION_PAUSE, "actual_iter": 2}
        )
    )
    migration.process_pending()
    assert all(queue.empty() for queue in communication.outgoing_queues.values())

    communication.incoming_queues["b"].put(
        json.dumps(
            {"type": IslandMigrationProcessor.SYNCHRONIZATION_PAUSE, "actual_iter": 2}
        )
    )
    migration.process_pending()
    assert all(queue.empty() for queue in communication.outgoing_queues.values())

    _state(communication, "b", {"b-0": 3}, 2)
    migration.process_pending()
    requests: list[dict[str, object]] = [
        json.loads(communication.outgoing_queues["a"].get_nowait())
    ]
    request = requests[0]
    assert request["type"] == IslandMigrationProcessor.MIGRATION_REQUEST
    assert all(queue.empty() for queue in communication.outgoing_queues.values())

    communication.incoming_queues["a"].put(
        json.dumps(
            {
                "type": IslandMigrationProcessor.PARTICLE,
                "request_id": request["request_id"],
                "particle": {"identifier": "a-0", "fitness": 1},
            }
        )
    )
    migration.process_pending()

    release = {
        "type": IslandMigrationProcessor.SYNCHRONIZATION_RELEASE,
        "actual_iter": 2,
    }
    assert json.loads(communication.outgoing_queues["a"].get_nowait()) == release
    assert json.loads(communication.outgoing_queues["b"].get_nowait()) == {
        "type": IslandMigrationProcessor.PARTICLE,
        "particle": {"identifier": "a-0", "fitness": 1},
        "actual_iter": 2,
    }
    assert json.loads(communication.outgoing_queues["b"].get_nowait()) == release


def test_driver_ignores_unusable_protocol_messages() -> None:
    migration = _ObservableIslandMigration(initial_iter=5)
    communication = _driver(migration)
    for message in (
        "not-json",
        "[]",
        json.dumps(
            {"type": IslandMigrationProcessor.STATE, "actual_iter": 2, "population": []}
        ),
        json.dumps(
            {"type": IslandMigrationProcessor.PARTICLE, "request_id": "unknown"}
        ),
        json.dumps(
            {"type": IslandMigrationProcessor.SYNCHRONIZATION_PAUSE, "actual_iter": 5}
        ),
    ):
        communication.incoming_queues["a"].put(message)

    migration.process_pending()

    assert all(queue.empty() for queue in communication.outgoing_queues.values())
    assert migration.consume_migration_history() == []
