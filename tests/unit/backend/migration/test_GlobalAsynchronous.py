from __future__ import annotations

import json
from queue import Queue
from threading import Event
from unittest.mock import MagicMock

import pytest

from tyrannis.backend.migration.global_asynchronous import (
    GlobalAsynchronous,
    GlobalAsynchronousProcessor,
)
from tyrannis.core.backend_communication import (
    CommunicationProcessorBase,
)
from tyrannis.core.signals import LocalEvent


class MockCommunicationProcessor(CommunicationProcessorBase):
    """In-memory communication processor used by the unit tests."""

    def __init__(self, identification: str, **kwargs: object) -> None:
        self.identification = identification
        self._messages: Queue[str] = Queue()
        self._outgoing_queue: Queue[str] = Queue()

        self.start_calls: list[LocalEvent] = []
        self.stop_calls = 0

    @property
    def messages(self) -> Queue[str]:
        return self._messages

    @property
    def outgoing_queue(self) -> Queue[str]:
        return self._outgoing_queue

    def start(self, stop_signal: LocalEvent) -> None:
        self.start_calls.append(stop_signal)

    def stop(self) -> None:
        self.stop_calls += 1


class MockCommunicationDriver:
    """In-memory communication driver used by the unit tests."""

    def __init__(self, island_ids: list[str]) -> None:
        self.incoming_queues = {island_id: Queue[str]() for island_id in island_ids}

        self.outgoing_queues = {island_id: Queue[str]() for island_id in island_ids}

        self.start_calls = 0
        self.stop_calls = 0

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1


def make_particle(
    identifier: str = "particle-0",
    fitness: float = 10.0,
    **kwargs: object,
) -> dict:
    """Create a serialized particle representation."""

    particle = {
        "identifier": identifier,
        "fitness": fitness,
    }
    particle.update(kwargs)
    return particle


def serialize_particle(
    identifier: str = "particle-0",
    fitness: float = 10.0,
    **kwargs: object,
) -> str:
    return json.dumps(
        make_particle(
            identifier=identifier,
            fitness=fitness,
            **kwargs,
        )
    )


def serialize_update(
    identifier: str = "particle-0",
    fitness: float = 10.0,
    **kwargs: object,
) -> str:
    return json.dumps(
        {
            "type": GlobalAsynchronousProcessor.MESSAGE_TYPE,
            "particle": make_particle(
                identifier=identifier,
                fitness=fitness,
                **kwargs,
            ),
        }
    )


# ============================================================================
# GlobalAsynchronousProcessor
# ============================================================================


class TestGlobalAsynchronousProcessorInitialization:
    """Tests for GlobalAsynchronousProcessor initialization."""

    def test_initializes_with_default_state(self) -> None:
        communication = MockCommunicationProcessor("island-0")

        processor = GlobalAsynchronousProcessor(
            initial_iter=10,
            check_interval=5,
            communication_processor=communication,
        )

        assert processor._initial_iter == 10
        assert processor._check_interval == 5
        assert processor._synchronization_iter == 10
        assert processor._communication_processor is communication
        assert processor._last_published_fitness is None

    @pytest.mark.parametrize(
        "initial_iter",
        [-1, -10],
    )
    def test_rejects_negative_initial_iter(
        self,
        initial_iter: int,
    ) -> None:
        communication = MockCommunicationProcessor("island-0")

        with pytest.raises(ValueError, match="initial_iter"):
            GlobalAsynchronousProcessor(
                initial_iter=initial_iter,
                check_interval=5,
                communication_processor=communication,
            )

    @pytest.mark.parametrize(
        "check_interval",
        [0, -1, -10],
    )
    def test_rejects_non_positive_check_interval(
        self,
        check_interval: int,
    ) -> None:
        communication = MockCommunicationProcessor("island-0")

        with pytest.raises(ValueError, match="check_interval"):
            GlobalAsynchronousProcessor(
                initial_iter=1,
                check_interval=check_interval,
                communication_processor=communication,
            )


class TestGlobalAsynchronousProcessorLifecycle:
    """Tests for the processor communication lifecycle."""

    def test_start_delegates_to_communication_processor(self) -> None:
        communication = MockCommunicationProcessor("island-0")

        processor = GlobalAsynchronousProcessor(
            initial_iter=1,
            check_interval=5,
            communication_processor=communication,
        )

        stop_signal = Event()

        processor.start(stop_signal)  # type: ignore

        assert communication.start_calls == [stop_signal]

    def test_stop_delegates_to_communication_processor(self) -> None:
        communication = MockCommunicationProcessor("island-0")

        processor = GlobalAsynchronousProcessor(
            initial_iter=1,
            check_interval=5,
            communication_processor=communication,
        )

        processor.stop()

        assert communication.stop_calls == 1


class TestGlobalAsynchronousProcessorSynchronization:
    """Tests for iteration-based migration scheduling."""

    def test_does_not_publish_before_initial_iter(self) -> None:
        communication = MockCommunicationProcessor("island-0")

        processor = GlobalAsynchronousProcessor(
            initial_iter=10,
            check_interval=5,
            communication_processor=communication,
        )

        local_best = serialize_particle(fitness=1.0)

        insert_arrival_particle = MagicMock()
        departure_particle = MagicMock()

        for actual_iter in range(10):
            processor.migration_control(
                actual_iter=actual_iter,
                population={},
                local_best=local_best,
                insert_arrival_particle=insert_arrival_particle,
                departure_particle=departure_particle,
            )

        assert communication.outgoing_queue.empty()
        assert processor._synchronization_iter == 10

    def test_publishes_at_initial_iter(self) -> None:
        communication = MockCommunicationProcessor("island-0")

        processor = GlobalAsynchronousProcessor(
            initial_iter=10,
            check_interval=5,
            communication_processor=communication,
        )

        local_best = serialize_particle(
            identifier="particle-1",
            fitness=1.0,
        )

        processor.migration_control(
            actual_iter=10,
            population={},
            local_best=local_best,
            insert_arrival_particle=MagicMock(),
            departure_particle=MagicMock(),
        )

        assert communication.outgoing_queue.qsize() == 1
        assert processor._synchronization_iter == 15

    def test_first_synchronization_point_is_initial_iter(self) -> None:
        communication = MockCommunicationProcessor("island-0")

        processor = GlobalAsynchronousProcessor(
            initial_iter=7,
            check_interval=4,
            communication_processor=communication,
        )

        assert processor._synchronization_iter == 7

    def test_synchronization_points_follow_interval(self) -> None:
        communication = MockCommunicationProcessor("island-0")

        processor = GlobalAsynchronousProcessor(
            initial_iter=7,
            check_interval=4,
            communication_processor=communication,
        )

        expected_points = [7, 11, 15, 19]

        for expected in expected_points:
            assert processor._synchronization_iter == expected

            processor.migration_control(
                actual_iter=expected,
                population={},
                local_best=None,
                insert_arrival_particle=MagicMock(),
                departure_particle=MagicMock(),
            )

    def test_large_iteration_jump_advances_to_next_future_point(
        self,
    ) -> None:
        communication = MockCommunicationProcessor("island-0")

        processor = GlobalAsynchronousProcessor(
            initial_iter=10,
            check_interval=5,
            communication_processor=communication,
        )

        processor.migration_control(
            actual_iter=27,
            population={},
            local_best=None,
            insert_arrival_particle=MagicMock(),
            departure_particle=MagicMock(),
        )

        assert processor._synchronization_iter == 30

    def test_large_iteration_jump_does_not_publish_multiple_times(
        self,
    ) -> None:
        communication = MockCommunicationProcessor("island-0")

        processor = GlobalAsynchronousProcessor(
            initial_iter=10,
            check_interval=5,
            communication_processor=communication,
        )

        local_best = serialize_particle(fitness=1.0)

        processor.migration_control(
            actual_iter=27,
            population={},
            local_best=local_best,
            insert_arrival_particle=MagicMock(),
            departure_particle=MagicMock(),
        )

        assert communication.outgoing_queue.qsize() == 1
        assert processor._synchronization_iter == 30

    def test_does_not_block_when_not_at_synchronization_point(
        self,
    ) -> None:
        communication = MockCommunicationProcessor("island-0")

        processor = GlobalAsynchronousProcessor(
            initial_iter=10,
            check_interval=5,
            communication_processor=communication,
        )

        communication.messages.put(
            serialize_update(
                identifier="remote-1",
                fitness=2.0,
            )
        )

        population = {
            "local-1": 10.0,
            "local-2": 20.0,
        }

        inserted = []
        departed = []

        processor.migration_control(
            actual_iter=3,
            population=population,  # type: ignore
            local_best=None,
            insert_arrival_particle=inserted.append,
            departure_particle=departed.append,
        )

        assert len(inserted) == 1
        assert inserted[0]["identifier"] == "local-2"
        assert inserted[0]["fitness"] == 2.0
        assert departed == ["local-2"]


class TestGlobalAsynchronousProcessorPublication:
    """Tests for publication of local best solutions."""

    def test_publishes_valid_local_best(self) -> None:
        communication = MockCommunicationProcessor("island-0")

        processor = GlobalAsynchronousProcessor(
            initial_iter=1,
            check_interval=5,
            communication_processor=communication,
        )

        local_best = serialize_particle(
            identifier="particle-1",
            fitness=10.5,
            position=[1.0, 2.0],
        )

        processor.migration_control(
            actual_iter=1,
            population={},
            local_best=local_best,
            insert_arrival_particle=MagicMock(),
            departure_particle=MagicMock(),
        )

        message = communication.outgoing_queue.get_nowait()
        payload = json.loads(message)

        assert payload["type"] == "global_best_update"
        assert payload["particle"]["identifier"] == "particle-1"
        assert payload["particle"]["fitness"] == 10.5
        assert payload["particle"]["position"] == [1.0, 2.0]

    def test_publishes_none_local_best_is_not_possible(self) -> None:
        communication = MockCommunicationProcessor("island-0")

        processor = GlobalAsynchronousProcessor(
            initial_iter=1,
            check_interval=5,
            communication_processor=communication,
        )

        processor.migration_control(
            actual_iter=1,
            population={},
            local_best=None,
            insert_arrival_particle=MagicMock(),
            departure_particle=MagicMock(),
        )

        assert communication.outgoing_queue.empty()

    @pytest.mark.parametrize(
        "local_best",
        [
            "",
            "not-json",
            "{",
            "[]",
            "null",
            '{"identifier": "particle"}',
            '{"identifier": "particle", "fitness": "1.0"}',
        ],
    )
    def test_invalid_local_best_is_ignored(
        self,
        local_best: str,
    ) -> None:
        communication = MockCommunicationProcessor("island-0")

        processor = GlobalAsynchronousProcessor(
            initial_iter=1,
            check_interval=1,
            communication_processor=communication,
        )

        processor.migration_control(
            actual_iter=1,
            population={},
            local_best=local_best,
            insert_arrival_particle=MagicMock(),
            departure_particle=MagicMock(),
        )

        assert communication.outgoing_queue.empty()

    def test_local_best_without_identifier_can_be_published(self) -> None:
        communication = MockCommunicationProcessor("island-0")

        processor = GlobalAsynchronousProcessor(
            initial_iter=1,
            check_interval=1,
            communication_processor=communication,
        )

        local_best = json.dumps({"fitness": 1.0})

        processor.migration_control(
            actual_iter=1,
            population={},
            local_best=local_best,
            insert_arrival_particle=MagicMock(),
            departure_particle=MagicMock(),
        )

        assert communication.outgoing_queue.qsize() == 1

        payload = json.loads(communication.outgoing_queue.get_nowait())

        assert payload["type"] == "global_best_update"
        assert payload["particle"]["fitness"] == 1.0

    def test_publishes_only_when_fitness_improves(self) -> None:
        communication = MockCommunicationProcessor("island-0")

        processor = GlobalAsynchronousProcessor(
            initial_iter=1,
            check_interval=1,
            communication_processor=communication,
        )

        for actual_iter, fitness in enumerate(
            [10.0, 10.0, 12.0, 8.0, 8.0],
            start=1,
        ):
            processor.migration_control(
                actual_iter=actual_iter,
                population={},
                local_best=serialize_particle(fitness=fitness),
                insert_arrival_particle=MagicMock(),
                departure_particle=MagicMock(),
            )

        messages = []

        while not communication.outgoing_queue.empty():
            messages.append(json.loads(communication.outgoing_queue.get_nowait()))

        assert len(messages) == 2
        assert messages[0]["particle"]["fitness"] == 10.0
        assert messages[1]["particle"]["fitness"] == 8.0

    def test_equal_fitness_is_not_published_again(self) -> None:
        communication = MockCommunicationProcessor("island-0")

        processor = GlobalAsynchronousProcessor(
            initial_iter=1,
            check_interval=1,
            communication_processor=communication,
        )

        local_best = serialize_particle(fitness=5.0)

        processor.migration_control(
            actual_iter=1,
            population={},
            local_best=local_best,
            insert_arrival_particle=MagicMock(),
            departure_particle=MagicMock(),
        )

        processor.migration_control(
            actual_iter=2,
            population={},
            local_best=local_best,
            insert_arrival_particle=MagicMock(),
            departure_particle=MagicMock(),
        )

        assert communication.outgoing_queue.qsize() == 1

    def test_worse_fitness_is_not_published(self) -> None:
        communication = MockCommunicationProcessor("island-0")

        processor = GlobalAsynchronousProcessor(
            initial_iter=1,
            check_interval=1,
            communication_processor=communication,
        )

        processor.migration_control(
            actual_iter=1,
            population={},
            local_best=serialize_particle(fitness=5.0),
            insert_arrival_particle=MagicMock(),
            departure_particle=MagicMock(),
        )

        processor.migration_control(
            actual_iter=2,
            population={},
            local_best=serialize_particle(fitness=10.0),
            insert_arrival_particle=MagicMock(),
            departure_particle=MagicMock(),
        )

        assert communication.outgoing_queue.qsize() == 1

    def test_improved_fitness_is_published(self) -> None:
        communication = MockCommunicationProcessor("island-0")

        processor = GlobalAsynchronousProcessor(
            initial_iter=1,
            check_interval=1,
            communication_processor=communication,
        )

        processor.migration_control(
            actual_iter=1,
            population={},
            local_best=serialize_particle(fitness=10.0),
            insert_arrival_particle=MagicMock(),
            departure_particle=MagicMock(),
        )

        processor.migration_control(
            actual_iter=2,
            population={},
            local_best=serialize_particle(fitness=5.0),
            insert_arrival_particle=MagicMock(),
            departure_particle=MagicMock(),
        )

        assert communication.outgoing_queue.qsize() == 2

        first = json.loads(communication.outgoing_queue.get_nowait())
        second = json.loads(communication.outgoing_queue.get_nowait())

        assert first["particle"]["fitness"] == 10.0
        assert second["particle"]["fitness"] == 5.0


class TestGlobalAsynchronousProcessorReception:
    """Tests for asynchronous reception of remote global-best messages."""

    def test_receives_remote_particle(self) -> None:
        communication = MockCommunicationProcessor("island-0")

        processor = GlobalAsynchronousProcessor(
            initial_iter=10,
            check_interval=5,
            communication_processor=communication,
        )

        communication.messages.put(
            serialize_update(
                identifier="remote-1",
                fitness=2.0,
                position=[3.0, 4.0],
            )
        )

        population = {
            "local-1": 10.0,
            "local-2": 20.0,
            "local-3": 50.0,
        }

        inserted = []
        departed = []

        processor.migration_control(
            actual_iter=1,
            population=population,  # type: ignore
            local_best=None,
            insert_arrival_particle=inserted.append,
            departure_particle=departed.append,
        )

        assert inserted == [
            {
                "identifier": "local-3",
                "fitness": 2.0,
                "position": [3.0, 4.0],
            }
        ]

        assert departed == ["local-3"]

    def test_replaces_worst_particle(self) -> None:
        communication = MockCommunicationProcessor("island-0")

        processor = GlobalAsynchronousProcessor(
            initial_iter=10,
            check_interval=5,
            communication_processor=communication,
        )

        communication.messages.put(
            serialize_update(
                identifier="remote-1",
                fitness=2.0,
            )
        )

        population = {
            "local-1": 10.0,
            "local-2": 20.0,
            "local-3": 50.0,
        }

        inserted = []
        departed = []

        processor.migration_control(
            actual_iter=1,
            population=population,  # type: ignore
            local_best=None,
            insert_arrival_particle=inserted.append,
            departure_particle=departed.append,
        )

        assert departed == ["local-3"]
        assert len(inserted) == 1
        assert inserted[0]["identifier"] == "local-3"
        assert inserted[0]["fitness"] == 2.0

    def test_preserves_replaced_particle_identifier(self) -> None:
        communication = MockCommunicationProcessor("island-0")

        processor = GlobalAsynchronousProcessor(
            initial_iter=10,
            check_interval=5,
            communication_processor=communication,
        )

        communication.messages.put(
            serialize_update(
                identifier="remote-1",
                fitness=2.0,
            )
        )

        population = {
            "particle:0": 10.0,
            "particle:1": 20.0,
            "particle:2": 50.0,
        }

        inserted = []
        departed = []

        processor.migration_control(
            actual_iter=1,
            population=population,  # type: ignore
            local_best=None,
            insert_arrival_particle=inserted.append,
            departure_particle=departed.append,
        )

        assert departed == ["particle:2"]
        assert inserted[0]["identifier"] == "particle:2"
        assert inserted[0]["fitness"] == 2.0

    def test_consumes_all_pending_messages(self) -> None:
        communication = MockCommunicationProcessor("island-0")

        processor = GlobalAsynchronousProcessor(
            initial_iter=10,
            check_interval=5,
            communication_processor=communication,
        )

        communication.messages.put(
            serialize_update(
                identifier="remote-1",
                fitness=10.0,
            )
        )
        communication.messages.put(
            serialize_update(
                identifier="remote-2",
                fitness=8.0,
            )
        )
        communication.messages.put(
            serialize_update(
                identifier="remote-3",
                fitness=5.0,
            )
        )

        population = {
            "local-1": 10.0,
            "local-2": 20.0,
            "local-3": 50.0,
        }

        inserted = []
        departed = []

        processor.migration_control(
            actual_iter=1,
            population=population,  # type: ignore
            local_best=None,
            insert_arrival_particle=inserted.append,
            departure_particle=departed.append,
        )

        assert [p["identifier"] for p in inserted] == [
            "local-3",
            "local-3",
            "local-3",
        ]

        assert [p["fitness"] for p in inserted] == [
            10.0,
            8.0,
            5.0,
        ]

        assert departed == [
            "local-3",
            "local-3",
            "local-3",
        ]

        assert communication.messages.empty()

    def test_incoming_particle_is_ignored_when_population_is_empty(
        self,
    ) -> None:
        communication = MockCommunicationProcessor("island-0")

        processor = GlobalAsynchronousProcessor(
            initial_iter=1,
            check_interval=1,
            communication_processor=communication,
        )

        communication.messages.put(
            serialize_update(
                identifier="remote-1",
                fitness=2.0,
            )
        )

        inserted = MagicMock()
        departure_particle = MagicMock()

        processor.migration_control(
            actual_iter=1,
            population={},
            local_best=None,
            insert_arrival_particle=inserted,
            departure_particle=departure_particle,
        )

        inserted.assert_not_called()
        departure_particle.assert_not_called()

    @pytest.mark.parametrize(
        "message",
        [
            "",
            "invalid-json",
            "{}",
            '{"type": "other"}',
            '{"type": "global_best_update"}',
            '{"type": "global_best_update", "particle": null}',
            '{"type": "global_best_update", "particle": []}',
            ('{"type": "global_best_update", "particle": {"fitness": 1.0}}'),
        ],
    )
    def test_invalid_incoming_messages_are_ignored(
        self,
        message: str,
    ) -> None:
        communication = MockCommunicationProcessor("island-0")

        processor = GlobalAsynchronousProcessor(
            initial_iter=1,
            check_interval=1,
            communication_processor=communication,
        )

        communication.messages.put(message)

        population = {
            "local-1": 10.0,
            "local-2": 20.0,
        }

        inserted = MagicMock()
        departure_particle = MagicMock()

        processor.migration_control(
            actual_iter=1,
            population=population,  # type: ignore
            local_best=None,
            insert_arrival_particle=inserted,
            departure_particle=departure_particle,
        )

        inserted.assert_not_called()
        departure_particle.assert_not_called()

    def test_receiving_message_does_not_change_synchronization_schedule(
        self,
    ) -> None:
        communication = MockCommunicationProcessor("island-0")

        processor = GlobalAsynchronousProcessor(
            initial_iter=10,
            check_interval=5,
            communication_processor=communication,
        )

        communication.messages.put(
            serialize_update(
                identifier="remote-1",
                fitness=1.0,
            )
        )

        population = {
            "local-1": 10.0,
            "local-2": 20.0,
        }

        processor.migration_control(
            actual_iter=3,
            population=population,  # type: ignore
            local_best=None,
            insert_arrival_particle=MagicMock(),
            departure_particle=MagicMock(),
        )

        assert processor._synchronization_iter == 10


class TestGlobalAsynchronousProcessorNoBarrier:
    """Tests that asynchronous migration does not introduce a barrier."""

    def test_migration_control_does_not_wait_for_messages(self) -> None:
        communication = MockCommunicationProcessor("island-0")

        processor = GlobalAsynchronousProcessor(
            initial_iter=10,
            check_interval=5,
            communication_processor=communication,
        )

        processor.migration_control(
            actual_iter=10,
            population={},
            local_best=None,
            insert_arrival_particle=MagicMock(),
            departure_particle=MagicMock(),
        )

        assert processor._synchronization_iter == 15

    def test_empty_message_queue_does_not_block(self) -> None:
        communication = MockCommunicationProcessor("island-0")

        processor = GlobalAsynchronousProcessor(
            initial_iter=10,
            check_interval=5,
            communication_processor=communication,
        )

        processor.migration_control(
            actual_iter=10,
            population={},
            local_best=None,
            insert_arrival_particle=MagicMock(),
            departure_particle=MagicMock(),
        )

        assert communication.messages.empty()


# ============================================================================
# GlobalAsynchronous driver
# ============================================================================


class TestGlobalAsynchronousInitialization:
    """Tests for GlobalAsynchronous driver initialization."""

    def test_initializes_with_defaults(self) -> None:
        migration = GlobalAsynchronous()

        assert migration._migration_processor_init_kargs == {
            "initial_iter": 1,
            "check_interval": 1,
        }

        assert migration._global_best_fitness is None
        assert migration._thread is None

    def test_initializes_with_custom_parameters(self) -> None:
        migration = GlobalAsynchronous(
            initial_iter=20,
            check_interval=10,
        )

        assert migration._migration_processor_init_kargs == {
            "initial_iter": 20,
            "check_interval": 10,
        }

    @pytest.mark.parametrize(
        "initial_iter",
        [-1, -5],
    )
    def test_rejects_negative_initial_iter(
        self,
        initial_iter: int,
    ) -> None:
        with pytest.raises(ValueError, match="initial_iter"):
            GlobalAsynchronous(
                initial_iter=initial_iter,
                check_interval=5,
            )

    @pytest.mark.parametrize(
        "check_interval",
        [0, -1, -5],
    )
    def test_rejects_non_positive_check_interval(
        self,
        check_interval: int,
    ) -> None:
        with pytest.raises(ValueError, match="check_interval"):
            GlobalAsynchronous(
                initial_iter=1,
                check_interval=check_interval,
            )


class TestGlobalAsynchronousContext:
    """Tests for communication context initialization."""

    def test_initialize_context_stores_communication_configuration(
        self,
    ) -> None:
        migration = GlobalAsynchronous(
            initial_iter=10,
            check_interval=5,
        )

        driver = MockCommunicationDriver(
            ["island-0", "island-1"],
        )

        migration.initialize_context(
            communication_driver=driver,  # type: ignore
            communication_processor_class=MockCommunicationProcessor,
            communication_processor_kargs={
                "host": "127.0.0.1",
                "port": 5000,
            },
        )

        assert migration._communication_driver is driver
        assert migration._communication_processor_class is MockCommunicationProcessor

        assert migration._migration_processor_init_kargs == {
            "initial_iter": 10,
            "check_interval": 5,
            "communication": {
                "host": "127.0.0.1",
                "port": 5000,
            },
        }

    def test_initialize_context_copies_communication_configuration(
        self,
    ) -> None:
        migration = GlobalAsynchronous()

        driver = MockCommunicationDriver(["island-0"])
        communication_kargs = {
            "host": "127.0.0.1",
            "port": 5000,
        }

        migration.initialize_context(
            communication_driver=driver,  # type: ignore
            communication_processor_class=MockCommunicationProcessor,
            communication_processor_kargs=communication_kargs,
        )

        communication_kargs["port"] = 9999

        assert (
            migration._migration_processor_init_kargs["communication"]["port"] == 5000
        )

    def test_create_processor_module(self) -> None:
        migration = GlobalAsynchronous(
            initial_iter=10,
            check_interval=5,
        )

        driver = MockCommunicationDriver(["island-0"])

        migration.initialize_context(
            communication_driver=driver,  # type: ignore
            communication_processor_class=MockCommunicationProcessor,
            communication_processor_kargs={
                "host": "127.0.0.1",
            },
        )

        processor = migration.create_processor_module("island-0")

        assert isinstance(
            processor,
            GlobalAsynchronousProcessor,
        )

        assert processor._initial_iter == 10
        assert processor._check_interval == 5
        assert processor._communication_processor.identification == "island-0"  # type: ignore


class TestGlobalAsynchronousRouting:
    """Tests for global-best routing through the driver."""

    def test_routes_new_global_best_to_all_other_islands(self) -> None:
        migration = GlobalAsynchronous(
            initial_iter=1,
            check_interval=5,
        )

        driver = MockCommunicationDriver(
            ["island-0", "island-1", "island-2"],
        )

        migration.initialize_context(
            communication_driver=driver,  # type: ignore
            communication_processor_class=MockCommunicationProcessor,
            communication_processor_kargs={},
        )

        message = serialize_update(
            identifier="particle-1",
            fitness=5.0,
        )

        migration._route_message(
            source_id="island-0",
            message=message,
        )

        assert driver.outgoing_queues["island-0"].empty()

        for island_id in ("island-1", "island-2"):
            assert driver.outgoing_queues[island_id].qsize() == 1

            payload = json.loads(driver.outgoing_queues[island_id].get_nowait())

            assert payload["type"] == "global_best_update"
            assert payload["particle"]["identifier"] == "particle-1"
            assert payload["particle"]["fitness"] == 5.0

    def test_does_not_send_global_best_back_to_source(self) -> None:
        migration = GlobalAsynchronous()

        driver = MockCommunicationDriver(
            ["island-0", "island-1"],
        )

        migration.initialize_context(
            communication_driver=driver,  # type: ignore
            communication_processor_class=MockCommunicationProcessor,
            communication_processor_kargs={},
        )

        migration._route_message(
            source_id="island-0",
            message=serialize_update(
                identifier="particle-1",
                fitness=1.0,
            ),
        )

        assert driver.outgoing_queues["island-0"].empty()
        assert driver.outgoing_queues["island-1"].qsize() == 1

    def test_ignores_worse_global_best(self) -> None:
        migration = GlobalAsynchronous()

        driver = MockCommunicationDriver(
            ["island-0", "island-1", "island-2"],
        )

        migration.initialize_context(
            communication_driver=driver,  # type: ignore
            communication_processor_class=MockCommunicationProcessor,
            communication_processor_kargs={},
        )

        migration._route_message(
            source_id="island-0",
            message=serialize_update(
                identifier="particle-1",
                fitness=5.0,
            ),
        )

        migration._route_message(
            source_id="island-1",
            message=serialize_update(
                identifier="particle-2",
                fitness=10.0,
            ),
        )

        assert migration._global_best_fitness == 5.0
        assert driver.outgoing_queues["island-2"].qsize() == 1

    def test_ignores_equal_global_best(self) -> None:
        migration = GlobalAsynchronous()

        driver = MockCommunicationDriver(
            ["island-0", "island-1", "island-2"],
        )

        migration.initialize_context(
            communication_driver=driver,  # type: ignore
            communication_processor_class=MockCommunicationProcessor,
            communication_processor_kargs={},
        )

        migration._route_message(
            source_id="island-0",
            message=serialize_update(
                identifier="particle-1",
                fitness=5.0,
            ),
        )

        driver.outgoing_queues["island-1"].get_nowait()
        driver.outgoing_queues["island-2"].get_nowait()

        migration._route_message(
            source_id="island-1",
            message=serialize_update(
                identifier="particle-2",
                fitness=5.0,
            ),
        )

        assert driver.outgoing_queues["island-0"].empty()
        assert driver.outgoing_queues["island-1"].empty()
        assert driver.outgoing_queues["island-2"].empty()

    def test_better_global_best_replaces_previous_global_best(
        self,
    ) -> None:
        migration = GlobalAsynchronous()

        driver = MockCommunicationDriver(
            ["island-0", "island-1", "island-2"],
        )

        migration.initialize_context(
            communication_driver=driver,  # type: ignore
            communication_processor_class=MockCommunicationProcessor,
            communication_processor_kargs={},
        )

        migration._route_message(
            source_id="island-0",
            message=serialize_update(
                identifier="particle-1",
                fitness=10.0,
            ),
        )

        for island_id in ("island-1", "island-2"):
            driver.outgoing_queues[island_id].get_nowait()

        migration._route_message(
            source_id="island-1",
            message=serialize_update(
                identifier="particle-2",
                fitness=5.0,
            ),
        )

        assert migration._global_best_fitness == 5.0

        for island_id in ("island-0", "island-2"):
            assert driver.outgoing_queues[island_id].qsize() == 1

        assert driver.outgoing_queues["island-1"].empty()

    @pytest.mark.parametrize(
        "message",
        [
            "",
            "invalid-json",
            "{}",
            '{"type": "other"}',
            '{"type": "global_best_update"}',
            '{"type": "global_best_update", "particle": null}',
            '{"type": "global_best_update", "particle": []}',
            ('{"type": "global_best_update", "particle": {"identifier": "p"}}'),
            (
                '{"type": "global_best_update", '
                '"particle": {"identifier": "p", "fitness": "5"}}'
            ),
        ],
    )
    def test_invalid_driver_messages_are_ignored(
        self,
        message: str,
    ) -> None:
        migration = GlobalAsynchronous()

        driver = MockCommunicationDriver(
            ["island-0", "island-1"],
        )

        migration.initialize_context(
            communication_driver=driver,  # type: ignore
            communication_processor_class=MockCommunicationProcessor,
            communication_processor_kargs={},
        )

        migration._route_message(
            source_id="island-0",
            message=message,
        )

        assert migration._global_best_fitness is None

        for queue in driver.outgoing_queues.values():
            assert queue.empty()


class TestGlobalAsynchronousProcessing:
    """Tests for processing incoming driver messages."""

    def test_process_incoming_processes_all_islands(self) -> None:
        migration = GlobalAsynchronous()

        driver = MockCommunicationDriver(
            ["island-0", "island-1", "island-2"],
        )

        migration.initialize_context(
            communication_driver=driver,  # type: ignore
            communication_processor_class=MockCommunicationProcessor,
            communication_processor_kargs={},
        )

        driver.incoming_queues["island-0"].put(
            serialize_update(
                identifier="particle-0",
                fitness=10.0,
            )
        )

        driver.incoming_queues["island-1"].put(
            serialize_update(
                identifier="particle-1",
                fitness=5.0,
            )
        )

        migration._process_incoming()

        assert migration._global_best_fitness == 5.0

        assert driver.incoming_queues["island-0"].empty()
        assert driver.incoming_queues["island-1"].empty()
        assert driver.incoming_queues["island-2"].empty()

        assert driver.outgoing_queues["island-0"].qsize() == 1
        assert driver.outgoing_queues["island-1"].qsize() == 1
        assert driver.outgoing_queues["island-2"].qsize() == 2

        island_0_message = json.loads(driver.outgoing_queues["island-0"].get_nowait())

        island_1_message = json.loads(driver.outgoing_queues["island-1"].get_nowait())

        island_2_first_message = json.loads(
            driver.outgoing_queues["island-2"].get_nowait()
        )

        island_2_second_message = json.loads(
            driver.outgoing_queues["island-2"].get_nowait()
        )

        assert island_0_message["particle"]["fitness"] == 5.0
        assert island_1_message["particle"]["fitness"] == 10.0

        assert island_2_first_message["particle"]["fitness"] == 10.0
        assert island_2_second_message["particle"]["fitness"] == 5.0

    def test_process_incoming_does_not_block_on_empty_queues(
        self,
    ) -> None:
        migration = GlobalAsynchronous()

        driver = MockCommunicationDriver(
            ["island-0", "island-1"],
        )

        migration.initialize_context(
            communication_driver=driver,  # type: ignore
            communication_processor_class=MockCommunicationProcessor,
            communication_processor_kargs={},
        )

        migration._process_incoming()

        assert migration._global_best_fitness is None

    def test_processes_messages_in_queue_order(self) -> None:
        migration = GlobalAsynchronous()

        driver = MockCommunicationDriver(
            ["island-0", "island-1"],
        )

        migration.initialize_context(
            communication_driver=driver,  # type: ignore
            communication_processor_class=MockCommunicationProcessor,
            communication_processor_kargs={},
        )

        driver.incoming_queues["island-0"].put(
            serialize_update(
                identifier="particle-1",
                fitness=10.0,
            )
        )

        driver.incoming_queues["island-0"].put(
            serialize_update(
                identifier="particle-2",
                fitness=5.0,
            )
        )

        migration._process_incoming()

        assert migration._global_best_fitness == 5.0

        first_message = driver.outgoing_queues["island-1"].get_nowait()
        second_message = driver.outgoing_queues["island-1"].get_nowait()

        first_payload = json.loads(first_message)
        second_payload = json.loads(second_message)

        assert first_payload["particle"]["identifier"] == "particle-1"
        assert first_payload["particle"]["fitness"] == 10.0

        assert second_payload["particle"]["identifier"] == "particle-2"
        assert second_payload["particle"]["fitness"] == 5.0

        assert driver.outgoing_queues["island-1"].empty()


class TestGlobalAsynchronousLifecycle:
    """Tests for driver start/stop behavior."""

    def test_start_starts_communication_driver(self) -> None:
        migration = GlobalAsynchronous()

        driver = MockCommunicationDriver(["island-0"])

        migration.initialize_context(
            communication_driver=driver,  # type: ignore
            communication_processor_class=MockCommunicationProcessor,
            communication_processor_kargs={},
        )

        migration.start()

        try:
            assert driver.start_calls == 1
            assert migration._running.is_set()
            assert migration._thread is not None
            assert migration._thread.is_alive()
        finally:
            migration.stop()

    def test_stop_stops_driver_and_router(self) -> None:
        migration = GlobalAsynchronous()

        driver = MockCommunicationDriver(["island-0"])

        migration.initialize_context(
            communication_driver=driver,  # type: ignore
            communication_processor_class=MockCommunicationProcessor,
            communication_processor_kargs={},
        )

        migration.start()
        migration.stop()

        assert driver.start_calls == 1
        assert driver.stop_calls == 1
        assert not migration._running.is_set()
        assert migration._thread is None

    def test_start_cannot_be_called_twice_while_running(self) -> None:
        migration = GlobalAsynchronous()

        driver = MockCommunicationDriver(["island-0"])

        migration.initialize_context(
            communication_driver=driver,  # type: ignore
            communication_processor_class=MockCommunicationProcessor,
            communication_processor_kargs={},
        )

        migration.start()

        try:
            with pytest.raises(
                RuntimeError,
                match="already running",
            ):
                migration.start()
        finally:
            migration.stop()

    def test_start_resets_global_best(self) -> None:
        migration = GlobalAsynchronous()

        driver = MockCommunicationDriver(["island-0"])

        migration.initialize_context(
            communication_driver=driver,  # type: ignore
            communication_processor_class=MockCommunicationProcessor,
            communication_processor_kargs={},
        )

        migration._global_best_fitness = 5.0

        migration.start()

        try:
            assert migration._global_best_fitness is None
        finally:
            migration.stop()

    def test_stop_without_start_is_safe(self) -> None:
        migration = GlobalAsynchronous()

        driver = MockCommunicationDriver(["island-0"])

        migration.initialize_context(
            communication_driver=driver,  # type: ignore
            communication_processor_class=MockCommunicationProcessor,
            communication_processor_kargs={},
        )

        migration.stop()

        assert driver.stop_calls == 1
