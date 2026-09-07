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
from tyrannis.core.backend_communication import CommunicationProcessorBase
from tyrannis.core.signals import LocalEvent


class MockCommunicationProcessor(CommunicationProcessorBase):
    def __init__(self, identification: str, **kwargs: object) -> None:
        self.identification = identification
        self._messages: Queue[str] = Queue()
        self._outgoing_queue: Queue[str] = Queue()
        self.start_signal = None
        self.stop_calls = 0
        self.message_signal = None

    @property
    def messages(self) -> Queue[str]:
        return self._messages

    @property
    def outgoing_queue(self) -> Queue[str]:
        return self._outgoing_queue

    def start(self, stop_signal: LocalEvent) -> None:
        self.start_signal = stop_signal

    def stop(self) -> None:
        self.stop_calls += 1

    def set_message_signal(
        self,
        message_signal: LocalEvent | None,
    ) -> None:
        self.message_signal = message_signal


class MockCommunicationDriver:
    def __init__(self, island_ids: list[str]) -> None:
        self.incoming_queues = {island_id: Queue[str]() for island_id in island_ids}
        self.outgoing_queues = {island_id: Queue[str]() for island_id in island_ids}
        self.start_calls = 0
        self.stop_calls = 0

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1


def particle(
    identifier: str = "particle-0",
    fitness: float = 10.0,
    **kwargs: object,
) -> dict:
    return {
        "identifier": identifier,
        "fitness": fitness,
        **kwargs,
    }


def serialized_particle(
    identifier: str = "particle-0",
    fitness: float = 10.0,
    **kwargs: object,
) -> str:
    return json.dumps(
        particle(
            identifier=identifier,
            fitness=fitness,
            **kwargs,
        )
    )


def update_message(
    identifier: str = "particle-0",
    fitness: float = 10.0,
    **kwargs: object,
) -> str:
    return json.dumps(
        {
            "type": GlobalAsynchronousProcessor.MESSAGE_TYPE,
            "particle": particle(
                identifier=identifier,
                fitness=fitness,
                **kwargs,
            ),
        }
    )


def make_processor(
    initial_iter: int = 1,
    check_interval: int = 1,
) -> tuple[
    GlobalAsynchronousProcessor,
    MockCommunicationProcessor,
    LocalEvent,
]:
    communication = MockCommunicationProcessor("island-0")

    processor = GlobalAsynchronousProcessor(
        initial_iter=initial_iter,
        check_interval=check_interval,
        communication_processor=communication,
    )

    signal = LocalEvent()
    processor.initialize_loop_context(signal)

    return processor, communication, signal


def make_migration(
    island_ids: list[str] | None = None,
) -> tuple[GlobalAsynchronous, MockCommunicationDriver]:
    migration = GlobalAsynchronous()

    driver = MockCommunicationDriver(
        island_ids or ["island-0", "island-1"],
    )

    migration.initialize_context(
        communication_driver=driver,  # type: ignore
        communication_processor_class=MockCommunicationProcessor,
        communication_processor_kargs={},
    )

    return migration, driver


class TestGlobalAsynchronousProcessor:
    def test_rejects_invalid_configuration(self) -> None:
        communication = MockCommunicationProcessor("island-0")

        with pytest.raises(ValueError, match="initial_iter"):
            GlobalAsynchronousProcessor(
                initial_iter=-1,
                check_interval=1,
                communication_processor=communication,
            )

        with pytest.raises(ValueError, match="check_interval"):
            GlobalAsynchronousProcessor(
                initial_iter=1,
                check_interval=0,
                communication_processor=communication,
            )

    def test_initializes_synchronization_state(self) -> None:
        communication = MockCommunicationProcessor("island-0")

        processor = GlobalAsynchronousProcessor(
            initial_iter=10,
            check_interval=5,
            communication_processor=communication,
        )

        assert processor._initial_iter == 10
        assert processor._check_interval == 5
        assert processor._synchronization_iter == 10

    def test_requires_loop_context(self) -> None:
        communication = MockCommunicationProcessor("island-0")

        processor = GlobalAsynchronousProcessor(
            initial_iter=1,
            check_interval=1,
            communication_processor=communication,
        )

        with pytest.raises(
            RuntimeError,
            match="loop context",
        ):
            processor.migration_control(
                actual_iter=1,
                population={},
                local_best=None,
                insert_arrival_particle=MagicMock(),
                departure_particle=MagicMock(),
            )

    def test_initializes_and_finalizes_loop_context(self) -> None:
        processor, communication, signal = make_processor()

        assert communication.message_signal is signal

        processor.finalize_loop_context()

        assert communication.message_signal is None

    def test_start_and_stop_delegate_to_communication(self) -> None:
        processor, communication, _ = make_processor()
        stop_signal = Event()

        processor.start(stop_signal)  # type: ignore[arg-type]
        processor.stop()

        assert communication.start_signal is stop_signal
        assert communication.stop_calls == 1

    def test_publishes_local_best_at_synchronization_point(self) -> None:
        processor, communication, _ = make_processor(
            initial_iter=5,
            check_interval=3,
        )

        processor.migration_control(
            actual_iter=5,
            population={},
            local_best=serialized_particle(
                identifier="particle-1",
                fitness=4.0,
                position=[1.0, 2.0],
            ),
            insert_arrival_particle=MagicMock(),
            departure_particle=MagicMock(),
        )

        message = json.loads(communication.outgoing_queue.get_nowait())

        assert message == {
            "type": "global_best_update",
            "particle": {
                "identifier": "particle-1",
                "fitness": 4.0,
                "position": [1.0, 2.0],
            },
        }

        assert processor._synchronization_iter == 8

    def test_publishes_only_improved_local_best(self) -> None:
        processor, communication, _ = make_processor(
            initial_iter=1,
            check_interval=1,
        )

        callbacks = {
            "insert_arrival_particle": MagicMock(),
            "departure_particle": MagicMock(),
        }

        for actual_iter, fitness in enumerate(
            (10.0, 10.0, 12.0, 5.0),
            start=1,
        ):
            processor.migration_control(
                actual_iter=actual_iter,
                population={},
                local_best=serialized_particle(fitness=fitness),
                **callbacks,
            )

        messages = []

        while not communication.outgoing_queue.empty():
            messages.append(json.loads(communication.outgoing_queue.get_nowait()))

        assert [message["particle"]["fitness"] for message in messages] == [
            10.0,
            5.0,
        ]

    def test_consumes_message_and_replaces_worst_particle(self) -> None:
        processor, communication, _ = make_processor(
            initial_iter=10,
            check_interval=5,
        )

        communication.messages.put(
            update_message(
                identifier="remote",
                fitness=2.0,
                position=[1.0, 2.0],
            )
        )

        insert = MagicMock()
        departure = MagicMock()

        processor.migration_control(
            actual_iter=1,
            population={
                "particle-0": 10.0,
                "particle-1": 20.0,
                "particle-2": 50.0,
            },
            local_best=None,
            insert_arrival_particle=insert,
            departure_particle=departure,
        )

        departure.assert_called_once_with("particle-2")
        insert.assert_called_once_with(
            {
                "identifier": "particle-2",
                "fitness": 2.0,
                "position": [1.0, 2.0],
            }
        )

    def test_ignores_invalid_message(self) -> None:
        processor, communication, _ = make_processor()

        communication.messages.put("not-json")

        insert = MagicMock()
        departure = MagicMock()

        processor.migration_control(
            actual_iter=1,
            population={"particle-0": 10.0},
            local_best=None,
            insert_arrival_particle=insert,
            departure_particle=departure,
        )

        insert.assert_not_called()
        departure.assert_not_called()

    def test_ignores_message_when_population_is_empty(self) -> None:
        processor, communication, _ = make_processor()

        communication.messages.put(update_message(fitness=1.0))

        insert = MagicMock()
        departure = MagicMock()

        processor.migration_control(
            actual_iter=1,
            population={},
            local_best=None,
            insert_arrival_particle=insert,
            departure_particle=departure,
        )

        insert.assert_not_called()
        departure.assert_not_called()

    def test_iteration_jump_moves_to_next_future_synchronization_point(
        self,
    ) -> None:
        processor, _, _ = make_processor(
            initial_iter=10,
            check_interval=5,
        )

        processor.migration_control(
            actual_iter=27,
            population={},
            local_best=None,
            insert_arrival_particle=MagicMock(),
            departure_particle=MagicMock(),
        )

        assert processor._synchronization_iter == 30


class TestGlobalAsynchronous:
    def test_rejects_invalid_configuration(self) -> None:
        with pytest.raises(ValueError, match="initial_iter"):
            GlobalAsynchronous(initial_iter=0)

        with pytest.raises(ValueError, match="check_interval"):
            GlobalAsynchronous(check_interval=0)

    def test_initialize_context_and_create_processor(self) -> None:
        migration, driver = make_migration(["island-0"])

        processor = migration.create_processor_module("island-0")

        assert migration._communication_driver is driver
        assert isinstance(
            processor,
            GlobalAsynchronousProcessor,
        )

    def test_routes_new_global_best_to_other_islands(self) -> None:
        migration, driver = make_migration(
            ["island-0", "island-1", "island-2"],
        )

        migration._route_message(
            source_id="island-0",
            message=update_message(
                identifier="particle-1",
                fitness=5.0,
            ),
        )

        assert driver.outgoing_queues["island-0"].empty()

        for island_id in ("island-1", "island-2"):
            message = json.loads(driver.outgoing_queues[island_id].get_nowait())

            assert message["type"] == "global_best_update"
            assert message["particle"]["fitness"] == 5.0

    def test_accepts_only_new_global_best(self) -> None:
        migration, driver = make_migration(
            ["island-0", "island-1"],
        )

        migration._route_message(
            source_id="island-0",
            message=update_message(fitness=5.0),
        )

        driver.outgoing_queues["island-1"].get_nowait()

        migration._route_message(
            source_id="island-1",
            message=update_message(fitness=10.0),
        )

        migration._route_message(
            source_id="island-1",
            message=update_message(fitness=5.0),
        )

        assert migration._global_best_fitness == 5.0
        assert driver.outgoing_queues["island-0"].empty()
        assert driver.outgoing_queues["island-1"].empty()

    def test_ignores_invalid_routing_message(self) -> None:
        migration, driver = make_migration()

        migration._route_message(
            source_id="island-0",
            message="not-json",
        )

        assert migration._global_best_fitness is None
        assert driver.outgoing_queues["island-1"].empty()

    def test_process_incoming_drains_all_island_queues(self) -> None:
        migration, driver = make_migration(
            ["island-0", "island-1"],
        )

        driver.incoming_queues["island-0"].put(update_message(fitness=10.0))
        driver.incoming_queues["island-1"].put(update_message(fitness=5.0))

        migration._process_incoming()

        assert migration._global_best_fitness == 5.0
        assert driver.incoming_queues["island-0"].empty()
        assert driver.incoming_queues["island-1"].empty()

    def test_start_and_stop_control_driver_and_router(self) -> None:
        migration, driver = make_migration()

        migration.start()

        assert driver.start_calls == 1
        assert migration._running.is_set()

        migration.stop()

        assert driver.stop_calls == 1
        assert not migration._running.is_set()
        assert migration._thread is None

    def test_start_cannot_be_called_twice(self) -> None:
        migration, _ = make_migration()

        migration.start()

        try:
            with pytest.raises(
                RuntimeError,
                match="already running",
            ):
                migration.start()
        finally:
            migration.stop()
