"""Contracts specific to the migration strategy for isolated islands."""

from unittest.mock import Mock, patch

import numpy as np

from tyrannis.backend.distributed.communication.no_communication import (
    NoCommunicationDriver,
    NoCommunicationProcessor,
)
from tyrannis.core.backend_communication import CommunicationProcessorBase
from tyrannis.core.results import HistoryConfig
from tyrannis.core.signals import LocalEvent
from tyrannis.migration import island_isolation
from tyrannis.migration.island_isolation import (
    IslandIsolation,
    IslandIsolationProcessor,
)


def test_processor_start_and_stop_delegate_to_communication() -> None:
    communication = NoCommunicationProcessor("island:a")
    processor = IslandIsolationProcessor(1, communication, None)

    with (
        patch.object(communication, "start") as start,
        patch.object(communication, "stop") as stop,
    ):
        processor.start()
        processor.stop()

    start.assert_called_once_with()
    stop.assert_called_once_with()


def test_processor_loop_and_controls_leave_migration_inert() -> None:
    communication = NoCommunicationProcessor("island:a")
    processor = IslandIsolationProcessor(1, communication, None)
    signal = LocalEvent()
    population = {"particle": np.float64(1.0)}
    insert = Mock()
    depart = Mock()

    with patch.object(communication, "set_message_signal") as set_message_signal:
        processor.initialize_loop_context(signal)
        processor.migration_control(1, population, "particle", insert, depart)
        processor.synchronization_control(1, insert, depart)
        processor.finalize_loop_context()

    assert population == {"particle": np.float64(1.0)}
    assert not signal.is_set()
    insert.assert_not_called()
    depart.assert_not_called()
    set_message_signal.assert_not_called()


def test_driver_uses_isolated_communication_and_delegates_lifecycle() -> None:
    supplied = NoCommunicationDriver(["island:a", "island:b"])
    configured = NoCommunicationDriver(["island:a", "island:b"])
    isolation = IslandIsolation()

    with (
        patch.object(
            island_isolation, "NoCommunicationDriver", return_value=configured
        ) as create_driver,
        patch.object(
            island_isolation, "NoCommunicationProcessor", wraps=NoCommunicationProcessor
        ) as create_processor,
    ):
        isolation.initialize_context(
            supplied,
            CommunicationProcessorBase,
            {"unused": True},
            HistoryConfig(),
            None,
        )
        _ = isolation.create_processor_module("island:a")

    create_driver.assert_called_once_with(island_ids=list(supplied.incoming_queues))
    create_processor.assert_called_once_with(identifier="island:a")

    with (
        patch.object(configured, "start") as start,
        patch.object(configured, "stop") as stop,
    ):
        isolation.start()
        isolation.stop()

    start.assert_called_once_with()
    stop.assert_called_once_with()
