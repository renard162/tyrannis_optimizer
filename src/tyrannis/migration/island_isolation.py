from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..core.backend_communication import CommunicationProcessorBase
from ..core.backend_migration import (
    MigrationDriverBase,
    MigrationProcessorBase,
)
from ..core.signals import LocalEvent
from ..backend.distributed.communication.no_communication import (
    NoCommunicationDriver,
    NoCommunicationProcessor,
)


class IslandIsolationProcessor(MigrationProcessorBase):
    """Inactive migration processor for isolated islands."""

    def __init__(
        self,
        initial_iter: int,
        communication_processor: CommunicationProcessorBase,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._initial_iter = initial_iter
        self._synchronization_iter = None
        self._communication_processor = communication_processor

    def start(self) -> None:
        """Start the inactive communication processor."""

        self._communication_processor.start()

    def stop(self) -> None:
        """Stop the inactive communication processor."""

        self._communication_processor.stop()

    def initialize_loop_context(self, migration_signal: LocalEvent) -> None:
        """Do nothing."""

    def finalize_loop_context(self) -> None:
        """Do nothing."""

    def migration_control(
        self,
        actual_iter: int,
        population: dict[str, float | None],
        local_best: str | None,
        insert_arrival_particle: Callable[[dict[str, Any]], None],
        departure_particle: Callable[[str], None],
    ) -> None:
        """Do nothing."""


class IslandIsolation(MigrationDriverBase):
    """Migration strategy that keeps all islands isolated."""

    _processor_class = IslandIsolationProcessor

    def __init__(
        self,
        initial_iter: int = 1,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._migration_processor_init_kargs = {
            "initial_iter": initial_iter,
        }

    def initialize_context(
        self,
        communication_driver,
        communication_processor_class,
        communication_processor_kargs: dict[str, Any],
    ) -> None:
        """Configure inactive communication for isolated islands."""

        island_ids = list(
            communication_driver.incoming_queues.keys(),
        )

        no_communication_stop_signal = LocalEvent()

        no_communication_driver = NoCommunicationDriver(
            island_ids=island_ids,
            stop_signal=no_communication_stop_signal,
        )

        super().initialize_context(
            communication_driver=no_communication_driver,
            communication_processor_class=NoCommunicationProcessor,
            communication_processor_kargs={},
        )

    def start(self) -> None:
        """Start the inactive migration strategy."""

        self._communication_driver.start()

    def stop(self) -> None:
        """Stop the inactive migration strategy."""

        self._communication_driver.stop()
