from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from .backend_communication import (
    CommunicationDriverBase,
    CommunicationProcessorBase,
)
from .signals import LocalEvent


class MigrationProcessorBase(ABC):
    """Base class for migration processor modules."""

    _initial_iter: int
    _synchronization_iter: int | None
    _communication_processor: CommunicationProcessorBase

    @abstractmethod
    def __init__(
        self,
        initial_iter: int,
        communication_processor: CommunicationProcessorBase,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the migration processor."""

    @abstractmethod
    def start(
        self,
        stop_signal: LocalEvent,
    ) -> None:
        """Start the migration communication protocol."""
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """Stop the migration processor."""
        raise NotImplementedError

    @abstractmethod
    def migration_control(
        self,
        actual_iter: int,
        local_best: str | None,
        insert_arrival_particle: Callable[[dict[str, Any]], None],
        departure_particle: Callable[[str], None],
    ) -> None:
        """
        Control migration and synchronization for the current iteration.

        The migration processor is responsible for determining whether the
        processor must wait for synchronization and for applying pending
        migration operations during the migration control window.

        When `_synchronization_iter` is not `None`, the processor must block
        whenever `actual_iter >= _synchronization_iter` until the
        synchronization condition defined by the migration strategy is
        satisfied.

        A value of `None` for `_synchronization_iter` disables
        iteration-based synchronization.

        Parameters
        ----------
        actual_iter:
            Current processor iteration.

        local_best:
            JSON-serialized representation of the best solution found by the
            processor so far. A value of `None` indicates that no local best
            solution has been established yet.

        insert_arrival_particle:
            Callback used to insert a particle received through migration into
            the processor population.

        departure_particle:
            Callback used to remove a particle selected for migration from the
            processor population.
        """
        raise NotImplementedError


class MigrationDriverBase(ABC):
    """Base class for migration driver modules."""

    _processor_class: type[MigrationProcessorBase]
    _communication_driver: CommunicationDriverBase
    _communication_processor_class: type[CommunicationProcessorBase]
    _migration_processor_init_kargs: dict[str, Any]

    @abstractmethod
    def __init__(
        self,
        initial_iter: int = 1,
        *args: Any,
        **kargs: Any,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def start(self) -> None:
        """Starts migration and communication listening."""
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """Stops migration and communication listening."""
        raise NotImplementedError

    def initialize_context(
        self,
        communication_driver: CommunicationDriverBase,
        communication_processor_class: type[CommunicationProcessorBase],
        communication_processor_kargs: dict[str, Any],
    ) -> None:
        self._communication_driver = communication_driver
        self._communication_processor_class = communication_processor_class

        self._migration_processor_init_kargs["communication"] = (
            communication_processor_kargs.copy()
        )

    def create_processor_module(
        self,
        identification: str,
    ) -> MigrationProcessorBase:
        communication_kargs = self._migration_processor_init_kargs[
            "communication"
        ].copy()

        communication_processor = self._communication_processor_class(
            **communication_kargs,
            identification=identification,
        )

        migration_processor_kargs = self._migration_processor_init_kargs.copy()
        migration_processor_kargs.pop("communication")

        return self._processor_class(
            communication_processor=communication_processor,
            **migration_processor_kargs,
        )
