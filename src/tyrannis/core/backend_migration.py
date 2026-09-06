from abc import ABC, abstractmethod
from typing import Any

from .backend_communication import (
    CommunicationDriverBase,
    CommunicationProcessorBase,
)
from .processor import LocalEvent


class MigrationProcessorBase(ABC):
    """Base class for migration processor modules."""

    _initial_iter: int
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
    def start(self, wait_signal: Any, stop_signal: Any) -> None:
        """Start the migration communication protocol."""
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """Stop the migration processor."""

    @abstractmethod
    def check_particles(self) -> None:
        """Check the processor particles for migration."""


class MigrationDriverBase(ABC):
    _processor_class: type[MigrationProcessorBase]
    _communication_driver: CommunicationDriverBase
    _communication_processor_class: type[CommunicationProcessorBase]
    _migration_processor_init_kargs: dict[str, Any]

    @abstractmethod
    def __init__(self, initial_iter: int = 1, *args: Any, **kargs: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def start(self) -> None:
        """Starts migration and communication listening"""
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """Stops migration and communication listening"""
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

    def create_processor_module(self, identification: str) -> MigrationProcessorBase:
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
