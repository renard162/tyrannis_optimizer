from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np

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
        population: dict[str, float | None],
        local_best: str | None,
        insert_arrival_particle: Callable[[dict[str, Any]], None],
        departure_particle: Callable[[str], None],
    ) -> None:
        """
        Control migration for the current iteration.

        Parameters
        ----------
        actual_iter:
            Current processor iteration.

        population:
            Current processor population represented as a dictionary mapping
            particle identifiers to their fitness values.

            The population is ordered by fitness in ascending order. Therefore,
            the first entry represents the best particle and the last entry
            represents the worst particle.

            Tyrannis minimizes objective functions, so lower fitness values are
            better.

        local_best:
            JSON-serialized representation of the best solution found by the
            processor so far.

        insert_arrival_particle:
            Callback used to insert or apply a particle received through migration.

        departure_particle:
            Callback used to remove a particle selected for migration.
        """
        raise NotImplementedError

    @staticmethod
    def _get_particle_fitness(
        population: dict[str, float | None],
        particle_id: str,
    ) -> float:
        """Return a particle fitness suitable for population ordering."""
        fitness = population[particle_id]

        if fitness is None:
            return np.inf

        return fitness


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
