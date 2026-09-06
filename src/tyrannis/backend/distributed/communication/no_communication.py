from __future__ import annotations

from queue import Queue

from ....core.backend_communication import (
    CommunicationDriverBase,
    CommunicationProcessorBase,
)
from ....core.signals import LocalEvent


class NoCommunicationProcessor(CommunicationProcessorBase):
    """Inactive communication layer for processor modules."""

    def __init__(
        self,
        identification: str,
        **kwargs: object,
    ) -> None:
        self._identification = identification

        self._messages: Queue[str] | None = None
        self._outgoing_queue: Queue[str] | None = None

    @property
    def messages(self) -> Queue[str]:
        if self._messages is None:
            raise RuntimeError("Communication processor is not running.")

        return self._messages

    @property
    def outgoing_queue(self) -> Queue[str]:
        if self._outgoing_queue is None:
            raise RuntimeError("Communication processor is not running.")

        return self._outgoing_queue

    def start(
        self,
        stop_signal: LocalEvent,
    ) -> None:
        self._messages = Queue()
        self._outgoing_queue = Queue()

    def stop(self) -> None:
        self._messages = None
        self._outgoing_queue = None


class NoCommunicationDriver(CommunicationDriverBase):
    """Inactive communication layer for driver modules."""

    def __init__(
        self,
        island_ids: list[str],
        stop_signal: LocalEvent,
        **kwargs: object,
    ) -> None:
        self._island_ids = set(island_ids)
        self._stop_signal = stop_signal

        self._incoming_queues: dict[str, Queue[str]] = {
            island_id: Queue() for island_id in self._island_ids
        }

        self._outgoing_queues: dict[str, Queue[str]] = {
            island_id: Queue() for island_id in self._island_ids
        }

    @property
    def incoming_queues(self) -> dict[str, Queue[str]]:
        """Queues containing messages received from processors."""
        return self._incoming_queues

    @property
    def outgoing_queues(self) -> dict[str, Queue[str]]:
        """Queues containing messages to be sent to processors."""
        return self._outgoing_queues

    def start(self) -> None:
        """Do nothing."""

    def stop(self) -> None:
        """Do nothing."""
