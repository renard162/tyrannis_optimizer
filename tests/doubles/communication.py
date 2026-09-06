from queue import Queue

from tyrannis.core.backend_communication import (
    CommunicationDriverBase,
    CommunicationProcessorBase,
)
from tyrannis.core.processor import LocalEvent


class DummyCommunicationProcessor(CommunicationProcessorBase):
    def __init__(
        self,
        identification: str,
        **kwargs: object,
    ) -> None:
        self._identification = identification
        self._messages = Queue()
        self._outgoing_queue = Queue()

    @property
    def messages(self) -> Queue[str]:
        return self._messages

    @property
    def outgoing_queue(self) -> Queue[str]:
        return self._outgoing_queue

    def start(self, stop_signal: LocalEvent) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError


def start(self, stop_signal: LocalEvent) -> None:
    self._stop_signal = stop_signal

    def stop(self) -> None:
        pass


class DummyCommunicationDriver(CommunicationDriverBase):
    def __init__(
        self,
        island_ids: list[str],
        stop_signal: LocalEvent,
        **kwargs: object,
    ) -> None:
        self._island_ids = island_ids
        self._stop_signal = stop_signal
        self._incoming_queues = {island_id: Queue() for island_id in island_ids}
        self._outgoing_queues = {island_id: Queue() for island_id in island_ids}

    @property
    def incoming_queues(self) -> dict[str, Queue[str]]:
        return self._incoming_queues

    @property
    def outgoing_queues(self) -> dict[str, Queue[str]]:
        return self._outgoing_queues

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass
