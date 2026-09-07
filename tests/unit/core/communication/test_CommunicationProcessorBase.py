from queue import Queue

from tyrannis.core.backend_communication import CommunicationProcessorBase
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
        self._message_signal = None

    def start(
        self,
        stop_signal: LocalEvent,
    ) -> None:
        pass

    def stop(self) -> None:
        pass

    def set_message_signal(
        self,
        message_signal: LocalEvent | None,
    ) -> None:
        self._message_signal = message_signal

    @property
    def messages(self) -> Queue:
        return self._messages

    @property
    def outgoing_queue(self) -> Queue:
        return self._outgoing_queue


def test_is_abstract() -> None:
    assert CommunicationProcessorBase.__abstractmethods__


def test_implementation_satisfies_contract() -> None:
    processor = DummyCommunicationProcessor("island:0")

    assert processor.messages is not processor.outgoing_queue


def test_messages_and_outgoing_queue_are_distinct() -> None:
    processor = DummyCommunicationProcessor("island:0")

    assert processor.messages is not processor.outgoing_queue


def test_implementation_accepts_backend_configuration() -> None:
    processor = DummyCommunicationProcessor(
        "island:0",
        driver_ip="127.0.0.1",
        port=5000,
    )

    assert processor.messages is not processor.outgoing_queue
