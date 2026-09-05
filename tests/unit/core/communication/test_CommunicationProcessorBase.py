import inspect

from doubles.communication import DummyCommunicationProcessor

from tyrannis.core.backend_communication import CommunicationProcessorBase


def test_is_abstract() -> None:
    assert inspect.isabstract(CommunicationProcessorBase)


def test_implementation_satisfies_contract() -> None:
    processor = DummyCommunicationProcessor("island:0")

    assert processor._identification == "island:0"
    assert processor.messages.empty()
    assert processor.outgoing_queue.empty()


def test_messages_and_outgoing_queue_are_distinct() -> None:
    processor = DummyCommunicationProcessor("island:0")

    assert processor.messages is not processor.outgoing_queue


def test_implementation_accepts_backend_configuration() -> None:
    processor = DummyCommunicationProcessor(
        "island:0",
        driver_ip="127.0.0.1",
        port=5000,
    )

    assert processor._identification == "island:0"
