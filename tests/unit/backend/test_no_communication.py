import pytest

from tyrannis.backend.distributed.communication.no_communication import (
    NoCommunicationDriver,
    NoCommunicationProcessor,
)


def test_processor_queues_are_unavailable_before_start() -> None:
    processor = NoCommunicationProcessor("island:0")

    for name in ("messages", "outgoing_queue"):
        with pytest.raises(
            RuntimeError, match="Communication processor is not running"
        ):
            getattr(processor, name)


def test_processor_start_exposes_stable_distinct_queues_and_rejects_reentry() -> None:
    processor = NoCommunicationProcessor("island:0")

    processor.start()
    messages = processor.messages
    outgoing = processor.outgoing_queue

    assert messages is not outgoing
    assert processor.messages is messages
    assert processor.outgoing_queue is outgoing
    with pytest.raises(RuntimeError, match="Communication is already running"):
        processor.start()


def test_processor_stop_makes_queues_unavailable() -> None:
    processor = NoCommunicationProcessor("island:0")
    processor.start()

    processor.stop()

    for name in ("messages", "outgoing_queue"):
        with pytest.raises(
            RuntimeError, match="Communication processor is not running"
        ):
            getattr(processor, name)


def test_driver_queues_remain_distinct_and_stable_through_lifecycle() -> None:
    driver = NoCommunicationDriver(["island:0", "island:1"])

    incoming = driver.incoming_queues
    outgoing = driver.outgoing_queues

    assert set(incoming) == {"island:0", "island:1"}
    assert set(outgoing) == {"island:0", "island:1"}
    assert incoming["island:0"] is not incoming["island:1"]
    assert outgoing["island:0"] is not outgoing["island:1"]
    driver.start()
    driver.stop()

    for island_id in ("island:0", "island:1"):
        assert incoming[island_id] is not outgoing[island_id]
        assert driver.incoming_queues[island_id] is incoming[island_id]
        assert driver.outgoing_queues[island_id] is outgoing[island_id]
    assert driver.incoming_queues is incoming
    assert driver.outgoing_queues is outgoing
