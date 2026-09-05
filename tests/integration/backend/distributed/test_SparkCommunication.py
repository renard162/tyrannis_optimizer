import time
from collections.abc import Callable
from threading import Event

import pytest

from tyrannis.backend.distributed.communication.spark_communication import (
    SparkCommunicationDriver,
    SparkCommunicationProcessor,
)
from tyrannis.core.processor import LocalEvent


HOST = "127.0.0.1"
PORT = 5000

FIRST_ISLAND = "island:0"
SECOND_ISLAND = "island:1"

TIMEOUT = 5.0


def wait_for(
    condition: Callable[[], bool],
    timeout: float = TIMEOUT,
) -> None:
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if condition():
            return

        time.sleep(0.01)

    raise AssertionError(
        "Condition was not satisfied before timeout.",
    )


def connect_processor(
    processor: SparkCommunicationProcessor,
    island_id: str,
) -> None:
    assert processor._socket is not None

    processor._socket.sendall(
        island_id.encode("utf-8"),
    )


@pytest.fixture
def communication() -> tuple[
    SparkCommunicationDriver,
    SparkCommunicationProcessor,
    LocalEvent,
    LocalEvent,
]:
    driver_stop_signal = Event()
    wait_signal = LocalEvent()
    stop_signal = LocalEvent()

    driver = SparkCommunicationDriver(
        island_ids=[FIRST_ISLAND],
        port=PORT,
        stop_signal=driver_stop_signal,
    )

    processor = SparkCommunicationProcessor(
        driver_ip=HOST,
        port=PORT,
    )

    driver.start()

    processor.start(
        wait_signal=wait_signal,
        stop_signal=stop_signal,
    )

    connect_processor(
        processor,
        FIRST_ISLAND,
    )

    wait_for(
        lambda: FIRST_ISLAND in driver._connections,
    )

    yield (
        driver,
        processor,
        wait_signal,
        stop_signal,
    )

    processor.stop()
    driver.stop()


def test_client_connects_to_driver(
    communication: tuple[
        SparkCommunicationDriver,
        SparkCommunicationProcessor,
        LocalEvent,
        LocalEvent,
    ],
) -> None:
    driver, _, _, _ = communication

    assert FIRST_ISLAND in driver._connections


def test_client_sends_message_to_driver(
    communication: tuple[
        SparkCommunicationDriver,
        SparkCommunicationProcessor,
        LocalEvent,
        LocalEvent,
    ],
) -> None:
    driver, processor, _, _ = communication

    assert processor._socket is not None

    message = "message-from-client"

    processor._socket.sendall(
        message.encode("utf-8"),
    )

    wait_for(
        lambda: not driver.incoming_queues[FIRST_ISLAND].empty(),
    )

    assert driver.incoming_queues[FIRST_ISLAND].get() == message


def test_driver_sends_message_to_client(
    communication: tuple[
        SparkCommunicationDriver,
        SparkCommunicationProcessor,
        LocalEvent,
        LocalEvent,
    ],
) -> None:
    driver, processor, _, _ = communication

    message = "message-from-driver"

    driver.outgoing_queues[FIRST_ISLAND].put(
        message,
    )

    wait_for(
        lambda: not processor.messages.empty(),
    )

    assert processor.messages.get() == message


def test_two_clients_connect_to_driver() -> None:
    driver_stop_signal = Event()

    driver = SparkCommunicationDriver(
        island_ids=[
            FIRST_ISLAND,
            SECOND_ISLAND,
        ],
        port=PORT,
        stop_signal=driver_stop_signal,
    )

    first_wait_signal = LocalEvent()
    first_stop_signal = LocalEvent()

    second_wait_signal = LocalEvent()
    second_stop_signal = LocalEvent()

    first = SparkCommunicationProcessor(
        driver_ip=HOST,
        port=PORT,
    )

    second = SparkCommunicationProcessor(
        driver_ip=HOST,
        port=PORT,
    )

    driver.start()

    first.start(
        wait_signal=first_wait_signal,
        stop_signal=first_stop_signal,
    )

    second.start(
        wait_signal=second_wait_signal,
        stop_signal=second_stop_signal,
    )

    try:
        connect_processor(
            first,
            FIRST_ISLAND,
        )

        connect_processor(
            second,
            SECOND_ISLAND,
        )

        wait_for(
            lambda: (
                set(driver._connections)
                == {
                    FIRST_ISLAND,
                    SECOND_ISLAND,
                }
            ),
        )

        assert FIRST_ISLAND in driver._connections
        assert SECOND_ISLAND in driver._connections

    finally:
        first.stop()
        second.stop()
        driver.stop()


def test_messages_are_routed_to_correct_clients() -> None:
    driver_stop_signal = Event()

    driver = SparkCommunicationDriver(
        island_ids=[
            FIRST_ISLAND,
            SECOND_ISLAND,
        ],
        port=PORT,
        stop_signal=driver_stop_signal,
    )

    first_wait_signal = LocalEvent()
    first_stop_signal = LocalEvent()

    second_wait_signal = LocalEvent()
    second_stop_signal = LocalEvent()

    first = SparkCommunicationProcessor(
        driver_ip=HOST,
        port=PORT,
    )

    second = SparkCommunicationProcessor(
        driver_ip=HOST,
        port=PORT,
    )

    driver.start()

    first.start(
        wait_signal=first_wait_signal,
        stop_signal=first_stop_signal,
    )

    second.start(
        wait_signal=second_wait_signal,
        stop_signal=second_stop_signal,
    )

    try:
        connect_processor(
            first,
            FIRST_ISLAND,
        )

        connect_processor(
            second,
            SECOND_ISLAND,
        )

        wait_for(
            lambda: (
                set(driver._connections)
                == {
                    FIRST_ISLAND,
                    SECOND_ISLAND,
                }
            ),
        )

        first_message = "message-for-first"
        second_message = "message-for-second"

        driver.outgoing_queues[FIRST_ISLAND].put(
            first_message,
        )

        driver.outgoing_queues[SECOND_ISLAND].put(
            second_message,
        )

        wait_for(
            lambda: not first.messages.empty() and not second.messages.empty(),
        )

        assert first.messages.get() == first_message
        assert second.messages.get() == second_message

        assert first.messages.empty()
        assert second.messages.empty()

    finally:
        first.stop()
        second.stop()
        driver.stop()


def test_multiple_messages_are_queued_in_order(
    communication: tuple[
        SparkCommunicationDriver,
        SparkCommunicationProcessor,
        LocalEvent,
        LocalEvent,
    ],
) -> None:
    driver, processor, _, _ = communication

    assert processor._socket is not None

    messages = [
        "message-1",
        "message-2",
        "message-3",
    ]

    for message in messages:
        driver.outgoing_queues[FIRST_ISLAND].put(
            message,
        )

    wait_for(
        lambda: processor.messages.qsize() == len(messages),
    )

    received = [processor.messages.get() for _ in messages]

    assert received == messages


def test_etx_sets_wait_signal(
    communication: tuple[
        SparkCommunicationDriver,
        SparkCommunicationProcessor,
        LocalEvent,
        LocalEvent,
    ],
) -> None:
    driver, _, wait_signal, stop_signal = communication

    driver.outgoing_queues[FIRST_ISLAND].put(
        driver.ETX,
    )

    wait_for(
        wait_signal.is_set,
    )

    assert wait_signal.is_set()
    assert not stop_signal.is_set()


def test_stop_sets_stop_signal(
    communication: tuple[
        SparkCommunicationDriver,
        SparkCommunicationProcessor,
        LocalEvent,
        LocalEvent,
    ],
) -> None:
    driver, _, wait_signal, stop_signal = communication

    driver.outgoing_queues[FIRST_ISLAND].put(
        driver.STOP,
    )

    wait_for(
        stop_signal.is_set,
    )

    assert stop_signal.is_set()
    assert not wait_signal.is_set()


def test_driver_stop_signal_is_sent_to_all_clients() -> None:
    driver_stop_signal = Event()

    driver = SparkCommunicationDriver(
        island_ids=[
            FIRST_ISLAND,
            SECOND_ISLAND,
        ],
        port=PORT,
        stop_signal=driver_stop_signal,
    )

    first_wait_signal = LocalEvent()
    first_stop_signal = LocalEvent()

    second_wait_signal = LocalEvent()
    second_stop_signal = LocalEvent()

    first = SparkCommunicationProcessor(
        driver_ip=HOST,
        port=PORT,
    )

    second = SparkCommunicationProcessor(
        driver_ip=HOST,
        port=PORT,
    )

    driver.start()

    first.start(
        wait_signal=first_wait_signal,
        stop_signal=first_stop_signal,
    )

    second.start(
        wait_signal=second_wait_signal,
        stop_signal=second_stop_signal,
    )

    try:
        connect_processor(
            first,
            FIRST_ISLAND,
        )

        connect_processor(
            second,
            SECOND_ISLAND,
        )

        wait_for(
            lambda: (
                set(driver._connections)
                == {
                    FIRST_ISLAND,
                    SECOND_ISLAND,
                }
            ),
        )

        driver_stop_signal.set()

        wait_for(
            lambda: first_stop_signal.is_set() and second_stop_signal.is_set(),
        )

        assert first_stop_signal.is_set()
        assert second_stop_signal.is_set()

    finally:
        first.stop()
        second.stop()
        driver.stop()


def test_stx_etx_frame_message(
    communication: tuple[
        SparkCommunicationDriver,
        SparkCommunicationProcessor,
        LocalEvent,
        LocalEvent,
    ],
) -> None:
    driver, processor, _, _ = communication

    assert processor._socket is not None

    message = "message"

    processor._socket.sendall(
        (processor.STX + message + processor.ETX).encode("utf-8"),
    )

    wait_for(
        lambda: not driver.incoming_queues[FIRST_ISLAND].empty(),
    )

    assert driver.incoming_queues[FIRST_ISLAND].get() == message
