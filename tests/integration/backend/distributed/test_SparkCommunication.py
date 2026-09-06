import time
from collections.abc import Callable, Generator
from queue import Queue
from threading import Event
from unittest.mock import Mock

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


@pytest.fixture
def communication() -> Generator[
    tuple[
        SparkCommunicationDriver,
        SparkCommunicationProcessor,
        LocalEvent,
    ],
    None,
    None,
]:
    driver_stop_signal = LocalEvent()
    stop_signal = LocalEvent()

    driver = SparkCommunicationDriver(
        island_ids=[FIRST_ISLAND],
        port=PORT,
        stop_signal=driver_stop_signal,
    )

    processor = SparkCommunicationProcessor(
        driver_ip=HOST,
        port=PORT,
        identification=FIRST_ISLAND,
    )

    driver.start()

    processor.start(
        stop_signal=stop_signal,
    )

    wait_for(
        lambda: FIRST_ISLAND in driver._connections,
    )

    yield (
        driver,
        processor,
        stop_signal,
    )

    processor.stop()
    driver.stop()


def test_client_connects_to_driver(
    communication: tuple[
        SparkCommunicationDriver,
        SparkCommunicationProcessor,
        LocalEvent,
    ],
) -> None:
    driver, _, _ = communication

    assert FIRST_ISLAND in driver._connections


def test_client_sends_message_to_driver(
    communication: tuple[
        SparkCommunicationDriver,
        SparkCommunicationProcessor,
        LocalEvent,
    ],
) -> None:
    driver, processor, _ = communication

    assert processor._socket is not None

    message = "message-from-client"

    processor._socket.sendall(
        (processor.STX + message + processor.ETX).encode("utf-8"),
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
    ],
) -> None:
    driver, processor, _ = communication

    message = "message-from-driver"

    driver.outgoing_queues[FIRST_ISLAND].put(
        message,
    )

    wait_for(
        lambda: not processor.messages.empty(),
    )

    assert processor.messages.get() == message


def test_two_clients_connect_to_driver() -> None:
    driver_stop_signal = LocalEvent()

    driver = SparkCommunicationDriver(
        island_ids=[
            FIRST_ISLAND,
            SECOND_ISLAND,
        ],
        port=PORT,
        stop_signal=driver_stop_signal,
    )

    first_stop_signal = LocalEvent()
    second_stop_signal = LocalEvent()

    first = SparkCommunicationProcessor(
        driver_ip=HOST,
        port=PORT,
        identification=FIRST_ISLAND,
    )

    second = SparkCommunicationProcessor(
        driver_ip=HOST,
        port=PORT,
        identification=SECOND_ISLAND,
    )

    driver.start()

    first.start(
        stop_signal=first_stop_signal,
    )

    second.start(
        stop_signal=second_stop_signal,
    )

    try:
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
    driver_stop_signal = LocalEvent()

    driver = SparkCommunicationDriver(
        island_ids=[
            FIRST_ISLAND,
            SECOND_ISLAND,
        ],
        port=PORT,
        stop_signal=driver_stop_signal,
    )

    first_stop_signal = LocalEvent()
    second_stop_signal = LocalEvent()

    first = SparkCommunicationProcessor(
        driver_ip=HOST,
        port=PORT,
        identification=FIRST_ISLAND,
    )

    second = SparkCommunicationProcessor(
        driver_ip=HOST,
        port=PORT,
        identification=SECOND_ISLAND,
    )

    driver.start()

    first.start(
        stop_signal=first_stop_signal,
    )

    second.start(
        stop_signal=second_stop_signal,
    )

    try:
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
    ],
) -> None:
    driver, processor, _ = communication

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


def test_empty_message_is_received_when_etx_is_sent(
    communication: tuple[
        SparkCommunicationDriver,
        SparkCommunicationProcessor,
        LocalEvent,
    ],
) -> None:
    driver, processor, stop_signal = communication

    driver.outgoing_queues[FIRST_ISLAND].put(
        "",
    )

    wait_for(
        lambda: not processor.messages.empty(),
    )

    assert processor.messages.get() == ""
    assert not stop_signal.is_set()


def test_stop_sets_stop_signal(
    communication: tuple[
        SparkCommunicationDriver,
        SparkCommunicationProcessor,
        LocalEvent,
    ],
) -> None:
    driver, _, stop_signal = communication

    driver_stop_signal = driver._stop_signal

    assert not driver_stop_signal.is_set()
    assert not stop_signal.is_set()

    driver_stop_signal.set()

    wait_for(
        stop_signal.is_set,
    )

    assert stop_signal.is_set()


def test_driver_stop_signal_is_sent_to_all_clients() -> None:
    driver_stop_signal = LocalEvent()

    driver = SparkCommunicationDriver(
        island_ids=[
            FIRST_ISLAND,
            SECOND_ISLAND,
        ],
        port=PORT,
        stop_signal=driver_stop_signal,
    )

    first_stop_signal = LocalEvent()
    second_stop_signal = LocalEvent()

    first = SparkCommunicationProcessor(
        driver_ip=HOST,
        port=PORT,
        identification=FIRST_ISLAND,
    )

    second = SparkCommunicationProcessor(
        driver_ip=HOST,
        port=PORT,
        identification=SECOND_ISLAND,
    )

    driver.start()

    first.start(
        stop_signal=first_stop_signal,
    )

    second.start(
        stop_signal=second_stop_signal,
    )

    try:
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
    ],
) -> None:
    driver, processor, _ = communication

    assert processor._socket is not None

    message = "message"

    processor._socket.sendall(
        (processor.STX + message + processor.ETX).encode("utf-8"),
    )

    wait_for(
        lambda: not driver.incoming_queues[FIRST_ISLAND].empty(),
    )

    assert driver.incoming_queues[FIRST_ISLAND].get() == message


def create_processor_runtime(
    processor: SparkCommunicationProcessor,
    *,
    running: bool = True,
) -> None:
    processor._running = Event()
    processor._messages = Queue()
    processor._outgoing_queue = Queue()
    processor._stop_signal = LocalEvent()

    if running:
        processor._running.set()


def test_processor_stop_handles_shutdown_error_and_joins_thread() -> None:
    processor = SparkCommunicationProcessor(
        HOST,
        PORT,
        FIRST_ISLAND,
    )

    socket_mock = Mock()
    socket_mock.shutdown.side_effect = OSError

    thread_mock = Mock()

    processor._socket = socket_mock
    processor._thread = thread_mock

    create_processor_runtime(processor)

    processor.stop()

    socket_mock.close.assert_called_once_with()
    thread_mock.join.assert_called_once_with()

    assert processor._socket is None
    assert processor._thread is None
    assert processor._running is None
    assert processor._messages is None
    assert processor._outgoing_queue is None
    assert processor._stop_signal is None


def test_processor_stop_with_no_socket_but_existing_thread() -> None:
    processor = SparkCommunicationProcessor(
        HOST,
        PORT,
        FIRST_ISLAND,
    )

    thread_mock = Mock()

    processor._thread = thread_mock

    processor.stop()

    thread_mock.join.assert_called_once_with()

    assert processor._thread is None


def test_processor_stop_with_socket_but_no_thread() -> None:
    processor = SparkCommunicationProcessor(
        HOST,
        PORT,
        FIRST_ISLAND,
    )

    socket_mock = Mock()

    processor._socket = socket_mock

    processor.stop()

    socket_mock.close.assert_called_once_with()

    assert processor._socket is None


def test_processor_receive_loop_requires_socket() -> None:
    processor = SparkCommunicationProcessor(
        HOST,
        PORT,
        FIRST_ISLAND,
    )

    with pytest.raises(
        RuntimeError,
        match="Communication socket is not initialized.",
    ):
        processor._receive_loop()


def test_processor_receive_loop_handles_timeout_and_clean_shutdown() -> None:
    processor = SparkCommunicationProcessor(
        HOST,
        PORT,
        FIRST_ISLAND,
    )

    socket_mock = Mock()

    socket_mock.recv.side_effect = [
        TimeoutError,
        b"",
    ]

    processor._socket = socket_mock

    create_processor_runtime(processor)

    processor._receive_loop()

    socket_mock.settimeout.assert_called_once_with(0.1)


def test_processor_receive_loop_sets_stop_on_socket_error() -> None:
    processor = SparkCommunicationProcessor(
        HOST,
        PORT,
        FIRST_ISLAND,
    )

    socket_mock = Mock()
    socket_mock.recv.side_effect = OSError

    processor._socket = socket_mock

    create_processor_runtime(processor)

    processor._receive_loop()

    assert processor._stop_signal is not None
    assert processor._stop_signal.is_set()


def test_processor_receive_loop_handles_invalid_utf8() -> None:
    processor = SparkCommunicationProcessor(
        HOST,
        PORT,
        FIRST_ISLAND,
    )

    socket_mock = Mock()
    socket_mock.recv.return_value = b"\xff"

    processor._socket = socket_mock

    create_processor_runtime(processor)

    processor._receive_loop()

    assert processor._stop_signal is not None
    assert processor._stop_signal.is_set()


def test_processor_receive_loop_exits_when_stopped_before_recv() -> None:
    processor = SparkCommunicationProcessor(
        HOST,
        PORT,
        FIRST_ISLAND,
    )

    socket_mock = Mock()

    processor._socket = socket_mock

    create_processor_runtime(
        processor,
        running=False,
    )

    processor._receive_loop()

    socket_mock.settimeout.assert_called_once_with(0.1)
    socket_mock.recv.assert_not_called()


def test_processor_process_buffer_discards_data_without_frame() -> None:
    processor = SparkCommunicationProcessor(
        HOST,
        PORT,
        FIRST_ISLAND,
    )

    create_processor_runtime(processor)

    processor._receive_buffer = "noise"

    processor._process_buffer()

    assert processor._receive_buffer == ""
    assert processor.messages.empty()


def test_processor_process_buffer_discards_prefix_before_frame() -> None:
    processor = SparkCommunicationProcessor(
        HOST,
        PORT,
        FIRST_ISLAND,
    )

    create_processor_runtime(processor)

    processor._receive_buffer = f"noise{processor.STX}message{processor.ETX}"

    processor._process_buffer()

    assert processor.messages.get() == "message"
    assert processor._receive_buffer == ""


def test_processor_process_buffer_handles_stop_after_frame() -> None:
    processor = SparkCommunicationProcessor(
        HOST,
        PORT,
        FIRST_ISLAND,
    )

    create_processor_runtime(processor)

    processor._receive_buffer = f"{processor.STX}message{processor.ETX}{processor.STOP}"

    processor._process_buffer()

    assert processor.messages.get() == "message"

    assert processor._stop_signal is not None
    assert processor._stop_signal.is_set()

    assert processor._receive_buffer == ""


def test_processor_send_pending_messages_without_socket() -> None:
    processor = SparkCommunicationProcessor(
        HOST,
        PORT,
        FIRST_ISLAND,
    )

    create_processor_runtime(processor)

    processor._send_pending_messages()

    assert processor.outgoing_queue.empty()


def test_processor_send_pending_messages_requeues_and_stops_on_error() -> None:
    processor = SparkCommunicationProcessor(
        HOST,
        PORT,
        FIRST_ISLAND,
    )

    socket_mock = Mock()
    socket_mock.sendall.side_effect = OSError

    processor._socket = socket_mock

    create_processor_runtime(processor)

    processor.outgoing_queue.put("message")

    processor._send_pending_messages()

    assert processor.outgoing_queue.get() == "message"

    assert processor._stop_signal is not None
    assert processor._stop_signal.is_set()


def test_processor_send_pending_messages_requeues_without_setting_stop_when_not_running() -> (
    None
):
    processor = SparkCommunicationProcessor(
        HOST,
        PORT,
        FIRST_ISLAND,
    )

    socket_mock = Mock()
    socket_mock.sendall.side_effect = OSError

    processor._socket = socket_mock

    create_processor_runtime(
        processor,
        running=False,
    )

    processor.outgoing_queue.put("message")

    processor._send_pending_messages()

    assert processor.outgoing_queue.get() == "message"

    assert processor._stop_signal is not None
    assert not processor._stop_signal.is_set()


def test_driver_start_while_running_raises() -> None:
    driver = SparkCommunicationDriver(
        [FIRST_ISLAND],
        PORT,
        LocalEvent(),
    )

    driver._thread = Mock()
    driver._thread.is_alive.return_value = True

    with pytest.raises(
        RuntimeError,
        match="Communication is already running.",
    ):
        driver.start()
