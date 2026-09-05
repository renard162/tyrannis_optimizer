import time
from collections.abc import Callable, Generator
from threading import Event
from unittest.mock import Mock, patch

import pytest

from tyrannis.backend.distributed.communication import (
    spark_communication as communication_module,
)
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
        LocalEvent,
    ],
    None,
    None,
]:
    driver_stop_signal = Event()
    wait_signal = LocalEvent()
    stop_signal = LocalEvent()

    driver = SparkCommunicationDriver(
        island_ids=[FIRST_ISLAND],
        port=PORT,
        stop_signal=driver_stop_signal,  # type: ignore
    )

    processor = SparkCommunicationProcessor(
        driver_ip=HOST,
        port=PORT,
        identification=FIRST_ISLAND,
    )

    driver.start()

    processor.start(
        wait_signal=wait_signal,
        stop_signal=stop_signal,
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
        stop_signal=driver_stop_signal,  # type: ignore
    )

    first_wait_signal = LocalEvent()
    first_stop_signal = LocalEvent()

    second_wait_signal = LocalEvent()
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
        wait_signal=first_wait_signal,
        stop_signal=first_stop_signal,
    )

    second.start(
        wait_signal=second_wait_signal,
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
    driver_stop_signal = Event()

    driver = SparkCommunicationDriver(
        island_ids=[
            FIRST_ISLAND,
            SECOND_ISLAND,
        ],
        port=PORT,
        stop_signal=driver_stop_signal,  # type: ignore
    )

    first_wait_signal = LocalEvent()
    first_stop_signal = LocalEvent()

    second_wait_signal = LocalEvent()
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
        wait_signal=first_wait_signal,
        stop_signal=first_stop_signal,
    )

    second.start(
        wait_signal=second_wait_signal,
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
        LocalEvent,
    ],
) -> None:
    driver, processor, _, _ = communication

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
        stop_signal=driver_stop_signal,  # type: ignore
    )

    first_wait_signal = LocalEvent()
    first_stop_signal = LocalEvent()

    second_wait_signal = LocalEvent()
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
        wait_signal=first_wait_signal,
        stop_signal=first_stop_signal,
    )

    second.start(
        wait_signal=second_wait_signal,
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


# ---------------------------------------------------------------------------
# Complementação de cobertura
# ---------------------------------------------------------------------------


def test_processor_stop_handles_shutdown_error_and_joins_thread() -> None:
    processor = SparkCommunicationProcessor(HOST, PORT, FIRST_ISLAND)
    socket_mock = Mock()
    socket_mock.shutdown.side_effect = OSError
    thread_mock = Mock()

    processor._socket = socket_mock
    processor._thread = thread_mock
    processor._running.set()

    processor.stop()

    socket_mock.close.assert_called_once_with()
    thread_mock.join.assert_called_once_with()

    assert processor._socket is None
    assert processor._thread is None


def test_processor_stop_with_no_socket_but_existing_thread() -> None:
    processor = SparkCommunicationProcessor(HOST, PORT, FIRST_ISLAND)
    thread_mock = Mock()

    processor._thread = thread_mock

    processor.stop()

    thread_mock.join.assert_called_once_with()
    assert processor._thread is None


def test_processor_stop_with_socket_but_no_thread() -> None:
    processor = SparkCommunicationProcessor(HOST, PORT, FIRST_ISLAND)
    socket_mock = Mock()

    processor._socket = socket_mock

    processor.stop()

    socket_mock.close.assert_called_once_with()
    assert processor._socket is None


def test_processor_receive_loop_requires_socket() -> None:
    processor = SparkCommunicationProcessor(HOST, PORT, FIRST_ISLAND)

    with pytest.raises(
        RuntimeError,
        match="Communication socket is not initialized.",
    ):
        processor._receive_loop()


def test_processor_receive_loop_handles_timeout_and_clean_shutdown() -> None:
    processor = SparkCommunicationProcessor(HOST, PORT, FIRST_ISLAND)
    socket_mock = Mock()

    socket_mock.recv.side_effect = [
        TimeoutError,
        b"",
    ]

    processor._socket = socket_mock
    processor._running.set()

    processor._receive_loop()

    socket_mock.settimeout.assert_called_once_with(0.1)


def test_processor_receive_loop_sets_stop_on_socket_error() -> None:
    processor = SparkCommunicationProcessor(HOST, PORT, FIRST_ISLAND)
    socket_mock = Mock()

    socket_mock.recv.side_effect = OSError

    processor._socket = socket_mock
    processor._running.set()

    processor._receive_loop()

    assert processor._stop_signal.is_set()


def test_processor_receive_loop_handles_invalid_utf8() -> None:
    processor = SparkCommunicationProcessor(HOST, PORT, FIRST_ISLAND)
    socket_mock = Mock()

    socket_mock.recv.return_value = b"\xff"

    processor._socket = socket_mock
    processor._running.set()

    processor._receive_loop()

    assert processor._stop_signal.is_set()


def test_processor_receive_loop_exits_when_stopped_before_recv() -> None:
    processor = SparkCommunicationProcessor(HOST, PORT, FIRST_ISLAND)
    socket_mock = Mock()

    processor._socket = socket_mock

    processor._receive_loop()

    socket_mock.settimeout.assert_called_once_with(0.1)
    socket_mock.recv.assert_not_called()


def test_processor_process_buffer_discards_data_without_frame() -> None:
    processor = SparkCommunicationProcessor(HOST, PORT, FIRST_ISLAND)
    processor._receive_buffer = "noise"

    processor._process_buffer()

    assert processor._receive_buffer == ""


def test_processor_process_buffer_discards_prefix_before_frame() -> None:
    processor = SparkCommunicationProcessor(HOST, PORT, FIRST_ISLAND)

    processor._receive_buffer = f"noise{processor.STX}message{processor.ETX}"

    processor._process_buffer()

    assert processor.messages.get() == "message"
    assert processor._receive_buffer == ""


def test_processor_process_buffer_handles_stop_after_frame() -> None:
    processor = SparkCommunicationProcessor(HOST, PORT, FIRST_ISLAND)

    processor._receive_buffer = f"{processor.STX}message{processor.ETX}{processor.STOP}"

    processor._process_buffer()

    assert processor.messages.get() == "message"
    assert processor._stop_signal.is_set()
    assert processor._receive_buffer == ""


def test_processor_send_pending_messages_without_socket() -> None:
    processor = SparkCommunicationProcessor(HOST, PORT, FIRST_ISLAND)

    processor._send_pending_messages()

    assert processor.outgoing_queue.empty()


def test_processor_send_pending_messages_requeues_and_stops_on_error() -> None:
    processor = SparkCommunicationProcessor(HOST, PORT, FIRST_ISLAND)
    socket_mock = Mock()

    socket_mock.sendall.side_effect = OSError

    processor._socket = socket_mock
    processor._running.set()
    processor.outgoing_queue.put("message")

    processor._send_pending_messages()

    assert processor.outgoing_queue.get() == "message"
    assert processor._stop_signal.is_set()


def test_processor_send_pending_messages_requeues_without_setting_stop_when_not_running() -> (
    None
):
    processor = SparkCommunicationProcessor(HOST, PORT, FIRST_ISLAND)
    socket_mock = Mock()

    socket_mock.sendall.side_effect = OSError

    processor._socket = socket_mock
    processor.outgoing_queue.put("message")

    processor._send_pending_messages()

    assert processor.outgoing_queue.get() == "message"
    assert not processor._stop_signal.is_set()


def test_driver_start_while_running_raises() -> None:
    driver = SparkCommunicationDriver(
        [FIRST_ISLAND],
        PORT,
        Event(),  # type: ignore
    )

    driver._thread = Mock()
    driver._thread.is_alive.return_value = True

    with pytest.raises(
        RuntimeError,
        match="Communication is already running.",
    ):
        driver.start()


def test_driver_stop_handles_connection_and_server_close_errors() -> None:
    driver = SparkCommunicationDriver(
        [FIRST_ISLAND],
        PORT,
        Event(),  # type: ignore
    )

    server_socket = Mock()
    server_socket.close.side_effect = OSError

    connection = Mock()
    connection.close.side_effect = OSError

    selector = Mock()
    thread = Mock()

    driver._server_socket = server_socket
    driver._connections = {
        FIRST_ISLAND: connection,
    }
    driver._selector = selector
    driver._thread = thread
    driver._running.set()

    driver.stop()

    thread.join.assert_called_once_with()
    selector.close.assert_called_once_with()

    assert driver._server_socket is None
    assert driver._thread is None
    assert driver._selector is None


def test_driver_stop_with_no_server_but_existing_thread() -> None:
    driver = SparkCommunicationDriver(
        [FIRST_ISLAND],
        PORT,
        Event(),  # type: ignore
    )

    thread_mock = Mock()
    driver._thread = thread_mock

    driver.stop()

    thread_mock.join.assert_called_once_with()
    assert driver._thread is None


def test_driver_stop_with_server_but_no_thread_or_selector() -> None:
    driver = SparkCommunicationDriver(
        [FIRST_ISLAND],
        PORT,
        Event(),  # type: ignore
    )

    server_socket = Mock()
    driver._server_socket = server_socket

    driver.stop()

    server_socket.close.assert_called_once_with()
    assert driver._server_socket is None


def test_driver_stop_with_no_selector() -> None:
    driver = SparkCommunicationDriver(
        [FIRST_ISLAND],
        PORT,
        Event(),  # type: ignore
    )

    driver.stop()

    assert driver._selector is None


def test_driver_communication_loop_requires_selector() -> None:
    driver = SparkCommunicationDriver(
        [FIRST_ISLAND],
        PORT,
        Event(),  # type: ignore
    )

    with pytest.raises(
        RuntimeError,
        match="Selector is not initialized.",
    ):
        driver._communication_loop()


def test_driver_communication_loop_sends_stop_once() -> None:
    driver = SparkCommunicationDriver(
        [FIRST_ISLAND],
        PORT,
        Event(),  # type: ignore
    )

    selector = Mock()
    selector.select.side_effect = [
        [],
        [],
    ]

    driver._selector = selector
    driver._server_socket = Mock()
    driver._stop_signal.set()
    driver._running.set()

    def stop_after_first_send() -> None:
        driver._running.clear()

    with patch.object(
        driver,
        "_send_stop_to_all",
        side_effect=stop_after_first_send,
    ) as send_stop:
        driver._communication_loop()

    send_stop.assert_called_once_with()
    assert driver._stop_sent


def test_driver_communication_loop_ignores_non_read_event() -> None:
    driver = SparkCommunicationDriver(
        [FIRST_ISLAND],
        PORT,
        Event(),  # type: ignore
    )

    selector = Mock()
    key = Mock()

    key.fileobj = Mock()

    selector.select.side_effect = [
        [(key, 0)],
    ]

    driver._selector = selector
    driver._server_socket = Mock()
    driver._running.set()

    def stop_loop() -> None:
        driver._running.clear()

    with patch.object(
        driver,
        "_send_pending_messages",
        side_effect=stop_loop,
    ):
        driver._communication_loop()


def test_driver_accept_connection_handles_missing_server() -> None:
    driver = SparkCommunicationDriver(
        [FIRST_ISLAND],
        PORT,
        Event(),  # type: ignore
    )

    driver._accept_connection()

    assert driver._connections == {}


def test_driver_accept_connection_requires_selector() -> None:
    driver = SparkCommunicationDriver(
        [FIRST_ISLAND],
        PORT,
        Event(),  # type: ignore
    )

    driver._server_socket = Mock()

    with pytest.raises(
        RuntimeError,
        match="Selector is not initialized.",
    ):
        driver._accept_connection()


def test_driver_accept_connection_handles_accept_error() -> None:
    driver = SparkCommunicationDriver(
        [FIRST_ISLAND],
        PORT,
        Event(),  # type: ignore
    )

    server_socket = Mock()
    server_socket.accept.side_effect = OSError

    driver._server_socket = server_socket
    driver._selector = Mock()

    driver._accept_connection()

    driver._selector.register.assert_not_called()


def test_driver_receive_ignores_non_socket_file_object() -> None:
    driver = SparkCommunicationDriver(
        [FIRST_ISLAND],
        PORT,
        Event(),  # type: ignore
    )

    key = Mock()
    key.fileobj = object()

    driver._receive(key)


def test_driver_receive_closes_connection_with_invalid_island_id() -> None:
    driver = SparkCommunicationDriver(
        [FIRST_ISLAND],
        PORT,
        Event(),  # type: ignore
    )

    connection = communication_module.socket.socketpair()[0]

    key = Mock()
    key.fileobj = connection
    key.data = 123

    with patch.object(
        driver,
        "_close_connection",
    ) as close_connection:
        driver._receive(key)

    close_connection.assert_called_once_with(connection)

    connection.close()


def test_driver_receive_handshake_requires_selector() -> None:
    driver = SparkCommunicationDriver(
        [FIRST_ISLAND],
        PORT,
        Event(),  # type: ignore
    )

    connection = communication_module.socket.socketpair()[0]

    with pytest.raises(
        RuntimeError,
        match="Selector is not initialized.",
    ):
        driver._receive_handshake(connection)

    connection.close()


def test_driver_receive_handshake_returns_when_no_data() -> None:
    driver = SparkCommunicationDriver(
        [FIRST_ISLAND],
        PORT,
        Event(),  # type: ignore
    )

    connection = communication_module.socket.socketpair()[0]

    driver._selector = Mock()

    with patch.object(
        driver,
        "_receive_data",
        return_value=False,
    ):
        driver._receive_handshake(connection)

    connection.close()


def test_driver_receive_handshake_with_no_complete_message() -> None:
    driver = SparkCommunicationDriver(
        [FIRST_ISLAND],
        PORT,
        Event(),  # type: ignore
    )

    connection = communication_module.socket.socketpair()[0]

    driver._selector = Mock()

    with (
        patch.object(
            driver,
            "_receive_data",
            return_value=True,
        ),
        patch.object(
            driver,
            "_extract_messages",
            return_value=[],
        ),
    ):
        driver._receive_handshake(connection)

    assert driver._connections == {}

    connection.close()


def test_driver_receive_handshake_rejects_unknown_island() -> None:
    driver = SparkCommunicationDriver(
        [FIRST_ISLAND],
        PORT,
        Event(),  # type: ignore
    )

    connection = communication_module.socket.socketpair()[0]

    driver._selector = Mock()
    driver._receive_buffers[connection] = ""

    with (
        patch.object(
            driver,
            "_receive_data",
            return_value=True,
        ),
        patch.object(
            driver,
            "_extract_messages",
            return_value=["unknown"],
        ),
        patch.object(
            driver,
            "_close_connection",
        ) as close_connection,
    ):
        driver._receive_handshake(connection)

    close_connection.assert_called_once_with(connection)

    connection.close()


def test_driver_receive_handshake_replaces_existing_connection() -> None:
    driver = SparkCommunicationDriver(
        [FIRST_ISLAND],
        PORT,
        Event(),  # type: ignore
    )

    old_connection, new_connection = communication_module.socket.socketpair()

    driver._selector = Mock()

    driver._connections[FIRST_ISLAND] = old_connection
    driver._receive_buffers[old_connection] = ""
    driver._receive_buffers[new_connection] = ""

    with (
        patch.object(
            driver,
            "_receive_data",
            return_value=True,
        ),
        patch.object(
            driver,
            "_extract_messages",
            return_value=[FIRST_ISLAND],
        ),
        patch.object(
            driver,
            "_close_connection",
            wraps=driver._close_connection,
        ) as close_connection,
    ):
        driver._receive_handshake(new_connection)

    close_connection.assert_called_once_with(
        old_connection,
        FIRST_ISLAND,
    )

    assert driver._connections[FIRST_ISLAND] is new_connection

    driver._close_connection(
        new_connection,
        FIRST_ISLAND,
    )

    old_connection.close()


def test_driver_receive_message_returns_when_no_data() -> None:
    driver = SparkCommunicationDriver(
        [FIRST_ISLAND],
        PORT,
        Event(),  # type: ignore
    )

    connection = communication_module.socket.socketpair()[0]

    with patch.object(
        driver,
        "_receive_data",
        return_value=False,
    ):
        driver._receive_message(
            connection,
            FIRST_ISLAND,
        )

    assert driver.incoming_queues[FIRST_ISLAND].empty()

    connection.close()


def test_driver_receive_data_handles_socket_error() -> None:
    driver = SparkCommunicationDriver(
        [FIRST_ISLAND],
        PORT,
        Event(),  # type: ignore
    )

    connection = Mock()
    connection.recv.side_effect = OSError

    with patch.object(
        driver,
        "_close_connection",
    ) as close_connection:
        assert not driver._receive_data(connection)

    close_connection.assert_called_once_with(connection)


def test_driver_receive_data_handles_connection_close() -> None:
    driver = SparkCommunicationDriver(
        [FIRST_ISLAND],
        PORT,
        Event(),  # type: ignore
    )

    connection = Mock()
    connection.recv.return_value = b""

    with patch.object(
        driver,
        "_close_connection",
    ) as close_connection:
        assert not driver._receive_data(connection)

    close_connection.assert_called_once_with(connection)


def test_driver_receive_data_handles_invalid_utf8() -> None:
    driver = SparkCommunicationDriver(
        [FIRST_ISLAND],
        PORT,
        Event(),  # type: ignore
    )

    connection = Mock()
    connection.recv.return_value = b"\xff"

    driver._receive_buffers[connection] = ""

    with patch.object(
        driver,
        "_close_connection",
    ) as close_connection:
        assert not driver._receive_data(connection)

    close_connection.assert_called_once_with(connection)


def test_driver_extract_messages_discards_noise() -> None:
    driver = SparkCommunicationDriver(
        [FIRST_ISLAND],
        PORT,
        Event(),  # type: ignore
    )

    connection = Mock()

    driver._receive_buffers[connection] = "noise"

    assert driver._extract_messages(connection) == []
    assert driver._receive_buffers[connection] == ""


def test_driver_extract_messages_discards_prefix_before_frame() -> None:
    driver = SparkCommunicationDriver(
        [FIRST_ISLAND],
        PORT,
        Event(),  # type: ignore
    )

    connection = Mock()

    driver._receive_buffers[connection] = f"noise{driver.STX}message{driver.ETX}"

    assert driver._extract_messages(connection) == ["message"]
    assert driver._receive_buffers[connection] == ""


def test_driver_extract_messages_handles_stop_after_frame() -> None:
    driver = SparkCommunicationDriver(
        [FIRST_ISLAND],
        PORT,
        Event(),  # type: ignore
    )

    connection = Mock()

    driver._receive_buffers[connection] = (
        f"{driver.STX}message{driver.ETX}{driver.STOP}"
    )

    assert driver._extract_messages(connection) == ["message"]
    assert driver._stop_signal.is_set()
    assert driver._receive_buffers[connection] == ""


def test_driver_send_pending_messages_skips_unconnected_island() -> None:
    driver = SparkCommunicationDriver(
        [FIRST_ISLAND],
        PORT,
        Event(),  # type: ignore
    )

    driver.outgoing_queues[FIRST_ISLAND].put("message")

    driver._send_pending_messages()

    assert driver.outgoing_queues[FIRST_ISLAND].get() == "message"


def test_driver_send_pending_messages_handles_send_error() -> None:
    driver = SparkCommunicationDriver(
        [FIRST_ISLAND],
        PORT,
        Event(),  # type: ignore
    )

    connection = Mock()
    connection.sendall.side_effect = OSError

    driver._connections[FIRST_ISLAND] = connection
    driver.outgoing_queues[FIRST_ISLAND].put("message")

    with patch.object(
        driver,
        "_close_connection",
    ) as close_connection:
        driver._send_pending_messages()

    assert driver.outgoing_queues[FIRST_ISLAND].get() == "message"

    close_connection.assert_called_once_with(
        connection,
        FIRST_ISLAND,
    )


def test_driver_send_stop_to_all_handles_send_error() -> None:
    driver = SparkCommunicationDriver(
        [FIRST_ISLAND],
        PORT,
        Event(),  # type: ignore
    )

    connection = Mock()
    connection.sendall.side_effect = OSError

    driver._connections[FIRST_ISLAND] = connection

    with patch.object(
        driver,
        "_close_connection",
    ) as close_connection:
        driver._send_stop_to_all()

    close_connection.assert_called_once_with(
        connection,
        FIRST_ISLAND,
    )


def test_driver_close_connection_ignores_unregister_error() -> None:
    driver = SparkCommunicationDriver(
        [FIRST_ISLAND],
        PORT,
        Event(),  # type: ignore
    )

    connection = Mock()
    selector = Mock()

    selector.unregister.side_effect = KeyError
    driver._selector = selector

    driver._close_connection(connection)

    connection.close.assert_called_once_with()


def test_driver_close_connection_ignores_close_error() -> None:
    driver = SparkCommunicationDriver(
        [FIRST_ISLAND],
        PORT,
        Event(),  # type: ignore
    )

    connection = Mock()
    connection.close.side_effect = OSError

    driver._selector = Mock()

    driver._close_connection(connection)


def test_driver_close_connection_removes_island_mapping() -> None:
    driver = SparkCommunicationDriver(
        [FIRST_ISLAND],
        PORT,
        Event(),  # type: ignore
    )

    connection = Mock()

    driver._selector = None
    driver._connections[FIRST_ISLAND] = connection
    driver._receive_buffers[connection] = "partial"

    driver._close_connection(
        connection,
        FIRST_ISLAND,
    )

    assert FIRST_ISLAND not in driver._connections
    assert connection not in driver._receive_buffers
