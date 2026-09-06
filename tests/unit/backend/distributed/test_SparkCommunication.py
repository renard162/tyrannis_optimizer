import selectors
import socket
from queue import Queue
from threading import Event
from unittest.mock import Mock, patch

import pytest

from tyrannis.backend.distributed.communication.spark_communication import (
    SparkCommunicationDriver,
    SparkCommunicationProcessor,
)
from tyrannis.core.signals import LocalEvent

FIRST_ISLAND = "island:0"
SECOND_ISLAND = "island:1"


def create_processor() -> SparkCommunicationProcessor:
    return SparkCommunicationProcessor(
        driver_ip="127.0.0.1",
        port=5000,
        identification=FIRST_ISLAND,
    )


def prepare_processor(
    processor: SparkCommunicationProcessor,
) -> None:
    processor._running = Event()
    processor._stop_signal = LocalEvent()
    processor._messages = Queue()
    processor._outgoing_queue = Queue()


def create_driver() -> SparkCommunicationDriver:
    return SparkCommunicationDriver(
        island_ids=[FIRST_ISLAND, SECOND_ISLAND],
        port=5000,
        stop_signal=LocalEvent(),
    )


def test_processor_init() -> None:
    processor = create_processor()

    assert processor._driver_ip == "127.0.0.1"
    assert processor._port == 5000
    assert processor._identification == FIRST_ISLAND

    assert processor._socket is None
    assert processor._thread is None
    assert processor._running is None

    assert processor._stop_signal is None

    assert processor._messages is None
    assert processor._outgoing_queue is None

    assert processor._receive_buffer == ""


def test_processor_properties_require_execution_context() -> None:
    processor = create_processor()

    with pytest.raises(
        RuntimeError,
        match="Communication processor is not running.",
    ):
        processor.messages  # noqa: B018

    with pytest.raises(
        RuntimeError,
        match="Communication processor is not running.",
    ):
        processor.outgoing_queue  # noqa: B018


def test_processor_start() -> None:
    processor = create_processor()
    stop_signal = LocalEvent()

    socket_mock = Mock()
    thread = Mock()
    thread.is_alive.return_value = False

    with (
        patch(
            "tyrannis.backend.distributed.communication.spark_communication.socket.socket",
            return_value=socket_mock,
        ),
        patch(
            "tyrannis.backend.distributed.communication.spark_communication.Thread",
            return_value=thread,
        ),
    ):
        processor.start(stop_signal)

    socket_mock.connect.assert_called_once_with(
        ("127.0.0.1", 5000),
    )

    socket_mock.sendall.assert_called_once_with(
        f"{processor.STX}{FIRST_ISLAND}{processor.ETX}".encode(),
    )

    thread.start.assert_called_once_with()

    assert processor._stop_signal is stop_signal
    assert processor._socket is socket_mock
    assert processor._thread is thread

    assert processor._running is not None
    assert processor._running.is_set()

    assert processor._messages is not None
    assert processor._outgoing_queue is not None


def test_processor_start_while_running() -> None:
    processor = create_processor()
    processor._thread = Mock()
    processor._thread.is_alive.return_value = True

    with pytest.raises(
        RuntimeError,
        match="Communication is already running.",
    ):
        processor.start(LocalEvent())


def test_processor_stop() -> None:
    processor = create_processor()

    socket_mock = Mock()
    thread = Mock()

    processor._socket = socket_mock
    processor._thread = thread
    processor._running = Event()
    processor._running.set()

    processor._stop_signal = LocalEvent()
    processor._messages = Queue()
    processor._outgoing_queue = Queue()
    processor._receive_buffer = "partial"

    processor.stop()

    socket_mock.shutdown.assert_called_once()
    socket_mock.close.assert_called_once()
    thread.join.assert_called_once_with()

    assert processor._running is None
    assert processor._socket is None
    assert processor._thread is None

    assert processor._stop_signal is None
    assert processor._messages is None
    assert processor._outgoing_queue is None

    assert processor._receive_buffer == ""


def test_processor_stop_ignores_socket_shutdown_error() -> None:
    processor = create_processor()

    socket_mock = Mock()
    socket_mock.shutdown.side_effect = OSError

    processor._socket = socket_mock
    processor._running = Event()

    processor.stop()

    socket_mock.shutdown.assert_called_once()
    socket_mock.close.assert_called_once()

    assert processor._socket is None


def test_processor_stop_ignores_socket_close_error() -> None:
    processor = create_processor()

    socket_mock = Mock()
    socket_mock.close.side_effect = OSError

    processor._socket = socket_mock
    processor._running = Event()

    processor.stop()

    socket_mock.shutdown.assert_called_once()
    socket_mock.close.assert_called_once()

    assert processor._socket is None


def test_processor_properties_return_internal_queues() -> None:
    processor = create_processor()
    prepare_processor(processor)

    assert processor.messages is processor._messages
    assert processor.outgoing_queue is processor._outgoing_queue


def test_processor_send_pending_messages() -> None:
    processor = create_processor()
    prepare_processor(processor)

    socket_mock = Mock()
    processor._socket = socket_mock

    processor.outgoing_queue.put("message")
    processor.outgoing_queue.put(processor.STOP)

    processor._send_pending_messages()

    assert socket_mock.sendall.call_args_list == [
        (
            (f"{processor.STX}message{processor.ETX}".encode(),),
            {},
        ),
        (
            (processor.STOP.encode("utf-8"),),
            {},
        ),
    ]

    assert processor.outgoing_queue.empty()


def test_processor_send_pending_messages_without_socket() -> None:
    processor = create_processor()
    prepare_processor(processor)

    processor.outgoing_queue.put("message")

    processor._send_pending_messages()

    assert processor.outgoing_queue.get() == "message"


def test_processor_send_pending_messages_without_running() -> None:
    processor = create_processor()

    processor._socket = Mock()
    processor._running = None
    processor._stop_signal = LocalEvent()
    processor._outgoing_queue = Queue()

    processor._outgoing_queue.put("message")

    processor._send_pending_messages()

    assert processor._outgoing_queue.get() == "message"


def test_processor_send_pending_messages_without_queue() -> None:
    processor = create_processor()

    processor._socket = Mock()
    processor._running = Event()
    processor._stop_signal = LocalEvent()
    processor._outgoing_queue = None

    processor._send_pending_messages()


def test_processor_send_pending_messages_without_stop_signal() -> None:
    processor = create_processor()

    processor._socket = Mock()
    processor._running = Event()
    processor._stop_signal = None
    processor._outgoing_queue = Queue()

    processor._outgoing_queue.put("message")

    processor._send_pending_messages()

    assert processor._outgoing_queue.get() == "message"


def test_processor_send_pending_messages_on_error() -> None:
    processor = create_processor()
    prepare_processor(processor)

    socket_mock = Mock()
    socket_mock.sendall.side_effect = OSError

    processor._socket = socket_mock
    assert processor._running is not None
    processor._running.set()

    processor.outgoing_queue.put("message")

    processor._send_pending_messages()

    assert processor.outgoing_queue.get() == "message"
    assert processor._stop_signal is not None
    assert processor._stop_signal.is_set()


def test_processor_send_message_requires_socket() -> None:
    processor = create_processor()

    with pytest.raises(
        RuntimeError,
        match="Communication socket is not initialized.",
    ):
        processor._send_message("message")


def test_processor_send_message() -> None:
    processor = create_processor()

    socket_mock = Mock()
    processor._socket = socket_mock

    processor._send_message("message")

    socket_mock.sendall.assert_called_once_with(
        f"{processor.STX}message{processor.ETX}".encode(),
    )


def test_processor_process_buffer() -> None:
    processor = create_processor()
    prepare_processor(processor)

    processor._receive_buffer = (
        f"{processor.STX}message-1{processor.ETX}"
        f"{processor.STX}message-2{processor.ETX}"
    )

    processor._process_buffer()

    assert [
        processor.messages.get(),
        processor.messages.get(),
    ] == [
        "message-1",
        "message-2",
    ]

    assert processor._receive_buffer == ""


def test_processor_process_buffer_discards_data_before_frame() -> None:
    processor = create_processor()
    prepare_processor(processor)

    processor._receive_buffer = f"discard{processor.STX}message{processor.ETX}"

    processor._process_buffer()

    assert processor.messages.get() == "message"
    assert processor._receive_buffer == ""


def test_processor_process_buffer_preserves_incomplete_frame() -> None:
    processor = create_processor()
    prepare_processor(processor)

    processor._receive_buffer = f"{processor.STX}partial"

    processor._process_buffer()

    assert processor.messages.empty()
    assert processor._receive_buffer == f"{processor.STX}partial"

    processor._receive_buffer += processor.ETX

    processor._process_buffer()

    assert processor.messages.get() == "partial"
    assert processor._receive_buffer == ""


def test_processor_process_buffer_handles_stop() -> None:
    processor = create_processor()
    prepare_processor(processor)

    processor._receive_buffer = f"{processor.STOP}{processor.STX}message{processor.ETX}"

    processor._process_buffer()

    assert processor._stop_signal is not None
    assert processor._stop_signal.is_set()

    assert processor._receive_buffer == (f"{processor.STX}message{processor.ETX}")


def test_processor_process_buffer_sets_stop_signal_when_stop_is_first() -> None:
    processor = create_processor()
    prepare_processor(processor)

    processor._receive_buffer = processor.STOP

    processor._process_buffer()

    assert processor._stop_signal is not None
    assert processor._stop_signal.is_set()
    assert processor._receive_buffer == ""


def test_processor_process_buffer_processes_frame_before_stop() -> None:
    processor = create_processor()
    prepare_processor(processor)

    processor._receive_buffer = f"{processor.STX}message{processor.ETX}{processor.STOP}"

    processor._process_buffer()

    assert processor.messages.get() == "message"

    assert processor._stop_signal is not None
    assert processor._stop_signal.is_set()

    assert processor._receive_buffer == ""


def test_processor_process_buffer_requires_stop_signal() -> None:
    processor = create_processor()

    processor._stop_signal = None
    processor._receive_buffer = processor.STOP

    with pytest.raises(
        RuntimeError,
        match="Communication stop signal is not initialized.",
    ):
        processor._process_buffer()


# ---------------------------------------------------------------------------
# SparkCommunicationDriver
# ---------------------------------------------------------------------------


def test_driver_init() -> None:
    stop_signal = LocalEvent()

    driver = SparkCommunicationDriver(
        island_ids=[FIRST_ISLAND, SECOND_ISLAND],
        port=5000,
        stop_signal=stop_signal,
    )

    assert driver._island_ids == {
        FIRST_ISLAND,
        SECOND_ISLAND,
    }

    assert driver._port == 5000
    assert driver._stop_signal is stop_signal

    assert set(driver.incoming_queues) == {
        FIRST_ISLAND,
        SECOND_ISLAND,
    }

    assert set(driver.outgoing_queues) == {
        FIRST_ISLAND,
        SECOND_ISLAND,
    }

    assert driver._connections == {}
    assert driver._connection_identifications == {}
    assert driver._receive_buffers == {}

    assert driver._server_socket is None
    assert driver._selector is None
    assert driver._thread is None

    assert not driver._running.is_set()
    assert not driver._stop_sent


def test_driver_start() -> None:
    driver = create_driver()

    server_socket = Mock()
    selector = Mock()
    thread = Mock()

    thread.is_alive.return_value = False

    with (
        patch(
            "tyrannis.backend.distributed.communication.spark_communication.socket.socket",
            return_value=server_socket,
        ),
        patch(
            "tyrannis.backend.distributed.communication.spark_communication.selectors.DefaultSelector",
            return_value=selector,
        ),
        patch(
            "tyrannis.backend.distributed.communication.spark_communication.Thread",
            return_value=thread,
        ),
    ):
        driver.start()

    server_socket.setsockopt.assert_called_once_with(
        socket.SOL_SOCKET,
        socket.SO_REUSEADDR,
        1,
    )

    server_socket.bind.assert_called_once_with(
        ("", 5000),
    )

    server_socket.listen.assert_called_once_with()
    server_socket.setblocking.assert_called_once_with(False)

    selector.register.assert_called_once_with(
        server_socket,
        selectors.EVENT_READ,
    )

    thread.start.assert_called_once_with()

    assert driver._server_socket is server_socket
    assert driver._selector is selector
    assert driver._thread is thread

    assert driver._running.is_set()
    assert not driver._stop_sent


def test_driver_start_while_running() -> None:
    driver = create_driver()

    driver._thread = Mock()
    driver._thread.is_alive.return_value = True

    with pytest.raises(
        RuntimeError,
        match="Communication is already running.",
    ):
        driver.start()


def test_driver_stop() -> None:
    driver = create_driver()

    server_socket = Mock()
    connection = Mock()
    selector = Mock()
    thread = Mock()

    driver._server_socket = server_socket
    driver._connections = {
        FIRST_ISLAND: connection,
    }
    driver._connection_identifications = {
        connection: FIRST_ISLAND,
    }
    driver._receive_buffers = {
        connection: "partial",
    }
    driver._selector = selector
    driver._thread = thread

    driver._running.set()
    driver._stop_sent = True

    driver.stop()

    server_socket.close.assert_called_once_with()
    thread.join.assert_called_once_with()
    connection.close.assert_called_once_with()
    selector.close.assert_called_once_with()

    assert not driver._running.is_set()

    assert driver._server_socket is None
    assert driver._thread is None

    assert driver._connections == {}
    assert driver._connection_identifications == {}
    assert driver._receive_buffers == {}

    assert driver._selector is None
    assert not driver._stop_sent


def test_driver_stop_without_server_socket() -> None:
    driver = create_driver()

    thread = Mock()
    driver._thread = thread

    driver.stop()

    thread.join.assert_called_once_with()

    assert driver._thread is None
    assert driver._server_socket is None


def test_driver_properties_return_internal_queues() -> None:
    driver = create_driver()

    assert driver.incoming_queues is driver._incoming_queues
    assert driver.outgoing_queues is driver._outgoing_queues


def test_driver_accept_connection() -> None:
    driver = create_driver()

    server_socket = Mock()
    selector = Mock()
    connection = Mock()

    server_socket.accept.return_value = (
        connection,
        ("127.0.0.1", 12345),
    )

    driver._server_socket = server_socket
    driver._selector = selector

    driver._accept_connection()

    server_socket.accept.assert_called_once_with()
    connection.setblocking.assert_called_once_with(False)

    selector.register.assert_called_once_with(
        connection,
        selectors.EVENT_READ,
    )

    assert driver._receive_buffers[connection] == ""


def test_driver_accept_connection_ignores_accept_error() -> None:
    driver = create_driver()

    server_socket = Mock()
    selector = Mock()

    server_socket.accept.side_effect = OSError

    driver._server_socket = server_socket
    driver._selector = selector

    driver._accept_connection()

    server_socket.accept.assert_called_once_with()
    selector.register.assert_not_called()


def test_driver_accept_connection_without_server_socket() -> None:
    driver = create_driver()

    driver._server_socket = None
    driver._selector = Mock()

    driver._accept_connection()


def test_driver_accept_connection_without_selector() -> None:
    driver = create_driver()

    driver._server_socket = Mock()
    driver._selector = None

    driver._accept_connection()


def test_driver_receive_from_connection_routes_data() -> None:
    driver = create_driver()

    connection = Mock()

    driver._receive_buffers[connection] = ""
    driver._connection_identifications[connection] = FIRST_ISLAND

    connection.recv.return_value = f"{driver.STX}message{driver.ETX}".encode()

    driver._receive_from_connection(connection)

    connection.recv.assert_called_once_with(4096)

    assert driver.incoming_queues[FIRST_ISLAND].get() == "message"
    assert driver._receive_buffers[connection] == ""


def test_driver_receive_from_connection_handles_disconnect() -> None:
    driver = create_driver()

    connection = Mock()
    connection.recv.return_value = b""

    driver._receive_buffers[connection] = ""
    driver._running.set()

    driver._receive_from_connection(connection)

    assert driver._stop_signal.is_set()
    connection.close.assert_called_once_with()


def test_driver_receive_from_connection_handles_recv_error() -> None:
    driver = create_driver()

    connection = Mock()
    connection.recv.side_effect = OSError

    driver._receive_buffers[connection] = ""
    driver._running.set()

    with patch.object(
        driver,
        "_close_connection",
    ) as close_connection:
        driver._receive_from_connection(connection)

    close_connection.assert_called_once_with(connection)


def test_driver_receive_from_connection_handles_invalid_utf8() -> None:
    driver = create_driver()

    connection = Mock()
    connection.recv.return_value = b"\xff"

    driver._receive_buffers[connection] = ""

    with patch.object(
        driver,
        "_close_connection",
    ) as close_connection:
        driver._receive_from_connection(connection)

    close_connection.assert_called_once_with(connection)
    assert driver._stop_signal.is_set()


def test_driver_process_buffer_registers_connection() -> None:
    driver = create_driver()

    connection = Mock()

    driver._receive_buffers[connection] = f"{driver.STX}{FIRST_ISLAND}{driver.ETX}"

    driver._process_driver_buffer(connection)

    assert driver._connection_identifications[connection] == FIRST_ISLAND
    assert driver._connections[FIRST_ISLAND] is connection
    assert driver._receive_buffers[connection] == ""


def test_driver_process_buffer_routes_registered_messages() -> None:
    driver = create_driver()

    connection = Mock()

    driver._connection_identifications[connection] = FIRST_ISLAND
    driver._connections[FIRST_ISLAND] = connection

    driver._receive_buffers[connection] = (
        f"{driver.STX}message-1{driver.ETX}{driver.STX}message-2{driver.ETX}"
    )

    driver._process_driver_buffer(connection)

    assert [
        driver.incoming_queues[FIRST_ISLAND].get(),
        driver.incoming_queues[FIRST_ISLAND].get(),
    ] == [
        "message-1",
        "message-2",
    ]

    assert driver.incoming_queues[SECOND_ISLAND].empty()
    assert driver._receive_buffers[connection] == ""


def test_driver_process_buffer_preserves_incomplete_frame() -> None:
    driver = create_driver()

    connection = Mock()

    driver._connection_identifications[connection] = FIRST_ISLAND
    driver._connections[FIRST_ISLAND] = connection

    driver._receive_buffers[connection] = f"{driver.STX}partial"

    driver._process_driver_buffer(connection)

    assert driver.incoming_queues[FIRST_ISLAND].empty()
    assert driver._receive_buffers[connection] == f"{driver.STX}partial"

    driver._receive_buffers[connection] += driver.ETX

    driver._process_driver_buffer(connection)

    assert driver.incoming_queues[FIRST_ISLAND].get() == "partial"
    assert driver._receive_buffers[connection] == ""


def test_driver_process_buffer_discards_data_before_frame() -> None:
    driver = create_driver()

    connection = Mock()

    driver._connection_identifications[connection] = FIRST_ISLAND
    driver._connections[FIRST_ISLAND] = connection

    driver._receive_buffers[connection] = f"discard{driver.STX}message{driver.ETX}"

    driver._process_driver_buffer(connection)

    assert driver.incoming_queues[FIRST_ISLAND].get() == "message"
    assert driver._receive_buffers[connection] == ""


def test_driver_process_buffer_handles_stop() -> None:
    driver = create_driver()

    connection = Mock()

    driver._receive_buffers[connection] = (
        f"{driver.STOP}{driver.STX}message{driver.ETX}"
    )

    driver._process_driver_buffer(connection)

    assert driver._stop_signal.is_set()
    assert driver._receive_buffers[connection] == (f"{driver.STX}message{driver.ETX}")


def test_driver_process_buffer_ignores_unknown_connection_buffer() -> None:
    driver = create_driver()

    connection = Mock()

    driver._process_driver_buffer(connection)

    assert connection not in driver._receive_buffers


def test_driver_register_connection_rejects_unknown_island() -> None:
    driver = create_driver()

    connection = Mock()

    driver._receive_buffers[connection] = "partial"

    result = driver._register_connection(
        connection,
        "unknown-island",
    )

    assert result is True
    assert driver._stop_signal.is_set()
    assert connection not in driver._receive_buffers


def test_driver_register_connection_rejects_duplicate_island() -> None:
    driver = create_driver()

    existing_connection = Mock()
    new_connection = Mock()

    driver._connections[FIRST_ISLAND] = existing_connection
    driver._receive_buffers[new_connection] = ""

    result = driver._register_connection(
        new_connection,
        FIRST_ISLAND,
    )

    assert result is True
    assert driver._stop_signal.is_set()
    assert new_connection not in driver._receive_buffers


def test_driver_register_connection_returns_false_for_registered_connection() -> None:
    driver = create_driver()

    connection = Mock()

    driver._connection_identifications[connection] = FIRST_ISLAND

    result = driver._register_connection(
        connection,
        "message",
    )

    assert result is False
    assert not driver._stop_signal.is_set()


def test_driver_route_message_requires_identification() -> None:
    driver = create_driver()

    connection = Mock()

    with patch.object(
        driver,
        "_close_connection",
    ) as close_connection:
        driver._route_message(
            connection,
            "message",
        )

    close_connection.assert_called_once_with(connection)
    assert driver._stop_signal.is_set()


def test_driver_send_pending_messages() -> None:
    driver = create_driver()

    connection = Mock()

    driver._connections[FIRST_ISLAND] = connection

    driver.outgoing_queues[FIRST_ISLAND].put("message")
    driver.outgoing_queues[FIRST_ISLAND].put(driver.STOP)

    driver._send_pending_messages()

    assert connection.sendall.call_args_list == [
        (
            (f"{driver.STX}message{driver.ETX}".encode(),),
            {},
        ),
        (
            (f"{driver.STX}{driver.STOP}{driver.ETX}".encode(),),
            {},
        ),
    ]

    assert driver.outgoing_queues[FIRST_ISLAND].empty()


def test_driver_send_pending_messages_skips_disconnected_island() -> None:
    driver = create_driver()

    driver.outgoing_queues[FIRST_ISLAND].put("message")

    driver._send_pending_messages()

    assert driver.outgoing_queues[FIRST_ISLAND].get() == "message"


def test_driver_send_pending_messages_on_error() -> None:
    driver = create_driver()

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

    close_connection.assert_called_once_with(connection)

    assert not driver._stop_signal.is_set()


def test_driver_send_pending_messages_on_error_while_stopped() -> None:
    driver = create_driver()

    connection = Mock()
    connection.sendall.side_effect = OSError

    driver._connections[FIRST_ISLAND] = connection
    driver.outgoing_queues[FIRST_ISLAND].put("message")

    driver._running.clear()

    with patch.object(
        driver,
        "_close_connection",
    ) as close_connection:
        driver._send_pending_messages()

    assert driver.outgoing_queues[FIRST_ISLAND].get() == "message"

    close_connection.assert_called_once_with(connection)
    assert not driver._stop_signal.is_set()


def test_driver_send_stop_signal() -> None:
    driver = create_driver()

    first = Mock()
    second = Mock()

    driver._connections = {
        FIRST_ISLAND: first,
        SECOND_ISLAND: second,
    }

    driver._send_stop_signal()

    first.sendall.assert_called_once_with(
        driver.STOP.encode("utf-8"),
    )

    second.sendall.assert_called_once_with(
        driver.STOP.encode("utf-8"),
    )

    assert driver._stop_sent


def test_driver_send_stop_signal_is_idempotent() -> None:
    driver = create_driver()

    connection = Mock()

    driver._connections[FIRST_ISLAND] = connection

    driver._send_stop_signal()
    driver._send_stop_signal()

    connection.sendall.assert_called_once_with(
        driver.STOP.encode("utf-8"),
    )


def test_driver_send_stop_signal_ignores_send_error() -> None:
    driver = create_driver()

    connection = Mock()
    connection.sendall.side_effect = OSError

    driver._connections[FIRST_ISLAND] = connection

    driver._send_stop_signal()

    connection.sendall.assert_called_once_with(
        driver.STOP.encode("utf-8"),
    )

    assert driver._stop_sent


def test_driver_close_connection_ignores_unregister_error() -> None:
    driver = create_driver()

    connection = Mock()
    selector = Mock()

    selector.unregister.side_effect = KeyError
    driver._selector = selector

    driver._close_connection(connection)

    selector.unregister.assert_called_once_with(connection)
    connection.close.assert_called_once_with()


def test_driver_close_connection_ignores_value_error() -> None:
    driver = create_driver()

    connection = Mock()
    selector = Mock()

    selector.unregister.side_effect = ValueError
    driver._selector = selector

    driver._close_connection(connection)

    connection.close.assert_called_once_with()


def test_driver_close_connection_ignores_os_error_from_unregister() -> None:
    driver = create_driver()

    connection = Mock()
    selector = Mock()

    selector.unregister.side_effect = OSError
    driver._selector = selector

    driver._close_connection(connection)

    connection.close.assert_called_once_with()


def test_driver_close_connection_ignores_close_error() -> None:
    driver = create_driver()

    connection = Mock()
    connection.close.side_effect = OSError

    driver._selector = Mock()

    driver._close_connection(connection)


def test_driver_close_connection_removes_island_mapping() -> None:
    driver = create_driver()

    connection = Mock()

    driver._selector = None
    driver._connections[FIRST_ISLAND] = connection
    driver._connection_identifications[connection] = FIRST_ISLAND
    driver._receive_buffers[connection] = "partial"

    driver._close_connection(connection)

    assert FIRST_ISLAND not in driver._connections
    assert connection not in driver._connection_identifications
    assert connection not in driver._receive_buffers


def test_driver_close_connection_does_not_remove_replaced_connection() -> None:
    driver = create_driver()

    connection = Mock()
    replacement = Mock()

    driver._connections[FIRST_ISLAND] = replacement
    driver._connection_identifications[connection] = FIRST_ISLAND
    driver._receive_buffers[connection] = "partial"

    driver._close_connection(connection)

    assert driver._connections[FIRST_ISLAND] is replacement
    assert connection not in driver._connection_identifications
    assert connection not in driver._receive_buffers


def test_driver_close_connection_without_island_mapping() -> None:
    driver = create_driver()

    connection = Mock()
    driver._receive_buffers[connection] = "partial"

    driver._close_connection(connection)

    assert connection not in driver._receive_buffers
    connection.close.assert_called_once_with()


def test_driver_communication_loop_requires_selector() -> None:
    driver = create_driver()

    driver._selector = None
    driver._server_socket = Mock()

    with pytest.raises(
        RuntimeError,
        match="Selector is not initialized.",
    ):
        driver._communication_loop()


def test_driver_communication_loop_requires_server_socket() -> None:
    driver = create_driver()

    driver._selector = Mock()
    driver._server_socket = None

    with pytest.raises(
        RuntimeError,
        match="Server socket is not initialized.",
    ):
        driver._communication_loop()


def test_driver_communication_loop_stops_when_stop_signal_is_set() -> None:
    driver = create_driver()

    driver._selector = Mock()
    driver._server_socket = Mock()

    driver._stop_signal.set()

    driver._communication_loop()

    driver._selector.select.assert_not_called()


def test_driver_communication_loop_accepts_connection() -> None:
    driver = create_driver()

    selector = Mock()
    server_socket = Mock()

    driver._selector = selector
    driver._server_socket = server_socket
    driver._running.set()

    selector.select.side_effect = [
        [
            (
                Mock(fileobj=server_socket),
                selectors.EVENT_READ,
            ),
        ],
    ]

    def stop_after_accept() -> None:
        driver._running.clear()

    with (
        patch.object(
            driver,
            "_accept_connection",
            side_effect=stop_after_accept,
        ) as accept_connection,
        patch.object(
            driver,
            "_send_pending_messages",
        ),
    ):
        driver._communication_loop()

    accept_connection.assert_called_once_with()
    driver._send_stop_signal()


def test_driver_communication_loop_receives_connection_data() -> None:
    driver = create_driver()

    selector = Mock()
    server_socket = Mock()
    connection = Mock()

    driver._selector = selector
    driver._server_socket = server_socket
    driver._running.set()

    selector.select.side_effect = [
        [
            (
                Mock(fileobj=connection),
                selectors.EVENT_READ,
            ),
        ],
    ]

    def stop_after_receive(
        received_connection: socket.socket,
    ) -> None:
        assert received_connection is connection
        driver._running.clear()

    with (
        patch.object(
            driver,
            "_receive_from_connection",
            side_effect=stop_after_receive,
        ) as receive_from_connection,
        patch.object(
            driver,
            "_send_pending_messages",
        ),
    ):
        driver._communication_loop()

    receive_from_connection.assert_called_once_with(connection)
    driver._send_stop_signal()
