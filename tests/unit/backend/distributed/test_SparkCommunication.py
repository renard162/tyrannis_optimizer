import selectors
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
    processor._wait_signal = LocalEvent()
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

    assert processor._wait_signal is None
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
    wait_signal = LocalEvent()
    stop_signal = LocalEvent()

    socket = Mock()
    thread = Mock()
    thread.is_alive.return_value = False

    with (
        patch(
            "tyrannis.backend.distributed.communication.spark_communication.socket.socket",
            return_value=socket,
        ),
        patch(
            "tyrannis.backend.distributed.communication.spark_communication.Thread",
            return_value=thread,
        ),
    ):
        processor.start(wait_signal, stop_signal)

    socket.connect.assert_called_once_with(("127.0.0.1", 5000))
    socket.sendall.assert_called_once_with(
        f"{processor.STX}{FIRST_ISLAND}{processor.ETX}".encode(),
    )
    thread.start.assert_called_once_with()

    assert processor._wait_signal is wait_signal
    assert processor._stop_signal is stop_signal
    assert processor._socket is socket
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
        processor.start(LocalEvent(), LocalEvent())


def test_processor_stop() -> None:
    processor = create_processor()

    socket = Mock()
    thread = Mock()

    processor._socket = socket
    processor._thread = thread
    processor._running = Event()
    processor._running.set()

    processor._wait_signal = LocalEvent()
    processor._stop_signal = LocalEvent()
    processor._messages = Queue()
    processor._outgoing_queue = Queue()
    processor._receive_buffer = "partial"

    processor.stop()

    socket.shutdown.assert_called_once()
    socket.close.assert_called_once()
    thread.join.assert_called_once_with()

    assert processor._running is None
    assert processor._socket is None
    assert processor._thread is None

    assert processor._wait_signal is None
    assert processor._stop_signal is None

    assert processor._messages is None
    assert processor._outgoing_queue is None

    assert processor._receive_buffer == ""


def test_processor_properties_return_internal_queues() -> None:
    processor = create_processor()
    prepare_processor(processor)

    assert processor.messages is processor._messages
    assert processor.outgoing_queue is processor._outgoing_queue


def test_processor_send_pending_messages() -> None:
    processor = create_processor()
    prepare_processor(processor)

    socket = Mock()
    processor._socket = socket

    processor.outgoing_queue.put("message")
    processor.outgoing_queue.put(processor.STOP)

    processor._send_pending_messages()

    assert socket.sendall.call_args_list == [
        (
            (f"{processor.STX}message{processor.ETX}".encode(),),
            {},
        ),
        (
            (processor.STOP.encode(),),
            {},
        ),
    ]

    assert processor.outgoing_queue.empty()


def test_processor_send_pending_messages_on_error() -> None:
    processor = create_processor()
    prepare_processor(processor)

    socket = Mock()
    socket.sendall.side_effect = OSError

    processor._socket = socket
    processor._running.set()  # type: ignore

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

    socket = Mock()
    processor._socket = socket

    processor._send_message("message")

    socket.sendall.assert_called_once_with(
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


def test_processor_process_buffer_preserves_incomplete_frame() -> None:
    processor = create_processor()
    prepare_processor(processor)

    processor._receive_buffer = f"{processor.STX}partial"

    processor._process_buffer()

    assert processor.messages.empty()
    assert processor._receive_buffer == f"{processor.STX}partial"

    processor._receive_buffer += f"{processor.ETX}"

    processor._process_buffer()

    assert processor.messages.get() == "partial"
    assert processor._receive_buffer == ""


def test_processor_process_buffer_sets_wait_signal_for_empty_frame() -> None:
    processor = create_processor()
    prepare_processor(processor)

    processor._receive_buffer = f"{processor.STX}{processor.ETX}"

    processor._process_buffer()

    assert processor._wait_signal is not None
    assert processor._wait_signal.is_set()
    assert processor.messages.empty()


def test_processor_process_buffer_sets_stop_signal() -> None:
    processor = create_processor()
    prepare_processor(processor)

    processor._receive_buffer = processor.STOP

    processor._process_buffer()

    assert processor._stop_signal is not None
    assert processor._stop_signal.is_set()
    assert processor._receive_buffer == ""


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

    server_socket.setsockopt.assert_called_once()
    server_socket.bind.assert_called_once_with(("", 5000))
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
    assert driver._receive_buffers == {}

    assert driver._selector is None
    assert not driver._stop_sent


def test_driver_properties_return_internal_queues() -> None:
    driver = create_driver()

    assert driver.incoming_queues is driver._incoming_queues
    assert driver.outgoing_queues is driver._outgoing_queues


def test_driver_extract_messages() -> None:
    driver = create_driver()
    connection = Mock()

    driver._receive_buffers[connection] = (
        f"discard{driver.STX}message-1{driver.ETX}{driver.STX}message-2{driver.ETX}"
    )

    messages = driver._extract_messages(connection)

    assert messages == [
        "message-1",
        "message-2",
    ]

    assert driver._receive_buffers[connection] == ""


def test_driver_extract_messages_preserves_incomplete_frame() -> None:
    driver = create_driver()
    connection = Mock()

    driver._receive_buffers[connection] = f"{driver.STX}partial"

    messages = driver._extract_messages(connection)

    assert messages == []
    assert driver._receive_buffers[connection] == (f"{driver.STX}partial")

    driver._receive_buffers[connection] += driver.ETX

    assert driver._extract_messages(connection) == [
        "partial",
    ]

    assert driver._receive_buffers[connection] == ""


def test_driver_extract_messages_handles_stop() -> None:
    driver = create_driver()
    connection = Mock()

    driver._receive_buffers[connection] = (
        f"{driver.STOP}{driver.STX}message{driver.ETX}"
    )

    messages = driver._extract_messages(connection)

    assert messages == []
    assert driver._stop_signal.is_set()
    assert driver._receive_buffers[connection] == (f"{driver.STX}message{driver.ETX}")


def test_driver_handle_stop() -> None:
    driver = create_driver()

    driver._handle_stop()

    assert driver._stop_signal.is_set()


def test_driver_receive_message_routes_messages() -> None:
    driver = create_driver()
    connection = Mock()

    with (
        patch.object(
            driver,
            "_receive_data",
            return_value=True,
        ),
        patch.object(
            driver,
            "_extract_messages",
            return_value=[
                "message-1",
                "message-2",
            ],
        ),
    ):
        driver._receive_message(
            connection,
            FIRST_ISLAND,
        )

    assert [
        driver.incoming_queues[FIRST_ISLAND].get(),
        driver.incoming_queues[FIRST_ISLAND].get(),
    ] == [
        "message-1",
        "message-2",
    ]

    assert driver.incoming_queues[SECOND_ISLAND].empty()


def test_driver_send_pending_messages() -> None:
    driver = create_driver()
    connection = Mock()

    driver._connections[FIRST_ISLAND] = connection

    driver.outgoing_queues[FIRST_ISLAND].put(
        "message",
    )
    driver.outgoing_queues[FIRST_ISLAND].put(
        driver.STOP,
    )

    driver._send_pending_messages()

    assert connection.sendall.call_args_list == [
        (
            (f"{driver.STX}message{driver.ETX}".encode(),),
            {},
        ),
        (
            (driver.STOP.encode(),),
            {},
        ),
    ]

    assert driver.outgoing_queues[FIRST_ISLAND].empty()


def test_driver_send_pending_messages_on_error() -> None:
    driver = create_driver()
    connection = Mock()

    connection.sendall.side_effect = OSError

    driver._connections[FIRST_ISLAND] = connection
    driver.outgoing_queues[FIRST_ISLAND].put(
        "message",
    )

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


def test_driver_send_stop_to_all() -> None:
    driver = create_driver()

    first = Mock()
    second = Mock()

    driver._connections = {
        FIRST_ISLAND: first,
        SECOND_ISLAND: second,
    }

    driver._send_stop_to_all()

    first.sendall.assert_called_once_with(
        driver.STOP.encode(),
    )
    second.sendall.assert_called_once_with(
        driver.STOP.encode(),
    )


def test_driver_send_stop_to_all_handles_send_error() -> None:
    driver = create_driver()
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
    driver = create_driver()

    connection = Mock()
    selector = Mock()

    selector.unregister.side_effect = KeyError
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
    driver._receive_buffers[connection] = "partial"

    driver._close_connection(
        connection,
        FIRST_ISLAND,
    )

    assert FIRST_ISLAND not in driver._connections
    assert connection not in driver._receive_buffers
