"""Deterministic transport contracts for Spark TCP communication."""

# The transport state machine exposes its testable contract through private methods.
# pyright: reportPrivateUsage=false

import selectors
import socket
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Never, final, override

import pytest

from tyrannis.backend.distributed.communication import spark_communication as transport
from tyrannis.backend.distributed.communication.spark_communication import (
    SparkCommunicationDriver,
    SparkCommunicationProcessor,
)
from tyrannis.core.signals import LocalEvent


@final
class ConnectionDouble(socket.socket):
    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self, replies: Sequence[bytes | Exception] | None = None
    ) -> None:
        # Do not initialize the C socket: this double never allocates an OS socket.
        self.sent: list[bytes] = []
        self.replies = list(replies or [])
        self.connected_to: tuple[str, int] | None = None
        self.timeout_value: float | None = None
        self.blocking: bool | None = None
        self.shutdown_calls: list[int] = []
        self.closed = False
        self.fail_send = False
        self.fail_shutdown = False
        self.fail_close = False

    @override
    def connect(self, address: object) -> None:
        assert isinstance(address, tuple)
        assert isinstance(address[0], str) and isinstance(address[1], int)
        self.connected_to = address

    @override
    def sendall(self, data: object, flags: int = 0) -> None:
        assert isinstance(data, bytes)
        assert flags == 0
        if self.fail_send:
            raise OSError("send failed")
        self.sent.append(data)

    @override
    def recv(self, size: int, flags: int = 0) -> bytes:
        assert size == 4096
        assert flags == 0
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    @override
    def settimeout(self, value: float | None) -> None:
        self.timeout_value = value

    @override
    def setblocking(self, value: bool) -> None:
        self.blocking = value

    @override
    def shutdown(self, how: int) -> None:
        self.shutdown_calls.append(how)
        if self.fail_shutdown:
            raise OSError("shutdown failed")

    @override
    def close(self) -> None:
        self.closed = True
        if self.fail_close:
            raise OSError("close failed")


@final
class ThreadDouble:
    def __init__(self, *, target: Callable[[], None], name: str, daemon: bool) -> None:
        self.target = target
        self.name = name
        self.daemon = daemon
        self.started = False
        self.joined = False

    def start(self) -> None:
        self.started = True

    def is_alive(self) -> bool:
        return self.started and not self.joined

    def join(self) -> None:
        self.joined = True


@final
class ServerDouble:
    def __init__(self) -> None:
        self.options: list[tuple[int, int, int]] = []
        self.bound_to: tuple[str, int] | None = None
        self.listening = False
        self.blocking: bool | None = None
        self.closed = False
        self.fail_close = False
        self.accept_reply: ConnectionDouble | Exception | None = None

    def setsockopt(self, level: int, option: int, value: int) -> None:
        self.options.append((level, option, value))

    def bind(self, address: tuple[str, int]) -> None:
        self.bound_to = address

    def listen(self) -> None:
        self.listening = True

    def setblocking(self, value: bool) -> None:
        self.blocking = value

    def accept(self) -> tuple[ConnectionDouble, tuple[str, int]]:
        reply = self.accept_reply
        if isinstance(reply, Exception):
            raise reply
        assert reply is not None
        return reply, ("192.0.2.2", 1234)

    def close(self) -> None:
        self.closed = True
        if self.fail_close:
            raise OSError("server close failed")


@final
class SelectorDouble:
    def __init__(self) -> None:
        self.registrations: list[tuple[object, int]] = []
        self.unregistered: list[object] = []
        self.events: list[tuple[SimpleNamespace, int]] = []
        self.on_select: Callable[[], None] | None = None
        self.closed = False

    def register(self, fileobj: object, events: int) -> None:
        self.registrations.append((fileobj, events))

    def unregister(self, fileobj: object) -> None:
        self.unregistered.append(fileobj)

    def select(self, timeout: float) -> list[tuple[SimpleNamespace, int]]:
        assert timeout == 0.1
        if self.on_select is not None:
            self.on_select()
        return self.events

    def close(self) -> None:
        self.closed = True


def _processor_runtime(
    monkeypatch: pytest.MonkeyPatch, connection: ConnectionDouble
) -> tuple[SparkCommunicationProcessor, ThreadDouble]:
    threads: list[ThreadDouble] = []

    def make_thread(
        *, target: Callable[[], None], name: str, daemon: bool
    ) -> ThreadDouble:
        thread = ThreadDouble(target=target, name=name, daemon=daemon)
        threads.append(thread)
        return thread

    def make_socket(family: int, kind: int) -> ConnectionDouble:
        assert (family, kind) == (socket.AF_INET, socket.SOCK_STREAM)
        return connection

    monkeypatch.setattr(socket, "socket", make_socket)
    monkeypatch.setattr(transport, "Thread", make_thread)
    processor = SparkCommunicationProcessor("192.0.2.1", 1234, "island:1")
    processor.start()
    return processor, threads[0]


def _driver_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[SparkCommunicationDriver, ServerDouble, SelectorDouble, ThreadDouble]:
    server = ServerDouble()
    selector = SelectorDouble()
    threads: list[ThreadDouble] = []

    def make_socket(family: int, kind: int) -> ServerDouble:
        assert (family, kind) == (socket.AF_INET, socket.SOCK_STREAM)
        return server

    def make_thread(
        *, target: Callable[[], None], name: str, daemon: bool
    ) -> ThreadDouble:
        thread = ThreadDouble(target=target, name=name, daemon=daemon)
        threads.append(thread)
        return thread

    monkeypatch.setattr(socket, "socket", make_socket)
    monkeypatch.setattr(selectors, "DefaultSelector", lambda: selector)
    monkeypatch.setattr(transport, "Thread", make_thread)
    driver = SparkCommunicationDriver(["island:1", "island:2"], 1234)
    driver.start()
    return driver, server, selector, threads[0]


def test_processor_constructor_is_inactive_and_properties_require_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_socket(family: int, kind: int) -> Never:
        pytest.fail(f"socket created early: {(family, kind)}")

    def unexpected_thread(
        *, target: Callable[[], None], name: str, daemon: bool
    ) -> Never:
        pytest.fail(f"thread created early: {(target, name, daemon)}")

    monkeypatch.setattr(socket, "socket", unexpected_socket)
    monkeypatch.setattr(transport, "Thread", unexpected_thread)
    processor = SparkCommunicationProcessor("192.0.2.1", 1234, "island:1")

    with pytest.raises(RuntimeError, match="not running"):
        _ = processor.messages
    with pytest.raises(RuntimeError, match="not running"):
        _ = processor.outgoing_queue

    first = LocalEvent()
    second = LocalEvent()
    processor.set_message_signal(first)
    assert processor._message_signal is first
    processor.set_message_signal(second)
    assert processor._message_signal is second
    processor.set_message_signal(None)
    assert processor._message_signal is None


def test_processor_start_connects_and_sends_framed_handshake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = ConnectionDouble()
    processor, thread = _processor_runtime(monkeypatch, connection)

    assert connection.connected_to == ("192.0.2.1", 1234)
    assert connection.sent == [b"\x02island:1\x03"]
    assert processor._running is not None and processor._running.is_set()
    assert processor._send_lock is not None
    assert processor.messages is processor.messages
    assert processor.outgoing_queue is processor.outgoing_queue
    assert thread.target == processor._receive_loop
    assert (thread.name, thread.daemon, thread.started) == (
        "processor-communication",
        True,
        True,
    )
    with pytest.raises(RuntimeError, match="already running"):
        processor.start()


def test_processor_outgoing_put_sends_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = ConnectionDouble()
    processor, _ = _processor_runtime(monkeypatch, connection)

    processor.outgoing_queue.put("opaque")

    assert connection.sent == [b"\x02island:1\x03", b"\x02opaque\x03"]
    assert processor.outgoing_queue.empty()


def test_processor_send_failure_requeues_without_recursive_flush(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = ConnectionDouble()
    processor, _ = _processor_runtime(monkeypatch, connection)
    connection.fail_send = True

    processor.outgoing_queue.put("retry")

    assert processor.outgoing_queue.get_nowait() == "retry"
    assert processor.outgoing_queue.empty()
    assert connection.sent == [b"\x02island:1\x03"]


@pytest.mark.parametrize(
    ("chunks", "expected", "remainder"),
    [
        (["garbage\x02one\x03"], ["one"], ""),
        (["garbage"], [], ""),
        (["\x02part", "ial\x03"], ["partial"], ""),
        (["\x02one\x03\x02two\x03\x02part"], ["one", "two"], "\x02part"),
    ],
    ids=[
        "garbage-prefix",
        "no-stx",
        "fragmented",
        "multiple-with-remainder",
    ],
)
def test_processor_buffer_publishes_only_complete_frames(
    monkeypatch: pytest.MonkeyPatch,
    chunks: list[str],
    expected: list[str],
    remainder: str,
) -> None:
    processor, _ = _processor_runtime(monkeypatch, ConnectionDouble())
    signal = LocalEvent()
    processor.set_message_signal(signal)

    for chunk in chunks:
        processor._receive_buffer += chunk
        processor._process_buffer()

    assert [processor.messages.get_nowait() for _ in expected] == expected
    assert processor.messages.empty()
    assert processor._receive_buffer == remainder
    assert signal.is_set() is bool(expected)


def test_processor_receive_loop_survives_timeout_and_flushes_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = ConnectionDouble([TimeoutError(), b"\x02arrived\x03", b""])
    processor, _ = _processor_runtime(monkeypatch, connection)
    send_lock = processor._send_lock
    monkeypatch.setattr(processor, "_send_lock", None)
    processor.outgoing_queue.put("queued")
    monkeypatch.setattr(processor, "_send_lock", send_lock)

    processor._receive_loop()

    assert connection.timeout_value == 0.1
    assert processor.messages.get_nowait() == "arrived"
    assert connection.sent == [b"\x02island:1\x03", b"\x02queued\x03"]


def test_processor_receive_loop_exits_when_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = ConnectionDouble([b"\x02unread\x03"])
    processor, _ = _processor_runtime(monkeypatch, connection)
    running = processor._running
    assert running is not None
    running.clear()

    processor._receive_loop()

    assert connection.replies == [b"\x02unread\x03"]
    assert processor.messages.empty()


@pytest.mark.parametrize(
    "reply", [OSError("recv failed"), b"", b"\xff"], ids=["os-error", "eof", "utf8"]
)
def test_processor_receive_loop_stops_on_terminal_input(
    monkeypatch: pytest.MonkeyPatch, reply: bytes | Exception
) -> None:
    connection = ConnectionDouble([reply])
    processor, _ = _processor_runtime(monkeypatch, connection)

    processor._receive_loop()

    assert processor.messages.empty()
    assert connection.replies == []


def test_processor_stop_releases_runtime_despite_socket_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = ConnectionDouble()
    processor, thread = _processor_runtime(monkeypatch, connection)
    running = processor._running
    processor.set_message_signal(LocalEvent())
    processor._receive_buffer = "\x02partial"
    connection.fail_shutdown = True
    connection.fail_close = True

    processor.stop()

    assert running is not None and not running.is_set()
    assert connection.shutdown_calls == [socket.SHUT_RDWR]
    assert connection.closed
    assert thread.joined
    assert processor._socket is None
    assert processor._thread is None
    assert processor._running is None
    assert processor._send_lock is None
    assert processor._receive_buffer == ""
    assert processor._message_signal is None
    with pytest.raises(RuntimeError, match="not running"):
        _ = processor.messages
    with pytest.raises(RuntimeError, match="not running"):
        _ = processor.outgoing_queue


def test_processor_stop_is_safe_before_start() -> None:
    processor = SparkCommunicationProcessor("192.0.2.1", 1234, "island:1")

    processor.stop()

    assert processor._running is None


def test_driver_constructor_creates_stable_queues_without_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_socket(family: int, kind: int) -> Never:
        pytest.fail(f"socket created early: {(family, kind)}")

    def unexpected_thread(
        *, target: Callable[[], None], name: str, daemon: bool
    ) -> Never:
        pytest.fail(f"thread created early: {(target, name, daemon)}")

    monkeypatch.setattr(socket, "socket", unexpected_socket)
    monkeypatch.setattr(transport, "Thread", unexpected_thread)
    driver = SparkCommunicationDriver(["island:1", "island:2", "island:1"], 1234)
    incoming = driver.incoming_queues
    outgoing = driver.outgoing_queues
    outgoing["island:1"].put("pending")

    assert set(incoming) == {"island:1", "island:2"}
    assert set(outgoing) == set(incoming)
    assert driver.incoming_queues is incoming
    assert driver.outgoing_queues is outgoing
    assert outgoing["island:1"].get_nowait() == "pending"


def test_driver_start_configures_server_selector_and_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver, server, selector, thread = _driver_runtime(monkeypatch)

    assert server.options == [(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)]
    assert server.bound_to == ("", 1234)
    assert server.listening
    assert server.blocking is False
    assert selector.registrations == [(server, selectors.EVENT_READ)]
    assert thread.started
    with pytest.raises(RuntimeError, match="already running"):
        driver.start()


def test_driver_handshake_registers_and_flushes_queued_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver, server, _, _ = _driver_runtime(monkeypatch)
    connection = ConnectionDouble([b"\x02island:1\x03"])
    server.accept_reply = connection
    driver._accept_connection()
    driver.outgoing_queues["island:1"].put("waiting")
    assert connection.sent == []

    driver._receive_from_connection(connection)

    assert driver._connections == {"island:1": connection}
    assert driver._connection_identifiers[connection] == "island:1"
    assert connection.sent == [b"\x02waiting\x03"]
    assert driver.outgoing_queues["island:1"].empty()


def test_driver_duplicate_identifier_keeps_original_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver, server, _, _ = _driver_runtime(monkeypatch)
    original = ConnectionDouble([b"\x02island:1\x03"])
    server.accept_reply = original
    driver._accept_connection()
    driver._receive_from_connection(original)
    duplicate = ConnectionDouble([b"\x02island:1\x03"])
    server.accept_reply = duplicate
    driver._accept_connection()

    driver._receive_from_connection(duplicate)

    assert duplicate.closed
    assert not original.closed
    assert driver._connections == {"island:1": original}
    assert driver._connection_identifiers == {original: "island:1"}


@pytest.mark.parametrize(
    ("chunks", "expected", "remainder"),
    [
        ([b"junk\x02island:1\x03\x02one\x03"], ["one"], ""),
        ([b"junk", b"\x02island:1\x03"], [], ""),
        ([b"\x02island:", b"1\x03\x02one\x03"], ["one"], ""),
        (
            [b"\x02island:1\x03\x02one\x03\x02two\x03\x02part"],
            ["one", "two"],
            "\x02part",
        ),
    ],
    ids=[
        "garbage-prefix",
        "no-stx",
        "fragmented",
        "multiple-with-remainder",
    ],
)
def test_driver_buffer_registers_then_routes_complete_frames(
    monkeypatch: pytest.MonkeyPatch,
    chunks: list[bytes],
    expected: list[str],
    remainder: str,
) -> None:
    driver, server, _, _ = _driver_runtime(monkeypatch)
    connection = ConnectionDouble(chunks)
    server.accept_reply = connection
    driver._accept_connection()

    for _ in chunks:
        driver._receive_from_connection(connection)

    assert [
        driver.incoming_queues["island:1"].get_nowait() for _ in expected
    ] == expected
    assert driver.incoming_queues["island:1"].empty()
    assert driver.incoming_queues["island:2"].empty()
    assert driver._receive_buffers[connection] == remainder


def test_driver_outgoing_put_sends_in_order_to_only_its_island(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver, server, _, _ = _driver_runtime(monkeypatch)
    first = ConnectionDouble([b"\x02island:1\x03"])
    second = ConnectionDouble([b"\x02island:2\x03"])
    for connection in (first, second):
        server.accept_reply = connection
        driver._accept_connection()
        driver._receive_from_connection(connection)

    driver.outgoing_queues["island:1"].put("one")
    driver.outgoing_queues["island:1"].put("two")
    driver.outgoing_queues["island:2"].put("other")

    assert first.sent == [b"\x02one\x03", b"\x02two\x03"]
    assert second.sent == [b"\x02other\x03"]
    assert all(queue.empty() for queue in driver.outgoing_queues.values())


def test_driver_send_failure_requeues_and_closes_broken_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver, server, selector, _ = _driver_runtime(monkeypatch)
    broken = ConnectionDouble([b"\x02island:1\x03"])
    server.accept_reply = broken
    driver._accept_connection()
    driver._receive_from_connection(broken)
    broken.fail_send = True

    driver.outgoing_queues["island:1"].put("retry")

    assert driver.outgoing_queues["island:1"].qsize() == 1
    assert broken.closed
    assert broken in selector.unregistered
    assert driver._connections == {}
    assert driver._connection_identifiers == {}
    assert broken not in driver._receive_buffers

    replacement = ConnectionDouble([b"\x02island:1\x03"])
    server.accept_reply = replacement
    driver._accept_connection()
    driver._receive_from_connection(replacement)
    assert replacement.sent == [b"\x02retry\x03"]
    assert driver.outgoing_queues["island:1"].empty()


@pytest.mark.parametrize(
    "reply", [OSError("recv failed"), b"", b"\xff"], ids=["os-error", "eof", "utf8"]
)
def test_driver_receive_closes_connection_on_terminal_input(
    monkeypatch: pytest.MonkeyPatch, reply: bytes | Exception
) -> None:
    driver, server, selector, _ = _driver_runtime(monkeypatch)
    connection = ConnectionDouble([reply])
    server.accept_reply = connection
    driver._accept_connection()

    driver._receive_from_connection(connection)

    assert connection.closed
    assert connection in selector.unregistered
    assert connection not in driver._receive_buffers


def test_driver_loop_dispatches_events_and_flushes_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver, server, selector, _ = _driver_runtime(monkeypatch)
    registered = ConnectionDouble([b"\x02island:1\x03"])
    server.accept_reply = registered
    driver._accept_connection()
    driver._receive_from_connection(registered)
    pending = ConnectionDouble([b"\x02island:2\x03\x02incoming\x03"])
    server.accept_reply = pending
    send_lock = driver._send_lock
    monkeypatch.setattr(driver, "_send_lock", None)
    driver.outgoing_queues["island:1"].put("flush")
    assert registered.sent == []
    monkeypatch.setattr(driver, "_send_lock", send_lock)
    running = driver._running
    assert running is not None
    selector.on_select = running.clear
    selector.events = [
        (SimpleNamespace(fileobj=server), selectors.EVENT_READ),
        (SimpleNamespace(fileobj=pending), selectors.EVENT_READ),
        (SimpleNamespace(fileobj=registered), 0),
    ]

    driver._communication_loop()

    assert pending.blocking is False
    assert driver.incoming_queues["island:2"].get_nowait() == "incoming"
    assert registered.sent == [b"\x02flush\x03"]


def test_driver_stop_releases_runtime_despite_close_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver, server, selector, thread = _driver_runtime(monkeypatch)
    connection = ConnectionDouble([b"\x02island:1\x03"])
    server.accept_reply = connection
    driver._accept_connection()
    driver._receive_from_connection(connection)
    running = driver._running
    server.fail_close = True
    connection.fail_close = True
    incoming = driver.incoming_queues
    outgoing = driver.outgoing_queues

    driver.stop()

    assert running is not None and not running.is_set()
    assert server.closed and connection.closed and selector.closed and thread.joined
    assert driver._connections == {}
    assert driver._connection_identifiers == {}
    assert driver._receive_buffers == {}
    assert driver._server_socket is None
    assert driver._selector is None
    assert driver._thread is None
    assert driver._running is None
    assert driver._send_lock is None
    assert driver.incoming_queues is incoming
    assert driver.outgoing_queues is outgoing


def test_driver_stop_is_safe_before_start() -> None:
    driver = SparkCommunicationDriver(["island:1"], 1234)

    driver.stop()

    assert driver._selector is None
    assert driver._running is None


def test_driver_rejected_handshake_does_not_restore_closed_receive_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver, server, selector, _ = _driver_runtime(monkeypatch)
    connection = ConnectionDouble([b"\x02unknown\x03"])
    server.accept_reply = connection
    driver._accept_connection()

    driver._receive_from_connection(connection)

    assert connection.closed
    assert connection in selector.unregistered
    assert driver._connections == {}
    assert driver._connection_identifiers == {}
    assert connection not in driver._receive_buffers
