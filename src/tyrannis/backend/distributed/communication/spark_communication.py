from __future__ import annotations

import selectors
import socket
from queue import Empty, Queue
from threading import Event, Thread
from typing import Final, cast

from ....core.backend_communication import (
    CommunicationDriverBase,
    CommunicationProcessorBase,
)
from ....core.signals import LocalEvent


class SparkCommunicationProcessor(CommunicationProcessorBase):
    """TCP communication layer for a processor."""

    STX: Final[str] = "\x02"
    ETX: Final[str] = "\x03"
    STOP: Final[str] = "\x04"

    def __init__(
        self,
        driver_ip: str,
        port: int,
        identification: str,
    ) -> None:
        self._driver_ip = driver_ip
        self._port = port
        self._identification = identification

        self._socket: socket.socket | None = None
        self._thread: Thread | None = None
        self._running: Event | None = None

        self._stop_signal: LocalEvent | None = None

        self._messages: Queue[str] | None = None
        self._outgoing_queue: Queue[str] | None = None

        self._receive_buffer = ""

    @property
    def messages(self) -> Queue[str]:
        if self._messages is None:
            raise RuntimeError(
                "Communication processor is not running.",
            )

        return self._messages

    @property
    def outgoing_queue(self) -> Queue[str]:
        if self._outgoing_queue is None:
            raise RuntimeError(
                "Communication processor is not running.",
            )

        return self._outgoing_queue

    def start(
        self,
        stop_signal: LocalEvent,
    ) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError(
                "Communication is already running.",
            )

        self._stop_signal = stop_signal

        self._running = Event()
        self._messages = Queue()
        self._outgoing_queue = Queue()

        self._socket = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        )

        self._socket.connect(
            (self._driver_ip, self._port),
        )

        self._running.set()

        self._send_message(
            self._identification,
        )

        self._thread = Thread(
            target=self._receive_loop,
            name="processor-communication",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._running is not None:
            self._running.clear()

        if self._socket is not None:
            try:
                self._socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

            try:
                self._socket.close()
            except OSError:
                pass

            self._socket = None

        if self._thread is not None:
            self._thread.join()
            self._thread = None

        self._running = None
        self._messages = None
        self._outgoing_queue = None

        self._receive_buffer = ""
        self._stop_signal = None

    def _receive_loop(self) -> None:
        communication_socket = self._socket
        running = self._running
        stop_signal = self._stop_signal

        if communication_socket is None:
            raise RuntimeError(
                "Communication socket is not initialized.",
            )

        if running is None:
            raise RuntimeError(
                "Communication running state is not initialized.",
            )

        if stop_signal is None:
            raise RuntimeError(
                "Communication stop signal is not initialized.",
            )

        communication_socket.settimeout(0.1)

        while running.is_set():
            self._send_pending_messages()

            try:
                data = communication_socket.recv(4096)
            except TimeoutError:
                continue
            except OSError:
                if running.is_set():
                    stop_signal.set()

                break

            if not data:
                if running.is_set():
                    stop_signal.set()

                break

            try:
                self._receive_buffer += data.decode("utf-8")
            except UnicodeDecodeError:
                stop_signal.set()
                break

            self._process_buffer()

    def _process_buffer(self) -> None:
        while self._receive_buffer:
            stop_position = self._receive_buffer.find(
                self.STOP,
            )

            if stop_position != -1:
                stx_position = self._receive_buffer.find(
                    self.STX,
                )

                if stx_position == -1 or stop_position < stx_position:
                    self._receive_buffer = self._receive_buffer[
                        stop_position + len(self.STOP) :
                    ]

                    stop_signal = self._stop_signal

                    if stop_signal is None:
                        raise RuntimeError(
                            "Communication stop signal is not initialized.",
                        )

                    stop_signal.set()
                    return

            stx_position = self._receive_buffer.find(
                self.STX,
            )

            if stx_position == -1:
                self._receive_buffer = ""
                return

            if stx_position > 0:
                self._receive_buffer = self._receive_buffer[stx_position:]

            etx_position = self._receive_buffer.find(
                self.ETX,
                len(self.STX),
            )

            if etx_position == -1:
                return

            message = self._receive_buffer[len(self.STX) : etx_position]

            self._receive_buffer = self._receive_buffer[etx_position + len(self.ETX) :]

            self.messages.put(message)

    def _send_pending_messages(self) -> None:
        communication_socket = self._socket
        running = self._running
        outgoing_queue = self._outgoing_queue
        stop_signal = self._stop_signal

        if communication_socket is None:
            return

        if running is None:
            return

        if outgoing_queue is None:
            return

        if stop_signal is None:
            return

        while True:
            try:
                message = outgoing_queue.get_nowait()
            except Empty:
                return

            try:
                if message == self.STOP:
                    data = self.STOP.encode("utf-8")
                else:
                    data = (f"{self.STX}{message}{self.ETX}").encode()

                communication_socket.sendall(data)

            except OSError:
                outgoing_queue.put(message)

                if running.is_set():
                    stop_signal.set()

                return

    def _send_message(
        self,
        message: str,
    ) -> None:
        communication_socket = self._socket

        if communication_socket is None:
            raise RuntimeError(
                "Communication socket is not initialized.",
            )

        data = (f"{self.STX}{message}{self.ETX}").encode()

        communication_socket.sendall(data)


class SparkCommunicationDriver(CommunicationDriverBase):
    """TCP communication layer for the driver."""

    STX: Final[str] = "\x02"
    ETX: Final[str] = "\x03"
    STOP: Final[str] = "\x04"

    def __init__(
        self,
        island_ids: list[str],
        port: int,
        stop_signal: LocalEvent,
    ) -> None:
        self._island_ids = set(island_ids)
        self._port = port
        self._stop_signal = stop_signal

        self._incoming_queues: dict[str, Queue[str]] = {
            island_id: Queue() for island_id in self._island_ids
        }

        self._outgoing_queues: dict[str, Queue[str]] = {
            island_id: Queue() for island_id in self._island_ids
        }

        self._connections: dict[str, socket.socket] = {}

        self._connection_identifications: dict[
            socket.socket,
            str,
        ] = {}

        self._receive_buffers: dict[
            socket.socket,
            str,
        ] = {}

        self._server_socket: socket.socket | None = None
        self._selector: selectors.BaseSelector | None = None

        self._thread: Thread | None = None
        self._running = Event()

        self._stop_sent = False

    @property
    def incoming_queues(self) -> dict[str, Queue[str]]:
        """Queues containing messages received from each island."""
        return self._incoming_queues

    @property
    def outgoing_queues(self) -> dict[str, Queue[str]]:
        """Queues containing messages to be sent to each island."""
        return self._outgoing_queues

    def start(self) -> None:
        """Start the TCP server and communication thread."""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError(
                "Communication is already running.",
            )

        self._selector = selectors.DefaultSelector()

        self._server_socket = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        )

        self._server_socket.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )

        self._server_socket.bind(
            ("", self._port),
        )

        self._server_socket.listen()
        self._server_socket.setblocking(False)

        self._selector.register(
            self._server_socket,
            selectors.EVENT_READ,
        )

        self._stop_sent = False
        self._running.set()

        self._thread = Thread(
            target=self._communication_loop,
            name="driver-communication",
            daemon=True,
        )

        self._thread.start()

    def stop(self) -> None:
        """Stop the TCP server and close all connections."""
        self._running.clear()

        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except OSError:
                pass

            self._server_socket = None

        if self._thread is not None:
            self._thread.join()
            self._thread = None

        for connection in self._connections.values():
            try:
                connection.close()
            except OSError:
                pass

        self._connections.clear()
        self._connection_identifications.clear()
        self._receive_buffers.clear()

        if self._selector is not None:
            self._selector.close()
            self._selector = None

        self._stop_sent = False

    def _communication_loop(self) -> None:
        selector = self._selector
        server_socket = self._server_socket

        if selector is None:
            raise RuntimeError(
                "Selector is not initialized.",
            )

        if server_socket is None:
            raise RuntimeError(
                "Server socket is not initialized.",
            )

        while self._running.is_set() and not self._stop_signal.is_set():
            self._send_pending_messages()

            events = selector.select(
                timeout=0.1,
            )

            for key, mask in events:
                if key.fileobj is server_socket:
                    self._accept_connection()

                elif mask & selectors.EVENT_READ:
                    self._receive_from_connection(
                        cast(socket.socket, key.fileobj),
                    )

        self._send_stop_signal()

    def _accept_connection(self) -> None:
        server_socket = self._server_socket
        selector = self._selector

        if server_socket is None:
            return

        if selector is None:
            return

        try:
            connection, _ = server_socket.accept()
        except OSError:
            return

        connection.setblocking(False)

        self._receive_buffers[connection] = ""

        selector.register(
            connection,
            selectors.EVENT_READ,
        )

    def _receive_from_connection(
        self,
        connection: socket.socket,
    ) -> None:
        try:
            data = connection.recv(4096)
        except OSError:
            self._close_connection(connection)
            return

        if not data:
            self._close_connection(connection)

            if self._running.is_set():
                self._stop_signal.set()

            return

        try:
            self._receive_buffers[connection] += data.decode(
                "utf-8",
            )
        except UnicodeDecodeError:
            self._close_connection(connection)
            self._stop_signal.set()
            return

        self._process_driver_buffer(
            connection,
        )

    def _process_driver_buffer(
        self,
        connection: socket.socket,
    ) -> None:
        buffer = self._receive_buffers.get(connection)

        if buffer is None:
            return

        while buffer:
            stop_position = buffer.find(
                self.STOP,
            )

            if stop_position != -1:
                stx_position = buffer.find(
                    self.STX,
                )

                if stx_position == -1 or stop_position < stx_position:
                    buffer = buffer[stop_position + len(self.STOP) :]

                    self._stop_signal.set()
                    break

            stx_position = buffer.find(
                self.STX,
            )

            if stx_position == -1:
                buffer = ""
                break

            if stx_position > 0:
                buffer = buffer[stx_position:]

            etx_position = buffer.find(
                self.ETX,
                len(self.STX),
            )

            if etx_position == -1:
                break

            message = buffer[len(self.STX) : etx_position]

            buffer = buffer[etx_position + len(self.ETX) :]

            if not self._register_connection(
                connection,
                message,
            ):
                self._route_message(
                    connection,
                    message,
                )

        self._receive_buffers[connection] = buffer

    def _register_connection(
        self,
        connection: socket.socket,
        message: str,
    ) -> bool:
        if connection in self._connection_identifications:
            return False

        if message not in self._island_ids:
            self._close_connection(connection)
            self._stop_signal.set()
            return True

        existing_connection = self._connections.get(
            message,
        )

        if existing_connection is not None:
            self._close_connection(connection)
            self._stop_signal.set()
            return True

        self._connection_identifications[connection] = message
        self._connections[message] = connection

        return True

    def _route_message(
        self,
        connection: socket.socket,
        message: str,
    ) -> None:
        island_id = self._connection_identifications.get(
            connection,
        )

        if island_id is None:
            self._close_connection(connection)
            self._stop_signal.set()
            return

        self._incoming_queues[island_id].put(
            message,
        )

    def _close_connection(
        self,
        connection: socket.socket,
    ) -> None:
        selector = self._selector

        if selector is not None:
            try:
                selector.unregister(connection)
            except (
                KeyError,
                ValueError,
                OSError,
            ):
                pass

        island_id = self._connection_identifications.pop(
            connection,
            None,
        )

        if island_id is not None:
            active_connection = self._connections.get(
                island_id,
            )

            if active_connection is connection:
                del self._connections[island_id]

        self._receive_buffers.pop(
            connection,
            None,
        )

        try:
            connection.close()
        except OSError:
            pass

    def _send_pending_messages(self) -> None:
        for island_id, outgoing_queue in self._outgoing_queues.items():
            connection = self._connections.get(
                island_id,
            )

            if connection is None:
                continue

            while True:
                try:
                    message = outgoing_queue.get_nowait()
                except Empty:
                    break

                try:
                    data = (f"{self.STX}{message}{self.ETX}").encode()

                    connection.sendall(data)

                except OSError:
                    outgoing_queue.put(message)
                    self._close_connection(connection)

                    if self._running.is_set():
                        self._stop_signal.set()

                    break

    def _send_stop_signal(self) -> None:
        if self._stop_sent:
            return

        self._stop_sent = True

        for connection in list(
            self._connections.values(),
        ):
            try:
                connection.sendall(
                    self.STOP.encode("utf-8"),
                )
            except OSError:
                pass
