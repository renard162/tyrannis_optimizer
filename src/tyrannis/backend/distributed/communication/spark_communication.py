from __future__ import annotations

import selectors
import socket
from queue import Empty, Queue
from threading import Event, Thread
from typing import Final

from ....core.processor import LocalEvent


class SparkCommunicationProcessor:
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

        self._running = Event()

        self._wait_signal: LocalEvent = LocalEvent()
        self._stop_signal: LocalEvent = LocalEvent()

        self._messages: Queue[str] = Queue()
        self._outgoing_queue: Queue[str] = Queue()

        self._receive_buffer = ""

    @property
    def messages(self) -> Queue[str]:
        return self._messages

    @property
    def outgoing_queue(self) -> Queue[str]:
        return self._outgoing_queue

    def start(
        self,
        wait_signal: LocalEvent,
        stop_signal: LocalEvent,
    ) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Communication is already running.")

        self._wait_signal = wait_signal
        self._stop_signal = stop_signal

        self._socket = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        )
        self._socket.connect(
            (self._driver_ip, self._port),
        )

        self._running.set()

        # The identification is sent as a normal framed message.
        self._send_message(self._identification)

        self._thread = Thread(
            target=self._receive_loop,
            name="processor-communication",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()

        if self._socket is not None:
            try:
                self._socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

            self._socket.close()
            self._socket = None

        if self._thread is not None:
            self._thread.join()
            self._thread = None

        self._receive_buffer = ""

        self._stop_signal = LocalEvent()
        self._wait_signal = LocalEvent()

    def _receive_loop(self) -> None:
        if self._socket is None:
            raise RuntimeError(
                "Communication socket is not initialized.",
            )

        self._socket.settimeout(0.1)

        while self._running.is_set():
            self._send_pending_messages()

            try:
                data = self._socket.recv(4096)
            except TimeoutError:
                continue
            except OSError:
                if self._running.is_set():
                    self._stop_signal.set()

                break

            if not data:
                break

            try:
                self._receive_buffer += data.decode("utf-8")
            except UnicodeDecodeError:
                self._stop_signal.set()
                break

            self._process_buffer()

    def _process_buffer(self) -> None:
        while self._receive_buffer:
            # STOP is a control signal and is not part of a frame.
            stop_position = self._receive_buffer.find(self.STOP)

            if stop_position != -1:
                stx_position = self._receive_buffer.find(self.STX)

                # If STOP occurs before the next frame, discard everything
                # before STOP and process the stop signal.
                if stx_position == -1 or stop_position < stx_position:
                    self._receive_buffer = self._receive_buffer[
                        stop_position + len(self.STOP) :
                    ]
                    self._stop_signal.set()
                    return

            # Locate the beginning of the next frame.
            stx_position = self._receive_buffer.find(self.STX)

            if stx_position == -1:
                # No frame can be completed with the current buffer.
                self._receive_buffer = ""
                return

            # Discard bytes preceding STX.
            if stx_position > 0:
                self._receive_buffer = self._receive_buffer[stx_position:]

            # Locate the end of the frame.
            etx_position = self._receive_buffer.find(
                self.ETX,
                len(self.STX),
            )

            if etx_position == -1:
                # The frame is incomplete. Keep it for the next recv().
                return

            message = self._receive_buffer[len(self.STX) : etx_position]

            self._receive_buffer = self._receive_buffer[etx_position + len(self.ETX) :]

            if message:
                self._messages.put(message)
            else:
                self._wait_signal.set()

    def _send_pending_messages(self) -> None:
        if self._socket is None:
            return

        while True:
            try:
                message = self._outgoing_queue.get_nowait()
            except Empty:
                return

            try:
                if message == self.STOP:
                    data = self.STOP.encode("utf-8")
                else:
                    data = f"{self.STX}{message}{self.ETX}".encode()

                self._socket.sendall(data)
            except OSError:
                self._outgoing_queue.put(message)

                if self._running.is_set():
                    self._stop_signal.set()

                return

    def _send_message(self, message: str) -> None:
        if self._socket is None:
            raise RuntimeError(
                "Communication socket is not initialized.",
            )

        data = f"{self.STX}{message}{self.ETX}".encode()
        self._socket.sendall(data)


class SparkCommunicationDriver:
    """TCP communication layer for the driver."""

    STX: Final[str] = "\x02"
    ETX: Final[str] = "\x03"
    STOP: Final[str] = "\x04"

    def __init__(
        self,
        island_ids: list[str],
        port: int,
        stop_signal: Event,
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

        # Maps an island ID to its active TCP connection.
        self._connections: dict[str, socket.socket] = {}

        # Keeps a receive buffer for each TCP connection.
        self._receive_buffers: dict[socket.socket, str] = {}

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

        self._server_socket.bind(("", self._port))
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
        self._receive_buffers.clear()

        if self._selector is not None:
            self._selector.close()
            self._selector = None

        self._stop_sent = False

    def _communication_loop(self) -> None:
        if self._selector is None:
            raise RuntimeError(
                "Selector is not initialized.",
            )

        while self._running.is_set():
            if self._stop_signal.is_set() and not self._stop_sent:
                self._send_stop_to_all()
                self._stop_sent = True

            events = self._selector.select(timeout=0.1)

            for key, mask in events:
                if key.fileobj is self._server_socket:
                    self._accept_connection()
                elif mask & selectors.EVENT_READ:
                    self._receive(key)

            self._send_pending_messages()

    def _accept_connection(self) -> None:
        if self._server_socket is None:
            return

        if self._selector is None:
            raise RuntimeError(
                "Selector is not initialized.",
            )

        try:
            connection, _ = self._server_socket.accept()
        except OSError:
            return

        connection.setblocking(False)

        self._receive_buffers[connection] = ""

        self._selector.register(
            connection,
            selectors.EVENT_READ,
            data=None,
        )

    def _receive(self, key: selectors.SelectorKey) -> None:
        connection = key.fileobj

        if not isinstance(connection, socket.socket):
            return

        # data=None means that the client has not yet identified itself.
        if key.data is None:
            self._receive_handshake(connection)
            return

        island_id = key.data

        if not isinstance(island_id, str):
            self._close_connection(connection)
            return

        self._receive_message(
            connection,
            island_id,
        )

    def _receive_handshake(
        self,
        connection: socket.socket,
    ) -> None:
        if self._selector is None:
            raise RuntimeError(
                "Selector is not initialized.",
            )

        if not self._receive_data(connection):
            return

        messages = self._extract_messages(connection)

        for island_id in messages:
            # The first framed message received from a new connection
            # must be the client's identification.
            if island_id not in self._island_ids:
                self._close_connection(connection)
                return

            # Only one active connection is allowed for each island.
            old_connection = self._connections.get(island_id)

            if old_connection is not None:
                self._close_connection(
                    old_connection,
                    island_id,
                )

            self._connections[island_id] = connection

            # From this point onward, messages received through this
            # socket are routed to this island's queue.
            self._selector.modify(
                connection,
                selectors.EVENT_READ,
                data=island_id,
            )

            return

    def _receive_message(
        self,
        connection: socket.socket,
        island_id: str,
    ) -> None:
        if not self._receive_data(connection):
            return

        for message in self._extract_messages(connection):
            self._incoming_queues[island_id].put(message)

    def _receive_data(
        self,
        connection: socket.socket,
    ) -> bool:
        try:
            data = connection.recv(4096)
        except OSError:
            self._close_connection(connection)
            return False

        if not data:
            self._close_connection(connection)
            return False

        try:
            self._receive_buffers[connection] += data.decode(
                "utf-8",
            )
        except UnicodeDecodeError:
            self._close_connection(connection)
            return False

        return True

    def _extract_messages(
        self,
        connection: socket.socket,
    ) -> list[str]:
        buffer = self._receive_buffers[connection]
        messages: list[str] = []

        while buffer:
            # STOP is independent from STX/ETX framing.
            stop_position = buffer.find(self.STOP)

            if stop_position != -1:
                stx_position = buffer.find(self.STX)

                if stx_position == -1 or stop_position < stx_position:
                    buffer = buffer[stop_position + len(self.STOP) :]

                    self._receive_buffers[connection] = buffer
                    self._handle_stop()

                    break

            stx_position = buffer.find(self.STX)

            if stx_position == -1:
                # Discard data that cannot form a frame.
                buffer = ""
                break

            if stx_position > 0:
                buffer = buffer[stx_position:]

            etx_position = buffer.find(
                self.ETX,
                len(self.STX),
            )

            if etx_position == -1:
                # Incomplete frame; preserve it for the next recv().
                break

            message = buffer[len(self.STX) : etx_position]

            messages.append(message)

            buffer = buffer[etx_position + len(self.ETX) :]

        self._receive_buffers[connection] = buffer

        return messages

    def _handle_stop(self) -> None:
        self._stop_signal.set()

    def _send_pending_messages(self) -> None:
        for island_id, queue in self._outgoing_queues.items():
            connection = self._connections.get(island_id)

            if connection is None:
                continue

            while True:
                try:
                    message = queue.get_nowait()
                except Empty:
                    break

                try:
                    # STOP is a control signal, not a framed message.
                    if message == self.STOP:
                        data = self.STOP.encode("utf-8")
                    else:
                        data = f"{self.STX}{message}{self.ETX}".encode()

                    connection.sendall(data)

                except OSError:
                    queue.put(message)

                    self._close_connection(
                        connection,
                        island_id,
                    )

                    break

    def _send_stop_to_all(self) -> None:
        data = self.STOP.encode("utf-8")

        for island_id, connection in list(
            self._connections.items(),
        ):
            try:
                connection.sendall(data)
            except OSError:
                self._close_connection(
                    connection,
                    island_id,
                )

    def _close_connection(
        self,
        connection: socket.socket,
        island_id: str | None = None,
    ) -> None:
        if self._selector is not None:
            try:
                self._selector.unregister(connection)
            except (KeyError, ValueError):
                pass

        try:
            connection.close()
        except OSError:
            pass

        self._receive_buffers.pop(connection, None)

        if island_id is not None:
            self._connections.pop(island_id, None)
