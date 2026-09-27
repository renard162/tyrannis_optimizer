from __future__ import annotations

from collections.abc import Callable
from queue import Empty, Queue
from threading import Event as ThreadEvent
from threading import Thread
from typing import TYPE_CHECKING, Final

from ....core.backend_communication import (
    CommunicationDriverBase,
    CommunicationProcessorBase,
)
from ....core.signals import EventProtocol

if TYPE_CHECKING:
    from mpi4py.MPI import Intracomm


_PROCESSOR_TO_DRIVER_TAG: Final[int] = 27101
_DRIVER_TO_PROCESSOR_TAG: Final[int] = 27102
_DRIVER_RANK: Final[int] = 0
_POLL_INTERVAL: Final[float] = 0.01

_REGISTER: Final[str] = "register"
_REGISTERED: Final[str] = "registered"
_REJECTED: Final[str] = "rejected"
_STOP: Final[str] = "stop"
_STOPPED: Final[str] = "stopped"


class _OutgoingQueue(Queue[str]):
    """Queue that wakes its communication thread when a message arrives."""

    def __init__(self, on_put: Callable[[], None]) -> None:
        super().__init__()
        self._on_put = on_put

    def put(self, item: str, block: bool = True, timeout: float | None = None) -> None:
        super().put(item, block=block, timeout=timeout)
        self._on_put()


def _get_world_communicator() -> Intracomm:
    from mpi4py import MPI

    if not MPI.Is_initialized():
        raise RuntimeError("MPI runtime is not initialized.")

    if MPI.Is_finalized():
        raise RuntimeError("MPI runtime has already been finalized.")

    if MPI.Query_thread() < MPI.THREAD_MULTIPLE:
        raise RuntimeError(
            "MPI communication requires MPI_THREAD_MULTIPLE support because "
            "Tyrannis runs communication asynchronously in background threads."
        )

    return MPI.COMM_WORLD


def _control_message(message_type: str, identifier: str) -> tuple[str, str]:
    return message_type, identifier


def _parse_control_message(payload: object) -> tuple[str, str] | None:
    if not isinstance(payload, tuple) or len(payload) != 2:
        return None

    message_type, identifier = payload

    if not isinstance(message_type, str) or not isinstance(identifier, str):
        return None

    return message_type, identifier


class MPICommunicationProcessor(CommunicationProcessorBase):
    """MPI communication layer for an optimization processor."""

    def __init__(self, identifier: str) -> None:
        self._identifier = identifier

        self._communicator: Intracomm | None = None
        self._rank: int | None = None

        self._thread: Thread | None = None
        self._running: ThreadEvent | None = None
        self._wakeup: ThreadEvent | None = None
        self._stop_requested: ThreadEvent | None = None
        self._stop_acknowledged: ThreadEvent | None = None
        self._stop_sent = False

        self._message_signal: EventProtocol | None = None
        self._messages: Queue[str] | None = None
        self._outgoing_queue: Queue[str] | None = None
        self._pending_outgoing: str | None = None

    @property
    def messages(self) -> Queue[str]:
        if self._messages is None:
            raise RuntimeError("Communication processor is not running.")

        return self._messages

    @property
    def outgoing_queue(self) -> Queue[str]:
        if self._outgoing_queue is None:
            raise RuntimeError("Communication processor is not running.")

        return self._outgoing_queue

    def set_message_signal(self, message_signal: EventProtocol | None) -> None:
        """Set the signal used to notify the processor of received messages."""
        self._message_signal = message_signal

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Communication is already running.")

        communicator = _get_world_communicator()
        rank = communicator.Get_rank()

        if rank == _DRIVER_RANK:
            raise RuntimeError(
                "MPI processor communication cannot run on the driver rank."
            )

        self._communicator = communicator
        self._rank = rank

        self._running = ThreadEvent()
        self._wakeup = ThreadEvent()
        self._stop_requested = ThreadEvent()
        self._stop_acknowledged = ThreadEvent()
        self._stop_sent = False

        self._messages = Queue()
        self._outgoing_queue = _OutgoingQueue(self._notify_outgoing)
        self._pending_outgoing = None

        from mpi4py import MPI

        try:
            communicator.send(
                _control_message(_REGISTER, self._identifier),
                dest=_DRIVER_RANK,
                tag=_PROCESSOR_TO_DRIVER_TAG,
            )
            response = communicator.recv(
                source=_DRIVER_RANK, tag=_DRIVER_TO_PROCESSOR_TAG
            )
        except MPI.Exception:
            self._reset_runtime()
            raise

        control = _parse_control_message(response)

        if control != (_REGISTERED, self._identifier):
            self._reset_runtime()
            raise RuntimeError(
                f"MPI communication registration was rejected for "
                f"processor {self._identifier!r}."
            )

        running = self._running

        if running is None:
            self._reset_runtime()
            raise RuntimeError("Communication running state is not initialized.")

        running.set()

        self._thread = Thread(
            target=self._communication_loop,
            name=f"mpi-processor-communication-{rank}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        thread = self._thread
        stop_requested = self._stop_requested
        wakeup = self._wakeup

        if thread is not None and thread.is_alive():
            if stop_requested is not None:
                stop_requested.set()

            if wakeup is not None:
                wakeup.set()

            thread.join()

        self._reset_runtime()
        self._message_signal = None

    def _reset_runtime(self) -> None:
        self._communicator = None
        self._rank = None

        self._thread = None
        self._running = None
        self._wakeup = None
        self._stop_requested = None
        self._stop_acknowledged = None
        self._stop_sent = False

        self._messages = None
        self._outgoing_queue = None
        self._pending_outgoing = None

    def _notify_outgoing(self) -> None:
        wakeup = self._wakeup

        if wakeup is not None:
            wakeup.set()

    def _communication_loop(self) -> None:
        running = self._running
        wakeup = self._wakeup
        stop_requested = self._stop_requested
        stop_acknowledged = self._stop_acknowledged

        if running is None:
            raise RuntimeError("Communication running state is not initialized.")

        if wakeup is None:
            raise RuntimeError("Communication wakeup event is not initialized.")

        if stop_requested is None or stop_acknowledged is None:
            raise RuntimeError("Communication stop state is not initialized.")

        while running.is_set():
            self._receive_pending_messages()
            self._send_pending_messages()

            if stop_requested.is_set():
                if not self._stop_sent and self._outgoing_is_drained():
                    self._stop_sent = self._send_stop_request()

                if stop_acknowledged.is_set():
                    running.clear()
                    break

            wakeup.wait(timeout=_POLL_INTERVAL)
            wakeup.clear()

    def _outgoing_is_drained(self) -> bool:
        outgoing_queue = self._outgoing_queue

        if outgoing_queue is None:
            return True

        return self._pending_outgoing is None and outgoing_queue.empty()

    def _send_stop_request(self) -> bool:
        communicator = self._communicator

        if communicator is None:
            return False

        from mpi4py import MPI

        try:
            communicator.send(
                _control_message(_STOP, self._identifier),
                dest=_DRIVER_RANK,
                tag=_PROCESSOR_TO_DRIVER_TAG,
            )
        except MPI.Exception:
            return False

        return True

    def _receive_pending_messages(self) -> None:
        communicator = self._communicator

        if communicator is None:
            return

        from mpi4py import MPI

        while communicator.Iprobe(source=_DRIVER_RANK, tag=_DRIVER_TO_PROCESSOR_TAG):
            try:
                payload = communicator.recv(
                    source=_DRIVER_RANK, tag=_DRIVER_TO_PROCESSOR_TAG
                )
            except MPI.Exception:
                return

            control = _parse_control_message(payload)

            if control is not None:
                message_type, identifier = control

                if message_type == _STOPPED and identifier == self._identifier:
                    stop_acknowledged = self._stop_acknowledged

                    if stop_acknowledged is not None:
                        stop_acknowledged.set()

                continue

            if not isinstance(payload, str):
                continue

            self.messages.put(payload)

            if self._message_signal is not None:
                self._message_signal.set()

    def _send_pending_messages(self) -> None:
        communicator = self._communicator
        outgoing_queue = self._outgoing_queue

        if communicator is None or outgoing_queue is None:
            return

        from mpi4py import MPI

        while True:
            message = self._pending_outgoing

            if message is None:
                try:
                    message = outgoing_queue.get_nowait()
                except Empty:
                    return

            try:
                communicator.send(
                    message, dest=_DRIVER_RANK, tag=_PROCESSOR_TO_DRIVER_TAG
                )
            except MPI.Exception:
                self._pending_outgoing = message
                return

            self._pending_outgoing = None


class MPICommunicationDriver(CommunicationDriverBase):
    """MPI communication layer for the distributed optimization driver."""

    def __init__(self, island_ids: list[str]) -> None:
        self._island_ids = set(island_ids)

        self._incoming_queues: dict[str, Queue[str]] = {
            island_id: Queue() for island_id in self._island_ids
        }

        self._outgoing_queues: dict[str, Queue[str]] = {
            island_id: _OutgoingQueue(self._notify_outgoing)
            for island_id in self._island_ids
        }

        self._communicator: Intracomm | None = None

        self._thread: Thread | None = None
        self._running: ThreadEvent | None = None
        self._wakeup: ThreadEvent | None = None

        self._island_ranks: dict[str, int] = {}
        self._rank_islands: dict[int, str] = {}

        self._pending_outgoing: dict[str, str] = {}
        self._pending_stop_acknowledgements: dict[int, str] = {}

    @property
    def incoming_queues(self) -> dict[str, Queue[str]]:
        """Queues containing messages received from each island."""
        return self._incoming_queues

    @property
    def outgoing_queues(self) -> dict[str, Queue[str]]:
        """Queues containing messages to be sent to each island."""
        return self._outgoing_queues

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Communication is already running.")

        communicator = _get_world_communicator()

        if communicator.Get_rank() != _DRIVER_RANK:
            raise RuntimeError("MPI driver communication must run on rank 0.")

        self._communicator = communicator

        self._island_ranks.clear()
        self._rank_islands.clear()
        self._pending_outgoing.clear()
        self._pending_stop_acknowledgements.clear()

        self._running = ThreadEvent()
        self._running.set()

        self._wakeup = ThreadEvent()

        self._thread = Thread(
            target=self._communication_loop,
            name="mpi-driver-communication",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._running is not None:
            self._running.clear()

        if self._wakeup is not None:
            self._wakeup.set()

        if self._thread is not None:
            self._thread.join()
            self._thread = None

        self._communicator = None
        self._running = None
        self._wakeup = None

        self._island_ranks.clear()
        self._rank_islands.clear()
        self._pending_outgoing.clear()
        self._pending_stop_acknowledgements.clear()

    def _notify_outgoing(self) -> None:
        wakeup = self._wakeup

        if wakeup is not None:
            wakeup.set()

    def _communication_loop(self) -> None:
        running = self._running
        wakeup = self._wakeup

        if running is None:
            raise RuntimeError("Communication running state is not initialized.")

        if wakeup is None:
            raise RuntimeError("Communication wakeup event is not initialized.")

        while running.is_set():
            self._receive_pending_messages()
            self._send_pending_messages()

            wakeup.wait(timeout=_POLL_INTERVAL)
            wakeup.clear()

        self._receive_pending_messages()
        self._send_stop_acknowledgements()

    def _receive_pending_messages(self) -> None:
        communicator = self._communicator

        if communicator is None:
            return

        from mpi4py import MPI

        while communicator.Iprobe(source=MPI.ANY_SOURCE, tag=_PROCESSOR_TO_DRIVER_TAG):
            status = MPI.Status()

            try:
                payload = communicator.recv(
                    source=MPI.ANY_SOURCE, tag=_PROCESSOR_TO_DRIVER_TAG, status=status
                )
            except MPI.Exception:
                return

            source_rank = status.Get_source()
            control = _parse_control_message(payload)

            if control is not None:
                self._process_control_message(
                    source_rank=source_rank,
                    message_type=control[0],
                    identifier=control[1],
                )
                continue

            if not isinstance(payload, str):
                continue

            island_id = self._rank_islands.get(source_rank)

            if island_id is None:
                continue

            self._incoming_queues[island_id].put(payload)

    def _process_control_message(
        self, source_rank: int, message_type: str, identifier: str
    ) -> None:
        if message_type == _REGISTER:
            self._register_processor(source_rank=source_rank, identifier=identifier)
            return

        if message_type == _STOP:
            self._stop_processor(source_rank=source_rank, identifier=identifier)

    def _register_processor(self, source_rank: int, identifier: str) -> None:
        communicator = self._communicator

        if communicator is None:
            return

        accepted = False

        registered_identifier = self._rank_islands.get(source_rank)
        registered_rank = self._island_ranks.get(identifier)

        if identifier in self._island_ids:
            if registered_identifier == identifier:
                accepted = True

            elif registered_identifier is None and registered_rank is None:
                self._rank_islands[source_rank] = identifier
                self._island_ranks[identifier] = source_rank
                accepted = True

        response_type = _REGISTERED if accepted else _REJECTED

        from mpi4py import MPI

        try:
            communicator.send(
                _control_message(response_type, identifier),
                dest=source_rank,
                tag=_DRIVER_TO_PROCESSOR_TAG,
            )
        except MPI.Exception:
            if accepted:
                self._rank_islands.pop(source_rank, None)
                self._island_ranks.pop(identifier, None)

            return

        if accepted:
            self._notify_outgoing()

    def _stop_processor(self, source_rank: int, identifier: str) -> None:
        if self._rank_islands.get(source_rank) != identifier:
            return

        if self._island_ranks.get(identifier) != source_rank:
            return

        self._rank_islands.pop(source_rank, None)
        self._island_ranks.pop(identifier, None)
        self._pending_outgoing.pop(identifier, None)

        self._pending_stop_acknowledgements[source_rank] = identifier

        self._notify_outgoing()

    def _send_pending_messages(self) -> None:
        self._send_stop_acknowledgements()

        communicator = self._communicator

        if communicator is None:
            return

        from mpi4py import MPI

        for island_id, outgoing_queue in self._outgoing_queues.items():
            destination_rank = self._island_ranks.get(island_id)

            if destination_rank is None:
                continue

            while True:
                message = self._pending_outgoing.get(island_id)

                if message is None:
                    try:
                        message = outgoing_queue.get_nowait()
                    except Empty:
                        break

                try:
                    communicator.send(
                        message, dest=destination_rank, tag=_DRIVER_TO_PROCESSOR_TAG
                    )
                except MPI.Exception:
                    self._pending_outgoing[island_id] = message
                    break

                self._pending_outgoing.pop(island_id, None)

    def _send_stop_acknowledgements(self) -> None:
        communicator = self._communicator

        if communicator is None:
            return

        from mpi4py import MPI

        for source_rank, identifier in list(
            self._pending_stop_acknowledgements.items()
        ):
            try:
                communicator.send(
                    _control_message(_STOPPED, identifier),
                    dest=source_rank,
                    tag=_DRIVER_TO_PROCESSOR_TAG,
                )
            except MPI.Exception:
                continue

            self._pending_stop_acknowledgements.pop(source_rank, None)
