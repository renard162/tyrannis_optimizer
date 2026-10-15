"""MPI transport contracts with a local peer and a cooperative scheduler."""

import sys
from collections import deque
from collections.abc import Callable
from threading import Event
from types import ModuleType
from typing import final, override

import pytest

from tyrannis.backend.distributed.communication import mpi_communication as transport
from tyrannis.backend.distributed.communication.mpi_communication import (
    MPICommunicationDriver,
    MPICommunicationProcessor,
)


@final
class MPIError(Exception):
    pass


@final
class Status:
    def __init__(self) -> None:
        self.source = -1

    def Get_source(self) -> int:
        return self.source


@final
class Peer:
    """Script only the MPI calls used by this transport; never import MPI."""

    def __init__(self) -> None:
        self.rank = 1
        self.inbox: deque[tuple[int, object]] = deque()
        self.sent: list[tuple[int, object]] = []
        self.attempts: list[tuple[int, object]] = []
        self.fail_once: list[object] = []
        self.calls = 0
        self.tags: set[int] = set()

    def Get_rank(self) -> int:
        self.calls += 1
        return self.rank

    def Iprobe(self, *, source: int, tag: int) -> bool:
        self.tags.add(tag)
        self.calls += 1
        return bool(self.inbox) and (source == -1 or self.inbox[0][0] == source)

    def recv(self, *, source: int, tag: int, status: Status | None = None) -> object:
        self.tags.add(tag)
        self.calls += 1
        sender, payload = self.inbox.popleft()
        assert source == -1 or source == sender
        if status is not None:
            status.source = sender
        return payload

    def send(self, payload: object, *, dest: int, tag: int) -> None:
        self.tags.add(tag)
        self.calls += 1
        self.attempts.append((dest, payload))
        if payload in self.fail_once:
            self.fail_once.remove(payload)
            raise MPIError
        self.sent.append((dest, payload))


@final
class ScheduledThread:
    """Defer the real loop to join; scripted event waits advance the peer."""

    def __init__(self, *, target: Callable[[], None], name: str, daemon: bool) -> None:
        self.target = target
        self.started = False
        self.joined = False
        self.executing = False
        self.finished = False

    def start(self) -> None:
        self.started = True

    def is_alive(self) -> bool:
        return self.started and not self.joined

    def run(self) -> None:
        self.executing = True
        try:
            self.target()
            self.finished = True
        finally:
            self.executing = False

    def join(self) -> None:
        # stop() has cleared the driver's running event. Complete its shutdown
        # tail before allowing stop() to reset resources; the outer loop unwinds.
        if self.executing:
            self.target()
        elif not self.finished:
            self.run()
        self.joined = True


@final
class Runtime:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.peer = Peer()
        self.threads: list[ScheduledThread] = []
        self.events: list[Event] = []
        self.steps: deque[Callable[[], None]] = deque()
        self.initialized = True
        self.finalized = False
        self.thread_level = 3
        self.queries = 0

        mpi = ModuleType("mpi4py.MPI")
        for name, value in {
            "COMM_WORLD": self.peer,
            "THREAD_MULTIPLE": 3,
            "ANY_SOURCE": -1,
            "Exception": MPIError,
            "Status": Status,
            "Is_initialized": self.is_initialized,
            "Is_finalized": lambda: self.finalized,
            "Query_thread": lambda: self.thread_level,
        }.items():
            monkeypatch.setattr(mpi, name, value, raising=False)
        package = ModuleType("mpi4py")
        monkeypatch.setattr(package, "MPI", mpi, raising=False)
        monkeypatch.setitem(sys.modules, "mpi4py", package)
        monkeypatch.setitem(sys.modules, "mpi4py.MPI", mpi)
        monkeypatch.setattr(transport, "Thread", self.make_thread)

        steps = self.steps

        class ScheduledEvent(Event):
            @override
            def wait(self, timeout: float | None = None) -> bool:
                assert steps, "Communication loop did not finish its scripted protocol"
                steps.popleft()()
                return self.is_set()

        def make_event() -> Event:
            event = ScheduledEvent()
            self.events.append(event)
            return event

        monkeypatch.setattr(transport, "ThreadEvent", make_event)

    def is_initialized(self) -> bool:
        self.queries += 1
        return self.initialized

    def make_thread(
        self, *, target: Callable[[], None], name: str, daemon: bool
    ) -> ScheduledThread:
        thread = ScheduledThread(target=target, name=name, daemon=daemon)
        self.threads.append(thread)
        return thread


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> Runtime:
    return Runtime(monkeypatch)


def assert_processor_inactive(processor: MPICommunicationProcessor) -> None:
    with pytest.raises(Exception):  # noqa: B017
        _ = processor.messages
    with pytest.raises(Exception):  # noqa: B017
        _ = processor.outgoing_queue


def test_construction_is_passive_and_start_requires_usable_mpi(
    runtime: Runtime,
) -> None:
    processor = MPICommunicationProcessor("a")
    driver = MPICommunicationDriver(["a", "b"])
    assert_processor_inactive(processor)
    assert runtime.queries == runtime.peer.calls == 0
    assert not runtime.threads and not runtime.events

    # Each prerequisite excludes a different unusable runtime state.
    for initialized, finalized, level in [
        (False, False, 3),
        (True, True, 3),
        (True, False, 2),
    ]:
        runtime.initialized = initialized
        runtime.finalized = finalized
        runtime.thread_level = level
        with pytest.raises(Exception):  # noqa: B017
            processor.start()
        with pytest.raises(Exception):  # noqa: B017
            driver.start()
    assert_processor_inactive(processor)
    assert not runtime.threads
    assert runtime.peer.calls == 0


def test_start_rejects_the_wrong_rank_for_each_role(runtime: Runtime) -> None:
    runtime.peer.rank = 0
    processor = MPICommunicationProcessor("a")
    with pytest.raises(Exception):  # noqa: B017
        processor.start()
    assert_processor_inactive(processor)
    runtime.peer.rank = 1
    with pytest.raises(Exception):  # noqa: B017
        MPICommunicationDriver(["a"]).start()
    assert not runtime.threads
    assert not runtime.peer.sent


def test_rejected_registration_leaves_processor_restartable(runtime: Runtime) -> None:
    processor = MPICommunicationProcessor("a")
    runtime.peer.inbox.append((0, ("rejected", "a")))
    with pytest.raises(Exception):  # noqa: B017
        processor.start()
    assert runtime.peer.sent == [(0, ("register", "a"))]
    assert_processor_inactive(processor)
    assert not runtime.threads

    runtime.peer.inbox.append((0, ("registered", "a")))
    processor.start()
    assert processor.messages.empty()
    assert processor.outgoing_queue.empty()
    runtime.steps.append(lambda: runtime.peer.inbox.append((0, ("stopped", "a"))))
    processor.stop()
    assert runtime.threads[0].joined
    assert_processor_inactive(processor)


def test_processor_delivers_and_drains_fifo_before_stop_ack(runtime: Runtime) -> None:
    processor = MPICommunicationProcessor("a")
    signal = Event()
    processor.set_message_signal(signal)
    runtime.peer.inbox.append((0, ("registered", "a")))
    processor.start()
    incoming, outgoing = processor.messages, processor.outgoing_queue
    assert incoming is not outgoing
    assert runtime.peer.sent == [(0, ("register", "a"))]
    assert not signal.is_set()
    runtime.peer.inbox.extend([(0, "in-one"), (0, "in-two")])
    outgoing.put("out-one")
    outgoing.put("out-two")
    runtime.peer.fail_once.append("out-one")

    def after_failure() -> None:
        assert signal.is_set()
        assert [incoming.get_nowait(), incoming.get_nowait()] == ["in-one", "in-two"]
        assert incoming.empty()
        assert processor.messages is incoming
        assert processor.outgoing_queue is outgoing
        assert runtime.peer.sent == [(0, ("register", "a"))]
        assert runtime.peer.attempts[-1] == (0, "out-one")
        signal.clear()
        processor.set_message_signal(None)
        runtime.peer.inbox.append((0, "unsignaled"))

    def after_drain() -> None:
        assert runtime.peer.sent == [
            (0, ("register", "a")),
            (0, "out-one"),
            (0, "out-two"),
            (0, ("stop", "a")),
        ]
        assert runtime.peer.attempts[1:4] == [
            (0, "out-one"),
            (0, "out-one"),
            (0, "out-two"),
        ]
        assert incoming.get_nowait() == "unsignaled"
        assert not signal.is_set()
        # An acknowledgement for another processor must not finish shutdown.
        runtime.peer.inbox.append((0, ("stopped", "other")))

    def acknowledge() -> None:
        assert processor.messages is incoming
        runtime.peer.inbox.append((0, ("stopped", "a")))

    runtime.steps.extend([after_failure, after_drain, acknowledge])
    processor.stop()
    assert not runtime.steps
    assert incoming.empty() and outgoing.empty()
    assert runtime.threads[0].joined
    assert_processor_inactive(processor)


def test_driver_registration_is_bijective_and_routes_only_registered_sources(
    runtime: Runtime,
) -> None:
    runtime.peer.rank = 0
    driver = MPICommunicationDriver(["a", "b"])
    incoming, outgoing = driver.incoming_queues, driver.outgoing_queues
    endpoints = [*incoming.values(), *outgoing.values()]
    assert len({id(queue) for queue in endpoints}) == 4
    driver.start()
    runtime.peer.inbox.extend(
        [
            (1, ("register", "a")),
            (2, ("register", "a")),  # One island cannot belong to two ranks.
            (1, ("register", "b")),  # One rank cannot belong to two islands.
            (3, ("register", "unknown")),
            (2, ("register", "b")),
            (1, "a-one"),
            (2, "b-one"),
            (1, "a-two"),
            (3, "still-unregistered"),
        ]
    )
    outgoing["a"].put("to-a")
    outgoing["b"].put("to-b")

    def verify_routing_and_stop() -> None:
        assert runtime.peer.sent[:5] == [
            (1, ("registered", "a")),
            (2, ("rejected", "a")),
            (1, ("rejected", "b")),
            (3, ("rejected", "unknown")),
            (2, ("registered", "b")),
        ]
        assert set(runtime.peer.sent[5:]) == {(1, "to-a"), (2, "to-b")}
        assert [incoming["a"].get_nowait(), incoming["a"].get_nowait()] == [
            "a-one",
            "a-two",
        ]
        assert incoming["b"].get_nowait() == "b-one"
        assert all(queue.empty() for queue in endpoints)
        assert driver.incoming_queues is incoming
        assert driver.outgoing_queues is outgoing
        assert [*incoming.values(), *outgoing.values()] == endpoints
        driver.stop()

    runtime.steps.append(verify_routing_and_stop)
    runtime.threads[0].run()
    assert runtime.threads[0].finished and runtime.threads[0].joined


def test_driver_retries_fifo_and_stop_ack_then_allows_rank_reassignment(
    runtime: Runtime,
) -> None:
    runtime.peer.rank = 0
    driver = MPICommunicationDriver(["a"])
    driver.start()
    runtime.peer.inbox.append((1, ("register", "a")))
    driver.outgoing_queues["a"].put("one")
    driver.outgoing_queues["a"].put("two")
    runtime.peer.fail_once.extend(["one", ("stopped", "a")])

    def after_failed_send() -> None:
        assert runtime.peer.sent == [(1, ("registered", "a"))]
        assert runtime.peer.attempts[-1] == (1, "one")
        # A STOP from another rank cannot unregister the legitimate endpoint.
        runtime.peer.inbox.append((2, ("stop", "a")))

    def after_retry() -> None:
        assert runtime.peer.sent == [(1, ("registered", "a")), (1, "one"), (1, "two")]
        assert runtime.peer.attempts[1:] == [(1, "one"), (1, "one"), (1, "two")]
        runtime.peer.inbox.append((1, ("stop", "a")))

    def after_failed_ack() -> None:
        assert runtime.peer.attempts[-1] == (1, ("stopped", "a"))
        assert (1, ("stopped", "a")) not in runtime.peer.sent
        runtime.peer.inbox.extend(
            [(1, "after-stop"), (2, ("register", "a")), (2, "new-owner")]
        )
        driver.outgoing_queues["a"].put("new-destination")

    def verify_ack_and_stop() -> None:
        assert runtime.peer.attempts.count((1, ("stopped", "a"))) == 2
        assert runtime.peer.sent.count((1, ("stopped", "a"))) == 1
        assert (2, ("registered", "a")) in runtime.peer.sent
        assert runtime.peer.sent[-1] == (2, "new-destination")
        assert driver.incoming_queues["a"].get_nowait() == "new-owner"
        assert driver.incoming_queues["a"].empty()
        driver.stop()

    runtime.steps.extend(
        [after_failed_send, after_retry, after_failed_ack, verify_ack_and_stop]
    )
    runtime.threads[0].run()
    assert not runtime.steps
    assert runtime.threads[0].finished and runtime.threads[0].joined


def test_driver_restart_does_not_reuse_registrations_or_pending_transmissions(
    runtime: Runtime,
) -> None:
    runtime.peer.rank = 0
    driver = MPICommunicationDriver(["a", "b"])
    driver.start()
    runtime.peer.inbox.extend(
        [(1, ("register", "a")), (2, ("register", "b")), (2, ("stop", "b"))]
    )
    driver.outgoing_queues["a"].put("obsolete")
    runtime.peer.fail_once.extend(["obsolete", ("stopped", "b"), ("stopped", "b")])

    def stop_with_pending_work() -> None:
        assert (1, "obsolete") in runtime.peer.attempts
        assert (2, ("stopped", "b")) in runtime.peer.attempts
        assert runtime.peer.sent == [(1, ("registered", "a")), (2, ("registered", "b"))]
        driver.stop()

    runtime.steps.append(stop_with_pending_work)
    runtime.threads[0].run()
    assert runtime.threads[0].finished and runtime.threads[0].joined
    runtime.peer.sent.clear()
    runtime.peer.attempts.clear()

    driver.start()
    runtime.peer.inbox.append((1, "stale-owner"))
    driver.outgoing_queues["a"].put("fresh")

    def register_new_owners() -> None:
        assert not runtime.peer.attempts
        assert driver.incoming_queues["a"].empty()
        runtime.peer.inbox.extend([(3, ("register", "a")), (4, ("register", "b"))])

    def verify_fresh_execution() -> None:
        assert runtime.peer.sent == [
            (3, ("registered", "a")),
            (4, ("registered", "b")),
            (3, "fresh"),
        ]
        driver.stop()

    runtime.steps.extend([register_new_owners, verify_fresh_execution])
    runtime.threads[1].run()
    assert not runtime.steps
    assert runtime.threads[1].finished and runtime.threads[1].joined
