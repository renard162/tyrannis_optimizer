"""Calibrate the integration transport with real Manager IPC and spawn."""

from collections.abc import Iterator
from multiprocessing import get_context
from multiprocessing.managers import SyncManager
from threading import enumerate as enumerate_threads

import pytest

# pytest adds tests/ to pythonpath; basedpyright does not read that pytest setting.
from _support.distributed_backend import (  # pyright: ignore[reportImplicitRelativeImport]
    WAIT_TIMEOUT,
    IslandTransport,
    TestBackendCommunicationDriver,
    TestBackendCommunicationProcessor,
    exchange_in_worker,
)

from tyrannis.core.signals import LocalEvent

pytestmark = pytest.mark.multiprocess
EndpointPair = tuple[TestBackendCommunicationDriver, TestBackendCommunicationProcessor]


@pytest.fixture
def manager() -> Iterator[SyncManager]:
    with get_context("spawn").Manager() as runtime:
        yield runtime


@pytest.fixture
def endpoints(manager: SyncManager) -> Iterator[EndpointPair]:
    transports = {"island:0": IslandTransport.create(manager)}
    driver = TestBackendCommunicationDriver(["island:0"], transports)
    processor = TestBackendCommunicationProcessor("island:0", transports)
    try:
        yield driver, processor
    finally:
        try:
            processor.stop()
        finally:
            driver.stop()


def test_bidirectional_fifo(endpoints: EndpointPair) -> None:
    driver, processor = endpoints
    driver.start()
    processor.start()
    for message in ("first", "second"):
        processor.outgoing_queue.put_nowait(message)
        driver.outgoing_queues["island:0"].put_nowait(message)
    assert [
        driver.incoming_queues["island:0"].get(timeout=WAIT_TIMEOUT) for _ in range(2)
    ] == ["first", "second"]
    assert [processor.messages.get(timeout=WAIT_TIMEOUT) for _ in range(2)] == [
        "first",
        "second",
    ]


def test_message_signal_can_be_detached(endpoints: EndpointPair) -> None:
    driver, processor = endpoints
    signal = LocalEvent()
    driver.start()
    processor.start()
    processor.set_message_signal(signal)
    driver.outgoing_queues["island:0"].put_nowait("signaled")
    assert processor.messages.get(timeout=WAIT_TIMEOUT) == "signaled"
    # Detachment synchronizes with the receive callback, avoiding a race between
    # insertion in the queue and signal.set().
    processor.set_message_signal(None)
    assert signal.is_set()
    signal.clear()
    driver.outgoing_queues["island:0"].put_nowait("detached")
    assert processor.messages.get(timeout=WAIT_TIMEOUT) == "detached"
    processor.stop()
    assert not signal.is_set()


def test_lifecycle_owns_and_releases_threads(manager: SyncManager) -> None:
    transports = {"island:0": IslandTransport.create(manager)}
    before = set(enumerate_threads())
    driver = TestBackendCommunicationDriver(["island:0"], transports)
    processor = TestBackendCommunicationProcessor("island:0", transports)
    assert set(enumerate_threads()) == before
    with pytest.raises(Exception):  # noqa: B017
        _ = processor.messages
    active = set(enumerate_threads()) - before
    try:
        driver.start()
        processor.start()
        active = set(enumerate_threads()) - before
        assert active
        assert all(not thread.daemon for thread in active)
        processor.outgoing_queue.put_nowait("running")
        assert driver.incoming_queues["island:0"].get(timeout=WAIT_TIMEOUT) == "running"
    finally:
        try:
            processor.stop()
        finally:
            driver.stop()
    assert all(not thread.is_alive() for thread in active)
    with pytest.raises(Exception):  # noqa: B017
        _ = processor.outgoing_queue


def test_configured_processor_crosses_spawn_boundary(endpoints: EndpointPair) -> None:
    driver, processor = endpoints
    driver.start()
    with get_context("spawn").Pool(1) as pool:
        task = pool.apply_async(exchange_in_worker, (processor,))
        driver.outgoing_queues["island:0"].put_nowait("round trip")
        assert (
            driver.incoming_queues["island:0"].get(timeout=WAIT_TIMEOUT) == "round trip"
        )
        task.get(timeout=WAIT_TIMEOUT)
