"""Integration-only distributed runtime, independent of Spark and MPI.

The caller owns the spawn Manager and must keep it alive until communication
and workers have stopped. Only queue proxies cross the process boundary.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing import get_context
from multiprocessing.managers import SyncManager
from os import getpid
from queue import Empty, Queue
from threading import Barrier, Event, Lock, Thread
from time import monotonic
from typing import TypeVar, final, override

from tyrannis.core.algorithm import AlgorithmBase, CostFunctionWrapperBase, ParticleBase
from tyrannis.core.backend_communication import (
    CommunicationDriverBase,
    CommunicationProcessorBase,
)
from tyrannis.core.backend_distributed import DistributedBackendBase
from tyrannis.core.backend_migration import MigrationDriverBase
from tyrannis.core.processor import ProcessorBase
from tyrannis.core.results import HistoryConfig, ProcessorResult
from tyrannis.core.signals import EventProtocol
from tyrannis.processor.serial import Serial

ParticleT = TypeVar("ParticleT", bound=ParticleBase)
SignalT = TypeVar("SignalT", bound=EventProtocol)

WAIT_TIMEOUT = 30.0
_POLL_INTERVAL = 0.02


@dataclass(frozen=True)
class IslandTransport:
    """Opaque, picklable Manager queue pair for one island."""

    to_driver: Queue[str]
    to_processor: Queue[str]

    @classmethod
    def create(cls, manager: SyncManager) -> IslandTransport:
        return cls(manager.Queue(), manager.Queue())


MessageReceiver = Callable[[str], None]


@final
class _Bridge:
    """One FIFO pump; stop drains queued work within a bounded grace period."""

    def __init__(self, source: Queue[str], deliver: MessageReceiver) -> None:
        self._stopping = Event()
        self._errors: Queue[Exception] = Queue()
        self._source = source
        self._deliver = deliver
        self._deadline: float | None = None
        self._thread = Thread(target=self._run, name="test-communication")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._deadline = monotonic() + WAIT_TIMEOUT
        self._stopping.set()
        if self._thread.ident is not None:
            self._thread.join(timeout=WAIT_TIMEOUT)
        if self._thread.is_alive():
            raise TimeoutError("Test communication bridge did not stop.")
        if not self._errors.empty():
            raise RuntimeError("Test communication transport failed.") from (
                self._errors.get_nowait()
            )

    def _run(self) -> None:
        try:
            while True:
                if self._deadline is not None and monotonic() >= self._deadline:
                    raise TimeoutError("Test communication did not drain.")
                try:
                    message = self._source.get(timeout=_POLL_INTERVAL)
                except Empty:
                    if self._stopping.is_set():
                        return
                    continue
                self._deliver(message)
        except Exception as error:
            logging.getLogger(__name__).exception("Test communication pump failed")
            self._errors.put_nowait(error)


@final
class TestBackendCommunicationProcessor(CommunicationProcessorBase):
    """Local application queues connected to the driver's Manager transport."""

    __test__ = False

    def __init__(self, identifier: str, transports: dict[str, IslandTransport]) -> None:
        self._identifier = identifier
        self._transport = transports[identifier]
        self._bridges: list[_Bridge] = []
        self._messages: Queue[str] | None = None
        self._outgoing_queue: Queue[str] | None = None
        self._message_signal: EventProtocol | None = None
        self._signal_lock: Lock | None = None

    @property
    @override
    def messages(self) -> Queue[str]:
        if self._messages is None:
            raise RuntimeError("Test communication processor is not running.")
        return self._messages

    @property
    @override
    def outgoing_queue(self) -> Queue[str]:
        if self._outgoing_queue is None:
            raise RuntimeError("Test communication processor is not running.")
        return self._outgoing_queue

    @override
    def set_message_signal(self, message_signal: EventProtocol | None) -> None:
        lock = self._signal_lock
        if lock is None:
            self._message_signal = message_signal
        else:
            with lock:
                self._message_signal = message_signal

    def _receive(self, message: str) -> None:
        lock = self._signal_lock
        assert lock is not None
        with lock:
            self.messages.put_nowait(message)
            if self._message_signal is not None:
                self._message_signal.set()

    @override
    def start(self) -> None:
        if self._bridges:
            raise RuntimeError("Test communication is already running.")
        self._messages = Queue()
        self._outgoing_queue = Queue()
        self._signal_lock = Lock()
        self._bridges = [
            _Bridge(self._transport.to_processor, self._receive),
            _Bridge(self._outgoing_queue, self._transport.to_driver.put_nowait),
        ]
        try:
            for bridge in self._bridges:
                bridge.start()
        except BaseException:
            self.stop()
            raise

    @override
    def stop(self) -> None:
        try:
            _stop_bridges(self._bridges)
        finally:
            self._message_signal = None
            self._signal_lock = None
            self._messages = None
            self._outgoing_queue = None


@final
class TestBackendCommunicationDriver(CommunicationDriverBase):
    """One FIFO bridge per registered island; no migration policy."""

    __test__ = False

    def __init__(
        self, island_ids: list[str], transports: dict[str, IslandTransport]
    ) -> None:
        self._transports = {
            identifier: transports[identifier] for identifier in island_ids
        }
        self._incoming_queues = {identifier: Queue[str]() for identifier in island_ids}
        self._outgoing_queues = {identifier: Queue[str]() for identifier in island_ids}
        self._bridges: list[_Bridge] = []

    @property
    @override
    def incoming_queues(self) -> dict[str, Queue[str]]:
        return self._incoming_queues

    @property
    @override
    def outgoing_queues(self) -> dict[str, Queue[str]]:
        return self._outgoing_queues

    @override
    def start(self) -> None:
        if self._bridges:
            raise RuntimeError("Test communication is already running.")
        try:
            for identifier, transport in self._transports.items():
                bridges = [
                    _Bridge(
                        transport.to_driver,
                        self._incoming_queues[identifier].put_nowait,
                    ),
                    _Bridge(
                        self._outgoing_queues[identifier],
                        transport.to_processor.put_nowait,
                    ),
                ]
                self._bridges.extend(bridges)
                for bridge in bridges:
                    bridge.start()
        except BaseException:
            self.stop()
            raise

    @override
    def stop(self) -> None:
        _stop_bridges(self._bridges)


def _stop_bridges(bridges: list[_Bridge]) -> None:
    errors: list[Exception] = []
    for bridge in bridges:
        try:
            bridge.stop()
        except (RuntimeError, TimeoutError) as error:
            errors.append(error)
    bridges.clear()
    if errors:
        raise ExceptionGroup("Test communication shutdown failed.", errors)


@final
class TestDistributedCostFunctionWrapper(CostFunctionWrapperBase):
    """Neutral wrapper identifying the integration backend."""

    __test__ = False


@final
class TestDistributedBackend(DistributedBackendBase):
    """Exactly two Serial islands in one spawn pool; the caller owns Manager."""

    __test__ = False

    def __init__(self, manager: SyncManager) -> None:
        self._manager = manager
        self._identifier = "TestDistributedBackend"
        self._cost_function_wrapper = TestDistributedCostFunctionWrapper
        self._n_executors = 2
        self._local_bests = {}
        self._worker_pids: dict[str, int] = {}

    @property
    def n_executors(self) -> int:
        return self._n_executors

    @property
    def local_bests(self) -> dict[str, ProcessorResult]:
        return self._local_bests

    @property
    def worker_pids(self) -> dict[str, int]:
        """Calibration evidence only; never included in optimizer results."""
        return self._worker_pids.copy()

    # Keep the requested signature style without a trailing argument comma.
    # fmt: off
    @override
    def initialize_context(
        self,
        algorithm: AlgorithmBase[ParticleT],
        n_iter: int,
        n_particles: int,
        migration: MigrationDriverBase,
        processor: ProcessorBase[SignalT] | None,
        fitness_failure_strategy: str,
        history_config: HistoryConfig,
        seed: int | None
    ) -> None:
        # fmt: on
        # The production base omits its Algorithm/Processor generic arguments.
        super().initialize_context(  # pyright: ignore[reportUnknownMemberType]
            algorithm=algorithm,
            n_iter=n_iter,
            n_particles=n_particles,
            migration=migration,
            processor=processor,
            fitness_failure_strategy=fitness_failure_strategy,
            history_config=history_config,
            seed=seed,
        )
        self._migration = migration
        _ = self._require_serial()
        island_ids = ["island:0", "island:1"]
        transports = {
            identifier: IslandTransport.create(self._manager)
            for identifier in island_ids
        }
        communication_driver = TestBackendCommunicationDriver(island_ids, transports)
        self._migration.initialize_context(
            communication_driver=communication_driver,
            communication_processor_class=TestBackendCommunicationProcessor,
            communication_processor_kargs={"transports": transports},
            history_config=history_config,
            seed=seed,
        )
        self._local_bests = {}
        self._worker_pids.clear()

    def _require_serial(self) -> Serial:
        # BackendBase leaves the Processor signal type unspecified; narrow it here.
        processor: object = self._processor  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        if not isinstance(processor, Serial):
            raise TypeError("TestDistributedBackend requires a Serial processor.")
        return processor

    @override
    def execute(self) -> None:
        processor = self._require_serial()
        if not processor.processors_pool:
            self.init_processors()
        processors = processor.processors_pool
        if set(processors) != {"island:0", "island:1"}:
            raise RuntimeError("TestDistributedBackend requires exactly two islands.")

        self._local_bests = {}
        self._worker_pids.clear()
        barrier = self._manager.Barrier(2, timeout=WAIT_TIMEOUT)
        try:
            self._migration.start()
            # Both tasks must occupy a worker before either can pass the barrier.
            # Pool's context terminates and reaps workers even on task failure.
            with get_context("spawn").Pool(processes=2) as pool:
                tasks = [
                    pool.apply_async(_run_island, (island, barrier))
                    for island in processors.values()
                ]
                deadline = monotonic() + WAIT_TIMEOUT
                for task in tasks:
                    identifier, result, pid = task.get(
                        timeout=max(0.0, deadline - monotonic())
                    )
                    self._local_bests[identifier] = result
                    self._worker_pids[identifier] = pid
                self.update_result()
        finally:
            self._migration.stop()


def _run_island(
    processor: Serial, barrier: Barrier
) -> tuple[str, ProcessorResult, int]:
    """Materialize execution resources in the worker, then always finalize them."""
    _ = barrier.wait(timeout=WAIT_TIMEOUT)
    try:
        processor.initialize_execution_context()
        processor.run()
        return processor.identifier, processor.result, getpid()
    finally:
        processor.finalize_execution_context()


def exchange_in_worker(processor: TestBackendCommunicationProcessor) -> None:
    """Echo one IPC message from a module importable by spawn without repo root."""
    try:
        processor.start()
        message = processor.messages.get(timeout=WAIT_TIMEOUT)
        processor.outgoing_queue.put_nowait(message)
    finally:
        processor.stop()
