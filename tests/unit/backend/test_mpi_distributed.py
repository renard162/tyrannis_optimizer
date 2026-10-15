"""MPI island orchestration contracts without an MPI runtime."""

from __future__ import annotations

import sys
from collections.abc import Callable
from importlib.machinery import ModuleSpec
from pickle import loads
from types import ModuleType, TracebackType
from typing import Protocol, Self, final
from unittest.mock import Mock

import pytest

from tests._support.numerics import BASE_SEED
from tests._support.processors import (
    ProcessorAlgorithmDouble,
    ProcessorMigrationDriverDouble,
    configure_processor_for_dispatch,
)
from tyrannis.backend.distributed.communication.mpi_communication import (
    MPICommunicationProcessor,
)
from tyrannis.backend.distributed.mpi_distributed import (
    MPIDistributed,
    MPIDistributedCostFunctionWrapper,
    _run_processor,  # pyright: ignore[reportPrivateUsage]  # Explicit worker boundary contract.
)
from tyrannis.core.processor import ProcessorBase
from tyrannis.core.results import HistoryConfig, ProcessorResult
from tyrannis.core.signals import LocalEvent
from tyrannis.processor.serial import Serial


@final
class _World:
    def __init__(self) -> None:
        self.rank = 0
        self.size = 3

    def Get_rank(self) -> int:
        return self.rank

    def Get_size(self) -> int:
        return self.size


@final
class _Executor:
    def __init__(self) -> None:
        self.num_workers = 2
        self.events: list[str] = []
        self.submit = Mock()

    def __enter__(self) -> Self:
        self.events.append("enter")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.events.append("exit")

    def bootup(self, *, wait: bool) -> None:
        assert wait is True
        self.events.append("bootup")


class _Context(Protocol):
    def initialize_context(
        self,
        algorithm: ProcessorAlgorithmDouble,
        n_iter: int,
        n_particles: int,
        migration: ProcessorMigrationDriverDouble,
        processor: ProcessorBase[LocalEvent] | None,
        fitness_failure_strategy: str,
        history_config: HistoryConfig,
        seed: int | None,
    ) -> None: ...


def _initialize(
    context: _Context,
    algorithm: ProcessorAlgorithmDouble,
    migration: ProcessorMigrationDriverDouble,
    history: HistoryConfig,
    processor: ProcessorBase[LocalEvent] | None = None,
) -> None:
    context.initialize_context(
        algorithm=algorithm,
        n_iter=1,
        n_particles=2,
        migration=migration,
        processor=processor,
        fitness_failure_strategy="raise",
        history_config=history,
        seed=BASE_SEED,
    )


@pytest.fixture
def mpi_runtime(monkeypatch: pytest.MonkeyPatch) -> tuple[_World, _Executor, Mock]:
    # Existing MPI helpers model particle batches or transport threads, neither
    # of which belongs to this backend's island orchestration boundary.
    world = _World()
    executor = _Executor()
    factory = Mock(return_value=executor)
    package = ModuleType("mpi4py")
    package.__spec__ = ModuleSpec("mpi4py", loader=None, is_package=True)
    mpi = ModuleType("mpi4py.MPI")
    futures = ModuleType("mpi4py.futures")
    monkeypatch.setattr(mpi, "COMM_WORLD", world, raising=False)
    monkeypatch.setattr(package, "MPI", mpi, raising=False)
    monkeypatch.setattr(futures, "MPIPoolExecutor", factory, raising=False)
    for name, module in {
        "mpi4py": package,
        "mpi4py.MPI": mpi,
        "mpi4py.futures": futures,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return world, executor, factory


@pytest.mark.parametrize(
    ("rank", "size"), [(1, 3), (0, 1)], ids=["not-driver", "no-worker"]
)
def test_constructor_rejects_unusable_topology(
    mpi_runtime: tuple[_World, _Executor, Mock], rank: int, size: int
) -> None:
    world, _, factory = mpi_runtime
    world.rank = rank
    world.size = size

    with pytest.raises(Exception):  # noqa: B017
        _ = MPIDistributed()

    factory.assert_not_called()


@pytest.mark.usefixtures("mpi_runtime")
def test_context_connects_worker_topology_to_migration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = MPIDistributed()
    algorithm = ProcessorAlgorithmDouble()
    configure = Mock()
    monkeypatch.setattr(algorithm, "configure", configure)
    migration = ProcessorMigrationDriverDouble()
    initialize_migration = Mock()
    monkeypatch.setattr(migration, "initialize_context", initialize_migration)
    communication = object()
    communication_factory = Mock(return_value=communication)
    monkeypatch.setattr(
        "tyrannis.backend.distributed.mpi_distributed.MPICommunicationDriver",
        communication_factory,
    )
    history = HistoryConfig(migration=True)

    _initialize(backend, algorithm, migration, history)

    assert backend.identifier == "MPIDistributed"
    configure.assert_called_once_with(
        identifier="MPIDistributed|algorithm",
        cost_function_wrapper=MPIDistributedCostFunctionWrapper,
        seed=BASE_SEED,
    )
    communication_factory.assert_called_once_with(island_ids=["island:0", "island:1"])
    initialize_migration.assert_called_once_with(
        communication_driver=communication,
        communication_processor_class=MPICommunicationProcessor,
        communication_processor_kargs={},
        history_config=history,
        seed=BASE_SEED,
    )


@pytest.fixture
def ready_backend(
    mpi_runtime: tuple[_World, _Executor, Mock], monkeypatch: pytest.MonkeyPatch
) -> tuple[MPIDistributed, Serial]:
    _, executor, _ = mpi_runtime
    backend = MPIDistributed()
    processor = Serial()
    _ = configure_processor_for_dispatch(processor, n_iterations=1, n_particles=2)
    algorithm = ProcessorAlgorithmDouble()
    monkeypatch.setattr(algorithm, "configure", Mock())
    migration = ProcessorMigrationDriverDouble()
    monkeypatch.setattr(migration, "initialize_context", Mock())
    monkeypatch.setattr(migration, "start", lambda: executor.events.append("start"))
    monkeypatch.setattr(migration, "stop", lambda: executor.events.append("stop"))
    _initialize(backend, algorithm, migration, HistoryConfig(), processor)
    return backend, processor


def test_execute_serializes_one_processor_per_worker_and_consolidates(
    mpi_runtime: tuple[_World, _Executor, Mock],
    ready_backend: tuple[MPIDistributed, Serial],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, executor, factory = mpi_runtime
    backend, processor = ready_backend
    workers = [Serial(), Serial()]
    for identifier, worker in zip(["island:0", "island:1"], workers, strict=True):
        worker.set_identifier(identifier)
    results = {worker.identifier: ProcessorResult() for worker in workers}
    dispatched: list[str] = []
    resolutions: list[str] = []

    def create_pool() -> None:
        assert executor.events[-1] == "bootup"
        executor.events.append("prepare")
        processor.processors_pool.update(
            {worker.identifier: worker for worker in workers}
        )

    def submit(
        worker: Callable[[bytes], tuple[str, ProcessorResult]], payload: bytes
    ) -> Mock:
        assert worker is _run_processor
        assert executor.events[-1] == "start"
        assert isinstance(payload, bytes)
        snapshot: object = loads(payload)  # pyright: ignore[reportAny]  # Narrow the pickle boundary below.
        assert isinstance(snapshot, Serial)
        assert snapshot is not processor.processors_pool[snapshot.identifier]
        dispatched.append(snapshot.identifier)

        def resolve() -> tuple[str, ProcessorResult]:
            assert executor.events[-1] == "start"
            resolutions.append(snapshot.identifier)
            return snapshot.identifier, results[snapshot.identifier]

        return Mock(result=Mock(side_effect=resolve))

    def consolidate() -> None:
        assert sorted(resolutions) == sorted(results)
        # No public API exposes the island results before base consolidation.
        assert vars(backend)["_local_bests"] == results
        executor.events.append("consolidate")

    initialize = Mock(side_effect=create_pool)
    update = Mock(side_effect=consolidate)
    monkeypatch.setattr(backend, "init_processors", initialize)
    monkeypatch.setattr(backend, "update_result", update)
    executor.submit.side_effect = submit

    backend.execute()

    factory.assert_called_once_with()
    initialize.assert_called_once_with()
    update.assert_called_once_with()
    assert sorted(dispatched) == sorted(results)
    assert executor.events == [
        "enter",
        "bootup",
        "prepare",
        "start",
        "consolidate",
        "stop",
        "exit",
    ]


@pytest.mark.parametrize("mismatch", ["runtime", "processors"])
def test_cardinality_mismatch_prevents_migration_and_dispatch(
    mpi_runtime: tuple[_World, _Executor, Mock],
    ready_backend: tuple[MPIDistributed, Serial],
    mismatch: str,
) -> None:
    _, executor, _ = mpi_runtime
    backend, processor = ready_backend
    processor.processors_pool.update({"island:0": Serial(), "island:1": Serial()})
    if mismatch == "runtime":
        executor.num_workers = 1
    else:
        _ = processor.processors_pool.pop("island:1")

    with pytest.raises(Exception):  # noqa: B017
        backend.execute()

    executor.submit.assert_not_called()
    assert executor.events == ["enter", "bootup", "exit"]


def test_future_failure_stops_migration_and_exits_executor(
    mpi_runtime: tuple[_World, _Executor, Mock],
    ready_backend: tuple[MPIDistributed, Serial],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, executor, _ = mpi_runtime
    backend, processor = ready_backend
    processor.processors_pool.update({"island:0": Serial(), "island:1": Serial()})

    def fail() -> None:
        assert executor.events[-1] == "start"
        executor.events.append("future_failure")
        raise ValueError("scripted future failure")

    executor.submit.return_value = Mock(result=Mock(side_effect=fail))
    update = Mock()
    monkeypatch.setattr(backend, "update_result", update)

    with pytest.raises(Exception):  # noqa: B017
        backend.execute()

    update.assert_not_called()
    assert executor.events == [
        "enter",
        "bootup",
        "start",
        "future_failure",
        "stop",
        "exit",
    ]


@pytest.mark.usefixtures("mpi_runtime")
@pytest.mark.parametrize(
    "fail_run", [False, True], ids=["return-result", "cleanup-on-failure"]
)
def test_worker_barrier_and_execution_context_lifecycle(
    monkeypatch: pytest.MonkeyPatch, fail_run: bool
) -> None:
    events: list[str] = []
    communicator = Mock(Barrier=Mock(side_effect=lambda: events.append("barrier")))
    get_workers = Mock(return_value=communicator)
    monkeypatch.setattr(
        sys.modules["mpi4py.futures"], "get_comm_workers", get_workers, raising=False
    )
    result = ProcessorResult(result={"fitness": 3.0})

    def run() -> None:
        events.append("run")
        if fail_run:
            raise ValueError("scripted processor failure")

    # Only the remote lifecycle is observed; processor internals have their own tests.
    processor = Mock(
        spec=ProcessorBase,
        identifier="island:0",
        result=result,
        initialize_execution_context=Mock(
            side_effect=lambda: events.append("initialize")
        ),
        run=Mock(side_effect=run),
        finalize_execution_context=Mock(side_effect=lambda: events.append("finalize")),
    )
    deserialize = Mock(return_value=processor)
    monkeypatch.setattr(
        "tyrannis.backend.distributed.mpi_distributed.cloudpickle.loads", deserialize
    )
    payload = b"serialized processor"

    if fail_run:
        with pytest.raises(Exception):  # noqa: B017
            _ = _run_processor(payload)
    else:
        returned = _run_processor(payload)
        assert returned == ("island:0", result)

    get_workers.assert_called_once_with()
    deserialize.assert_called_once_with(payload)
    assert events == ["barrier", "initialize", "run", "finalize"]
