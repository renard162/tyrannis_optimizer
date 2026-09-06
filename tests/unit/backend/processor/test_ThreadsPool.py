import pytest

from tyrannis.backend.processor.threads import (
    ThreadsPool,
    ThreadsPoolCostFunctionWrapper,
)
from tyrannis.core.backend_migration import MigrationProcessorBase
from tyrannis.core.signals import LocalEvent


class DummyMigrationProcessor(MigrationProcessorBase):
    def __init__(
        self,
        initial_iter: int = 1,
        communication_processor=None,
        *args,
        **kwargs,
    ) -> None:
        self._initial_iter = initial_iter
        self._synchronization_iter = None
        self._communication_processor = communication_processor

        self.start_calls: list[LocalEvent] = []
        self.stop_calls = 0
        self.migration_control_calls: list[int] = []

    def start(
        self,
        stop_signal: LocalEvent,
    ) -> None:
        self.start_calls.append(stop_signal)

    def stop(self) -> None:
        self.stop_calls += 1

    def migration_control(
        self,
        actual_iter: int,
        local_best: str | None,
        insert_arrival_particle,
        departure_particle,
    ) -> None:
        self.migration_control_calls.append(actual_iter)


def create_processor() -> ThreadsPool:
    processor = ThreadsPool()
    processor._migration_processor = DummyMigrationProcessor()

    return processor


def test_init() -> None:
    processor = ThreadsPool()

    assert processor._n_process is None
    assert processor._chunksize == 1
    assert processor._cost_function_wrapper is ThreadsPoolCostFunctionWrapper


def test_init_with_parameters() -> None:
    processor = ThreadsPool(
        n_process=4,
        chunksize=3,
    )

    assert processor._n_process == 4
    assert processor._chunksize == 3


def test_init_rejects_invalid_chunksize() -> None:
    with pytest.raises(
        ValueError,
        match="chunksize must be greater than zero",
    ):
        ThreadsPool(chunksize=0)


def test_initialize_execution_context() -> None:
    processor = create_processor()

    processor.initialize_execution_context()

    assert processor._stop_signal is not None
    assert not processor._stop_signal.is_set()

    migration_processor = processor._migration_processor

    assert migration_processor is not None
    assert migration_processor.start_calls == [  # type: ignore
        processor._stop_signal,
    ]


def test_finalize_execution_context() -> None:
    processor = create_processor()

    processor.initialize_execution_context()

    stop_signal = processor._stop_signal
    migration_processor = processor._migration_processor

    assert stop_signal is not None
    assert migration_processor is not None

    stop_signal.set()

    processor.finalize_execution_context()

    assert migration_processor.stop_calls == 1  # type: ignore

    assert isinstance(
        processor._stop_signal,
        LocalEvent,
    )

    assert not processor._stop_signal.is_set()


def test_initialize_execution_context_requires_migration_processor() -> None:
    processor = ThreadsPool()
    processor._migration_processor = None

    with pytest.raises(
        RuntimeError,
        match="Migration processor has not been initialized.",
    ):
        processor.initialize_execution_context()


def test_run_requires_algorithm() -> None:
    processor = ThreadsPool()
    processor._algorithm = None  # type: ignore
    processor._n_iter = 0
    processor._n_particles = 0
    processor._stop_signal = LocalEvent()
    processor._migration_processor = DummyMigrationProcessor()

    with pytest.raises(
        AttributeError,
        match="population",
    ):
        processor.run()


def test_migration_control_delegates_to_migration_processor() -> None:
    processor = create_processor()

    processor._local_best = '{"fitness": 1.0}'

    migration_processor = processor._migration_processor

    assert migration_processor is not None

    processor.migration_control(5)

    assert migration_processor.migration_control_calls == [5]  # type: ignore
