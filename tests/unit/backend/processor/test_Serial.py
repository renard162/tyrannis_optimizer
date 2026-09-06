import numpy as np
from doubles.algorithm import DummyAlgorithm

from tyrannis.backend.processor.serial import (
    Serial,
    SerialCostFunctionWrapper,
)
from tyrannis.core.backend_migration import (
    MigrationDriverBase,
    MigrationProcessorBase,
)
from tyrannis.core.processor import LocalEvent


class DummyMigrationProcessor(MigrationProcessorBase):
    def __init__(
        self,
        initial_iter: int = 1,
        communication_processor=None,
        *args,
        **kwargs,
    ) -> None:
        self._initial_iter = initial_iter
        self._communication_processor = communication_processor

    def start(
        self,
        wait_signal: LocalEvent,
        stop_signal: LocalEvent,
    ) -> None:
        pass

    def stop(self) -> None:
        pass

    def check_particles(self) -> None:
        pass


class DummyMigrationDriver(MigrationDriverBase):
    _processor_class = DummyMigrationProcessor

    def __init__(
        self,
        initial_iter: int = 1,
        *args,
        **kwargs,
    ) -> None:
        self._migration_processor_init_kargs = {
            "initial_iter": initial_iter,
        }

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


def create_processor() -> Serial:
    algorithm = DummyAlgorithm()
    algorithm.initialize_context(
        fitness_function=lambda variables: float(
            np.sum(np.fromiter(variables.values(), dtype=float) ** 2)
        ),
        boundaries={"x": (-1.0, 1.0)},
    )

    migration_driver = DummyMigrationDriver()

    processor = Serial()
    processor.initialize_context(
        algorithm=algorithm,
        n_iter=1,
        n_particles=2,
        migration_driver=migration_driver,
        seed=42,
    )

    processor._migration_processor = DummyMigrationProcessor()

    return processor


def test_init() -> None:
    processor = Serial()

    assert processor._cost_function_wrapper is SerialCostFunctionWrapper


def test_initialize_execution_context() -> None:
    processor = create_processor()

    processor.initialize_execution_context()

    assert isinstance(processor._stop_signal, LocalEvent)
    assert isinstance(processor._wait_signal, LocalEvent)

    assert not processor._stop_signal.is_set()
    assert not processor._wait_signal.is_set()


def test_finalize_execution_context() -> None:
    processor = create_processor()

    processor.initialize_execution_context()
    processor._stop_signal.set()
    processor._wait_signal.set()

    processor.finalize_execution_context()

    assert processor._stop_signal.is_set()
    assert processor._wait_signal.is_set()


def test_run() -> None:
    processor = create_processor()

    processor.initialize_execution_context()
    processor.run()

    assert processor._algorithm is not None
    assert len(processor._algorithm.population) == 2

    assert processor._status.actual_iter == 1
    assert processor._status.population

    for particle in processor._algorithm.population.values():
        assert particle.fitness is not None
