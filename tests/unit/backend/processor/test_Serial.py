import numpy as np
from doubles.algorithm import DummyAlgorithm

from tyrannis.processor.serial import (
    Serial,
    SerialCostFunctionWrapper,
)
from tyrannis.core.backend_migration import (
    MigrationDriverBase,
    MigrationProcessorBase,
)
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
        self.population_calls: list[dict[str, float | None]] = []

    def initialize_loop_context(
        self,
        migration_signal: LocalEvent,
    ) -> None:
        pass

    def finalize_loop_context(self) -> None:
        pass

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
        population: dict[str, float | None],
        local_best: str | None,
        insert_arrival_particle,
        departure_particle,
    ) -> None:
        self.migration_control_calls.append(actual_iter)
        self.population_calls.append(population.copy())


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
            np.sum(
                np.fromiter(
                    variables.values(),
                    dtype=float,
                )
                ** 2
            )
        ),
        boundaries={"x": (-1.0, 1.0)},
    )

    algorithm.create_random_cache = lambda particle_ids: None

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
    assert not processor._stop_signal.is_set()

    migration_processor = processor._migration_processor

    assert migration_processor is not None
    assert migration_processor.start_calls == [  # type: ignore
        processor._stop_signal,
    ]


def test_finalize_execution_context() -> None:
    processor = create_processor()

    processor.initialize_execution_context()

    processor._stop_signal.set()

    migration_processor = processor._migration_processor

    assert migration_processor is not None

    processor.finalize_execution_context()

    assert processor._stop_signal.is_set()
    assert migration_processor.stop_calls == 1  # type: ignore


def test_run() -> None:
    processor = create_processor()

    processor.initialize_execution_context()
    processor.run()

    assert processor._algorithm is not None
    assert len(processor._algorithm.population) == 2

    migration_processor = processor._migration_processor

    assert migration_processor is not None
    assert migration_processor.migration_control_calls == [0, 1]  # type: ignore

    population_calls = migration_processor.population_calls  # type: ignore

    assert len(population_calls) == 2
    assert population_calls[0] == {}

    assert set(population_calls[1]) == set(processor._algorithm.population)

    assert population_calls[1] == {
        particle.identifier: particle.fitness
        for particle in processor._algorithm.population.values()
    }

    assert processor.population == population_calls[1]

    for particle in processor._algorithm.population.values():
        assert particle.fitness is not None

    assert processor._local_best is None
