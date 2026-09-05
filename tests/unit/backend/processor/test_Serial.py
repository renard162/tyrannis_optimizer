import numpy as np
from doubles.algorithm import DummyAlgorithm

from tyrannis.backend.processor.serial import (
    Serial,
    SerialCostFunctionWrapper,
)
from tyrannis.core.processor import LocalEvent


def create_processor() -> Serial:
    algorithm = DummyAlgorithm()
    algorithm.initialize_context(
        fitness_function=lambda variables: float(
            np.sum(np.fromiter(variables.values(), dtype=float) ** 2)
        ),
        boundaries={"x": (-1.0, 1.0)},
    )

    processor = Serial()
    processor.initialize_context(
        algorithm=algorithm,
        n_iter=1,
        n_particles=2,
        seed=42,
    )

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
