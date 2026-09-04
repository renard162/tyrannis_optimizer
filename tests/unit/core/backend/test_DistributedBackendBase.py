import pytest
from doubles.algorithm import DummyAlgorithm
from doubles.backend import DummyBackend
from doubles.processor import DummyProcessor

from tyrannis.core.backend_distributed import DistributedBackendBase


class DummyDistributedBackend(DistributedBackendBase, DummyBackend):
    _n_executors = 3


def create_algorithm() -> DummyAlgorithm:
    algorithm = DummyAlgorithm()

    algorithm.initialize_context(
        fitness_function=lambda variables: sum(variables.values()),
        boundaries={"x": (-1.0, 1.0)},
    )

    return algorithm


def create_backend(
    processor: DummyProcessor | None = None,
) -> DummyDistributedBackend:
    backend = DummyDistributedBackend()

    algorithm = DummyAlgorithm()
    algorithm.initialize_context(
        fitness_function=lambda variables: sum(variables.values()),
        boundaries={"x": (-1.0, 1.0)},
    )

    backend.initialize_context(
        algorithm=algorithm,
        n_iter=10,
        n_particles=3,
        processor=processor,
        seed=42,
    )

    return backend


def test_init_processors_requires_processor() -> None:
    backend = create_backend()

    with pytest.raises(
        RuntimeError,
        match="Processor is not initialized",
    ):
        backend.init_processors()


def test_init_processors_creates_pool() -> None:
    processor = DummyProcessor()

    processor.initialize_context(
        algorithm=create_algorithm(),
        n_iter=10,
        n_particles=3,
        seed=42,
    )

    backend = create_backend(processor)

    backend.init_processors()

    assert list(processor.processors_pool) == [
        "island:0",
        "island:1",
        "island:2",
    ]


def test_update_result_selects_best_candidate() -> None:
    backend = create_backend()

    backend._local_bests = {
        "island:0": {
            "fitness": 5.0,
            "variables": {"x": 0.5},
        },
        "island:1": {
            "fitness": 2.0,
            "variables": {"x": 0.2},
        },
        "island:2": {
            "fitness": 8.0,
            "variables": {"x": 0.8},
        },
    }

    backend.update_result()

    assert backend.result == {
        "fitness": 2.0,
        "variables": {"x": 0.2},
    }


def test_update_result_ignores_invalid_candidates() -> None:
    backend = create_backend()

    backend._local_bests = {
        "island:0": None,
        "island:1": {
            "fitness": 2.0,
            "variables": {"x": 0.2},
        },
        "island:2": {
            "fitness": 8.0,
            "variables": {"x": 0.8},
        },
    }

    backend.update_result()

    assert backend.result == {
        "fitness": 2.0,
        "variables": {"x": 0.2},
    }


def test_update_result_ignores_non_float_fitness() -> None:
    backend = create_backend()

    backend._local_bests = {
        "island:0": {
            "fitness": 1,
            "variables": {"x": 0.1},
        },
        "island:1": {
            "fitness": 2.0,
            "variables": {"x": 0.2},
        },
    }

    backend.update_result()

    assert backend.result == {
        "fitness": 2.0,
        "variables": {"x": 0.2},
    }
