import json
from unittest.mock import Mock

from doubles.algorithm import DummyAlgorithm

from tyrannis.backend.local import Local, LocalCostFunctionWrapper
from tyrannis.migration.island_isolation import IslandIsolation


def create_algorithm() -> DummyAlgorithm:
    algorithm = DummyAlgorithm()

    algorithm.initialize_context(
        fitness_function=lambda variables: float(
            sum(value**2 for value in variables.values())
        ),
        boundaries={"x": (-1.0, 1.0)},
    )

    return algorithm


def create_migration() -> IslandIsolation:
    return IslandIsolation()


def test_init() -> None:
    backend = Local()

    assert backend.identifier == "Local"
    assert backend._cost_function_wrapper is LocalCostFunctionWrapper
    assert backend._global_best_data is None


def test_execute() -> None:
    backend = Local()
    algorithm = create_algorithm()

    executor = Mock()
    executor.local_best = json.dumps(
        {
            "identifier": "particle:0",
            "fitness": 0.25,
            "variables": {"x": 0.5},
        }
    )

    processor = Mock()
    processor.processors_pool = {"island:0": executor}

    backend.initialize_context(
        algorithm=algorithm,
        n_iter=1,
        n_particles=1,
        migration=create_migration(),
        processor=processor,
        seed=42,
    )

    backend.execute()

    processor.create_processors_pool.assert_called_once_with(1)
    executor.initialize_execution_context.assert_called_once_with()
    executor.run.assert_called_once_with()
    processor.update_processors_pool.assert_called_once_with([executor])

    assert backend._global_best_data == executor.local_best
    assert backend.result == {
        "identifier": "particle:0",
        "fitness": 0.25,
        "variables": {"x": 0.5},
    }


def test_update_result() -> None:
    backend = Local()

    backend.initialize_context(
        algorithm=create_algorithm(),
        n_iter=1,
        n_particles=1,
        migration=create_migration(),
    )

    backend._global_best_data = json.dumps(
        {
            "identifier": "particle:0",
            "variables": {"x": 0.5},
            "fitness": 0.25,
            "velocity": {"x": 0.1},
        }
    )

    backend.update_result()

    assert backend.result == {
        "identifier": "particle:0",
        "fitness": 0.25,
        "variables": {"x": 0.5},
    }


def test_update_result_without_global_best() -> None:
    backend = Local()

    backend.initialize_context(
        algorithm=create_algorithm(),
        n_iter=1,
        n_particles=1,
        migration=create_migration(),
    )

    backend.update_result()

    assert backend.result is None
