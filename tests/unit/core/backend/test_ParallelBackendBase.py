from typing import Any

from doubles.algorithm import DummyAlgorithm
from doubles.backend import DummyBackend

from tyrannis.core.backend_migration import MigrationDriverBase
from tyrannis.core.backend_parallel import ParallelBackendBase


class DummyMigration(MigrationDriverBase):
    def __init__(
        self,
        initial_iter: int = 1,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._migration_processor_init_kargs = {
            "initial_iter": initial_iter,
        }

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class DummyParallelBackend(ParallelBackendBase, DummyBackend):
    pass


def create_migration() -> DummyMigration:
    return DummyMigration()


def create_backend() -> DummyParallelBackend:
    backend = DummyParallelBackend()

    algorithm = DummyAlgorithm()
    algorithm.initialize_context(
        fitness_function=lambda variables: sum(variables.values()),
        boundaries={"x": (-1.0, 1.0)},
    )

    backend.initialize_context(
        algorithm=algorithm,
        n_iter=10,
        n_particles=3,
        migration=create_migration(),
        seed=42,
    )

    return backend


def test_init_particles() -> None:
    backend = create_backend()

    backend.init_particles()

    assert list(backend._algorithm.population) == [
        "DummyBackend|particle:0",
        "DummyBackend|particle:1",
        "DummyBackend|particle:2",
    ]

    population = backend._algorithm.population

    backend.init_particles()

    assert backend._algorithm.population is population


def test_update_result() -> None:
    backend = create_backend()

    backend._algorithm.create_particle(
        identifier="DummyBackend|particle:0",
        variables={"x": 0.5},
        fitness=1.0,
    )
    backend._algorithm._local_best = backend._algorithm.population[
        "DummyBackend|particle:0"
    ]

    backend.update_result()

    assert backend.result == {
        "identifier": "DummyBackend|particle:0",
        "fitness": 1.0,
        "variables": {"x": 0.5},
    }


def test_update_result_without_local_best() -> None:
    backend = create_backend()

    backend.update_result()

    assert backend.result is None
