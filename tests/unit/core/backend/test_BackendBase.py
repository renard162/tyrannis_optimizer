import inspect
from typing import Any

from doubles.algorithm import DummyAlgorithm
from doubles.backend import DummyBackend
from doubles.processor import DummyProcessor

from tyrannis.core.backend import BackendBase
from tyrannis.core.backend_migration import MigrationDriverBase


class DummyMigration(MigrationDriverBase):
    _processor_class = None  # type: ignore
    _communication_driver = None  # type: ignore
    _communication_processor_class = None  # type: ignore

    def __init__(self, initial_iter: int = 1, *args: Any, **kwargs: Any) -> None:
        self._migration_processor_init_kargs = {
            "initial_iter": initial_iter,
        }

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


def create_migration() -> DummyMigration:
    return DummyMigration()


def create_algorithm() -> DummyAlgorithm:
    algorithm = DummyAlgorithm()

    algorithm.initialize_context(
        fitness_function=lambda variables: sum(variables.values()),
        boundaries={"x": (-1.0, 1.0)},
    )

    return algorithm


def create_backend() -> DummyBackend:
    backend = DummyBackend()

    algorithm = create_algorithm()
    processor = DummyProcessor()
    migration = create_migration()

    backend.initialize_context(
        algorithm=algorithm,
        n_iter=10,
        n_particles=20,
        migration=migration,
        processor=processor,
        seed=42,
    )

    return backend


def test_initialize_context() -> None:
    backend = DummyBackend()
    algorithm = create_algorithm()
    processor = DummyProcessor()
    migration = create_migration()

    backend.initialize_context(
        algorithm=algorithm,
        n_iter=10,
        n_particles=20,
        migration=migration,
        processor=processor,
        seed=42,
    )

    assert backend._algorithm is algorithm
    assert backend._n_iter == 10
    assert backend._n_particles == 20
    assert backend._migration is migration
    assert backend._processor is processor
    assert backend._fitness_failure_strategy == "invalidate"
    assert backend._seed == 42
    assert backend._result is None

    assert algorithm.identifier == "DummyBackend|algorithm"


def test_properties() -> None:
    backend = create_backend()

    assert backend.identifier == "DummyBackend"
    assert backend.result is None

    backend._result = {
        "identifier": "particle:0",
        "fitness": 1.0,
        "variables": {"x": 0.5},
    }

    assert backend.result == {
        "identifier": "particle:0",
        "fitness": 1.0,
        "variables": {"x": 0.5},
    }


def test_is_abstract() -> None:
    assert inspect.isabstract(BackendBase)
