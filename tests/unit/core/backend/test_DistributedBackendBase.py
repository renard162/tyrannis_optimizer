import inspect
from queue import Queue

import pytest

from doubles.algorithm import DummyAlgorithm
from doubles.backend import DummyBackend
from doubles.processor import DummyProcessor

from tyrannis.core.backend_communication import CommunicationProcessorBase
from tyrannis.core.backend_distributed import DistributedBackendBase
from tyrannis.core.backend_migration import (
    MigrationDriverBase,
    MigrationProcessorBase,
)
from tyrannis.core.processor import LocalEvent


class DummyCommunicationProcessor(CommunicationProcessorBase):
    def __init__(
        self,
        identification: str,
        **kwargs: object,
    ) -> None:
        self._identification = identification
        self._messages = Queue()
        self._outgoing_queue = Queue()

    @property
    def messages(self) -> Queue[str]:
        return self._messages

    @property
    def outgoing_queue(self) -> Queue[str]:
        return self._outgoing_queue

    def start(
        self,
        wait_signal: LocalEvent,
        stop_signal: LocalEvent,
    ) -> None:
        pass

    def stop(self) -> None:
        pass


class DummyMigrationProcessor(MigrationProcessorBase):
    def __init__(
        self,
        initial_iter: int,
        communication_processor: CommunicationProcessorBase,
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


class DummyMigration(MigrationDriverBase):
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


class DummyDistributedBackend(DistributedBackendBase, DummyBackend):
    _n_executors = 3

    def initialize_context(
        self,
        algorithm,
        n_iter,
        n_particles,
        migration,
        processor=None,
        fitness_failure_strategy="invalidate",
        seed=None,
    ) -> None:
        super().initialize_context(
            algorithm=algorithm,
            n_iter=n_iter,
            n_particles=n_particles,
            migration=migration,
            processor=processor,
            fitness_failure_strategy=fitness_failure_strategy,
            seed=seed,
        )


def create_algorithm() -> DummyAlgorithm:
    algorithm = DummyAlgorithm()

    algorithm.initialize_context(
        fitness_function=lambda variables: sum(variables.values()),
        boundaries={"x": (-1.0, 1.0)},
    )

    return algorithm


def create_migration() -> DummyMigration:
    migration = DummyMigration()

    migration.initialize_context(
        communication_driver=None,  # type: ignore
        communication_processor_class=DummyCommunicationProcessor,
        communication_processor_kargs={},
    )

    return migration


def create_backend(
    processor: DummyProcessor | None = None,
) -> DummyDistributedBackend:
    backend = DummyDistributedBackend()

    backend.initialize_context(
        algorithm=create_algorithm(),
        n_iter=10,
        n_particles=3,
        migration=create_migration(),
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
        migration_driver=create_migration(),
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


def test_is_abstract() -> None:
    assert inspect.isabstract(DistributedBackendBase)
