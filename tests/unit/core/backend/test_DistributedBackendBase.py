import inspect
from collections.abc import Callable
from queue import Queue
from typing import Any

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
        self._messages: Queue[str] = Queue()
        self._outgoing_queue: Queue[str] = Queue()
        self._started = False
        self._stopped = False
        self._stop_signal: LocalEvent | None = None
        self._message_signal: LocalEvent | None = None

    @property
    def messages(self) -> Queue[str]:
        return self._messages

    @property
    def outgoing_queue(self) -> Queue[str]:
        return self._outgoing_queue

    def set_message_signal(  # type: ignore
        self,
        message_signal: LocalEvent,
    ) -> None:
        self._message_signal = message_signal

    def start(
        self,
        stop_signal: LocalEvent,
    ) -> None:
        self._stop_signal = stop_signal
        self._started = True

    def stop(self) -> None:
        self._stopped = True


class DummyMigrationProcessor(MigrationProcessorBase):
    def __init__(
        self,
        initial_iter: int,
        communication_processor: CommunicationProcessorBase,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._initial_iter = initial_iter
        self._communication_processor = communication_processor
        self._started = False
        self._stopped = False
        self._stop_signal: LocalEvent | None = None
        self._migration_control_calls: list[
            tuple[
                int,
                dict[str, float | None],
                str | None,
                Callable[[dict[str, Any]], None],
                Callable[[str], None],
            ]
        ] = []

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
        self._stop_signal = stop_signal
        self._started = True

    def stop(self) -> None:
        self._stopped = True

    def migration_control(  # type: ignore
        self,
        actual_iter: int,
        population: dict[str, float | None],
        local_best: str | None,
        insert_arrival_particle: Callable[[dict[str, Any]], None],
        departure_particle: Callable[[str], None],
    ) -> None:
        self._migration_control_calls.append(
            (
                actual_iter,
                population,
                local_best,
                insert_arrival_particle,
                departure_particle,
            )
        )


class DummyMigration(MigrationDriverBase):
    _processor_class = DummyMigrationProcessor

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


class DummyDistributedBackend(DistributedBackendBase, DummyBackend):
    _n_executors = 3

    def initialize_context(  # type: ignore
        self,
        algorithm: DummyAlgorithm,
        n_iter: int,
        n_particles: int,
        migration: DummyMigration,
        processor: DummyProcessor | None = None,
        fitness_failure_strategy: str = "invalidate",
        seed: int | None = None,
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
    algorithm = DummyAlgorithm()  # type: ignore

    algorithm.initialize_context(
        fitness_function=lambda variables: sum(variables.values()),
        boundaries={"x": (-1.0, 1.0)},
    )

    return algorithm


def create_migration() -> DummyMigration:
    migration = DummyMigration()

    migration.initialize_context(
        communication_driver=None,  # type: ignore[arg-type]
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


def create_initialized_processor() -> DummyProcessor:
    processor = DummyProcessor()

    processor.initialize_context(
        algorithm=create_algorithm(),
        n_iter=10,
        n_particles=3,
        migration_driver=create_migration(),
        seed=42,
    )

    return processor


def test_init_processors_requires_processor() -> None:
    backend = create_backend()

    with pytest.raises(
        RuntimeError,
        match="Processor is not initialized.",
    ):
        backend.init_processors()


def test_init_processors_creates_pool() -> None:
    processor = create_initialized_processor()
    backend = create_backend(processor)

    backend.init_processors()

    assert list(processor.processors_pool) == [
        "island:0",
        "island:1",
        "island:2",
    ]


def test_init_processors_uses_configured_number_of_executors() -> None:
    processor = create_initialized_processor()
    backend = create_backend(processor)

    backend._n_executors = 5
    backend.init_processors()

    assert list(processor.processors_pool) == [
        "island:0",
        "island:1",
        "island:2",
        "island:3",
        "island:4",
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


def test_update_result_ignores_none_candidates() -> None:
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


def test_update_result_preserves_only_result_keys() -> None:
    backend = create_backend()

    backend._local_bests = {
        "island:0": {
            "identifier": "island:0",
            "fitness": 2.0,
            "variables": {"x": 0.2},
            "extra": "ignored",
        },  # type: ignore
    }

    backend.update_result()

    assert backend.result == {
        "identifier": "island:0",
        "fitness": 2.0,
        "variables": {"x": 0.2},
    }


def test_update_result_does_not_change_result_when_no_valid_candidate() -> None:
    backend = create_backend()

    backend._result = {
        "fitness": 10.0,
        "variables": {"x": 1.0},
    }

    backend._local_bests = {
        "island:0": None,
        "island:1": {
            "fitness": 1,
            "variables": {"x": 0.1},
        },
    }

    backend.update_result()

    assert backend.result == {
        "fitness": 10.0,
        "variables": {"x": 1.0},
    }


def test_update_result_does_not_change_result_for_empty_candidates() -> None:
    backend = create_backend()

    backend._result = {
        "fitness": 10.0,
        "variables": {"x": 1.0},
    }

    backend._local_bests = {}

    backend.update_result()

    assert backend.result == {
        "fitness": 10.0,
        "variables": {"x": 1.0},
    }


def test_update_result_selects_global_minimum_fitness() -> None:
    backend = create_backend()

    backend._local_bests = {
        "island:0": {
            "fitness": 10.0,
            "variables": {"x": 0.1},
        },
        "island:1": {
            "fitness": -5.0,
            "variables": {"x": -0.5},
        },
        "island:2": {
            "fitness": 0.0,
            "variables": {"x": 0.0},
        },
    }

    backend.update_result()

    assert backend.result == {
        "fitness": -5.0,
        "variables": {"x": -0.5},
    }


def test_initialize_context_configures_algorithm() -> None:
    backend = DummyDistributedBackend()
    algorithm = create_algorithm()
    migration = create_migration()

    backend.initialize_context(
        algorithm=algorithm,
        n_iter=20,
        n_particles=7,
        migration=migration,
        fitness_failure_strategy="raise",
        seed=123,
    )

    assert backend._algorithm is algorithm
    assert backend._n_iter == 20
    assert backend._n_particles == 7
    assert backend._migration is migration
    assert backend._fitness_failure_strategy == "raise"
    assert backend._seed == 123
    assert backend._processor is None


def test_identifier_property() -> None:
    backend = create_backend()

    assert backend.identifier == "DummyBackend"


def test_result_is_none_before_update() -> None:
    backend = create_backend()

    assert backend.result is None


def test_is_abstract() -> None:
    assert inspect.isabstract(DistributedBackendBase)


def test_distributed_backend_requires_initialize_context_implementation() -> None:
    assert "initialize_context" in DistributedBackendBase.__abstractmethods__
