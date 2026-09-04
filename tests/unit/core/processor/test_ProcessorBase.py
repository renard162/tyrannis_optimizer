import inspect

import pytest
from doubles.algorithm import DummyAlgorithm, DummyParticle
from doubles.processor import DummyProcessor

from tyrannis.core.processor import (
    ControlVariables,
    LocalEvent,
    StatusVariables,
)


def create_processor() -> DummyProcessor:
    algorithm = DummyAlgorithm()
    algorithm.initialize_context(
        fitness_function=lambda variables: sum(variables.values()),
        boundaries={"x": (-1.0, 1.0)},
    )

    processor = DummyProcessor(algorithm)
    processor.initialize_context(
        algorithm=algorithm,
        n_iter=10,
        n_particles=3,
        seed=42,
    )

    return processor


def test_initialize_context_sets_initial_state() -> None:
    processor = create_processor()

    assert processor._algorithm is not None
    assert processor._n_iter == 10
    assert processor._n_particles == 3
    assert processor._fitness_failure_strategy == "invalidate"
    assert processor.identifier == "MainProcessor"
    assert processor.processors_pool == {}
    assert processor.local_best == ""
    assert isinstance(processor._control, ControlVariables)
    assert isinstance(processor._status, StatusVariables)


def test_initialize_context_rejects_invalid_fitness_strategy() -> None:
    algorithm = DummyAlgorithm()
    algorithm.initialize_context(
        fitness_function=lambda variables: sum(variables.values()),
        boundaries={"x": (-1.0, 1.0)},
    )

    processor = DummyProcessor(algorithm)

    with pytest.raises(
        ValueError,
        match="Invalid fitness failure strategy",
    ):
        processor.initialize_context(
            algorithm=algorithm,
            n_iter=10,
            n_particles=3,
            fitness_failure_strategy="ignore",
        )


def test_identifier_and_pool_properties() -> None:
    processor = create_processor()

    assert processor.identifier == "MainProcessor"
    assert processor.processors_pool == {}
    assert processor.local_best == ""

    processor.set_identifier("processor:1")

    assert processor.identifier == "processor:1"


def test_init_particles_creates_population_once() -> None:
    processor = create_processor()

    processor.init_particles()

    assert processor._algorithm is not None
    assert list(processor._algorithm.population) == [
        "MainProcessor|particle:0",
        "MainProcessor|particle:1",
        "MainProcessor|particle:2",
    ]

    population = processor._algorithm.population

    processor.init_particles()

    assert processor._algorithm.population is population


def test_initialize_and_finalize_execution_context() -> None:
    processor = create_processor()

    processor.initialize_execution_context()

    assert isinstance(processor._stop_signal, LocalEvent)
    assert isinstance(processor._wait_signal, LocalEvent)

    processor._stop_signal.set()
    processor._wait_signal.set()

    processor.finalize_execution_context()

    assert isinstance(processor._stop_signal, LocalEvent)
    assert isinstance(processor._wait_signal, LocalEvent)
    assert not processor._stop_signal.is_set()
    assert not processor._wait_signal.is_set()


def test_deepcopy_excludes_execution_and_pool_state() -> None:
    processor = create_processor()
    processor.initialize_execution_context()

    processor._stop_signal.set()
    processor._pool_count_sequence.spawn()
    processor._status.actual_iter = 5

    replica = processor.__deepcopy__({})

    assert replica is not processor
    assert replica._status is not processor._status
    assert replica._status.actual_iter == 5

    assert not hasattr(replica, "_stop_signal")
    assert not hasattr(replica, "_wait_signal")
    assert replica._seed_sequence is None
    assert replica._pool_count_sequence is None
    assert replica._processors_pool == {}


def test_create_processors_pool_creates_islands() -> None:
    processor = create_processor()

    processor.create_processors_pool(3)

    assert list(processor.processors_pool) == [
        "island:0",
        "island:1",
        "island:2",
    ]

    for identifier, replica in processor.processors_pool.items():
        assert replica.identifier == identifier
        assert replica is not processor
        assert replica._algorithm is not None
        assert replica._algorithm.identifier == f"{identifier}|algorithm"


def test_update_processors_pool_adds_and_replaces_processors() -> None:
    processor = create_processor()

    first = create_processor()
    first.set_identifier("island:0")

    replacement = create_processor()
    replacement.set_identifier("island:0")

    second = create_processor()
    second.set_identifier("island:1")

    processor.update_processors_pool([first])
    processor.update_processors_pool([replacement, second])

    assert processor.processors_pool == {
        "island:0": replacement,
        "island:1": second,
    }


def test_wait_sync_returns_without_waiting_when_no_limit() -> None:
    processor = create_processor()
    processor.initialize_execution_context()

    processor.wait_sync(actual_iter=10)

    assert processor._status.iter_waiting is False


def test_update_iter_counter_updates_status() -> None:
    processor = create_processor()

    processor.update_iter_counter(7)

    assert processor._status.actual_iter == 7


def test_update_status_updates_population_and_partial_result() -> None:
    processor = create_processor()

    best = DummyParticle(
        identifier="particle:0",
        variables={"x": 1.0},
        fitness=1.0,
    )
    worst = DummyParticle(
        identifier="particle:1",
        variables={"x": 2.0},
        fitness=4.0,
    )

    assert processor._algorithm is not None
    processor._algorithm.update_population([best, worst])
    processor._algorithm._iter_best = best.identifier
    processor._algorithm._iter_worst = worst.identifier

    processor.update_status()

    assert processor._status.population == {
        "particle:0": 1.0,
        "particle:1": 4.0,
    }
    assert processor._status.best_particle_fitness == 1.0
    assert processor._status.best_particle_data == best.dump()
    assert processor._status.worst_particle_fitness == 4.0
    assert processor._status.worst_particle_data == worst.dump()
    assert processor.local_best == best.dump()


def test_update_status_without_iteration_best_only_updates_population() -> None:
    processor = create_processor()

    particle = DummyParticle(
        identifier="particle:0",
        variables={"x": 1.0},
        fitness=1.0,
    )

    assert processor._algorithm is not None
    processor._algorithm.update_population([particle])

    processor.update_status()

    assert processor._status.population == {"particle:0": 1.0}
    assert processor._status.best_particle_data == ""
    assert processor._status.best_particle_fitness is None


def test_migration_control_inserts_arrival_particle() -> None:
    processor = create_processor()

    particle = DummyParticle(
        identifier="arrival",
        variables={"x": 2.0},
        fitness=4.0,
    )

    processor._control.arrival_particle_id = particle.identifier
    processor._control.arrival_particle_data = particle.dump()

    processor.migration_control()

    assert processor._algorithm is not None
    assert processor._algorithm.population[particle.identifier]() == particle()


def test_migration_control_rejects_missing_arrival_data() -> None:
    processor = create_processor()

    processor._control.arrival_particle_id = "arrival"

    with pytest.raises(
        ValueError,
        match="Arrival particle data cannot be None",
    ):
        processor.migration_control()


def test_migration_control_ignores_existing_arrival_particle() -> None:
    processor = create_processor()

    particle = DummyParticle(
        identifier="arrival",
        variables={"x": 1.0},
        fitness=1.0,
    )

    assert processor._algorithm is not None
    processor._algorithm.update_population([particle])

    processor._control.arrival_particle_id = particle.identifier
    processor._control.arrival_particle_data = DummyParticle(
        identifier="arrival",
        variables={"x": 9.0},
        fitness=81.0,
    ).dump()

    processor.migration_control()

    assert processor._algorithm.population["arrival"] is particle


def test_is_abstract() -> None:
    from tyrannis.core.processor import ProcessorBase

    assert inspect.isabstract(ProcessorBase)
