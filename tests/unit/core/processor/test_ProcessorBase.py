import inspect
import json

import pytest
from doubles.algorithm import DummyAlgorithm, DummyParticle
from doubles.processor import DummyProcessor

from tyrannis.backend.distributed.communication.no_communication import (
    NoCommunicationProcessor,
)
from tyrannis.core.backend_migration import (
    MigrationDriverBase,
    MigrationProcessorBase,
)
from tyrannis.core.processor import (
    LocalEvent,
    ProcessorBase,
    evaluate_particle,
)


class DummyMigrationProcessor(MigrationProcessorBase):
    def __init__(
        self,
        initial_iter: int,
        communication_processor: NoCommunicationProcessor,
        processor=None,
        *args,
        **kwargs,
    ) -> None:
        self._initial_iter = initial_iter
        self._communication_processor = communication_processor
        self._processor = processor
        self.start_called = False
        self.stop_called = False
        self.migration_control_called = False
        self.start_stop_signal = None
        self.migration_control_args = None

    def start(
        self,
        stop_signal: LocalEvent,
    ) -> None:
        self.start_called = True
        self.start_stop_signal = stop_signal

    def stop(self) -> None:
        self.stop_called = True

    def migration_control(
        self,
        actual_iter: int,
        population: dict[str, float | None],
        local_best: str | None,
        insert_arrival_particle,
        departure_particle,
    ) -> None:
        self.migration_control_called = True
        self.migration_control_args = {
            "actual_iter": actual_iter,
            "population": population,
            "local_best": local_best,
            "insert_arrival_particle": insert_arrival_particle,
            "departure_particle": departure_particle,
        }


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


def create_migration() -> DummyMigration:
    migration = DummyMigration()

    migration.initialize_context(
        communication_driver=None,  # type: ignore[arg-type]
        communication_processor_class=NoCommunicationProcessor,
        communication_processor_kargs={},
    )

    return migration


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
        migration_driver=create_migration(),
        seed=42,
    )

    return processor


def create_migration_processor(
    processor: DummyProcessor,
) -> DummyMigrationProcessor:
    migration = create_migration()

    migration._migration_processor_init_kargs["processor"] = processor

    return migration.create_processor_module(
        identification="MainProcessor",
    )  # type: ignore[return-value]


def test_initialize_context_sets_initial_state() -> None:
    processor = create_processor()

    assert processor._algorithm is not None
    assert processor._n_iter == 10
    assert processor._n_particles == 3
    assert processor._fitness_failure_strategy == "invalidate"
    assert processor.identifier == "MainProcessor"
    assert processor.processors_pool == {}
    assert processor.local_best is None
    assert processor._migration_processor is None


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
            migration_driver=create_migration(),
            fitness_failure_strategy="ignore",
        )


@pytest.mark.parametrize(
    "strategy",
    [
        "invalidate",
        "raise",
    ],
)
def test_initialize_context_accepts_valid_fitness_strategy(
    strategy: str,
) -> None:
    processor = create_processor()

    processor.initialize_context(
        algorithm=processor._algorithm,  # type: ignore
        n_iter=10,
        n_particles=3,
        migration_driver=create_migration(),
        fitness_failure_strategy=strategy,
    )

    assert processor._fitness_failure_strategy == strategy


def test_identifier_and_pool_properties() -> None:
    processor = create_processor()

    assert processor.identifier == "MainProcessor"
    assert processor.processors_pool == {}
    assert processor.local_best is None

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


def test_init_particles_does_not_replace_existing_population() -> None:
    processor = create_processor()

    assert processor._algorithm is not None

    particle = DummyParticle(
        identifier="existing",
        variables={"x": 0.5},
        fitness=0.25,
    )

    processor._algorithm.update_population([particle])

    processor.init_particles()

    assert list(processor._algorithm.population) == ["existing"]
    assert processor._algorithm.population["existing"] is particle


def test_initialize_execution_context_creates_stop_signal() -> None:
    processor = create_processor()

    processor.initialize_execution_context()

    assert isinstance(processor._stop_signal, LocalEvent)
    assert not processor._stop_signal.is_set()


def test_finalize_execution_context_replaces_stop_signal() -> None:
    processor = create_processor()

    processor.initialize_execution_context()

    processor._stop_signal.set()

    assert processor._stop_signal.is_set()

    processor.finalize_execution_context()

    assert isinstance(processor._stop_signal, LocalEvent)
    assert not processor._stop_signal.is_set()


def test_deepcopy_excludes_execution_and_pool_state() -> None:
    processor = create_processor()

    processor.initialize_execution_context()
    processor._stop_signal.set()

    processor._pool_count_sequence.spawn()

    replica = processor.__deepcopy__({})

    assert replica is not processor

    assert not hasattr(replica, "_stop_signal")

    assert replica._seed_sequence is None
    assert replica._pool_count_sequence is None
    assert replica._processors_pool == {}
    assert replica._migration_driver is None
    assert replica._migration_processor is None


def test_deepcopy_preserves_local_best() -> None:
    processor = create_processor()

    particle = DummyParticle(
        identifier="particle:0",
        variables={"x": 0.5},
        fitness=0.25,
    )

    assert processor._algorithm is not None

    processor._algorithm.update_population([particle])
    processor._algorithm._local_best = particle
    processor.update_status()

    replica = processor.__deepcopy__({})

    assert replica.local_best == processor.local_best


def test_deepcopy_preserves_processor_state() -> None:
    processor = create_processor()

    processor.set_identifier("processor:1")

    replica = processor.__deepcopy__({})

    assert replica.identifier == "processor:1"
    assert replica._n_iter == processor._n_iter
    assert replica._n_particles == processor._n_particles
    assert replica._fitness_failure_strategy == processor._fitness_failure_strategy


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
        assert replica._migration_processor is not None


def test_create_processors_pool_creates_independent_algorithms() -> None:
    processor = create_processor()

    processor.create_processors_pool(2)

    first = processor.processors_pool["island:0"]
    second = processor.processors_pool["island:1"]

    assert first._algorithm is not second._algorithm
    assert first._algorithm is not None
    assert first._algorithm.identifier == "island:0|algorithm"
    assert second._algorithm is not None
    assert second._algorithm.identifier == "island:1|algorithm"


def test_create_processors_pool_assigns_migration_processors() -> None:
    processor = create_processor()

    processor.create_processors_pool(2)

    for replica in processor.processors_pool.values():
        assert isinstance(
            replica._migration_processor,
            DummyMigrationProcessor,
        )


def test_update_processors_pool_adds_processors() -> None:
    processor = create_processor()

    first = create_processor()
    first.set_identifier("island:0")

    processor.update_processors_pool([first])

    assert processor.processors_pool == {
        "island:0": first,
    }


def test_update_processors_pool_replaces_processor_with_same_identifier() -> None:
    processor = create_processor()

    first = create_processor()
    first.set_identifier("island:0")

    second = create_processor()
    second.set_identifier("island:0")

    processor.update_processors_pool([first])
    processor.update_processors_pool([second])

    assert processor.processors_pool == {
        "island:0": second,
    }


def test_start_migration_requires_migration_processor() -> None:
    processor = create_processor()

    with pytest.raises(
        RuntimeError,
        match="Migration processor has not been initialized",
    ):
        processor.start_migration()


def test_start_migration() -> None:
    processor = create_processor()

    migration_processor = create_migration_processor(processor)

    processor._migration_processor = migration_processor
    processor.initialize_execution_context()

    processor.start_migration()

    assert migration_processor.start_called is True
    assert migration_processor.start_stop_signal is processor._stop_signal


def test_stop_migration_without_migration_processor() -> None:
    processor = create_processor()

    processor.stop_migration()


def test_stop_migration() -> None:
    processor = create_processor()

    migration_processor = create_migration_processor(processor)

    processor._migration_processor = migration_processor

    processor.stop_migration()

    assert migration_processor.stop_called is True


def test_migration_control_requires_migration_processor() -> None:
    processor = create_processor()

    with pytest.raises(
        RuntimeError,
        match="Migration processor has not been initialized",
    ):
        processor.migration_control(actual_iter=0)


def test_migration_control_delegates_to_migration_processor() -> None:
    processor = create_processor()

    migration_processor = create_migration_processor(processor)

    processor._migration_processor = migration_processor

    processor.migration_control(actual_iter=7)

    assert migration_processor.migration_control_called is True

    assert migration_processor.migration_control_args is not None
    assert migration_processor.migration_control_args["actual_iter"] == 7
    assert migration_processor.migration_control_args["population"] == {}
    assert migration_processor.migration_control_args["local_best"] is None
    assert (
        migration_processor.migration_control_args["insert_arrival_particle"]
        == processor._insert_arrival_particle
    )
    assert (
        migration_processor.migration_control_args["departure_particle"]
        == processor._departure_particle
    )


def test_migration_control_passes_local_best() -> None:
    processor = create_processor()

    particle = DummyParticle(
        identifier="particle:0",
        variables={"x": 1.0},
        fitness=1.0,
    )

    assert processor._algorithm is not None

    processor._algorithm.update_population([particle])
    processor._algorithm._local_best = particle

    processor.update_status()

    migration_processor = create_migration_processor(processor)
    processor._migration_processor = migration_processor

    processor.migration_control(actual_iter=5)

    assert migration_processor.migration_control_args is not None
    assert migration_processor.migration_control_args["local_best"] == particle.dump()
    assert migration_processor.migration_control_args["population"] == {
        "particle:0": 1.0,
    }


def test_insert_arrival_particle_creates_particle() -> None:
    processor = create_processor()

    particle = DummyParticle(
        identifier="arrival",
        variables={"x": 0.5},
        fitness=0.25,
    )

    processor._insert_arrival_particle(json.loads(particle.dump()))

    assert processor._algorithm is not None
    assert processor._algorithm.population["arrival"]() == particle()


def test_insert_arrival_particle_creates_particle_without_fitness() -> None:
    processor = create_processor()

    particle_data = {
        "identifier": "arrival",
        "variables": {"x": 0.5},
        "fitness": None,
    }

    processor._insert_arrival_particle(particle_data)

    assert processor._algorithm is not None
    assert "arrival" in processor._algorithm.population
    assert processor._algorithm.population["arrival"].fitness is None


def test_insert_arrival_particle_rejects_missing_identifier() -> None:
    processor = create_processor()

    with pytest.raises(
        ValueError,
        match="Arrival particle data must contain an identifier",
    ):
        processor._insert_arrival_particle(
            {
                "variables": {"x": 0.5},
                "fitness": 0.25,
            }
        )


def test_insert_arrival_particle_ignores_existing_particle() -> None:
    processor = create_processor()

    particle = DummyParticle(
        identifier="arrival",
        variables={"x": 0.5},
        fitness=0.25,
    )

    assert processor._algorithm is not None

    processor._algorithm.update_population([particle])

    replacement = DummyParticle(
        identifier="arrival",
        variables={"x": 0.9},
        fitness=0.81,
    )

    processor._insert_arrival_particle(json.loads(replacement.dump()))

    assert processor._algorithm.population["arrival"] is particle


def test_departure_particle_removes_particle() -> None:
    processor = create_processor()

    particle = DummyParticle(
        identifier="departure",
        variables={"x": 0.5},
        fitness=0.25,
    )

    assert processor._algorithm is not None

    processor._algorithm.update_population([particle])

    processor._departure_particle("departure")

    assert "departure" not in processor._algorithm.population


def test_departure_particle_ignores_missing_particle() -> None:
    processor = create_processor()

    processor._departure_particle("missing")


def test_update_status_updates_local_best() -> None:
    processor = create_processor()

    best = DummyParticle(
        identifier="particle:0",
        variables={"x": 1.0},
        fitness=1.0,
    )

    assert processor._algorithm is not None

    processor._algorithm.update_population([best])
    processor._algorithm._local_best = best

    processor.update_status()

    assert processor.local_best == best.dump()


def test_update_status_without_local_best_does_nothing() -> None:
    processor = create_processor()

    assert processor.local_best is None

    processor.update_status()

    assert processor.local_best is None


def test_update_status_preserves_existing_local_best() -> None:
    processor = create_processor()

    best = DummyParticle(
        identifier="particle:0",
        variables={"x": 1.0},
        fitness=1.0,
    )

    assert processor._algorithm is not None

    processor._algorithm.update_population([best])
    processor._algorithm._local_best = best

    processor.update_status()

    first_local_best = processor.local_best

    processor._algorithm._local_best = None

    processor.update_status()

    assert processor.local_best == first_local_best


def test_evaluate_particle_updates_particle() -> None:
    processor = create_processor()

    processor.initialize_execution_context()
    processor.init_particles()

    particle_id = "MainProcessor|particle:0"

    particle = evaluate_particle(
        particle_id=particle_id,
        algorithm=processor._algorithm,  # type: ignore
        stop_signal=processor._stop_signal,
        fitness_failure_strategy="invalidate",
    )

    assert particle.identifier == particle_id
    assert particle.candidate_fitness is not None
    assert particle.fitness is None


def test_evaluate_particle_initializes_particle() -> None:
    processor = create_processor()

    processor.initialize_execution_context()
    processor.init_particles()

    particle_id = "MainProcessor|particle:0"

    particle = evaluate_particle(
        particle_id=particle_id,
        algorithm=processor._algorithm,  # type: ignore
        stop_signal=processor._stop_signal,
        fitness_failure_strategy="invalidate",
        initialize_particle=True,
    )

    assert particle.identifier == particle_id
    assert particle.fitness is not None
    assert particle.new_particle is False


def test_evaluate_particle_returns_unmodified_particle_when_stopped() -> None:
    processor = create_processor()

    processor.initialize_execution_context()
    processor.init_particles()

    particle_id = "MainProcessor|particle:0"

    assert processor._algorithm is not None
    particle = processor._algorithm.population[particle_id]

    particle.candidate_variables = {"x": 99.0}
    particle.candidate_fitness = 99.0

    processor._stop_signal.set()

    result = evaluate_particle(
        particle_id=particle_id,
        algorithm=processor._algorithm,  # type: ignore
        stop_signal=processor._stop_signal,
        fitness_failure_strategy="invalidate",
    )

    assert result is particle
    assert result.candidate_variables == result.variables
    assert result.candidate_fitness == result.fitness


def test_evaluate_particle_invalidates_particle_when_fitness_fails() -> None:
    processor = create_processor()

    processor.initialize_execution_context()
    processor.init_particles()

    particle_id = "MainProcessor|particle:0"

    def failing_fitness(variables):
        raise RuntimeError("fitness failure")

    assert processor._algorithm is not None
    processor._algorithm._fitness_function = failing_fitness

    particle = evaluate_particle(
        particle_id=particle_id,
        algorithm=processor._algorithm,  # type: ignore
        stop_signal=processor._stop_signal,
        fitness_failure_strategy="invalidate",
    )

    assert particle.candidate_fitness == float("inf")


def test_evaluate_particle_raises_fitness_exception() -> None:
    processor = create_processor()

    processor.initialize_execution_context()
    processor.init_particles()

    particle_id = "MainProcessor|particle:0"

    def failing_fitness(variables):
        raise RuntimeError("fitness failure")

    assert processor._algorithm is not None
    processor._algorithm._fitness_function = failing_fitness

    with pytest.raises(RuntimeError, match="fitness failure"):
        evaluate_particle(
            particle_id=particle_id,
            algorithm=processor._algorithm,  # type: ignore
            stop_signal=processor._stop_signal,
            fitness_failure_strategy="raise",
        )


def test_evaluate_particle_does_not_initialize_when_stopped() -> None:
    processor = create_processor()

    processor.initialize_execution_context()
    processor.init_particles()

    particle_id = "MainProcessor|particle:0"

    processor._stop_signal.set()

    particle = evaluate_particle(
        particle_id=particle_id,
        algorithm=processor._algorithm,  # type: ignore
        stop_signal=processor._stop_signal,
        fitness_failure_strategy="invalidate",
        initialize_particle=True,
    )

    assert particle.identifier == particle_id


def test_is_abstract() -> None:
    assert inspect.isabstract(ProcessorBase)


def test_processor_base_requires_run_implementation() -> None:
    assert "run" in ProcessorBase.__abstractmethods__


def test_processor_base_requires_init_implementation() -> None:
    assert "__init__" in ProcessorBase.__abstractmethods__
