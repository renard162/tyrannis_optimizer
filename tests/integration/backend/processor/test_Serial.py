from doubles.algorithm import DummyAlgorithm

from tyrannis.processor.serial import Serial
from tyrannis.core.signals import LocalEvent

N_ITER = 10
N_PARTICLES = 5
SEED = 42


def create_algorithm() -> DummyAlgorithm:
    algorithm = DummyAlgorithm()

    algorithm.initialize_context(
        fitness_function=lambda variables: sum(variables.values()),
        boundaries={"x": (-1.0, 1.0)},
    )

    return algorithm


def create_processor() -> Serial:
    processor = Serial()

    processor.initialize_context(
        algorithm=create_algorithm(),
        n_iter=N_ITER,
        n_particles=N_PARTICLES,
        migration_driver=None,  # type: ignore
        seed=SEED,
    )

    return processor


def run_processor() -> Serial:
    processor = create_processor()

    processor._migration_processor = DummyMigrationProcessor()  # type: ignore

    processor.initialize_execution_context()

    try:
        processor.run()
    finally:
        processor.finalize_execution_context()

    return processor


class DummyMigrationProcessor:
    def __init__(self) -> None:
        self.start_called = False
        self.stop_called = False
        self.migration_control_calls = []
        self.migration_signal = None

    def initialize_loop_context(
        self,
        migration_signal: LocalEvent,
    ) -> None:
        self.migration_signal = migration_signal

    def finalize_loop_context(self) -> None:
        self.migration_signal = None

    def start(self, stop_signal: LocalEvent) -> None:
        self.start_called = True
        self.stop_signal = stop_signal

    def stop(self) -> None:
        self.stop_called = True

    def migration_control(
        self,
        actual_iter: int,
        population: dict[str, float | None],
        local_best,
        insert_arrival_particle,
        departure_particle,
    ) -> None:
        self.migration_control_calls.append(
            {
                "actual_iter": actual_iter,
                "population": population,
                "local_best": local_best,
                "insert_arrival_particle": insert_arrival_particle,
                "departure_particle": departure_particle,
            }
        )


def test_run_executes_algorithm() -> None:
    processor = run_processor()

    algorithm = processor._algorithm

    assert len(algorithm.population) == N_PARTICLES

    for particle in algorithm.population.values():
        assert particle.fitness is not None


def test_run_updates_status() -> None:
    processor = run_processor()

    algorithm = processor._algorithm

    assert len(algorithm.population) == N_PARTICLES

    # O estado publicado pelo ProcessorBase agora é local_best.
    # Não existe mais processor._status.
    if algorithm.local_best is not None:
        assert processor.local_best == algorithm.local_best.dump()


def test_initialize_execution_context_starts_migration() -> None:
    processor = create_processor()

    migration_processor = DummyMigrationProcessor()
    processor._migration_processor = migration_processor  # type: ignore

    processor.initialize_execution_context()

    try:
        assert migration_processor.start_called
        assert migration_processor.stop_called is False
        assert processor._stop_signal is not None
        assert migration_processor.stop_signal is processor._stop_signal
    finally:
        processor.finalize_execution_context()


def test_finalize_execution_context_stops_migration() -> None:
    processor = create_processor()

    migration_processor = DummyMigrationProcessor()
    processor._migration_processor = migration_processor  # type: ignore

    processor.initialize_execution_context()
    processor.finalize_execution_context()

    assert migration_processor.start_called
    assert migration_processor.stop_called
