import numpy as np
from doubles.cost_functions import sphere

from tyrannis.algorithm.pso import PSO
from tyrannis.backend.processor.process import ProcessPool

N_ITER = 20
N_PARTICLES = 10
N_PROCESS = 2
SEED = 42

BOUNDARIES = {
    "x": (-5.12, 5.12),
    "y": (-5.12, 5.12),
}


def create_processor(seed: int = SEED) -> ProcessPool:
    algorithm = PSO()

    algorithm.initialize_context(
        fitness_function=sphere,
        boundaries=BOUNDARIES,
    )

    processor = ProcessPool(
        n_process=N_PROCESS,
        chunksize=1,
    )

    processor.initialize_context(
        algorithm=algorithm,
        n_iter=N_ITER,
        n_particles=N_PARTICLES,
        seed=seed,
    )

    return processor


def run_processor(seed: int = SEED) -> ProcessPool:
    processor = create_processor(seed)

    processor.create_processors_pool(1)

    executor = next(iter(processor.processors_pool.values()))

    executor.initialize_execution_context()
    executor.run()
    executor.clear_execution_context()

    return executor


def test_run_executes_algorithm() -> None:
    processor = run_processor()

    algorithm = processor._algorithm

    assert len(algorithm.population) == N_PARTICLES
    assert processor._status.actual_iter == N_ITER

    for particle in algorithm.population.values():
        assert particle.fitness is not None
        assert np.isfinite(particle.fitness)


def test_run_updates_status() -> None:
    processor = run_processor()

    assert len(processor._status.population) == N_PARTICLES
    assert processor._status.best_particle_data != ""
    assert processor._status.best_particle_fitness is not None
    assert np.isfinite(processor._status.best_particle_fitness)


def test_run_updates_local_best() -> None:
    processor = run_processor()

    best_particle = processor._algorithm.local_best

    assert best_particle is not None
    assert best_particle.fitness is not None
    assert np.isfinite(best_particle.fitness)

    assert processor.local_best != ""


def test_run_uses_configured_number_of_processes() -> None:
    processor = run_processor()

    np.testing.assert_equal(
        processor._status.n_process,
        N_PROCESS,
    )


def test_processors_pool_creates_process_executor() -> None:
    processor = create_processor()

    processor.create_processors_pool(1)

    assert len(processor.processors_pool) == 1

    executor = next(iter(processor.processors_pool.values()))

    assert isinstance(executor, ProcessPool)
    assert executor.identifier == "island:0"
