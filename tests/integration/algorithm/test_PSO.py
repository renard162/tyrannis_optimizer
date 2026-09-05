import numpy as np
from doubles.cost_functions import sphere

from tyrannis.algorithm.pso import PSO
from tyrannis.backend.processor.serial import Serial

SEED = 42
N_ITER = 20
N_PARTICLES = 10

BOUNDARIES = {
    "x": (-5.12, 5.12),
    "y": (-5.12, 5.12),
}

EXPECTED_BEST_FITNESS = 0.0004345841025887674

EXPECTED_BEST_VARIABLES = {
    "x": -0.01698341924029871,
    "y": -0.012089151066018614,
}


def run_pso(seed: int = SEED) -> Serial:
    algorithm = PSO()

    algorithm.initialize_context(
        fitness_function=sphere,
        boundaries=BOUNDARIES,
    )

    processor = Serial()

    processor.initialize_context(
        algorithm=algorithm,
        n_iter=N_ITER,
        n_particles=N_PARTICLES,
        seed=seed,
    )

    processor.create_processors_pool(1)

    executor = next(iter(processor.processors_pool.values()))

    executor.initialize_execution_context()
    executor.run()
    executor.finalize_execution_context()

    return executor


def test_pso_converges_on_sphere() -> None:
    processor = run_pso()

    assert processor.local_best != ""

    best_particle = processor._algorithm.local_best

    assert best_particle is not None
    assert best_particle.fitness is not None

    np.testing.assert_allclose(
        best_particle.fitness,
        EXPECTED_BEST_FITNESS,
    )

    np.testing.assert_allclose(
        best_particle.variables["x"],
        EXPECTED_BEST_VARIABLES["x"],
    )

    np.testing.assert_allclose(
        best_particle.variables["y"],
        EXPECTED_BEST_VARIABLES["y"],
    )


def test_pso_is_reproducible() -> None:
    first = run_pso()
    second = run_pso()

    first_best = first._algorithm.local_best
    second_best = second._algorithm.local_best

    assert first_best is not None
    assert second_best is not None

    assert first_best.fitness is not None
    assert second_best.fitness is not None

    np.testing.assert_allclose(
        first_best.fitness,
        second_best.fitness,
    )

    np.testing.assert_allclose(
        first_best.variables["x"],
        second_best.variables["x"],
    )

    np.testing.assert_allclose(
        first_best.variables["y"],
        second_best.variables["y"],
    )
