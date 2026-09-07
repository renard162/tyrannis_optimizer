import numpy as np
from doubles.cost_functions import sphere

from tyrannis.algorithm.pso import PSO
from tyrannis.backend.local import Local
from tyrannis.migration.island_isolation import IslandIsolation
from tyrannis.processor.serial import Serial

N_ITER = 20
N_PARTICLES = 10
SEED = 42

BOUNDARIES = {
    "x": (-5.12, 5.12),
    "y": (-5.12, 5.12),
}


def create_backend(seed: int = SEED) -> Local:
    algorithm = PSO()

    algorithm.initialize_context(
        fitness_function=sphere,
        boundaries=BOUNDARIES,
    )

    migration = IslandIsolation()

    processor = Serial()

    processor.initialize_context(
        algorithm=algorithm,
        n_iter=N_ITER,
        n_particles=N_PARTICLES,
        migration_driver=migration,
        seed=seed,
    )

    backend = Local()

    backend.initialize_context(
        algorithm=algorithm,
        n_iter=N_ITER,
        n_particles=N_PARTICLES,
        migration=migration,
        processor=processor,
        seed=seed,
    )

    return backend


def test_execute_runs_pso() -> None:
    backend = create_backend()

    backend.execute()

    result = backend.result

    assert result is not None

    assert isinstance(result["identifier"], str)
    assert isinstance(result["variables"], dict)

    fitness = result["fitness"]

    assert isinstance(fitness, float)
    assert np.isfinite(fitness)


def test_execute_updates_result() -> None:
    backend = create_backend()

    backend.execute()

    result = backend.result

    assert result is not None
    assert set(result) == {
        "identifier",
        "fitness",
        "variables",
    }

    assert isinstance(result["identifier"], str)
    assert isinstance(result["variables"], dict)

    fitness = result["fitness"]

    assert isinstance(fitness, float)
    assert np.isfinite(fitness)


def test_execute_finds_valid_solution() -> None:
    backend = create_backend()

    backend.execute()

    result = backend.result

    assert result is not None

    fitness = result["fitness"]

    assert isinstance(fitness, float)
    assert np.isfinite(fitness)
    assert fitness >= 0.0


def test_execute_is_reproducible() -> None:
    first = create_backend(seed=SEED)
    second = create_backend(seed=SEED)

    first.execute()
    second.execute()

    first_result = first.result
    second_result = second.result

    assert first_result is not None
    assert second_result is not None

    first_fitness = first_result["fitness"]
    second_fitness = second_result["fitness"]

    assert isinstance(first_fitness, float)
    assert isinstance(second_fitness, float)

    np.testing.assert_allclose(
        first_fitness,
        second_fitness,
    )

    first_variables = first_result["variables"]
    second_variables = second_result["variables"]

    assert isinstance(first_variables, dict)
    assert isinstance(second_variables, dict)

    np.testing.assert_allclose(
        first_variables["x"],
        second_variables["x"],
    )

    np.testing.assert_allclose(
        first_variables["y"],
        second_variables["y"],
    )
