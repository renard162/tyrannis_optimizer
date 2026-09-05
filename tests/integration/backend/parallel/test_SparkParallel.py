import numpy as np
from doubles.cost_functions import sphere
from pyspark.sql import SparkSession

from tyrannis.algorithm.pso import PSO
from tyrannis.backend.parallel.spark_parallel import SparkParallel

N_ITER = 2
N_PARTICLES = 20
SEED = 42

BOUNDARIES = {
    "x": (-5.12, 5.12),
    "y": (-5.12, 5.12),
}


def create_backend(
    spark: SparkSession,
    seed: int = SEED,
) -> SparkParallel:
    algorithm = PSO()

    algorithm.initialize_context(
        fitness_function=sphere,
        boundaries=BOUNDARIES,
    )

    backend = SparkParallel(spark)

    backend.initialize_context(
        algorithm=algorithm,
        n_iter=N_ITER,
        n_particles=N_PARTICLES,
        seed=seed,
    )

    return backend


def test_execute_runs_pso_on_spark(
    spark: SparkSession,
) -> None:
    backend = create_backend(spark)

    backend.execute()

    assert backend.actual_iter == N_ITER

    assert len(backend._algorithm.population) == N_PARTICLES

    for particle in backend._algorithm.population.values():
        assert particle.fitness is not None
        assert np.isfinite(particle.fitness)

    assert backend._algorithm.local_best is not None


def test_execute_updates_result(
    spark: SparkSession,
) -> None:
    backend = create_backend(spark)

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


def test_execute_updates_local_best(
    spark: SparkSession,
) -> None:
    backend = create_backend(spark)

    backend.execute()

    local_best = backend.local_best

    assert local_best != ""

    best_particle = backend._algorithm.local_best
    result = backend.result

    assert best_particle is not None
    assert best_particle.fitness is not None
    assert result is not None

    fitness = result["fitness"]

    assert isinstance(fitness, float)

    np.testing.assert_allclose(
        best_particle.fitness,
        fitness,
    )


def test_execute_is_reproducible(
    spark: SparkSession,
) -> None:
    first = create_backend(spark, seed=SEED)
    second = create_backend(spark, seed=SEED)

    first.execute()
    second.execute()

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
