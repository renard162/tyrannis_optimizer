from __future__ import annotations

import math

import pytest

from tyrannis.algorithm.pso import PSO
from tyrannis.backend.distributed.spark_distributed import SparkDistributed
from tyrannis.backend.local import Local
from tyrannis.backend.migration.island_isolation import IslandIsolation
from tyrannis.backend.parallel.spark_parallel import SparkParallel
from tyrannis.backend.processor.process import ProcessPool
from tyrannis.backend.processor.serial import Serial
from tyrannis.space.continuous import Continuous

SEED = 42
N_ITER = 2
N_PARTICLES = 3


def list_sphere(*values: float) -> float:
    """Sphere cost function using positional arguments."""
    return float(sum(value**2 for value in values))


def create_algorithm(space: Continuous) -> PSO:
    """Create and initialize a PSO using the search space."""
    space.initialize_context(SEED)

    algorithm = PSO()

    algorithm.initialize_context(
        fitness_function=space,
        boundaries=space.encoded_boundaries,
    )

    return algorithm


def assert_result(backend) -> None:
    """Validate the result exposed by the backend."""
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
    assert math.isfinite(fitness)


def test_list_boundaries_with_local_serial() -> None:
    """A list-based space wraps a positional cost function for Tyrannis."""
    space = Continuous(
        cost_function=list_sphere,
        boundaries=[
            (0.0, 1.0),
            (10.0, 20.0),
        ],
    )

    algorithm = create_algorithm(space)

    assert space.is_kargs is False

    assert space.encoded_boundaries == {
        "0": (0.0, 1.0),
        "1": (10.0, 20.0),
    }

    migration = IslandIsolation()

    processor = Serial()

    processor.initialize_context(
        algorithm=algorithm,
        n_iter=N_ITER,
        n_particles=N_PARTICLES,
        migration_driver=migration,
        fitness_failure_strategy="raise",
        seed=SEED,
    )

    backend = Local()

    backend.initialize_context(
        algorithm=algorithm,
        n_iter=N_ITER,
        n_particles=N_PARTICLES,
        migration=migration,
        processor=processor,
        fitness_failure_strategy="raise",
        seed=SEED,
    )

    backend.execute()

    assert_result(backend)

    result = backend.result

    assert result is not None

    variables = result["variables"]

    assert set(variables) == {"0", "1"}  # type: ignore

    assert 0.0 <= variables["0"] <= 1.0  # type: ignore
    assert 10.0 <= variables["1"] <= 20.0  # type: ignore


def test_dict_boundaries_with_local_serial() -> None:
    """A dictionary-based space preserves the named cost-function inputs."""

    def dict_sphere(**values: float) -> float:
        return float(sum(value**2 for value in values.values()))

    space = Continuous(
        cost_function=dict_sphere,
        boundaries={
            "x": (-5.0, 5.0),
            "y": (-5.0, 5.0),
        },
    )

    algorithm = create_algorithm(space)

    assert space.is_kargs is True

    assert space.encoded_boundaries == {
        "x": (-5.0, 5.0),
        "y": (-5.0, 5.0),
    }

    migration = IslandIsolation()

    processor = Serial()

    processor.initialize_context(
        algorithm=algorithm,
        n_iter=N_ITER,
        n_particles=N_PARTICLES,
        migration_driver=migration,
        fitness_failure_strategy="raise",
        seed=SEED,
    )

    backend = Local()

    backend.initialize_context(
        algorithm=algorithm,
        n_iter=N_ITER,
        n_particles=N_PARTICLES,
        migration=migration,
        processor=processor,
        fitness_failure_strategy="raise",
        seed=SEED,
    )

    backend.execute()

    assert_result(backend)

    result = backend.result

    assert result is not None

    variables = result["variables"]

    assert set(variables) == {"x", "y"}  # type: ignore

    assert -5.0 <= variables["x"] <= 5.0  # type: ignore
    assert -5.0 <= variables["y"] <= 5.0  # type: ignore


def test_continuous_with_process_pool() -> None:
    """The complete Space wrapper can be serialized by ProcessPool."""

    def dict_sphere(**values: float) -> float:
        return float(sum(value**2 for value in values.values()))

    space = Continuous(
        cost_function=dict_sphere,
        boundaries={
            "x": (-5.0, 5.0),
            "y": (-5.0, 5.0),
        },
    )

    algorithm = create_algorithm(space)

    migration = IslandIsolation()

    processor = ProcessPool(
        n_process=2,
        chunksize=1,
    )

    processor.initialize_context(
        algorithm=algorithm,
        n_iter=N_ITER,
        n_particles=N_PARTICLES,
        migration_driver=migration,
        fitness_failure_strategy="raise",
        seed=SEED,
    )

    backend = Local()

    backend.initialize_context(
        algorithm=algorithm,
        n_iter=N_ITER,
        n_particles=N_PARTICLES,
        migration=migration,
        processor=processor,
        fitness_failure_strategy="raise",
        seed=SEED,
    )

    backend.execute()

    assert_result(backend)


@pytest.mark.usefixtures("spark")
def test_continuous_with_spark_parallel(spark) -> None:
    """The complete Space wrapper can be serialized by SparkParallel."""
    from tyrannis.examples.bowl_shaped import sphere

    space = Continuous(
        cost_function=sphere,
        boundaries={
            "x": (-5.0, 5.0),
            "y": (-5.0, 5.0),
        },
    )

    algorithm = create_algorithm(space)

    backend = SparkParallel(spark)

    backend.initialize_context(
        algorithm=algorithm,
        n_iter=N_ITER,
        n_particles=N_PARTICLES,
        migration=IslandIsolation(),
        fitness_failure_strategy="raise",
        seed=SEED,
    )

    backend.execute()

    assert_result(backend)


@pytest.mark.usefixtures("spark")
def test_continuous_with_spark_distributed_and_island_isolation(
    spark,
) -> None:
    """The Space wrapper works with SparkDistributed and IslandIsolation."""
    from tyrannis.examples.bowl_shaped import sphere

    space = Continuous(
        cost_function=sphere,
        boundaries={
            "x": (-5.0, 5.0),
            "y": (-5.0, 5.0),
        },
    )

    algorithm = create_algorithm(space)

    migration = IslandIsolation()

    processor = Serial()

    processor.initialize_context(
        algorithm=algorithm,
        n_iter=N_ITER,
        n_particles=N_PARTICLES,
        migration_driver=migration,
        fitness_failure_strategy="raise",
        seed=SEED,
    )

    backend = SparkDistributed(
        spark,
        n_executors=1,
    )

    backend.initialize_context(
        algorithm=algorithm,
        n_iter=N_ITER,
        n_particles=N_PARTICLES,
        migration=migration,
        processor=processor,
        fitness_failure_strategy="raise",
        seed=SEED,
    )

    backend.execute()

    assert_result(backend)
