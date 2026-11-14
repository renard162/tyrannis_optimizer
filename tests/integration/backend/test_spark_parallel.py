"""Spark workers return processed particles to the Optimizer."""

from importlib.util import find_spec
from pathlib import Path

import pytest

from tests._support.assertions import assert_optimizer_result_consistent
from tests._support.integration_probe import IntegrationProbeAlgorithm
from tests._support.numerics import BASE_SEED
from tests._support.objectives import sphere
from tyrannis import Optimizer
from tyrannis.backend import SparkParallel
from tyrannis.space import Continuous


@pytest.mark.optional
@pytest.mark.multiprocess
def test_optimizer_consolidates_particles_from_spark_workers() -> None:
    for dependency in ("pyspark", "pandas", "pyarrow"):
        if find_spec(dependency) is None:
            pytest.skip(f"SparkParallel requires {dependency}")

    from pyspark import TaskContext
    from pyspark.sql import SparkSession

    def worker_fitness(x: float) -> float:
        assert TaskContext.get() is not None
        # Updates away from the initial midpoint improve the objective.
        return -sphere(x)

    probe = IntegrationProbeAlgorithm()
    builder = (
        SparkSession.builder.master("local[2]")
        .appName("tyrannis-spark-parallel-test")
        # Local workers need the same import root for the serialized probe as pytest.
        .config(
            "spark.executorEnv.PYTHONPATH", str(Path(__file__).resolve().parents[3])
        )
    )
    try:
        spark = builder.getOrCreate()
    except Exception as exc:  # noqa: BLE001 - Any Spark startup error skips this test.
        pytest.skip(f"Spark runtime unavailable: {exc}")

    with spark:
        optimizer = Optimizer(
            space=Continuous(
                boundaries={"x": (-1.0, 1.0)}, cost_function=worker_fitness
            ),
            algorithm=probe,
            backend=SparkParallel(spark),
            n_particles=2,
            n_iterations=1,
            seed=BASE_SEED,
        ).fit()

    assert len(probe.population) == 2
    for identifier, particle in probe.population.items():
        assert [(event["particle_id"], event["phase"]) for event in particle.trace] == [
            (identifier, "initialize"),
            (identifier, "update"),
        ]
        assert particle.fitness == -sphere(**particle.variables) < 0.0

    best = min(probe.population.values(), key=lambda particle: particle.fitness)
    assert optimizer.result_ == {
        "identifier": best.identifier,
        "variables": best.variables,
        "fitness": best.fitness,
    }
    assert_optimizer_result_consistent(optimizer)
