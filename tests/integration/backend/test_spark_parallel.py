"""Spark workers return processed particles to the Optimizer."""

from collections.abc import Iterator
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from _support.assertions import assert_optimizer_result_consistent
from _support.integration_probe import (
    IntegrationProbeAlgorithm,
    assert_parallel_probe_continues,
)
from _support.numerics import BASE_SEED
from _support.objectives import constant_objective, sphere
from scipy.optimize import minimize

from tyrannis import Optimizer
from tyrannis.backend import SparkParallel
from tyrannis.compatibility.scipy import ScipyMinimizer
from tyrannis.space import Continuous

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


@pytest.fixture(scope="module")
def spark_session() -> Iterator["SparkSession"]:
    for dependency in ("pyspark", "pandas", "pyarrow"):
        if find_spec(dependency) is None:
            pytest.skip(f"SparkParallel requires {dependency}")

    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder.master("local[2]")
        .appName("tyrannis-spark-parallel-test")
        # Local workers need the same import root for the serialized probe as pytest.
        .config(
            "spark.executorEnv.PYTHONPATH", str(Path(__file__).resolve().parents[2])
        )
    )
    try:
        spark = builder.getOrCreate()
    except Exception as exc:  # noqa: BLE001 - Any Spark startup error skips this test.
        pytest.skip(f"Spark runtime unavailable: {exc}")

    with spark:
        try:
            result = spark.sparkContext.parallelize([1], 1).map(lambda x: x).collect()
            assert result == [1]
        except Exception as exc:  # noqa: BLE001 - Any Spark startup error skips this test.
            pytest.skip(f"Spark Python workers unavailable: {exc}")

        yield spark


@pytest.mark.optional
@pytest.mark.multiprocess
@pytest.mark.timeout(240, method="thread")
def test_optimizer_consolidates_particles_from_spark_workers(
    spark_session: "SparkSession",
) -> None:
    from pyspark import TaskContext

    def worker_fitness(x: float) -> float:
        assert TaskContext.get() is not None
        # Updates away from the initial midpoint improve the objective.
        return -sphere(x)

    probe = IntegrationProbeAlgorithm()
    optimizer = Optimizer(
        space=Continuous(boundaries={"x": (-1.0, 1.0)}, cost_function=worker_fitness),
        algorithm=probe,
        backend=SparkParallel(spark_session),
        n_particles=2,
        n_iterations=1,
        seed=BASE_SEED,
        history="iteration",
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
    assert_parallel_probe_continues(optimizer, probe)


@pytest.mark.optional
@pytest.mark.multiprocess
@pytest.mark.timeout(240, method="thread")
def test_scipy_observes_force_stop_on_the_parallel_driver(
    spark_session: "SparkSession",
) -> None:
    method = ScipyMinimizer(
        IntegrationProbeAlgorithm(),
        2,
        1,
        seed=BASE_SEED,
        backend=SparkParallel(spark_session),
        history="iteration",
    )
    with pytest.warns(UserWarning, match="probe requested early stop"):
        stopped = minimize(
            constant_objective,
            [0],
            method=method,
            bounds=[(0, 1)],
            options={
                "algorithm": IntegrationProbeAlgorithm(force_stop_at_iteration=1),
                "maxiter": 4,
            },
        )
    assert (stopped.status, stopped.success) == (2, False)
    assert "nit" not in stopped
    step = method.history.columns.index("iteration")
    assert [row[step] for row in method.history.generator] == [0, 1]
    complete = minimize(constant_objective, [0], method=method, bounds=[(0, 1)])
    assert (complete.status, complete.success, complete.nit) == (0, True, 2)
