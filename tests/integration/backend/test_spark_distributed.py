"""The Optimizer selects the best particle returned by isolated Spark islands."""

from importlib.util import find_spec
from pathlib import Path

import pytest

from tests._support.assertions import assert_optimizer_result_consistent
from tests._support.integration_probe import IntegrationProbeAlgorithm
from tests._support.numerics import BASE_SEED
from tests._support.objectives import sphere
from tyrannis import Optimizer
from tyrannis.backend import SparkDistributed
from tyrannis.migration.island_isolation import IslandIsolation
from tyrannis.space import Continuous


@pytest.mark.optional
@pytest.mark.multiprocess
def test_optimizer_consolidates_results_from_spark_islands() -> None:
    for dependency in ("pyspark", "pandas", "pyarrow"):
        if find_spec(dependency) is None:
            pytest.skip(f"SparkDistributed requires {dependency}")

    from pyspark import TaskContext
    from pyspark.sql import SparkSession

    def worker_fitness(x: float) -> float:
        assert TaskContext.get() is not None
        # Either island's update improves on the probe's initial midpoint.
        return -sphere(x)

    builder = (
        SparkSession.builder.master("local[2]")
        .appName("tyrannis-spark-distributed-test")
        .config(
            "spark.executorEnv.PYTHONPATH", str(Path(__file__).resolve().parents[3])
        )
    )
    try:
        spark = builder.getOrCreate()
    except Exception as exc:  # noqa: BLE001 - Spark startup can fail outside Python.
        pytest.skip(f"Spark runtime unavailable: {exc}")

    with spark:
        try:
            result = spark.sparkContext.parallelize([1], 1).map(lambda x: x).collect()
            assert result == [1]
        except Exception as exc:  # noqa: BLE001 - Worker startup errors vary by runtime.
            pytest.skip(f"Spark Python workers unavailable: {exc}")

        with pytest.warns(UserWarning, match="CRITICAL WARNING"):
            optimizer = Optimizer(
                space=Continuous({"x": (-1.0, 1.0)}, cost_function=worker_fitness),
                algorithm=IntegrationProbeAlgorithm(),
                backend=SparkDistributed(
                    spark, n_executors=2, communication_port=18081
                ),
                migration=IslandIsolation(),
                n_particles=1,
                n_iterations=1,
                seed=BASE_SEED,
                history="iteration",
            ).fit()

    candidates: list[tuple[float, str, float]] = []
    for values in optimizer.history_generator:
        row: dict[str, object] = dict(
            zip(optimizer.history_columns, values, strict=True)
        )
        if row["iteration"] == 1:
            identifier, x = row["identifier"], row["x"]
            assert isinstance(identifier, str) and isinstance(x, float)
            assert identifier == f"{row['origin']}|particle:0"
            fitness = -sphere(x)
            assert row["fitness"] == fitness < 0.0
            candidates.append((fitness, identifier, x))

    assert sorted(identifier for _, identifier, _ in candidates) == [
        "island:0|particle:0",
        "island:1|particle:0",
    ]
    fitness, identifier, x = min(candidates)
    assert optimizer.result_ == {
        "identifier": identifier,
        "variables": {"x": x},
        "fitness": fitness,
    }
    assert_optimizer_result_consistent(optimizer)
