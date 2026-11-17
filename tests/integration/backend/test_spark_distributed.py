"""Spark islands exchange a best particle and return their local updates."""

import json
import secrets
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

import pytest
from _support.assertions import assert_optimizer_result_consistent
from _support.integration_probe import IntegrationProbeAlgorithm
from _support.numerics import BASE_SEED
from _support.objectives import sphere
from integration.migration._history import HistoryRecord, population_at

from tyrannis import Optimizer
from tyrannis.backend import SparkDistributed
from tyrannis.migration import GlobalBest
from tyrannis.space import Continuous


@pytest.mark.optional
@pytest.mark.multiprocess
@pytest.mark.timeout(240, method="thread")
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
            "spark.executorEnv.PYTHONPATH", str(Path(__file__).resolve().parents[2])
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

        communication_port = 18081 + secrets.randbelow(101)
        backend = SparkDistributed(
            spark, n_executors=2, communication_port=communication_port
        )
        with pytest.warns(UserWarning, match="CRITICAL WARNING"):
            optimizer = Optimizer(
                space=Continuous({"x": (-1.0, 1.0)}, cost_function=worker_fitness),
                algorithm=IntegrationProbeAlgorithm(),
                backend=backend,
                migration=GlobalBest(
                    synchronous=True, initial_iter=1, check_interval=1
                ),
                n_particles=1,
                n_iterations=1,
                seed=BASE_SEED,
                history="iteration",
            ).fit()

    records: list[HistoryRecord] = [
        json.loads(entry) for entry in backend.result.history
    ]
    candidates: list[tuple[float, str, float]] = []
    source_by_slot: dict[str, str] = {}
    for values in optimizer.history_generator:
        row: dict[str, object] = dict(
            zip(optimizer.history_columns, values, strict=True)
        )
        if row["iteration"] == 1:
            origin, identifier, x = row["origin"], row["identifier"], row["x"]
            assert isinstance(origin, str)
            assert isinstance(identifier, str) and isinstance(x, float)
            assert identifier == f"{origin}|particle:0"
            fitness = -sphere(x)
            assert row["fitness"] == fitness < 0.0
            candidates.append((fitness, identifier, x))

            # The checkpoint copies the source's trace into the receiver's slot.
            # The subsequent update records the local ID on that same particle.
            trace = population_at(records, origin, 1)[identifier]["trace"]
            source = trace[0]["particle_id"]
            assert isinstance(source, str)
            assert trace[-1]["particle_id"] == identifier
            source_by_slot[identifier] = source

    assert sorted(identifier for _, identifier, _ in candidates) == [
        "island:0|particle:0",
        "island:1|particle:0",
    ]
    # Either island can win the tie at initialization.
    (donor,) = set(source_by_slot.values())
    assert donor in source_by_slot
    fitness, identifier, x = min(candidates)
    assert optimizer.result_ == {
        "identifier": identifier,
        "variables": {"x": x},
        "fitness": fitness,
    }
    assert_optimizer_result_consistent(optimizer)
    first_result = optimizer.result_
    first_history = list(optimizer.internal_history_generator_)
    with patch.object(backend, "execute") as execute:
        with pytest.raises(RuntimeError, match="local and parallel"):
            _ = optimizer.fit()
        execute.assert_not_called()
    assert optimizer.result_ == first_result
    assert list(optimizer.internal_history_generator_) == first_history
