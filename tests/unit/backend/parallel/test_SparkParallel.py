import cloudpickle
import numpy as np
import pandas as pd
import pytest
from doubles.algorithm import DummyAlgorithm

from tyrannis.backend.parallel.spark_parallel import (
    SparkParallel,
    SparkParallelCostFunctionWrapper,
    _process_particle_batches,
)


class DummySparkDataFrame:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.map_function = None
        self.map_schema = None

    def mapInPandas(
        self,
        function: object,
        schema: object,
    ) -> "DummySparkDataFrame":
        self.map_function = function
        self.map_schema = schema
        return self

    def collect(self) -> list[dict[str, object]]:
        assert self.map_function is not None

        batches = iter(
            [
                pd.DataFrame(
                    {
                        "particle_id": [row["particle_id"] for row in self.rows],
                    }
                )
            ]
        )

        result = self.map_function(batches)  # type: ignore

        return [
            {
                "particles": row["particles"],
            }
            for batch in result
            for _, row in batch.iterrows()
        ]


class DummySparkSession:
    def __init__(self) -> None:
        self.created_rows: list[tuple[str]] = []

    def createDataFrame(
        self,
        data: list[tuple[str]],
        columns: list[str],
    ) -> DummySparkDataFrame:
        assert columns == ["particle_id"]

        self.created_rows = data

        return DummySparkDataFrame(
            [{"particle_id": particle_id} for (particle_id,) in data]
        )


def create_failing_algorithm() -> DummyAlgorithm:
    algorithm = DummyAlgorithm()

    def failing_fitness(variables: dict[str, float]) -> float:
        raise RuntimeError("fitness failure")

    algorithm.initialize_context(
        fitness_function=failing_fitness,
        boundaries={"x": (-1.0, 1.0)},
    )

    algorithm.configure(
        identifier="test",
        cost_function_wrapper=SparkParallelCostFunctionWrapper,
        seed=42,
    )

    algorithm.create_particle(
        identifier="particle:0",
        variables={"x": 0.5},
    )

    return algorithm


def create_algorithm() -> DummyAlgorithm:
    algorithm = DummyAlgorithm()

    algorithm.initialize_context(
        fitness_function=lambda variables: float(
            np.sum(
                np.fromiter(
                    variables.values(),
                    dtype=float,
                )
                ** 2
            )
        ),
        boundaries={"x": (-1.0, 1.0)},
    )

    algorithm.configure(
        identifier="test",
        cost_function_wrapper=SparkParallelCostFunctionWrapper,
        seed=42,
    )

    algorithm.create_particle(
        identifier="particle:0",
        variables={"x": 0.5},
    )

    return algorithm


def test_init() -> None:
    spark = DummySparkSession()

    backend = SparkParallel(spark)  # type: ignore

    assert backend._spark is spark
    assert backend._actual_iter == -1
    assert backend.identifier == "SparkParallel"
    assert backend._cost_function_wrapper is SparkParallelCostFunctionWrapper


def test_init_rejects_none_spark() -> None:
    with pytest.raises(
        ValueError,
        match="Spark session cannot be None",
    ):
        SparkParallel(None)  # type: ignore


def test_actual_iter() -> None:
    backend = SparkParallel(DummySparkSession())  # type: ignore

    assert backend.actual_iter == -1

    backend._actual_iter = 3

    assert backend.actual_iter == 3


def test_local_best_is_empty_without_solution() -> None:
    algorithm = create_algorithm()

    backend = SparkParallel(DummySparkSession())  # type: ignore

    backend.initialize_context(
        algorithm=algorithm,
        n_iter=1,
        n_particles=1,
    )

    assert backend.local_best == ""


def test_initialize_context() -> None:
    spark = DummySparkSession()
    algorithm = create_algorithm()

    backend = SparkParallel(spark)  # type: ignore

    backend.initialize_context(
        algorithm=algorithm,
        n_iter=5,
        n_particles=10,
        fitness_failure_strategy="raise",
        seed=42,
    )

    assert backend._algorithm is algorithm
    assert backend._n_iter == 5
    assert backend._n_particles == 10
    assert backend._fitness_failure_strategy == "raise"
    assert backend._seed == 42


def test_parallel_initialize_particles() -> None:
    spark = DummySparkSession()
    algorithm = create_algorithm()

    backend = SparkParallel(spark)  # type: ignore

    backend.initialize_context(
        algorithm=algorithm,
        n_iter=1,
        n_particles=1,
    )

    particles = backend._parallel_initialize_particles(
        ["particle:0"],
    )

    assert len(particles) == 1
    assert particles[0].identifier == "particle:0"
    assert particles[0].fitness is not None

    np.testing.assert_allclose(
        particles[0].fitness,
        0.25,
    )

    assert spark.created_rows == [("particle:0",)]


def test_parallel_process_particles_returns_empty_for_empty_input() -> None:
    backend = SparkParallel(DummySparkSession())  # type: ignore

    result = backend._parallel_process_particles(
        particle_ids=[],
        initialize_particle=True,
    )

    assert result == []


def test_parallel_update_particles_accepts_population_dict() -> None:
    spark = DummySparkSession()
    algorithm = create_algorithm()

    particle = algorithm.population["particle:0"]

    algorithm.initialize_particle("particle:0")
    algorithm.update_solution_state()

    backend = SparkParallel(spark)  # type: ignore

    backend.initialize_context(
        algorithm=algorithm,
        n_iter=1,
        n_particles=1,
    )

    particles = backend._parallel_update_particles(
        {
            particle.identifier: particle,
        }
    )

    assert len(particles) == 1
    assert particles[0].identifier == "particle:0"
    assert particles[0].candidate_fitness is not None


def test_process_particle_batches_initializes_particles() -> None:
    algorithm = create_algorithm()

    serialized_algorithm = __import__("pyspark").cloudpickle.dumps(algorithm)

    batches = iter(
        [
            pd.DataFrame(
                {
                    "particle_id": ["particle:0"],
                }
            )
        ]
    )

    result = list(
        _process_particle_batches(
            batches=batches,
            serialized_algorithm=serialized_algorithm,
            initialize_particle=True,
            fitness_failure_strategy="raise",
        )
    )

    assert len(result) == 1
    assert "particles" in result[0]

    particles = __import__("pyspark").cloudpickle.loads(result[0]["particles"].iloc[0])

    assert len(particles) == 1
    assert particles[0].fitness is not None

    np.testing.assert_allclose(
        particles[0].fitness,
        0.25,
    )


def test_process_particle_batches_invalidates_failed_particle() -> None:
    algorithm = create_failing_algorithm()

    serialized_algorithm = cloudpickle.dumps(algorithm)

    batches = iter(
        [
            pd.DataFrame(
                {
                    "particle_id": ["particle:0"],
                }
            )
        ]
    )

    result = list(
        _process_particle_batches(
            batches=batches,
            serialized_algorithm=serialized_algorithm,
            initialize_particle=True,
            fitness_failure_strategy="invalidate",
        )
    )

    particles = cloudpickle.loads(
        result[0]["particles"].iloc[0],
    )

    assert len(particles) == 1
    assert particles[0].candidate_fitness is not None

    np.testing.assert_equal(
        particles[0].candidate_fitness,
        np.inf,
    )


def test_process_particle_batches_raises_when_requested() -> None:
    algorithm = create_failing_algorithm()

    serialized_algorithm = cloudpickle.dumps(algorithm)

    batches = iter(
        [
            pd.DataFrame(
                {
                    "particle_id": ["particle:0"],
                }
            )
        ]
    )

    with pytest.raises(
        RuntimeError,
        match="fitness failure",
    ):
        list(
            _process_particle_batches(
                batches=batches,
                serialized_algorithm=serialized_algorithm,
                initialize_particle=True,
                fitness_failure_strategy="raise",
            )
        )
