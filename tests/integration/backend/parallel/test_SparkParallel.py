import importlib
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from pyspark.sql import SparkSession

from tyrannis.algorithm.pso import PSO
from tyrannis.backend.parallel.spark_parallel import SparkParallel
from tyrannis.examples.bowl_shaped import sphere

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
    code_archive: str | Path | None = None,
    fitness_function=sphere,
) -> SparkParallel:
    algorithm = PSO()

    algorithm.initialize_context(
        fitness_function=fitness_function,
        boundaries=BOUNDARIES,
    )

    backend = SparkParallel(
        spark,
        code_archive=code_archive,
    )

    backend.initialize_context(
        algorithm=algorithm,
        n_iter=N_ITER,
        n_particles=N_PARTICLES,
        seed=seed,
    )

    return backend


def create_cost_function_archive(
    tmp_path: Path,
) -> Path:
    package_dir = tmp_path / "custom_cost"

    package_dir.mkdir()

    (package_dir / "__init__.py").write_text(
        "",
        encoding="utf-8",
    )

    (package_dir / "functions.py").write_text(
        """
import numpy as np


def archive_cost(x: dict[str, float]) -> float:
    values = np.fromiter(x.values(), dtype=float)

    return float(
        1000.0 + np.sum(values**2)
    )
""",
        encoding="utf-8",
    )

    archive = tmp_path / "custom_cost.zip"

    with zipfile.ZipFile(
        archive,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zip_file:
        zip_file.write(
            package_dir / "__init__.py",
            "custom_cost/__init__.py",
        )
        zip_file.write(
            package_dir / "functions.py",
            "custom_cost/functions.py",
        )

    return archive


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
    first = create_backend(
        spark,
        seed=SEED,
    )

    second = create_backend(
        spark,
        seed=SEED,
    )

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


def test_execute_with_cost_function_from_code_archive(
    spark: SparkSession,
    tmp_path: Path,
) -> None:
    if sys.platform == "win32":
        pytest.skip("Code archive integration test requires a Linux Spark driver.")

    archive = create_cost_function_archive(tmp_path)

    sys.path.insert(0, str(archive))

    try:
        module = importlib.import_module("custom_cost.functions")
        archive_cost = module.archive_cost
    finally:
        sys.path.remove(str(archive))

    sys.modules.pop("custom_cost.functions", None)
    sys.modules.pop("custom_cost", None)

    backend = create_backend(
        spark=spark,
        code_archive=archive,
        fitness_function=archive_cost,
    )

    backend.execute()

    assert backend.actual_iter == N_ITER

    assert len(backend._algorithm.population) == N_PARTICLES

    for particle in backend._algorithm.population.values():
        assert particle.fitness is not None
        assert np.isfinite(particle.fitness)
        assert particle.fitness >= 1000.0

    assert backend._algorithm.local_best is not None

    result = backend.result

    assert result is not None
    assert result["fitness"] >= 1000.0
