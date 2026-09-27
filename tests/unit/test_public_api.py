"""Stable public package exports."""

from importlib import import_module
from types import ModuleType

import pytest

import tyrannis
from tyrannis import algorithm, backend, core, migration, processor, space


@pytest.mark.parametrize(
    ("package", "exports"),
    [
        pytest.param(tyrannis, {"Optimizer": "tyrannis.main"}, id="tyrannis"),
        pytest.param(
            algorithm,
            {
                "AntColony": "tyrannis.algorithm.ant_colony",
                "BeeColony": "tyrannis.algorithm.bee_colony",
                "CMAES": "tyrannis.algorithm.cma_es",
                "DifferentialEvolution": "tyrannis.algorithm.differential_evolution",
                "GeneticAlgorithm": "tyrannis.algorithm.genetic_algorithm",
                "GSA": "tyrannis.algorithm.gravitational",
                "GreyWolf": "tyrannis.algorithm.grey_wolf",
                "PSO": "tyrannis.algorithm.pso",
                "PSOGSA": "tyrannis.algorithm.psogsa",
                "WhaleAlgorithm": "tyrannis.algorithm.whale_algorithm",
            },
            id="algorithm",
        ),
        pytest.param(
            backend,
            {
                "SparkDistributed": "tyrannis.backend.distributed.spark_distributed",
                "SparkParallel": "tyrannis.backend.parallel.spark_parallel",
            },
            id="backend",
        ),
        pytest.param(
            core,
            {
                "FITNESS_UNDEFINED": "tyrannis.core.algorithm",
                "AlgorithmBase": "tyrannis.core.algorithm",
                "ParticleBase": "tyrannis.core.algorithm",
            },
            id="core",
        ),
        pytest.param(
            migration,
            {
                "GlobalBest": "tyrannis.migration.global_best",
                "IslandMigration": "tyrannis.migration.island_migration",
            },
            id="migration",
        ),
        pytest.param(
            processor,
            {
                "Joblib": "tyrannis.processor.joblib",
                "ProcessPool": "tyrannis.processor.process",
                "ThreadsPool": "tyrannis.processor.threads",
            },
            id="processor",
        ),
        pytest.param(
            space,
            {
                "Binary": "tyrannis.space.binary",
                "Categorical": "tyrannis.space.categorical",
                "Continuous": "tyrannis.space.continuous",
                "Integer": "tyrannis.space.integer",
                "Mixed": "tyrannis.space.mixed",
                "Ordinal": "tyrannis.space.ordinal",
                "Permutation": "tyrannis.space.permutation",
                "Sequence": "tyrannis.space.sequence",
            },
            id="space",
        ),
    ],
)
def test_stable_public_exports_remain_available(
    package: ModuleType, exports: dict[str, str]
) -> None:
    assert set(exports) <= set(package.__all__)

    for name, source_name in exports.items():
        assert getattr(package, name) is getattr(import_module(source_name), name)
