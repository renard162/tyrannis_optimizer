"""Package exports and the backend's small lazy-loading contract."""

from importlib import import_module
from types import ModuleType, SimpleNamespace

import pytest

import tyrannis
from tyrannis import algorithm, backend, core, migration, processor, space


@pytest.mark.parametrize(
    ("package", "declared", "exports"),
    [
        pytest.param(
            tyrannis, tyrannis.__all__, {"Optimizer": "tyrannis.main"}, id="tyrannis"
        ),
        pytest.param(
            algorithm,
            algorithm.__all__,
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
            core,
            core.__all__,
            {
                "FITNESS_UNDEFINED": "tyrannis.core.algorithm",
                "AlgorithmBase": "tyrannis.core.algorithm",
                "ParticleBase": "tyrannis.core.algorithm",
            },
            id="core",
        ),
        pytest.param(
            migration,
            migration.__all__,
            {
                "GlobalBest": "tyrannis.migration.global_best",
                "IslandMigration": "tyrannis.migration.island_migration",
            },
            id="migration",
        ),
        pytest.param(
            processor,
            processor.__all__,
            {
                "Joblib": "tyrannis.processor.joblib",
                "ProcessPool": "tyrannis.processor.process",
                "ThreadsPool": "tyrannis.processor.threads",
            },
            id="processor",
        ),
        pytest.param(
            space,
            space.__all__,
            {
                "Binary": "tyrannis.space.binary",
                "Categorical": "tyrannis.space.categorical",
                "Continuous": "tyrannis.space.continuous",
                "Integer": "tyrannis.space.integer",
                "Mixed": "tyrannis.space.mixed",
                "Permutation": "tyrannis.space.permutation",
            },
            id="space",
        ),
    ],
)
def test_package_exports_are_the_concrete_objects(
    package: ModuleType, declared: list[str], exports: dict[str, str]
) -> None:
    assert set(declared) == set(exports)
    for name, source_name in exports.items():
        assert getattr(package, name) is getattr(import_module(source_name), name)


def test_backend_rejects_unknown_export() -> None:
    with pytest.raises(AttributeError, match="has no attribute 'unknown'"):
        backend.__getattr__("unknown")


@pytest.mark.parametrize(
    ("name", "source"),
    [
        ("SparkParallel", ".parallel.spark_parallel"),
        ("SparkDistributed", ".distributed.spark_distributed"),
    ],
    ids=["parallel", "distributed"],
)
def test_backend_loads_known_export_lazily(
    monkeypatch: pytest.MonkeyPatch, name: str, source: str
) -> None:
    exported = object()
    calls: list[tuple[str, str]] = []

    def import_module_stub(module_name: str, package_name: str) -> SimpleNamespace:
        calls.append((module_name, package_name))
        return SimpleNamespace(**{name: exported})

    monkeypatch.setattr(backend.importlib, "import_module", import_module_stub)

    assert backend.__getattr__(name) is exported
    assert calls == [(source, "tyrannis.backend")]
