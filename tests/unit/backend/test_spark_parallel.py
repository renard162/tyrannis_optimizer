"""Unit contracts for the Spark parallel backend without a Spark cluster."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from copy import deepcopy
from pathlib import Path
from pickle import dumps
from typing import Protocol, final, runtime_checkable
from unittest.mock import Mock

import joblib  # pyright: ignore[reportMissingTypeStubs]  # Joblib lacks stubs.
import numpy as np
import pytest

from tests._support.numerics import BASE_SEED
from tests._support.objectives import constant_objective
from tests._support.processors import (
    ProcessorAlgorithmDouble,
    ProcessorMigrationDriverDouble,
    ProcessorParticleDouble,
    assert_second_phase_dispatch,
)
from tyrannis.core.results import HistoryConfig, ProcessorResult
from tyrannis.core.signals import EventProtocol

pytest.importorskip("pandas")
pytest.importorskip("pyspark")

import pandas as pd  # pyright: ignore[reportMissingTypeStubs]  # Optional extra lacks stubs.
from pyspark import cloudpickle
from pyspark.sql import SparkSession
from pyspark.sql.types import BinaryType, StructType

from tyrannis.backend.parallel.spark_parallel import (
    SparkParallel,
    SparkParallelCostFunctionWrapper,
    _process_particle_batches,  # pyright: ignore[reportPrivateUsage]  # Direct worker unit.
)
from tyrannis.core.algorithm import FITNESS_UNDEFINED, ParticleBase
from tyrannis.core.backend_migration import MigrationDriverBase
from tyrannis.core.processor import ProcessorBase

pytestmark = pytest.mark.optional


class _SparkContext(Protocol):
    def initialize_context(
        self,
        algorithm: ProcessorAlgorithmDouble,
        n_iter: int,
        n_particles: int,
        migration: MigrationDriverBase,
        processor: ProcessorBase[EventProtocol] | None,
        fitness_failure_strategy: str,
        history_config: HistoryConfig,
        seed: int | None,
    ) -> None: ...


@runtime_checkable
class _SerializedColumn(Protocol):
    def __iter__(self) -> Iterator[bytes]: ...


class _ObservableBackend(SparkParallel):
    def update_ids(
        self, particle_ids: dict[str, ParticleBase] | list[str]
    ) -> list[ParticleBase]:
        return self._parallel_update_particles(particle_ids)


@final
class _CollectedFrame:
    def __init__(self, rows: list[dict[str, bytes]]) -> None:
        self.rows = rows

    def collect(self) -> list[dict[str, bytes]]:
        return self.rows


@final
class _InputFrame:
    def __init__(self, rows: list[dict[str, bytes]]) -> None:
        self.rows = rows
        self.schema: object | None = None
        self.worker: (
            Callable[[Iterable[pd.DataFrame]], Iterator[pd.DataFrame]] | None
        ) = None

    def mapInPandas(
        self,
        worker: Callable[[Iterable[pd.DataFrame]], Iterator[pd.DataFrame]],
        schema: object,
    ) -> _CollectedFrame:
        self.worker = worker
        self.schema = schema
        return _CollectedFrame(self.rows)


def _spark() -> SparkSession:
    return Mock(spec=SparkSession)


def _invoke_constructor(
    constructor: Callable[..., SparkParallel], kwargs: dict[str, object]
) -> SparkParallel:
    # Invalid runtime arguments must bypass the static constructor signature.
    return constructor(**kwargs)


def _construct_runtime(**kwargs: object) -> SparkParallel:
    return _invoke_constructor(SparkParallel, kwargs)


def _configure(
    backend: _SparkContext,
    algorithm: ProcessorAlgorithmDouble,
    *,
    n_iter: int = 0,
    n_particles: int = 0,
    history_config: HistoryConfig | None = None,
) -> None:
    algorithm.initialize_context(
        fitness_function=lambda variables: np.float64(constant_objective(variables)),
        boundaries={},
        n_iter=n_iter,
        n_particles=n_particles,
    )
    backend.initialize_context(
        algorithm=algorithm,
        n_iter=n_iter,
        n_particles=n_particles,
        migration=ProcessorMigrationDriverDouble(),
        processor=None,
        fitness_failure_strategy="raise",
        history_config=history_config or HistoryConfig(),
        seed=BASE_SEED,
    )


def _stub_particle_processing(
    backend: SparkParallel,
    algorithm: ProcessorAlgorithmDouble,
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[str, list[str], bool]],
) -> None:
    def initialize(particle_ids: list[str]) -> list[ParticleBase]:
        calls.append(("initialize", list(particle_ids), False))
        return [
            algorithm.initialize_particle(identifier) for identifier in particle_ids
        ]

    def update(
        particle_ids: list[str] | dict[str, ParticleBase],
        second_update: bool = False,
    ) -> list[ParticleBase]:
        identifiers = list(particle_ids)
        calls.append(("update", identifiers, second_update))
        operation = (
            algorithm.second_update_particle
            if second_update
            else algorithm.update_particle
        )
        return [operation(identifier) for identifier in identifiers]

    monkeypatch.setattr(backend, "_parallel_initialize_particles", initialize)
    monkeypatch.setattr(backend, "_parallel_update_particles", update)


def _worker_particles(
    algorithm: ProcessorAlgorithmDouble,
    batches: list[list[str]],
    *,
    initialize: bool = False,
    second_update: bool = False,
    strategy: str = "raise",
) -> list[ProcessorParticleDouble]:
    frames = (pd.DataFrame({"particle_id": identifiers}) for identifiers in batches)
    outputs = list(
        _process_particle_batches(
            batches=iter(frames),
            serialized_algorithm=cloudpickle.dumps(algorithm),  # pyright: ignore[reportUnknownMemberType]  # PySpark lacks a typed signature.
            initialize_particle=initialize,
            second_update=second_update,
            fitness_failure_strategy=strategy,
        )
    )
    assert len(outputs) == 1
    column: object = outputs[0]["particles"]  # pyright: ignore[reportUnknownVariableType]  # Pandas extra lacks stubs.
    assert isinstance(column, _SerializedColumn)
    serialized = next(iter(column))
    loaded: object = cloudpickle.loads(serialized)  # pyright: ignore[reportAny]  # Narrow after the pickle boundary.
    assert isinstance(loaded, list)
    items: Iterable[object] = loaded  # pyright: ignore[reportUnknownVariableType]  # Pickle returns an untyped list.
    particles: list[ProcessorParticleDouble] = []
    for item in items:  # pyright: ignore[reportUnknownVariableType]  # Elements are checked below.
        assert isinstance(item, ProcessorParticleDouble)
        particles.append(item)
    return particles


@pytest.mark.parametrize("n_jobs", [0, True, "2"], ids=["zero", "bool", "str"])
def test_constructor_rejects_invalid_aux_jobs(n_jobs: object) -> None:
    expected = ValueError if n_jobs == 0 else TypeError
    with pytest.raises(expected):
        _ = _construct_runtime(spark=_spark(), n_aux_jobs=n_jobs)


@pytest.mark.parametrize(
    "backend_name,expected",
    [(0, TypeError), ("unknown", ValueError)],
    ids=["type", "name"],
)
def test_constructor_rejects_invalid_joblib_backend(
    backend_name: object, expected: type[Exception]
) -> None:
    with pytest.raises(expected):
        _ = _construct_runtime(spark=_spark(), aux_backend=backend_name)


@pytest.mark.parametrize(
    "batch_size,expected",
    [
        (0, ValueError),
        (True, TypeError),
        ("invalid", ValueError),
    ],
    ids=["zero", "bool", "string"],
)
def test_constructor_rejects_invalid_aux_batch_size(
    batch_size: object, expected: type[Exception]
) -> None:
    with pytest.raises(expected):
        _ = _construct_runtime(spark=_spark(), aux_batch_size=batch_size)


@pytest.mark.parametrize(
    "pre_dispatch,expected",
    [(0, ValueError), (True, TypeError), (1.5, TypeError)],
    ids=["zero", "bool", "float"],
)
def test_constructor_rejects_invalid_aux_pre_dispatch(
    pre_dispatch: object, expected: type[Exception]
) -> None:
    with pytest.raises(expected):
        _ = _construct_runtime(spark=_spark(), aux_pre_dispatch=pre_dispatch)


def test_constructor_rejects_missing_spark_session() -> None:
    with pytest.raises(ValueError, match="Spark session cannot be None"):
        _ = _construct_runtime(spark=None)


def test_initialize_context_configures_spark_wrapper_and_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = SparkParallel(_spark())
    algorithm = ProcessorAlgorithmDouble()
    configured: dict[str, object] = {}

    def record_configuration(**kwargs: object) -> None:
        configured.update(kwargs)

    monkeypatch.setattr(algorithm, "configure", record_configuration)
    _configure(backend, algorithm)

    assert configured == {
        "identifier": "SparkParallel|algorithm",
        "cost_function_wrapper": SparkParallelCostFunctionWrapper,
        "seed": BASE_SEED,
    }
    assert isinstance(backend.result, ProcessorResult)
    assert backend.result.result is None
    assert backend.result.history == []


def test_execute_distributes_existing_code_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "code.zip"
    archive.touch()
    spark = _spark()
    add_py_file = Mock()
    monkeypatch.setattr(spark.sparkContext, "addPyFile", add_py_file)
    backend = SparkParallel(spark, spark_code_archive=archive)
    _configure(backend, ProcessorAlgorithmDouble())

    backend.execute()

    add_py_file.assert_called_once_with(str(archive))


def test_execute_rejects_missing_code_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "missing.zip"
    spark = _spark()
    add_py_file = Mock()
    monkeypatch.setattr(spark.sparkContext, "addPyFile", add_py_file)
    backend = SparkParallel(spark, spark_code_archive=archive)
    _configure(backend, ProcessorAlgorithmDouble())

    with pytest.raises(FileNotFoundError, match="Spark code archive not found"):
        backend.execute()

    add_py_file.assert_not_called()


def test_execute_runs_initialization_and_iteration_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = SparkParallel(_spark())
    algorithm = ProcessorAlgorithmDouble()
    _configure(
        backend,
        algorithm,
        n_iter=1,
        n_particles=2,
        history_config=HistoryConfig(
            pre_iteration=True, new_particle=True, iteration=True
        ),
    )
    calls: list[tuple[str, list[str], bool]] = []
    _stub_particle_processing(backend, algorithm, monkeypatch, calls)
    initial_result = backend.result

    backend.execute()

    identifiers = ["SparkParallel|particle:0", "SparkParallel|particle:1"]
    assert list(algorithm.population) == identifiers
    assert algorithm.iterations == [("pre", 0), ("post", 0), ("pre", 1), ("post", 1)]
    assert calls == [
        ("initialize", identifiers, False),
        ("update", identifiers, False),
    ]
    assert algorithm.random_cache_calls == [(identifiers, True), (identifiers, False)]
    assert {particle.updates for particle in algorithm.population.values()} == {1}
    assert backend.result is initial_result
    assert len(backend.result.history) == 10
    for event, expected_count in (
        ("pre_iteration", 4),
        ("new_particle", 2),
        ("iteration", 4),
    ):
        name = HistoryConfig.get_event(event)
        assert (
            sum(f'"event": "{name}"' in row for row in backend.result.history)
            == expected_count
        )


def test_execute_passes_auxiliary_joblib_settings_and_uses_consolidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = SparkParallel(
        _spark(),
        n_aux_jobs=2,
        aux_backend="threading",
        aux_batch_size=3,
        aux_pre_dispatch=4,
    )
    algorithm = ProcessorAlgorithmDouble()
    _configure(backend, algorithm, n_particles=1)
    calls: list[tuple[str, list[str], bool]] = []
    _stub_particle_processing(backend, algorithm, monkeypatch, calls)
    parallel_factory = Mock(wraps=joblib.Parallel)
    monkeypatch.setattr(
        "tyrannis.backend.parallel.spark_parallel.Parallel", parallel_factory
    )
    backend.execute()

    parallel_factory.assert_called_once_with(
        n_jobs=2,
        backend="threading",
        batch_size=3,
        pre_dispatch=4,
        return_as="list",
    )
    assert calls == [("initialize", ["SparkParallel|particle:0"], False)]
    assert not algorithm.population["SparkParallel|particle:0"].new_particle


def test_execute_skips_new_particle_block_when_population_is_initialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = SparkParallel(_spark())
    algorithm = ProcessorAlgorithmDouble()
    _configure(backend, algorithm, n_iter=1, n_particles=1)
    algorithm.create_particle("SparkParallel|particle:0")
    algorithm.population["SparkParallel|particle:0"].mark_initialized()
    calls: list[tuple[str, list[str], bool]] = []
    _stub_particle_processing(backend, algorithm, monkeypatch, calls)

    backend.execute()

    assert calls == [("update", ["SparkParallel|particle:0"], False)]
    assert algorithm.random_cache_calls == [(["SparkParallel|particle:0"], False)]
    assert algorithm.population["SparkParallel|particle:0"].updates == 1


def test_execute_double_check_with_empty_selection_skips_second_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = SparkParallel(_spark())
    algorithm = ProcessorAlgorithmDouble()
    _configure(backend, algorithm, n_iter=1, n_particles=1)
    algorithm.double_particle_check = True
    calls: list[tuple[str, list[str], bool]] = []
    _stub_particle_processing(backend, algorithm, monkeypatch, calls)

    backend.execute()

    assert_second_phase_dispatch(algorithm, ["SparkParallel|particle:0"])
    assert [call for call in calls if call[2]] == []


def test_execute_double_check_reuses_selected_ids_after_first_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = SparkParallel(_spark())
    algorithm = ProcessorAlgorithmDouble()
    _configure(backend, algorithm, n_iter=1, n_particles=2)
    algorithm.double_particle_check = True
    selected = ["SparkParallel|particle:1"]
    algorithm.requested_double_check_ids = selected
    calls: list[tuple[str, list[str], bool]] = []
    _stub_particle_processing(backend, algorithm, monkeypatch, calls)

    backend.execute()

    assert_second_phase_dispatch(
        algorithm, ["SparkParallel|particle:0", "SparkParallel|particle:1"]
    )
    assert calls[-1] == ("update", selected, True)


def test_execute_logs_errors_and_best_and_updates_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = SparkParallel(_spark())
    algorithm = ProcessorAlgorithmDouble()
    _configure(
        backend,
        algorithm,
        n_iter=1,
        n_particles=1,
        history_config=HistoryConfig(error=True, best=True),
    )
    calls: list[tuple[str, list[str], bool]] = []
    _stub_particle_processing(backend, algorithm, monkeypatch, calls)
    original_post = algorithm.post_iteration

    def record_best(actual_iter: int) -> None:
        original_post(actual_iter)
        vars(algorithm)["_local_best"] = next(iter(algorithm.population.values()))

    def update_with_error(identifier: str) -> ProcessorParticleDouble:
        particle = deepcopy(algorithm.population[identifier])
        particle.error_fitness = np.float64(7.0)
        return particle

    monkeypatch.setattr(algorithm, "post_iteration", record_best)
    monkeypatch.setattr(algorithm, "update_particle", update_with_error)

    backend.execute()

    error_event = HistoryConfig.get_event("error")
    best_event = HistoryConfig.get_event("local_best")
    assert (
        sum(f'"event": "{error_event}"' in row for row in backend.result.history) == 1
    )
    assert sum(f'"event": "{best_event}"' in row for row in backend.result.history) == 2
    assert algorithm.local_best is not None
    assert backend.result.result == algorithm.local_best()


def test_worker_initializes_all_particles_across_batches() -> None:
    algorithm = ProcessorAlgorithmDouble()
    _configure(SparkParallel(_spark()), algorithm)
    algorithm.create_particle("SparkParallel|particle:0")
    algorithm.create_particle("SparkParallel|particle:1")

    particles = _worker_particles(
        algorithm,
        [["SparkParallel|particle:0"], ["SparkParallel|particle:1"]],
        initialize=True,
    )

    assert [particle.identifier for particle in particles] == [
        "SparkParallel|particle:0",
        "SparkParallel|particle:1",
    ]
    assert all(not particle.new_particle for particle in particles)


@pytest.mark.parametrize("second_update", [False, True], ids=["normal", "second"])
def test_worker_uses_requested_update_operation(
    monkeypatch: pytest.MonkeyPatch, second_update: bool
) -> None:
    algorithm = ProcessorAlgorithmDouble()
    _configure(SparkParallel(_spark()), algorithm)
    algorithm.create_particle("SparkParallel|particle:0")

    def second_update_operation(identifier: str) -> ProcessorParticleDouble:
        particle = deepcopy(algorithm.population[identifier])
        particle.updates = 2
        return particle

    monkeypatch.setattr(algorithm, "second_update_particle", second_update_operation)

    particles = _worker_particles(
        algorithm, [["SparkParallel|particle:0"]], second_update=second_update
    )

    assert len(particles) == 1
    assert particles[0].updates == (2 if second_update else 1)


def test_worker_propagates_algorithm_exception_with_raise_strategy() -> None:
    identifier = "SparkParallel|particle:0"
    algorithm = ProcessorAlgorithmDouble(fail_on_update=identifier)
    _configure(SparkParallel(_spark()), algorithm)
    algorithm.create_particle(identifier)

    with pytest.raises(ValueError, match="failed item"):
        _ = _worker_particles(algorithm, [[identifier]])


def test_worker_recovers_original_particle_after_update_failure() -> None:
    identifier = "SparkParallel|particle:0"
    algorithm = ProcessorAlgorithmDouble(fail_on_update=identifier)
    _configure(SparkParallel(_spark()), algorithm)
    algorithm.create_particle(identifier)

    particles = _worker_particles(algorithm, [[identifier]], strategy="continue")

    assert len(particles) == 1
    assert particles[0].identifier == identifier
    assert particles[0].updates == 0
    assert particles[0].candidate_fitness is not None
    assert np.isinf(particles[0].candidate_fitness)
    assert particles[0].error_fitness == FITNESS_UNDEFINED


def test_worker_preserves_existing_error_fitness_after_failure() -> None:
    identifier = "SparkParallel|particle:0"
    algorithm = ProcessorAlgorithmDouble(fail_on_update=identifier)
    _configure(SparkParallel(_spark()), algorithm)
    algorithm.create_particle(identifier)
    algorithm.population[identifier].error_fitness = np.float64(-7.0)

    particles = _worker_particles(algorithm, [[identifier]], strategy="continue")

    assert particles[0].error_fitness == np.float64(-7.0)
    assert particles[0].candidate_fitness is not None
    assert np.isinf(particles[0].candidate_fitness)


def test_worker_propagates_nan_candidate_with_raise_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identifier = "SparkParallel|particle:0"
    algorithm = ProcessorAlgorithmDouble()
    _configure(SparkParallel(_spark()), algorithm)
    algorithm.create_particle(identifier)

    def nan_update(particle_id: str) -> ProcessorParticleDouble:
        particle = deepcopy(algorithm.population[particle_id])
        particle.candidate_fitness = np.float64(np.nan)
        return particle

    monkeypatch.setattr(algorithm, "update_particle", nan_update)

    with pytest.raises(ValueError, match="NaN is an invalid cost function result"):
        _ = _worker_particles(algorithm, [[identifier]])


def test_worker_records_nan_and_recovers_original_particle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identifier = "SparkParallel|particle:0"
    algorithm = ProcessorAlgorithmDouble()
    _configure(SparkParallel(_spark()), algorithm)
    algorithm.create_particle(identifier)

    def nan_update(particle_id: str) -> ProcessorParticleDouble:
        particle = deepcopy(algorithm.population[particle_id])
        particle.candidate_fitness = np.float64(np.nan)
        return particle

    monkeypatch.setattr(algorithm, "update_particle", nan_update)

    particles = _worker_particles(algorithm, [[identifier]], strategy="continue")

    assert particles[0].updates == 0
    assert particles[0].candidate_fitness is not None
    assert np.isinf(particles[0].candidate_fitness)
    assert particles[0].error_fitness is not None
    assert np.isnan(particles[0].error_fitness)


@pytest.mark.parametrize("as_mapping", [False, True], ids=["ids", "population"])
def test_spark_adapter_submits_population_and_collects_particles(
    monkeypatch: pytest.MonkeyPatch, as_mapping: bool
) -> None:
    spark = _spark()
    backend = _ObservableBackend(spark)
    algorithm = ProcessorAlgorithmDouble()
    _configure(backend, algorithm)
    identifiers = ["SparkParallel|particle:0", "SparkParallel|particle:1"]
    for identifier in identifiers:
        algorithm.create_particle(identifier)
    particles = list(algorithm.population.values())
    frame = _InputFrame(
        [
            {"particles": dumps([particles[0]])},
            {"particles": dumps([particles[1]])},
        ]
    )
    submitted: list[tuple[list[tuple[str]], list[str]]] = []

    def create_data_frame(rows: list[tuple[str]], columns: list[str]) -> _InputFrame:
        submitted.append((rows, columns))
        return frame

    monkeypatch.setattr(spark, "createDataFrame", create_data_frame)

    source: dict[str, ParticleBase] | list[str] = (
        dict(algorithm.population) if as_mapping else identifiers
    )
    returned = backend.update_ids(source)

    assert submitted == [([(identifiers[0],), (identifiers[1],)], ["particle_id"])]
    assert [particle.identifier for particle in returned] == identifiers
    assert isinstance(frame.schema, StructType)
    assert frame.schema.fieldNames() == ["particles"]
    assert isinstance(frame.schema["particles"].dataType, BinaryType)
    assert not frame.schema["particles"].nullable
    assert frame.worker is not None
    worker_frames = list(frame.worker([pd.DataFrame({"particle_id": identifiers})]))
    assert len(worker_frames) == 1
