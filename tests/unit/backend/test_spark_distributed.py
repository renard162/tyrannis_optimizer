"""Contracts owned by the Spark distributed backend."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, final, override
from unittest.mock import Mock

import numpy as np
import pytest
from pyspark.sql import SparkSession

from tests._support.numerics import BASE_SEED
from tests._support.objectives import encoded_sphere
from tests._support.processors import (
    ProcessorAlgorithmDouble,
    ProcessorMigrationDriverDouble,
)
from tyrannis.backend.distributed.communication.spark_communication import (
    SparkCommunicationProcessor,
)
from tyrannis.backend.distributed.spark_distributed import (
    SparkDistributed,
    SparkDistributedCostFunctionWrapper,
    _run_processor,  # pyright: ignore[reportPrivateUsage]  # Spark worker boundary is an explicit unit contract.
)
from tyrannis.core.algorithm import CostFunctionWrapperBase
from tyrannis.core.processor import ProcessorBase
from tyrannis.core.results import HistoryConfig, ProcessorResult
from tyrannis.core.signals import EventProtocol


class _SparkContext(Protocol):
    def initialize_context(
        self,
        algorithm: ProcessorAlgorithmDouble,
        n_iter: int,
        n_particles: int,
        migration: ProcessorMigrationDriverDouble,
        processor: ProcessorBase[EventProtocol] | None,
        fitness_failure_strategy: str,
        history_config: HistoryConfig,
        seed: int | None,
    ) -> None: ...


@final
class _WorkerProcessor(ProcessorBase[EventProtocol]):
    def __init__(self, events: list[str], *, failure: Exception | None = None) -> None:
        super().__init__()
        self._cost_function_wrapper = CostFunctionWrapperBase
        self._identifier = "island:0"
        self._result = ProcessorResult()
        self.events = events
        self.failure = failure

    @override
    def initialize_execution_context(self) -> None:
        self.events.append("initialize")

    @override
    def run(self) -> None:
        self.events.append("run")
        if self.failure is not None:
            raise self.failure

    @override
    def finalize_execution_context(self) -> None:
        self.events.append("finalize")

    @override
    def initialize_loop_context(self) -> None:
        return None

    @override
    def finalize_loop_context(self) -> None:
        return None


def _spark() -> SparkSession:
    return Mock(spec=SparkSession)


def _invoke_constructor(
    constructor: Callable[..., SparkDistributed], kwargs: dict[str, object]
) -> SparkDistributed:
    return constructor(**kwargs)


def _construct_runtime(**kwargs: object) -> SparkDistributed:
    return _invoke_constructor(SparkDistributed, kwargs)


def _initialize(
    backend: _SparkContext,
    migration: ProcessorMigrationDriverDouble,
    *,
    history_config: HistoryConfig | None = None,
    processor: ProcessorBase[EventProtocol] | None = None,
) -> None:
    algorithm = ProcessorAlgorithmDouble()
    algorithm.initialize_context(
        fitness_function=lambda variables: np.float64(encoded_sphere(variables)),
        boundaries={},
        n_iter=0,
        n_particles=0,
    )
    backend.initialize_context(
        algorithm=algorithm,
        n_iter=0,
        n_particles=0,
        migration=migration,
        processor=processor,
        fitness_failure_strategy="raise",
        history_config=history_config or HistoryConfig(),
        seed=BASE_SEED,
    )


def _ready_backend(
    monkeypatch: pytest.MonkeyPatch,
    *,
    archive: Path | None = None,
    processor: ProcessorBase[EventProtocol] | None = None,
) -> tuple[SparkDistributed, SparkSession, ProcessorMigrationDriverDouble]:
    spark = _spark()
    backend = SparkDistributed(
        spark, n_executors=2, communication_port=23456, code_archive=archive
    )
    migration = ProcessorMigrationDriverDouble()
    monkeypatch.setattr(migration, "initialize_context", Mock())
    _initialize(backend, migration, processor=processor)
    return backend, spark, migration


def test_constructor_keeps_spark_configuration_and_wrapper(tmp_path: Path) -> None:
    spark = _spark()
    archive = tmp_path / "code.zip"

    backend = SparkDistributed(
        spark, n_executors=3, communication_port=23456, code_archive=archive
    )

    assert vars(backend)["_spark"] is spark
    assert vars(backend)["_n_executors"] == 3
    assert vars(backend)["_communication_port"] == 23456
    assert vars(backend)["_code_archive"] == archive
    assert backend.identifier == "SparkDistributed"
    assert (
        vars(backend)["_cost_function_wrapper"] is SparkDistributedCostFunctionWrapper
    )


def test_constructor_rejects_missing_spark() -> None:
    with pytest.raises(ValueError, match="Spark session cannot be None"):
        _ = _construct_runtime(spark=None, n_executors=1, communication_port=23456)


@pytest.mark.parametrize(
    "n_executors, error",
    [("2", TypeError), (True, TypeError), (0, ValueError)],
    ids=["type", "boolean", "nonpositive"],
)
def test_constructor_rejects_invalid_executor_count(
    n_executors: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        _ = _construct_runtime(
            spark=_spark(), n_executors=n_executors, communication_port=23456
        )


@pytest.mark.parametrize(
    "port, error",
    [("18081", TypeError), (True, TypeError), (0, ValueError), (65536, ValueError)],
    ids=["type", "boolean", "below_range", "above_range"],
)
def test_constructor_rejects_invalid_port(port: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        _ = _construct_runtime(spark=_spark(), n_executors=1, communication_port=port)


def test_constructor_warns_when_using_default_port() -> None:
    with pytest.warns(UserWarning, match="default communication port"):
        backend = SparkDistributed(_spark(), n_executors=1)

    assert vars(backend)["_communication_port"] == 18081


def test_initialize_context_configures_migration_for_each_island(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spark = _spark()
    get_host = Mock(return_value="10.0.0.1")
    monkeypatch.setattr(spark.conf, "get", get_host)
    backend = SparkDistributed(spark, n_executors=3, communication_port=23456)
    migration = ProcessorMigrationDriverDouble()
    configure_migration = Mock()
    communication_driver = object()
    communication_factory = Mock(return_value=communication_driver)
    monkeypatch.setattr(migration, "initialize_context", configure_migration)
    monkeypatch.setattr(
        "tyrannis.backend.distributed.spark_distributed.SparkCommunicationDriver",
        communication_factory,
    )
    history = HistoryConfig()

    _initialize(backend, migration, history_config=history)

    communication_factory.assert_called_once_with(
        island_ids=["island:0", "island:1", "island:2"], port=23456
    )
    get_host.assert_called_once_with("spark.driver.host")
    configure_migration.assert_called_once_with(
        communication_driver=communication_driver,
        communication_processor_class=SparkCommunicationProcessor,
        communication_processor_kargs={"driver_ip": "10.0.0.1", "port": 23456},
        history_config=history,
        seed=BASE_SEED,
    )


def test_initialize_context_warns_when_history_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = SparkDistributed(_spark(), n_executors=1, communication_port=23456)
    migration = ProcessorMigrationDriverDouble()
    monkeypatch.setattr(migration, "initialize_context", Mock())

    with pytest.warns(UserWarning, match="SPARK HISTORY ENABLED"):
        _initialize(backend, migration, history_config=HistoryConfig(iteration=True))


def test_execute_creates_pool_distributes_processors_and_consolidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    processor = Mock(spec=ProcessorBase)
    pool: dict[str, ProcessorBase[EventProtocol]] = {}
    processor.processors_pool = pool
    backend, spark, migration = _ready_backend(monkeypatch, processor=processor)
    workers: list[ProcessorBase[EventProtocol]] = [
        Mock(spec=ProcessorBase),
        Mock(spec=ProcessorBase),
    ]
    results = [("island:0", ProcessorResult()), ("island:1", ProcessorResult())]

    def create_pool() -> None:
        events.append("create_pool")
        pool.update({"island:0": workers[0], "island:1": workers[1]})

    def parallelize(*, c: list[object], numSlices: int) -> Mock:
        events.append("parallelize")
        assert c == workers
        assert numSlices == 2
        return rdd

    rdd = Mock()
    collect = Mock(return_value=results)
    mapped = Mock()
    monkeypatch.setattr(mapped, "collect", collect)
    map_processors = Mock(return_value=mapped)
    monkeypatch.setattr(rdd, "map", map_processors)
    monkeypatch.setattr(backend, "init_processors", create_pool)
    monkeypatch.setattr(spark.sparkContext, "parallelize", parallelize)
    monkeypatch.setattr(migration, "start", lambda: events.append("start"))
    monkeypatch.setattr(migration, "stop", lambda: events.append("stop"))
    monkeypatch.setattr(
        backend, "update_result", lambda: events.append("update_result")
    )

    backend.execute()

    map_processors.assert_called_once_with(_run_processor)
    collect.assert_called_once_with()
    assert vars(backend)["_local_bests"] == dict(results)
    assert events == ["create_pool", "start", "parallelize", "update_result", "stop"]


def test_execute_rejects_missing_processor(monkeypatch: pytest.MonkeyPatch) -> None:
    backend, _, migration = _ready_backend(monkeypatch)
    start = Mock()
    monkeypatch.setattr(migration, "start", start)

    with pytest.raises(RuntimeError, match="Processor cannot be None"):
        backend.execute()

    start.assert_not_called()


def test_execute_adds_archive_and_stops_migration_after_spark_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive = tmp_path / "code.zip"
    archive.touch()
    processor = Mock(spec=ProcessorBase)
    processor.processors_pool = {"island:0": Mock(spec=ProcessorBase)}
    backend, spark, migration = _ready_backend(
        monkeypatch, archive=archive, processor=processor
    )
    add_file = Mock()
    create_pool = Mock()
    monkeypatch.setattr(spark.sparkContext, "addPyFile", add_file)
    monkeypatch.setattr(backend, "init_processors", create_pool)
    events: list[str] = []
    monkeypatch.setattr(
        spark.sparkContext, "parallelize", Mock(side_effect=ValueError("Spark failed"))
    )
    monkeypatch.setattr(migration, "start", lambda: events.append("start"))
    monkeypatch.setattr(migration, "stop", lambda: events.append("stop"))

    with pytest.raises(ValueError, match="Spark failed"):
        backend.execute()

    add_file.assert_called_once_with(str(archive))
    create_pool.assert_not_called()
    assert events == ["start", "stop"]


def test_execute_rejects_missing_archive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    processor = Mock(spec=ProcessorBase)
    backend, spark, migration = _ready_backend(
        monkeypatch, archive=tmp_path / "missing.zip", processor=processor
    )
    add_file = Mock()
    start = Mock()
    monkeypatch.setattr(spark.sparkContext, "addPyFile", add_file)
    monkeypatch.setattr(migration, "start", start)

    with pytest.raises(FileNotFoundError, match="Spark code archive not found"):
        backend.execute()

    add_file.assert_not_called()
    start.assert_not_called()


def test_run_processor_initializes_runs_and_finalizes() -> None:
    events: list[str] = []
    processor = _WorkerProcessor(events)

    returned = _run_processor(processor)

    assert returned == ("island:0", processor.result)
    assert events == ["initialize", "run", "finalize"]


def test_run_processor_finalizes_after_run_failure() -> None:
    events: list[str] = []
    failure = ValueError("worker failed")
    processor = _WorkerProcessor(events, failure=failure)

    with pytest.raises(ValueError) as error:
        _ = _run_processor(processor)

    assert error.value is failure
    assert events == ["initialize", "run", "finalize"]
