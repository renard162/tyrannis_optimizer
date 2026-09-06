import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from doubles.algorithm import DummyAlgorithm
from doubles.processor import DummyProcessor

from tyrannis.backend.distributed.spark_distributed import (
    SparkDistributed,
    SparkDistributedCostFunctionWrapper,
    _run_processor,
)
from tyrannis.core.backend_migration import MigrationDriverBase


class DummySparkContext:
    def __init__(self) -> None:
        self.added_py_files: list[str] = []
        self.parallelized_data = None
        self.parallelized_slices = None

    def addPyFile(self, path: str) -> None:
        self.added_py_files.append(path)

    def parallelize(
        self,
        data: list[DummyProcessor],
        numSlices: int,
    ) -> "DummyRDD":
        self.parallelized_data = data
        self.parallelized_slices = numSlices

        return DummyRDD(data)


class DummyRDD:
    def __init__(
        self,
        data: list[DummyProcessor],
    ) -> None:
        self.data = data

    def map(
        self,
        function: object,
    ) -> "DummyMappedRDD":
        return DummyMappedRDD(
            self.data,
            function,
        )


class DummyMappedRDD:
    def __init__(
        self,
        data: list[DummyProcessor],
        function: object,
    ) -> None:
        self.data = data
        self.function = function

    def collect(self) -> list[tuple[str, dict]]:
        return [
            self.function(processor)  # type: ignore
            for processor in self.data
        ]


class DummySparkConf:
    def get(self, key: str) -> str:
        if key == "spark.driver.host":
            return "127.0.0.1"

        raise KeyError(key)


class DummySparkSession:
    def __init__(self) -> None:
        self.sparkContext = DummySparkContext()
        self.conf = DummySparkConf()


def create_algorithm() -> DummyAlgorithm:
    algorithm = DummyAlgorithm()

    algorithm.initialize_context(
        fitness_function=lambda variables: sum(variables.values()),
        boundaries={"x": (-1.0, 1.0)},
    )

    return algorithm


def create_migration() -> Mock:
    return Mock(spec=MigrationDriverBase)


def create_processor(
    migration: MigrationDriverBase | None = None,
) -> DummyProcessor:
    if migration is None:
        migration = create_migration()

    processor = DummyProcessor()

    processor.initialize_context(
        algorithm=create_algorithm(),
        n_iter=1,
        n_particles=1,
        migration_driver=migration,
        seed=42,
    )

    return processor


def create_processor_with_local_best(
    migration: MigrationDriverBase | None = None,
) -> DummyProcessor:
    processor = create_processor(
        migration=migration,
    )

    processor._status.best_particle_data = json.dumps(
        {
            "fitness": 1.0,
            "variables": {"x": 0.0},
        }
    )

    return processor


def create_backend(
    spark: DummySparkSession | None = None,
    processor: DummyProcessor | None = None,
    migration: MigrationDriverBase | None = None,
    n_executors: int = 2,
    code_archive: str | Path | None = None,
) -> SparkDistributed:
    if spark is None:
        spark = DummySparkSession()

    if migration is None:
        migration = create_migration()

    if processor is None:
        processor = create_processor_with_local_best(
            migration=migration,
        )

    backend = SparkDistributed(
        spark,  # type: ignore
        n_executors=n_executors,
        code_archive=code_archive,
    )

    backend.initialize_context(
        algorithm=processor._algorithm,  # type: ignore
        n_iter=1,
        n_particles=1,
        migration=migration,
        processor=processor,
        seed=42,
    )

    return backend


def test_init() -> None:
    spark = DummySparkSession()

    backend = SparkDistributed(  # type: ignore
        spark,  # type: ignore
        n_executors=3,
    )

    assert backend._spark is spark
    assert backend._code_archive is None
    assert backend._communication_port == 6062
    assert backend._n_executors == 3
    assert backend._identifier == "SparkDistributed"
    assert backend._cost_function_wrapper is SparkDistributedCostFunctionWrapper
    assert backend._local_bests == {}


def test_init_converts_code_archive_to_path(tmp_path) -> None:
    archive = tmp_path / "code.zip"

    backend = SparkDistributed(  # type: ignore
        DummySparkSession(),  # type: ignore
        n_executors=2,
        code_archive=str(archive),
    )

    assert backend._code_archive == archive
    assert isinstance(backend._code_archive, Path)


def test_init_accepts_path_code_archive(tmp_path) -> None:
    archive = tmp_path / "code.zip"

    backend = SparkDistributed(  # type: ignore
        DummySparkSession(),  # type: ignore
        n_executors=2,
        code_archive=archive,
    )

    assert backend._code_archive == archive


def test_init_rejects_none_spark() -> None:
    with pytest.raises(
        ValueError,
        match="Spark session cannot be None",
    ):
        SparkDistributed(None)  # type: ignore


def test_get_n_executors() -> None:
    spark = DummySparkSession()

    jsc = Mock()

    executor_memory_status = Mock()
    executor_memory_status.size = 5

    jsc.sc().getExecutorMemoryStatus.return_value = executor_memory_status

    spark.sparkContext._jsc = jsc  # type: ignore

    backend = SparkDistributed(  # type: ignore
        spark,  # type: ignore
        n_executors=None,
    )

    assert backend._n_executors == 5


def test_execute_distributes_code_archive(tmp_path) -> None:
    archive = tmp_path / "code.zip"
    archive.write_bytes(b"zip")

    spark = DummySparkSession()

    backend = create_backend(
        spark=spark,
        code_archive=archive,
    )

    backend.execute()

    assert spark.sparkContext.added_py_files == [
        str(archive),
    ]


def test_execute_without_code_archive_does_not_distribute_file() -> None:
    spark = DummySparkSession()

    backend = create_backend(
        spark=spark,
    )

    backend.execute()

    assert spark.sparkContext.added_py_files == []


def test_execute_rejects_missing_code_archive(tmp_path) -> None:
    archive = tmp_path / "missing.zip"

    backend = create_backend(
        code_archive=archive,
    )

    with pytest.raises(
        FileNotFoundError,
        match="Spark code archive not found",
    ):
        backend.execute()


def test_execute_creates_processors_and_collects_results() -> None:
    spark = DummySparkSession()
    processor = create_processor_with_local_best()

    backend = create_backend(
        spark=spark,
        processor=processor,
        n_executors=3,
    )

    backend.execute()

    assert len(spark.sparkContext.parallelized_data) == 3  # type: ignore

    assert spark.sparkContext.parallelized_slices == 3

    assert set(backend._local_bests) == {
        "island:0",
        "island:1",
        "island:2",
    }

    assert backend.result == {
        "fitness": 1.0,
        "variables": {"x": 0.0},
    }


def test_run_processor_returns_empty_result_without_local_best() -> None:
    processor = create_processor()

    result = _run_processor(processor)

    assert result == (
        processor.identifier,
        {},
    )


def test_run_processor_decodes_local_best() -> None:
    processor = create_processor()

    local_best = {
        "fitness": 1.5,
        "variables": {"x": 0.25},
    }

    processor._status.best_particle_data = json.dumps(
        local_best,
    )

    result = _run_processor(processor)

    assert result == (
        processor.identifier,
        local_best,
    )


def test_run_processor_finalizes_execution_context_on_error() -> None:
    processor = create_processor()

    processor.run = Mock(
        side_effect=RuntimeError(
            "processor failure",
        )
    )

    with pytest.raises(
        RuntimeError,
        match="processor failure",
    ):
        _run_processor(processor)

    assert processor._stop_signal is not None
    assert processor._wait_signal is not None


def test_initialize_context_configures_migration() -> None:
    spark = DummySparkSession()
    migration = create_migration()
    processor = create_processor(
        migration=migration,
    )

    backend = SparkDistributed(  # type: ignore
        spark,  # type: ignore
        n_executors=3,
        communication_port=6063,
    )

    backend.initialize_context(
        algorithm=processor._algorithm,  # type: ignore
        n_iter=1,
        n_particles=1,
        migration=migration,
        processor=processor,
        seed=42,
    )

    migration.initialize_context.assert_called_once()

    kwargs = migration.initialize_context.call_args.kwargs

    assert kwargs["communication_processor_class"] is not None
    assert kwargs["communication_processor_kargs"] == {
        "driver_ip": spark.conf.get(  # type: ignore
            "spark.driver.host",
        ),
        "port": 6063,
    }
