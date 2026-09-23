"""Contracts shared by concrete distributed backends."""

from typing import override
from unittest.mock import patch

import pytest

from tests._support.numerics import BASE_SEED
from tests._support.processors import (
    ProcessorAlgorithmDouble,
    ProcessorMigrationDriverDouble,
    ProcessorParticleDouble,
)
from tyrannis.core.algorithm import AlgorithmBase, CostFunctionWrapperBase
from tyrannis.core.backend_distributed import DistributedBackendBase
from tyrannis.core.backend_migration import MigrationDriverBase
from tyrannis.core.processor import ProcessorBase
from tyrannis.core.results import HistoryConfig, ProcessorResult
from tyrannis.core.signals import LocalEvent
from tyrannis.processor.serial import Serial


class _DistributedBackend(DistributedBackendBase):
    def __init__(self, n_executors: int = 3) -> None:
        self._identifier: str = "test-distributed"
        self._cost_function_wrapper: type[CostFunctionWrapperBase] = (
            CostFunctionWrapperBase
        )
        self._n_executors: int = n_executors

    @override
    def initialize_context(
        self,
        algorithm: AlgorithmBase[ProcessorParticleDouble],
        n_iter: int,
        n_particles: int,
        migration: MigrationDriverBase,
        processor: ProcessorBase[LocalEvent] | None,
        fitness_failure_strategy: str,
        history_config: HistoryConfig,
        seed: int | None,
    ) -> None:
        super().initialize_context(  # pyright: ignore[reportUnknownMemberType]  # The base omits generic type arguments.
            algorithm=algorithm,
            n_iter=n_iter,
            n_particles=n_particles,
            migration=migration,
            processor=processor,
            fitness_failure_strategy=fitness_failure_strategy,
            history_config=history_config,
            seed=seed,
        )

    @override
    def execute(self) -> None:
        pass


def _initialized_backend(
    processor: Serial | None,
) -> tuple[_DistributedBackend, ProcessorMigrationDriverDouble]:
    backend = _DistributedBackend()
    algorithm = ProcessorAlgorithmDouble()
    migration = ProcessorMigrationDriverDouble()
    with patch.object(algorithm, "configure"):
        backend.initialize_context(
            algorithm=algorithm,
            n_iter=2,
            n_particles=4,
            migration=migration,
            processor=processor,
            fitness_failure_strategy="raise",
            history_config=HistoryConfig(),
            seed=BASE_SEED,
        )
    return backend, migration


def test_initialize_context_preserves_processor_migration_and_result() -> None:
    processor = Serial()

    backend, migration = _initialized_backend(processor)

    assert vars(backend)["_processor"] is processor
    assert vars(backend)["_migration"] is migration
    assert isinstance(backend.result, ProcessorResult)


def test_init_processors_rejects_missing_processor() -> None:
    backend, _ = _initialized_backend(None)

    with pytest.raises(RuntimeError, match="Processor is not initialized"):
        backend.init_processors()


def test_init_processors_uses_configured_executor_count() -> None:
    processor = Serial()
    backend, _ = _initialized_backend(processor)

    with patch.object(processor, "create_processors_pool") as create_pool:
        backend.init_processors()

    create_pool.assert_called_once_with(3)


def test_update_result_collects_all_histories_and_selects_valid_minimum() -> None:
    backend = _DistributedBackend()
    migration = ProcessorMigrationDriverDouble()
    first = ProcessorResult(result={"fitness": 5.0}, history=["island-a"])
    absent = ProcessorResult(result=None, history=["island-b"])
    invalid = ProcessorResult(result={"fitness": "invalid"}, history=["island-c"])
    best = ProcessorResult(result={"fitness": 2.0}, history=["island-d"])
    vars(backend)["_result"] = None
    vars(backend)["_migration"] = migration
    vars(backend)["_local_bests"] = {
        "a": first,
        "b": absent,
        "c": invalid,
        "d": best,
    }

    with patch.object(
        migration, "consume_migration_history", return_value=["migration"]
    ):
        backend.update_result()

    assert backend.result.result is best.result
    assert len(backend.result.history) == 5
    assert set(backend.result.history) == {
        "island-a",
        "island-b",
        "island-c",
        "island-d",
        "migration",
    }
    assert all(result.history == [] for result in (first, absent, invalid, best))


def test_update_result_preserves_then_improves_best_across_calls() -> None:
    backend = _DistributedBackend()
    migration = ProcessorMigrationDriverDouble()
    vars(backend)["_result"] = ProcessorResult(result={"fitness": "invalid"})
    vars(backend)["_migration"] = migration
    original = ProcessorResult(result={"fitness": 1.5}, history=["first"])
    worse = ProcessorResult(result={"fitness": 4.0}, history=["second"])
    improved = ProcessorResult(result={"fitness": 0.5}, history=["third"])

    with patch.object(
        migration,
        "consume_migration_history",
        side_effect=[["migration-first"], ["migration-second"], []],
    ):
        vars(backend)["_local_bests"] = {"island": original}
        backend.update_result()
        assert backend.result.result is original.result

        vars(backend)["_local_bests"] = {"island": worse}
        backend.update_result()
        assert backend.result.result is original.result

        vars(backend)["_local_bests"] = {"island": improved}
        backend.update_result()

    assert backend.result.result is improved.result
    assert len(backend.result.history) == 5
    assert set(backend.result.history) == {
        "first",
        "second",
        "third",
        "migration-first",
        "migration-second",
    }
    assert all(result.history == [] for result in (original, worse, improved))
