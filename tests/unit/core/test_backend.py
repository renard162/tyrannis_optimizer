"""Concrete context and property contracts of BackendBase."""

from typing import Protocol, final, override
from unittest.mock import patch

from tests._support.numerics import BASE_SEED
from tests._support.processors import (
    ProcessorAlgorithmDouble,
    ProcessorMigrationDriverDouble,
)
from tyrannis.core.algorithm import CostFunctionWrapperBase
from tyrannis.core.backend import BackendBase
from tyrannis.core.results import HistoryConfig, ProcessorResult
from tyrannis.processor.serial import Serial


class _ContextInitializer(Protocol):
    def initialize_context(
        self,
        algorithm: ProcessorAlgorithmDouble,
        n_iter: int,
        n_particles: int,
        migration: ProcessorMigrationDriverDouble,
        processor: Serial | None,
        fitness_failure_strategy: str,
        history_config: HistoryConfig,
        seed: int | None,
    ) -> None: ...


def _as_context_initializer(backend: _ContextInitializer) -> _ContextInitializer:
    return backend


@final
class _Backend(BackendBase):
    def __init__(self) -> None:
        self._identifier = "test-backend"
        self._cost_function_wrapper = CostFunctionWrapperBase

    @override
    def execute(self) -> None:
        pass


def test_initialize_context_stores_configuration_and_exposes_empty_result() -> None:
    backend = _Backend()
    algorithm = ProcessorAlgorithmDouble()
    migration = ProcessorMigrationDriverDouble()
    processor = Serial()
    history_config = HistoryConfig(best=True)

    context_initializer = _as_context_initializer(backend)
    with patch.object(algorithm, "configure") as configure:
        context_initializer.initialize_context(
            algorithm=algorithm,
            n_iter=3,
            n_particles=5,
            migration=migration,
            processor=processor,
            fitness_failure_strategy="invalidate",
            history_config=history_config,
            seed=BASE_SEED,
        )

    context = vars(backend)
    assert context["_algorithm"] is algorithm
    assert context["_n_iter"] == 3
    assert context["_n_particles"] == 5
    assert context["_processor"] is processor
    assert context["_migration"] is migration
    assert context["_fitness_failure_strategy"] == "invalidate"
    assert context["_seed"] == BASE_SEED
    assert context["_history_config"] is history_config
    assert backend.identifier == "test-backend"
    assert isinstance(backend.result, ProcessorResult)
    assert backend.result is context["_result"]
    assert backend.result.result is None
    assert backend.result.history == []
    configure.assert_called_once_with(
        identifier="test-backend|algorithm",
        cost_function_wrapper=CostFunctionWrapperBase,
        seed=BASE_SEED,
    )
