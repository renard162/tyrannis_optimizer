from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, final, override

import numpy as np
import pytest

from tests._support.numerics import BASE_SEED
from tests._support.objectives import encoded_sphere
from tests._support.processors import (
    ProcessorAlgorithmDouble,
    ProcessorMigrationDriverDouble,
)
from tyrannis.backend.distributed.communication.no_communication import (
    NoCommunicationDriver,
    NoCommunicationProcessor,
)
from tyrannis.backend.local import Local, LocalCostFunctionWrapper
from tyrannis.core.algorithm import CostFunctionWrapperBase
from tyrannis.core.processor import ProcessorBase
from tyrannis.core.results import HistoryConfig, ProcessorResult
from tyrannis.core.signals import EventProtocol


class _Migration(ProcessorMigrationDriverDouble):
    def __init__(self) -> None:
        super().__init__()
        self._migration_processor_init_kargs: dict[str, object] = {}


class _LocalContext(Protocol):
    def initialize_context(
        self,
        algorithm: ProcessorAlgorithmDouble,
        n_iter: int,
        n_particles: int,
        migration: _Migration,
        processor: ProcessorBase[EventProtocol] | None,
        fitness_failure_strategy: str,
        history_config: HistoryConfig,
        seed: int | None,
    ) -> None: ...


@final
class _Processor(ProcessorBase[EventProtocol]):
    def __init__(
        self,
        events: list[str],
        result: ProcessorResult | None = None,
        *,
        executor: _Processor | None = None,
        failure: Exception | None = None,
    ) -> None:
        super().__init__()
        self.events = events
        self._identifier = "island:0"
        self._cost_function_wrapper = CostFunctionWrapperBase
        self._result = result if result is not None else ProcessorResult()
        self._processors_pool: dict[str, ProcessorBase[EventProtocol]] = {}
        self.executor = executor
        self.failure = failure
        self.pool_sizes: list[int] = []
        self.updated: list[ProcessorBase[EventProtocol]] = []

    @override
    def create_processors_pool(self, n_islands: int) -> None:
        self.pool_sizes.append(n_islands)
        self.events.append("create_pool")
        if self.executor is None:
            raise AssertionError("The test executor was not supplied")
        self._processors_pool[self.executor.identifier] = self.executor

    @override
    def update_processors_pool(
        self, new_processors: Iterable[ProcessorBase[EventProtocol]]
    ) -> None:
        returned = list(new_processors)
        self.updated.extend(returned)
        self.events.append("update_pool")
        self._processors_pool.update(
            {processor.identifier: processor for processor in returned}
        )

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


def _configure(
    backend: _LocalContext,
    algorithm: ProcessorAlgorithmDouble,
    migration: _Migration,
    history_config: HistoryConfig,
    processor: ProcessorBase[EventProtocol] | None = None,
) -> None:
    backend.initialize_context(
        algorithm=algorithm,
        n_iter=3,
        n_particles=5,
        migration=migration,
        processor=processor,
        fitness_failure_strategy="raise",
        history_config=history_config,
        seed=BASE_SEED,
    )


def _initialize(
    backend: _LocalContext, processor: ProcessorBase[EventProtocol] | None = None
) -> None:
    algorithm = ProcessorAlgorithmDouble()
    algorithm.initialize_context(
        fitness_function=lambda variables: np.float64(encoded_sphere(variables)),
        boundaries={},
        n_iter=3,
        n_particles=5,
    )
    _configure(backend, algorithm, _Migration(), HistoryConfig(), processor)


def test_constructor_sets_local_identifier() -> None:
    assert Local().identifier == "Local"


def test_initialize_context_configures_local_collaborators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = Local()
    algorithm = ProcessorAlgorithmDouble()
    migration = _Migration()
    history_config = HistoryConfig(migration=True)
    algorithm_configuration: dict[str, object] = {}
    migration_configuration: dict[str, object] = {}

    def record_algorithm_configuration(**kwargs: object) -> None:
        algorithm_configuration.update(kwargs)

    def record_migration_configuration(**kwargs: object) -> None:
        migration_configuration.update(kwargs)

    monkeypatch.setattr(algorithm, "configure", record_algorithm_configuration)
    monkeypatch.setattr(migration, "initialize_context", record_migration_configuration)

    _configure(backend, algorithm, migration, history_config)

    assert algorithm_configuration == {
        "identifier": "Local|algorithm",
        "cost_function_wrapper": LocalCostFunctionWrapper,
        "seed": BASE_SEED,
    }
    assert isinstance(backend.result, ProcessorResult)
    assert backend.result.result is None
    assert backend.result.history == []

    assert migration_configuration["history_config"] is history_config
    assert migration_configuration["seed"] == BASE_SEED
    assert (
        migration_configuration["communication_processor_class"]
        is NoCommunicationProcessor
    )
    assert migration_configuration["communication_processor_kargs"] == {}
    communication = migration_configuration["communication_driver"]
    assert isinstance(communication, NoCommunicationDriver)
    assert set(communication.incoming_queues) == {"island:0"}
    assert set(communication.outgoing_queues) == {"island:0"}


def test_execute_requires_configured_processor() -> None:
    backend = Local()
    _initialize(backend)

    with pytest.raises(RuntimeError, match="Processor cannot be None"):
        backend.execute()


def test_execute_creates_one_local_executor_and_publishes_its_result() -> None:
    events: list[str] = []
    executor_result = ProcessorResult(result={"fitness": 1.0}, history=["event"])
    executor = _Processor(events, executor_result)
    processor = _Processor(events, executor=executor)
    backend = Local()
    _initialize(backend, processor)
    initial_result = backend.result

    backend.execute()

    assert processor.pool_sizes == [1]
    assert processor.updated == [executor]
    assert processor.processors_pool == {"island:0": executor}
    assert events == ["create_pool", "initialize", "run", "update_pool", "finalize"]
    assert backend.result is executor_result
    assert backend.result is not initial_result


def test_execute_reuses_an_existing_local_executor() -> None:
    events: list[str] = []
    executor_result = ProcessorResult(result={"fitness": 2.0})
    executor = _Processor(events, executor_result)
    processor = _Processor(events)
    processor.processors_pool[executor.identifier] = executor
    backend = Local()
    _initialize(backend, processor)

    backend.execute()

    assert processor.pool_sizes == []
    assert processor.updated == [executor]
    assert events == ["initialize", "run", "update_pool", "finalize"]
    assert backend.result is executor_result


def test_execute_finalizes_context_and_preserves_run_error() -> None:
    events: list[str] = []
    failure = ValueError("processor run failed")
    executor = _Processor(events, failure=failure)
    processor = _Processor(events, executor=executor)
    backend = Local()
    _initialize(backend, processor)
    initial_result = backend.result

    with pytest.raises(ValueError) as raised:
        backend.execute()

    assert raised.value is failure
    assert events == ["create_pool", "initialize", "run", "finalize"]
    assert processor.updated == []
    assert backend.result is initial_result
