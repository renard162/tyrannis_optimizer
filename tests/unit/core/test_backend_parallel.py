"""Shared state and history contracts of ParallelBackendBase."""

import json
from typing import Protocol, TypedDict, final, override
from unittest.mock import patch

import numpy as np

from tests._support.numerics import BASE_SEED
from tests._support.processors import (
    ProcessorAlgorithmDouble,
    ProcessorMigrationDriverDouble,
    ProcessorParticleDouble,
)
from tyrannis.core.algorithm import CostFunctionWrapperBase
from tyrannis.core.backend_migration import MigrationDriverBase
from tyrannis.core.backend_parallel import ParallelBackendBase
from tyrannis.core.results import HistoryConfig


class _HistoryEvent(TypedDict):
    iteration: int
    event: str
    origin: str
    particle: dict[str, object]


class _ParallelContext(Protocol):
    def initialize_context(
        self,
        algorithm: ProcessorAlgorithmDouble,
        n_iter: int,
        n_particles: int,
        migration: MigrationDriverBase,
        processor: None,
        fitness_failure_strategy: str,
        history_config: HistoryConfig,
        seed: int | None,
    ) -> None: ...


def _as_context(backend: _ParallelContext) -> _ParallelContext:
    return backend


@final
class _ParallelBackend(ParallelBackendBase):
    def __init__(self) -> None:
        self._identifier = "test-parallel"
        self._cost_function_wrapper = CostFunctionWrapperBase

    @override
    def execute(self) -> None:
        pass


def _configured_backend(
    history_config: HistoryConfig | None = None, *, n_particles: int = 2
) -> tuple[_ParallelBackend, ProcessorAlgorithmDouble]:
    backend = _ParallelBackend()
    algorithm = ProcessorAlgorithmDouble()
    with patch.object(algorithm, "configure"):
        _as_context(backend).initialize_context(
            algorithm=algorithm,
            n_iter=3,
            n_particles=n_particles,
            migration=ProcessorMigrationDriverDouble(),
            processor=None,
            fitness_failure_strategy="raise",
            history_config=history_config or HistoryConfig(),
            seed=BASE_SEED,
        )
    return backend, algorithm


def test_initialize_context_keeps_shared_history_and_result() -> None:
    history_config = HistoryConfig(pre_iteration=True)
    backend, algorithm = _configured_backend(history_config)
    algorithm.create_particle("test-parallel|particle:0")
    result = backend.result

    backend.pre_iteration_log(0)

    assert vars(backend)["_history_config"] is history_config
    assert backend.result is result
    assert len(result.history) == 1


def test_init_particles_creates_full_population_with_parallel_identifiers() -> None:
    backend, algorithm = _configured_backend(n_particles=3)

    backend.init_particles()

    assert list(algorithm.population) == [
        "test-parallel|particle:0",
        "test-parallel|particle:1",
        "test-parallel|particle:2",
    ]


def test_init_particles_preserves_existing_population() -> None:
    backend, algorithm = _configured_backend(n_particles=3)
    algorithm.create_particle("existing|particle:7")
    original = algorithm.population["existing|particle:7"]

    backend.init_particles()

    assert algorithm.population == {original.identifier: original}


def test_disabled_logging_leaves_history_unchanged() -> None:
    backend, algorithm = _configured_backend()
    particle = ProcessorParticleDouble("one", 1)
    algorithm.population[particle.identifier] = particle
    vars(algorithm)["_local_best"] = particle
    backend.result.history.append("previous")

    backend.pre_iteration_log(1)
    backend.iteration_log(1)
    backend.new_particle_log(1, [particle])
    backend.error_log(1, [particle])
    backend.best_log(1)

    assert backend.result.history == ["previous"]


def test_population_and_new_particle_logs_record_selected_particles() -> None:
    backend, algorithm = _configured_backend(
        HistoryConfig(pre_iteration=True, iteration=True, new_particle=True)
    )
    first = ProcessorParticleDouble("first", 1)
    second = ProcessorParticleDouble("second", 2)
    algorithm.population.update({first.identifier: first, second.identifier: second})

    backend.pre_iteration_log(2)
    backend.iteration_log(2)
    backend.new_particle_log(2, [second])

    events: list[_HistoryEvent] = [json.loads(row) for row in backend.result.history]
    assert [(event["event"], event["particle"]["identifier"]) for event in events] == [
        (HistoryConfig.get_event("pre_iteration"), first.identifier),
        (HistoryConfig.get_event("pre_iteration"), second.identifier),
        (HistoryConfig.get_event("iteration"), first.identifier),
        (HistoryConfig.get_event("iteration"), second.identifier),
        (HistoryConfig.get_event("new_particle"), second.identifier),
    ]
    assert all(
        event["iteration"] == 2 and event["origin"] == backend.identifier
        for event in events
    )


def test_error_log_records_failed_candidate_only() -> None:
    backend, _ = _configured_backend(HistoryConfig(error=True))
    failed = ProcessorParticleDouble("failed", 1)
    failed.candidate_variables = {"item": 2.0}
    failed.error_fitness = np.float64(9)
    successful = ProcessorParticleDouble("successful", 3)

    backend.error_log(3, [successful, failed])

    assert [json.loads(row) for row in backend.result.history] == [
        {
            "iteration": 3,
            "event": HistoryConfig.get_event("error"),
            "origin": backend.identifier,
            "particle": {
                "identifier": "failed",
                "item": 1,
                "variables": {"item": 2.0},
                "fitness": 9.0,
            },
        }
    ]


def test_best_log_records_local_best_and_optional_iteration_status() -> None:
    backend, algorithm = _configured_backend(HistoryConfig(best=True))
    first = ProcessorParticleDouble("first", 1)
    second = ProcessorParticleDouble("second", 2)
    algorithm.population.update({first.identifier: first, second.identifier: second})
    vars(algorithm)["_local_best"] = first
    vars(algorithm)["_iter_best"] = first.identifier
    vars(algorithm)["_iter_worst"] = second.identifier

    backend.best_log(4)
    vars(backend)["_history_config"] = HistoryConfig(status=True)
    backend.best_log(5)

    events: list[_HistoryEvent] = [json.loads(row) for row in backend.result.history]
    assert [
        (event["iteration"], event["event"], event["particle"]["identifier"])
        for event in events
    ] == [
        (4, HistoryConfig.get_event("local_best"), first.identifier),
        (5, HistoryConfig.get_event("local_best"), first.identifier),
        (5, HistoryConfig.get_event("iter_best"), first.identifier),
        (5, HistoryConfig.get_event("iter_worst"), second.identifier),
    ]


def test_update_result_tracks_local_best_without_replacing_history() -> None:
    backend, algorithm = _configured_backend()
    result = backend.result
    result.history.append("existing event")

    backend.update_result()
    assert result.result is None

    best = ProcessorParticleDouble("best", 1)
    vars(algorithm)["_local_best"] = best
    backend.update_result()

    assert backend.result is result
    assert result.result == best()
    assert result.history == ["existing event"]
