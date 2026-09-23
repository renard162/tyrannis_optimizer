"""Contracts implemented by ProcessorBase and its particle evaluation helper."""

from __future__ import annotations

import json
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest

from tests._support.numerics import BASE_SEED
from tests._support.objectives import encoded_sphere
from tests._support.processors import (
    ProcessorAlgorithmDouble,
    ProcessorMigrationDriverDouble,
    ProcessorParticleDouble,
    configure_processor_for_dispatch,
)
from tyrannis.core.algorithm import FITNESS_UNDEFINED, ParticleBase
from tyrannis.core.processor import IntSequence, evaluate_particle
from tyrannis.core.results import HistoryConfig
from tyrannis.core.signals import LocalEvent
from tyrannis.processor.serial import Serial


def test_int_sequences_advance_independently() -> None:
    first, second = IntSequence(), IntSequence()

    assert [first.spawn(), first.spawn(), second.spawn(), first.spawn()] == [0, 1, 0, 2]


def test_initialize_context_sets_a_replicable_template() -> None:
    processor = Serial()
    algorithm, _ = configure_processor_for_dispatch(
        processor, n_iterations=2, n_particles=3
    )

    assert processor.identifier == "MainProcessor"
    assert processor.processors_pool == {}
    assert processor.population == {}
    assert processor.result.result is None
    assert processor.result.history == []
    assert processor._algorithm is algorithm


@pytest.mark.parametrize(
    "strategy, valid",
    [("raise", True), ("invalidate", True), ("ignore", False), ("", False)],
    ids=["raise", "invalidate", "unknown", "empty"],
)
def test_initialize_context_accepts_only_fitness_failure_strategies(
    strategy: str, valid: bool
) -> None:
    processor = Serial()
    algorithm = ProcessorAlgorithmDouble()
    driver = ProcessorMigrationDriverDouble()

    if valid:
        processor.initialize_context(
            algorithm, 1, 2, driver, HistoryConfig(), strategy, BASE_SEED
        )
        assert processor._fitness_failure_strategy == strategy
    else:
        with pytest.raises(ValueError, match="fitness failure strategy"):
            processor.initialize_context(
                algorithm, 1, 2, driver, HistoryConfig(), strategy, BASE_SEED
            )


def test_pool_replicates_independent_islands_and_continues_identifiers() -> None:
    processor = Serial()
    algorithm = ProcessorAlgorithmDouble()
    algorithm.initialize_context(
        fitness_function=lambda variables: np.float64(encoded_sphere(variables)),
        boundaries={},
        n_iter=1,
        n_particles=1,
    )
    driver = ProcessorMigrationDriverDouble()
    processor.initialize_context(
        algorithm, 1, 1, driver, HistoryConfig(), "raise", BASE_SEED
    )
    processor._migration_signal = LocalEvent()
    processor.result.history.append("template only")

    processor.create_processors_pool(2)
    first, second = (processor.processors_pool[f"island:{index}"] for index in range(2))

    assert list(processor.processors_pool) == ["island:0", "island:1"]
    assert first is not processor and second is not processor and first is not second
    assert first._algorithm is not algorithm and second._algorithm is not algorithm
    assert first._algorithm is not second._algorithm
    assert [first._algorithm.identifier, second._algorithm.identifier] == [
        "island:0|algorithm",
        "island:1|algorithm",
    ]
    assert (
        first._algorithm._rng.bit_generator.state
        != second._algorithm._rng.bit_generator.state
    )
    assert first._migration_processor is not second._migration_processor
    assert first._migration_processor is not None
    assert second._migration_processor is not None
    assert driver.created_identifiers == ["island:0", "island:1"]
    assert first._migration_signal is None
    assert first._migration_driver is None
    assert first.processors_pool == {}
    first.result.history.append("replica only")
    assert processor.result.history == ["template only"]
    assert second.result.history == ["template only"]

    processor.create_processors_pool(1)
    assert list(processor.processors_pool) == ["island:0", "island:1", "island:2"]
    assert driver.created_identifiers == ["island:0", "island:1", "island:2"]

    returned = deepcopy(first)
    processor.update_processors_pool([returned])
    assert processor.processors_pool["island:0"] is returned
    assert processor.processors_pool["island:1"] is second


def test_init_particles_creates_only_missing_population() -> None:
    processor = Serial()
    algorithm, _ = configure_processor_for_dispatch(
        processor, n_iterations=0, n_particles=3
    )

    processor.init_particles()
    expected = [f"MainProcessor|particle:{index}" for index in range(3)]
    assert list(algorithm.population) == expected
    originals = list(algorithm.population.values())

    processor.init_particles()
    assert list(algorithm.population.values()) == originals


def test_migration_start_stop_and_missing_processor_contract() -> None:
    processor = Serial()
    _, migration = configure_processor_for_dispatch(
        processor, n_iterations=0, n_particles=0
    )

    processor.start_migration()
    processor.stop_migration()
    assert migration.started and migration.stopped

    processor._migration_processor = None
    processor.stop_migration()
    with pytest.raises(RuntimeError, match="Migration processor"):
        processor.start_migration()
    with pytest.raises(RuntimeError, match="Migration processor"):
        processor.migration_control(0)


def test_migration_control_delegates_population_best_callbacks_and_update() -> None:
    processor = Serial()
    algorithm, migration = configure_processor_for_dispatch(
        processor, n_iterations=1, n_particles=1
    )
    particle_id = "MainProcessor|particle:0"
    algorithm.create_particle(particle_id)
    algorithm._iter_best = particle_id
    processor.update_status()
    with (
        patch.object(migration, "migration_control") as migration_call,
        patch.object(migration, "synchronization_control") as sync_call,
        patch.object(algorithm, "update_n_particles") as update_call,
    ):
        processor.migration_control(1)

    migration_call.assert_called_once()
    sync_call.assert_called_once()
    update_call.assert_called_once_with()
    first = migration_call.call_args.kwargs
    second = sync_call.call_args.kwargs
    assert first["actual_iter"] == second["actual_iter"] == 1
    assert first["population"] is processor.population
    assert first["iter_best"] == particle_id
    assert first["insert_arrival_particle"] == second["insert_arrival_particle"]
    assert first["departure_particle"] == second["departure_particle"]
    assert first["insert_arrival_particle"].__self__ is processor
    assert first["departure_particle"].__self__ is processor


def test_arrival_requires_identifier_and_does_not_duplicate_particle() -> None:
    processor = Serial()
    algorithm, _ = configure_processor_for_dispatch(
        processor, n_iterations=0, n_particles=0
    )

    with pytest.raises(ValueError, match="identifier"):
        processor._insert_arrival_particle({"item": 1})

    processor._insert_arrival_particle({"identifier": "arrival:1"})
    original = algorithm.population["arrival:1"]
    processor._insert_arrival_particle({"identifier": "arrival:1"})
    assert algorithm.population == {"arrival:1": original}


def test_departure_serializes_and_removes_particle_from_both_populations() -> None:
    processor = Serial()
    algorithm, _ = configure_processor_for_dispatch(
        processor, n_iterations=0, n_particles=0
    )
    algorithm.create_particle("arrival:1")
    processor.update_status()
    original = algorithm.population["arrival:1"]

    assert processor._departure_particle("arrival:1") == original()
    assert algorithm.population == {}
    assert processor.population == {}
    assert processor._departure_particle("arrival:1") is None


def test_update_status_uses_local_best_and_orders_population_by_fitness() -> None:
    processor = Serial()
    algorithm, _ = configure_processor_for_dispatch(
        processor, n_iterations=0, n_particles=0
    )
    processor.update_status()
    assert processor.result.result is None

    high = ProcessorParticleDouble("island:0|particle:0", 0)
    low = ProcessorParticleDouble("island:0|particle:1", 1)
    high._fitness = np.float64(3)
    low._fitness = np.float64(1)
    algorithm.population.update({high.identifier: high, low.identifier: low})
    algorithm._local_best = low

    processor.update_status()

    assert processor.result.result == low()
    assert list(processor.population.items()) == [
        (low.identifier, low.fitness),
        (high.identifier, high.fitness),
    ]


def test_disabled_history_flags_leave_history_empty() -> None:
    processor = Serial()
    algorithm, _ = configure_processor_for_dispatch(
        processor, n_iterations=0, n_particles=0
    )
    particle = ProcessorParticleDouble("island:0|particle:0", 0)
    algorithm.population[particle.identifier] = particle
    algorithm._local_best = particle

    processor.pre_iteration_log(1)
    processor.iteration_log(1)
    processor.new_particle_log(1, [particle])
    processor.error_log(1, [particle])
    processor.best_log(1)

    assert processor.result.history == []


def test_history_logs_population_and_only_supplied_new_particles() -> None:
    processor = Serial()
    algorithm, _ = configure_processor_for_dispatch(
        processor, n_iterations=0, n_particles=0
    )
    processor._history_config = HistoryConfig(
        pre_iteration=True, iteration=True, new_particle=True
    )
    first = ProcessorParticleDouble("island:0|particle:0", 0)
    second = ProcessorParticleDouble("island:0|particle:1", 1)
    algorithm.population.update({first.identifier: first, second.identifier: second})

    processor.pre_iteration_log(2)
    processor.iteration_log(2)
    processor.new_particle_log(2, [second])

    events = [json.loads(record) for record in processor.result.history]
    assert [(event["event"], event["particle"]["identifier"]) for event in events] == [
        (HistoryConfig.get_event("pre_iteration"), first.identifier),
        (HistoryConfig.get_event("pre_iteration"), second.identifier),
        (HistoryConfig.get_event("iteration"), first.identifier),
        (HistoryConfig.get_event("iteration"), second.identifier),
        (HistoryConfig.get_event("new_particle"), second.identifier),
    ]
    assert all(
        event["iteration"] == 2 and event["origin"] == processor.identifier
        for event in events
    )


def test_error_history_uses_failed_candidate_and_skips_other_particles() -> None:
    processor = Serial()
    configure_processor_for_dispatch(processor, n_iterations=0, n_particles=0)
    processor._history_config = HistoryConfig(error=True)
    failed = ParticleBase("failed", {"item": 1.0}, np.float64(1))
    failed.candidate_variables = {"item": 2.0}
    failed.error_fitness = np.float64(9)
    successful = ParticleBase("successful", {"item": 3.0}, np.float64(3))

    processor.error_log(3, [successful, failed])

    assert [json.loads(record) for record in processor.result.history] == [
        {
            "iteration": 3,
            "event": HistoryConfig.get_event("error"),
            "origin": processor.identifier,
            "particle": {
                "identifier": "failed",
                "variables": {"item": 2.0},
                "fitness": 9.0,
            },
        }
    ]


def test_best_history_logs_local_best_and_status_extrema() -> None:
    processor = Serial()
    algorithm, _ = configure_processor_for_dispatch(
        processor, n_iterations=0, n_particles=0
    )
    first = ProcessorParticleDouble("island:0|particle:0", 0)
    second = ProcessorParticleDouble("island:0|particle:1", 1)
    algorithm.population.update({first.identifier: first, second.identifier: second})
    algorithm._local_best = first
    algorithm._iter_best = first.identifier
    algorithm._iter_worst = second.identifier

    processor._history_config = HistoryConfig(best=True)
    processor.best_log(4)
    processor._history_config = HistoryConfig(status=True)
    processor.best_log(5)

    events = [json.loads(record) for record in processor.result.history]
    assert [
        (event["iteration"], event["event"], event["particle"]["identifier"])
        for event in events
    ] == [
        (4, HistoryConfig.get_event("local_best"), first.identifier),
        (5, HistoryConfig.get_event("local_best"), first.identifier),
        (5, HistoryConfig.get_event("iter_best"), first.identifier),
        (5, HistoryConfig.get_event("iter_worst"), second.identifier),
    ]


@pytest.mark.parametrize(
    "initialize, second, expected",
    [
        (True, False, "initialized_ids"),
        (False, False, "updated_ids"),
        (False, True, "second_updated_ids"),
    ],
    ids=["initialization", "regular", "second"],
)
def test_evaluate_particle_dispatches_the_requested_phase(
    initialize: bool, second: bool, expected: str
) -> None:
    algorithm = ProcessorAlgorithmDouble()
    particle_id = "island:0|particle:0"
    algorithm.create_particle(particle_id)

    result = evaluate_particle(
        particle_id,
        algorithm,
        "raise",
        initialize_particle=initialize,
        second_update=second,
    )

    assert result.identifier == particle_id
    assert getattr(algorithm, expected) == [particle_id]
    assert (
        sum(
            map(
                len,
                (
                    algorithm.initialized_ids,
                    algorithm.updated_ids,
                    algorithm.second_updated_ids,
                ),
            )
        )
        == 1
    )


@pytest.mark.parametrize(
    "strategy", ["raise", "invalidate"], ids=["raise", "invalidate"]
)
def test_evaluate_particle_marks_nan_and_applies_failure_strategy(
    strategy: str,
) -> None:
    algorithm = ProcessorAlgorithmDouble()
    particle_id = "island:0|particle:0"
    algorithm.create_particle(particle_id)
    original = algorithm.population[particle_id]
    original.candidate_fitness = np.float64(np.nan)

    if strategy == "raise":
        with pytest.raises(ValueError, match="NaN"):
            evaluate_particle(particle_id, algorithm, strategy)
    else:
        result = evaluate_particle(particle_id, algorithm, strategy)
        assert result is original
        assert result.candidate_fitness == FITNESS_UNDEFINED

    assert original.error_fitness is not None
    assert np.isnan(original.error_fitness)


@pytest.mark.parametrize(
    "strategy, previous_error",
    [("raise", None), ("invalidate", None), ("invalidate", np.float64(7))],
    ids=["raise", "invalidate-new-error", "invalidate-existing-error"],
)
def test_evaluate_particle_handles_update_exception_and_preserves_error(
    strategy: str, previous_error: np.float64 | None
) -> None:
    particle_id = "island:0|particle:0"
    algorithm = ProcessorAlgorithmDouble(fail_on_update=particle_id)
    algorithm.create_particle(particle_id)
    original = algorithm.population[particle_id]
    if previous_error is not None:
        original.error_fitness = previous_error

    if strategy == "raise":
        with pytest.raises(ValueError, match="failed item"):
            evaluate_particle(particle_id, algorithm, strategy)
        assert original.error_fitness is None
    else:
        result = evaluate_particle(particle_id, algorithm, strategy)
        assert result is original
        assert result.candidate_fitness == FITNESS_UNDEFINED
        assert result.error_fitness == (
            previous_error if previous_error is not None else FITNESS_UNDEFINED
        )
