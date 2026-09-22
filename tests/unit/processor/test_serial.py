from threading import get_ident

import pytest
from _support.processors import (
    assert_second_phase_dispatch,
    configure_processor_for_dispatch,
)

from tyrannis.processor.serial import Serial


def test_serial_executes_every_particle_on_the_calling_thread() -> None:
    processor = Serial()
    algorithm, migration = configure_processor_for_dispatch(
        processor,
        n_iterations=1,
        n_particles=3,
    )
    calling_thread = get_ident()

    processor.run()

    expected_ids = [f"MainProcessor|particle:{index}" for index in range(3)]
    assert set(algorithm.initialized_ids) == set(expected_ids)
    assert set(algorithm.updated_ids) == set(expected_ids)
    assert len(algorithm.initialized_ids) == len(expected_ids)
    assert len(algorithm.updated_ids) == len(expected_ids)
    assert set(algorithm.population) == set(expected_ids)
    assert [particle.updates for particle in algorithm.population.values()] == [1, 1, 1]
    assert algorithm.worker_threads == [calling_thread] * 6
    assert migration.loop_initialized
    assert migration.loop_finalized


def test_serial_handles_an_empty_population() -> None:
    processor = Serial()
    algorithm, migration = configure_processor_for_dispatch(
        processor,
        n_iterations=0,
        n_particles=0,
    )

    processor.run()

    assert algorithm.population == {}
    assert algorithm.iterations == [("pre", 0), ("post", 0)]
    assert migration.loop_finalized


def test_serial_propagates_particle_exceptions_and_finalizes_loop() -> None:
    failing_id = "MainProcessor|particle:1"
    processor = Serial()
    _, migration = configure_processor_for_dispatch(
        processor,
        n_iterations=1,
        n_particles=2,
        fail_on_update=failing_id,
    )

    with pytest.raises(ValueError, match=failing_id):
        processor.run()

    assert migration.loop_finalized


def test_serial_execution_context_starts_and_stops_migration() -> None:
    processor = Serial()
    _, migration = configure_processor_for_dispatch(
        processor,
        n_iterations=0,
        n_particles=0,
    )

    processor.initialize_execution_context()
    processor.finalize_execution_context()

    assert migration.started
    assert migration.stopped


@pytest.mark.parametrize(
    "checked_indexes",
    [[1], []],
    ids=["selected-particle", "empty-selection"],
)
def test_serial_runs_inter_iteration_and_only_selected_second_updates(
    checked_indexes: list[int],
) -> None:
    processor = Serial()
    algorithm, _ = configure_processor_for_dispatch(
        processor, n_iterations=1, n_particles=3
    )
    ids = [f"MainProcessor|particle:{index}" for index in range(3)]
    algorithm.double_particle_check = True
    algorithm.requested_double_check_ids = [ids[index] for index in checked_indexes]

    processor.run()

    assert algorithm.second_updated_ids == algorithm.requested_double_check_ids
    assert_second_phase_dispatch(algorithm, ids)
