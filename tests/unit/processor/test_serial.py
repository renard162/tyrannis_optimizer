from threading import get_ident

import pytest

from _support.processors import configure_processor_for_dispatch
from tyrannis.processor.serial import Serial


def test_serial_executes_every_particle_in_order_on_the_calling_thread() -> None:
    processor = Serial()
    algorithm, migration = configure_processor_for_dispatch(
        processor,
        n_iterations=1,
        n_particles=3,
    )
    calling_thread = get_ident()

    processor.run()

    expected_ids = [f"MainProcessor|particle:{index}" for index in range(3)]
    assert algorithm.initialized_ids == expected_ids
    assert algorithm.updated_ids == expected_ids
    assert list(algorithm.population) == expected_ids
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
