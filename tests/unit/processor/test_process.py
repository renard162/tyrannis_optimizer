from __future__ import annotations

import pickle
from collections.abc import Callable, Iterable
from typing import Any, Self

import numpy as np
import pytest
from _support.processors import (
    assert_second_phase_dispatch,
    configure_processor_for_dispatch,
)

import tyrannis.processor.process as process_module
from tyrannis.processor import ProcessPool
from tyrannis.processor.process import PoolSignal, ProcessPoolCostFunctionWrapper


class _EventDouble:
    def __init__(self) -> None:
        self.value = False

    def set(self) -> None:
        self.value = True

    def clear(self) -> None:
        self.value = False

    def is_set(self) -> bool:
        return self.value


class _RecordingManager:
    def __init__(self) -> None:
        self.exited = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        del args
        self.exited = True

    def Event(self) -> _EventDouble:
        return _EventDouble()


class _RecordingProcessPool:
    def __init__(self, processes: int | None, maxtasksperchild: int | None) -> None:
        self.processes = processes
        self.maxtasksperchild = maxtasksperchild
        self.map_calls: list[tuple[list[Any], int | None]] = []
        self.exited = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        del args
        self.exited = True

    def map(
        self,
        func: Callable[[Any], Any],
        iterable: Iterable[Any],
        chunksize: int | None = None,
    ) -> list[Any]:
        items = list(iterable)
        self.map_calls.append((items, chunksize))
        return [func(item) for item in items]


class _RecordingContext:
    def __init__(self) -> None:
        self.manager = _RecordingManager()
        self.pool: _RecordingProcessPool | None = None

    def Manager(self) -> _RecordingManager:
        return self.manager

    def Pool(
        self,
        processes: int | None,
        maxtasksperchild: int | None,
    ) -> _RecordingProcessPool:
        self.pool = _RecordingProcessPool(processes, maxtasksperchild)
        return self.pool


def test_pool_signal_mirrors_state_to_manager_and_releases_it() -> None:
    signal = PoolSignal()
    manager_signal = _EventDouble()
    assert not signal.is_set()

    signal.set_manager_signal(manager_signal)
    assert signal.manager_signal is manager_signal
    assert not manager_signal.is_set()

    signal.set()
    assert signal.is_set()
    assert manager_signal.is_set()

    signal.clear()
    assert not signal.is_set()
    assert not manager_signal.is_set()

    signal.set()
    signal.clear_manager_signal()
    with pytest.raises(RuntimeError, match="Manager signal"):
        _ = signal.manager_signal

    signal.set_manager_signal(manager_signal)
    assert manager_signal.is_set()


def test_process_pool_forwards_context_and_pool_configuration_and_closes_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _RecordingContext()
    requested_contexts: list[str] = []

    def context_factory(name: str) -> _RecordingContext:
        requested_contexts.append(name)
        return context

    monkeypatch.setattr(process_module, "get_context", context_factory)
    processor = ProcessPool(
        n_jobs=3,
        multiprocessing_context="spawn",
        maxtasksperchild=4,
        chunksize=2,
    )
    algorithm, _ = configure_processor_for_dispatch(
        processor,
        n_iterations=1,
        n_particles=2,
    )

    processor.run()

    assert requested_contexts == ["spawn"]
    assert context.pool is not None
    assert context.pool.processes == 3
    assert context.pool.maxtasksperchild == 4
    assert [chunksize for _, chunksize in context.pool.map_calls] == [2, 2, 2]
    assert [particle.updates for particle in algorithm.population.values()] == [1, 1]
    assert context.pool.exited
    assert context.manager.exited


@pytest.mark.multiprocess
def test_process_pool_dispatches_and_returns_every_particle_with_spawn() -> None:
    processor = ProcessPool(
        n_jobs=2,
        multiprocessing_context="spawn",
        chunksize=1,
    )
    algorithm, migration = configure_processor_for_dispatch(
        processor,
        n_iterations=1,
        n_particles=3,
    )

    processor.run()

    assert len(algorithm.population) == 3
    assert {particle.item for particle in algorithm.population.values()} == {0, 1, 2}
    assert all(particle.updates == 1 for particle in algorithm.population.values())
    assert migration.loop_finalized


@pytest.mark.multiprocess
def test_process_pool_propagates_worker_exceptions_and_finalizes_loop() -> None:
    failing_id = "MainProcessor|particle:1"
    processor = ProcessPool(
        n_jobs=2,
        multiprocessing_context="spawn",
        chunksize=1,
    )
    _, migration = configure_processor_for_dispatch(
        processor,
        n_iterations=1,
        n_particles=2,
        fail_on_update=failing_id,
    )

    with pytest.raises(ValueError, match=failing_id):
        processor.run()

    assert migration.loop_finalized


def test_process_pool_wrapper_round_trips_a_cloudpickle_callable() -> None:
    offset = 2.0
    wrapper = ProcessPoolCostFunctionWrapper(lambda value: float(value + offset))

    restored = pickle.loads(pickle.dumps(wrapper))

    assert restored(3.0) == np.float64(5.0)


def test_process_pool_rejects_unknown_multiprocessing_context() -> None:
    with pytest.raises(ValueError, match="Invalid multiprocessing context"):
        ProcessPool(multiprocessing_context="not-a-context")


@pytest.mark.parametrize(
    "option",
    [
        pytest.param("chunksize", id="chunksize"),
        pytest.param("maxtasksperchild", id="maxtasksperchild"),
    ],
)
def test_process_pool_rejects_non_positive_pool_options(
    option: str,
) -> None:
    with pytest.raises(ValueError):
        if option == "chunksize":
            ProcessPool(chunksize=0)
        else:
            ProcessPool(maxtasksperchild=0)


@pytest.mark.multiprocess
@pytest.mark.parametrize(
    "checked_indexes", [[1], []], ids=["selected-particle", "empty-selection"]
)
def test_process_pool_runs_only_selected_second_updates(
    checked_indexes: list[int],
) -> None:
    processor = ProcessPool(n_jobs=2, multiprocessing_context="spawn", chunksize=1)
    algorithm, _ = configure_processor_for_dispatch(
        processor, n_iterations=1, n_particles=3
    )
    ids = [f"MainProcessor|particle:{index}" for index in range(3)]
    algorithm.double_particle_check = True
    algorithm.requested_double_check_ids = [ids[index] for index in checked_indexes]

    processor.run()

    assert_second_phase_dispatch(algorithm, ids)


def test_process_pool_execution_context_starts_and_stops_migration() -> None:
    processor = ProcessPool(n_jobs=1)
    _, migration = configure_processor_for_dispatch(
        processor, n_iterations=0, n_particles=0
    )

    processor.initialize_execution_context()
    processor.finalize_execution_context()

    assert migration.started
    assert migration.stopped
