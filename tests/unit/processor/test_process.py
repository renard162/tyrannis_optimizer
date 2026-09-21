from __future__ import annotations

import pickle
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import pytest
from _support.processors import configure_processor_for_dispatch

import tyrannis.processor.process as process_module
from tyrannis.processor import ProcessPool
from tyrannis.processor.process import ProcessPoolCostFunctionWrapper


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

    def __enter__(self) -> _RecordingManager:
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

    def __enter__(self) -> _RecordingProcessPool:
        return self

    def __exit__(self, *args: object) -> None:
        del args
        self.exited = True

    def map(
        self,
        function: Callable[[Any], Any],
        inputs: Iterable[Any],
        chunksize: int | None = None,
    ) -> list[Any]:
        items = list(inputs)
        self.map_calls.append((items, chunksize))
        return [function(item) for item in items]


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
    wrapper = ProcessPoolCostFunctionWrapper(lambda value: np.float64(value + offset))

    restored = pickle.loads(pickle.dumps(wrapper))

    assert restored(3.0) == np.float64(5.0)


def test_process_pool_rejects_unknown_multiprocessing_context() -> None:
    with pytest.raises(ValueError, match="Invalid multiprocessing context"):
        ProcessPool(multiprocessing_context="not-a-context")


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"chunksize": 0}, id="chunksize"),
        pytest.param({"maxtasksperchild": 0}, id="maxtasksperchild"),
    ],
)
def test_process_pool_rejects_non_positive_pool_options(
    kwargs: dict[str, int],
) -> None:
    with pytest.raises(ValueError):
        ProcessPool(**kwargs)
