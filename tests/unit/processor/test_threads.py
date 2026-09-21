from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import pytest
from _support.processors import configure_processor_for_dispatch

import tyrannis.processor.threads as threads_module
from tyrannis.processor import ThreadsPool


class _RecordingThreadPool:
    def __init__(self, processes: int | None) -> None:
        self.processes = processes
        self.map_calls: list[tuple[list[Any], int | None]] = []
        self.exited = False

    def __enter__(self) -> _RecordingThreadPool:
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


def test_threads_pool_forwards_worker_count_and_chunksize_and_closes_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded_pool: _RecordingThreadPool | None = None

    def pool_factory(processes: int | None) -> _RecordingThreadPool:
        nonlocal recorded_pool
        recorded_pool = _RecordingThreadPool(processes)
        return recorded_pool

    monkeypatch.setattr(threads_module, "ThreadPool", pool_factory)
    processor = ThreadsPool(n_jobs=2, chunksize=2)
    algorithm, _ = configure_processor_for_dispatch(
        processor,
        n_iterations=1,
        n_particles=3,
    )

    processor.run()

    assert recorded_pool is not None
    assert recorded_pool.processes == 2
    assert [chunksize for _, chunksize in recorded_pool.map_calls] == [2, 2, 2]
    assert [particle.updates for particle in algorithm.population.values()] == [1, 1, 1]
    assert recorded_pool.exited


def test_threads_pool_dispatches_and_returns_every_particle() -> None:
    processor = ThreadsPool(n_jobs=2, chunksize=1)
    algorithm, migration = configure_processor_for_dispatch(
        processor,
        n_iterations=1,
        n_particles=4,
    )

    processor.run()

    assert len(algorithm.population) == 4
    assert {particle.item for particle in algorithm.population.values()} == {0, 1, 2, 3}
    assert all(particle.updates == 1 for particle in algorithm.population.values())
    assert migration.loop_finalized


def test_threads_pool_accepts_default_worker_count_and_empty_dispatch() -> None:
    processor = ThreadsPool()
    algorithm, migration = configure_processor_for_dispatch(
        processor,
        n_iterations=0,
        n_particles=0,
    )

    processor.run()

    assert algorithm.population == {}
    assert migration.loop_finalized


@pytest.mark.parametrize("chunksize", [0, -1], ids=["zero", "negative"])
def test_threads_pool_rejects_non_positive_chunksize(chunksize: int) -> None:
    with pytest.raises(ValueError, match="chunksize"):
        ThreadsPool(chunksize=chunksize)


def test_threads_pool_propagates_worker_exceptions_and_finalizes_loop() -> None:
    failing_id = "MainProcessor|particle:1"
    processor = ThreadsPool(n_jobs=2, chunksize=1)
    _, migration = configure_processor_for_dispatch(
        processor,
        n_iterations=1,
        n_particles=2,
        fail_on_update=failing_id,
    )

    with pytest.raises(ValueError, match=failing_id):
        processor.run()

    assert migration.loop_finalized
