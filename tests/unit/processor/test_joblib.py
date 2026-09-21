from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import pytest
from _support.processors import configure_processor_for_dispatch

import tyrannis.processor.joblib as joblib_module
from tyrannis.processor import Joblib


class _RecordingParallel:
    def __init__(
        self,
        n_jobs: int,
        backend: str,
        batch_size: int | str,
        pre_dispatch: int | str,
        return_as: str,
    ) -> None:
        self.n_jobs = n_jobs
        self.backend = backend
        self.batch_size = batch_size
        self.pre_dispatch = pre_dispatch
        self.return_as = return_as
        self.calls: list[
            list[
                tuple[
                    Callable[..., Any],
                    tuple[Any, ...],
                    dict[str, Any],
                ]
            ]
        ] = []
        self.exited = False

    def __enter__(self) -> _RecordingParallel:
        return self

    def __exit__(self, *args: object) -> None:
        del args
        self.exited = True

    def __call__(
        self,
        tasks: Iterable[
            tuple[
                Callable[..., Any],
                tuple[Any, ...],
                dict[str, Any],
            ]
        ],
    ) -> list[Any]:
        task_list = list(tasks)
        self.calls.append(task_list)

        return [function(*args, **kwargs) for function, args, kwargs in task_list]


def test_joblib_forwards_parallel_configuration_and_closes_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded_parallel: _RecordingParallel | None = None

    def parallel_factory(
        *,
        n_jobs: int,
        backend: str,
        batch_size: int | str,
        pre_dispatch: int | str,
        return_as: str,
    ) -> _RecordingParallel:
        nonlocal recorded_parallel
        recorded_parallel = _RecordingParallel(
            n_jobs=n_jobs,
            backend=backend,
            batch_size=batch_size,
            pre_dispatch=pre_dispatch,
            return_as=return_as,
        )
        return recorded_parallel

    monkeypatch.setattr(joblib_module, "Parallel", parallel_factory)
    processor = Joblib(
        n_jobs=3,
        joblib_backend="threading",
        batch_size=2,
        pre_dispatch=4,
    )
    algorithm, _ = configure_processor_for_dispatch(
        processor,
        n_iterations=1,
        n_particles=2,
    )

    processor.run()

    assert recorded_parallel is not None
    assert recorded_parallel.n_jobs == 3
    assert recorded_parallel.backend == "threading"
    assert recorded_parallel.batch_size == 2
    assert recorded_parallel.pre_dispatch == 4
    assert recorded_parallel.return_as == "list"
    assert len(recorded_parallel.calls) == 3
    assert [particle.updates for particle in algorithm.population.values()] == [1, 1]
    assert recorded_parallel.exited


@pytest.mark.parametrize(
    "joblib_backend",
    [
        pytest.param(
            "loky",
            marks=pytest.mark.multiprocess,
            id="loky",
        ),
        pytest.param(
            "threading",
            id="threading",
        ),
        pytest.param(
            "multiprocessing",
            marks=pytest.mark.multiprocess,
            id="multiprocessing",
        ),
        pytest.param(
            "sequential",
            id="sequential",
        ),
    ],
)
def test_joblib_dispatches_and_returns_every_particle_with_supported_backend(
    joblib_backend: str,
) -> None:
    processor = Joblib(
        n_jobs=2,
        joblib_backend=joblib_backend,
        batch_size=1,
        pre_dispatch=2,
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


def test_joblib_accepts_default_configuration_and_empty_dispatch() -> None:
    processor = Joblib()
    algorithm, migration = configure_processor_for_dispatch(
        processor,
        n_iterations=0,
        n_particles=0,
    )

    processor.run()

    assert algorithm.population == {}
    assert migration.loop_finalized


@pytest.mark.parametrize(
    "joblib_backend",
    [
        pytest.param(
            "loky",
            marks=pytest.mark.multiprocess,
            id="loky",
        ),
        pytest.param(
            "threading",
            id="threading",
        ),
        pytest.param(
            "multiprocessing",
            marks=pytest.mark.multiprocess,
            id="multiprocessing",
        ),
        pytest.param(
            "sequential",
            id="sequential",
        ),
    ],
)
def test_joblib_propagates_worker_exceptions_and_finalizes_loop(
    joblib_backend: str,
) -> None:
    failing_id = "MainProcessor|particle:1"
    processor = Joblib(
        n_jobs=2,
        joblib_backend=joblib_backend,
        batch_size=1,
        pre_dispatch=2,
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


def test_joblib_rejects_zero_worker_count() -> None:
    with pytest.raises(ValueError, match="n_jobs"):
        Joblib(n_jobs=0)


def test_joblib_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Invalid Joblib backend"):
        Joblib(joblib_backend="not-a-backend")


@pytest.mark.parametrize(
    "batch_size",
    [0, -1],
    ids=["zero", "negative"],
)
def test_joblib_rejects_non_positive_batch_size(batch_size: int) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        Joblib(batch_size=batch_size)


@pytest.mark.parametrize(
    "pre_dispatch",
    [0, -1],
    ids=["zero", "negative"],
)
def test_joblib_rejects_non_positive_pre_dispatch(pre_dispatch: int) -> None:
    with pytest.raises(ValueError, match="pre_dispatch"):
        Joblib(pre_dispatch=pre_dispatch)
