"""MPI contracts exercised with real serialization and no MPI runtime."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable
from importlib.machinery import ModuleSpec
from pickle import loads
from types import ModuleType, TracebackType
from typing import Protocol, Self, final
from unittest.mock import Mock, PropertyMock

import joblib  # pyright: ignore[reportMissingTypeStubs]  # Joblib has no stubs.
import numpy as np
import pytest

from tests._support.numerics import BASE_SEED
from tests._support.objectives import constant_objective
from tests._support.processors import (
    ProcessorAlgorithmDouble,
    ProcessorMigrationDriverDouble,
    ProcessorParticleDouble,
    assert_second_phase_dispatch,
)
from tyrannis.backend.parallel.mpi_parallel import (
    MPIParallel,
    MPIParallelCostFunctionWrapper,
)
from tyrannis.core.algorithm import FITNESS_UNDEFINED
from tyrannis.core.backend_migration import MigrationDriverBase
from tyrannis.core.results import HistoryConfig

_Task = tuple[bytes, list[str], bool, bool, str]
_Worker = Callable[[bytes, list[str], bool, bool, str], bytes]


@final
class _Executor:
    """Only the executor interface consumed by MPIParallel; work stays local."""

    def __init__(self) -> None:
        self.num_workers = 2
        self.events: list[str] = []
        self.batches: list[list[list[str]]] = []
        self.snapshots: list[ProcessorAlgorithmDouble] = []

    def __enter__(self) -> Self:
        self.events.append("enter")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.events.append("exit")

    def bootup(self, *, wait: bool) -> None:
        assert wait is True
        self.events.append("bootup")

    def starmap(
        self, worker: _Worker, tasks: Iterable[_Task], *, chunksize: int
    ) -> list[bytes]:
        assert self.events[0:2] == ["enter", "bootup"]
        assert chunksize == 1
        self.events.append("dispatch")
        batches: list[list[str]] = []
        results: list[bytes] = []
        # Narrow pickle's dynamic boundary before inspecting algorithm state.
        for task in tasks:
            snapshot: object = loads(task[0])  # pyright: ignore[reportAny]  # Pickle is untyped; narrowed below.
            assert isinstance(snapshot, ProcessorAlgorithmDouble)
            self.snapshots.append(snapshot)
            batches.append(list(task[1]))
            serialized = worker(*task)
            particles: list[ProcessorParticleDouble] = loads(serialized)  # pyright: ignore[reportAny]  # Pickle result checked below.
            assert all(isinstance(p, ProcessorParticleDouble) for p in particles)
            assert [p.identifier for p in particles] == task[1]
            results.append(serialized)
        self.batches.append(batches)
        return results


@pytest.fixture
def mpi_executor(monkeypatch: pytest.MonkeyPatch) -> tuple[_Executor, Mock]:
    executor = _Executor()
    factory = Mock(return_value=executor)
    # A discoverable package satisfies _check_dependencies without loading MPI.
    package = ModuleType("mpi4py")
    package.__spec__ = ModuleSpec("mpi4py", loader=None, is_package=True)
    futures = ModuleType("mpi4py.futures")
    monkeypatch.setattr(futures, "MPIPoolExecutor", factory, raising=False)
    monkeypatch.setitem(sys.modules, "mpi4py", package)
    monkeypatch.setitem(sys.modules, "mpi4py.futures", futures)
    return executor, factory


class _Context(Protocol):
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


def _configure(
    backend: _Context,
    algorithm: ProcessorAlgorithmDouble,
    *,
    n_iter: int = 0,
    n_particles: int = 1,
    strategy: str = "raise",
) -> None:
    # Backend configuration is under test; do not pre-configure a generic wrapper.
    algorithm.initialize_context(
        fitness_function=lambda variables: np.float64(constant_objective(variables)),
        boundaries={},
        n_iter=n_iter,
        n_particles=n_particles,
    )
    backend.initialize_context(
        algorithm=algorithm,
        n_iter=n_iter,
        n_particles=n_particles,
        migration=ProcessorMigrationDriverDouble(),
        processor=None,
        fitness_failure_strategy=strategy,
        history_config=HistoryConfig(iteration=True),
        seed=BASE_SEED,
    )


def _construct(
    constructor: Callable[..., MPIParallel], kwargs: dict[str, object]
) -> MPIParallel:
    return constructor(**kwargs)


@pytest.mark.usefixtures("mpi_executor")
@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("batch_size", 0),
        ("batch_size", 1.5),
        ("batch_size", True),
        ("n_aux_jobs", 0),
        ("n_aux_jobs", 1.5),
        ("n_aux_jobs", True),
        ("aux_backend", "unknown"),
        ("aux_backend", 1),
        ("aux_batch_size", 0),
        ("aux_batch_size", "invalid"),
        ("aux_batch_size", True),
        ("aux_pre_dispatch", 0),
        ("aux_pre_dispatch", 1.5),
        ("aux_pre_dispatch", True),
    ],
)
def test_constructor_rejects_invalid_configuration(option: str, value: object) -> None:
    # Invalid inputs intentionally bypass the static constructor signature.
    with pytest.raises(Exception):  # noqa: B017
        _ = _construct(MPIParallel, {option: value})


@pytest.mark.usefixtures("mpi_executor")
def test_context_registers_mpi_identity_wrapper_and_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = MPIParallel()
    algorithm = ProcessorAlgorithmDouble()
    configure = Mock(wraps=algorithm.configure)
    monkeypatch.setattr(algorithm, "configure", configure)

    _configure(backend, algorithm)

    configure.assert_called_once_with(
        identifier="MPIParallel|algorithm",
        cost_function_wrapper=MPIParallelCostFunctionWrapper,
        seed=BASE_SEED,
    )


def test_execute_reuses_executor_and_consolidates_serialized_phases_on_driver(
    mpi_executor: tuple[_Executor, Mock], monkeypatch: pytest.MonkeyPatch
) -> None:
    executor, factory = mpi_executor
    executor.num_workers = 3  # More workers than particles still yields unit batches.
    backend = MPIParallel(
        n_aux_jobs=2, aux_backend="threading", aux_batch_size=3, aux_pre_dispatch=4
    )
    algorithm = ProcessorAlgorithmDouble()
    _configure(backend, algorithm, n_iter=1, n_particles=2)
    identifiers = ["MPIParallel|particle:0", "MPIParallel|particle:1"]
    algorithm.double_particle_check = True
    algorithm.requested_double_check_ids = identifiers[1:]
    consolidated: list[str] = []
    second_updates: list[str] = []
    second_update = ProcessorAlgorithmDouble.second_update_particle

    def observe_second_update(
        current: ProcessorAlgorithmDouble, identifier: str
    ) -> ProcessorParticleDouble:
        second_updates.append(identifier)
        return second_update(current, identifier)

    monkeypatch.setattr(
        ProcessorAlgorithmDouble, "second_update_particle", observe_second_update
    )

    def consolidate(particle: ProcessorParticleDouble) -> ProcessorParticleDouble:
        assert executor.events[-1] == "dispatch"
        assert not particle.new_particle
        consolidated.append(particle.identifier)
        particle.item += 10
        return particle

    def best(current: ProcessorAlgorithmDouble) -> ProcessorParticleDouble | None:
        return next(iter(current.population.values()), None)

    monkeypatch.setattr(algorithm, "consolidate_new_particles", consolidate)
    monkeypatch.setattr(ProcessorAlgorithmDouble, "local_best", property(best))
    parallel_factory = Mock(wraps=joblib.Parallel)
    monkeypatch.setattr(
        "tyrannis.backend.parallel.mpi_parallel.Parallel", parallel_factory
    )
    result = backend.result

    backend.execute()

    assert second_updates == identifiers[1:]
    factory.assert_called_once_with()
    parallel_factory.assert_called_once_with(
        n_jobs=2, backend="threading", batch_size=3, pre_dispatch=4, return_as="list"
    )
    assert executor.events == [
        "enter",
        "bootup",
        "dispatch",
        "dispatch",
        "dispatch",
        "exit",
    ]
    assert executor.batches == [
        [[identifiers[0]], [identifiers[1]]],
        [[identifiers[0]], [identifiers[1]]],
        [[identifiers[1]]],
    ]
    assert sorted(consolidated) == identifiers
    assert_second_phase_dispatch(algorithm, identifiers)
    assert algorithm.iterations == [("pre", 0), ("post", 0), ("pre", 1), ("post", 1)]
    # Worker bookkeeping must not be used to update the driver's global state.
    assert (
        algorithm.initialized_ids
        == algorithm.updated_ids
        == algorithm.second_updated_ids
        == []
    )
    assert [p.item for p in algorithm.population.values()] == [10, 11]
    assert backend.result is result
    assert result.result == algorithm.population[identifiers[0]]()
    assert len(result.history) == 4
    # Every batch must see the current cache and the previously consolidated phase.
    assert [snapshot.random_cache_calls[-1] for snapshot in executor.snapshots] == [
        (identifiers, True),
        (identifiers, True),
        (identifiers, False),
        (identifiers, False),
        (identifiers[1:], False),
    ]
    assert [
        [(p.new_particle, p.updates, p.item) for p in snapshot.population.values()]
        for snapshot in executor.snapshots
    ] == [
        [(True, 0, 0), (True, 0, 1)],
        [(True, 0, 0), (True, 0, 1)],
        [(False, 0, 10), (False, 0, 11)],
        [(False, 0, 10), (False, 0, 11)],
        [(False, 1, 10), (False, 1, 11)],
    ]


@pytest.mark.parametrize(
    ("batch_size", "expected"),
    [(None, [[0, 1, 2], [3, 4]]), (2, [[0, 1], [2, 3], [4]])],
    ids=["automatic-ceiling", "explicit-size"],
)
def test_execute_batches_identifiers_in_order(
    mpi_executor: tuple[_Executor, Mock],
    batch_size: int | None,
    expected: list[list[int]],
) -> None:
    executor, _ = mpi_executor
    backend = MPIParallel(batch_size=batch_size)
    algorithm = ProcessorAlgorithmDouble()
    _configure(backend, algorithm, n_particles=5)

    backend.execute()

    assert executor.batches == [
        [[f"MPIParallel|particle:{item}" for item in batch] for batch in expected]
    ]
    assert [p.item for p in algorithm.population.values()] == [0, 1, 2, 3, 4]
    assert all(not particle.new_particle for particle in algorithm.population.values())


def test_empty_double_check_selection_avoids_remote_work_and_cache(
    mpi_executor: tuple[_Executor, Mock],
) -> None:
    executor, _ = mpi_executor
    backend = MPIParallel()
    algorithm = ProcessorAlgorithmDouble()
    _configure(backend, algorithm, n_iter=1)
    algorithm.double_particle_check = True

    backend.execute()

    assert_second_phase_dispatch(algorithm, ["MPIParallel|particle:0"])
    assert len(executor.batches) == 2


def test_zero_workers_rejects_execution_before_particle_processing(
    mpi_executor: tuple[_Executor, Mock],
) -> None:
    executor, _ = mpi_executor
    executor.num_workers = 0
    backend = MPIParallel()
    algorithm = ProcessorAlgorithmDouble()
    _configure(backend, algorithm)

    with pytest.raises(Exception):  # noqa: B017
        backend.execute()

    assert executor.events == ["enter", "bootup", "exit"]
    assert executor.batches == []
    assert algorithm.iterations == algorithm.random_cache_calls == []


def test_active_force_stop_prevents_starting_iteration(
    mpi_executor: tuple[_Executor, Mock], monkeypatch: pytest.MonkeyPatch
) -> None:
    executor, _ = mpi_executor
    backend = MPIParallel()
    algorithm = ProcessorAlgorithmDouble()
    _configure(backend, algorithm, n_iter=1)
    monkeypatch.setattr(
        ProcessorAlgorithmDouble, "force_stop", PropertyMock(return_value=True)
    )

    backend.execute()

    assert algorithm.iterations == algorithm.random_cache_calls == []
    assert executor.batches == []
    assert backend.result.history == []


@pytest.mark.parametrize("strategy", ["invalidate", "raise"])
def test_worker_failure_strategy_and_executor_cleanup(
    mpi_executor: tuple[_Executor, Mock], strategy: str
) -> None:
    executor, _ = mpi_executor
    identifier = "MPIParallel|particle:0"
    backend = MPIParallel()
    algorithm = ProcessorAlgorithmDouble(fail_on_update=identifier)
    _configure(backend, algorithm, n_iter=1, strategy=strategy)

    if strategy == "raise":
        with pytest.raises(Exception):  # noqa: B017
            backend.execute()
    else:
        backend.execute()
        particle = algorithm.population[identifier]
        assert particle.candidate_fitness == FITNESS_UNDEFINED
        assert particle.error_fitness == FITNESS_UNDEFINED
        assert algorithm.iterations[-1] == ("post", 1)

    assert executor.events[-1] == "exit"
