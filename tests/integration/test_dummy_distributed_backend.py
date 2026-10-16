"""Calibrate two real Serial islands without a production distributed runtime."""

import json
from collections.abc import Iterator
from multiprocessing import active_children, get_context
from multiprocessing.managers import SyncManager
from os import getpid
from typing import TypedDict

import pytest

# pytest adds tests/ to pythonpath; basedpyright does not read that pytest setting.
from _support.assertions import (  # pyright: ignore[reportImplicitRelativeImport]
    assert_optimizer_result_consistent,
    assert_reproducible,
)
from _support.distributed_backend import (  # pyright: ignore[reportImplicitRelativeImport]
    TestDistributedBackend,
)

# The existing factory omits the Algorithm generic argument.
from _support.factories import (  # pyright: ignore[reportImplicitRelativeImport]
    make_optimizer,  # pyright: ignore[reportUnknownVariableType]
)
from _support.integration_probe import (  # pyright: ignore[reportImplicitRelativeImport]
    IntegrationProbeAlgorithm,
)
from _support.objectives import (  # pyright: ignore[reportImplicitRelativeImport]
    fails_for_positive,
    sphere,
)

from tyrannis import Optimizer
from tyrannis.core.results import ProcessorResult
from tyrannis.migration.island_isolation import IslandIsolation
from tyrannis.processor.serial import Serial
from tyrannis.processor.threads import ThreadsPool
from tyrannis.space.continuous import Continuous

pytestmark = pytest.mark.multiprocess


class _Trace(TypedDict):
    scalar: float
    array: list[float]
    draws: list[float]


class _ParticleSnapshot(TypedDict):
    identifier: str
    fitness: float
    trace: list[_Trace]


class _HistoryRecord(TypedDict):
    origin: str
    iteration: int
    particle: _ParticleSnapshot


@pytest.fixture
def manager() -> Iterator[SyncManager]:
    with get_context("spawn").Manager() as runtime:
        yield runtime


def _optimizer(
    backend: TestDistributedBackend, processor: Serial | None = None
) -> Optimizer:
    return make_optimizer(
        Continuous({"x": (1.0, 3.0)}, cost_function=sphere),
        IntegrationProbeAlgorithm(),
        n_iterations=1,
        n_particles=2,
        backend=backend,
        processor=processor,
        migration=IslandIsolation(),
        history="iteration",
    )


def test_two_islands_and_consolidation(manager: SyncManager) -> None:
    backend = TestDistributedBackend(manager)
    before = {child.pid for child in active_children()}
    optimizer = _optimizer(backend).fit()
    assert {child.pid for child in active_children()} == before
    island_ids = {"island:0", "island:1"}

    assert backend.n_executors == 2
    assert set(backend.local_bests) == island_ids
    assert all(
        isinstance(result, ProcessorResult) for result in backend.local_bests.values()
    )
    assert set(backend.worker_pids) == island_ids
    assert len(set(backend.worker_pids.values())) == 2
    assert getpid() not in backend.worker_pids.values()

    records: list[_HistoryRecord] = [
        json.loads(entry) for entry in backend.result.history
    ]
    for identifier in island_ids:
        island_records = [
            record for record in records if record["origin"] == identifier
        ]
        assert (
            len(island_records) == 4
        )  # Two particles at initialization and iteration 1.
        assert {record["iteration"] for record in island_records} == {0, 1}
        assert {record["particle"]["identifier"] for record in island_records} == {
            f"{identifier}|particle:0",
            f"{identifier}|particle:1",
        }
        assert all(record["particle"]["trace"] for record in island_records)
    candidates = [result.result for result in backend.local_bests.values()]
    assert all(candidate is not None for candidate in candidates)
    best_fitness = min(record["particle"]["fitness"] for record in records)
    assert backend.result.result is not None
    assert backend.result.result["fitness"] == best_fitness
    assert all(not result.history for result in backend.local_bests.values())
    assert_optimizer_result_consistent(optimizer)


def test_reproducible_independent_islands(manager: SyncManager) -> None:
    backends: list[TestDistributedBackend] = []

    def factory() -> Optimizer:
        backend = TestDistributedBackend(manager)
        backends.append(backend)
        return _optimizer(backend, Serial())

    assert_reproducible(factory)
    assert backends[0].result.history == backends[1].result.history
    # History observes workers, rather than the untouched Algorithm template.
    records: list[_HistoryRecord] = [
        json.loads(entry) for entry in backends[0].result.history
    ]
    samples = {
        record["origin"]: record["particle"]["trace"][-1]
        for record in records
        if record["iteration"] == 1
        and record["particle"]["identifier"].endswith("|particle:0")
    }
    first, second = samples["island:0"], samples["island:1"]
    assert (first["scalar"], first["array"], first["draws"]) != (
        second["scalar"],
        second["array"],
        second["draws"],
    )


def test_backend_rejects_a_non_serial_processor(manager: SyncManager) -> None:
    with pytest.raises(Exception):  # noqa: B017
        _ = make_optimizer(
            Continuous({"x": (1.0, 3.0)}, cost_function=sphere),
            IntegrationProbeAlgorithm(),
            n_iterations=1,
            n_particles=1,
            backend=TestDistributedBackend(manager),
            processor=ThreadsPool(n_jobs=1),
            migration=IslandIsolation(),
        )


def test_worker_failure_reaps_the_pool(manager: SyncManager) -> None:
    backend = TestDistributedBackend(manager)
    optimizer = make_optimizer(
        Continuous({"x": (1.0, 3.0)}, cost_function=fails_for_positive),
        IntegrationProbeAlgorithm(),
        n_iterations=1,
        n_particles=1,
        backend=backend,
        migration=IslandIsolation(),
    )
    before = {child.pid for child in active_children()}
    with pytest.raises(Exception):  # noqa: B017
        _ = optimizer.fit()
    assert {child.pid for child in active_children()} == before
