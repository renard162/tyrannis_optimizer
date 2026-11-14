"""GlobalBest copies a solution through the real distributed lifecycle."""

import json
from math import isfinite
from multiprocessing.managers import SyncManager

import pytest

# pytest adds tests/ to pythonpath; basedpyright does not read that setting.
from _support.assertions import (  # pyright: ignore[reportImplicitRelativeImport]
    assert_optimizer_result_consistent,
)
from _support.distributed_backend import (  # pyright: ignore[reportImplicitRelativeImport]
    PacedSerial,
    TestDistributedBackend,
)

# The shared factory's production AlgorithmBase annotation omits its type argument.
from _support.factories import (  # pyright: ignore[reportImplicitRelativeImport]
    make_optimizer,  # pyright: ignore[reportUnknownVariableType]
)
from _support.integration_probe import (  # pyright: ignore[reportImplicitRelativeImport]
    IntegrationProbeAlgorithm,
)
from _support.objectives import sphere  # pyright: ignore[reportImplicitRelativeImport]

from tyrannis.core.results import HistoryConfig
from tyrannis.migration import GlobalBest
from tyrannis.processor.serial import Serial
from tyrannis.space.continuous import Continuous

from ._history import HistoryRecord, population_at

pytestmark = pytest.mark.multiprocess


def test_global_best_copies_to_the_worst_slot_and_preserves_both_populations(
    manager: SyncManager,
) -> None:
    backend = TestDistributedBackend(manager)
    # Initialization uses equal midpoints. One ordinary update before the only
    # checkpoint makes best/worst observable without prescribing a tie breaker.
    optimizer = make_optimizer(
        Continuous({"x": (1.0, 3.0)}, cost_function=sphere),
        IntegrationProbeAlgorithm(double_eval=False, force_stop_at_iteration=None),
        n_iterations=2,
        n_particles=2,
        backend=backend,
        processor=Serial(),
        migration=GlobalBest(initial_iter=2, check_interval=1, synchronous=True),
        history=["iteration", "migration"],
    ).fit()

    islands = {"island:0", "island:1"}
    assert set(backend.local_bests) == islands
    records: list[HistoryRecord] = [
        json.loads(entry) for entry in backend.result.history
    ]
    migrations = [
        record
        for record in records
        if record["event"] == HistoryConfig.get_event("migration")
    ]
    assert len(migrations) == 1
    migration = migrations[0]
    assert migration["iteration"] == 2
    assert migration.get("destination") == "all"
    origin = migration["origin"]
    assert origin in islands
    (receiver,) = islands - {origin}
    before = {island: population_at(records, island, 1) for island in islands}
    after = {island: population_at(records, island, 2) for island in islands}

    particle = migration["particle"]
    identifier = particle["identifier"]
    assert identifier in before[origin]
    assert isfinite(particle["fitness"])
    assert particle["fitness"] == min(
        member["fitness"]
        for population in before.values()
        for member in population.values()
    )
    assert particle["fitness"] == before[origin][identifier]["fitness"]
    assert particle["trace"] == before[origin][identifier]["trace"]
    assert particle["trace"]

    # The source retains every particle and its own lineage after the checkpoint.
    for island in islands:
        assert len(before[island]) == len(after[island]) == 2
        assert set(after[island]) == set(before[island])
    for member_id, member in before[origin].items():
        assert (
            after[origin][member_id]["trace"][: len(member["trace"])] == member["trace"]
        )

    worst_fitness = max(member["fitness"] for member in before[receiver].values())
    (worst_id,) = [
        member_id
        for member_id, member in before[receiver].items()
        if member["fitness"] == worst_fitness
    ]
    # GlobalBest reuses the receiver's worst slot ID. Its incoming trace proves
    # that the solution was copied, even though the probe then updates it again.
    copied = after[receiver][worst_id]
    prefix_length = len(particle["trace"])
    assert copied["trace"][:prefix_length] == particle["trace"]
    assert copied["trace"][:prefix_length] != before[receiver][worst_id]["trace"]
    assert copied["evaluation_count"] == particle["evaluation_count"] + 1
    for member_id in set(before[receiver]) - {worst_id}:
        trace = before[receiver][member_id]["trace"]
        assert after[receiver][member_id]["trace"][: len(trace)] == trace

    local_fitnesses: list[float] = []
    for result in backend.local_bests.values():
        assert result.result is not None
        candidate: dict[str, object] = result.result
        fitness = candidate["fitness"]
        assert isinstance(fitness, float)
        assert isfinite(fitness)
        local_fitnesses.append(fitness)
    assert optimizer.best_fitness == min(local_fitnesses)
    assert_optimizer_result_consistent(optimizer)


def test_async_global_best_reaches_an_island_before_its_first_checkpoint(
    manager: SyncManager,
) -> None:
    donor, receiver = "island:1", "island:0"
    ready, delivered = manager.Event(), manager.Event()
    backend = TestDistributedBackend(manager, first_deliveries={receiver: delivered})
    processor = PacedSerial()
    optimizer = make_optimizer(
        Continuous({"x": (1.0, 3.0)}, cost_function=sphere),
        IntegrationProbeAlgorithm(),
        n_iterations=2,
        n_particles=1,
        backend=backend,
        processor=processor,
        migration=GlobalBest(initial_iter=2, synchronous=False),
        history=["iteration", "pre_iteration", "migration"],
    )
    backend.init_processors()
    # The receiver has a populated slot but has not reached its own checkpoint.
    processor.processors_pool[receiver].pause_before(1, delivered, ready=ready)
    processor.processors_pool[donor].pause_before(2, ready)
    _ = optimizer.fit()

    records: list[HistoryRecord] = [json.loads(row) for row in backend.result.history]
    (publication,) = [
        row
        for row in records
        if row["event"] == HistoryConfig.get_event("migration")
        and row["origin"] == donor
    ]
    sent = publication["particle"]
    assert sent == population_at(records, donor, 1)[sent["identifier"]]
    assert set(population_at(records, donor, 2)) == {sent["identifier"]}
    ((slot, previous),) = population_at(records, receiver, 0).items()
    assert sent["fitness"] < previous["fitness"]
    received = population_at(records, receiver, 1, event="pre_iteration")
    assert sent["trace"]
    assert received == {slot: {**sent, "identifier": slot}}
    assert set(population_at(records, receiver, 2)) == {slot}
    assert optimizer.best_fitness == min(
        row["particle"]["fitness"]
        for row in records
        if row["event"] == HistoryConfig.get_event("iteration")
    )
    assert_optimizer_result_consistent(optimizer)
