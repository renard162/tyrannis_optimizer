"""IslandMigration transfers ownership through the real distributed lifecycle."""

import json
from collections import Counter
from math import isfinite
from multiprocessing.managers import SyncManager

import pytest

# pytest adds tests/ to pythonpath; basedpyright does not read that setting.
from _support.assertions import (  # pyright: ignore[reportImplicitRelativeImport]
    assert_optimizer_result_consistent,
)
from _support.distributed_backend import (  # pyright: ignore[reportImplicitRelativeImport]
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
from tyrannis.migration import IslandMigration
from tyrannis.processor.serial import Serial
from tyrannis.space.continuous import Continuous

from ._history import HistoryRecord, population_at

pytestmark = pytest.mark.multiprocess


def test_island_migration_transfers_one_particle_and_conserves_global_population(
    manager: SyncManager,
) -> None:
    backend = TestDistributedBackend(manager)
    optimizer = make_optimizer(
        Continuous({"x": (1.0, 3.0)}, cost_function=sphere),
        IntegrationProbeAlgorithm(double_eval=False, force_stop_at_iteration=None),
        n_iterations=1,
        n_particles=2,
        backend=backend,
        processor=Serial(),
        migration=IslandMigration(
            trigger="synchronous",
            movement_strategy="ring",
            migration_size=1,
            particle_selection="best",
            initial_iter=1,
            min_interval=1,
            min_population=1,
            balance_population=False,
        ),
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
    assert migration["iteration"] == 1
    donor = migration["origin"]
    receiver = migration.get("destination")
    assert isinstance(receiver, str)
    assert donor != receiver
    assert {donor, receiver} == islands
    before = {island: population_at(records, island, 0) for island in islands}
    after = {island: population_at(records, island, 1) for island in islands}

    particle = migration["particle"]
    identifier = particle["identifier"]
    assert all(len(population) == 2 for population in before.values())
    assert identifier in before[donor]
    assert identifier not in before[receiver]
    assert isfinite(particle["fitness"])
    assert particle["fitness"] == before[donor][identifier]["fitness"]
    assert particle["fitness"] == min(
        member["fitness"] for member in before[donor].values()
    )
    assert set(after[donor]) == set(before[donor]) - {identifier}
    assert set(after[receiver]) == set(before[receiver]) | {identifier}
    assert len(after[donor]) == 1
    assert len(after[receiver]) == 3

    # Multiset equality catches losses, fabricated IDs and simultaneous copies.
    initial_ids = Counter(
        member_id for population in before.values() for member_id in population
    )
    final_ids = Counter(
        member_id for population in after.values() for member_id in population
    )
    assert final_ids == initial_ids
    assert final_ids.total() == 4
    assert set(final_ids.values()) == {1}

    # Transport preserves the donor's lineage; only the subsequent ordinary
    # update extends it. Comparing whole snapshots would reject valid progress.
    assert particle["trace"] == before[donor][identifier]["trace"]
    assert particle["trace"]
    assert particle["evaluation_count"] == before[donor][identifier]["evaluation_count"]
    received = after[receiver][identifier]
    assert received["trace"][: len(particle["trace"])] == particle["trace"]
    assert received["evaluation_count"] == particle["evaluation_count"] + 1

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
