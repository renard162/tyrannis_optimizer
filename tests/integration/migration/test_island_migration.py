"""IslandMigration transfers ownership through the real distributed lifecycle."""

import json
from collections import Counter
from collections.abc import Callable
from math import isfinite
from multiprocessing import active_children
from multiprocessing.managers import SyncManager
from threading import Event
from typing import NotRequired, TypedDict, final

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

from tyrannis.core.backend_migration import RETIRE, RETIRE_ACK
from tyrannis.core.results import HistoryConfig
from tyrannis.migration import IslandMigration
from tyrannis.migration.island_migration import IslandMigrationProcessor
from tyrannis.processor.serial import Serial
from tyrannis.space.continuous import Continuous

from ._history import HistoryRecord, ParticleSnapshot, population_at

pytestmark = pytest.mark.multiprocess


class _WireMessage(TypedDict):
    type: str
    request_id: NotRequired[str]
    particle: NotRequired[ParticleSnapshot | None]


_wire_message: Callable[[str], _WireMessage] = json.loads


@final
class _RouterTrace:
    """Signal one observed message and retain FIFO operations for assertions."""

    def __init__(self, endpoint: str, kind: str, ready: Event) -> None:
        self.trigger = (endpoint, kind)
        self.ready = ready
        self.messages: list[tuple[str, str, _WireMessage]] = []

    def __call__(self, island: str, direction: str, message: str) -> None:
        payload = _wire_message(message)
        self.messages.append((island, direction, payload))
        if (island, payload["type"]) == self.trigger:
            self.ready.set()

    def exchange(
        self, island: str, direction: str, kind: str
    ) -> tuple[int, _WireMessage]:
        (match,) = [
            (index, payload)
            for index, (endpoint, flow, payload) in enumerate(self.messages)
            if (endpoint, flow, payload["type"]) == (island, direction, kind)
        ]
        return match


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


def test_force_stop_releases_the_other_island_from_its_checkpoint(
    manager: SyncManager,
) -> None:
    stopping, continuing = "island:0", "island:1"
    paused = manager.Event()
    trace = _RouterTrace(
        continuing, IslandMigrationProcessor.SYNCHRONIZATION_PAUSE, paused
    )
    backend = TestDistributedBackend(manager, observe=trace)
    processor = PacedSerial()
    optimizer = make_optimizer(
        Continuous({"x": (1.0, 3.0)}, cost_function=sphere),
        IntegrationProbeAlgorithm(
            force_stop_at_iteration=0, force_stop_on_island=stopping
        ),
        n_iterations=1,
        n_particles=1,
        backend=backend,
        processor=processor,
        migration=IslandMigration(trigger="synchronous"),
        history="iteration",
    )
    backend.init_processors()
    processor.processors_pool[stopping].pause_before_retirement(paused)
    children = {child.pid for child in active_children()}
    _ = optimizer.fit()
    assert {child.pid for child in active_children()} == children

    records: list[HistoryRecord] = [json.loads(row) for row in backend.result.history]
    for island, iterations in ((stopping, [0]), (continuing, [0, 1])):
        local = [r for r in records if r["origin"] == island]
        assert sorted(r["iteration"] for r in local) == iterations
        candidate = backend.local_bests[island].result
        assert candidate is not None
        assert candidate in [r["particle"] for r in local]
        assert candidate["fitness"] == min(r["particle"]["fitness"] for r in local)
    assert optimizer.best_fitness == min(r["particle"]["fitness"] for r in records)
    assert_optimizer_result_consistent(optimizer)

    assert [
        payload["type"] for island, _, payload in trace.messages if island == stopping
    ] == [
        RETIRE,
        RETIRE_ACK,
    ]
    pause, _ = trace.exchange(
        continuing, "received", IslandMigrationProcessor.SYNCHRONIZATION_PAUSE
    )
    retire, _ = trace.exchange(stopping, "received", RETIRE)
    release, _ = trace.exchange(
        continuing, "sent", IslandMigrationProcessor.SYNCHRONIZATION_RELEASE
    )
    assert pause < retire < release


def test_async_island_migration_transfers_ownership_during_active_iterations(
    manager: SyncManager,
) -> None:
    deliveries = {island: manager.Event() for island in ("island:0", "island:1")}
    backend = TestDistributedBackend(manager, first_deliveries=deliveries)
    processor = PacedSerial()
    optimizer = make_optimizer(
        Continuous({"x": (1.0, 3.0)}, cost_function=sphere),
        IntegrationProbeAlgorithm(),
        n_iterations=2,
        n_particles=2,
        backend=backend,
        processor=processor,
        migration=IslandMigration(
            trigger="n_iter",
            movement_strategy="ring_sequential",
            min_interval=2,
            min_population=1,
        ),
        history=["iteration", "pre_iteration", "migration"],
    )
    backend.init_processors()
    # Both islands publish state in their first iteration. Each next control
    # runs after its own delivery: a request at the donor, a particle at the receiver.
    for island, delivered in deliveries.items():
        processor.processors_pool[island].pause_before(2, delivered)
    _ = optimizer.fit()

    records: list[HistoryRecord] = [json.loads(row) for row in backend.result.history]
    (transfer,) = [
        row for row in records if row["event"] == HistoryConfig.get_event("migration")
    ]
    assert "destination" in transfer
    donor, receiver = transfer["origin"], transfer["destination"]
    assert {donor, receiver} == set(deliveries)
    sent = transfer["particle"]
    identifier = sent["identifier"]
    source = population_at(records, donor, 1)
    destination = population_at(records, receiver, 1)
    assert source.keys().isdisjoint(destination)
    assert sent == source[identifier]
    assert set(population_at(records, donor, 2)) == source.keys() - {identifier}
    assert set(population_at(records, receiver, 2)) == destination.keys() | {identifier}

    # Snapshot on arrival, before local evaluation changes fitness or extends the trace.
    arrivals = population_at(records, receiver, 2, event="pre_iteration")
    assert sent["trace"] and arrivals[identifier] == sent
    assert optimizer.best_fitness == min(
        row["particle"]["fitness"]
        for row in records
        if row["event"] == HistoryConfig.get_event("iteration")
    )
    assert_optimizer_result_consistent(optimizer)


def test_retiring_island_refuses_a_departure_and_drains_an_authorized_arrival(
    manager: SyncManager,
) -> None:
    stopping, continuing = "island:0", "island:1"
    request_delivered, retiring = manager.Event(), manager.Event()
    trace = _RouterTrace(stopping, RETIRE, retiring)
    backend = TestDistributedBackend(
        manager, first_deliveries={continuing: request_delivered}, observe=trace
    )
    processor = PacedSerial()
    optimizer = make_optimizer(
        Continuous({"x": (1.0, 3.0)}, cost_function=sphere),
        IntegrationProbeAlgorithm(
            force_stop_at_iteration=1, force_stop_on_island=stopping
        ),
        n_iterations=2,
        n_particles=2,
        backend=backend,
        processor=processor,
        # Each island can donate exactly one particle at the first checkpoint.
        migration=IslandMigration(migration_size=2, min_population=1),
        history=["iteration", "migration"],
    )
    backend.init_processors()
    # Neither request can be consumed during iteration 1: authorization requires
    # the states published by both islands after they drain their inboxes.
    processor.processors_pool[stopping].pause_before_retirement(request_delivered)
    processor.processors_pool[continuing].pause_before(2, retiring)
    _ = optimizer.fit()

    retire, _ = trace.exchange(stopping, "received", RETIRE)
    ack, _ = trace.exchange(stopping, "sent", RETIRE_ACK)
    for island in (stopping, continuing):
        request, authorization = trace.exchange(
            island, "sent", IslandMigrationProcessor.MIGRATION_REQUEST
        )
        response, reply = trace.exchange(
            island, "received", IslandMigrationProcessor.PARTICLE
        )
        assert request < retire < response < ack
        assert (
            "request_id" in authorization
            and reply.get("request_id") == authorization["request_id"]
        )
        if island == stopping:
            assert "particle" in reply and reply["particle"] is None

    arrival, delivery = trace.exchange(
        stopping, "sent", IslandMigrationProcessor.PARTICLE
    )
    sent = delivery.get("particle")
    assert sent is not None and retire < arrival < ack
    _, donor_reply = trace.exchange(
        continuing, "received", IslandMigrationProcessor.PARTICLE
    )
    assert donor_reply.get("particle") == sent
    assert [
        p["type"]
        for endpoint, flow, p in trace.messages[retire + 1 :]
        if endpoint == stopping and flow == "received"
    ] == [IslandMigrationProcessor.PARTICLE]

    records: list[HistoryRecord] = [json.loads(row) for row in backend.result.history]
    (transfer,) = [
        r for r in records if r["event"] == HistoryConfig.get_event("migration")
    ]
    assert (transfer["origin"], transfer.get("destination"), transfer["particle"]) == (
        continuing,
        stopping,
        sent,
    )
    identifier = sent["identifier"]
    before = set(population_at(records, continuing, 1))
    assert identifier in before
    last = set(population_at(records, stopping, 1))
    assert last.isdisjoint(before)
    # Retirement arrivals are visible only in the worker's final population.
    assert backend.final_populations == {
        stopping: last | {identifier},
        continuing: before - {identifier},
    }
    assert population_at(records, continuing, 2).keys() == before - {identifier}
    assert not population_at(records, stopping, 2)
