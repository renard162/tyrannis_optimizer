"""Serializable test doubles shared by isolated processor unit tests."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from copy import deepcopy
from threading import get_ident
from typing import Any, final, override

import numpy as np
from numpy.random import SeedSequence

from tyrannis.core.algorithm import (
    FITNESS_UNDEFINED,
    AlgorithmBase,
    ParticleBase,
    Serializable,
)
from tyrannis.core.backend_communication import CommunicationProcessorBase
from tyrannis.core.backend_migration import (
    MigrationDriverBase,
    MigrationProcessorBase,
)
from tyrannis.core.processor import ProcessorBase
from tyrannis.core.results import HistoryConfig
from tyrannis.core.signals import EventProtocol

from .numerics import BASE_SEED


@final
class ProcessorParticleDouble(ParticleBase):
    """Minimal serializable particle state returned by processor workers."""

    def __init__(self, identifier: str, item: int) -> None:
        super().__init__(identifier, {"item": float(item)}, np.float64(0.0))
        self.item = item
        self.updates = 0

    @override
    def __call__(self) -> dict[str, Serializable]:
        return {"identifier": self.identifier, "item": self.item}

    @override
    def dump(self) -> str:
        return self.identifier

    def mark_initialized(self) -> None:
        self._new_particle = False


@final
class ProcessorAlgorithmDouble(AlgorithmBase[ProcessorParticleDouble]):
    """Deterministic algorithm protocol double used to observe dispatch."""

    def __init__(self, *, fail_on_update: str | None = None) -> None:
        self._population: dict[str, ProcessorParticleDouble] = {}
        self._local_best: ProcessorParticleDouble | None = None
        self._iter_best: str | None = None
        self._iter_worst: str | None = None
        self._double_particle_check = False
        self._double_check_ids: list[str] = []
        self.requested_double_check_ids: list[str] = []
        self.initialized_ids: list[str] = []
        self.updated_ids: list[str] = []
        self.second_updated_ids: list[str] = []
        self.inter_iterations: list[int] = []
        self.random_cache_calls: list[tuple[list[str], bool]] = []
        self.phase_events: list[str] = []
        self.worker_threads: list[int] = []
        self.iterations: list[tuple[str, int]] = []
        self.fail_on_update = fail_on_update

    @property
    @override
    def double_particle_check(self) -> bool:
        return self._double_particle_check

    @double_particle_check.setter
    def double_particle_check(self, value: bool) -> None:
        self._double_particle_check = value

    @override
    def create_particle(
        self,
        identifier: str,
        variables: dict[str, float] | None = None,
        fitness: np.float64 = FITNESS_UNDEFINED,
        *args: object,
        **kwargs: object,
    ) -> None:
        del variables, fitness, args, kwargs
        item = int(identifier.rsplit(":", maxsplit=1)[1])
        self.population[identifier] = ProcessorParticleDouble(identifier, item)

    @override
    def delete_particle(self, identifier: str | None) -> None:
        if identifier is not None:
            _ = self.population.pop(identifier, None)

    @override
    def initialize_particle(self, identifier: str) -> ProcessorParticleDouble:
        self.initialized_ids.append(identifier)
        self.worker_threads.append(get_ident())
        particle = deepcopy(self.population[identifier])
        particle.mark_initialized()
        return particle

    @staticmethod
    @override
    def consolidate_new_particles(
        particle: ProcessorParticleDouble,
    ) -> ProcessorParticleDouble:
        return particle

    @override
    def update_particle(self, identifier: str) -> ProcessorParticleDouble:
        self.updated_ids.append(identifier)
        self.worker_threads.append(get_ident())
        if identifier == self.fail_on_update:
            raise ValueError(f"failed item: {identifier}")
        particle = deepcopy(self.population[identifier])
        particle.updates += 1
        return particle

    @override
    def second_update_particle(self, identifier: str) -> ProcessorParticleDouble:
        self.second_updated_ids.append(identifier)
        particle = deepcopy(self.population[identifier])
        particle.updates += 1
        return particle

    @override
    def create_random_cache(self, particle_ids: list[str], initialize: bool) -> None:
        self.random_cache_calls.append((list(particle_ids), initialize))
        self.phase_events.append("initial_cache" if initialize else "update_cache")

    @override
    def update_population(
        self, new_population: Iterable[ProcessorParticleDouble]
    ) -> None:
        self.phase_events.append("population_update")
        self.population.update(
            {particle.identifier: particle for particle in new_population}
        )

    @override
    def update_n_particles(self) -> None:
        return None

    @override
    def pre_iteration(self, actual_iter: int) -> None:
        self.iterations.append(("pre", actual_iter))
        self.phase_events.append("pre")

    @override
    def inter_iteration(self, actual_iter: int) -> None:
        self.inter_iterations.append(actual_iter)
        self.phase_events.append("inter")
        self.double_check_ids = list(self.requested_double_check_ids)

    @override
    def post_iteration(self, actual_iter: int) -> None:
        self.iterations.append(("post", actual_iter))
        self.phase_events.append("post")


class ProcessorMigrationDouble(MigrationProcessorBase):
    """Minimal migration lifecycle double for isolated processor execution."""

    def __init__(
        self,
        initial_iter: int = 1,
        communication_processor: CommunicationProcessorBase | None = None,
        seed: SeedSequence | int | None = None,
    ) -> None:
        del initial_iter, communication_processor, seed
        self.started: bool = False
        self.stopped: bool = False
        self.loop_initialized: bool = False
        self.loop_finalized: bool = False

    @override
    def start(self) -> None:
        self.started = True

    @override
    def stop(self) -> None:
        self.stopped = True

    @override
    def initialize_loop_context(self, migration_signal: EventProtocol) -> None:
        del migration_signal
        self.loop_initialized = True

    @override
    def finalize_loop_context(self) -> None:
        self.loop_finalized = True

    @override
    def migration_control(
        self,
        actual_iter: int,
        population: dict[str, np.float64],
        iter_best: str | None,
        insert_arrival_particle: Callable[[dict[str, object]], None],
        departure_particle: Callable[[str], dict[str, object] | None],
    ) -> None:
        del actual_iter, population, iter_best, insert_arrival_particle
        del departure_particle

    @override
    def synchronization_control(
        self,
        actual_iter: int,
        insert_arrival_particle: Callable[[dict[str, object]], None],
        departure_particle: Callable[[str], dict[str, object] | None],
    ) -> None:
        del actual_iter, insert_arrival_particle, departure_particle


class ProcessorMigrationDriverDouble(MigrationDriverBase):
    """Nominal migration driver placeholder for isolated processor setup."""

    def __init__(self, initial_iter: int = 1, *args: object, **kwargs: object) -> None:
        del initial_iter, args, kwargs
        self.created_identifiers: list[str] = []

    @override
    def create_processor_module(self, identifier: str) -> MigrationProcessorBase:
        self.created_identifiers.append(identifier)
        if hasattr(self, "_processor_class"):
            return super().create_processor_module(identifier)
        return ProcessorMigrationDouble()

    @override
    def start(self) -> None:
        return None

    @override
    def stop(self) -> None:
        return None


def configure_processor_for_dispatch(
    processor: ProcessorBase[Any],
    *,
    n_iterations: int,
    n_particles: int,
    fail_on_update: str | None = None,
) -> tuple[ProcessorAlgorithmDouble, ProcessorMigrationDouble]:
    """Attach deterministic isolated execution state to a processor."""
    algorithm = ProcessorAlgorithmDouble(fail_on_update=fail_on_update)
    migration = ProcessorMigrationDouble()
    processor.initialize_context(
        algorithm=algorithm,
        n_iter=n_iterations,
        n_particles=n_particles,
        migration_driver=ProcessorMigrationDriverDouble(),
        history_config=HistoryConfig(),
        fitness_failure_strategy="raise",
        seed=BASE_SEED,
    )
    processor._migration_processor = migration
    return algorithm, migration


def assert_second_phase_dispatch(
    algorithm: ProcessorAlgorithmDouble, particle_ids: list[str]
) -> None:
    """Assert the processor's two-phase dispatch and returned particle state."""
    checked_ids = algorithm.requested_double_check_ids
    assert algorithm.inter_iterations == [1]
    assert {
        identifier: particle.updates
        for identifier, particle in algorithm.population.items()
    } == {
        identifier: 2 if identifier in checked_ids else 1 for identifier in particle_ids
    }
    assert algorithm.random_cache_calls == [
        (particle_ids, True),
        (particle_ids, False),
        *([(checked_ids, False)] if checked_ids else []),
    ]
    assert algorithm.phase_events == [
        "pre",
        "initial_cache",
        "population_update",
        "post",
        "pre",
        "update_cache",
        "population_update",
        "inter",
        *(["update_cache", "population_update"] if checked_ids else []),
        "post",
    ]
