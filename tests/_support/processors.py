"""Serializable test doubles shared by isolated processor unit tests."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from threading import get_ident
from typing import Any

import numpy as np

from _support.numerics import BASE_SEED
from tyrannis.core.processor import ProcessorBase
from tyrannis.core.results import HistoryConfig


@dataclass
class ProcessorParticleDouble:
    """Minimal serializable particle state returned by processor workers."""

    identifier: str
    item: int
    new_particle: bool = True
    updates: int = 0
    fitness: np.float64 = np.float64(0.0)  # noqa: RUF009
    candidate_fitness: np.float64 | None = np.float64(0.0)  # noqa: RUF009
    error_fitness: np.float64 | None = None
    candidate_variables: dict[str, float] | None = None
    random_cache: dict[str, Any] = field(default_factory=dict)

    def __call__(self) -> dict[str, Any]:
        return {"identifier": self.identifier, "item": self.item}

    def dump(self) -> str:
        return self.identifier


class ProcessorAlgorithmDouble:
    """Deterministic algorithm protocol double used to observe dispatch."""

    def __init__(self, *, fail_on_update: str | None = None) -> None:
        self.population: dict[str, ProcessorParticleDouble] = {}
        self.local_best = None
        self.iter_best = None
        self.iter_worst = None
        self.double_particle_check = False
        self.double_check_ids: list[str] = []
        self.initialized_ids: list[str] = []
        self.updated_ids: list[str] = []
        self.worker_threads: list[int] = []
        self.iterations: list[tuple[str, int]] = []
        self.fail_on_update = fail_on_update

    @property
    def new_particles_id(self) -> list[str]:
        return [
            identifier
            for identifier, particle in self.population.items()
            if particle.new_particle
        ]

    def create_particle(self, identifier: str) -> None:
        item = int(identifier.rsplit(":", maxsplit=1)[1])
        self.population[identifier] = ProcessorParticleDouble(identifier, item)

    def initialize_particle(self, identifier: str) -> ProcessorParticleDouble:
        self.initialized_ids.append(identifier)
        self.worker_threads.append(get_ident())
        particle = deepcopy(self.population[identifier])
        particle.new_particle = False
        return particle

    def consolidate_new_particles(
        self, particle: ProcessorParticleDouble
    ) -> ProcessorParticleDouble:
        return particle

    def update_particle(self, identifier: str) -> ProcessorParticleDouble:
        self.updated_ids.append(identifier)
        self.worker_threads.append(get_ident())
        if identifier == self.fail_on_update:
            raise ValueError(f"failed item: {identifier}")
        particle = deepcopy(self.population[identifier])
        particle.updates += 1
        return particle

    def second_update_particle(self, identifier: str) -> ProcessorParticleDouble:
        raise AssertionError(f"unexpected second update: {identifier}")

    def create_random_cache(self, particle_ids: list[str], initialize: bool) -> None:
        del particle_ids, initialize

    def update_population(self, particles: list[ProcessorParticleDouble]) -> None:
        self.population.update(
            {particle.identifier: particle for particle in particles}
        )

    def update_n_particles(self) -> None:
        return None

    def pre_iteration(self, actual_iter: int) -> None:
        self.iterations.append(("pre", actual_iter))

    def inter_iteration(self, actual_iter: int) -> None:
        raise AssertionError(f"unexpected inter iteration: {actual_iter}")

    def post_iteration(self, actual_iter: int) -> None:
        self.iterations.append(("post", actual_iter))


class ProcessorMigrationDouble:
    """Minimal migration lifecycle double for isolated processor execution."""

    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.loop_initialized = False
        self.loop_finalized = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def initialize_loop_context(self, migration_signal: object) -> None:
        del migration_signal
        self.loop_initialized = True

    def finalize_loop_context(self) -> None:
        self.loop_finalized = True

    def migration_control(self, **kwargs: Any) -> None:
        del kwargs

    def synchronization_control(self, **kwargs: Any) -> None:
        del kwargs


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
        algorithm=algorithm,  # type: ignore[arg-type]
        n_iter=n_iterations,
        n_particles=n_particles,
        migration_driver=None,  # type: ignore[arg-type]
        history_config=HistoryConfig(),
        fitness_failure_strategy="raise",
        seed=BASE_SEED,
    )
    processor._migration_processor = migration  # type: ignore[assignment]
    return algorithm, migration
