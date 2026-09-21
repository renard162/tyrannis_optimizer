from __future__ import annotations

from typing import Protocol, final, override

import numpy as np

from tyrannis.backend.local import Local
from tyrannis.core.algorithm import (
    FITNESS_UNDEFINED,
    AlgorithmBase,
    CostFunctionWrapperBase,
    ParticleBase,
)
from tyrannis.core.backend_migration import MigrationDriverBase
from tyrannis.core.results import HistoryConfig


@final
class _Algorithm(AlgorithmBase[ParticleBase]):
    def __init__(self) -> None:
        pass

    @override
    def configure(
        self,
        identifier: str,
        cost_function_wrapper: type[CostFunctionWrapperBase],
        seed: int | np.random.SeedSequence | None,
    ) -> None:
        del identifier, cost_function_wrapper, seed

    @override
    def create_particle(
        self,
        identifier: str,
        variables: dict[str, float] | None = None,
        fitness: np.float64 = FITNESS_UNDEFINED,
        *args: object,
        **kwargs: object,
    ) -> None:
        raise NotImplementedError

    @override
    def delete_particle(self, identifier: str | None) -> None:
        raise NotImplementedError

    @override
    def pre_iteration(self, actual_iter: int) -> None:
        raise NotImplementedError

    @override
    def create_random_cache(self, particle_ids: list[str], initialize: bool) -> None:
        raise NotImplementedError

    @override
    def initialize_particle(self, identifier: str) -> ParticleBase:
        raise NotImplementedError

    @staticmethod
    @override
    def consolidate_new_particles(particle: ParticleBase) -> ParticleBase:
        raise NotImplementedError

    @override
    def update_particle(self, identifier: str) -> ParticleBase:
        raise NotImplementedError

    @override
    def post_iteration(self, actual_iter: int) -> None:
        raise NotImplementedError


class _Migration(MigrationDriverBase):
    def __init__(self, initial_iter: int = 1) -> None:
        del initial_iter
        self._migration_processor_init_kargs: dict[str, object] = {}

    @override
    def start(self) -> None:
        raise NotImplementedError

    @override
    def stop(self) -> None:
        raise NotImplementedError


class _LocalContext(Protocol):
    def initialize_context(
        self,
        algorithm: _Algorithm,
        n_iter: int,
        n_particles: int,
        migration: _Migration,
        processor: None,
        fitness_failure_strategy: str,
        history_config: HistoryConfig,
        seed: int | None,
    ) -> None: ...


def _initialize(context: _LocalContext) -> None:
    context.initialize_context(
        algorithm=_Algorithm(),
        n_iter=1,
        n_particles=1,
        migration=_Migration(),
        processor=None,
        fitness_failure_strategy="raise",
        history_config=HistoryConfig(),
        seed=None,
    )


def test_result_is_absent_until_local_execution_produces_it() -> None:
    backend = Local()
    _initialize(backend)

    assert backend.result is None
