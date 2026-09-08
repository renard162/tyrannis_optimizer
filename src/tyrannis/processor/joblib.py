from collections.abc import Iterable
from functools import partial
from multiprocessing import Event
from typing import cast

from joblib import Parallel, delayed

from ..core.algorithm import CostFunctionWrapperBase, ParticleBase
from ..core.processor import (
    ProcessorBase,
    evaluate_particle,
)


class JoblibCostFunctionWrapper(CostFunctionWrapperBase):
    """Threads pool processor cost-function wrapper."""


class Joblib(ProcessorBase):
    def __init__(
        self,
        n_process: int | None = None,
        joblib_backend: str = "loky",
    ) -> None:
        self._n_process = n_process
        self._joblib_backend = joblib_backend

        self._cost_function_wrapper = JoblibCostFunctionWrapper

    def initialize_execution_context(self) -> None:
        self.start_migration()

    def finalize_execution_context(self) -> None:
        self.stop_migration()

    def initialize_loop_context(self) -> None:
        self._migration_signal = Event()

        if self._migration_processor is None:
            raise RuntimeError("Migration processor cannot be None.")

        self._migration_processor.initialize_loop_context(
            migration_signal=self._migration_signal,
        )

    def finalize_loop_context(self) -> None:
        if self._migration_processor is None:
            raise RuntimeError("Migration processor cannot be None.")

        self._migration_processor.finalize_loop_context()
        self._migration_signal = None

    def run(self) -> None:
        self.init_particles()
        self.initialize_loop_context()

        initialize_worker = partial(
            evaluate_particle,
            algorithm=self._algorithm,
            fitness_failure_strategy=self._fitness_failure_strategy,
            initialize_particle=True,
        )

        update_worker = partial(
            evaluate_particle,
            algorithm=self._algorithm,
            fitness_failure_strategy=self._fitness_failure_strategy,
            initialize_particle=False,
        )

        try:
            with Parallel(
                n_jobs=self._n_process,
                backend=self._joblib_backend,
                return_as="generator_unordered",
            ) as parallel:
                for actual_iter in range(self._n_iter + 1):
                    self.migration_control(actual_iter)

                    self._algorithm.pre_iteration(actual_iter)

                    new_particles_ids = self._algorithm.new_particles_id

                    if new_particles_ids:
                        self._algorithm.create_random_cache(new_particles_ids)

                        new_particles = parallel(
                            delayed(initialize_worker)(particle_id)
                            for particle_id in new_particles_ids
                        )

                        self._algorithm.update_population(
                            cast(Iterable[ParticleBase], new_particles)
                        )

                    if actual_iter > 0:
                        population = list(self._algorithm.population)

                        self._algorithm.create_random_cache(population)

                        processed_particles = parallel(
                            delayed(update_worker)(particle_id)
                            for particle_id in population
                        )

                        self._algorithm.update_population(
                            cast(Iterable[ParticleBase], processed_particles)
                        )

                    self._algorithm.post_iteration(actual_iter)

                    self.update_status()

        finally:
            self.finalize_loop_context()
