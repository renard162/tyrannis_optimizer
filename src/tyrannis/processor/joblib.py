from collections.abc import Iterable
from functools import partial
from multiprocessing import Event
from typing import cast

from joblib import Parallel, delayed
from joblib.parallel import BACKENDS

from ..core.algorithm import CostFunctionWrapperBase, ParticleBase
from ..core.processor import (
    ProcessorBase,
    evaluate_particle,
)


class JoblibCostFunctionWrapper(CostFunctionWrapperBase):
    """Joblib processor cost-function wrapper."""


class Joblib(ProcessorBase):
    def __init__(
        self,
        n_jobs: int = -1,
        joblib_backend: str = "loky",
        batch_size: int | str = "auto",
        pre_dispatch: int | str = "2 * n_jobs",
    ) -> None:
        if not isinstance(n_jobs, int) or isinstance(n_jobs, bool):
            raise TypeError("n_jobs must be an integer.")

        if n_jobs == 0:
            raise ValueError("n_jobs cannot be zero.")

        if not isinstance(joblib_backend, str):
            raise TypeError("joblib_backend must be a string.")

        if joblib_backend not in BACKENDS:
            available_backends = ", ".join(sorted(BACKENDS))
            raise ValueError(
                f"Invalid Joblib backend {joblib_backend!r}. "
                f"Available backends are: {available_backends}."
            )

        if isinstance(batch_size, bool):
            raise TypeError("batch_size must be a positive integer or 'auto'.")

        if isinstance(batch_size, int):
            if batch_size <= 0:
                raise ValueError("batch_size must be greater than zero.")
        elif batch_size != "auto":
            raise ValueError("batch_size must be a positive integer or 'auto'.")

        if isinstance(pre_dispatch, bool):
            raise TypeError("pre_dispatch must be a positive integer or a string.")

        if isinstance(pre_dispatch, int):
            if pre_dispatch <= 0:
                raise ValueError("pre_dispatch must be greater than zero.")
        elif not isinstance(pre_dispatch, str):
            raise TypeError("pre_dispatch must be a positive integer or a string.")

        self._n_process = n_jobs
        self._joblib_backend = joblib_backend
        self._batch_size = batch_size
        self._pre_dispatch = pre_dispatch

        self._return_as = (
            "list" if joblib_backend == "multiprocessing" else "generator_unordered"
        )

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
                return_as=self._return_as,
                batch_size=cast(str, self._batch_size),
                pre_dispatch=cast(str, self._pre_dispatch),
            ) as parallel:
                for actual_iter in range(self._n_iter + 1):
                    self.migration_control(actual_iter)

                    self._algorithm.pre_iteration(actual_iter)

                    new_particles_ids = self._algorithm.new_particles_id

                    if new_particles_ids:
                        self._algorithm.create_random_cache(
                            particle_ids=new_particles_ids,
                            initialize=True,
                        )

                        new_particles = parallel(
                            delayed(initialize_worker)(particle_id)
                            for particle_id in new_particles_ids
                        )

                        self._algorithm.update_population(
                            cast(Iterable[ParticleBase], new_particles)
                        )

                    if actual_iter > 0:
                        population = list(self._algorithm.population)

                        self._algorithm.create_random_cache(
                            particle_ids=population,
                            initialize=False,
                        )

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
