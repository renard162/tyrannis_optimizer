from functools import partial
from multiprocessing import Event
from multiprocessing.pool import ThreadPool

from ..core.algorithm import CostFunctionWrapperBase
from ..core.processor import (
    ProcessorBase,
    evaluate_particle,
)


class ThreadsPoolCostFunctionWrapper(CostFunctionWrapperBase):
    """Threads pool processor cost-function wrapper."""


class ThreadsPool(ProcessorBase):
    def __init__(
        self,
        n_jobs: int | None = None,
        chunksize: int = 1,
    ) -> None:
        if chunksize <= 0:
            raise ValueError("chunksize must be greater than zero.")

        self._n_process = n_jobs
        self._chunksize = chunksize

        self._cost_function_wrapper = ThreadsPoolCostFunctionWrapper

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

        with ThreadPool(processes=self._n_process) as pool:
            for actual_iter in range(self._n_iter + 1):
                self.migration_control(actual_iter)

                self._algorithm.pre_iteration(actual_iter)

                new_particles_ids = self._algorithm.new_particles_id
                if new_particles_ids:
                    self._algorithm.create_random_cache(new_particles_ids)
                    worker = partial(
                        evaluate_particle,
                        algorithm=self._algorithm,
                        fitness_failure_strategy=self._fitness_failure_strategy,
                        initialize_particle=True,
                    )
                    new_particles = pool.imap_unordered(
                        worker,
                        new_particles_ids,
                        chunksize=self._chunksize,
                    )
                    self._algorithm.update_population(new_particles)

                if actual_iter > 0:
                    self._algorithm.create_random_cache(
                        [idx for idx in self._algorithm.population]
                    )
                    worker = partial(
                        evaluate_particle,
                        algorithm=self._algorithm,
                        fitness_failure_strategy=self._fitness_failure_strategy,
                        initialize_particle=False,
                    )
                    processed_particles = pool.imap_unordered(
                        worker,
                        self._algorithm.population,
                        chunksize=self._chunksize,
                    )
                    self._algorithm.update_population(processed_particles)

                self._algorithm.post_iteration(actual_iter)

                self.update_status()
        self.finalize_loop_context()
