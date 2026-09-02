from functools import partial
from multiprocessing import Event
from multiprocessing.pool import ThreadPool

from ...algorithm.base import CostFunctionWrapperBase
from .base import (
    LocalEvent,
    ProcessorBase,
    evaluate_particle,
)


class ThreadsPoolCostFunctionWrapper(CostFunctionWrapperBase):
    """Threads pool processor cost-function wrapper."""


class ThreadsPool(ProcessorBase):
    def __init__(
        self,
        n_process: int | None = None,
        chunksize: int = 1,
    ) -> None:
        if chunksize <= 0:
            raise ValueError("chunksize must be greater than zero.")

        self._n_process = n_process
        self._chunksize = chunksize

        self._cost_function_wrapper = ThreadsPoolCostFunctionWrapper

    def initialize_execution_context(self) -> None:
        self._stop_signal = Event()
        self._wait_signal = Event()

    def finalize_execution_context(self) -> None:
        self._stop_signal = LocalEvent()
        self._wait_signal = LocalEvent()

    def run(self) -> None:
        self.init_particles()
        self._stop_signal.clear()

        with ThreadPool(processes=self._n_process) as pool:
            self._status.n_process = pool._processes  # type: ignore

            for actual_iter in range(self._n_iter + 1):
                self.update_iter_counter(actual_iter)

                self.wait_sync(actual_iter)
                if self._stop_signal.is_set():
                    break

                self.migration_control()

                self._algorithm.pre_iteration(actual_iter)

                new_particles_ids = self._algorithm.new_particles_id
                if new_particles_ids:
                    worker = partial(
                        evaluate_particle,
                        stop_signal=self._stop_signal,
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
                    worker = partial(
                        evaluate_particle,
                        stop_signal=self._stop_signal,
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
