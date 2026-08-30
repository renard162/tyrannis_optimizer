from copy import deepcopy
from functools import partial
from multiprocessing import Event
from multiprocessing.pool import ThreadPool
from typing import Any, Self

from .base import ProcessorBase, evaluate_particle


class ThreadsPool(ProcessorBase):
    def __init__(self, n_process: int | None = None) -> None:
        self._n_process = n_process
        self._stop_signal = Event()
        self._wait_signal = Event()

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        new_processor = self.__class__.__new__(self.__class__)
        memo[id(self)] = new_processor

        for key, value in self.__dict__.items():
            if key in {"_stop_signal", "_wait_signal"}:
                continue

            setattr(new_processor, key, deepcopy(value, memo))

        new_processor._stop_signal = Event()
        new_processor._wait_signal = Event()

        return new_processor

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
                    new_particles = pool.imap_unordered(worker, new_particles_ids)
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
                        worker, self._algorithm.population
                    )
                    self._algorithm.update_population(processed_particles)

                self._algorithm.post_iteration(actual_iter)

                self.update_status()


if __name__ == "__main__":
    # from time import perf_counter, sleep

    # import numpy as np

    # from ...algorithm.pso import PSO
    # from ...examples.many_local_minima import ackley

    # def test_function(x):
    #     # sleep(0.25)
    #     result = ackley(x)
    #     # for _ in range(5_000_001):
    #     #     result = result * 1.0000001
    #     # result /= np.exp(1)
    #     return float(result)

    # algo = PSO()
    # algo.initialize_context(
    #     fitness_function=test_function,
    #     boundaries={f"{n}": (-32.768, 32.768) for n in range(2)},
    # )

    # processor = ThreadsPool(n_process=8)
    # processor.initialize_context(
    #     algorithm=algo,
    #     n_iter=5,
    #     n_particles=13,
    # )
    # obj = processor.replicate_processor("1")

    # start = perf_counter()
    # obj.run()
    # total_time = perf_counter() - start
    # print(f"{total_time=}")

    print("Breakpoint here")
