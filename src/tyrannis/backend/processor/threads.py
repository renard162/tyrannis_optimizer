from functools import partial
from multiprocessing import Event
from multiprocessing.pool import ThreadPool

from ...algorithm.base import AlgorithmBase
from .base import ProcessorBase, evaluate_particle


class ThreadsPool(ProcessorBase):
    def __init__(
        self,
        identifier: str,
        algorithm: AlgorithmBase,
        n_iter: int,
        n_particles: int,
        n_process: int | None = None,
        fitness_failure_strategy: str = "invalidate",
    ) -> None:

        self._stop_signal = Event()
        self._n_process = n_process
        super().__init__(
            identifier=identifier,
            algorithm=algorithm,
            n_iter=n_iter,
            n_particles=n_particles,
            fitness_failure_strategy=fitness_failure_strategy,
        )

    def run(self) -> None:
        self._stop_signal.clear()
        with ThreadPool(processes=self._n_process) as pool:
            self._status.n_process = pool._processes  # type: ignore
            for actual_iter in range(self._n_iter + 1):
                self.update_iter_counter(actual_iter)

                if self._stop_signal.is_set():
                    break

                self.wait_sync(actual_iter)

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
    #     sleep(0.25)
    #     result = ackley(x)
    #     for _ in range(5_000_001):
    #         result = result * 1.0000001
    #     result /= np.exp(1)
    #     return float(result)

    # algo = PSO(
    #     identifier="pso",
    #     fitness_function=test_function,
    #     boundaries={f"{n}": (-32.768, 32.768) for n in range(2)},
    #     seed=42,
    # )

    # obj = ThreadsPool(  # execution time 138s with 8 threads
    #     identifier="1",
    #     algorithm=algo,
    #     n_iter=30,
    #     n_particles=13,
    # )

    # start = perf_counter()
    # obj.run()
    # total_time = perf_counter() - start
    # print(f"{total_time=}")

    print("Breakpoint here")
