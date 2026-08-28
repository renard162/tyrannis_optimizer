from multiprocessing import Event
from multiprocessing.pool import ThreadPool

from ...algorithm.base import AlgorithmBase, ParticleBase
from .base import ProcessorBase


class ThreadsPool(ProcessorBase):
    def __init__(
        self,
        identifier: str,
        algorithm: AlgorithmBase,
        n_iter: int,
        n_particles: int,
        n_process: int | None = None,
    ) -> None:

        self._stop_signal = Event()
        self._n_process = n_process
        super().__init__(
            identifier=identifier,
            algorithm=algorithm,
            n_iter=n_iter,
            n_particles=n_particles,
        )

    def _update_particle(self, particle_id) -> ParticleBase:
        if self._stop_signal.is_set():
            return self._algorithm.population[particle_id]
        return self._algorithm.update_particle(particle_id)

    def run(self) -> None:
        for actual_iter in range(self._n_iter + 1):
            self.update_iter_counter(actual_iter)

            if self._stop_signal.is_set():
                break

            self.wait_sync(actual_iter)

            self.migration_control()

            self._algorithm.pre_iteration(actual_iter)

            with ThreadPool(processes=self._n_process) as pool:
                self._status.n_process = pool._processes  # type: ignore

                processed_particles = pool.imap_unordered(
                    self._update_particle, self._algorithm.population
                )

                self._algorithm.update_population(processed_particles)

            self._algorithm.post_iteration(actual_iter)

            self.update_status()


if __name__ == "__main__":
    from time import perf_counter, sleep

    from ...algorithm.pso import PSO
    from ...examples.many_local_minima import ackley

    def test_function(x):
        sleep(0.1)
        return ackley(x)

    algo = PSO(
        identifier="pso",
        fitness_function=test_function,
        boundaries={f"{n}": (-32.768, 32.768) for n in range(2)},
        seed=42,
    )

    obj = ThreadsPool(  # execution time 17s with 8 process
        identifier="1",
        algorithm=algo,
        n_iter=40,
        n_particles=25,
    )

    start = perf_counter()
    obj.run()
    total_time = perf_counter() - start
    print(f"{total_time=}")

    print("Breakpoint here")
