from functools import partial
from multiprocessing import Event, Manager, Pool
from typing import Protocol

from ...algorithm.base import AlgorithmBase, ParticleBase
from .base import ProcessorBase


class StopSignal(Protocol):
    """Just to use in _update_particle signature"""

    def is_set(self) -> bool: ...
    def set(self) -> None: ...
    def clear(self) -> None: ...


def _update_particle(
    particle_id: str,
    algorithm: AlgorithmBase,
    stop_signal: StopSignal,
) -> ParticleBase:
    if stop_signal.is_set():
        return algorithm.population[particle_id]
    return algorithm.update_particle(particle_id)


class ProcessPool(ProcessorBase):
    def __init__(
        self,
        identifier: str,
        algorithm: AlgorithmBase,
        n_iter: int,
        n_particles: int,
        n_process: int | None = None,
    ) -> None:

        # Just a dummy, the real event is created when run method is called
        self._stop_signal = Event()

        self._n_process = n_process
        super().__init__(
            identifier=identifier,
            algorithm=algorithm,
            n_iter=n_iter,
            n_particles=n_particles,
        )

    def run(self) -> None:
        with Manager() as manager:
            self._stop_signal = manager.Event()
            with Pool(processes=self._n_process) as pool:
                self._status.n_process = pool._processes  # type: ignore
                for actual_iter in range(self._n_iter + 1):
                    self.update_iter_counter(actual_iter)

                    if self._stop_signal.is_set():
                        break

                    self.wait_sync(actual_iter)

                    self.migration_control()

                    self._algorithm.pre_iteration(actual_iter)

                    worker = partial(
                        _update_particle,
                        stop_signal=self._stop_signal,
                        algorithm=self._algorithm,
                    )
                    processed_particles = pool.map(worker, self._algorithm.population)
                    self._algorithm.update_population(processed_particles)

                    self._algorithm.post_iteration(actual_iter)

                    self.update_status()

        self._stop_signal = Event()


if __name__ == "__main__":
    from time import perf_counter, sleep

    import numpy as np

    from ...algorithm.pso import PSO
    from ...examples.many_local_minima import ackley

    def test_function(x):
        sleep(0.25)
        result = ackley(x)
        for _ in range(5_000_001):
            result = result * 1.0000001
        result /= np.exp(1)
        return float(result)

    algo = PSO(
        identifier="pso",
        fitness_function=test_function,
        boundaries={f"{n}": (-32.768, 32.768) for n in range(2)},
        seed=42,
    )

    obj = ProcessPool(  # execution time 46s with 8 process
        identifier="1",
        algorithm=algo,
        n_iter=30,
        n_particles=13,
    )

    start = perf_counter()
    obj.run()
    total_time = perf_counter() - start
    print(f"{total_time=}")

    print("Breakpoint here")
