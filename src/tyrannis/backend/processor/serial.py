from functools import partial

from ...algorithm.base import AlgorithmBase
from .base import ProcessorBase, evaluate_particle


class Event:
    def __init__(self) -> None:
        self._state: bool = False

    def set(self) -> None:
        self._state = True

    def clear(self) -> None:
        self._state = False

    def is_set(self) -> bool:
        return self._state


class Serial(ProcessorBase):
    def __init__(
        self,
        identifier: str,
        algorithm: AlgorithmBase,
        n_iter: int,
        n_particles: int,
        fitness_failure_strategy: str = "invalidate",
    ) -> None:

        self._stop_signal = Event()
        super().__init__(
            identifier=identifier,
            algorithm=algorithm,
            n_iter=n_iter,
            n_particles=n_particles,
            fitness_failure_strategy=fitness_failure_strategy,
        )

    def run(self) -> None:
        self._stop_signal.clear()
        for actual_iter in range(self._n_iter + 1):
            self.update_iter_counter(actual_iter)

            if self._stop_signal.is_set():
                break

            self.wait_sync(actual_iter)

            self.migration_control()

            self._algorithm.pre_iteration(actual_iter)

            worker = partial(
                evaluate_particle,
                stop_signal=self._stop_signal,
                algorithm=self._algorithm,
                fitness_failure_strategy=self._fitness_failure_strategy,
            )
            processed_particles = [
                worker(particle_id) for particle_id in self._algorithm.population
            ]
            self._algorithm.update_population(processed_particles)

            self._algorithm.post_iteration(actual_iter)

            self.update_status()


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

    obj = Serial(  # execution time 238s
        identifier="1",
        algorithm=algo,
        n_iter=5,
        n_particles=13,
    )

    start = perf_counter()
    obj.run()
    total_time = perf_counter() - start
    print(f"{total_time=}")

    print("Breakpoint here")
