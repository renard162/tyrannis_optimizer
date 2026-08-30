from time import perf_counter, sleep

import numpy as np

from .algorithm.pso import PSO
from .backend.processor import ProcessPool, Serial, ThreadsPool
from .examples.many_local_minima import ackley


def test_function(x):
    result = ackley(x)
    # for _ in range(5_000_001):
    #     result = result * 1.0000001
    # result /= np.exp(1)
    # sleep(0.25)
    return float(result)


def main():
    algo = PSO()
    algo.initialize_context(
        fitness_function=test_function,
        boundaries={f"{n}": (-32.768, 32.768) for n in range(2)},
    )

    # processor = Serial()
    # processor = ThreadsPool()
    processor = ProcessPool()
    processor.initialize_context(
        algorithm=algo,
        n_iter=30,
        n_particles=13,
        seed=42,
    )
    processor.create_processors_pool(1)

    start = perf_counter()
    processor.processors_pool[0].run()
    total_time = perf_counter() - start

    best_particle = processor.processors_pool[0]._status.best_particle_data

    print(f"{total_time=}\n{best_particle}")
    print("Breakpoint here")


if __name__ == "__main__":
    main()
