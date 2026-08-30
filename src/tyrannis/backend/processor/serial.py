from functools import partial

from .base import (
    LocalEvent,
    ProcessorBase,
    evaluate_particle,
)


class Serial(ProcessorBase):
    def __init__(self) -> None:
        return

    def initialize_execution_context(self) -> None:
        self._stop_signal = LocalEvent()
        self._wait_signal = LocalEvent()

    def finalize_execution_context(self) -> None:
        return

    def run(self) -> None:
        self.init_particles()
        self._stop_signal.clear()
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

                new_particles = [
                    worker(particle_id) for particle_id in new_particles_ids
                ]
                self._algorithm.update_population(new_particles)

            if actual_iter > 0:
                worker = partial(
                    evaluate_particle,
                    stop_signal=self._stop_signal,
                    algorithm=self._algorithm,
                    fitness_failure_strategy=self._fitness_failure_strategy,
                    initialize_particle=False,
                )

                processed_particles = [
                    worker(particle_id) for particle_id in self._algorithm.population
                ]
                self._algorithm.update_population(processed_particles)

            self._algorithm.post_iteration(actual_iter)

            self.update_status()
