from ...algorithm.base import AlgorithmBase
from .base import ProcessorBase


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
    ) -> None:

        self._stop_signal = Event()
        super().__init__(
            identifier=identifier,
            algorithm=algorithm,
            n_iter=n_iter,
            n_particles=n_particles,
        )

    def run(self) -> None:
        for actual_iter in range(self._n_iter + 1):
            self.wait_sync(actual_iter)

            self.migration_control()

            self._algorithm.pre_iteration(actual_iter)

            processed_particles = []
            for particle_id in self._algorithm.population:
                processed_particles.append(self._algorithm.update_particle(particle_id))
            self._algorithm.update_population(processed_particles)

            self._algorithm.post_iteration(actual_iter)

            self.update_status()
