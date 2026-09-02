from ..base import BackendBase


class ParallelBackendBase(BackendBase):
    """Base class for parallel optimization backends."""

    def init_particles(self) -> None:
        if self._algorithm.population:
            return

        for p_idx in range(self._n_particles):
            self._algorithm.create_particle(
                identifier=f"{self._identifier}|particle:{p_idx}",
            )

    def update_result(self) -> None:
        if self._algorithm.local_best is None:
            return
        self._result = self._algorithm.local_best.get_result_data()
