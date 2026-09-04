import numpy as np

from .backend import BackendBase


class DistributedBackendBase(BackendBase):
    """Base class for distributed optimization backends."""

    _local_bests: dict[str, dict[str, float | dict[str, float]] | None]
    _n_executors: int

    def init_processors(self) -> None:
        if self._processor is None:
            raise RuntimeError("Processor is not initialized.")

        self._processor.create_processors_pool(
            self._n_executors,
        )

    def update_result(self) -> None:
        best_candidate = None
        best_fitness = np.inf

        for candidate in self._local_bests.values():
            if candidate is None:
                continue

            fitness = candidate["fitness"]

            if not isinstance(fitness, float):
                continue

            if fitness < best_fitness:
                best_fitness = fitness
                best_candidate = candidate

        if best_candidate is None:
            return

        self._result = {
            key: value
            for key, value in best_candidate.items()
            if key in self._result_keys
        }
