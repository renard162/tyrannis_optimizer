from ...algorithm.base import ParticleBase
from ..base import BackendBase


class DistributedBackendBase(BackendBase):
    """Base class for distributed optimization backends."""

    _local_bests: dict[str, float | dict[str, float]]
    _n_executors: int

    def init_processors(self) -> None:
        if self._processor is None:
            raise RuntimeError("Processor is not initialized.")

        self._processor.create_processors_pool(
            self._n_executors,
        )

    def update_result(self) -> None:
        return
