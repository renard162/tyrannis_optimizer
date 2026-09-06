import json

from ..backend.distributed.communication.no_communication import (
    NoCommunicationDriver,
    NoCommunicationProcessor,
)
from ..core.algorithm import CostFunctionWrapperBase
from ..core.backend import BackendBase
from ..core.signals import LocalEvent


class LocalCostFunctionWrapper(CostFunctionWrapperBase):
    """Spark parallel cost-function wrapper."""


class Local(BackendBase):
    def __init__(self) -> None:
        self._cost_function_wrapper = LocalCostFunctionWrapper
        self._identifier = "Local"
        self._global_best_data: str | None = None

    def initialize_context(
        self,
        algorithm,
        n_iter: int,
        n_particles: int,
        migration,
        processor=None,
        fitness_failure_strategy: str = "invalidate",
        seed: int | None = None,
    ) -> None:
        super().initialize_context(
            algorithm=algorithm,
            n_iter=n_iter,
            n_particles=n_particles,
            migration=migration,
            processor=processor,
            fitness_failure_strategy=fitness_failure_strategy,
            seed=seed,
        )

        migration.initialize_context(
            communication_driver=NoCommunicationDriver(
                island_ids=["island:0"],
                stop_signal=LocalEvent(),
            ),
            communication_processor_class=NoCommunicationProcessor,
            communication_processor_kargs={},
        )

    def execute(self) -> None:
        if self._processor is None:
            raise RuntimeError("Processor cannot be None")

        self._processor.create_processors_pool(1)
        executor = next(iter(self._processor.processors_pool.values()))

        executor.initialize_execution_context()

        try:
            executor.run()
            self._global_best_data = executor.local_best
            self._processor.update_processors_pool([executor])
            self.update_result()

        finally:
            executor.finalize_execution_context()

    def update_result(self) -> None:
        if self._global_best_data is None:
            return

        particle_data = json.loads(self._global_best_data)

        self._result = {
            key: value
            for key, value in particle_data.items()
            if key in self._result_keys
        }
