from ..backend.distributed.communication.no_communication import (
    NoCommunicationDriver,
    NoCommunicationProcessor,
)
from ..core.algorithm import AlgorithmBase, CostFunctionWrapperBase
from ..core.backend import BackendBase
from ..core.backend_migration import MigrationDriverBase
from ..core.processor import ProcessorBase
from ..core.results import HistoryConfig


class LocalCostFunctionWrapper(CostFunctionWrapperBase):
    """Local cost-function wrapper."""


class Local(BackendBase):
    def __init__(self) -> None:
        self._cost_function_wrapper = LocalCostFunctionWrapper
        self._identifier = "Local"

    def initialize_context(
        self,
        algorithm: AlgorithmBase,
        n_iter: int,
        n_particles: int,
        migration: MigrationDriverBase,
        processor: ProcessorBase | None,
        fitness_failure_strategy: str,
        history_config: HistoryConfig,
        seed: int | None,
    ) -> None:
        super().initialize_context(
            algorithm=algorithm,
            n_iter=n_iter,
            n_particles=n_particles,
            migration=migration,
            processor=processor,
            fitness_failure_strategy=fitness_failure_strategy,
            seed=seed,
            history_config=history_config,
        )

        migration.initialize_context(
            communication_driver=NoCommunicationDriver(
                island_ids=["island:0"],
            ),
            communication_processor_class=NoCommunicationProcessor,
            communication_processor_kargs={},
            history_config=history_config,
        )

    def execute(self) -> None:
        if self._processor is None:
            raise RuntimeError("Processor cannot be None")

        if len(self._processor.processors_pool) == 0:
            self._processor.create_processors_pool(1)
        executor = next(iter(self._processor.processors_pool.values()))

        executor.initialize_execution_context()

        try:
            executor.run()
            self._processor.update_processors_pool([executor])
            self._result = executor.result

        finally:
            executor.finalize_execution_context()
