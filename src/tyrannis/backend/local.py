import json

from ..core.algorithm import CostFunctionWrapperBase
from ..core.backend import BackendBase


class LocalCostFunctionWrapper(CostFunctionWrapperBase):
    """Spark parallel cost-function wrapper."""


class Local(BackendBase):
    def __init__(self) -> None:
        self._cost_function_wrapper = LocalCostFunctionWrapper
        self._identifier = "Local"
        self._global_best_data = None

    def execute(self) -> None:
        if self._processor is None:
            raise RuntimeError("Processor cannot be None")

        self._processor.create_processors_pool(1)
        executor = next(iter(self._processor.processors_pool.values()))
        executor.initialize_execution_context()
        executor.run()
        self._global_best_data = executor.local_best
        self._processor.update_processors_pool([executor])
        self.update_result()

    def update_result(self) -> None:
        if self._global_best_data is None:
            return
        particle_data = json.loads(self._global_best_data)
        self._result = {
            key: value
            for key, value in particle_data.items()
            if key in self._result_keys
        }
