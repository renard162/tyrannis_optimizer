from typing import Any

from tyrannis.core.algorithm import CostFunctionWrapperBase
from tyrannis.core.backend import BackendBase


class DummyCostFunctionWrapper(CostFunctionWrapperBase):
    pass


class DummyBackend(BackendBase):
    _identifier = "DummyBackend"
    _cost_function_wrapper = DummyCostFunctionWrapper

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def execute(self) -> None:
        pass
