import inspect

from doubles.algorithm import DummyCostFunctionWrapper

from tyrannis.core.algorithm import CostFunctionWrapperBase


def test_call_forwards_arguments_to_function() -> None:
    def cost_function(x: float, y: float) -> float:
        return x**2 + y

    wrapper = DummyCostFunctionWrapper(cost_function)

    assert wrapper(3.0, 4.0) == 13.0
