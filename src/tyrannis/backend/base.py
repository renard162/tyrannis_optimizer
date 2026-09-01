from abc import ABC
from collections.abc import Callable
from typing import Any


class CostFunctionWrapperBase(ABC):
    """Neutral wrapper for a cost function."""

    def __init__(self, function: Callable[..., float]) -> None:
        self._function = function

    def __call__(self, *args: Any, **kwargs: Any) -> float:
        function = self._function
        return function(*args, **kwargs)
