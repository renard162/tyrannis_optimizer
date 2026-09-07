from collections.abc import Callable
from typing import Any

import pytest

from tyrannis.core.space import SpaceBase


class DummySpace(SpaceBase):
    """Concrete implementation used to test the SpaceBase contract."""

    def __init__(
        self,
        cost_function: Callable[..., float] | None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._cost_function = cost_function
        self._args = args
        self._kwargs = kwargs

    def initialize_context(self, seed: int | None) -> None:  # type: ignore
        raise NotImplementedError

    def decode(self, float_inputs: dict[str, float]) -> Any:
        raise NotImplementedError

    @property
    def is_kargs(self) -> bool:
        raise NotImplementedError


class PositionalCallableDummySpace(SpaceBase):
    """Concrete implementation used to test positional cost-function calls."""

    def __init__(
        self,
        cost_function: Callable[..., float] | None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._cost_function = cost_function
        self._args = args
        self._kwargs = kwargs

    def initialize_context(self, seed: int | None) -> None:  # type: ignore
        pass

    def decode(self, float_inputs: dict[str, float]) -> list[float]:
        return [
            float_inputs["x"] * 2.0,
            float_inputs["y"] * 2.0,
        ]

    @property
    def is_kargs(self) -> bool:
        return False


class KeywordCallableDummySpace(SpaceBase):
    """Concrete implementation used to test keyword cost-function calls."""

    def __init__(
        self,
        cost_function: Callable[..., float] | None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._cost_function = cost_function
        self._args = args
        self._kwargs = kwargs

    def initialize_context(self, seed: int | None) -> None:  # type: ignore
        pass

    def decode(self, float_inputs: dict[str, float]) -> dict[str, float]:
        return {key: value * 2.0 for key, value in float_inputs.items()}

    @property
    def is_kargs(self) -> bool:
        return True


def test_space_base_is_abstract() -> None:
    with pytest.raises(TypeError):
        SpaceBase(None)  # type: ignore


def test_init_is_abstract() -> None:
    class IncompleteSpace(SpaceBase):
        def initialize_context(self, seed: int | None) -> None:  # type: ignore
            pass

        def decode(self, float_inputs: dict[str, float]) -> Any:
            return float_inputs

        @property
        def is_kargs(self) -> bool:
            return False

    with pytest.raises(TypeError):
        IncompleteSpace(None)  # type: ignore


def test_initialize_context_is_abstract() -> None:
    space = DummySpace(None)

    with pytest.raises(NotImplementedError):
        space.initialize_context(42)


def test_decode_is_abstract() -> None:
    space = DummySpace(None)

    with pytest.raises(NotImplementedError):
        space.decode({"0": 1.0})


def test_is_kargs_is_abstract() -> None:
    space = DummySpace(None)

    with pytest.raises(NotImplementedError):
        space.is_kargs  # noqa: B018


def test_call_decodes_inputs_before_evaluating_positional_cost_function() -> None:
    received_inputs: list[tuple[float, float]] = []

    def cost_function(x: float, y: float) -> float:
        received_inputs.append((x, y))

        return x + y

    space = PositionalCallableDummySpace(cost_function)

    result = space({"x": 2.0, "y": 3.0})

    assert result == 10.0
    assert received_inputs == [(4.0, 6.0)]


def test_call_decodes_inputs_before_evaluating_keyword_cost_function() -> None:
    received_inputs: list[dict[str, float]] = []

    def cost_function(**inputs: float) -> float:
        received_inputs.append(inputs)

        return sum(inputs.values())

    space = KeywordCallableDummySpace(cost_function)

    result = space({"x": 2.0, "y": 3.0})

    assert result == 10.0
    assert received_inputs == [
        {
            "x": 4.0,
            "y": 6.0,
        }
    ]


def test_call_returns_positional_cost_function_result() -> None:
    def cost_function(x: float) -> float:
        return x**2

    class SingleInputPositionalSpace(PositionalCallableDummySpace):
        def decode(self, float_inputs: dict[str, float]) -> list[float]:
            return [float_inputs["x"] * 2.0]

    space = SingleInputPositionalSpace(cost_function)

    result = space({"x": 3.0})

    assert result == 36.0


def test_call_returns_keyword_cost_function_result() -> None:
    def cost_function(x: float) -> float:
        return x**2

    class SingleInputKeywordSpace(KeywordCallableDummySpace):
        def decode(self, float_inputs: dict[str, float]) -> dict[str, float]:
            return {"x": float_inputs["x"] * 2.0}

    space = SingleInputKeywordSpace(cost_function)

    result = space({"x": 3.0})

    assert result == 36.0


def test_call_rejects_none_cost_function() -> None:
    space = PositionalCallableDummySpace(None)

    with pytest.raises(
        ValueError,
        match="cost_function cannot be None",
    ):
        space({"x": 1.0})
