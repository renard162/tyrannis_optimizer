"""Contracts implemented by SpaceBase independently of concrete spaces."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, final, override

import numpy as np
import pytest

from tests._support.objectives import CountingObjective, sphere
from tyrannis.core.space import SpaceBase

Decoded = tuple[float, ...] | dict[str, float]


class MetadataView(Protocol):
    @property
    def input_arguments(self) -> dict[str, object]: ...


def _metadata(space: MetadataView) -> dict[str, object]:
    return space.input_arguments


@final
class SimpleSpace(SpaceBase):
    """A reversible tuple or mapping representation with no domain conversion."""

    def __init__(
        self,
        cost_function: Callable[..., float] | None = sphere,
        *,
        is_kwargs: bool = False,
        use_cache: bool = False,
        cache_type: str = "lru",
        cache_size: int = 8,
    ) -> None:
        super().__init__(cost_function, use_cache, cache_type, cache_size)
        self._is_kwargs = is_kwargs
        self._type = "simple"
        self._boundaries: dict[str, tuple[float, float]] = {
            "x": (-1.0, 1.0),
            "y": (0.0, 2.0),
        }
        self._configs = {"is_kwargs": is_kwargs}
        self._args = ()
        self._kwargs = {}

    @override
    def initialize_context(self, seed: int | None = None) -> None:
        del seed
        self._variable_names = ["x", "y"]
        self._encoded_boundaries = self._boundaries.copy()

    @override
    def decode(self, float_inputs: dict[str, float]) -> Decoded:
        self._check_input_bounds(float_inputs)
        if self._is_kwargs:
            return dict(float_inputs)
        return tuple(float_inputs[name] for name in self._variable_names)

    @override
    def encode_cache(self, inputs: Decoded) -> tuple[float, ...]:
        if isinstance(inputs, dict):
            return tuple(inputs[name] for name in self._variable_names)
        return inputs

    @override
    def decode_cache(self, inputs: tuple[float, ...]) -> Decoded:
        if self._is_kwargs:
            return dict(zip(self._variable_names, inputs, strict=True))
        return inputs


def test_disabled_cache_ignores_inert_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_dependency_lookup(_name: str) -> None:
        pytest.fail("disabled cache inspected an optional dependency")

    monkeypatch.setattr("tyrannis.core.space.find_spec", unexpected_dependency_lookup)
    space = SimpleSpace(use_cache=False, cache_type="disk", cache_size=0)
    space.initialize_context()

    assert space({"x": 0.0, "y": 1.0}) == 1.0


@pytest.mark.parametrize(
    ("cache_type", "cache_size", "exception", "message"),
    [
        (1, 8, TypeError, "cache_type must be a string"),
        ("unknown", 8, ValueError, "Invalid cache_type"),
        ("lru", 1.5, TypeError, "cache_size must be an integer"),
        ("lru", True, TypeError, "cache_size must be an integer"),
        ("lru", 0, ValueError, "cache_size must be greater than zero"),
    ],
    ids=[
        "non-string-type",
        "unknown-type",
        "non-integer-size",
        "boolean-size",
        "non-positive-size",
    ],
)
def test_enabled_cache_validates_configuration(
    cache_type: str,
    cache_size: int,
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        _ = SimpleSpace(use_cache=True, cache_type=cache_type, cache_size=cache_size)


def test_common_metadata_and_boundary_copy() -> None:
    space = SimpleSpace(is_kwargs=True)
    space.initialize_context()

    assert space.variable_names == ["x", "y"]
    assert space.encoded_variable_names == ["x", "y"]
    assert _metadata(space) == {
        "space": "simple",
        "boundaries": {"x": (-1.0, 1.0), "y": (0.0, 2.0)},
        "configs": {"is_kwargs": True},
    }
    exposed = space.encoded_boundaries
    exposed["x"] = (-100.0, 100.0)
    assert space.encoded_boundaries["x"] == (-1.0, 1.0)


@pytest.mark.parametrize(
    ("inputs", "exception"),
    [
        ({"missing": 0.0}, KeyError),
        ({"x": -1.1}, ValueError),
        ({"x": 1.1}, ValueError),
    ],
    ids=["unknown-variable", "below-lower-bound", "above-upper-bound"],
)
def test_common_bounds_reject_invalid_inputs(
    inputs: dict[str, float], exception: type[Exception]
) -> None:
    space = SimpleSpace()
    space.initialize_context()

    with pytest.raises(exception):
        _ = space.decode(inputs)


def test_common_bounds_accept_endpoints() -> None:
    space = SimpleSpace()
    space.initialize_context()

    assert space.decode({"x": -1.0, "y": 2.0}) == (-1.0, 2.0)


def test_enabled_cache_reports_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_dependency(_name: str) -> None:
        return None

    monkeypatch.setattr("tyrannis.core.space.find_spec", missing_dependency)

    with pytest.raises(ImportError, match="cachetools"):
        _ = SimpleSpace(use_cache=True, cache_type="lfu")


@pytest.mark.parametrize("is_kwargs", [False, True], ids=["positional", "keyword"])
def test_call_decodes_and_normalizes_fitness(is_kwargs: bool) -> None:
    space = SimpleSpace(is_kwargs=is_kwargs)
    space.initialize_context()

    result = space({"x": -1.0, "y": 2.0})

    assert type(result) is np.float64
    assert result == 5.0


def test_call_requires_cost_function() -> None:
    space = SimpleSpace(cost_function=None)

    with pytest.raises(ValueError, match="cost_function cannot be None"):
        _ = space({"x": 0.0, "y": 1.0})


def test_lru_cache_reuses_same_key_and_evaluates_new_key(
    counting_objective: CountingObjective,
) -> None:
    space = SimpleSpace(cost_function=counting_objective, use_cache=True)
    space.initialize_context()

    assert space({"x": 0.0, "y": 1.0}) == 1.0
    assert counting_objective.calls == 1
    assert space({"x": 0.0, "y": 1.0}) == 1.0
    assert counting_objective.calls == 1
    assert space({"x": 1.0, "y": 1.0}) == 1.0
    assert counting_objective.calls == 2


@pytest.mark.parametrize(
    ("cache_type", "dependency"),
    [
        pytest.param("lfu", "cachetools", marks=pytest.mark.optional, id="lfu"),
        pytest.param("fifo", "cachetools", marks=pytest.mark.optional, id="fifo"),
        pytest.param("rr", "cachetools", marks=pytest.mark.optional, id="rr"),
        pytest.param("disk", None, id="disk"),
    ],
)
def test_configured_cache_reuses_same_key_and_evaluates_new_key(
    cache_type: str, dependency: str | None, counting_objective: CountingObjective
) -> None:
    if dependency is not None:
        pytest.importorskip(dependency)
    space = SimpleSpace(
        cost_function=counting_objective, use_cache=True, cache_type=cache_type
    )
    space.initialize_context()

    assert space({"x": 0.0, "y": 1.0}) == 1.0
    assert counting_objective.calls == 1
    assert space({"x": 0.0, "y": 1.0}) == 1.0
    assert counting_objective.calls == 1
    assert space({"x": 1.0, "y": 1.0}) == 1.0
    assert counting_objective.calls == 2


def test_serialized_state_excludes_runtime_cache(
    counting_objective: CountingObjective,
) -> None:
    space = SimpleSpace(cost_function=counting_objective, use_cache=True)
    space.initialize_context()
    _ = space({"x": 0.0, "y": 1.0})

    state = space.__getstate__()

    assert state == {**vars(space), "_cached_cost_function": None}
    assert vars(space)["_cached_cost_function"] is not None
