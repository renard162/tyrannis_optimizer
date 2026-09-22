import numpy as np
import pytest

from tests._support.objectives import CountingObjective
from tyrannis.space import Continuous


def test_named_initialization_exposes_encoded_metadata_and_isolation() -> None:
    boundaries = {"x": (-1.0, 1.0), "y": (0.0, 2.0)}
    space = Continuous(boundaries)

    space.initialize_context()

    assert space.variable_names == ["x", "y"]
    assert space.encoded_variable_names == ["x", "y"]
    assert space.encoded_boundaries == boundaries
    assert space.input_arguments == {
        "space": "continuous",
        "boundaries": boundaries,
        "configs": {},
    }

    exposed_boundaries = space.encoded_boundaries
    exposed_boundaries["x"] = (-99.0, 99.0)
    assert space.encoded_boundaries == boundaries


def test_positional_single_interval_preserves_representation_for_decode_cache_and_call() -> (
    None
):
    def shift(value: float) -> float:
        return value + 1.0

    space = Continuous(
        (-2.0, 3.0),
        cost_function=shift,
    )
    space.initialize_context()

    encoded = {"0": -2.0}

    assert space.variable_names == ["0"]
    assert space.encoded_boundaries == {"0": (-2.0, 3.0)}
    assert space.decode(encoded) == [-2.0]
    assert space.decode_cache(space.encode_cache([-2.0])) == [-2.0]

    result = space(encoded)
    assert result.dtype == np.dtype("float64")
    assert result == -1.0


def test_named_decode_preserves_mapping_and_accepts_both_boundary_endpoints() -> None:
    space = Continuous({"left": (-2.0, 2.0), "right": (-3.0, 3.0)})
    space.initialize_context()

    decoded = space.decode({"left": -2.0, "right": 3.0})

    assert isinstance(decoded, dict)
    assert decoded == {"left": -2.0, "right": 3.0}
    assert all(isinstance(value, float) for value in decoded.values())


def test_positional_intervals_keep_order_and_dimension() -> None:
    space = Continuous([(-2.0, -1.0), (3.0, 4.0)])
    space.initialize_context()

    assert space.variable_names == ["0", "1"]
    assert space.encoded_variable_names == ["0", "1"]
    assert space.encoded_boundaries == {"0": (-2.0, -1.0), "1": (3.0, 4.0)}
    assert space.decode({"0": -2.0, "1": 4.0}) == [-2.0, 4.0]


def test_named_cache_key_is_canonical_and_cache_is_local_to_each_space(
    counting_objective: CountingObjective,
) -> None:
    first = Continuous(
        {"x": (-1.0, 1.0), "y": (-1.0, 1.0)},
        cost_function=counting_objective,
        use_cache=True,
    )
    second = Continuous(
        {"x": (-1.0, 1.0), "y": (-1.0, 1.0)},
        cost_function=counting_objective,
        use_cache=True,
    )
    first.initialize_context()
    second.initialize_context()

    assert first.encode_cache({"y": 0.5, "x": -0.5}) == (
        ("x", -0.5),
        ("y", 0.5),
    )
    assert first.decode_cache(first.encode_cache({"y": 0.5, "x": -0.5})) == {
        "x": -0.5,
        "y": 0.5,
    }
    assert first({"x": -0.5, "y": 0.5}) == 1.0
    assert first({"y": 0.5, "x": -0.5}) == 1.0
    assert counting_objective.calls == 1
    assert second({"x": -0.5, "y": 0.5}) == 1.0
    assert counting_objective.calls == 2


@pytest.mark.parametrize(
    ("inputs", "exception"),
    [
        ({"x": 1.1}, ValueError),
        ({"unknown": 0.0}, KeyError),
    ],
    ids=["outside-boundary", "unknown-variable"],
)
def test_decode_rejects_invalid_encoded_inputs(
    inputs: dict[str, float], exception: type[Exception]
) -> None:
    space = Continuous({"x": (-1.0, 1.0)})
    space.initialize_context()

    with pytest.raises(exception):
        _ = space.decode(inputs)


def test_constructor_requires_lower_boundary_to_be_smaller() -> None:
    with pytest.raises(ValueError):
        _ = Continuous((2.0, 1.0))
