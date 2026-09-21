import numpy as np
import pytest

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


def test_positional_single_interval_preserves_representation_for_decode_cache_and_call() -> None:
    space = Continuous(
        (-2.0, 3.0),
        cost_function=lambda value: value + 1.0,
    )
    space.initialize_context()

    encoded = {"0": -2.0}

    assert space.variable_names == ["0"]
    assert space.encoded_boundaries == {"0": (-2.0, 3.0)}
    assert space.decode(encoded) == [-2.0]
    assert space.decode_cache(space.encode_cache([-2.0])) == [-2.0]

    result = space(encoded)
    assert isinstance(result, np.float64)
    assert result == -1.0


def test_named_decode_preserves_mapping_and_accepts_both_boundary_endpoints() -> None:
    space = Continuous({"left": (-2.0, 2.0), "right": (-3.0, 3.0)})
    space.initialize_context()

    decoded = space.decode({"left": -2.0, "right": 3.0})

    assert isinstance(decoded, dict)
    assert decoded == {"left": -2.0, "right": 3.0}
    assert all(isinstance(value, float) for value in decoded.values())


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
        space.decode(inputs)


@pytest.mark.parametrize(
    "boundaries",
    [(1.0, 1.0), (2.0, 1.0)],
    ids=["equal-endpoints", "reversed-endpoints"],
)
def test_constructor_rejects_non_increasing_boundaries(
    boundaries: tuple[float, float],
) -> None:
    with pytest.raises(ValueError):
        Continuous(boundaries)
