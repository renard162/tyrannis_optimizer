import numpy as np
import pytest

from tyrannis.space import Binary, Categorical, Continuous, Integer, Mixed

from tests._support.objectives import constant_objective


def test_mixed_aggregates_named_spaces_and_routes_decoding_to_cost_function():
    space = Mixed(
        {
            "position": Continuous((-2.0, 2.0)),
            "count": Integer((0, 4)),
        },
        cost_function=lambda position, count: position + count,
    )

    assert space.input_arguments == {
        "space": "mixed",
        "boundaries": [],
        "configs": {},
    }

    space.initialize_context(seed=11)

    assert space.variable_names == ["position", "count"]
    assert space.encoded_variable_names == ["position", "count"]
    assert space.encoded_boundaries == {
        "position": (-2.0, 2.0),
        "count": (0.0, 4.0),
    }
    assert space.decode({"position": 1.25, "count": 2.0}) == {
        "position": 1.25,
        "count": 2,
    }

    value = space({"position": 1.25, "count": 2.0})

    assert isinstance(value, np.floating)
    assert value == pytest.approx(3.25)


def test_mixed_groups_equivalent_spaces_without_losing_component_boundaries():
    space = Mixed(
        {
            "left": Continuous((-1.0, 1.0)),
            "right": Continuous((-3.0, 3.0)),
            "count": Integer((0, 2)),
        },
        cost_function=constant_objective,
    )

    space.initialize_context(seed=7)
    first_boundaries = space.encoded_boundaries
    space.initialize_context(seed=7)

    assert space.variable_names == ["left", "right", "count"]
    assert space.encoded_variable_names == ["left", "right", "count"]
    assert space.encoded_boundaries == {
        "left": (-1.0, 1.0),
        "right": (-3.0, 3.0),
        "count": (0.0, 2.0),
    }
    assert space.encoded_boundaries == first_boundaries
    assert space.decode({"left": -1.0, "right": 3.0, "count": 2.0}) == {
        "left": -1.0,
        "right": 3.0,
        "count": 2,
    }


def test_mixed_composes_binary_and_stochastic_categorical_representations():
    space = Mixed(
        {
            "enabled": Binary(),
            "color": Categorical(["red", "blue"], decoder="softmax"),
        },
        cost_function=constant_objective,
    )

    space.initialize_context(seed=23)

    assert space.variable_names == ["enabled", "color"]
    assert space.encoded_variable_names == ["enabled", "color-red", "color-blue"]
    assert space.encoded_boundaries == {
        "enabled": (-2.0, 2.0),
        "color-red": (-1.0, 1.0),
        "color-blue": (-1.0, 1.0),
    }

    decoded = space.decode(
        {"enabled": 0.0, "color-red": 1.0, "color-blue": 0.0}
    )

    assert isinstance(decoded["enabled"], bool)
    assert decoded["color"] in {"red", "blue"}
    assert set(decoded) == {"enabled", "color"}


def test_mixed_cache_codec_round_trips_each_component_value():
    space = Mixed(
        {
            "position": Continuous((-1.0, 1.0)),
            "count": Integer((0, 3)),
        },
        cost_function=constant_objective,
    )

    cache_key = space.encode_cache({"position": 0.5, "count": 2})

    assert space.decode_cache(cache_key) == {"position": 0.5, "count": 2}


def test_mixed_decode_rejects_missing_encoded_variable():
    space = Mixed(
        {
            "position": Continuous((-1.0, 1.0)),
            "count": Integer((0, 3)),
        },
        cost_function=constant_objective,
    )
    space.initialize_context(seed=3)

    with pytest.raises(KeyError, match="count"):
        space.decode({"position": 0.0})


def test_mixed_decode_rejects_unrecognized_encoded_variable():
    space = Mixed(
        {
            "position": Continuous((-1.0, 1.0)),
            "count": Integer((0, 3)),
        },
        cost_function=constant_objective,
    )
    space.initialize_context(seed=3)

    with pytest.raises(KeyError, match="unexpected"):
        space.decode({"position": 0.0, "count": 1.0, "unexpected": 0.0})


@pytest.mark.parametrize(
    ("spaces", "error"),
    [
        pytest.param([], TypeError, id="non-mapping"),
        pytest.param({"position": object()}, TypeError, id="non-space-component"),
    ],
)
def test_mixed_requires_a_mapping_of_space_components(spaces, error):
    with pytest.raises(error):
        Mixed(spaces, cost_function=constant_objective)


def test_mixed_rejects_nested_mixed_space():
    nested_space = Mixed(
        {"position": Continuous((-1.0, 1.0))},
        cost_function=constant_objective,
    )

    with pytest.raises(ValueError, match="cannot contain another Mixed"):
        Mixed({"nested": nested_space}, cost_function=constant_objective)
