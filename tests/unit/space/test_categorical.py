import warnings

import numpy as np
import pytest

from _support.numerics import seed_for
from tyrannis.space import Categorical


def test_named_one_hot_initialization_exposes_one_encoded_value_per_choice() -> None:
    space = Categorical(
        {"color": ["red", "blue"], "size": ("small", "large")},
        decoder="one-hot",
    )

    space.initialize_context(seed=seed_for(301))

    assert space.variable_names == ["color", "size"]
    assert space.encoded_variable_names == [
        "color-red",
        "color-blue",
        "size-small",
        "size-large",
    ]
    assert space.encoded_boundaries == {
        "color-red": (-1.0, 1.0),
        "color-blue": (-1.0, 1.0),
        "size-small": (-1.0, 1.0),
        "size-large": (-1.0, 1.0),
    }


def test_scalar_initialization_uses_one_encoded_value_per_variable_and_custom_bounds() -> None:
    space = Categorical(
        [["small", "large"], ["red", "blue", "green"]],
        decoder="scalar",
        bounds=(-2.0, 2.0),
    )

    space.initialize_context(seed=seed_for(302))

    assert space.variable_names == ["0", "1"]
    assert space.encoded_variable_names == ["0", "1"]
    assert space.encoded_boundaries == {"0": (-2.0, 2.0), "1": (-2.0, 2.0)}
    assert space.decode({"0": -2.0, "1": 2.0}) == ["small", "green"]


def test_one_hot_decode_returns_named_choices_from_the_configured_domain() -> None:
    choices = {"color": ["red", "blue"], "size": ["small", "large"]}
    space = Categorical(choices, decoder="one-hot")
    space.initialize_context(seed=seed_for(303))

    decoded = space.decode(
        {
            "color-red": -1.0,
            "color-blue": 1.0,
            "size-small": 1.0,
            "size-large": -1.0,
        }
    )

    assert isinstance(decoded, dict)
    assert decoded == {"color": "blue", "size": "small"}
    assert decoded["color"] in choices["color"]
    assert decoded["size"] in choices["size"]


@pytest.mark.parametrize(
    ("choices", "inputs"),
    [
        (["red", "blue"], ["blue"]),
        ({"color": ["red", "blue"]}, {"color": "blue"}),
    ],
    ids=["positional", "keyword"],
)
def test_cache_codec_preserves_categorical_input_representation(
    choices: list[str] | dict[str, list[str]], inputs: list[str] | dict[str, str]
) -> None:
    space = Categorical(choices)
    space.initialize_context(seed=seed_for(304))

    assert space.decode_cache(space.encode_cache(inputs)) == inputs


@pytest.mark.parametrize(
    "decoder",
    ["softmax", "gumbel-softmax"],
    ids=["softmax", "gumbel-softmax"],
)
def test_stochastic_decoders_warn_without_seed_and_replay_with_same_seed(
    decoder: str,
) -> None:
    unseeded = Categorical(["red", "blue"], decoder=decoder)
    with pytest.warns(RuntimeWarning, match="without a seed"):
        unseeded.initialize_context()

    inputs = {"0-red": -0.5, "0-blue": 0.5}
    first = Categorical(["red", "blue"], decoder=decoder, params={"temperature": 0.5})
    second = Categorical(["red", "blue"], decoder=decoder, params={"temperature": 0.5})
    first.initialize_context(seed=seed_for(305))
    second.initialize_context(seed=seed_for(305))

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert [first.decode(inputs) for _ in range(3)] == [
            second.decode(inputs) for _ in range(3)
        ]


@pytest.mark.parametrize(
    ("inputs", "exception"),
    [
        ({"0-red": 1.1, "0-blue": 0.0}, ValueError),
        ({"unknown": 0.0}, KeyError),
    ],
    ids=["outside-encoded-boundary", "unknown-variable"],
)
def test_decode_rejects_invalid_encoded_inputs(
    inputs: dict[str, float], exception: type[Exception]
) -> None:
    space = Categorical(["red", "blue"])
    space.initialize_context(seed=seed_for(306))

    with pytest.raises(exception):
        space.decode(inputs)


@pytest.mark.parametrize(
    ("choices", "exception"),
    [
        ([], ValueError),
        ({"color": []}, ValueError),
        ({"color": 3}, TypeError),
        (np.asarray([[["red", "blue"]]]), ValueError),
    ],
    ids=["empty", "empty-named", "non-iterable-variable", "three-dimensional-array"],
)
def test_constructor_rejects_invalid_choice_collections(
    choices: object, exception: type[Exception]
) -> None:
    with pytest.raises(exception):
        Categorical(choices)  # type: ignore[arg-type]


def test_constructor_accepts_one_dimensional_numpy_choices() -> None:
    space = Categorical(np.asarray(["red", "blue"]))

    space.initialize_context(seed=seed_for(307))

    assert space.variable_names == ["0"]
    assert space.encoded_variable_names == ["0-red", "0-blue"]


@pytest.mark.parametrize(
    ("bounds", "exception"),
    [
        ((0.0,), ValueError),
        ((1.0, 1.0), ValueError),
    ],
    ids=["wrong-dimension", "equal-endpoints"],
)
def test_constructor_rejects_invalid_bounds(
    bounds: object, exception: type[Exception]
) -> None:
    with pytest.raises(exception):
        Categorical(["red"], bounds=bounds)  # type: ignore[arg-type]


def test_gumbel_softmax_rejects_temperature_outside_documented_domain() -> None:
    with pytest.raises(Exception):
        Categorical(
            ["red", "blue"],
            decoder="gumbel-softmax",
            params={"temperature": 0.0},
        )


def test_constructor_preserves_all_choices_from_a_documented_iterable() -> None:
    space = Categorical(iter(["red", "blue"]))

    space.initialize_context(seed=seed_for(308))

    assert space.encoded_variable_names == ["0-red", "0-blue"]


def test_encoded_representation_distinguishes_choices_with_equal_string_forms() -> None:
    space = Categorical([1, "1"])

    space.initialize_context(seed=seed_for(309))

    assert len(space.encoded_variable_names) == 2


def test_constructor_rejects_unknown_decoder() -> None:
    with pytest.raises(ValueError, match="Invalid decoder"):
        Categorical(["red"], decoder="invalid")
