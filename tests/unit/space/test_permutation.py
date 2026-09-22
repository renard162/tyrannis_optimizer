import warnings

import numpy as np
import pytest
from _support.assertions import assert_valid_permutation
from _support.numerics import seed_for

from tyrannis.space import Permutation


def _encoded_inputs(space: Permutation) -> dict[str, float]:
    """Build valid neutral encoded values from the public encoded representation."""
    return {name: 0.0 for name in space.encoded_variable_names}


def test_named_random_keys_initialization_exposes_one_encoded_value_per_choice() -> (
    None
):
    space = Permutation({"route": ["A", "B", "C"]}, decoder="random-keys")

    space.initialize_context(seed=seed_for(401))

    assert space.variable_names == ["route"]
    assert space.encoded_variable_names == ["route-A", "route-B", "route-C"]
    assert space.encoded_boundaries == {
        "route-A": (0.0, 1.0),
        "route-B": (0.0, 1.0),
        "route-C": (0.0, 1.0),
    }


@pytest.mark.parametrize(
    ("decoder", "expected_dimension"),
    [
        ("random-keys", 3),
        ("gumbel-random-keys", 3),
        ("plackett-luce", 3),
        ("gumbel-sinkhorn", 9),
    ],
    ids=["random-keys", "gumbel-random-keys", "plackett-luce", "gumbel-sinkhorn"],
)
def test_decoder_initialization_exposes_expected_encoded_dimensionality(
    decoder: str, expected_dimension: int
) -> None:
    space = Permutation({"route": ["A", "B", "C"]}, decoder=decoder)

    space.initialize_context(seed=seed_for(402))

    assert space.variable_names == ["route"]
    assert len(space.encoded_variable_names) == expected_dimension
    assert len(space.encoded_boundaries) == expected_dimension
    assert set(space.encoded_boundaries.values()) == {(0.0, 1.0)}


def test_gumbel_sinkhorn_configuration_keeps_custom_bounds_and_parameters() -> None:
    space = Permutation(
        {"route": ["A", "B"]},
        decoder="gumbel-sinkhorn",
        bounds=(-2.0, 2.0),
        params={"temperature": 0.5, "sinkhorn_iterations": 7},
    )

    space.initialize_context(seed=seed_for(403))

    assert space.encoded_boundaries == {
        "route-A-A": (-2.0, 2.0),
        "route-A-B": (-2.0, 2.0),
        "route-B-A": (-2.0, 2.0),
        "route-B-B": (-2.0, 2.0),
    }
    assert space.input_arguments["configs"] == {
        "decoder": "gumbel-sinkhorn",
        "bounds": (-2.0, 2.0),
        "params": {"sinkhorn_iterations": 7, "temperature": 0.5},
    }


@pytest.mark.parametrize(
    ("choices", "inputs", "result_type"),
    [
        (["A", "B", "C"], {"0-A": 0.2, "0-B": 0.1, "0-C": 0.3}, list),
        (
            {"route": ["A", "B", "C"]},
            {"route-A": 0.2, "route-B": 0.1, "route-C": 0.3},
            dict,
        ),
    ],
    ids=["positional", "keyword"],
)
def test_random_keys_decode_preserves_permutation_structure_and_representation(
    choices: list[str] | dict[str, list[str]],
    inputs: dict[str, float],
    result_type: type[object],
) -> None:
    space = Permutation(choices, decoder="random-keys")
    space.initialize_context(seed=seed_for(404))

    decoded = space.decode(inputs)

    assert isinstance(decoded, result_type)
    values = decoded["route"] if isinstance(decoded, dict) else decoded[0]
    assert_valid_permutation(values, ["A", "B", "C"])


@pytest.mark.parametrize(
    "decoder",
    ["random-keys", "gumbel-random-keys", "plackett-luce", "gumbel-sinkhorn"],
    ids=["random-keys", "gumbel-random-keys", "plackett-luce", "gumbel-sinkhorn"],
)
def test_each_decoder_returns_a_complete_permutation(decoder: str) -> None:
    space = Permutation({"route": ["A", "B", "C"]}, decoder=decoder)
    space.initialize_context(seed=seed_for(405))

    decoded = space.decode(_encoded_inputs(space))

    assert isinstance(decoded, dict)
    assert_valid_permutation(decoded["route"], ["A", "B", "C"])


@pytest.mark.parametrize(
    ("choices", "inputs"),
    [
        (["A", "B"], [["B", "A"]]),
        ({"route": ["A", "B"]}, {"route": ["B", "A"]}),
    ],
    ids=["positional", "keyword"],
)
def test_cache_codec_preserves_permutation_input_representation(
    choices: list[str] | dict[str, list[str]],
    inputs: list[list[str]] | dict[str, list[str]],
) -> None:
    space = Permutation(choices)
    space.initialize_context(seed=seed_for(406))

    assert space.decode_cache(space.encode_cache(inputs)) == inputs


@pytest.mark.parametrize(
    "decoder",
    ["gumbel-random-keys", "plackett-luce", "gumbel-sinkhorn"],
    ids=["gumbel-random-keys", "plackett-luce", "gumbel-sinkhorn"],
)
def test_stochastic_decoders_warn_without_seed_and_replay_with_same_seed(
    decoder: str,
) -> None:
    unseeded = Permutation(["A", "B", "C"], decoder=decoder)
    with pytest.warns(RuntimeWarning, match="without a seed"):
        unseeded.initialize_context()

    first = Permutation(["A", "B", "C"], decoder=decoder)
    second = Permutation(["A", "B", "C"], decoder=decoder)
    first.initialize_context(seed=seed_for(407))
    second.initialize_context(seed=seed_for(407))
    first_inputs = _encoded_inputs(first)
    second_inputs = _encoded_inputs(second)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert [first.decode(first_inputs) for _ in range(3)] == [
            second.decode(second_inputs) for _ in range(3)
        ]


@pytest.mark.parametrize(
    ("inputs", "exception"),
    [
        ({"0-A": 1.1, "0-B": 0.0}, ValueError),
        ({"unknown": 0.0}, KeyError),
    ],
    ids=["outside-encoded-boundary", "unknown-variable"],
)
def test_decode_rejects_invalid_encoded_inputs(
    inputs: dict[str, float], exception: type[Exception]
) -> None:
    space = Permutation(["A", "B"])
    space.initialize_context(seed=seed_for(408))

    with pytest.raises(exception):
        space.decode(inputs)


@pytest.mark.parametrize(
    ("choices", "exception"),
    [
        ([], ValueError),
        ({"route": []}, ValueError),
        ({"route": 3}, TypeError),
        (np.asarray([[["A", "B"]]]), ValueError),
    ],
    ids=["empty", "empty-named", "non-iterable-variable", "three-dimensional-array"],
)
def test_constructor_rejects_invalid_choice_collections(
    choices: object, exception: type[Exception]
) -> None:
    with pytest.raises(exception):
        Permutation(choices)  # type: ignore[arg-type]


def test_constructor_accepts_one_dimensional_numpy_choices() -> None:
    space = Permutation(np.asarray(["A", "B"]))

    space.initialize_context(seed=seed_for(409))

    assert space.variable_names == ["0"]
    assert space.encoded_variable_names == ["0-A", "0-B"]


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
        Permutation(["A"], bounds=bounds)  # type: ignore[arg-type]


def test_constructor_preserves_all_choices_from_a_documented_iterable() -> None:
    space = Permutation(iter(["A", "B"]))

    space.initialize_context(seed=seed_for(410))

    assert space.encoded_variable_names == ["0-A", "0-B"]


def test_encoded_representation_distinguishes_choices_with_equal_string_forms() -> None:
    space = Permutation([1, "1"])

    space.initialize_context(seed=seed_for(411))

    assert len(space.encoded_variable_names) == 2


def test_constructor_rejects_unknown_decoder() -> None:
    with pytest.raises(ValueError, match="Invalid decoder"):
        Permutation(["A"], decoder="invalid")
