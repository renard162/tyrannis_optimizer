import warnings
from collections.abc import Iterable
from typing import cast

import numpy as np
import pytest

from tests._support.numerics import seed_for
from tyrannis.space import Ordinal
from tyrannis.space.ordinal import PositionBoundaries, Positions


@pytest.mark.parametrize(
    "choices",
    [
        ["low", "high"],
        ("low", "high"),
        iter(["low", "high"]),
        np.array(["low", "high"]),
    ],
    ids=["list", "tuple", "iterator", "numpy"],
)
def test_single_collection_materializes_choices_and_default_configuration(
    choices: Iterable[str],
) -> None:
    space = Ordinal(choices)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        space.initialize_context()

    assert space.variable_names == ["0"]
    assert space.encoded_variable_names == ["0"]
    assert space.encoded_boundaries == {"0": (-0.5, 1.5)}
    # SpaceBase annotates this public dictionary as Any; retain opaque values.
    assert cast("dict[str, object]", space.input_arguments) == {
        "space": "ordinal",
        "boundaries": [["low", "high"]],
        "configs": {"positions": None, "decoder": "rank", "params": {}},
    }


@pytest.mark.parametrize(
    ("choices", "positions", "names", "boundaries", "expected"),
    [
        (
            [("low", "high"), ("small", "large")],
            [(-2.0, 2.0), (0.0, 2.0)],
            ["0", "1"],
            {"0": (-4.0, 4.0), "1": (-1.0, 3.0)},
            ["high", "large"],
        ),
        (
            np.array([["low", "high"], ["small", "large"]]),
            np.array([[-2.0, 2.0], [0.0, 2.0]]),
            ["0", "1"],
            {"0": (-4.0, 4.0), "1": (-1.0, 3.0)},
            ["high", "large"],
        ),
        (
            {"grade": ("low", "high"), "size": ("small", "large")},
            {"size": (0.0, 2.0), "grade": (-2.0, 2.0)},
            ["grade", "size"],
            {"grade": (-4.0, 4.0), "size": (-1.0, 3.0)},
            {"grade": "high", "size": "large"},
        ),
    ],
    ids=["positional", "numpy-matrix", "keyword"],
)
def test_custom_positions_preserve_variable_metadata_and_representation(
    choices: Iterable[object],
    positions: Positions | PositionBoundaries,
    names: list[str],
    boundaries: dict[str, tuple[float, float]],
    expected: list[str] | dict[str, str],
) -> None:
    space = Ordinal(choices, positions=positions)
    space.initialize_context()

    assert space.variable_names == names
    assert space.encoded_variable_names == names
    assert space.encoded_boundaries == boundaries
    assert space.decode(dict.fromkeys(reversed(names), 2.0)) == expected


@pytest.mark.parametrize(
    "positions",
    [(-2, 2), iter([-2, 2]), np.array([-2, 2])],
    ids=["tuple", "iterator", "numpy"],
)
def test_single_positions_are_materialized_in_public_configuration(
    positions: Positions,
) -> None:
    space = Ordinal(["low", "high"], positions=positions)

    # SpaceBase annotates this public dictionary as Any; retain opaque values.
    assert cast("dict[str, object]", space.input_arguments)["configs"] == {
        "positions": [-2.0, 2.0],
        "decoder": "rank",
        "params": {},
    }


@pytest.mark.parametrize(
    "choices",
    [[], {}, {"grade": []}, [["low"], []], {"grade": 3}, np.array([[["low"]]])],
    ids=[
        "empty",
        "empty-mapping",
        "empty-named",
        "empty-positional",
        "scalar-variable",
        "3d-array",
    ],
)
def test_constructor_rejects_invalid_choice_collections(
    choices: Iterable[object],
) -> None:
    with pytest.raises(Exception):  # noqa: B017
        _ = Ordinal(choices)


@pytest.mark.parametrize(
    ("choices", "positions"),
    [
        ({"grade": ["low", "high"]}, [0, 1]),
        (["low", "high"], {"grade": [0, 1]}),
        ({"grade": ["low", "high"]}, {"other": [0, 1]}),
        (["low", "high"], [[0, 1]]),
        ([["low", "high"]], [0, 1]),
        ([["low"], ["high"]], [[0]]),
        (["low", "high"], [0]),
        (["low", "high"], [0, "one"]),
        (["low", "high"], [False, 1]),
        (["low", "high"], [0, np.inf]),
        (["low", "high"], [0, np.nan]),
        (["low", "high"], [0, 0]),
        (["low", "high"], [1, 0]),
        (["only"], [1e20]),
        ({"grade": ["low", "high"]}, {"grade": 1}),
        ({"grade": ["low", "high"]}, {"grade": np.array([[0, 1]])}),
    ],
    ids=[
        "named-needs-mapping",
        "positional-rejects-mapping",
        "wrong-keys",
        "single-versus-nested",
        "nested-versus-single",
        "variable-count",
        "choice-count",
        "nonnumeric",
        "boolean",
        "infinite",
        "nan",
        "equal",
        "descending",
        "collapsed-boundaries",
        "scalar-positions",
        "2d-positions",
    ],
)
def test_constructor_rejects_invalid_positions(
    choices: Iterable[object],
    positions: object,
) -> None:
    with pytest.raises(Exception):  # noqa: B017
        # Invalid runtime inputs intentionally exceed the positions annotation.
        _ = Ordinal(
            choices, positions=cast("Positions | PositionBoundaries", positions)
        )


def test_constructor_rejects_unknown_decoder() -> None:
    with pytest.raises(Exception):  # noqa: B017
        _ = Ordinal(["low", "high"], decoder="invalid")


@pytest.mark.parametrize(
    ("decoder", "temperature"),
    [
        ("cumulative-logit", 0),
        ("cumulative-logit", -1),
        ("cumulative-logit", True),
        ("cumulative-logit", "1"),
        ("cumulative-logit", np.inf),
        ("cumulative-logit", np.nan),
        ("cumulative-probit", 0),
        ("distance-softmax", 0),
    ],
    ids=[
        "zero",
        "negative",
        "boolean",
        "string",
        "infinite",
        "nan",
        "probit",
        "softmax",
    ],
)
def test_temperature_decoders_reject_values_outside_the_parameter_domain(
    decoder: str,
    temperature: object,
) -> None:
    with pytest.raises(Exception):  # noqa: B017
        _ = Ordinal(
            ["low", "high"], decoder=decoder, params={"temperature": temperature}
        )


@pytest.mark.parametrize("decoder", ["rank", "nearest-stochastic"])
def test_decoders_without_temperature_do_not_validate_it(decoder: str) -> None:
    space = Ordinal(["low", "high"], decoder=decoder, params={"temperature": 0})
    space.initialize_context(seed=seed_for(501))

    assert space.decode({"0": 0.25}) in (["low"], ["high"])


@pytest.mark.parametrize(
    "decoder",
    ["nearest-stochastic", "cumulative-logit", "cumulative-probit", "distance-softmax"],
)
def test_stochastic_decoders_warn_without_seed_and_replay_with_same_seed(
    decoder: str,
) -> None:
    first = Ordinal(["low", "medium", "high"], decoder=decoder)
    with pytest.warns(RuntimeWarning, match="without a seed"):
        first.initialize_context()

    # An omitted temperature uses 1.0, including when replaying seeded state.
    second = Ordinal(
        ["low", "medium", "high"], decoder=decoder, params={"temperature": 1.0}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        first.initialize_context(seed=seed_for(502))
        second.initialize_context(seed=seed_for(502))
        first_values = [first.decode({"0": 0.75}) for _ in range(4)]
        assert first_values == [second.decode({"0": 0.75}) for _ in range(4)]
        assert all(value in (["low"], ["medium"], ["high"]) for value in first_values)


@pytest.mark.parametrize(
    "decoder", ["cumulative-logit", "cumulative-probit", "distance-softmax"]
)
def test_temperature_decoders_expose_and_accept_custom_configuration(
    decoder: str,
) -> None:
    space = Ordinal(
        {"grade": ["low", "high"]},
        positions={"grade": [-2.0, 2.0]},
        decoder=decoder,
        params={"temperature": np.float64(0.5)},
    )
    space.initialize_context(seed=seed_for(503))

    # SpaceBase exposes input_arguments as Any rather than a typed dictionary.
    assert cast("dict[str, object]", space.input_arguments)["configs"] == {
        "positions": {"grade": [-2.0, 2.0]},
        "decoder": decoder,
        "params": {"temperature": 0.5},
    }
    assert space.decode({"grade": 0.0}) in ({"grade": "low"}, {"grade": "high"})


@pytest.mark.parametrize(
    "inputs",
    [{"0": -0.6}, {"0": 1.6}, {"unknown": 0.0}, {}],
    ids=["below-boundary", "above-boundary", "unknown-variable", "missing-variable"],
)
def test_decode_rejects_invalid_encoded_inputs(inputs: dict[str, float]) -> None:
    space = Ordinal(["low", "high"])
    space.initialize_context()

    with pytest.raises(Exception):  # noqa: B017
        _ = space.decode(inputs)


@pytest.mark.parametrize(
    ("choices", "inputs", "key"),
    [
        ([["low", "high"], ["small", "large"]], ["high", "small"], "[1,0]"),
        (
            {"size": ["small", "large"], "grade": ["low", "high"]},
            {"size": "small", "grade": "high"},
            '{"grade":1,"size":0}',
        ),
    ],
    ids=["positional", "keyword"],
)
def test_cache_codec_round_trip_uses_canonical_indices(
    choices: Iterable[object],
    inputs: list[str] | dict[str, str],
    key: str,
) -> None:
    space = Ordinal(choices)

    assert space.encode_cache(inputs) == key
    assert space.decode_cache(key) == inputs
    if isinstance(inputs, dict):
        assert space.encode_cache(dict(reversed(inputs.items()))) == key


def test_cache_codec_preserves_unhashable_choices_without_serializing_the_values() -> (
    None
):
    choice = {"payload": [1, 2]}
    space = Ordinal({"grade": [choice]}, use_cache=True)

    key = space.encode_cache({"grade": {"payload": [1, 2]}})
    decoded = space.decode_cache(key)

    assert key == '{"grade":0}'
    assert isinstance(decoded, dict)
    assert decoded["grade"] is choice


@pytest.mark.parametrize(
    ("choices", "inputs"),
    [
        (["low", "high"], []),
        (["low", "high"], ["low", "high"]),
        (["low", "high"], {"0": "low"}),
        ({"grade": ["low", "high"]}, ["low"]),
        ({"grade": ["low", "high"]}, {"other": "low"}),
        ({"grade": ["low", "high"]}, {"grade": "low", "extra": "high"}),
        (["low", "high"], ["unknown"]),
    ],
    ids=[
        "missing-value",
        "extra-value",
        "unexpected-mapping",
        "expected-mapping",
        "wrong-keys",
        "extra-key",
        "unknown-choice",
    ],
)
def test_encode_cache_rejects_inputs_outside_configured_variables_and_choices(
    choices: Iterable[object],
    inputs: list[str] | dict[str, str],
) -> None:
    space = Ordinal(choices)

    with pytest.raises(Exception):  # noqa: B017
        _ = space.encode_cache(inputs)


@pytest.mark.parametrize(
    ("choices", "key"),
    [
        (["low", "high"], '{"0":0}'),
        ({"grade": ["low", "high"]}, "[0]"),
        (["low", "high"], "[]"),
        (["low", "high"], "[0,1]"),
        ({"grade": ["low", "high"]}, '{"other":0}'),
        ({"grade": ["low", "high"]}, '{"grade":0,"extra":1}'),
        (["low", "high"], "[-1]"),
        (["low", "high"], "[2]"),
        (["low", "high"], "[true]"),
        (["low", "high"], "[0.5]"),
        ({"grade": ["low", "high"]}, '{"grade":"0"}'),
        ({"grade": ["low", "high"]}, '{"grade":2}'),
        (["low", "high"], "invalid-json"),
    ],
    ids=[
        "unexpected-mapping",
        "expected-mapping",
        "missing-index",
        "extra-index",
        "wrong-keys",
        "extra-key",
        "negative-index",
        "index-out-of-range",
        "boolean-index",
        "fractional-index",
        "string-index",
        "named-index-out-of-range",
        "malformed-json",
    ],
)
def test_decode_cache_rejects_invalid_structure_or_indices(
    choices: Iterable[object],
    key: str,
) -> None:
    space = Ordinal(choices)

    with pytest.raises(Exception):  # noqa: B017
        _ = space.decode_cache(key)
