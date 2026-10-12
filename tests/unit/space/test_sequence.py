import warnings
from typing import cast

import numpy as np
import pytest

from tests._support.numerics import seed_for
from tyrannis.space import Sequence
from tyrannis.space.sequence import StopPositions


def test_default_categorical_represents_one_sequence_with_optional_positions() -> None:
    space = Sequence(np.array(["A", "B"]), min_choices=1, max_choices=3)
    space.initialize_context()

    assert space.variable_names == ["0"]
    # Only optional positions add a termination alternative; its name is private.
    choice_names = [f"{step}-{choice}" for step in range(3) for choice in ("A", "B")]
    stop_names = set(space.encoded_variable_names) - set(choice_names)
    assert len(space.encoded_variable_names) == 8
    assert len(stop_names) == 2
    assert space.encoded_boundaries == dict.fromkeys(
        [*choice_names, *stop_names], (-1.0, 1.0)
    )
    # SpaceBase exposes this public metadata as Any; keep its values opaque.
    assert cast("dict[str, object]", space.input_arguments) == {
        "space": "sequence",
        "boundaries": [["A", "B"]],
        "configs": {
            "min_choices": 1,
            "max_choices": 3,
            "positions": None,
            "stop_positions": None,
            "decoder": "categorical-one-hot",
            "params": {},
            "groupable": False,
        },
    }
    inputs = dict.fromkeys(space.encoded_variable_names, -1.0)
    inputs.update(dict.fromkeys(stop_names, 1.0))
    inputs["0-B"] = 1.0
    # Favor a real choice after the first STOP: it must still be discarded.
    inputs.update({name: -1.0 for name in stop_names if name.startswith("2-")})
    inputs["2-A"] = 1.0
    assert space.decode(inputs) == [["B"]]

    inputs.update(dict.fromkeys(stop_names, -1.0))
    inputs["1-B"] = 1.0
    assert space.decode(inputs) == [["B", "B", "A"]]


@pytest.mark.parametrize("stops", [1.0, [1.0, -2.0]], ids=["scalar", "per-position"])
def test_explicit_ordinal_positions_define_named_sequence_termination(
    stops: StopPositions,
) -> None:
    space = Sequence(
        {"route": ["A", "B"]},
        min_choices=1,
        max_choices=3,
        positions=np.array([0.0, 2.0]),
        stop_positions=stops,
    )
    space.initialize_context()

    assert space.variable_names == ["route"]
    assert space.encoded_variable_names == ["route-0", "route-1", "route-2"]
    # SpaceBase exposes this public metadata as Any; keep its values opaque.
    assert cast("dict[str, object]", space.input_arguments) == {
        "space": "sequence",
        "boundaries": {"route": ["A", "B"]},
        "configs": {
            "min_choices": 1,
            "max_choices": 3,
            "positions": [0.0, 2.0],
            "stop_positions": stops,
            "decoder": "ordinal-rank",
            "params": {},
            "groupable": False,
        },
    }
    assert space.decode({"route-2": 2.0, "route-1": 1.0, "route-0": 0.0}) == {
        "route": ["A"]
    }
    last_stop = stops if isinstance(stops, float) else -2.0
    assert space.decode({"route-0": 2.0, "route-1": 0.0, "route-2": last_stop}) == {
        "route": ["B", "A"]
    }


def test_automatic_ordinal_positions_allow_an_empty_sequence() -> None:
    space = Sequence(["A", "B"], min_choices=0, max_choices=2, decoder="ordinal-rank")
    space.initialize_context()

    assert space.encoded_boundaries == {"0": (-0.5, 2.5), "1": (-0.5, 2.5)}
    assert space.decode({"0": 2.0, "1": 0.0}) == [[]]
    assert space.decode({"0": 1.0, "1": 0.0}) == [["B", "A"]]


def test_fixed_length_ordinal_sequence_needs_no_stop_position() -> None:
    space = Sequence(["A", "B"], min_choices=2, max_choices=2, positions=[0.0, 2.0])
    space.initialize_context()

    assert space.encoded_boundaries == {"0": (-1.0, 3.0), "1": (-1.0, 3.0)}
    assert space.decode({"0": 0.0, "1": 2.0}) == [["A", "B"]]


def test_categorical_bounds_parameter_is_forwarded_to_the_base_space() -> None:
    space = Sequence(
        ["A", "B"],
        min_choices=1,
        max_choices=2,
        decoder="categorical-scalar",
        params={"bounds": (-2.0, 2.0)},
    )
    space.initialize_context()

    assert space.encoded_boundaries == {"0": (-2.0, 2.0), "1": (-2.0, 2.0)}
    assert space.decode({"0": -2.0, "1": 2.0}) == [["A"]]


def test_initialize_context_forwards_seed_and_resets_stochastic_decoding() -> None:
    space = Sequence(
        ["A", "B"], min_choices=1, max_choices=2, decoder="categorical-softmax"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        space.initialize_context(seed=seed_for(601))
        inputs = dict.fromkeys(space.encoded_variable_names, 0.0)
        first = [space.decode(inputs) for _ in range(4)]
        space.initialize_context(seed=seed_for(601))
        assert [space.decode(inputs) for _ in range(4)] == first


@pytest.mark.parametrize(
    "choices",
    [{"left": ["A"], "right": ["B"]}, np.array([["A", "B"]])],
    ids=["multiple-variables", "multidimensional-array"],
)
def test_choices_must_describe_a_single_sequence(
    choices: list[str] | dict[str, list[str]],
) -> None:
    with pytest.raises(Exception):  # noqa: B017
        _ = Sequence(choices, min_choices=1, max_choices=2)


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    [(0.5, 2), (0, 2.5), (True, 2), (0, True), (-1, 2), (0, 0), (3, 2)],
    ids=[
        "minimum-not-integer",
        "maximum-not-integer",
        "boolean-minimum",
        "boolean-maximum",
        "negative-minimum",
        "nonpositive-maximum",
        "reversed-limits",
    ],
)
def test_length_limits_reject_values_outside_the_sequence_domain(
    minimum: int,
    maximum: int,
) -> None:
    with pytest.raises(Exception):  # noqa: B017
        _ = Sequence(["A", "B"], min_choices=minimum, max_choices=maximum)


@pytest.mark.parametrize(
    "decoder",
    ["rank", "categorical-unknown", "ordinal-unknown"],
    ids=["missing-family", "unsupported-categorical", "unsupported-ordinal"],
)
def test_decoder_must_select_a_supported_family_and_method(decoder: str) -> None:
    with pytest.raises(Exception):  # noqa: B017
        _ = Sequence(["A", "B"], min_choices=1, max_choices=2, decoder=decoder)


@pytest.mark.parametrize(
    ("decoder", "positions", "stops", "minimum", "params"),
    [
        ("categorical-one-hot", [0.0, 2.0], None, 1, {}),
        ("categorical-one-hot", None, 1.0, 1, {}),
        ("ordinal-rank", None, 1.0, 1, {}),
        ("ordinal-rank", [0.0, 2.0], None, 1, {}),
        ("ordinal-rank", [0.0, 2.0], 1.0, 2, {}),
        ("ordinal-rank", None, None, 1, {"bounds": (-1.0, 1.0)}),
    ],
    ids=[
        "categorical-with-positions",
        "categorical-with-stop",
        "automatic-positions-with-stop",
        "explicit-positions-without-stop",
        "fixed-length-with-stop",
        "ordinal-with-categorical-bounds",
    ],
)
def test_decoder_configuration_rejects_incompatible_sequence_options(
    decoder: str,
    positions: list[float] | None,
    stops: float | None,
    minimum: int,
    params: dict[str, tuple[float, float]],
) -> None:
    with pytest.raises(Exception):  # noqa: B017
        _ = Sequence(
            ["A", "B"],
            min_choices=minimum,
            max_choices=2,
            decoder=decoder,
            positions=positions,
            stop_positions=stops,
            params=params,
        )


@pytest.mark.parametrize(
    "positions",
    [
        {"route": [0.0, 2.0]},
        np.array([[0.0, 2.0]]),
        [0.0],
        [0.0, "two"],
        [False, 2.0],
        [0.0, np.inf],
        [2.0, 0.0],
    ],
    ids=[
        "mapping-instead-of-flat-collection",
        "multidimensional-array",
        "choice-count-mismatch",
        "nonnumeric",
        "boolean",
        "nonfinite",
        "not-increasing-before-stop-insertion",
    ],
)
def test_explicit_positions_are_a_flat_ordered_numeric_scale(
    positions: list[float],
) -> None:
    with pytest.raises(Exception):  # noqa: B017
        _ = Sequence(
            {"route": ["A", "B"]},
            min_choices=1,
            max_choices=3,
            positions=positions,
            stop_positions=1.0,
        )


@pytest.mark.parametrize(
    "stops",
    ["stop", True, np.inf, [1.0], [1.0, 2.0]],
    ids=["nonnumeric", "boolean", "nonfinite", "optional-count-mismatch", "collision"],
)
def test_stop_positions_require_finite_noncolliding_values_for_optional_steps(
    stops: StopPositions,
) -> None:
    with pytest.raises(Exception):  # noqa: B017
        _ = Sequence(
            ["A", "B"],
            min_choices=1,
            max_choices=3,
            positions=[0.0, 2.0],
            stop_positions=stops,
        )


@pytest.mark.parametrize(
    ("decoder", "choices", "inputs", "key_type"),
    [
        ("categorical-one-hot", ["A", "B"], [["B", "B"]], tuple),
        ("categorical-one-hot", {"route": ["A", "B"]}, {"route": []}, tuple),
        ("ordinal-rank", ["A", "B"], [["B", "B"]], str),
        ("ordinal-rank", {"route": ["A", "B"]}, {"route": ["A"]}, str),
    ],
    ids=[
        "categorical-positional",
        "categorical-keyword",
        "ordinal-positional",
        "ordinal-keyword",
    ],
)
def test_cache_round_trip_preserves_one_sequence_and_hides_stop_padding(
    decoder: str,
    choices: list[str] | dict[str, list[str]],
    inputs: list[list[str]] | dict[str, list[str]],
    key_type: type[object],
) -> None:
    ordinal = decoder == "ordinal-rank"
    space = Sequence(
        choices,
        min_choices=0,
        max_choices=3,
        decoder=decoder,
        positions=[0.0, 2.0] if ordinal else None,
        # STOP precedes both choices, so ordinal cache indices must reflect it.
        stop_positions=-2.0 if ordinal else None,
    )

    key = space.encode_cache(inputs)

    assert isinstance(key, key_type)
    assert space.decode_cache(key) == inputs


@pytest.mark.parametrize(
    ("choices", "inputs"),
    [
        (["A", "B"], {"0": ["A"]}),
        (["A", "B"], [["A"], ["B"]]),
        ({"route": ["A", "B"]}, [["A"]]),
        ({"route": ["A", "B"]}, {"other": ["A"]}),
        (["A", "B"], [("A",)]),
    ],
    ids=[
        "positional-rejects-mapping",
        "multiple-positional-sequences",
        "keyword-requires-mapping",
        "wrong-keyword",
        "inner-sequence-requires-list",
    ],
)
def test_cache_inputs_must_match_the_sequence_representation(
    choices: list[str] | dict[str, list[str]],
    inputs: list[list[str]] | dict[str, list[str]],
) -> None:
    space = Sequence(choices, min_choices=1, max_choices=2)

    with pytest.raises(Exception):  # noqa: B017
        _ = space.encode_cache(inputs)


@pytest.mark.parametrize(
    ("decoder", "key"),
    [("categorical-one-hot", "[0,0]"), ("ordinal-rank", ("A", "B"))],
    ids=["categorical-requires-tuple", "ordinal-requires-string"],
)
def test_cache_key_type_must_match_the_selected_base(
    decoder: str,
    key: str | tuple[str, ...],
) -> None:
    space = Sequence(["A", "B"], min_choices=1, max_choices=2, decoder=decoder)

    with pytest.raises(Exception):  # noqa: B017
        _ = space.decode_cache(key)


@pytest.mark.parametrize(
    "values",
    [list[str](), ["A", "B", "A"]],
    ids=["below-minimum", "above-maximum"],
)
def test_cache_sequence_length_must_respect_both_limits(values: list[str]) -> None:
    space = Sequence(["A", "B"], min_choices=1, max_choices=2)

    with pytest.raises(Exception):  # noqa: B017
        _ = space.encode_cache([values])
