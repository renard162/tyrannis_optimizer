import warnings

import pytest

from _support.numerics import seed_for
from tyrannis.space import Binary


def test_default_configuration_creates_one_positional_bit() -> None:
    space = Binary()

    space.initialize_context(seed=seed_for(201))

    assert space.variable_names == ["0"]
    assert space.encoded_variable_names == ["0"]
    assert space.encoded_boundaries == {"0": (-2.0, 2.0)}
    assert space.input_arguments == {
        "space": "binary",
        "boundaries": 1,
        "configs": {"decoder": "angle_modulation", "bounds": None, "params": {}},
    }


@pytest.mark.parametrize(
    ("decoder", "expected_boundary"),
    [("angle_modulation", (-2.0, 2.0)), ("s-shape", (-6.0, 6.0))],
    ids=["angle-modulation", "s-shape"],
)
def test_decoder_initialization_uses_its_default_encoded_bounds(
    decoder: str, expected_boundary: tuple[float, float]
) -> None:
    space = Binary(2, decoder=decoder)

    space.initialize_context(seed=seed_for(202))

    assert space.variable_names == ["0", "1"]
    assert space.encoded_boundaries == {
        "0": expected_boundary,
        "1": expected_boundary,
    }


def test_named_configuration_preserves_custom_bounds_and_decoder_parameters() -> None:
    space = Binary(
        ["enabled", "cached"],
        decoder="s-shape",
        bounds=(-1.5, 1.5),
        params={"alpha": 2.0},
    )

    space.initialize_context(seed=seed_for(203))

    assert space.variable_names == ["enabled", "cached"]
    assert space.encoded_variable_names == ["enabled", "cached"]
    assert space.encoded_boundaries == {
        "enabled": (-1.5, 1.5),
        "cached": (-1.5, 1.5),
    }
    assert space.input_arguments["configs"] == {
        "decoder": "s-shape",
        "bounds": (-1.5, 1.5),
        "params": {"alpha": 2.0},
    }


@pytest.mark.parametrize(
    ("bits", "inputs", "result_type"),
    [
        (2, {"0": -2.0, "1": 2.0}, list),
        (["first", "second"], {"first": -2.0, "second": 2.0}, dict),
    ],
    ids=["positional", "keyword"],
)
def test_angle_modulation_decode_returns_boolean_values_in_expected_representation(
    bits: int | list[str], inputs: dict[str, float], result_type: type[object]
) -> None:
    space = Binary(bits, decoder="angle_modulation")
    space.initialize_context(seed=seed_for(204))

    decoded = space.decode(inputs)

    assert isinstance(decoded, result_type)
    values = decoded.values() if isinstance(decoded, dict) else decoded
    assert all(isinstance(value, bool) for value in values)


@pytest.mark.parametrize(
    ("bits", "inputs"),
    [
        (2, [True, False]),
        (["first", "second"], {"second": False, "first": True}),
    ],
    ids=["positional", "keyword"],
)
def test_cache_codec_preserves_binary_input_representation(
    bits: int | list[str], inputs: list[bool] | dict[str, bool]
) -> None:
    space = Binary(bits)
    space.initialize_context(seed=seed_for(205))

    assert space.decode_cache(space.encode_cache(inputs)) == inputs


@pytest.mark.parametrize(
    ("inputs", "exception"),
    [
        ({"0": 2.1}, ValueError),
        ({"unknown": 0.0}, KeyError),
    ],
    ids=["outside-encoded-boundary", "unknown-variable"],
)
def test_decode_rejects_invalid_encoded_inputs(
    inputs: dict[str, float], exception: type[Exception]
) -> None:
    space = Binary(1)
    space.initialize_context(seed=seed_for(206))

    with pytest.raises(exception):
        space.decode(inputs)


@pytest.mark.parametrize(
    ("bits", "exception"),
    [
        (0, ValueError),
        (-1, ValueError),
        (["same", "same"], ValueError),
        (["valid", 1], TypeError),
        ("not-a-list", TypeError),
    ],
    ids=["zero", "negative", "duplicate-names", "non-string-name", "invalid-type"],
)
def test_constructor_rejects_invalid_bit_definitions(
    bits: object, exception: type[Exception]
) -> None:
    with pytest.raises(exception):
        Binary(bits)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("bounds", "exception"),
    [
        ([0.0, 1.0], TypeError),
        ((0.0, "high"), TypeError),
        ((1.0, 1.0), ValueError),
        ((2.0, 1.0), ValueError),
    ],
    ids=["not-tuple", "non-numeric", "equal-endpoints", "reversed-endpoints"],
)
def test_constructor_rejects_invalid_bounds(
    bounds: object, exception: type[Exception]
) -> None:
    with pytest.raises(exception):
        Binary(bounds=bounds)  # type: ignore[arg-type]


def test_s_shape_warns_without_seed_and_replays_with_same_seed() -> None:
    unseeded = Binary(3, decoder="s-shape")
    with pytest.warns(RuntimeWarning, match="without a seed"):
        unseeded.initialize_context()

    inputs = {"0": -1.0, "1": 0.0, "2": 1.0}
    first = Binary(3, decoder="s-shape")
    second = Binary(3, decoder="s-shape")
    first.initialize_context(seed=seed_for(207))
    second.initialize_context(seed=seed_for(207))

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert [first.decode(inputs) for _ in range(3)] == [
            second.decode(inputs) for _ in range(3)
        ]


def test_constructor_rejects_unknown_decoder() -> None:
    with pytest.raises(ValueError, match="Invalid decoder"):
        Binary(decoder="invalid")
