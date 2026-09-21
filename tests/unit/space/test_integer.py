import warnings

import pytest

from _support.numerics import seed_for
from tyrannis.space import Integer


@pytest.mark.parametrize(
    ("decoder", "expected_boundary"),
    [
        ("round", (-3.0, 4.0)),
        ("stochastic_round", (-3.0, 4.0)),
        ("scaling", (0.0, 1.0)),
        ("transfer_function", (-6.0, 6.0)),
    ],
    ids=["round", "stochastic-round", "scaling", "transfer-function"],
)
def test_decoder_initialization_exposes_expected_encoded_boundaries(
    decoder: str, expected_boundary: tuple[float, float]
) -> None:
    space = Integer({"count": (-3, 4)}, decoder=decoder)

    space.initialize_context(seed=seed_for(101))

    assert space.variable_names == ["count"]
    assert space.encoded_variable_names == ["count"]
    assert space.encoded_boundaries == {"count": expected_boundary}


def test_transfer_function_configuration_keeps_custom_bounds_and_parameters() -> None:
    space = Integer(
        {"count": (2, 9)},
        decoder="transfer_function",
        custom_bounds=(-4.0, 4.0),
        params={"alpha": 2.5},
    )

    space.initialize_context(seed=seed_for(102))

    assert space.encoded_boundaries == {"count": (-4.0, 4.0)}
    assert space.input_arguments == {
        "space": "integer",
        "boundaries": {"count": (2, 9)},
        "configs": {
            "decoder": "transfer_function",
            "custom_bounds": (-4.0, 4.0),
            "params": {"alpha": 2.5},
        },
    }


@pytest.mark.parametrize(
    ("boundaries", "inputs", "expected"),
    [
        ([(0, 5), (-2, 2)], {"0": 1.0, "1": -2.0}, [1, -2]),
        ({"rows": (0, 5)}, {"rows": 2.0}, {"rows": 2}),
    ],
    ids=["positional", "keyword"],
)
def test_round_decoder_preserves_input_representation_and_cache_round_trip(
    boundaries: list[tuple[int, int]] | dict[str, tuple[int, int]],
    inputs: dict[str, float],
    expected: list[int] | dict[str, int],
) -> None:
    space = Integer(boundaries, decoder="round")
    space.initialize_context(seed=seed_for(103))

    decoded = space.decode(inputs)

    assert isinstance(decoded, type(expected))
    assert decoded == expected
    values = decoded.values() if isinstance(decoded, dict) else decoded
    assert all(isinstance(value, int) for value in values)
    assert space.decode_cache(space.encode_cache(decoded)) == expected


@pytest.mark.parametrize(
    ("decoder", "inputs", "exception"),
    [
        ("round", {"count": 4.0}, ValueError),
        ("scaling", {"unknown": 0.5}, KeyError),
    ],
    ids=["outside-encoded-boundary", "unknown-variable"],
)
def test_decode_rejects_invalid_encoded_inputs(
    decoder: str, inputs: dict[str, float], exception: type[Exception]
) -> None:
    space = Integer({"count": (0, 3)}, decoder=decoder)
    space.initialize_context(seed=seed_for(104))

    with pytest.raises(exception):
        space.decode(inputs)


def test_stochastic_rounding_warns_without_seed_but_not_with_seed() -> None:
    unseeded = Integer((0, 3), decoder="stochastic_round")
    with pytest.warns(RuntimeWarning, match="without a seed"):
        unseeded.initialize_context()

    seeded = Integer((0, 3), decoder="stochastic_round")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        seeded.initialize_context(seed=seed_for(105))


def test_constructor_rejects_unknown_decoder() -> None:
    with pytest.raises(ValueError, match="Invalid decoder"):
        Integer((0, 3), decoder="invalid")


@pytest.mark.parametrize(
    "boundaries",
    [(1, 1), (3, 1)],
    ids=["equal-endpoints", "reversed-endpoints"],
)
def test_constructor_rejects_non_increasing_boundaries(
    boundaries: tuple[int, int],
) -> None:
    with pytest.raises(ValueError):
        Integer(boundaries)
