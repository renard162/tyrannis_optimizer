import cloudpickle
import pytest

from tyrannis.space.continuous import Continuous


def list_cost_function(*inputs: float) -> float:
    return sum(value**2 for value in inputs)


def dict_cost_function(**inputs: float) -> float:
    return sum(value**2 for value in inputs.values())


def test_init_with_list_boundaries() -> None:
    boundaries = [
        (-1.0, 1.0),
        (0.0, 10.0),
        (100.0, 200.0),
    ]

    space = Continuous(
        list_cost_function,
        boundaries,
    )

    assert space._cost_function is list_cost_function
    assert space._boundaries == boundaries


def test_init_with_dict_boundaries() -> None:
    boundaries = {
        "x": (-1.0, 1.0),
        "y": (0.0, 10.0),
        "z": (100.0, 200.0),
    }

    space = Continuous(
        dict_cost_function,
        boundaries,
    )

    assert space._cost_function is dict_cost_function
    assert space._boundaries == boundaries


def test_init_accepts_none_cost_function() -> None:
    boundaries = [
        (-1.0, 1.0),
        (0.0, 10.0),
    ]

    space = Continuous(
        None,
        boundaries,
    )

    assert space._cost_function is None
    assert space._boundaries == boundaries


def test_initialize_context_with_list_boundaries() -> None:
    boundaries = [
        (-1.0, 1.0),
        (0.0, 10.0),
        (100.0, 200.0),
    ]

    space = Continuous(
        list_cost_function,
        boundaries,
    )

    space.initialize_context(42)

    assert space._encoded_boundaries == {
        "0": (-1.0, 1.0),
        "1": (0.0, 10.0),
        "2": (100.0, 200.0),
    }


def test_initialize_context_with_dict_boundaries() -> None:
    boundaries = {
        "x": (-1.0, 1.0),
        "y": (0.0, 10.0),
        "z": (100.0, 200.0),
    }

    space = Continuous(
        dict_cost_function,
        boundaries,
    )

    space.initialize_context(42)

    assert space._encoded_boundaries == boundaries


def test_initialize_context_with_dict_boundaries_creates_copy() -> None:
    boundaries = {
        "x": (-1.0, 1.0),
        "y": (0.0, 10.0),
    }

    space = Continuous(
        dict_cost_function,
        boundaries,
    )

    space.initialize_context(42)

    assert space._encoded_boundaries == boundaries
    assert space._encoded_boundaries is not boundaries


def test_initialize_context_accepts_none_cost_function() -> None:
    space = Continuous(
        None,
        [
            (-1.0, 1.0),
            (0.0, 10.0),
        ],
    )

    space.initialize_context(42)

    assert space._encoded_boundaries == {
        "0": (-1.0, 1.0),
        "1": (0.0, 10.0),
    }


def test_initialize_context_ignores_seed() -> None:
    boundaries = [
        (-1.0, 1.0),
        (0.0, 10.0),
    ]

    space = Continuous(
        list_cost_function,
        boundaries,
    )

    space.initialize_context(0)
    first_context = space._encoded_boundaries.copy()

    space.initialize_context(999999)
    second_context = space._encoded_boundaries.copy()

    assert first_context == second_context


def test_is_kwargs_is_false_for_list_boundaries() -> None:
    space = Continuous(
        list_cost_function,
        [
            (-1.0, 1.0),
            (0.0, 10.0),
        ],
    )

    assert space.is_kwargs is False


def test_is_kwargs_is_true_for_dict_boundaries() -> None:
    space = Continuous(
        dict_cost_function,
        {
            "x": (-1.0, 1.0),
            "y": (0.0, 10.0),
        },
    )

    assert space.is_kwargs is True


def test_decode_list_boundaries_returns_ordered_list() -> None:
    space = Continuous(
        list_cost_function,
        [
            (0.0, 10.0),
            (10.0, 20.0),
            (20.0, 30.0),
            (30.0, 40.0),
        ],
    )

    space.initialize_context(42)

    float_inputs = {
        "3": 35.0,
        "1": 15.0,
        "0": 5.0,
        "2": 25.0,
    }

    result = space.decode(float_inputs)

    assert result == [
        5.0,
        15.0,
        25.0,
        35.0,
    ]


def test_decode_list_boundaries_uses_numeric_key_order() -> None:
    space = Continuous(
        list_cost_function,
        [
            (0.0, 10.0),
            (0.0, 10.0),
            (0.0, 10.0),
        ],
    )

    space.initialize_context(42)

    result = space.decode(
        {
            "2": 2.0,
            "0": 0.0,
            "1": 1.0,
        }
    )

    assert result == [0.0, 1.0, 2.0]


def test_decode_dict_boundaries_returns_input_dictionary() -> None:
    space = Continuous(
        dict_cost_function,
        {
            "x": (-1.0, 1.0),
            "y": (0.0, 10.0),
        },
    )

    space.initialize_context(42)

    float_inputs = {
        "x": 0.5,
        "y": 5.0,
    }

    result = space.decode(float_inputs)

    assert result is float_inputs


def test_decode_accepts_lower_boundary() -> None:
    space = Continuous(
        list_cost_function,
        [
            (-1.0, 1.0),
        ],
    )

    space.initialize_context(42)

    assert space.decode({"0": -1.0}) == [-1.0]


def test_decode_accepts_upper_boundary() -> None:
    space = Continuous(
        list_cost_function,
        [
            (-1.0, 1.0),
        ],
    )

    space.initialize_context(42)

    assert space.decode({"0": 1.0}) == [1.0]


def test_decode_rejects_value_below_boundary() -> None:
    space = Continuous(
        list_cost_function,
        [
            (-1.0, 1.0),
        ],
    )

    space.initialize_context(42)

    with pytest.raises(
        ValueError,
        match="outside the boundaries",
    ):
        space.decode({"0": -1.000001})


def test_decode_rejects_value_above_boundary() -> None:
    space = Continuous(
        list_cost_function,
        [
            (-1.0, 1.0),
        ],
    )

    space.initialize_context(42)

    with pytest.raises(
        ValueError,
        match="outside the boundaries",
    ):
        space.decode({"0": 1.000001})


def test_decode_rejects_unknown_variable() -> None:
    space = Continuous(
        list_cost_function,
        [
            (-1.0, 1.0),
        ],
    )

    space.initialize_context(42)

    with pytest.raises(
        KeyError,
        match="Unknown continuous-space variable",
    ):
        space.decode(
            {
                "0": 0.0,
                "1": 0.0,
            }
        )


def test_decode_validates_dict_boundaries() -> None:
    space = Continuous(
        dict_cost_function,
        {
            "x": (-1.0, 1.0),
            "y": (0.0, 10.0),
        },
    )

    space.initialize_context(42)

    with pytest.raises(
        ValueError,
        match="outside the boundaries",
    ):
        space.decode(
            {
                "x": 0.5,
                "y": 10.000001,
            }
        )


def test_decode_rejects_unknown_dict_variable() -> None:
    space = Continuous(
        dict_cost_function,
        {
            "x": (-1.0, 1.0),
        },
    )

    space.initialize_context(42)

    with pytest.raises(
        KeyError,
        match="Unknown continuous-space variable",
    ):
        space.decode(
            {
                "x": 0.0,
                "unknown": 0.0,
            }
        )


def test_call_with_list_boundaries() -> None:
    space = Continuous(
        list_cost_function,
        [
            (-1.0, 1.0),
            (-2.0, 2.0),
        ],
    )

    space.initialize_context(42)

    result = space(
        {
            "1": 2.0,
            "0": 1.0,
        }
    )

    assert result == 5.0


def test_call_with_dict_boundaries() -> None:
    space = Continuous(
        dict_cost_function,
        {
            "x": (-1.0, 1.0),
            "y": (-2.0, 2.0),
        },
    )

    space.initialize_context(42)

    result = space(
        {
            "x": 1.0,
            "y": 2.0,
        }
    )

    assert result == 5.0


def test_call_rejects_none_cost_function() -> None:
    space = Continuous(
        None,
        [
            (-1.0, 1.0),
        ],
    )

    space.initialize_context(42)

    with pytest.raises(
        ValueError,
        match="cost_function cannot be None",
    ):
        space({"0": 0.0})


def test_continuous_is_serializable() -> None:
    space = Continuous(
        list_cost_function,
        [
            (-1.0, 1.0),
            (0.0, 10.0),
        ],
    )

    space.initialize_context(42)

    serialized = cloudpickle.dumps(space)
    restored = cloudpickle.loads(serialized)

    assert isinstance(restored, Continuous)
    assert restored._boundaries == space._boundaries
    assert restored._encoded_boundaries == space._encoded_boundaries
    assert restored.is_kwargs is False

    assert (
        restored(
            {
                "0": 0.5,
                "1": 2.0,
            }
        )
        == 4.25
    )
