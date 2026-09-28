"""Continuous identity contract: D = product([L_i, U_i]), decode(x) = x.

The specified mathematical contract requires the encoded domain to equal D
and evaluation to equal f(x), with only the argument representation adapted.
The asymmetric objective f(x, y) = 2*x + 7*y detects coordinate swaps;
all points and arithmetic below are exactly representable in binary floats.
"""

import numpy as np

from tyrannis.space import Continuous


def test_positional_domain_and_evaluation_preserve_identity() -> None:
    def objective(x: float, y: float, /) -> float:
        return 2.0 * x + 7.0 * y

    space = Continuous(
        [(-3.5, 2.25), (-1.75, 8.5)],
        cost_function=objective,
    )
    space.initialize_context()

    # Equality of ordered intervals proves that encoding leaves D unchanged.
    assert list(space.encoded_boundaries.items()) == [
        ("0", (-3.5, 2.25)),
        ("1", (-1.75, 8.5)),
    ]
    for x, y in (
        (-3.5, -1.75),  # Lower corner.
        (2.25, 8.5),  # Upper corner.
        (-1.25, 4.125),  # Interior, negative x and positive y.
        (1.75, -0.375),  # Interior, positive x and negative y.
    ):
        # Coordinate indices, not dictionary insertion order, define the point.
        encoded = {"1": y, "0": x}
        assert space.decode(encoded) == [x, y]

        result = space(encoded)
        assert type(result) is np.float64
        assert result == 2.0 * x + 7.0 * y


def test_named_domain_and_evaluation_preserve_identity() -> None:
    def objective(*, x: float, y: float) -> float:
        return 2.0 * x + 7.0 * y

    space = Continuous(
        {"y": (-1.75, 8.5), "x": (-3.5, 2.25)},
        cost_function=objective,
    )
    space.initialize_context()

    # Preserve the original coordinate order as well as each exact interval.
    assert list(space.encoded_boundaries.items()) == [
        ("y", (-1.75, 8.5)),
        ("x", (-3.5, 2.25)),
    ]
    for x, y in (
        (-3.5, -1.75),
        (2.25, 8.5),
        (-1.25, 4.125),
        (1.75, -0.375),
    ):
        # Names identify coordinates even when input and domain orders differ.
        encoded = {"x": x, "y": y}
        assert space.decode(encoded) == {"x": x, "y": y}

        result = space(encoded)
        assert type(result) is np.float64
        assert result == 2.0 * x + 7.0 * y
