"""Binary maps: deterministic angle modulation and independent Bernoulli bits.

Kennedy & Eberhart (1997), sec. 3, defines sampling by U < S(x):
https://doi.org/10.1109/ICSMC.1997.637339.
The alpha-logistic and encoded bounds follow the Tyrannis contract.

Pampara, Franken & Engelbrecht (2005), eq. (4), gives the general angle
generator: https://doi.org/10.1109/CEC.2005.1554671.
Tyrannis declares the specialization a=d=0, b=c=1, applied directly to each
coordinate: g(x)=sin(2*pi*x*cos(2*pi*x)), D(x)=[g(x)>0]. It does not evolve
four coefficients to sample an entire bitstring as in the general method.
"""

from collections import Counter
from math import cos, exp, log, sin, tau

import pytest

from tests._support.numerics import (
    BASE_SEED,
    STRICT_ATOL,
    STRICT_RTOL,
    frequency_tolerance,
    seed_for,
)
from tests._support.objectives import constant_objective
from tyrannis.space import Binary


def _angle_generator(x: float) -> float:
    """Evaluate the declared scalar equation independently of NumPy/decoders."""
    return sin(tau * x * cos(tau * x))


def test_angle_modulation_matches_declared_generating_function() -> None:
    space = Binary(1, constant_objective, decoder="angle_modulation", use_cache=False)
    space.initialize_context()
    assert space.encoded_boundaries == {"0": (-2.0, 2.0)}

    outputs: dict[float, bool] = {}
    for x in (0.0, 0.125, -0.125, 0.35, -0.35, 0.8, -0.8):
        generated = _angle_generator(x)
        # This is a witness-selection guard, not a floating-point tolerance:
        # nonzero cases have amplitude > 1/2, far from the sign boundary.
        if x != 0.0:
            assert abs(generated) > 0.5
        decoded = space.decode({"0": x})
        assert isinstance(decoded, list)
        assert len(decoded) == 1
        assert type(decoded[0]) is bool
        assert decoded[0] is (generated > 0.0), (x, generated)
        outputs[x] = decoded[0]

    assert _angle_generator(0.0) == 0.0
    assert outputs[0.0] is False
    assert set(outputs.values()) == {False, True}
    # cos is even and sin is odd; away from zero, the bits are complementary.
    for x in (0.125, 0.35, 0.8):
        assert _angle_generator(-x) == pytest.approx(
            -_angle_generator(x), rel=STRICT_RTOL, abs=STRICT_ATOL
        )
        assert outputs[-x] is not outputs[x]


def test_angle_modulation_preserves_positional_and_named_coordinate_mapping() -> None:
    def positional_objective(enabled: bool, cached: bool, /) -> float:
        assert type(enabled) is bool and type(cached) is bool
        return 2.0 * enabled + 7.0 * cached

    def named_objective(*, enabled: bool, cached: bool) -> float:
        assert type(enabled) is bool and type(cached) is bool
        return 2.0 * enabled + 7.0 * cached

    positional = Binary(
        2, positional_objective, decoder="angle_modulation", use_cache=False
    )
    named = Binary(
        ["enabled", "cached"],
        named_objective,
        decoder="angle_modulation",
        bounds=(-1.5, 1.0),
        use_cache=False,
    )
    positional.initialize_context()
    named.initialize_context()
    assert named.encoded_boundaries == {
        "enabled": (-1.5, 1.0),
        "cached": (-1.5, 1.0),
    }

    outputs: list[list[bool]] = []
    # Change one coordinate at a time, visiting TF, FF, FT, TT. The asymmetric
    # custom domain must not translate or scale the named decoder's inputs.
    for x, y in ((0.125, 0.35), (-0.125, 0.35), (-0.125, -0.35), (0.125, -0.35)):
        expected = [_angle_generator(x) > 0.0, _angle_generator(y) > 0.0]
        # Reverse insertion order: indices/names must still identify the bits.
        positional_inputs = {"1": y, "0": x}
        named_inputs = {"cached": y, "enabled": x}
        positional_decoded = positional.decode(positional_inputs)
        named_decoded = named.decode(named_inputs)
        assert isinstance(positional_decoded, list)
        assert isinstance(named_decoded, dict)
        assert all(type(bit) is bool for bit in positional_decoded)
        assert all(type(bit) is bool for bit in named_decoded.values())
        assert positional_decoded == expected
        assert named_decoded == {"enabled": expected[0], "cached": expected[1]}
        expected_fitness = 2.0 * expected[0] + 7.0 * expected[1]
        assert positional(positional_inputs) == expected_fitness
        assert named(named_inputs) == expected_fitness
        outputs.append(positional_decoded)

    assert outputs[0][0] is True and outputs[0][1] is False
    assert outputs[0][1] is outputs[1][1]  # Changing x leaves D(y) unchanged.
    assert outputs[1][0] is outputs[2][0]  # Changing y leaves D(x) unchanged.
    assert outputs[2][1] is outputs[3][1]


@pytest.mark.statistical
@pytest.mark.parametrize(
    ("alpha", "bounds", "case_id"),
    [(None, None, 0), (2.0 * log(3.0), (-1.5, 1.5), 1)],
    ids=["default-alpha-and-domain", "alpha-log-nine-custom-domain"],
)
def test_s_shape_frequency_matches_logistic_probability_and_alpha(
    alpha: float | None, bounds: tuple[float, float] | None, case_id: int
) -> None:
    space = Binary(
        3,
        constant_objective,
        decoder="s-shape",
        bounds=bounds,
        params=None if alpha is None else {"alpha": alpha},
        use_cache=False,
    )
    seed = seed_for(case_id)
    space.initialize_context(seed=seed)
    expected_bounds = (-6.0, 6.0) if bounds is None else bounds
    assert space.encoded_boundaries == dict.fromkeys(("0", "1", "2"), expected_bounds)

    slope = 1.0 if alpha is None else alpha
    points = (-1.0, 0.0, 1.0)
    # math.exp is independent of production's expit. Custom bounds do not
    # normalize x; alpha=2*ln(3) gives p=(-1: 1/10, 0: 1/2, 1: 9/10).
    probabilities = [1.0 / (1.0 + exp(-slope * x)) for x in points]
    inputs = {str(index): x for index, x in enumerate(points)}
    # 20,000 draws per coordinate give shared six-sigma margins <= 0.022,
    # separating both slopes and detecting normalization or a reversed sign.
    sample_size = 20_000
    counts = [0, 0, 0]
    for _ in range(sample_size):
        decoded = space.decode(inputs)
        assert isinstance(decoded, list)
        assert len(decoded) == len(points)
        for index, bit in enumerate(decoded):
            assert type(bit) is bool
            counts[index] += bit

    frequencies = [count / sample_size for count in counts]
    for x, probability, count, frequency in zip(
        points, probabilities, counts, frequencies, strict=True
    ):
        assert 0 < count < sample_size  # Both Boolean states were observed.
        tolerance = frequency_tolerance(probability, sample_size)
        assert abs(frequency - probability) <= tolerance, (x, slope, seed, counts)
        print(
            f"alpha={slope}, x={x}, seed={seed}, N={sample_size},",
            f"p={probability}, frequency={frequency}, tolerance={tolerance}",
        )
    assert frequencies[0] < 0.5 < frequencies[2]
    assert abs(frequencies[1] - 0.5) <= frequency_tolerance(0.5, sample_size)


@pytest.mark.statistical
def test_s_shape_bits_are_independent_bernoulli_draws() -> None:
    space = Binary(2, constant_objective, decoder="s-shape", use_cache=False)
    space.initialize_context(seed=BASE_SEED)
    sample_size = 20_000
    counts: Counter[tuple[bool, bool]] = Counter()
    for _ in range(sample_size):
        decoded = space.decode({"0": 0.0, "1": 0.0})
        assert isinstance(decoded, list)
        assert len(decoded) == 2
        assert all(type(bit) is bool for bit in decoded)
        counts[(decoded[0], decoded[1])] += 1

    # Independent Bernoulli(1/2) draws give P(b0,b1)=1/4 for each pair.
    # A shared random scalar would keep the marginals but eliminate FT/TF.
    states = ((False, False), (False, True), (True, False), (True, True))
    assert set(counts) == set(states)
    tolerance = frequency_tolerance(0.25, sample_size)
    for state in states:
        frequency = counts[state] / sample_size
        assert abs(frequency - 0.25) <= tolerance, (state, BASE_SEED, counts)
        print(
            f"state={state}, seed={BASE_SEED}, N={sample_size},",
            f"frequency={frequency}, tolerance={tolerance}",
        )
