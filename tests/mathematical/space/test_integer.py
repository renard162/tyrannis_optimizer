"""Integer decoders as maps from continuous intervals to integer lattices.

Round, affine scaling and alpha-logistic mapping follow the Tyrannis contract;
nearest-integer ties go to the even integer. Stochastic rounding specializes
Connolly, Higham & Mary (2021), eq. (1.1), to a lattice with unit spacing:
P(D(x) = floor(x) + 1) = x - floor(x), hence E[D(x)] = x.
Reference: https://doi.org/10.1137/20M1334796 (43(1), A566-A585).

The alpha-logistic followed by affine scaling and rounding is a Tyrannis
contract, not attributed to Chen et al. (2010), whose eq. (4.2) concerns binary
PSO positions rather than this integer decoder.
"""

from collections import Counter
from math import exp, floor, log

import pytest

from tests._support.assertions import assert_variables_within_bounds
from tests._support.numerics import BASE_SEED, frequency_tolerance, seed_for
from tests._support.objectives import constant_objective
from tyrannis.space import Integer


def test_round_decoder_matches_integer_partitions_and_ties_to_even() -> None:
    lower, upper = -3, 4
    bounds = {"count": (lower, upper)}
    space = Integer(bounds, constant_objective, decoder="round", use_cache=False)
    space.initialize_context()
    assert space.encoded_boundaries == bounds

    # All eight integers witness surjectivity; quarter points lie inside cells.
    cases = {float(k): k for k in range(lower, upper + 1)}
    for k in range(lower, upper + 1):
        for offset in (-0.25, 0.25):
            if lower <= k + offset <= upper:
                cases[k + offset] = k

    # Half-integers and parity are exact in binary. Epsilon is far above ulps.
    epsilon = 2.0**-20
    for k in range(lower, upper):
        threshold = k + 0.5
        cases[threshold - epsilon] = k
        cases[threshold] = k if k % 2 == 0 else k + 1
        cases[threshold + epsilon] = k + 1

    outputs: list[int] = []
    for x, expected in sorted(cases.items()):
        decoded = space.decode({"count": x})
        assert isinstance(decoded, dict)
        assert decoded == {"count": expected}, x
        assert isinstance(decoded["count"], int)
        assert_variables_within_bounds(decoded, bounds)
        outputs.append(decoded["count"])

    assert set(outputs) == set(range(lower, upper + 1))
    assert outputs == sorted(outputs)


def test_transfer_function_matches_logistic_mapping() -> None:
    # Width 1000 makes the finite logistic tails distinguishable after rounding.
    lower, upper = -300, 700
    bounds = {"count": (lower, upper)}
    space = Integer(
        bounds, constant_objective, decoder="transfer_function", use_cache=False
    )
    space.initialize_context()
    assert space.encoded_boundaries == {"count": (-6.0, 6.0)}

    outputs: list[int] = []
    for x in (-6.0, -2.0, -0.5, 0.0, 0.5, 2.0, 6.0):
        probability = 1.0 / (1.0 + exp(-x))
        assert 0.0 < probability < 1.0
        expected = round(lower + probability * (upper - lower))
        decoded = space.decode({"count": x})
        assert isinstance(decoded, dict)
        assert decoded == {"count": expected}, x
        assert isinstance(decoded["count"], int)
        assert_variables_within_bounds(decoded, bounds)
        if x == 0.0:
            assert decoded["count"] == (lower + upper) // 2
        outputs.append(decoded["count"])

    assert outputs == sorted(outputs)
    # Finite encoded bounds do not guarantee surjectivity onto the full lattice.
    assert lower < outputs[0]
    assert outputs[-1] < upper


def test_transfer_function_alpha_and_custom_bounds_preserve_declared_geometry() -> None:
    lower, upper = -30, 71
    bounds = {"count": (lower, upper)}
    custom_bounds = (-3.0, 2.0)
    negative_outputs: list[int] = []
    positive_outputs: list[int] = []
    epsilon = 2.0**-20
    for alpha in (0.5, 2.0):
        space = Integer(
            bounds,
            constant_objective,
            decoder="transfer_function",
            custom_bounds=custom_bounds,
            params={"alpha": alpha},
            use_cache=False,
        )
        space.initialize_context()
        assert space.encoded_boundaries == {"count": custom_bounds}

        k = 30
        probability = (k + 0.5 - lower) / (upper - lower)
        threshold = log(probability / (1.0 - probability)) / alpha
        assert (
            custom_bounds[0]
            < threshold - epsilon
            < threshold + epsilon
            < custom_bounds[1]
        )
        # Avoid an artificial floating-point tie at the nonzero inverse-logistic
        # threshold. At x=0, S=1/2 and s=20.5 are exact: ties-to-even gives 20.
        transition_cases = {
            -epsilon: 20,
            0.0: 20,
            epsilon: 21,
            threshold - epsilon: k,
            threshold + epsilon: k + 1,
        }
        points = sorted({-3.0, -1.0, 0.0, 1.0, 2.0, *transition_cases})
        outputs: list[int] = []
        for x in points:
            expected = round(lower + (upper - lower) / (1.0 + exp(-alpha * x)))
            decoded = space.decode({"count": x})
            assert isinstance(decoded, dict)
            assert decoded == {"count": expected}, (alpha, x)
            assert isinstance(decoded["count"], int)
            assert_variables_within_bounds(decoded, bounds)
            if x in transition_cases:
                assert decoded["count"] == transition_cases[x]
            if x == -1.0:
                negative_outputs.append(decoded["count"])
            if x == 1.0:
                positive_outputs.append(decoded["count"])
            outputs.append(decoded["count"])
        assert outputs == sorted(outputs)

    # Larger alpha steepens both sides, with a width that keeps the effect visible.
    assert negative_outputs[1] < negative_outputs[0]
    assert positive_outputs[1] > positive_outputs[0]


def test_integer_coordinates_preserve_positional_and_named_evaluation() -> None:
    def positional_objective(count: int, level: int, /) -> float:
        assert isinstance(count, int) and isinstance(level, int)
        return 2.0 * count + 7.0 * level

    def named_objective(*, count: int, level: int) -> float:
        assert isinstance(count, int) and isinstance(level, int)
        return 2.0 * count + 7.0 * level

    bounds = {"count": (-3, 4), "level": (10, 20)}
    positional = Integer(
        [bounds["count"], bounds["level"]],
        positional_objective,
        decoder="scaling",
        use_cache=False,
    )
    named = Integer(
        {"level": bounds["level"], "count": bounds["count"]},
        named_objective,
        decoder="scaling",
        use_cache=False,
    )
    positional.initialize_context()
    named.initialize_context()
    assert positional.encoded_boundaries == {"0": (0.0, 1.0), "1": (0.0, 1.0)}
    assert named.encoded_boundaries == {"level": (0.0, 1.0), "count": (0.0, 1.0)}

    for x, y in ((0.25, 0.75), (0.75, 0.125)):
        count = round(-3 + x * 7)
        level = round(10 + y * 10)
        # Reverse input insertion order relative to the configured coordinates.
        positional_inputs = {"1": y, "0": x}
        named_inputs = {"count": x, "level": y}
        positional_decoded = positional.decode(positional_inputs)
        named_decoded = named.decode(named_inputs)
        assert isinstance(positional_decoded, list)
        assert isinstance(named_decoded, dict)
        assert positional_decoded == [count, level]
        assert named_decoded == {"count": count, "level": level}
        assert all(isinstance(value, int) for value in positional_decoded)
        assert all(isinstance(value, int) for value in named_decoded.values())
        assert_variables_within_bounds(named_decoded, bounds)
        assert_variables_within_bounds(
            dict(zip(("count", "level"), positional_decoded, strict=True)), bounds
        )
        expected_fitness = 2.0 * count + 7.0 * level
        assert positional(positional_inputs) == expected_fitness
        assert named(named_inputs) == expected_fitness


def test_scaling_decoder_is_affine_then_nearest_integer() -> None:
    lower, upper = -3, 5
    bounds = {"count": (lower, upper)}
    space = Integer(bounds, constant_objective, decoder="scaling", use_cache=False)
    space.initialize_context()
    assert space.encoded_boundaries == {"count": (0.0, 1.0)}

    # Every integer has preimage (k-L)/(U-L), including both endpoints.
    preimages = {(k - lower) / (upper - lower): k for k in range(lower, upper + 1)}
    points = set(preimages)
    epsilon = 2.0**-20
    # Both parities: s(x*) = -1.5 -> -2 and 1.5 -> 2, exactly representable.
    for k in (-2, 1):
        threshold = (k + 0.5 - lower) / (upper - lower)
        points.update((threshold - epsilon, threshold, threshold + epsilon))
    points.update((1.0 / 32.0, 23.0 / 32.0))

    outputs: list[int] = []
    for x in sorted(points):
        expected = round(lower + x * (upper - lower))
        decoded = space.decode({"count": x})
        assert isinstance(decoded, dict)
        assert decoded == {"count": expected}, x
        assert isinstance(decoded["count"], int)
        assert_variables_within_bounds(decoded, bounds)
        if x in preimages:
            assert decoded["count"] == preimages[x]
        outputs.append(decoded["count"])

    assert outputs[0] == lower
    assert outputs[-1] == upper
    assert set(outputs) == set(range(lower, upper + 1))
    assert outputs == sorted(outputs)


@pytest.mark.statistical
def test_stochastic_rounding_matches_bernoulli_probabilities_and_is_unbiased() -> None:
    bounds = {"count": (-3, 4)}
    space = Integer(
        bounds, constant_objective, decoder="stochastic_round", use_cache=False
    )
    space.initialize_context(seed=BASE_SEED)
    assert space.encoded_boundaries == bounds

    # r=0 is deterministic, including the endpoints of the encoded domain.
    for k in (-3, 0, 4):
        decoded = space.decode({"count": float(k)})
        assert isinstance(decoded, dict)
        assert decoded == {"count": k}
        assert isinstance(decoded["count"], int)
        assert_variables_within_bounds(decoded, bounds)

    # At p=1/4 or 3/4, 20,000 trials give a six-sigma margin below 0.02.
    # No PRNG sequence is an oracle: seeds only make the experiment repeatable.
    sample_size = 20_000
    for case_id, x in enumerate((1.25, -1.25)):
        seed = seed_for(case_id)
        space.initialize_context(seed=seed)
        lower_integer = floor(x)
        upper_probability = x - lower_integer
        counts: Counter[int] = Counter()
        for _ in range(sample_size):
            decoded = space.decode({"count": x})
            assert isinstance(decoded, dict)
            assert isinstance(decoded["count"], int)
            assert_variables_within_bounds(decoded, bounds)
            counts[decoded["count"]] += 1

        assert set(counts) == {lower_integer, lower_integer + 1}
        for integer, probability in (
            (lower_integer, 1.0 - upper_probability),
            (lower_integer + 1, upper_probability),
        ):
            frequency = counts[integer] / sample_size
            assert abs(frequency - probability) <= frequency_tolerance(
                probability, sample_size
            ), (x, seed, counts)

        mean = sum(integer * count for integer, count in counts.items()) / sample_size
        # D=n+B, B~Bernoulli(r): mean(D)-x = frequency(B=1)-r.
        # The spacing is one, so the same six-sigma frequency bound applies
        # to the mean in integer units: SD(mean)=sqrt(r*(1-r)/N).
        mean_tolerance = frequency_tolerance(upper_probability, sample_size)
        assert abs(mean - x) <= mean_tolerance, (x, seed, mean)
        print(
            f"x={x}, seed={seed}, N={sample_size}, counts={dict(counts)},",
            f"mean={mean}, six_sigma={mean_tolerance}",
        )
