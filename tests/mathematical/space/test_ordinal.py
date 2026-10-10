"""Ordinal maps: nearest cells, unbiased rounding and categorical laws.

Croci et al. (2022), sec. 2, supplies stochastic rounding, here applied to
configured ordinal positions: https://doi.org/10.1098/rsos.211631.
Cumulative links follow McCullagh (1980), sec. 2 (logistic and normal):
https://doi.org/10.1111/j.2517-6161.1980.tb01109.x; the probit antecedent is
Aitchison & Silvey (1957): https://doi.org/10.1093/biomet/44.1-2.131.

Equidistant defaults, half-gap bounds, midpoint cut points, lower-level ties
and temperature-scaled absolute-distance softmax are Tyrannis contracts.
Bradley et al. (1962), https://doi.org/10.1007/BF02289644, studies optimal
ordinal scores, not these fixed positions. Snell et al. (2017), eq. (2),
https://arxiv.org/abs/1703.05175, gives softmax over distances; the particular
law exp(-abs(p_i-x)/T) tested here is the Tyrannis specialization.
"""

from collections import Counter
from collections.abc import Callable
from itertools import pairwise
from math import erf, exp, sqrt

import pytest

from tests._support.numerics import (
    BASE_SEED,
    STRICT_ATOL,
    STRICT_RTOL,
    frequency_tolerance,
    seed_for,
)
from tests._support.objectives import constant_objective
from tyrannis.space import Ordinal


def _logistic_cdf(z: float) -> float:
    return 1.0 / (1.0 + exp(-z))


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def _check_categorical_law(
    decoder: str,
    positions: tuple[float, ...],
    x: float,
    temperature: float,
    probabilities: list[float],
    seed: int,
) -> list[float]:
    """Sample the public decoder; the caller supplies an independent law."""
    choices = ("low", "medium", "high")
    space = Ordinal(
        choices,
        constant_objective,
        positions=positions,
        decoder=decoder,
        params={"temperature": temperature},
        use_cache=False,
    )
    space.initialize_context(seed=seed)
    # Each category indicator is Bernoulli(p), even in a multinomial sample.
    # N=5,000 gives 6*sqrt(p*(1-p)/N) <= 0.043. floor=0 retains the
    # variance-derived bound for smaller categories instead of a fixed floor.
    sample_size = 5_000
    counts: Counter[str] = Counter()
    for _ in range(sample_size):
        decoded = space.decode({"0": x})
        assert isinstance(decoded, list)
        counts[decoded[0]] += 1
    diagnostic = (
        f"decoder={decoder}, seed={seed}, x={x}, T={temperature}, "
        f"N={sample_size}, counts={dict(counts)}, expected={probabilities}"
    )
    assert set(counts) == set(choices), diagnostic
    assert sum(probabilities) == pytest.approx(1.0, rel=STRICT_RTOL, abs=STRICT_ATOL)
    frequencies = [counts[choice] / sample_size for choice in choices]
    for frequency, probability in zip(frequencies, probabilities, strict=True):
        assert 0.0 < probability < 1.0
        assert abs(frequency - probability) <= frequency_tolerance(
            probability, sample_size, floor=0.0
        ), diagnostic
    print(diagnostic)
    return frequencies


@pytest.mark.parametrize(
    "positions", [None, (-4.0, -2.0, 4.0, 8.0)], ids=["default", "nonuniform"]
)
def test_position_geometry_defines_bounds_and_level_preimages(
    positions: tuple[float, ...] | None,
) -> None:
    choices = ("low", "medium", "high", "peak")
    points = (
        tuple(float(i) for i in range(len(choices))) if positions is None else positions
    )
    lower = points[0] - (points[1] - points[0]) / 2.0
    upper = points[-1] + (points[-1] - points[-2]) / 2.0
    for decoder in ("rank", "nearest-stochastic"):
        space = Ordinal(
            choices,
            constant_objective,
            positions=positions,
            decoder=decoder,
            use_cache=False,
        )
        space.initialize_context(seed=BASE_SEED)
        assert space.encoded_boundaries == {"0": (lower, upper)}
        for point, choice in zip(points, choices, strict=True):
            assert lower <= point <= upper
            assert space.decode({"0": point}) == [choice]
        # Extrapolated half-gaps belong to the extreme levels, including endpoints.
        for x in (lower, (lower + points[0]) / 2.0):
            assert space.decode({"0": x}) == [choices[0]]
        for x in ((points[-1] + upper) / 2.0, upper):
            assert space.decode({"0": x}) == [choices[-1]]


def test_rank_partitions_at_midpoints_with_lower_level_ties() -> None:
    choices = ("low", "medium", "high", "peak")
    positions = (-4.0, -2.0, 4.0, 8.0)
    thresholds = [(a + b) / 2.0 for a, b in pairwise(positions)]
    lower = positions[0] - (positions[1] - positions[0]) / 2.0
    upper = positions[-1] + (positions[-1] - positions[-2]) / 2.0
    space = Ordinal(choices, constant_objective, positions=positions, use_cache=False)
    space.initialize_context()

    cases = dict(zip(positions, range(len(choices)), strict=True))
    edges = [lower, *thresholds, upper]
    for index, (left, right) in enumerate(pairwise(edges)):
        cases[(left + right) / 2.0] = index
    cases[lower], cases[upper] = 0, len(choices) - 1
    # Integer midpoints and binary epsilon are exact; epsilon exceeds local ulps
    # without approaching another cell boundary, even across unequal gaps.
    epsilon = 2.0**-20
    for index, threshold in enumerate(thresholds):
        cases[threshold - epsilon] = index
        cases[threshold] = index
        cases[threshold + epsilon] = index + 1

    observed: list[int] = []
    for x, expected in sorted(cases.items()):
        decoded = space.decode({"0": x})
        assert decoded == [choices[expected]], (x, expected)
        assert isinstance(decoded, list)
        observed.append(choices.index(decoded[0]))
    assert observed == sorted(observed)
    assert set(observed) == set(range(len(choices)))


@pytest.mark.statistical
def test_nearest_stochastic_is_unbiased_on_nonuniform_positions() -> None:
    choices = ("low", "medium", "high", "peak")
    positions = (-4.0, -2.0, 4.0, 8.0)
    space = Ordinal(
        choices,
        constant_objective,
        positions=positions,
        decoder="nearest-stochastic",
        use_cache=False,
    )
    space.initialize_context(seed=BASE_SEED)
    for point, choice in zip(positions, choices, strict=True):
        assert space.decode({"0": point}) == [choice]
    for x, choice in ((-5.0, "low"), (-4.5, "low"), (9.0, "peak"), (10.0, "peak")):
        assert space.decode({"0": x}) == [choice]

    sample_size = 5_000
    for index, upper_probability in enumerate((0.25, 0.75, 0.5)):
        lower, upper = positions[index : index + 2]
        gap = upper - lower
        x = lower + upper_probability * gap
        seed = seed_for(index)
        space.initialize_context(seed=seed)
        counts: Counter[str] = Counter()
        for _ in range(sample_size):
            decoded = space.decode({"0": x})
            assert isinstance(decoded, list)
            counts[decoded[0]] += 1
        probabilities = {
            choices[index]: 1.0 - upper_probability,
            choices[index + 1]: upper_probability,
        }
        diagnostic = (
            f"decoder=nearest-stochastic, seed={seed}, x={x}, "
            f"N={sample_size}, counts={dict(counts)}, expected={probabilities}"
        )
        assert set(counts) == set(probabilities), diagnostic
        for choice, probability in probabilities.items():
            assert abs(
                counts[choice] / sample_size - probability
            ) <= frequency_tolerance(probability, sample_size, floor=0.0), diagnostic
        mean = (
            sum(
                point * counts[choice]
                for point, choice in zip(positions, choices, strict=True)
            )
            / sample_size
        )
        # p_Y = lower + gap*B, B~Bernoulli(q): E[p_Y]=x and
        # SD(mean(p_Y))=gap*sqrt(q*(1-q)/N), not the SD of level indices.
        mean_tolerance = gap * frequency_tolerance(
            upper_probability, sample_size, floor=0.0
        )
        assert abs(mean - x) <= mean_tolerance, (diagnostic, mean, mean_tolerance)
        print(diagnostic, f"mean={mean}, six_sigma={mean_tolerance}")


@pytest.mark.statistical
@pytest.mark.parametrize(
    ("decoder", "cdf", "case_id"),
    [("cumulative-logit", _logistic_cdf, 10), ("cumulative-probit", _normal_cdf, 20)],
    ids=["logistic", "normal"],
)
def test_cumulative_links_match_cutpoint_laws_and_temperature(
    decoder: str,
    cdf: Callable[[float], float],
    case_id: int,
) -> None:
    positions = (-2.0, 0.0, 3.0)
    thresholds = [(a + b) / 2.0 for a, b in pairwise(positions)]
    x = 0.5  # Strictly inside the middle cell, asymmetrically between cut points.
    expected: list[list[float]] = []
    observed: list[list[float]] = []
    for index, temperature in enumerate((1.0, 2.0)):
        # F_i=P(Y<=i). Add F_{-1}=0 and F_{K-1}=1 so that each
        # categorical probability is F_i-F_{i-1}, including both tails.
        cumulative = [0.0, *(cdf((t - x) / temperature) for t in thresholds), 1.0]
        probabilities = [right - left for left, right in pairwise(cumulative)]
        expected.append(probabilities)
        observed.append(
            _check_categorical_law(
                decoder,
                positions,
                x,
                temperature,
                probabilities,
                seed_for(case_id + index),
            )
        )

    # Increasing T moves both finite CDFs toward 1/2: each tail grows,
    # while the middle cell loses mass. This is not uniform softmax flattening.
    for distributions in (expected, observed):
        cold, warm = distributions
        assert cold[0] < warm[0]
        assert cold[1] > warm[1]
        assert cold[2] < warm[2]


@pytest.mark.statistical
def test_distance_softmax_matches_absolute_distances_and_temperature() -> None:
    positions = (-2.0, 0.0, 3.0)
    x = 0.75  # Distances (2.75, 0.75, 2.25), all different.
    expected: list[list[float]] = []
    observed: list[list[float]] = []
    for index, temperature in enumerate((1.0, 2.0)):
        weights = [exp(-abs(point - x) / temperature) for point in positions]
        probabilities = [weight / sum(weights) for weight in weights]
        expected.append(probabilities)
        observed.append(
            _check_categorical_law(
                "distance-softmax",
                positions,
                x,
                temperature,
                probabilities,
                seed_for(30 + index),
            )
        )
    for distributions in (expected, observed):
        cold, warm = distributions
        assert cold[1] > cold[2] > cold[0]
        assert warm[1] > warm[2] > warm[0]
        assert cold[1] > warm[1]  # Lower T favors the uniquely nearest position.
        # Flattening decreases the probability range toward the uniform limit.
        assert max(cold) - min(cold) > max(warm) - min(warm)


@pytest.mark.statistical
@pytest.mark.parametrize(
    "decoder",
    [
        "rank",
        "nearest-stochastic",
        "cumulative-logit",
        "cumulative-probit",
        "distance-softmax",
    ],
    ids=[
        "rank",
        "nearest-stochastic",
        "cumulative-logit",
        "cumulative-probit",
        "distance-softmax",
    ],
)
def test_single_choice_is_a_point_mass_over_the_encoded_interval(decoder: str) -> None:
    position = -2.25
    space = Ordinal(
        ["only"],
        constant_objective,
        positions=[position],
        decoder=decoder,
        use_cache=False,
    )
    space.initialize_context(seed=BASE_SEED)
    assert space.encoded_boundaries == {"0": (position - 0.5, position + 0.5)}
    # With singleton support, every draw has mass one: no statistical tolerance.
    for offset in (-0.5, -0.25, 0.0, 0.25, 0.5):
        assert space.decode({"0": position + offset}) == ["only"]


def test_coordinates_preserve_positional_and_named_objective_arguments() -> None:
    def positional_objective(size: int, grade: int, /) -> float:
        return 2.0 * size + 7.0 * grade

    def named_objective(*, size: int, grade: int) -> float:
        return 2.0 * size + 7.0 * grade

    sizes, grades = (11, 23, 47), (101, 307)
    size_positions, grade_positions = (-4.0, -2.0, 4.0), (10.0, 18.0)
    positional = Ordinal(
        [sizes, grades],
        positional_objective,
        positions=[size_positions, grade_positions],
        use_cache=False,
    )
    named = Ordinal(
        {"grade": grades, "size": sizes},
        named_objective,
        positions={"size": size_positions, "grade": grade_positions},
        use_cache=False,
    )
    positional.initialize_context()
    named.initialize_context()
    assert positional.encoded_boundaries == {"0": (-5.0, 7.0), "1": (6.0, 22.0)}
    assert named.encoded_boundaries == {"grade": (6.0, 22.0), "size": (-5.0, 7.0)}
    for x, y, size, grade in ((-3.5, 17.0, 11, 307), (3.0, 11.0, 47, 101)):
        # Reverse insertion order relative to each configured variable order.
        positional_inputs = {"1": y, "0": x}
        named_inputs = {"size": x, "grade": y}
        assert positional.decode(positional_inputs) == [size, grade]
        assert named.decode(named_inputs) == {"size": size, "grade": grade}
        expected = 2.0 * size + 7.0 * grade
        assert positional(positional_inputs) == expected
        assert named(named_inputs) == expected
