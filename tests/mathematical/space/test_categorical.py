"""Categorical maps: argmax, softmax sampling and uniform scalar partitions.

Jang et al. (2017), eq. (2), defines the relaxed Gumbel-Softmax vector q:
https://arxiv.org/abs/1611.01144. Tyrannis additionally draws Y | q ~ Cat(q),
so P(Y=i)=E[q_i], unlike the paper's argmax discretization (sec. 2.2).
The docstring's temperature description is consistent with this extra draw,
but does not explicitly spell out the two stages or claim strict equivalence.

Argmax ties and equal-width scalar cells are Tyrannis conventions. Neither
Cerda et al.'s similarity features (2018, eq. (11), arXiv:1806.00979) nor
Bean's random-key framework (1994, doi:10.1287/ijoc.6.2.154) is used as the
source of these decoder equations. Category values are labels, not distances.
"""

from collections import Counter
from math import exp, log, log1p
from typing import cast

import pytest

# SciPy ships no integrate stubs in this environment. Only this external import
# lacks typing; quad's default real-valued result is narrowed at its call below.
from scipy.integrate import (  # pyright: ignore[reportMissingTypeStubs]
    quad,  # pyright: ignore[reportUnknownVariableType]
)

from tests._support.assertions import assert_variables_within_bounds
from tests._support.numerics import (
    BASE_SEED,
    STRICT_ATOL,
    STRICT_RTOL,
    frequency_tolerance,
    seed_for,
)
from tests._support.objectives import constant_objective
from tyrannis.space import Categorical


def _assert_categorical_frequencies(
    space: Categorical,
    inputs: dict[str, float],
    probabilities: dict[int, float],
    *,
    decoder: str,
    seed: int,
    temperature: float | None = None,
) -> dict[int, float]:
    space.initialize_context(seed=seed)
    assert_variables_within_bounds(inputs, space.encoded_boundaries)
    # At N=20,000 the shared six-sigma margin is at most 0.022 for any p.
    # Seeds reproduce experiments; neither their sequences nor labels are oracles.
    sample_size = 20_000
    counts: Counter[int] = Counter()
    for _ in range(sample_size):
        decoded = space.decode(inputs)
        assert isinstance(decoded, list)
        assert len(decoded) == 1 and isinstance(decoded[0], int)
        counts[decoded[0]] += 1
    diagnostic = (
        f"decoder={decoder}, seed={seed}, logits={inputs}, temperature={temperature}, "
        f"N={sample_size}, probabilities={probabilities}, counts={dict(counts)}"
    )
    assert set(counts) == set(probabilities), diagnostic
    frequencies = {label: counts[label] / sample_size for label in probabilities}
    for label, probability in probabilities.items():
        assert abs(frequencies[label] - probability) <= frequency_tolerance(
            probability, sample_size
        ), diagnostic
    return frequencies


def _binary_gumbel_marginal(delta: float, temperature: float) -> float:
    # If E_i=exp(-g_i), then E_i ~ Exp(1) and g_1-g_0=log(E_0/E_1).
    # Its CDF is sigmoid(d), so D=log(U/(1-U)) has that same logistic law.
    # Total expectation for Tyrannis's extra draw gives integral E[sigmoid].
    def integrand(u: float) -> float:
        return 1.0 / (1.0 + exp(-(delta + log(u) - log1p(-u)) / temperature))

    # With full_output=0 and complex_func=False, quad returns (float, float).
    probability, error = cast(
        "tuple[float, float]",
        quad(integrand, 0.0, 1.0, epsabs=STRICT_ATOL, epsrel=STRICT_RTOL),
    )
    # A deterministic 1e-12 integration budget is negligible beside sampling
    # error (~1e-2); it is checked separately, never added to statistical slack.
    assert error <= STRICT_ATOL, (delta, temperature, probability, error)
    return probability


@pytest.mark.parametrize(
    "decoder",
    ["one-hot", "softmax", "gumbel-softmax", "scalar"],
    ids=["one-hot", "softmax", "gumbel-softmax", "scalar"],
)
def test_encoded_domain_dimension_follows_cardinalities(decoder: str) -> None:
    for choices in ([[10, 20, 1000]], [[10, 20, 1000], ["red", "blue"]]):
        dimension = len(choices) if decoder == "scalar" else sum(map(len, choices))
        for bounds in (None, (-3.0, 5.0)):
            space = Categorical(
                choices, constant_objective, decoder=decoder, bounds=bounds
            )
            space.initialize_context(seed=BASE_SEED)
            interval = bounds or ((0.0, 1.0) if decoder == "scalar" else (-1.0, 1.0))
            # The solver domain is a Cartesian product, with no simplex constraint.
            assert len(space.encoded_boundaries) == dimension
            assert list(space.encoded_boundaries.values()) == [interval] * dimension
            for endpoint in interval:
                inputs = dict.fromkeys(space.encoded_boundaries, endpoint)
                decoded = space.decode(inputs)
                assert isinstance(decoded, list)
                assert len(decoded) == len(choices)
                assert all(
                    decoded[index] in domain for index, domain in enumerate(choices)
                )


def test_one_hot_selects_relative_maximum_and_first_tie() -> None:
    choices = (10, 20, 1000)
    for bounds in (None, (-3.0, 5.0)):
        space = Categorical(choices, constant_objective, bounds=bounds)
        space.initialize_context()
        for winner, category in enumerate(choices):
            # Every coordinate wins, even with all-negative, non-unit scores.
            scores = [-0.75, -0.625, -0.5]
            scores[winner] = -0.25
            transforms = ((1.0, 0.0), (1.0, 0.875), (0.5, 0.0))
            if bounds is not None:
                transforms += ((4.0, 2.0),)
            for scale, shift in transforms:
                inputs = {
                    f"0-{choice}": scale * score + shift
                    for choice, score in zip(choices, scores, strict=True)
                }
                assert_variables_within_bounds(inputs, space.encoded_boundaries)
                assert space.decode(inputs) == [category], inputs

        # First maximum is NumPy/Tyrannis's deterministic convention, not a
        # universal property of categorical encoding; reverse input insertion.
        assert space.decode({"0-1000": 0.5, "0-20": 0.5, "0-10": -0.5}) == [20]


@pytest.mark.statistical
def test_softmax_frequencies_match_exponential_weights_and_common_shifts() -> None:
    choices = (10, 20, 1000)
    logits = (-0.75, -0.125, 0.5)
    # Independent of scipy.special.softmax; the common exponential factor
    # cancels exactly in theory under z -> z+c.
    weights = [exp(z - max(logits)) for z in logits]
    probabilities = {
        label: weight / sum(weights)
        for label, weight in zip(choices, weights, strict=True)
    }
    for case_id, (shift, bounds) in enumerate(
        ((0.0, None), (0.375, None), (3.0, (-3.0, 5.0)))
    ):
        space = Categorical(
            choices, constant_objective, decoder="softmax", bounds=bounds
        )
        inputs = {
            f"0-{label}": z + shift for label, z in zip(choices, logits, strict=True)
        }
        _ = _assert_categorical_frequencies(
            space, inputs, probabilities, decoder="softmax", seed=seed_for(case_id)
        )


@pytest.mark.statistical
def test_gumbel_softmax_marginals_integrate_extra_draw_and_temperature() -> None:
    choices = (10, 1000)
    logits = (-0.75, 0.75)
    delta = logits[1] - logits[0]
    temperatures = (0.5, 1.0, 3.0)
    probabilities = [_binary_gumbel_marginal(delta, tau) for tau in temperatures]
    # At tau=1, direct integration of a*u/(1+(a-1)*u), a=exp(delta),
    # also yields a*(a-1-log(a))/(a-1)^2: an analytic check of quadrature.
    a = exp(delta)
    assert probabilities[1] == pytest.approx(
        a * (a - 1.0 - delta) / (a - 1.0) ** 2, rel=STRICT_RTOL, abs=STRICT_ATOL
    )
    hard_probability = 1.0 / (1.0 + exp(-delta))
    assert (
        0.5 < probabilities[2] < probabilities[1] < probabilities[0] < hard_probability
    )
    # At tau=1 both incorrect alternatives, softmax(z) and softmax(z/tau),
    # are separated from the true marginal by more than two tolerance bands.
    assert hard_probability - probabilities[1] > 2.0 * frequency_tolerance(
        probabilities[1], 20_000
    )
    frequencies: list[float] = []
    for case_id, (temperature, probability) in enumerate(
        zip(temperatures, probabilities, strict=True)
    ):
        # A common shift into custom bounds preserves delta and its marginal;
        # encoded bounds must not normalize or otherwise rescale the logits.
        shift = 3.0 if temperature == 3.0 else 0.0
        space = Categorical(
            choices,
            constant_objective,
            decoder="gumbel-softmax",
            bounds=(-3.0, 5.0) if shift else None,
            params={"temperature": temperature},
        )
        inputs = {
            f"0-{label}": z + shift for label, z in zip(choices, logits, strict=True)
        }
        observed = _assert_categorical_frequencies(
            space,
            inputs,
            {10: 1.0 - probability, 1000: probability},
            decoder="gumbel-softmax",
            seed=seed_for(case_id + 10),
            temperature=temperature,
        )
        frequencies.append(observed[1000])
    separation = probabilities[0] - probabilities[2]
    uncertainty = sum(
        frequency_tolerance(p, 20_000) for p in (probabilities[0], probabilities[2])
    )
    assert separation > uncertainty
    assert abs((frequencies[0] - frequencies[2]) - separation) <= uncertainty


@pytest.mark.statistical
def test_gumbel_softmax_equal_logits_are_uniform_at_distinct_temperatures() -> None:
    # Exchangeability and sum(q)=1 imply E[q_i]=1/K for every temperature.
    choices = (10, 20, 1000)
    for case_id, temperature in enumerate((0.25, 1.0, 4.0)):
        space = Categorical(
            choices,
            constant_objective,
            decoder="gumbel-softmax",
            bounds=(-3.0, 5.0),
            params={"temperature": temperature},
        )
        _ = _assert_categorical_frequencies(
            space,
            dict.fromkeys((f"0-{label}" for label in choices), 2.0),
            dict.fromkeys(choices, 1.0 / len(choices)),
            decoder="gumbel-softmax",
            seed=seed_for(case_id + 20),
            temperature=temperature,
        )


@pytest.mark.parametrize(
    "bounds", [None, (-3.0, 5.0)], ids=["default-domain", "custom-domain"]
)
def test_scalar_cells_cover_all_labels_with_right_continuous_thresholds(
    bounds: tuple[float, float] | None,
) -> None:
    choices = (1000, -7, 20, 10)
    space = Categorical(choices, constant_objective, decoder="scalar", bounds=bounds)
    space.initialize_context()
    lower, upper = (0.0, 1.0) if bounds is None else bounds
    assert space.encoded_boundaries == {"0": (lower, upper)}
    width = (upper - lower) / len(choices)
    # Four cells make thresholds exactly representable in both domains.
    # Epsilon is tiny relative to a cell but far above floating-point ulps.
    epsilon = width * 2.0**-20
    cases = {lower: 0, upper: len(choices) - 1}
    for index in range(len(choices)):
        cases[lower + (index + 0.5) * width] = index
    for index in range(1, len(choices)):
        threshold = lower + index * width
        cases[threshold - epsilon] = index - 1
        cases[threshold] = index
        cases[threshold + epsilon] = index

    indices: list[int] = []
    for x, expected_index in sorted(cases.items()):
        decoded = space.decode({"0": x})
        assert isinstance(decoded, list)
        assert decoded == [choices[expected_index]], (bounds, x, expected_index)
        indices.append(choices.index(decoded[0]))
    assert set(indices) == set(range(len(choices)))
    # Monotone cell indices do not impose a semantic ordering on the labels.
    assert indices == sorted(indices)


@pytest.mark.parametrize(
    "decoder",
    ["one-hot", "softmax", "gumbel-softmax", "scalar"],
    ids=["one-hot", "softmax", "gumbel-softmax", "scalar"],
)
def test_single_choice_collapses_the_entire_domain(decoder: str) -> None:
    space = Categorical(["only"], constant_objective, decoder=decoder)
    space.initialize_context(seed=BASE_SEED)
    assert len(space.encoded_boundaries) == 1
    key, (lower, upper) = next(iter(space.encoded_boundaries.items()))
    # K=1 gives exactly q=1 even with Gumbel noise; no frequency estimate needed.
    for fraction in (0.0, 0.125, 0.5, 0.875, 1.0):
        x = lower + fraction * (upper - lower)
        assert space.decode({key: x}) == ["only"]


def test_categorical_coordinates_preserve_positional_and_named_evaluation() -> None:
    def positional_objective(color: str, code: int, /) -> float:
        return {"red": 2.0, "blue": 11.0}[color] + {10: 3.0, 20: 7.0, 1000: 19.0}[code]

    def named_objective(*, color: str, code: int) -> float:
        return {"red": 2.0, "blue": 11.0}[color] + {10: 3.0, 20: 7.0, 1000: 19.0}[code]

    colors = ("red", "blue")
    codes = (10, 20, 1000)
    positional = Categorical([colors, codes], positional_objective)
    named = Categorical({"code": codes, "color": colors}, named_objective)
    positional.initialize_context()
    named.initialize_context()
    assert len(positional.encoded_boundaries) == len(named.encoded_boundaries) == 5
    for color_index, code_index, fitness in ((0, 1, 9.0), (1, 1, 18.0), (1, 2, 30.0)):
        color_scores = [0.75 if index == color_index else -0.5 for index in range(2)]
        code_scores = [0.5 if index == code_index else -0.75 for index in range(3)]
        # Input insertion reverses each configured block and its coordinates.
        positional_inputs = {
            "1-1000": code_scores[2],
            "1-20": code_scores[1],
            "1-10": code_scores[0],
            "0-blue": color_scores[1],
            "0-red": color_scores[0],
        }
        named_inputs = {
            "color-blue": color_scores[1],
            "color-red": color_scores[0],
            "code-1000": code_scores[2],
            "code-20": code_scores[1],
            "code-10": code_scores[0],
        }
        assert positional.decode(positional_inputs) == [
            colors[color_index],
            codes[code_index],
        ]
        assert named.decode(named_inputs) == {
            "color": colors[color_index],
            "code": codes[code_index],
        }
        assert positional(positional_inputs) == fitness
        assert named(named_inputs) == fitness
