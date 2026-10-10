"""Permutation maps: Bean (1994), random keys, doi:10.1287/ijoc.6.2.154;
Yellott (1977), Gumbel choice laws, doi:10.1016/0022-2496(77)90026-8;
Plackett (1975), sequential rankings, doi:10.2307/2346567;
Mena et al. (2018), Sinkhorn relaxation, https://arxiv.org/abs/1802.08665.

Stable ties, ascending noisy keys, exp(r/T) weights, rows as choices and
maximum assignment on the finite Sinkhorn output are Tyrannis conventions.
Ascending Gumbel keys are not identified with the Plackett-Luce ranking law.
For Sinkhorn, temperature softens S((V+G)/T); it does not remove Gumbel noise.
The docstring's generic "more deterministic" wording must not be read as a
noise-free ranking limit (Mena, sec. 4). Hardening the finite S is tested here.
"""

from collections import Counter
from collections.abc import Callable
from itertools import permutations
from math import exp, prod
from typing import cast, override

import numpy as np
import pytest
from numpy.typing import NDArray

from tests._support.assertions import (
    assert_valid_permutation,
    assert_variables_within_bounds,
)
from tests._support.numerics import (
    BASE_SEED,
    STRICT_ATOL,
    STRICT_RTOL,
    frequency_tolerance,
    seed_for,
)
from tests._support.objectives import constant_objective
from tyrannis.space import Permutation


def _assert_ranking_law(
    space: Permutation,
    choices: tuple[int, ...],
    scores: tuple[float, ...],
    probabilities: dict[tuple[int, ...], float],
    *,
    decoder: str,
    seed: int,
    temperature: float,
) -> tuple[dict[tuple[int, ...], float], str]:
    """Sample real decode calls; the caller supplies the independent law."""
    space.initialize_context(seed=seed)
    inputs = dict(zip(space.encoded_variable_names, scores, strict=True))
    assert_variables_within_bounds(inputs, space.encoded_boundaries)
    # Each ranking indicator is Bernoulli(p), including multinomial samples.
    # N=5,000 gives a six-sigma margin <= 0.043; no RNG replay is an oracle.
    sample_size = 5_000
    counts: Counter[tuple[int, ...]] = Counter()
    for _ in range(sample_size):
        decoded = space.decode(inputs)
        assert isinstance(decoded, list)
        assert len(decoded) == 1
        assert_valid_permutation(decoded[0], choices)
        counts[tuple(decoded[0])] += 1
    diagnostic = (
        f"decoder={decoder}, seed={seed}, temperature={temperature}, inputs={inputs}, "
        f"N={sample_size}, probabilities={probabilities}, counts={dict(counts)}"
    )
    assert sum(probabilities.values()) == pytest.approx(
        1.0, rel=STRICT_RTOL, abs=STRICT_ATOL
    ), diagnostic
    frequencies = {order: counts[order] / sample_size for order in probabilities}
    for order, probability in probabilities.items():
        assert abs(frequencies[order] - probability) <= frequency_tolerance(
            probability, sample_size, floor=0.0
        ), diagnostic
    return frequencies, diagnostic


def _plackett_luce_law(
    choices: tuple[int, ...], scores: tuple[float, ...], temperature: float
) -> dict[tuple[int, ...], float]:
    """P(pi)=product_t w[pi_t]/sum_{j>=t} w[pi_j], w_i=exp(r_i/T)."""
    weights = {
        choice: exp(score / temperature)
        for choice, score in zip(choices, scores, strict=True)
    }
    return {
        order: prod(
            weights[choice] / sum(weights[other] for other in order[position:])
            for position, choice in enumerate(order)
        )
        for order in permutations(choices)
    }


def _one_sinkhorn_iteration(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Scalar row probabilities followed by column normalization (Mena, eq. 1)."""
    shift = max(float(value) for value in matrix.flat)
    flat = [exp(float(value) - shift) for value in matrix.flat]
    width = len(matrix)
    weights = [flat[start : start + width] for start in range(0, len(flat), width)]
    rows = [[weight / sum(row) for weight in row] for row in weights]
    columns = [sum(column) for column in zip(*rows, strict=True)]
    return np.asarray(
        [
            [value / total for value, total in zip(row, columns, strict=True)]
            for row in rows
        ]
    )


@pytest.mark.parametrize(
    "decoder",
    ["random-keys", "gumbel-random-keys", "plackett-luce", "gumbel-sinkhorn"],
    ids=["random-keys", "gumbel-random-keys", "plackett-luce", "gumbel-sinkhorn"],
)
def test_encoded_geometry_is_a_product_of_score_intervals(decoder: str) -> None:
    for choices in ([[10, 1000, -7]], [[10, 1000, -7], ["red", "blue"]]):
        power = 2 if decoder == "gumbel-sinkhorn" else 1
        dimension = sum(len(block) ** power for block in choices)
        for bounds in (None, (-3.0, 5.0)):
            space = Permutation(
                choices, constant_objective, decoder=decoder, bounds=bounds
            )
            space.initialize_context(seed=BASE_SEED)
            interval = bounds or (0.0, 1.0)
            assert len(space.encoded_boundaries) == dimension
            assert list(space.encoded_boundaries.values()) == [interval] * dimension
            for endpoint in interval:
                decoded = space.decode(
                    dict.fromkeys(space.encoded_boundaries, endpoint)
                )
                assert isinstance(decoded, list)
                for output, block in zip(decoded, choices, strict=True):
                    assert_valid_permutation(output, block)


def test_random_keys_sort_ascending_preserve_ties_and_positive_affine_maps() -> None:
    choices = (10, 1000, -7, 42)  # Identifiers, not numerical magnitudes.
    # Every choice occupies every position across the cyclic orders.
    orders = [(0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2), (3, 2, 1, 0)]
    for bounds in (None, (-3.0, 5.0)):
        space = Permutation(choices, constant_objective, bounds=bounds)
        space.initialize_context(seed=BASE_SEED)
        for order in orders:
            scores = [0.0] * len(choices)
            for index, value in zip(order, (0.125, 0.25, 0.625, 0.875), strict=True):
                scores[index] = value
            transforms = ((1.0, 0.0), (0.5, 0.125))
            if bounds is not None:
                transforms += ((4.0, -2.5),)
            for scale, shift in transforms:
                inputs = dict(
                    zip(
                        space.encoded_variable_names,
                        (scale * value + shift for value in scores),
                        strict=True,
                    )
                )
                assert_variables_within_bounds(inputs, space.encoded_boundaries)
                decoded = space.decode(dict(reversed(list(inputs.items()))))
                assert isinstance(decoded, list)
                assert_valid_permutation(decoded[0], choices)
                assert decoded == [[choices[index] for index in order]]
        # Stable ties follow configured choice order, not input insertion order.
        # This deterministic convention is specific to Tyrannis, not Bean's law.
        for scores, order in (
            ((0.5, 0.125, 0.5, 0.125), (1, 3, 0, 2)),
            ((0.5, 0.5, 0.5, 0.5), (0, 1, 2, 3)),
        ):
            inputs = dict(zip(space.encoded_variable_names, scores, strict=True))
            decoded = space.decode(dict(reversed(list(inputs.items()))))
            assert isinstance(decoded, list)
            assert_valid_permutation(decoded[0], choices)
            assert decoded == [[choices[index] for index in order]]


@pytest.mark.parametrize(
    "decoder",
    ["random-keys", "gumbel-random-keys", "plackett-luce", "gumbel-sinkhorn"],
    ids=["random-keys", "gumbel-random-keys", "plackett-luce", "gumbel-sinkhorn"],
)
def test_singleton_is_a_point_mass_for_every_score(decoder: str) -> None:
    space = Permutation(["only"], constant_objective, decoder=decoder)
    space.initialize_context(seed=BASE_SEED)
    assert len(space.encoded_boundaries) == 1  # K = K**2 = 1.
    for value in (0.0, 0.125, 0.5, 1.0):
        decoded = space.decode(dict.fromkeys(space.encoded_boundaries, value))
        assert isinstance(decoded, list)
        assert_valid_permutation(decoded[0], ["only"])
        assert decoded == [["only"]]


@pytest.mark.statistical
def test_gumbel_random_keys_precedence_is_logistic_with_temperature() -> None:
    choices = (10, 1000)
    delta = 0.75
    temperatures = (0.25, 1.0, 4.0)
    # E_i=exp(-G_i) are iid Exp(1). Integrating their ratio gives
    # P(G_A-G_B < d)=1/(1+exp(-d)). Ascending scores give d=(r_B-r_A)/T.
    probabilities = [1.0 / (1.0 + exp(-delta / t)) for t in temperatures]
    assert 0.5 < probabilities[2] < probabilities[1] < probabilities[0] < 1.0
    observed: list[float] = []
    for case_id, (temperature, probability) in enumerate(
        zip(temperatures, probabilities, strict=True)
    ):
        custom = temperature == 1.0
        space = Permutation(
            choices,
            constant_objective,
            decoder="gumbel-random-keys",
            bounds=(-3.0, 5.0) if custom else None,
            params={} if custom else {"temperature": temperature},
        )
        # A custom interval must not divide the score gap by its width (8).
        scores = (-1.0, -0.25) if custom else (0.125, 0.875)
        frequencies, _ = _assert_ranking_law(
            space,
            choices,
            scores,
            {choices: probability, choices[::-1]: 1.0 - probability},
            decoder="gumbel-random-keys",
            seed=seed_for(10 + case_id),
            temperature=temperature,
        )
        observed.append(frequencies[choices])
    # The analytic bands are separated, so they prove the temperature effect.
    for index in range(2):
        separation = probabilities[index] - probabilities[index + 1]
        uncertainty = sum(
            frequency_tolerance(p, 5_000, floor=0.0)
            for p in probabilities[index : index + 2]
        )
        assert separation > uncertainty
        assert abs(observed[index] - observed[index + 1] - separation) <= uncertainty


@pytest.mark.statistical
@pytest.mark.parametrize(
    "decoder",
    ["gumbel-random-keys", "gumbel-sinkhorn"],
    ids=["iid-key-rankings", "iid-matrix-assignments"],
)
def test_equal_scores_give_all_six_rankings_by_exchangeability(decoder: str) -> None:
    choices = (10, 1000, -7)
    # Relabeling iid keys/rows leaves the law unchanged and acts transitively
    # on the six rankings. Sinkhorn and unique assignment respect row relabeling.
    probabilities: dict[tuple[int, ...], float] = dict.fromkeys(
        permutations(choices), 1.0 / 6.0
    )
    dimension = 9 if decoder == "gumbel-sinkhorn" else 3
    space = Permutation(choices, constant_objective, decoder=decoder)
    _ = _assert_ranking_law(
        space,
        choices,
        (0.5,) * dimension,
        probabilities,
        decoder=decoder,
        seed=BASE_SEED,
        temperature=1.0,
    )


@pytest.mark.statistical
def test_plackett_luce_full_law_first_position_shift_and_temperature() -> None:
    choices, scores = (10, 1000, -7), (0.0, 0.25, 0.875)
    first_laws: list[list[float]] = []
    for case_id, (temperature, shift) in enumerate(
        ((0.25, 0.0), (1.0, 0.0), (4.0, 0.0), (1.0, 2.0))
    ):
        probabilities = _plackett_luce_law(choices, scores, temperature)
        shifted = tuple(score + shift for score in scores)
        # exp((r+c)/T)=exp(c/T)*exp(r/T); each conditional factor cancels.
        assert _plackett_luce_law(choices, shifted, temperature) == pytest.approx(
            probabilities, rel=STRICT_RTOL, abs=STRICT_ATOL
        )
        space = Permutation(
            choices,
            constant_objective,
            decoder="plackett-luce",
            bounds=(-3.0, 5.0) if shift else None,
            params={} if temperature == 1.0 else {"temperature": temperature},
        )
        frequencies, diagnostic = _assert_ranking_law(
            space,
            choices,
            shifted,
            probabilities,
            decoder="plackett-luce",
            seed=seed_for(20 + case_id),
            temperature=temperature,
        )
        weights = [exp(score / temperature) for score in scores]
        first = [weight / sum(weights) for weight in weights]
        for choice, probability in zip(choices, first, strict=True):
            marginal = sum(
                p for order, p in probabilities.items() if order[0] == choice
            )
            assert marginal == pytest.approx(
                probability, rel=STRICT_RTOL, abs=STRICT_ATOL
            )
            frequency = sum(f for order, f in frequencies.items() if order[0] == choice)
            assert abs(frequency - probability) <= frequency_tolerance(
                probability, 5_000, floor=0.0
            ), diagnostic
        if shift == 0.0:
            first_laws.append(first)
    # Largest score (label -7) wins most often, opposite to ascending keys.
    assert all(first[2] > first[1] > first[0] for first in first_laws)
    assert first_laws[0][2] > first_laws[1][2] > first_laws[2][2] > 1.0 / 3.0
    assert max(abs(p - 1.0 / 3.0) for p in first_laws[2]) < max(
        abs(p - 1.0 / 3.0) for p in first_laws[0]
    )


def test_sinkhorn_one_iteration_convergence_and_global_shift() -> None:
    matrix = np.asarray([[0.2, -0.5, 1.2], [1.1, 0.3, -0.2], [-0.7, 0.8, 0.1]])
    expected = _one_sinkhorn_iteration(matrix)
    # White-box access is confined to this pure primitive, with a scalar oracle.
    # Production declares bare ndarray parameters/results. Specialize that
    # incomplete annotation to the float64 matrices supplied here, without a wrapper.
    sinkhorn = cast(
        "Callable[[NDArray[np.float64], int], NDArray[np.float64]]",
        Permutation._sinkhorn,  # pyright: ignore[reportPrivateUsage]
    )
    once = sinkhorn(matrix.copy(), 1)
    converged = sinkhorn(matrix.copy(), 20)
    np.testing.assert_allclose(once, expected, rtol=STRICT_RTOL, atol=STRICT_ATOL)
    row_errors: list[float] = []
    for iterations, result in ((1, once), (20, converged)):
        assert np.all(result > 0.0)
        entries = list(map(float, result.flat))
        row_sums = [sum(entries[3 * row : 3 * row + 3]) for row in range(3)]
        column_sums = [sum(entries[column::3]) for column in range(3)]
        row_errors.append(max(abs(total - 1.0) for total in row_sums))
        # Columns are normalized last: they already sum to one after one step.
        np.testing.assert_allclose(
            column_sums, np.ones(3), rtol=STRICT_RTOL, atol=STRICT_ATOL
        )
        if iterations == 20:
            np.testing.assert_allclose(
                row_sums, np.ones(3), rtol=STRICT_RTOL, atol=STRICT_ATOL
            )
        shifted = sinkhorn(matrix + 8.0, iterations)
        np.testing.assert_allclose(shifted, result, rtol=STRICT_RTOL, atol=STRICT_ATOL)
    assert row_errors[0] > 0.01  # Nontrivial witness, not a convergence tolerance.
    assert row_errors[1] < row_errors[0]


def test_gumbel_sinkhorn_real_assignment_temperature_and_unscaled_scores() -> None:
    snapshots: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []

    class ObservedPermutation(Permutation):
        @staticmethod
        @override
        def _sinkhorn(
            matrix: NDArray[np.float64], iterations: int
        ) -> NDArray[np.float64]:
            # Static super delegation preserves production, RNG and return value.
            before = matrix.copy()
            # Same bare-ndarray annotation limitation as the primitive test.
            primitive = cast(
                "Callable[[NDArray[np.float64], int], NDArray[np.float64]]",
                super(ObservedPermutation, ObservedPermutation)._sinkhorn,
            )
            result = primitive(matrix, iterations)
            snapshots.append((before, result.copy()))
            return result

    choices = (10, 1000, -7)
    values = np.asarray([[0.125, 0.875, 0.25], [0.5, 0.0, 0.75], [1.0, 0.375, 0.625]])
    delta = np.asarray([[-1.0, 2.0, -0.5], [1.5, -2.0, 0.25], [0.0, 1.0, -1.5]])
    for temperature, iterations, custom in (
        (0.5, 1, False),
        (0.5, 20, False),
        (2.0, 20, False),
        (0.5, 20, True),
    ):
        scores = values + delta if custom else values
        space = ObservedPermutation(
            choices,
            constant_objective,
            decoder="gumbel-sinkhorn",
            bounds=(-3.0, 5.0) if custom else None,
            params={"temperature": temperature, "sinkhorn_iterations": iterations},
        )
        space.initialize_context(seed=BASE_SEED)
        inputs = dict(
            zip(space.encoded_variable_names, map(float, scores.flat), strict=True)
        )
        assert_variables_within_bounds(inputs, space.encoded_boundaries)
        decoded = space.decode(dict(reversed(list(inputs.items()))))
        assert isinstance(decoded, list)
        assert_valid_permutation(decoded[0], choices)
        before, after = snapshots[-1]
        # Pure one-step oracle also checks the iteration parameter on real decode.
        if iterations == 1:
            np.testing.assert_allclose(
                after,
                _one_sinkhorn_iteration(before),
                rtol=STRICT_RTOL,
                atol=STRICT_ATOL,
            )
        entries = tuple(map(float, after.flat))
        assignments = sorted(
            (sum(entries[3 * row + column] for column, row in enumerate(order)), order)
            for order in permutations(range(3))
        )
        best_score, best_order = assignments[-1]
        runner_up = assignments[-2][0]
        # Reject an ambiguous witness instead of assuming SciPy's tie-breaking.
        assert best_score - runner_up > STRICT_ATOL + STRICT_RTOL * abs(best_score)
        assert decoded == [[choices[row] for row in best_order]], (
            before,
            after,
            assignments,
        )
    assert len(snapshots) == 4
    cold, repeated, warm, changed = [before for before, _ in snapshots]
    np.testing.assert_allclose(cold, repeated, rtol=STRICT_RTOL, atol=STRICT_ATOL)
    row_errors: list[float] = []
    for _, after in snapshots[:2]:
        entries = tuple(map(float, after.flat))
        row_errors.append(
            max(abs(sum(entries[3 * row : 3 * row + 3]) - 1.0) for row in range(3))
        )
    # The public iterations parameter improves normalization of the same draw.
    assert row_errors[1] < row_errors[0]
    # The common seed couples experiments, without reconstructing any Gumbel.
    # M_T=(V+G)/T => T_cold*M_cold=T_warm*M_warm.
    np.testing.assert_allclose(
        0.5 * cold, 2.0 * warm, rtol=STRICT_RTOL, atol=STRICT_ATOL
    )
    # Entrywise changes verify row=choice, column=position and no bounds rescaling.
    np.testing.assert_allclose(
        0.5 * (changed - cold), delta, rtol=STRICT_RTOL, atol=STRICT_ATOL
    )


def test_independent_blocks_preserve_positional_named_mapping_and_fitness() -> None:
    def positional_objective(codes: list[int], colors: list[str], /) -> float:
        return float(
            sum(10**position * code for position, code in enumerate(codes))
            + 10_000
            * sum(
                (position + 1) * {"red": 2, "blue": 7}[color]
                for position, color in enumerate(colors)
            )
        )

    def named_objective(*, codes: list[int], colors: list[str]) -> float:
        return positional_objective(codes, colors)

    codes, colors = (10, 1000, -7), ("red", "blue")
    positional = Permutation([codes, colors], positional_objective)
    named = Permutation({"colors": colors, "codes": codes}, named_objective)
    positional.initialize_context(seed=BASE_SEED)
    named.initialize_context(seed=BASE_SEED)
    # Change one block at a time; named configuration reverses the argument order.
    for code_scores, color_scores, expected_codes, expected_colors in (
        ((0.625, 0.125, 0.875), (0.75, 0.25), [1000, 10, -7], ["blue", "red"]),
        ((0.875, 0.625, 0.125), (0.75, 0.25), [-7, 1000, 10], ["blue", "red"]),
        ((0.875, 0.625, 0.125), (0.25, 0.75), [-7, 1000, 10], ["red", "blue"]),
    ):
        expected_fitness = float(
            expected_codes[0]
            + 10 * expected_codes[1]
            + 100 * expected_codes[2]
            + 10_000
            * (
                {"red": 2, "blue": 7}[expected_colors[0]]
                + 2 * {"red": 2, "blue": 7}[expected_colors[1]]
            )
        )
        for space, scores, expected in (
            (
                positional,
                (*code_scores, *color_scores),
                [expected_codes, expected_colors],
            ),
            (
                named,
                (*color_scores, *code_scores),
                {"codes": expected_codes, "colors": expected_colors},
            ),
        ):
            inputs = dict(zip(space.encoded_variable_names, scores, strict=True))
            inputs = dict(reversed(list(inputs.items())))
            decoded = space.decode(inputs)
            assert decoded == expected
            blocks = (
                [decoded["codes"], decoded["colors"]]
                if isinstance(decoded, dict)
                else decoded
            )
            for output, choices in zip(blocks, (codes, colors), strict=True):
                assert_valid_permutation(output, choices)
            assert space(inputs) == expected_fitness
