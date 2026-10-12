"""Tyrannis sequences form union(C**k, k=m,...,M), with order and repetition.

There are M=max_choices decision positions in a fixed continuous domain:
M coordinates for scalar/ordinal, m*K+(M-m)*(K+1) for vector categorical.
STOP exists only in optional positions; its first occurrence sets the length.
Categorical and Ordinal equations are validated separately. Here we protect
their composition, including configurable ordinal STOP geometry.

This is Tyrannis's unified contract, not a verbatim published encoding.
The docstring references Bean (1994), doi:10.1287/ijoc.6.2.154;
Zhang & Ding (2016), doi:10.1016/j.eswa.2016.04.005;
Beke et al. (2021), doi:10.1007/s42979-021-00512-z; and
Ni et al. (2021), doi:10.1016/j.infsof.2021.106565, as background.
The first-STOP rule and its ordinal positions are Tyrannis contracts.
"""

from collections import Counter

import pytest

from tests._support.assertions import assert_variables_within_bounds
from tests._support.numerics import BASE_SEED, frequency_tolerance, seed_for
from tests._support.objectives import constant_objective
from tyrannis.space import Sequence


def _cell(index: int, levels: int) -> float:
    """Interior of scalar cell [j/n, (j+1)/n] in the default unit interval."""
    return (index + 0.5) / levels


def _ordinal_bounds(levels: tuple[float, ...]) -> tuple[float, float]:
    """Extend a sorted scale by half of each extreme gap."""
    return (
        levels[0] - (levels[1] - levels[0]) / 2.0,
        levels[-1] + (levels[-1] - levels[-2]) / 2.0,
    )


@pytest.mark.parametrize(
    "decoder",
    [
        None,
        "categorical-one-hot",
        "categorical-softmax",
        "categorical-gumbel-softmax",
        "categorical-scalar",
    ],
    ids=["default", "one-hot", "softmax", "gumbel-softmax", "scalar"],
)
def test_categorical_geometry_has_mandatory_and_optional_blocks(
    decoder: str | None,
) -> None:
    choices = ("A", "B", "C")
    minimum, maximum = 2, 4
    cardinality = len(choices)
    scalar = decoder == "categorical-scalar"
    dimension = (
        maximum
        if scalar
        else minimum * cardinality + (maximum - minimum) * (cardinality + 1)
    )
    space = Sequence(
        choices,
        min_choices=minimum,
        max_choices=maximum,
        decoder=decoder,
        cost_function=constant_objective,
    )
    space.initialize_context(seed=BASE_SEED)
    boundaries = space.encoded_boundaries
    interval = (0.0, 1.0) if scalar else (-1.0, 1.0)
    assert len(boundaries) == dimension
    assert list(boundaries.values()) == [interval] * dimension
    if not scalar:
        for step in range(maximum):
            block = [key for key in boundaries if key.startswith(f"{step}-")]
            assert len(block) == cardinality + (step >= minimum)
            assert {f"{step}-{choice}" for choice in choices} <= set(block)

    if decoder in (None, "categorical-one-hot"):
        # Public coordinates identify the choices; no private STOP token needed.
        full = dict.fromkeys(boundaries, -0.5)
        short = dict.fromkeys(boundaries, 0.75)
        for step in range(maximum):
            full[f"{step}-A"] = 0.5
            for choice in choices:
                short[f"{step}-{choice}"] = -0.5
            if step < minimum:
                short[f"{step}-A"] = 0.5
        assert space.decode(short) == [["A"] * minimum]
        assert space.decode(full) == [["A"] * maximum]
        assert space.encoded_boundaries == boundaries


def test_scalar_first_stop_selects_prefix_with_fixed_encoded_dimension() -> None:
    choices = ("A", "B", "C")
    cardinality, maximum = len(choices), 4
    space = Sequence(
        choices, min_choices=1, max_choices=maximum, decoder="categorical-scalar"
    )
    space.initialize_context(seed=BASE_SEED)
    boundaries = {str(i): (0.0, 1.0) for i in range(maximum)}
    assert space.encoded_boundaries == boundaries
    stop = _cell(cardinality, cardinality + 1)
    # Witness C**1 through C**4, including repeated choices and distinct orders.
    for sequence in (["A"], ["A", "B"], ["A", "B", "A"], ["A", "A", "B", "A"]):
        values = [
            _cell(choices.index(choice), cardinality if i == 0 else cardinality + 1)
            for i, choice in enumerate(sequence)
        ]
        values += [stop] * (maximum - len(sequence))
        inputs = dict(zip(boundaries, values, strict=True))
        assert_variables_within_bounds(inputs, boundaries)
        assert space.decode(inputs) == [sequence]
        assert space.encoded_boundaries == boundaries

    # A mandatory endpoint selects C; the optional endpoint terminates.
    assert space.decode(dict.fromkeys(boundaries, 1.0)) == [["C"]]
    # Optional STOP occupies [K/(K+1), 1]; the threshold belongs to STOP.
    threshold = cardinality / (cardinality + 1)
    epsilon = 2.0**-20
    for x, expected in ((threshold - epsilon, ["A", "C"]), (threshold, ["A"])):
        assert space.decode({"0": _cell(0, 3), "1": x, "2": stop, "3": 0.0}) == [
            expected
        ]
    for tail in (_cell(0, 4), _cell(2, 4), stop):
        inputs = {"0": _cell(0, 3), "1": _cell(1, 4), "2": stop, "3": tail}
        assert_variables_within_bounds(inputs, boundaries)
        assert space.decode(inputs) == [["A", "B"]]


def test_empty_and_fixed_length_domains_allow_repetition() -> None:
    for minimum in (0, 3):
        space = Sequence(
            ("A", "B"), min_choices=minimum, max_choices=3, decoder="categorical-scalar"
        )
        space.initialize_context(seed=BASE_SEED)
        assert space.encoded_boundaries == {str(i): (0.0, 1.0) for i in range(3)}
        if minimum == 0:
            assert space.decode({"0": _cell(2, 3), "1": 0.0, "2": 0.0}) == [[]]
            assert space.decode({"0": _cell(0, 3), "1": _cell(2, 3), "2": 1.0}) == [
                ["A"]
            ]
        else:
            # With no optional positions, even x=1 is B, never STOP.
            assert space.decode({"0": 1.0, "1": 0.0, "2": 1.0}) == [["B", "A", "B"]]


@pytest.mark.statistical
def test_softmax_length_follows_truncated_first_stop_distribution() -> None:
    choices, minimum, maximum = ("A", "B"), 1, 4
    space = Sequence(
        choices, min_choices=minimum, max_choices=maximum, decoder="categorical-softmax"
    )
    seed = seed_for(0)
    space.initialize_context(seed=seed)
    inputs = dict.fromkeys(space.encoded_variable_names, 0.0)
    q = 1.0 / (len(choices) + 1)
    probabilities = {
        length: (1.0 - q) ** (length - minimum) * q
        for length in range(minimum, maximum)
    }
    probabilities[maximum] = (1.0 - q) ** (maximum - minimum)
    sample_size = 5_000
    counts: Counter[int] = Counter()
    for _ in range(sample_size):
        decoded = space.decode(inputs)
        assert isinstance(decoded, list)
        assert len(decoded) == 1
        counts[len(decoded[0])] += 1
    diagnostic = (
        f"seed={seed}, N={sample_size}, expected={probabilities}, counts={counts}"
    )
    assert set(counts) == set(probabilities), diagnostic
    # Each length indicator is Bernoulli(p). Six sigma, no RNG replay.
    for length, probability in probabilities.items():
        assert abs(counts[length] / sample_size - probability) <= frequency_tolerance(
            probability, sample_size, floor=0.0
        ), diagnostic


def test_automatic_ordinal_stop_threshold_controls_length() -> None:
    choices = ("A", "B", "C")
    cardinality = len(choices)
    space = Sequence(
        choices, min_choices=1, max_choices=3, decoder="ordinal-rank", positions=None
    )
    space.initialize_context(seed=BASE_SEED)
    assert space.encoded_boundaries == {
        "0": (-0.5, cardinality - 0.5),
        "1": (-0.5, cardinality + 0.5),
        "2": (-0.5, cardinality + 0.5),
    }
    threshold = cardinality - 0.5
    # Same exact binary epsilon as the Ordinal midpoint tests, well above ulps.
    epsilon = 2.0**-20
    for x, expected in (
        (threshold, ["C", "C", "A"]),
        (threshold + epsilon, ["C"]),
    ):
        # The mandatory upper endpoint still selects C, with no STOP there.
        assert space.decode({"0": threshold, "1": x, "2": 0.0}) == [expected]
    assert space.decode({"0": 0.0, "1": 0.0, "2": float(cardinality)}) == [["A", "A"]]


def test_explicit_ordinal_scalar_stop_is_broadcast_with_interior_geometry() -> None:
    positions = (-4.0, 1.0, 7.0)
    stop = 3.0
    # decoder=None with explicit positions must realize the ordinal-rank map.
    space = Sequence(
        ("A", "B", "C"),
        min_choices=1,
        max_choices=3,
        positions=positions,
        stop_positions=stop,
        decoder=None,
    )
    space.initialize_context(seed=BASE_SEED)
    optional_levels = (-4.0, 1.0, stop, 7.0)
    assert space.encoded_boundaries == {
        "0": _ordinal_bounds(positions),
        "1": _ordinal_bounds(optional_levels),
        "2": _ordinal_bounds(optional_levels),
    }
    # Inserting STOP changes the upper half-gap from (7-1)/2 to (7-3)/2.
    left, right = (positions[1] + stop) / 2.0, (stop + positions[2]) / 2.0
    epsilon = 2.0**-20
    # Lower-level ties make the STOP cell (left, right], not [left, right).
    for x, expected in (
        (left - epsilon, ["B", "B", "C"]),
        (left, ["B", "B", "C"]),
        (left + epsilon, ["B"]),
        (stop, ["B"]),
        (right, ["B"]),
        (right + epsilon, ["B", "C", "C"]),
    ):
        inputs = {"0": stop, "1": x, "2": positions[2]}
        assert_variables_within_bounds(inputs, space.encoded_boundaries)
        assert space.decode(inputs) == [expected], x
    # The same s also terminates at the second optional position, but is B
    # in the mandatory position, whose scale does not contain STOP.
    assert space.decode({"0": stop, "1": positions[0], "2": stop}) == [["B", "A"]]


def test_explicit_ordinal_per_step_stops_change_the_termination_geometry() -> None:
    positions = (-4.0, 1.0, 7.0)
    stops = (-8.0, 10.0)
    space = Sequence(
        ("A", "B", "C"),
        min_choices=1,
        max_choices=3,
        positions=positions,
        stop_positions=stops,
        decoder="ordinal-rank",
    )
    space.initialize_context(seed=BASE_SEED)
    assert space.encoded_boundaries == {
        "0": _ordinal_bounds(positions),
        "1": _ordinal_bounds((stops[0], *positions)),
        "2": _ordinal_bounds((*positions, stops[1])),
    }
    for first, second, expected in (
        (stops[0], stops[1], ["B"]),
        # x=10 is C at the first optional step, STOP at the second.
        (stops[1], stops[1], ["B", "C"]),
        (positions[0], positions[2], ["B", "A", "C"]),
    ):
        inputs = {"0": positions[1], "1": first, "2": second}
        assert_variables_within_bounds(inputs, space.encoded_boundaries)
        assert space.decode(inputs) == [expected], inputs


def test_positional_and_named_objectives_receive_only_the_ordered_prefix() -> None:
    # Decimal place weights distinguish order, length and repeated elements.
    # Shared sphere/constant objectives cannot express this sequence contract.
    def positional_objective(route: list[int], /) -> float:
        return float(sum(10**i * choice for i, choice in enumerate(route)))

    def named_objective(*, route: list[int]) -> float:
        return positional_objective(route)

    choices, maximum = (2, 7), 4
    positional = Sequence(
        choices,
        min_choices=1,
        max_choices=maximum,
        decoder="categorical-scalar",
        cost_function=positional_objective,
    )
    named = Sequence(
        {"route": choices},
        min_choices=1,
        max_choices=maximum,
        decoder="categorical-scalar",
        cost_function=named_objective,
    )
    positional.initialize_context(seed=BASE_SEED)
    named.initialize_context(seed=BASE_SEED)
    stop = _cell(2, 3)
    fitnesses: list[float] = []
    for sequence in ([2, 7], [2, 7, 2], [7, 2, 2], [2, 7, 7]):
        values = [
            _cell(choices.index(choice), 2 if i == 0 else 3)
            for i, choice in enumerate(sequence)
        ]
        values += [stop] * (maximum - len(sequence))
        expected_fitness = float(
            sum(choice * 10**i for i, choice in enumerate(sequence))
        )
        fitnesses.append(expected_fitness)
        # For the shorter prefix, vary only the inactive final coordinate.
        tails = (_cell(0, 3), _cell(1, 3), stop) if len(sequence) == 2 else (stop,)
        for tail in tails:
            values[-1] = tail
            # Reverse dictionary insertion to check positional mapping by name.
            positional_inputs = {str(i): values[i] for i in reversed(range(maximum))}
            named_inputs = {f"route-{i}": values[i] for i in reversed(range(maximum))}
            for space, inputs, expected in (
                (positional, positional_inputs, [sequence]),
                (named, named_inputs, {"route": sequence}),
            ):
                assert_variables_within_bounds(inputs, space.encoded_boundaries)
                assert space.decode(inputs) == expected
                assert space(inputs) == expected_fitness
    assert len(set(fitnesses)) == len(fitnesses)
