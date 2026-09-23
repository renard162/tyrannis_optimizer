"""WOA mathematics through Optimizer + Continuous + the real Serial executor.

The current WhaleAlgorithm docstring identifies the Tyrannis variant; the
published equations below are independent mathematical oracles for it.
WOA.m is evidence specifically for a2/l, never the specification of this suite.

Mirjalili & Lewis (2016), doi:10.1016/j.advengsoft.2016.01.008, section 2.2:
encircling (2.1)-(2.4), spiral (2.5)-(2.6), random search (2.7)-(2.8).
Article text: https://www.scribd.com/document/396718054/mirjalili2016-pdf
The article describes vector coefficients and l in [-1, 1].

Author's WOA.m (version 1.0.0.0, downloaded from the official File Exchange):
https://www.mathworks.com/matlabcentral/fileexchange/55667-the-whale-optimization-algorithm/files/WOA/WOA.m
Its a2=-1-t/T and l=(a2-1)*rand+1 instead give l in [a2, 1]. Other MATLAB
implementation choices do not override the documented Tyrannis variant.

Tyrannis explicitly adopts separate r1/r2 vectors, one infinity-norm decision
and one reference whale per movement, synchronous consolidated inputs,
immediate clipping, unconditional movement, and a historical best. Its
documented power schedule 2*(1-(t/T)**exponent) extends the linear schedule;
setup is separate from updates t=0..T-1. Infinite fitness affects ranking,
not position geometry; no particular tied infinite best is prescribed.

Nonlinear schedule context: Yang et al. (2025), section 2.1, equation (1),
https://doi.org/10.3390/biomimetics10050273 (gamma > 1); Zhong & Long (2017),
https://doi.org/10.1051/matecconf/201713900157 (alternative nonlinear curves).
The full positive-exponent domain and its interpretation follow Tyrannis'
docstring, not the other mechanisms of those modified WOA algorithms.
"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from math import cos, exp, isfinite, pi, pow
from typing import override

import numpy as np
import pytest

from tests._support import factories
from tests._support.assertions import (
    assert_optimizer_result_consistent,
    assert_particle_state_consistent,
)
from tests._support.numerics import BASE_SEED, STRICT_ATOL, STRICT_RTOL, seed_for
from tests._support.objectives import sphere
from tyrannis.algorithm.whale_algorithm import WhaleAlgorithm, WhaleParticle
from tyrannis.core.algorithm import FITNESS_UNDEFINED
from tyrannis.processor.serial import Serial
from tyrannis.space.continuous import Continuous


@dataclass
class MovementInputs:
    positions: dict[str, dict[str, float]]
    best: WhaleParticle
    random: dict[str, float | int]


class ObservedWhaleAlgorithm(WhaleAlgorithm):
    """Copy mathematical inputs/candidates only; super() owns all dynamics."""

    @override
    def pre_iteration(self, actual_iter: int) -> None:
        self.actual_iter: int = actual_iter
        if actual_iter == 0:
            self.initial: dict[str, WhaleParticle] = deepcopy(self.population)
            self.initialized: dict[str, WhaleParticle] = {}
            self.inputs: dict[tuple[int, str], MovementInputs] = {}
            self.candidates: dict[tuple[int, str], WhaleParticle] = {}
            self.states: dict[int, dict[str, WhaleParticle]] = {}
            self.best: dict[int, WhaleParticle] = {}
            self.pending_best: dict[int, WhaleParticle] = {}
            self.controls: dict[int, tuple[float, float]] = {}
        super().pre_iteration(actual_iter)
        # a2 has no public accessor; observation only, never an expected value.
        self.controls[actual_iter] = (self.a, self._a2)

    @override
    def initialize_particle(self, identifier: str) -> WhaleParticle:
        particle = super().initialize_particle(identifier)
        self.initialized[identifier] = deepcopy(particle)
        return particle

    @override
    def update_particle(self, identifier: str) -> WhaleParticle:
        assert self.local_best is not None
        self.inputs[self.actual_iter, identifier] = MovementInputs(
            positions={key: p.variables.copy() for key, p in self.population.items()},
            best=deepcopy(self.local_best),
            random=self.population[identifier].random_cache.copy(),
        )
        particle = super().update_particle(identifier)
        self.candidates[self.actual_iter, identifier] = deepcopy(particle)
        return particle

    @override
    def post_iteration(self, actual_iter: int) -> None:
        if actual_iter > 0:
            assert self.local_best is not None
            self.pending_best[actual_iter] = deepcopy(self.local_best)
        super().post_iteration(actual_iter)
        self.states[actual_iter] = deepcopy(self.population)
        assert self.local_best is not None
        self.best[actual_iter] = deepcopy(self.local_best)


def run_whales(
    bounds: dict[str, tuple[float, float]],
    *,
    exponent: float = 1.0,
    spiral: float = 1.0,
    objective: Callable[..., float] = sphere,
    seed: int = BASE_SEED,
    n_particles: int = 5,
    n_iterations: int = 5,
) -> ObservedWhaleAlgorithm:
    template = ObservedWhaleAlgorithm(spiral, exponent)
    processor = Serial()
    optimizer = factories.make_optimizer(
        Continuous(bounds, cost_function=objective),
        template,
        processor=processor,
        n_particles=n_particles,
        n_iterations=n_iterations,
        seed=seed,
        history="iteration",
        fitness_failure_strategy="raise",
    ).fit()
    (executor,) = processor.processors_pool.values()
    # Public pool locates the real replica, whose algorithm has no public accessor.
    assert isinstance(executor._algorithm, ObservedWhaleAlgorithm)  # pyright: ignore[reportPrivateUsage]
    algorithm = executor._algorithm  # pyright: ignore[reportPrivateUsage]
    assert algorithm is not template
    assert (
        set(algorithm.states) == set(algorithm.controls) == set(range(n_iterations + 1))
    )
    assert (
        set(algorithm.inputs)
        == set(algorithm.candidates)
        == {(i, key) for i in range(1, n_iterations + 1) for key in algorithm.initial}
    )
    for population in algorithm.states.values():
        assert len(population) == n_particles
        for particle in population.values():
            assert_particle_state_consistent(particle, bounds)
            assert all(np.isfinite(value) for value in particle.variables.values())
            assert particle.fitness == pytest.approx(
                objective(**particle.variables), rel=STRICT_RTOL, abs=STRICT_ATOL
            )
            assert particle.candidate_variables is None
            assert particle.candidate_fitness is None
    assert_initial_state(algorithm, bounds, objective)
    # Minimal history check: it exposes these consolidated positions and fitnesses.
    rows: list[list[object]] = list(optimizer.internal_history_generator_)
    assert len(rows) == n_particles * (n_iterations + 1)
    for row in rows:
        iteration, identifier = row[1], row[2]
        assert isinstance(iteration, int) and isinstance(identifier, str)
        particle = algorithm.states[iteration][identifier]
        assert row[5:-1] == [particle.fitness, *particle.variables.values()]
    assert_optimizer_result_consistent(optimizer)
    assert optimizer.best_fitness == min(
        p.fitness
        for population in algorithm.states.values()
        for p in population.values()
    )
    return algorithm


def assert_initial_state(
    algorithm: ObservedWhaleAlgorithm,
    bounds: dict[str, tuple[float, float]],
    objective: Callable[..., float],
) -> None:
    """Evaluated creation positions form t=0 inputs, with no hunting movement."""
    population = algorithm.states[0]
    assert algorithm.controls[0] == (2.0, -1.0)
    assert set(algorithm.initialized) == set(algorithm.initial) == set(population)
    assert not any(i == 0 for i, _ in algorithm.inputs)
    assert not any(i == 0 for i, _ in algorithm.candidates)
    for key, particle in population.items():
        evaluated = algorithm.initialized[key]
        assert_particle_state_consistent(evaluated, bounds)
        assert evaluated.candidate_variables == algorithm.initial[key].variables
        assert particle.variables == evaluated.candidate_variables
        assert evaluated.random_cache == particle.random_cache == {}
        assert evaluated.candidate_fitness == pytest.approx(
            objective(**particle.variables), rel=STRICT_RTOL, abs=STRICT_ATOL
        )
        assert particle.fitness == evaluated.candidate_fitness
    best = algorithm.best[0]
    assert best.fitness == min(p.fitness for p in population.values())
    # Membership, not a tie-breaking identity (in particular when all are infinite).
    assert any(
        best.variables == p.variables and best.fitness == p.fitness
        for p in population.values()
    )


@pytest.fixture
def whales(small_bounds: dict[str, tuple[float, float]]) -> ObservedWhaleAlgorithm:
    # Mixed |A_j|, clipping, earlier donor, and prey absent from the current swarm.
    return run_whales(small_bounds, seed=seed_for(5))


def test_initial_state_is_mathematically_valid(whales: ObservedWhaleAlgorithm) -> None:
    # run_whales validates setup in every scenario, including infinite objectives.
    first = next(iter(whales.initial))
    assert whales.inputs[1, first].positions == {
        key: p.variables for key, p in whales.states[0].items()
    }


def assert_movement_equations(
    algorithm: ObservedWhaleAlgorithm,
    bounds: dict[str, tuple[float, float]],
    *,
    exponent: float = 1.0,
    spiral: float = 1.0,
) -> set[str]:
    """Published vector equations, conditioned only on observed random draws.

    No state is advanced here: each equation starts from a consolidated snapshot.
    Witnesses ensure the assertions have discriminating, non-vacuous examples.
    """
    witnessed = assert_historical_best_and_consolidation(algorithm)
    names = tuple(bounds)
    lower = np.array([bounds[j][0] for j in names])
    upper = np.array([bounds[j][1] for j in names])
    total = len(algorithm.states) - 1
    for (iteration, key), inputs in algorithm.inputs.items():
        population = algorithm.states[iteration - 1]
        target = population[key]
        candidate = algorithm.candidates[iteration, key]
        assert inputs.positions == {k: p.variables for k, p in population.items()}
        assert candidate.variables == target.variables
        assert candidate.fitness == target.fitness
        assert inputs.best.variables == algorithm.best[iteration - 1].variables
        assert inputs.best.fitness == algorithm.best[iteration - 1].fitness
        assert candidate.random_cache == inputs.random
        assert candidate.candidate_variables is not None
        assert candidate.candidate_fitness is not None
        cache = inputs.random
        t = iteration - 1
        a = 2 * (1 - pow(t / total, exponent))
        a2 = -1 - t / total
        assert algorithm.controls[iteration] == pytest.approx(
            (a, a2), rel=STRICT_RTOL, abs=STRICT_ATOL
        )
        assert np.isfinite(algorithm.controls[iteration]).all()
        assert 0 <= cache["p"] <= 1
        assert a2 <= cache["l"] <= 1
        x = np.array([target.variables[j] for j in names])
        best = np.array([algorithm.best[iteration - 1].variables[j] for j in names])
        # A/C are vectors: every component has its own observed random input.
        r1 = np.array([cache[f"{j}-r1"] for j in names])
        r2 = np.array([cache[f"{j}-r2"] for j in names])
        assert ((0 <= r1) & (r1 <= 1)).all()
        assert ((0 <= r2) & (r2 <= 1)).all()
        coefficient_a = a * (2 * r1 - 1)
        coefficient_c = 2 * r2
        norm = max(abs(a * (2 * cache[f"{j}-r1"] - 1)) for j in names)
        assert norm <= a + STRICT_ATOL
        assert np.isfinite(coefficient_a).all() and np.isfinite(coefficient_c).all()
        if a < 1:
            assert norm < 1
            if cache["p"] < 0.5:
                witnessed.add("late_encircling")
        if cache["p"] >= 0.5:
            witnessed.add("spiral")
            # (2.5): independent of A, C, r1 and r2.
            distance = np.abs(best - x)
            raw = best + distance * exp(spiral * cache["l"]) * cos(2 * pi * cache["l"])
            if cache["l"] < -1 and np.any(distance > 0):
                witnessed.add("dynamic_spiral")
            if spiral != 1:
                default_raw = best + distance * exp(cache["l"]) * cos(
                    2 * pi * cache["l"]
                )
                if not np.allclose(
                    np.clip(raw, lower, upper),
                    np.clip(default_raw, lower, upper),
                    rtol=STRICT_RTOL,
                    atol=STRICT_ATOL,
                ):
                    witnessed.add("nondefault_spiral")
        else:
            branch = "encircling" if norm < 1 else "search"
            witnessed.add(branch)
            index = cache["random_particle"]
            assert isinstance(index, int) and 0 <= index < len(population)
            reference_key = tuple(population)[index]
            reference = (
                best
                if norm < 1
                else np.array([population[reference_key].variables[j] for j in names])
            )
            # (2.1)-(2.2) / (2.7)-(2.8): ONE reference vector for every component.
            distance = np.abs(coefficient_c * reference - x)
            raw = reference - coefficient_a * distance
            clipped = np.clip(raw, lower, upper)
            if norm >= 1 and np.any(np.abs(coefficient_a) < 1):
                # A coordinate-wise decision would incorrectly encircle on these axes.
                per_axis_reference = np.where(
                    np.abs(coefficient_a) < 1, best, reference
                )
                per_axis_raw = per_axis_reference - coefficient_a * np.abs(
                    coefficient_c * per_axis_reference - x
                )
                if not np.allclose(
                    clipped,
                    np.clip(per_axis_raw, lower, upper),
                    rtol=STRICT_RTOL,
                    atol=STRICT_ATOL,
                ):
                    witnessed.add("mixed_norm_decision")
            scalar_a_raw = reference - a * (2 * cache[f"{names[0]}-r1"] - 1) * distance
            scalar_c_raw = reference - coefficient_a * np.abs(
                2 * cache[f"{names[0]}-r2"] * reference - x
            )
            if not np.allclose(
                clipped,
                np.clip(scalar_a_raw, lower, upper),
                rtol=STRICT_RTOL,
                atol=STRICT_ATOL,
            ):
                witnessed.add("vector_a")
            if not np.allclose(
                clipped,
                np.clip(scalar_c_raw, lower, upper),
                rtol=STRICT_RTOL,
                atol=STRICT_ATOL,
            ):
                witnessed.add("vector_c")
            if exponent != 1 and t > 0:
                linear_a = 2 * (1 - t / total) * (2 * r1 - 1)
                linear_raw = reference - linear_a * distance
                if not np.allclose(
                    clipped,
                    np.clip(linear_raw, lower, upper),
                    rtol=STRICT_RTOL,
                    atol=STRICT_ATOL,
                ):
                    witnessed.add("nonlinear_movement")
            if norm >= 1 and index < tuple(population).index(key):
                earlier = algorithm.candidates[
                    iteration, reference_key
                ].candidate_variables
                assert earlier is not None
                if earlier != population[reference_key].variables:
                    witnessed.add("earlier_reference")
        assert np.isfinite(distance).all() and np.isfinite(raw).all()
        if np.any((raw < lower) | (raw > upper)):
            witnessed.add("clipped")
        else:
            witnessed.add("unclipped")
        np.testing.assert_allclose(
            [candidate.candidate_variables[j] for j in names],
            np.clip(raw, lower, upper),
            rtol=STRICT_RTOL,
            atol=STRICT_ATOL,
            err_msg=f"{iteration=}, {key=}",
        )
        assert np.isfinite(list(candidate.candidate_variables.values())).all()
    return witnessed


def assert_historical_best_and_consolidation(
    algorithm: ObservedWhaleAlgorithm,
) -> set[str]:
    """The minimum of evaluated history is independent of current best bookkeeping."""
    witnessed: set[str] = set()
    history: list[WhaleParticle] = []
    for iteration, population in algorithm.states.items():
        history.extend(population.values())
        minimum = min(p.fitness for p in history)
        best = algorithm.best[iteration]
        assert best.fitness == minimum
        assert any(
            p.fitness == minimum and p.variables == best.variables for p in history
        )
        if iteration == 0:
            continue
        previous_best = algorithm.best[iteration - 1]
        if previous_best.fitness < min(
            p.fitness for p in algorithm.states[iteration - 1].values()
        ):
            witnessed.add("historical_prey")
        pending = algorithm.pending_best[iteration]
        assert pending.variables == previous_best.variables
        assert pending.fitness == previous_best.fitness
        if minimum < previous_best.fitness:
            witnessed.add("best_improved")
        else:
            witnessed.add("best_preserved")
            assert best.variables == previous_best.variables
        earlier_improvement = False
        for key, current in population.items():
            previous = algorithm.states[iteration - 1][key]
            candidate = algorithm.candidates[iteration, key]
            assert candidate.candidate_variables is not None
            assert candidate.candidate_fitness is not None
            # WOA accepts ALL moves, including finite -> inf and inf -> inf.
            assert current.variables == candidate.candidate_variables
            assert current.fitness == candidate.candidate_fitness
            if current.fitness < previous.fitness:
                witnessed.add("improved_move")
            if current.fitness > previous.fitness:
                witnessed.add("worsened_move")
            inputs = algorithm.inputs[iteration, key]
            assert inputs.best.fitness == previous_best.fitness
            assert inputs.best.variables == previous_best.variables
            # A later spiral must still use the old prey after an earlier improvement.
            if earlier_improvement and inputs.random["p"] >= 0.5:
                witnessed.add("earlier_improvement")
            earlier_improvement |= current.fitness < previous_best.fitness
    return witnessed


@pytest.mark.parametrize(
    "exponent", [1.0, 0.5, 2.0], ids=["linear", "early_decay", "late_decay"]
)
def test_control_schedules_and_their_effect_on_movements(
    small_bounds: dict[str, tuple[float, float]],
    exponent: float,
) -> None:
    algorithm = run_whales(small_bounds, exponent=exponent)
    # Five real updates: t=0..4, never an extra t=T update.
    expected_a = [2 * (1 - pow(t / 5, exponent)) for t in range(5)]
    expected_a2 = [-1 - t / 5 for t in range(5)]
    np.testing.assert_allclose(
        [algorithm.controls[i][0] for i in range(1, 6)],
        expected_a,
        rtol=STRICT_RTOL,
        atol=STRICT_ATOL,
    )
    np.testing.assert_allclose(
        [algorithm.controls[i][1] for i in range(1, 6)],
        expected_a2,
        rtol=STRICT_RTOL,
        atol=STRICT_ATOL,
    )
    assert expected_a[0] == 2 and expected_a[-1] > 0
    assert expected_a2[0] == -1 and expected_a2[-1] > -2
    linear = [2, 8 / 5, 6 / 5, 4 / 5, 2 / 5]
    if exponent < 1:
        assert all(a < b for a, b in zip(expected_a[1:], linear[1:], strict=True))
    elif exponent > 1:
        assert all(a > b for a, b in zip(expected_a[1:], linear[1:], strict=True))
    else:
        np.testing.assert_allclose(
            expected_a, linear, rtol=STRICT_RTOL, atol=STRICT_ATOL
        )
    evidence = assert_movement_equations(algorithm, small_bounds, exponent=exponent)
    assert "dynamic_spiral" in evidence
    if exponent != 1:
        assert "nonlinear_movement" in evidence


def test_three_branches_vector_coefficients_one_reference_and_clipping(
    whales: ObservedWhaleAlgorithm,
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    evidence = assert_movement_equations(whales, small_bounds)
    assert {
        "encircling",
        "search",
        "spiral",
        "mixed_norm_decision",
        "vector_a",
        "vector_c",
        "earlier_reference",
        "late_encircling",
        "clipped",
        "unclipped",
        "dynamic_spiral",
    } <= evidence


def test_historical_best_is_synchronous_and_moves_are_not_greedy(
    whales: ObservedWhaleAlgorithm,
) -> None:
    evidence = assert_historical_best_and_consolidation(whales)
    assert {
        "best_improved",
        "best_preserved",
        "improved_move",
        "worsened_move",
        "earlier_improvement",
        "historical_prey",
    } <= evidence


def test_nondefault_spiral_coefficient_changes_the_actual_position(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    algorithm = run_whales(small_bounds, spiral=0.5, n_particles=3, n_iterations=3)
    evidence = assert_movement_equations(algorithm, small_bounds, spiral=0.5)
    assert "spiral" in evidence and "nondefault_spiral" in evidence


def half_sphere(x: float, y: float) -> float:
    """A deterministic infeasible half-plane; geometry is finite on both sides."""
    return sphere(x=x, y=y) if x >= 0 else np.inf


def test_infinite_fitness_does_not_corrupt_geometry_or_prevent_consolidation(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    algorithm = run_whales(small_bounds, objective=half_sphere)
    initial = algorithm.states[0]
    assert sum(p.fitness == np.inf for p in initial.values()) >= 2
    assert any(np.isfinite(p.fitness) for p in initial.values())
    assert all(np.isfinite(best.fitness) for best in algorithm.best.values())
    evidence = assert_movement_equations(algorithm, small_bounds)
    assert {"encircling", "search", "spiral"} <= evidence
    transitions: set[tuple[bool, bool]] = set()
    recovered_best = False
    for (iteration, key), candidate in algorithm.candidates.items():
        previous = algorithm.states[iteration - 1][key]
        assert candidate.candidate_fitness is not None
        if candidate.candidate_variables != previous.variables:
            transitions.add(
                (isfinite(previous.fitness), isfinite(candidate.candidate_fitness))
            )
        if (
            previous.fitness == np.inf
            and candidate.candidate_fitness < algorithm.best[iteration - 1].fitness
        ):
            recovered_best = True
            assert algorithm.best[iteration].fitness <= candidate.candidate_fitness
        if np.isfinite(previous.fitness) and candidate.candidate_fitness == np.inf:
            assert algorithm.states[iteration][key].fitness == np.inf
            assert (
                algorithm.best[iteration].fitness
                <= algorithm.best[iteration - 1].fitness
            )
    assert {(False, True), (True, False), (False, False)} <= transitions
    assert recovered_best


def test_all_infinite_population_has_a_usable_best_and_defined_movements() -> None:
    # Same objective, but this entire box is infeasible. No infinity tie identity.
    bounds = {"x": (-5.0, -1.0), "y": (-3.0, 3.0)}
    algorithm = run_whales(bounds, objective=half_sphere, n_particles=3, n_iterations=3)
    evidence = assert_movement_equations(algorithm, bounds)
    assert {"encircling", "search", "spiral", "best_preserved"} <= evidence
    assert all(
        p.fitness == np.inf
        for population in algorithm.states.values()
        for p in population.values()
    )
    assert all(p.fitness == np.inf for p in algorithm.best.values())
    assert any(
        candidate.candidate_variables != candidate.variables
        for candidate in algorithm.candidates.values()
    )


def test_negative_infinite_fitness_is_preserved_as_historical_best(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    # MATHTESTS 34.3 protects historical minima; WOA's current moves are non-greedy.
    def objective(**variables: float) -> float:
        x = variables["x"]
        return -np.inf if -1 < x < 1 else np.inf if x > 1 else sphere(**variables)

    assert np.isposinf(FITNESS_UNDEFINED) and -np.inf != FITNESS_UNDEFINED
    algorithm = run_whales(
        small_bounds, objective=objective, seed=seed_for(125), n_iterations=5
    )
    assert isfinite(algorithm.best[0].fitness)
    assert any(np.isposinf(p.fitness) for p in algorithm.states[0].values())
    evidence = assert_movement_equations(algorithm, small_bounds)
    assert {
        "encircling",
        "search",
        "spiral",
        "best_improved",
        "best_preserved",
    } <= evidence
    first = min(t for t, best in algorithm.best.items() if np.isneginf(best.fitness))
    assert first > 0
    assert any(
        isfinite(algorithm.states[first - 1][key].fitness) and np.isneginf(p.fitness)
        for key, p in algorithm.states[first].items()
    ), "A finite whale must discover the first -inf solution"
    for iteration, population in algorithm.states.items():
        assert all(not p.new_particle for p in population.values())
        if iteration >= first:
            best = algorithm.best[iteration]
            assert np.isneginf(best.fitness)
            assert best.variables == algorithm.best[first].variables
            assert np.isneginf(objective(**best.variables))
            assert all(isfinite(value) for value in best.variables.values())

    departures: set[str] = set()
    for (iteration, key), candidate in algorithm.candidates.items():
        before = algorithm.states[iteration - 1][key]
        after = algorithm.states[iteration][key]
        assert candidate.candidate_fitness is not None
        if np.isneginf(before.fitness):
            if isfinite(candidate.candidate_fitness):
                departures.add("finite")
            elif np.isposinf(candidate.candidate_fitness):
                departures.add("+inf")
            # Movement is accepted while the historical optimum remains protected.
            assert after.variables == candidate.candidate_variables
            assert after.fitness == candidate.candidate_fitness
            assert np.isneginf(algorithm.best[iteration].fitness)
    assert departures == {"finite", "+inf"}
