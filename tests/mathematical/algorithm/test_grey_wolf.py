"""GWO/EEGWO mathematics through Optimizer + Continuous + real Serial.

Mirjalili, Mirjalili & Lewis (2014), doi:10.1016/j.advengsoft.2013.12.007,
section 3.2, equations (3.3)-(3.7). Full article consulted at:
https://tiezhongyu2005.github.io/resources/popularization/GWO_2014.pdf
Equation (3.3) implies A in [-a,a]; section 3.2.4's prose [-2a,2a]
is inconsistent with that equation and is not used as the oracle.

Long, Jiao, Liang & Tang (2018), doi:10.1016/j.engappai.2017.10.024:
https://www.researchgate.net/publication/332032985
The accessible abstract confirms random-individual guidance and increasing
nonlinear control; full text was unavailable. The exact EEGWO weights and
power 1.5 below are the explicit task/Tyrannis contract, not independently
verified quotations from that paper. Its fourth author is Tang, not Cai as
listed in the current Tyrannis docstring and task bibliography.

Tyrannis extensions/conventions: configurable positive power (classical p=1),
setup before updates t=0..T-1, distinct historical leader identifiers,
strict snapshot replacement, frozen sweeps, clipping after the full equation,
unconditional consolidation, and ordered infinite fitness used only in ranking.
"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from math import pow
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
from tyrannis.algorithm.grey_wolf import GreyWolf, GreyWolfParticle
from tyrannis.core.algorithm import FITNESS_UNDEFINED
from tyrannis.processor.serial import Serial
from tyrannis.space.continuous import Continuous


@dataclass
class MovementInputs:
    positions: dict[str, dict[str, float]]
    leaders: tuple[GreyWolfParticle, ...]
    a: float
    random: dict[str, float | str]


class ObservedGreyWolf(GreyWolf):
    """Only copy observations; every lifecycle operation delegates to super()."""

    def leader_snapshots(self) -> tuple[GreyWolfParticle, ...]:
        # Public alpha/beta/delta expose IDs only; geometry needs the snapshots.
        return tuple(
            deepcopy(p) for p in (self._alpha, self._beta, self._delta) if p is not None
        )

    @override
    def pre_iteration(self, actual_iter: int) -> None:
        self.actual_iter: int = actual_iter
        if actual_iter == 0:
            self.initial: dict[str, GreyWolfParticle] = deepcopy(self.population)
            self.initialized: dict[str, GreyWolfParticle] = {}
            self.inputs: dict[tuple[int, str], MovementInputs] = {}
            self.candidates: dict[tuple[int, str], GreyWolfParticle] = {}
            self.states: dict[int, dict[str, GreyWolfParticle]] = {}
            self.best: dict[int, GreyWolfParticle] = {}
            self.controls: dict[int, float] = {}
            self.leaders: dict[int, tuple[GreyWolfParticle, ...]] = {}
            self.post_leaders: dict[int, tuple[GreyWolfParticle, ...]] = {}
        super().pre_iteration(actual_iter)
        self.controls[actual_iter] = self.a
        self.leaders[actual_iter] = self.leader_snapshots()

    @override
    def initialize_particle(self, identifier: str) -> GreyWolfParticle:
        particle = super().initialize_particle(identifier)
        self.initialized[identifier] = deepcopy(particle)
        return particle

    @override
    def update_particle(self, identifier: str) -> GreyWolfParticle:
        self.inputs[self.actual_iter, identifier] = MovementInputs(
            positions={key: p.variables.copy() for key, p in self.population.items()},
            leaders=self.leader_snapshots(),
            a=self.a,
            random=self.population[identifier].random_cache.copy(),
        )
        particle = super().update_particle(identifier)
        self.candidates[self.actual_iter, identifier] = deepcopy(particle)
        return particle

    @override
    def post_iteration(self, actual_iter: int) -> None:
        super().post_iteration(actual_iter)
        self.states[actual_iter] = deepcopy(self.population)
        self.post_leaders[actual_iter] = self.leader_snapshots()
        assert self.local_best is not None
        self.best[actual_iter] = deepcopy(self.local_best)


def run_wolves(
    bounds: dict[str, tuple[float, float]],
    *,
    template: ObservedGreyWolf | None = None,
    objective: Callable[..., float] = sphere,
    n_iterations: int = 5,
    seed: int = BASE_SEED,
) -> ObservedGreyWolf:
    if template is None:
        template = ObservedGreyWolf()
    processor = Serial()
    optimizer = factories.make_optimizer(
        Continuous(bounds, cost_function=objective),
        template,
        processor=processor,
        n_particles=5,
        n_iterations=n_iterations,
        seed=seed,
        history="iteration",
        fitness_failure_strategy="raise",
    ).fit()
    (executor,) = processor.processors_pool.values()
    # Observe the real replica located via the public pool (no algorithm accessor).
    assert isinstance(executor._algorithm, ObservedGreyWolf)  # pyright: ignore[reportPrivateUsage]
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
        assert len(population) == 5
        for particle in population.values():
            assert_particle_state_consistent(particle, bounds)
            assert all(np.isfinite(x) for x in particle.variables.values())
            assert not np.isnan(particle.fitness)
            assert particle.fitness == pytest.approx(
                objective(**particle.variables), rel=STRICT_RTOL, abs=STRICT_ATOL
            )
            assert particle.candidate_variables is None
            assert particle.candidate_fitness is None
            assert not particle.new_particle
    assert_initial_state(algorithm)
    rows: list[list[object]] = list(optimizer.internal_history_generator_)
    assert len(rows) == 5 * (n_iterations + 1)
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


def assert_initial_state(algorithm: ObservedGreyWolf) -> None:
    """Setup evaluates creation positions without forming leaders or moving wolves."""
    assert algorithm.controls[0] == 0
    assert np.isposinf(FITNESS_UNDEFINED)
    assert algorithm.leaders[0] == algorithm.post_leaders[0] == ()
    population = algorithm.states[0]
    assert set(algorithm.initialized) == set(algorithm.initial) == set(population)
    assert len(population) == 5
    for key, particle in population.items():
        created, evaluated = algorithm.initial[key], algorithm.initialized[key]
        assert created.candidate_variables is None
        assert created.candidate_fitness is None
        assert np.isposinf(created.fitness)
        assert particle.variables == evaluated.candidate_variables == created.variables
        assert particle.fitness == evaluated.candidate_fitness
        assert particle.random_cache == evaluated.random_cache == {}
    best = algorithm.best[0]
    assert best.fitness == min(p.fitness for p in population.values())
    assert best() == population[best.identifier]()


@pytest.fixture
def wolves(small_bounds: dict[str, tuple[float, float]]) -> ObservedGreyWolf:
    # This small trajectory includes clipping, improving and worsening moves.
    return run_wolves(small_bounds, seed=seed_for(0))


def test_initial_state_and_first_leaders_are_mathematically_valid(
    wolves: ObservedGreyWolf,
) -> None:
    # Setup checks run in every scenario, including both signs of infinite fitness.
    leaders = wolves.leaders[1]
    population = wolves.states[0]
    assert len(leaders) == len({p.identifier for p in leaders}) == 3
    assert [p.fitness for p in leaders] == sorted(
        p.fitness for p in population.values()
    )[:3]
    for leader in leaders:
        assert leader() == population[leader.identifier]()
    for key in population:
        assert wolves.inputs[1, key].positions == {
            k: p.variables for k, p in population.items()
        }


def assert_transitions(
    algorithm: ObservedGreyWolf,
    bounds: dict[str, tuple[float, float]],
    *,
    exponent: float = 1.0,
    enhanced: bool = False,
) -> set[str]:
    """Check each observed transition against vector equations.

    This advances no simulated state: inputs come from independent snapshots,
    and the schedule comes from t/T, never from the observed control value.
    """
    witnessed: set[str] = set()
    total = len(algorithm.states) - 1
    for (iteration, key), inputs in algorithm.inputs.items():
        population = algorithm.states[iteration - 1]
        before, after = population[key], algorithm.states[iteration][key]
        candidate = algorithm.candidates[iteration, key]
        assert inputs.positions == {k: p.variables for k, p in population.items()}
        assert [p() for p in inputs.leaders] == [
            p() for p in algorithm.leaders[iteration]
        ]
        assert [p() for p in algorithm.post_leaders[iteration]] == [
            p() for p in inputs.leaders
        ]
        assert candidate.variables == before.variables
        assert candidate.fitness == before.fitness
        assert candidate.random_cache == inputs.random
        assert candidate.candidate_variables is not None
        assert candidate.candidate_fitness is not None
        assert after.variables == candidate.candidate_variables
        assert after.fitness == candidate.candidate_fitness
        if after.fitness < before.fitness:
            witnessed.add("improvement")
        if after.fitness > before.fitness:
            witnessed.add("worsening")

        t = iteration - 1
        fraction = (total - t) / total if enhanced else t / total
        a = 2 * (1 - pow(fraction, exponent))
        assert inputs.a == pytest.approx(a, rel=STRICT_RTOL, abs=STRICT_ATOL)
        assert algorithm.controls[iteration] == inputs.a
        cache = inputs.random
        names = tuple(bounds)
        lower = np.array([bounds[j][0] for j in names], dtype=np.float64)
        upper = np.array([bounds[j][1] for j in names], dtype=np.float64)
        x = np.array([before.variables[j] for j in names], dtype=np.float64)
        actual = np.array(
            [candidate.candidate_variables[j] for j in names], dtype=np.float64
        )
        leaders = np.array(
            [[leader.variables[j] for j in names] for leader in inputs.leaders],
            dtype=np.float64,
        )
        # Rows are leaders; columns are dimensions. Only observed draws are inputs.
        r1 = np.array(
            [
                [float(cache[f"{j}-{label}-r1"]) for j in names]
                for label in ("alpha", "beta", "delta")
            ],
            dtype=np.float64,
        )
        r2 = np.array(
            [
                [float(cache[f"{j}-{label}-r2"]) for j in names]
                for label in ("alpha", "beta", "delta")
            ],
            dtype=np.float64,
        )
        assert ((0 <= r1) & (r1 <= 1)).all()
        assert ((0 <= r2) & (r2 <= 1)).all()
        coefficient_a, coefficient_c = a * (2 * r1 - 1), 2 * r2
        distances = np.abs(coefficient_c * leaders - x)
        terms = leaders - coefficient_a * distances
        weights = np.full(3, 1 / 3, dtype=np.float64)
        mean = weights @ terms
        assert np.isfinite([a, inputs.a]).all()
        for matrix in (
            x,
            leaders,
            coefficient_a,
            coefficient_c,
            distances,
            terms,
            mean,
        ):
            assert np.isfinite(matrix).all()
        assert (np.abs(coefficient_a) <= a).all()
        if a > 1 and np.any(np.abs(coefficient_a) > 1):
            witnessed.add("exploration")
        if a < 1:
            assert (np.abs(coefficient_a) < 1).all()
            witnessed.add("exploitation")
        if np.unique(np.concatenate((r1, r2))).size == 12:
            # Discriminating realized draws, not a statistical independence claim.
            witnessed.add("distinct_leader_dimension_draws")

        raw = mean
        if enhanced:
            donor = cache["random_particle"]
            assert isinstance(donor, str)
            assert donor in population and donor != key
            assert inputs.positions[donor] == population[donor].variables
            reference = np.array(
                [population[donor].variables[j] for j in names], dtype=np.float64
            )
            direction = reference - x
            r3 = np.array([float(cache[f"{j}-r3"]) for j in names], dtype=np.float64)
            r4 = np.array([float(cache[f"{j}-r4"]) for j in names], dtype=np.float64)
            assert ((0 <= r3) & (r3 <= 1)).all()
            assert ((0 <= r4) & (r4 <= 1)).all()
            guidance, exploration = 0.1 * r3 * mean, 0.9 * r4 * direction
            raw = guidance + exploration
            for vector in (reference, direction, guidance, exploration):
                assert np.isfinite(vector).all()
            if t == 0:
                assert a == 0
                assert np.count_nonzero(coefficient_a) == 0
                assert terms == pytest.approx(leaders, rel=STRICT_RTOL, abs=STRICT_ATOL)
                witnessed.add("zero_a_endpoint")
            active = (
                (lower < raw)
                & (raw < upper)
                & (np.abs(guidance) > STRICT_ATOL)
                & (np.abs(exploration) > STRICT_ATOL)
            )
            if np.any(active):
                assert actual[active] - exploration[active] == pytest.approx(
                    guidance[active], rel=STRICT_RTOL, abs=STRICT_ATOL
                )
                assert actual[active] - guidance[active] == pytest.approx(
                    exploration[active], rel=STRICT_RTOL, abs=STRICT_ATOL
                )
                witnessed.add("both_eegwo_terms")
            if np.unique(np.concatenate((r3, r4))).size == 4:
                witnessed.add("dimension_eegwo_draws")
            # An earlier candidate must not replace the consolidated reference.
            order = list(population)
            if order.index(donor) < order.index(key):
                pending = algorithm.candidates[iteration, donor].candidate_variables
                assert pending is not None
                asynchronous_direction = (
                    np.array([pending[j] for j in names], dtype=np.float64) - x
                )
                asynchronous_raw = guidance + 0.9 * r4 * asynchronous_direction
                if not np.allclose(
                    np.clip(raw, lower, upper),
                    np.clip(asynchronous_raw, lower, upper),
                    rtol=STRICT_RTOL,
                    atol=STRICT_ATOL,
                ):
                    witnessed.add("earlier_donor")
        else:
            interior = (lower < raw) & (raw < upper)
            if (
                interior.all()
                and (np.abs(terms) > STRICT_ATOL).all()
                and all(
                    not np.allclose(term, mean, rtol=STRICT_RTOL, atol=STRICT_ATOL)
                    for term in np.vsplit(terms, 3)
                )
            ):
                witnessed.add("three_leaders")
            if exponent != 1:
                linear_terms = leaders - 2 * (1 - t / total) * (2 * r1 - 1) * distances
                if np.any(
                    interior & (np.abs(raw - weights @ linear_terms) > STRICT_ATOL)
                ):
                    witnessed.add("nondefault_a_in_movement")
        assert np.isfinite(raw).all() and np.isfinite(actual).all()
        assert actual == pytest.approx(
            np.clip(raw, lower, upper), rel=STRICT_RTOL, abs=STRICT_ATOL
        ), (iteration, key, raw)
        witnessed.add(
            "clipping" if np.any((raw < lower) | (raw > upper)) else "no_clipping"
        )
    return witnessed


def test_standard_gwo_transition_matches_published_equations(
    wolves: ObservedGreyWolf, small_bounds: dict[str, tuple[float, float]]
) -> None:
    witnessed = assert_transitions(wolves, small_bounds)
    assert {
        "clipping",
        "no_clipping",
        "improvement",
        "worsening",
        "three_leaders",
        "exploration",
        "exploitation",
        "distinct_leader_dimension_draws",
    } <= witnessed


def test_eegwo_transition_matches_declared_equation(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    algorithm = run_wolves(
        small_bounds, template=ObservedGreyWolf(1.5, True), seed=seed_for(3)
    )
    witnessed = assert_transitions(algorithm, small_bounds, exponent=1.5, enhanced=True)
    _ = assert_leadership_and_best(algorithm)
    assert {
        "zero_a_endpoint",
        "both_eegwo_terms",
        "dimension_eegwo_draws",
        "earlier_donor",
        "clipping",
        "no_clipping",
        "improvement",
        "worsening",
        "distinct_leader_dimension_draws",
    } <= witnessed


def assert_leadership_and_best(algorithm: ObservedGreyWolf) -> set[str]:
    """Order statistics on available snapshots; no simulated leader lifecycle."""
    witnessed: set[str] = set()
    for iteration in algorithm.states:
        history = [
            p
            for i, state in algorithm.states.items()
            if i <= iteration
            for p in state.values()
        ]
        best = algorithm.best[iteration]
        assert best.fitness == min(p.fitness for p in history)
        assert any(best() == p() for p in history)
        if iteration == 0:
            continue
        previous_best = algorithm.best[iteration - 1]
        if best.fitness == previous_best.fitness:
            assert best() == previous_best()
        previous = {p.identifier: p for p in algorithm.leaders[iteration - 1]}
        current = algorithm.states[iteration - 1]
        available = [*previous.values(), *current.values()]
        # One order statistic per identifier, regardless of how many snapshots exist.
        minima = {
            key: min(p.fitness for p in available if p.identifier == key)
            for key in current
        }
        leaders = algorithm.leaders[iteration]
        assert len(leaders) == len({p.identifier for p in leaders}) == 3
        assert [p.fitness for p in leaders] == sorted(minima.values())[:3]
        for leader in leaders:
            key = leader.identifier
            assert any(leader() == p() for p in available)
            assert leader.fitness == minima[key]
            old = previous.get(key)
            if old is None:
                assert leader() == current[key]()
                if iteration > 1:
                    witnessed.add("new_leader")
            elif current[key].fitness < old.fitness:
                assert leader() == current[key]()
                witnessed.add("updated_snapshot")
            else:
                # Strict replacement preserves even tied historical positions.
                assert leader() == old()
                if current[key].fitness > old.fitness:
                    assert leader.variables != current[key].variables
                    witnessed.add("preserved_snapshot")
                    if np.isneginf(old.fitness):
                        assert np.isneginf(leader.fitness)
                        witnessed.add("preserved_negative_infinity")
        assert [p() for p in algorithm.post_leaders[iteration]] == [
            p() for p in leaders
        ]
        if best.fitness < leaders[0].fitness:
            witnessed.add("best_leader_lag")
            assert best.fitness < previous_best.fitness
            if iteration + 1 in algorithm.leaders:
                assert algorithm.leaders[iteration + 1][0].fitness == best.fitness
                witnessed.add("next_pre_incorporates_best")
    return witnessed


def test_historical_leaders_are_preserved_and_updated_at_the_correct_time(
    wolves: ObservedGreyWolf,
) -> None:
    assert {
        "new_leader",
        "updated_snapshot",
        "preserved_snapshot",
        "best_leader_lag",
        "next_pre_incorporates_best",
    } <= assert_leadership_and_best(wolves)


@pytest.mark.parametrize(
    ("exponent", "enhanced", "updates"),
    [
        pytest.param(0.5, False, 5, id="gwo-fast-decay"),
        pytest.param(1.0, False, 5, id="gwo-linear"),
        pytest.param(2.0, False, 5, id="gwo-slow-decay"),
        pytest.param(1.5, True, 5, id="eegwo-published-power"),
        pytest.param(1.0, False, 1, id="gwo-one-update"),
        pytest.param(1.5, True, 1, id="eegwo-one-update"),
    ],
)
def test_convergence_schedules_match_declared_formulations(
    small_bounds: dict[str, tuple[float, float]],
    exponent: float,
    enhanced: bool,
    updates: int,
) -> None:
    algorithm = run_wolves(
        small_bounds,
        template=ObservedGreyWolf(exponent, enhanced),
        n_iterations=updates,
    )
    expected = [
        2 * (1 - pow((updates - t) / updates if enhanced else t / updates, exponent))
        for t in range(updates)
    ]
    assert [algorithm.controls[i] for i in range(1, updates + 1)] == pytest.approx(
        expected, rel=STRICT_RTOL, abs=STRICT_ATOL
    )
    assert algorithm.controls[1] == (0 if enhanced else 2)
    assert len(algorithm.inputs) == 5 * updates
    witnessed = assert_transitions(
        algorithm, small_bounds, exponent=exponent, enhanced=enhanced
    )
    _ = assert_leadership_and_best(algorithm)
    if not enhanced and exponent != 1:
        assert "nondefault_a_in_movement" in witnessed
        for t in range(1, updates):
            linear = 2 * (1 - t / updates)
            if exponent < 1:
                assert algorithm.controls[t + 1] < linear
            else:
                assert algorithm.controls[t + 1] > linear


def positive_infinite_region(x: float, y: float) -> float:
    """A feasible half-plane, with hard positive-infinite penalties elsewhere."""
    return sphere(x, y) if x < -2 else np.inf


def signed_infinite_regions(x: float, y: float) -> float:
    """Three ordered fitness classes with finite geometry throughout the domain."""
    if x < -1:
        return -np.inf
    return np.inf if x > 2 else sphere(x, y)


def positive_infinite_plateau(x: float, y: float) -> float:
    del x, y
    return np.inf


def negative_infinite_plateau(x: float, y: float) -> float:
    del x, y
    return -np.inf


@pytest.mark.parametrize(
    ("enhanced", "plateau"),
    [
        pytest.param(False, False, id="gwo-mixed"),
        pytest.param(True, False, id="eegwo-mixed"),
        pytest.param(False, True, id="gwo-all-positive-infinity"),
        pytest.param(True, True, id="eegwo-all-positive-infinity"),
    ],
)
def test_positive_infinite_fitness_preserves_defined_leadership_and_dynamics(
    small_bounds: dict[str, tuple[float, float]], enhanced: bool, plateau: bool
) -> None:
    exponent = 1.5 if enhanced else 1.0
    algorithm = run_wolves(
        small_bounds,
        template=ObservedGreyWolf(exponent, enhanced),
        objective=positive_infinite_plateau if plateau else positive_infinite_region,
        seed=BASE_SEED if plateau else seed_for(2),
        n_iterations=2 if plateau else 5,
    )
    fitnesses = [p.fitness for p in algorithm.states[0].values()]
    assert sum(np.isposinf(f) for f in fitnesses) >= 2
    if plateau:
        assert all(np.isposinf(f) for f in fitnesses)
        assert all(np.isposinf(p.fitness) for p in algorithm.best.values())
        assert all(
            np.isposinf(p.fitness)
            for leaders in algorithm.leaders.values()
            for p in leaders
        )
    else:
        finite_count = sum(np.isfinite(f) for f in fitnesses)
        assert 0 < finite_count < 3
        leaders = algorithm.leaders[1]
        assert all(np.isfinite(p.fitness) for p in leaders[:finite_count])
        assert all(np.isposinf(p.fitness) for p in leaders[finite_count:])
        assert np.isfinite(algorithm.best[0].fitness)
    _ = assert_leadership_and_best(algorithm)
    witnessed = assert_transitions(
        algorithm, small_bounds, exponent=exponent, enhanced=enhanced
    )
    assert "distinct_leader_dimension_draws" in witnessed
    if enhanced:
        assert {"both_eegwo_terms", "earlier_donor"} <= witnessed


@pytest.mark.parametrize(
    ("enhanced", "plateau"),
    [
        pytest.param(False, False, id="gwo-mixed-signs"),
        pytest.param(True, False, id="eegwo-mixed-signs"),
        pytest.param(False, True, id="gwo-all-negative-infinity"),
        pytest.param(True, True, id="eegwo-all-negative-infinity"),
    ],
)
def test_negative_infinite_fitness_is_best_and_preserved_historically(
    small_bounds: dict[str, tuple[float, float]], enhanced: bool, plateau: bool
) -> None:
    assert np.isposinf(FITNESS_UNDEFINED)
    assert -np.inf != FITNESS_UNDEFINED
    exponent = 1.5 if enhanced else 1.0
    algorithm = run_wolves(
        small_bounds,
        template=ObservedGreyWolf(exponent, enhanced),
        objective=negative_infinite_plateau if plateau else signed_infinite_regions,
        seed=BASE_SEED if plateau else seed_for(3),
        n_iterations=2 if plateau else 5,
    )
    initial = algorithm.states[0]
    fitnesses = [p.fitness for p in initial.values()]
    assert sum(np.isneginf(f) for f in fitnesses) >= 2
    assert all(np.isneginf(p.fitness) for p in algorithm.best.values())
    # -inf is evaluated and participates in motion, never reclassified as undefined.
    assert all(not p.new_particle for p in initial.values())
    assert all((1, p.identifier) in algorithm.inputs for p in initial.values())
    if plateau:
        assert all(np.isneginf(f) for f in fitnesses)
        assert all(
            np.isneginf(p.fitness)
            for leaders in algorithm.leaders.values()
            for p in leaders
        )
    else:
        assert any(np.isfinite(f) for f in fitnesses)
        assert any(np.isposinf(f) for f in fitnesses)
        leaders = algorithm.leaders[1]
        assert np.isneginf(leaders[0].fitness) and np.isneginf(leaders[1].fitness)
        assert np.isfinite(leaders[2].fitness)
        assert -np.inf < leaders[2].fitness < np.inf
    witnessed = assert_leadership_and_best(algorithm)
    if not plateau:
        assert "preserved_negative_infinity" in witnessed
    movement = assert_transitions(
        algorithm, small_bounds, exponent=exponent, enhanced=enhanced
    )
    assert "distinct_leader_dimension_draws" in movement
    if enhanced:
        assert {"both_eegwo_terms", "earlier_donor"} <= movement
