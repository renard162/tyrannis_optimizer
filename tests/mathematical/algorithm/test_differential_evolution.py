"""DE/rand/1/bin through Optimizer + Continuous + the real Serial executor.

Storn & Price (1997), doi:10.1023/A:1008202821328, section 2, pp. 344-346:
mutation (2), binomial crossover (4), and strict greedy selection (p. 345).
Full text: https://sci2s.ugr.es/sites/default/files/files/Teaching/
GraduatesCourses/Metaheuristicas/Bibliography/DE-Storn-Price-1997.pdf

Clipping selected mutant coordinates is Tyrannis' bounded-domain convention,
not equation (2). Strict selection also follows DifferentialEvolution's
documented contract; it must not be replaced with a non-worsening (<=) rule.
The paper does not specify infinite fitness. No documented DE exception permits
replacing an infinite target by an equally infinite trial. ParticleBase's
unconditional consolidation of infinite targets is therefore tested against
the strict DE contract, not adopted as an independent oracle.
"""

from collections.abc import Callable
from copy import deepcopy
from math import isfinite
from typing import override

import numpy as np
import pytest

from tests._support.assertions import (
    assert_optimizer_result_consistent,
    assert_particle_state_consistent,
)
from tests._support.factories import make_optimizer
from tests._support.numerics import BASE_SEED, STRICT_ATOL, STRICT_RTOL, seed_for
from tests._support.objectives import constant_objective, sphere
from tyrannis.algorithm.differential_evolution import DEParticle, DifferentialEvolution
from tyrannis.core.algorithm import FITNESS_UNDEFINED
from tyrannis.processor.serial import Serial
from tyrannis.space.continuous import Continuous


class ObservedDifferentialEvolution(DifferentialEvolution):
    """Copy only mathematical state; super() owns RNG and every transition."""

    @override
    def pre_iteration(self, actual_iter: int) -> None:
        self.actual_iter: int = actual_iter
        if actual_iter == 0:
            self.initial: dict[str, DEParticle] = deepcopy(self.population)
            self.initialized: dict[str, DEParticle] = {}
            self.positions: dict[tuple[int, str], dict[str, dict[str, float]]] = {}
            self.candidates: dict[tuple[int, str], DEParticle] = {}
            self.before_post: dict[int, dict[str, DEParticle]] = {}
            self.states: dict[int, dict[str, DEParticle]] = {}
            self.best: dict[int, DEParticle] = {}
        super().pre_iteration(actual_iter)

    @override
    def initialize_particle(self, identifier: str) -> DEParticle:
        particle = super().initialize_particle(identifier)
        self.initialized[identifier] = deepcopy(particle)
        return particle

    @override
    def update_particle(self, identifier: str) -> DEParticle:
        self.positions[self.actual_iter, identifier] = {
            key: p.variables.copy() for key, p in self.population.items()
        }
        particle = super().update_particle(identifier)
        self.candidates[self.actual_iter, identifier] = deepcopy(particle)
        return particle

    @override
    def post_iteration(self, actual_iter: int) -> None:
        self.before_post[actual_iter] = deepcopy(self.population)
        super().post_iteration(actual_iter)
        self.states[actual_iter] = deepcopy(self.population)
        assert self.local_best is not None
        self.best[actual_iter] = deepcopy(self.local_best)


def run_de(
    bounds: dict[str, tuple[float, float]],
    *,
    mutation_factor: float = 0.8,
    crossover_rate: float = 0.5,
    n_particles: int = 5,
    n_iterations: int = 3,
    objective: Callable[..., float] = sphere,
    seed: int = BASE_SEED,
) -> ObservedDifferentialEvolution:
    algorithm = ObservedDifferentialEvolution(mutation_factor, crossover_rate)
    processor = Serial()
    optimizer = make_optimizer(
        Continuous(bounds, cost_function=objective),
        algorithm,
        processor=processor,
        n_particles=n_particles,
        n_iterations=n_iterations,
        seed=seed,
        history="iteration",
        fitness_failure_strategy="raise",
    ).fit()
    (executor,) = processor.processors_pool.values()
    # Only private access: observe the replica actually run by Serial.
    assert isinstance(executor._algorithm, ObservedDifferentialEvolution)  # pyright: ignore[reportPrivateUsage]
    executed = executor._algorithm  # pyright: ignore[reportPrivateUsage]
    assert executed is not algorithm
    assert set(executed.states) == set(range(n_iterations + 1))
    assert all(len(pop) == n_particles for pop in executed.states.values())
    assert set(executed.candidates) == {
        (t, key) for t in range(1, n_iterations + 1) for key in executed.initial
    }
    for population in executed.states.values():
        for particle in population.values():
            assert_particle_state_consistent(particle, bounds)
            assert all(np.isfinite(x) for x in particle.variables.values())
            assert not np.isnan(particle.fitness)
            assert particle.fitness == pytest.approx(
                objective(**particle.variables), rel=STRICT_RTOL, abs=STRICT_ATOL
            )
            assert particle.candidate_variables is None
            assert particle.candidate_fitness is None
    # History exposes exactly the consolidated states used by the oracle.
    rows: list[list[object]] = list(optimizer.internal_history_generator_)
    assert len(rows) == n_particles * (n_iterations + 1)
    for row in rows:
        iteration, identifier = row[1], row[2]
        assert isinstance(iteration, int) and isinstance(identifier, str)
        particle = executed.states[iteration][identifier]
        assert row[5:-1] == [particle.fitness, *particle.variables.values()]
    assert_optimizer_result_consistent(optimizer)
    assert optimizer.best_fitness == min(
        p.fitness
        for population in executed.states.values()
        for p in population.values()
    )
    assert_initial_population(executed, bounds, objective)
    return executed


def assert_initial_population(
    algorithm: ObservedDifferentialEvolution,
    bounds: dict[str, tuple[float, float]],
    objective: Callable[..., float],
) -> None:
    """Setup evaluates created positions without DE mutation or replacement."""
    initial = algorithm.states[0]
    assert set(algorithm.initialized) == set(initial) == set(algorithm.initial)
    assert not any(t == 0 for t, _ in algorithm.candidates)
    for key, particle in initial.items():
        evaluated = algorithm.initialized[key]
        assert_particle_state_consistent(evaluated, bounds)
        assert evaluated.random_cache == particle.random_cache == {}
        assert evaluated.candidate_variables == algorithm.initial[key].variables
        assert particle.variables == evaluated.candidate_variables
        assert evaluated.candidate_fitness == pytest.approx(
            objective(**particle.variables), rel=STRICT_RTOL, abs=STRICT_ATOL
        )
        assert particle.fitness == evaluated.candidate_fitness
        before_post = algorithm.before_post[0][key]
        assert before_post.variables == particle.variables
        assert before_post.fitness == particle.fitness
        assert before_post.candidate_variables is None
        assert before_post.candidate_fitness is None
    if any(np.isfinite(p.fitness) for p in initial.values()):
        best = algorithm.best[0]
        assert best.fitness == min(p.fitness for p in initial.values())
        assert best.variables == initial[best.identifier].variables


def test_initial_population_is_mathematically_valid(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    algorithm = run_de(small_bounds)
    first_key = next(iter(algorithm.initial))
    assert algorithm.positions[1, first_key] == {
        key: p.variables for key, p in algorithm.states[0].items()
    }


def assert_trial_equations(
    algorithm: ObservedDifferentialEvolution,
    bounds: dict[str, tuple[float, float]],
    *,
    factor: float,
    rate: float,
    objective: Callable[..., float] = sphere,
) -> set[str]:
    """Check published equations on observed inputs, tracking exercised branches."""
    witnessed: set[str] = set()
    for (iteration, key), trial in algorithm.candidates.items():
        previous = algorithm.states[iteration - 1]
        target = previous[key]
        # Every Serial call reads the SAME consolidated generation, including
        # donors whose candidates have already been produced in this sweep.
        assert algorithm.positions[iteration, key] == {
            donor: p.variables for donor, p in previous.items()
        }
        assert trial.variables == target.variables
        assert trial.fitness == target.fitness
        pending = algorithm.before_post[iteration][key]
        assert pending.variables == target.variables
        assert pending.fitness == target.fitness
        assert pending.candidate_variables == trial.candidate_variables
        assert pending.candidate_fitness == trial.candidate_fitness
        cache: dict[str, object] = trial.random_cache
        donors = [cache[f"donor-{i}"] for i in (1, 2, 3)]
        assert all(isinstance(donor, str) for donor in donors)
        donor_ids = [donor for donor in donors if isinstance(donor, str)]
        assert len(donor_ids) == len(set(donor_ids)) == 3
        assert key not in donor_ids
        assert set(donor_ids) <= set(previous)
        if len(previous) == 4:
            assert set(donor_ids) == set(previous) - {key}
        forced = cache["forced-variable"]
        assert isinstance(forced, str) and forced in bounds
        r1, r2, r3 = (previous[donor] for donor in donor_ids)
        assert trial.candidate_variables is not None
        assert trial.candidate_fitness is not None
        expected: dict[str, float] = {}
        for name, (lower, upper) in bounds.items():
            random = cache[f"{name}-crossover"]
            assert isinstance(random, float) and 0 <= random <= 1
            if random <= rate or name == forced:
                # Storn-Price (2), conditioned on the actual donor IDs at G.
                difference = r2.variables[name] - r3.variables[name]
                raw = r1.variables[name] + factor * difference
                assert np.isfinite(raw)
                expected[name] = min(upper, max(lower, raw))
                observed = trial.candidate_variables[name]
                if lower <= raw <= upper:
                    witnessed.add("unclipped")
                    assert observed == pytest.approx(
                        raw, rel=STRICT_RTOL, abs=STRICT_ATOL
                    )
                    if difference != 0:
                        witnessed.add("active-factor")
                        # A nonzero, unclipped difference distinguishes F from
                        # both an omitted differential and a hard-coded default.
                        assert raw != r1.variables[name]
                        if factor != 0.5:
                            assert raw != r1.variables[name] + 0.5 * difference
                else:
                    witnessed.add("clipped")
                    assert observed == (lower if raw < lower else upper)
                    assert observed != raw
                if random <= rate:
                    witnessed.add("probabilistic-mutant")
                if (
                    name == forced
                    and random > rate
                    and expected[name] != target.variables[name]
                ):
                    witnessed.add("forced-only-change")
                if any(p.fitness == np.inf for p in (r1, r2, r3)):
                    # Fitness is intentionally absent from equation (2).
                    witnessed.add("infinite-donor")
                for donor, weight in zip(
                    donor_ids, (1.0, factor, -factor), strict=True
                ):
                    earlier = algorithm.candidates.get((iteration, donor))
                    order = list(previous)
                    if earlier is not None and order.index(donor) < order.index(key):
                        assert earlier.candidate_variables is not None
                        # A candidate substituted for this donor must actually
                        # change a selected coordinate, even after clipping.
                        asynchronous_raw = raw + weight * (
                            earlier.candidate_variables[name]
                            - previous[donor].variables[name]
                        )
                        if min(upper, max(lower, asynchronous_raw)) != expected[name]:
                            witnessed.add("earlier-donor-candidate-differs")
            else:
                # Equation (4): inherited coordinates need no mutation oracle.
                witnessed.add("target")
                expected[name] = target.variables[name]
                assert trial.candidate_variables[name] == target.variables[name]
        assert trial.candidate_variables == pytest.approx(
            expected, rel=STRICT_RTOL, abs=STRICT_ATOL
        )
        assert_particle_state_consistent(trial, bounds)
        assert all(np.isfinite(value) for value in trial.candidate_variables.values())
        assert not np.isnan(trial.candidate_fitness)
        # For sphere this is directly sum(x_j**2), independent of the wrapper.
        expected_fitness = (
            sum(value**2 for value in expected.values())
            if objective is sphere
            else objective(**expected)
        )
        assert trial.candidate_fitness == pytest.approx(
            expected_fitness, rel=STRICT_RTOL, abs=STRICT_ATOL
        )
    return witnessed


def assert_strict_selection(algorithm: ObservedDifferentialEvolution) -> set[str]:
    """Finite ties retain the target; infinite ties have a dedicated regression."""
    branches: set[str] = set()
    for (iteration, key), trial in algorithm.candidates.items():
        before = algorithm.states[iteration - 1][key]
        after = algorithm.states[iteration][key]
        assert trial.candidate_fitness is not None
        if before.fitness == trial.candidate_fitness == np.inf:
            # Do not canonize ParticleBase's undocumented inf->inf override.
            # The minimal all-infinite test below asserts strict preservation.
            continue
        improved = trial.candidate_fitness < before.fitness
        branches.add(
            "better"
            if improved
            else "equal"
            if trial.candidate_fitness == before.fitness
            else "worse"
        )
        assert after.variables == (
            trial.candidate_variables if improved else before.variables
        )
        assert after.fitness == (
            trial.candidate_fitness if improved else before.fitness
        )
    return branches


def test_rand1_bin_transition_matches_published_equations(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    algorithm = run_de(small_bounds)
    branches = assert_trial_equations(algorithm, small_bounds, factor=0.8, rate=0.5)
    assert {
        "probabilistic-mutant",
        "target",
        "forced-only-change",
        "active-factor",
        "earlier-donor-candidate-differs",
    } <= branches
    assert {"better", "worse"} <= assert_strict_selection(algorithm)


@pytest.mark.parametrize("rate", [0.0, 1.0], ids=["forced-only", "all-mutant"])
def test_crossover_rate_endpoints(
    small_bounds: dict[str, tuple[float, float]],
    rate: float,
) -> None:
    # Four particles also proves that precisely the other three suffice.
    algorithm = run_de(small_bounds, crossover_rate=rate, n_particles=4, n_iterations=1)
    branches = assert_trial_equations(algorithm, small_bounds, factor=0.8, rate=rate)
    for trial in algorithm.candidates.values():
        assert all(trial.random_cache[f"{name}-crossover"] > 0 for name in small_bounds)
    if rate == 0:
        assert {"target", "forced-only-change"} <= branches
        assert "probabilistic-mutant" not in branches
    else:
        assert "probabilistic-mutant" in branches
        assert "target" not in branches
        assert "forced-only-change" not in branches
    _ = assert_strict_selection(algorithm)


def test_mutation_endpoint_clips_only_out_of_bounds_values(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    # F=2 is the documented upper endpoint; CR=1 exposes every mutant coordinate.
    algorithm = run_de(
        small_bounds, mutation_factor=2.0, crossover_rate=1.0, n_iterations=1
    )
    branches = assert_trial_equations(algorithm, small_bounds, factor=2.0, rate=1.0)
    assert {"clipped", "unclipped", "active-factor"} <= branches
    _ = assert_strict_selection(algorithm)


def test_equal_finite_fitness_preserves_target_despite_changed_trial(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    algorithm = run_de(small_bounds, objective=constant_objective, n_iterations=1)
    _ = assert_trial_equations(
        algorithm, small_bounds, factor=0.8, rate=0.5, objective=constant_objective
    )
    assert any(
        p.candidate_variables != p.variables for p in algorithm.candidates.values()
    )
    assert assert_strict_selection(algorithm) == {"equal"}


def test_mixed_infinite_fitness_uses_positions_and_recovers_finite_solutions(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    def objective(**variables: float) -> float:
        return np.inf if variables["x"] > 0 else sphere(**variables)

    algorithm = run_de(small_bounds, objective=objective, seed=seed_for(1))
    initial = algorithm.states[0]
    assert sum(p.fitness == np.inf for p in initial.values()) >= 2
    assert any(np.isfinite(p.fitness) for p in initial.values())
    branches = assert_trial_equations(
        algorithm, small_bounds, factor=0.8, rate=0.5, objective=objective
    )
    assert "infinite-donor" in branches
    directions: set[tuple[bool, bool]] = set()
    for trial in algorithm.candidates.values():
        assert trial.candidate_fitness is not None
        directions.add((isfinite(trial.fitness), isfinite(trial.candidate_fitness)))
    assert {(False, True), (True, False), (False, False)} <= directions
    assert {"better", "worse"} <= assert_strict_selection(algorithm)


def test_all_infinite_fitness_keeps_equations_defined_and_preserves_tied_targets(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    def objective(**variables: float) -> float:
        del variables
        return np.inf

    # Minimal DE population and one real generation reproduce the violation;
    # this is not a schedule endpoint test. Setup is independently checked too.
    algorithm = run_de(small_bounds, objective=objective, n_particles=4, n_iterations=1)
    branches = assert_trial_equations(
        algorithm, small_bounds, factor=0.8, rate=0.5, objective=objective
    )
    assert "infinite-donor" in branches
    assert all(p.fitness == np.inf for p in algorithm.states[0].values())
    assert all(p.fitness == np.inf for p in algorithm.states[1].values())
    assert all(p.candidate_fitness == np.inf for p in algorithm.candidates.values())
    assert any(
        p.candidate_variables != p.variables for p in algorithm.candidates.values()
    ), "An unchanged trial cannot distinguish the two consolidation contracts"
    # All donors, draws, trials, fitnesses, synchronization, history and absence
    # of NaN have been checked above, before testing the disputed selection.
    # DE post_iteration passes consolidate_new=False because inf < inf is false.
    # ParticleBase.consolidate nevertheless copies the candidate whenever the
    # current fitness is infinite. Its docstring and the DE documentation give
    # no infeasible-region exploration exception to strict greedy replacement.
    for key, before in algorithm.states[0].items():
        after = algorithm.states[1][key]
        assert after.variables == before.variables, (
            "PRODUCTION_CONTRACT_VIOLATION [DE greedy selection / np.inf / "
            "lifecycle-consolidation]: an infinite trial replaces an infinite "
            "target without strict improvement; ParticleBase overrides DE's "
            "rejection. No documented exception justifies this movement. "
            f"target={before.variables}, trial={algorithm.candidates[1, key].candidate_variables}, "
            f"consolidated={after.variables}"
        )


def test_negative_infinite_fitness_is_best_under_de_selection(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    # Strict DE selection and MATHTESTS 34.3 apply to evaluated -inf targets too.
    def objective(**variables: float) -> float:
        x = variables["x"]
        return -np.inf if -1 < x < 1 else np.inf if x > 1 else sphere(**variables)

    assert np.isposinf(FITNESS_UNDEFINED) and -np.inf != FITNESS_UNDEFINED
    # Two independent first generations cover all comparisons without relying
    # on a trajectory already changed by a wrong consolidation. 0 denotes finite.
    required = {
        (0.0, -np.inf),
        (np.inf, -np.inf),
        (-np.inf, 0.0),
        (-np.inf, np.inf),
        (-np.inf, -np.inf),
        (np.inf, 0.0),
    }
    witnessed: set[tuple[float, float]] = set()
    negative_donor = False
    violations: list[str] = []
    for case in (6, 10):
        algorithm = run_de(
            small_bounds, objective=objective, n_iterations=1, seed=seed_for(case)
        )
        _ = assert_trial_equations(
            algorithm, small_bounds, factor=0.8, rate=0.5, objective=objective
        )
        initial = algorithm.states[0]
        assert any(np.isneginf(p.fitness) for p in initial.values())
        assert any(np.isposinf(p.fitness) for p in initial.values())
        assert any(-np.inf < p.fitness < np.inf for p in initial.values())
        assert all(np.isneginf(best.fitness) for best in algorithm.best.values())
        for (iteration, key), trial in algorithm.candidates.items():
            before, after = initial[key], algorithm.states[iteration][key]
            assert not before.new_particle and not after.new_particle
            assert trial.candidate_fitness is not None
            cache: dict[str, object] = trial.random_cache
            for index in (1, 2, 3):
                donor = cache[f"donor-{index}"]
                assert isinstance(donor, str)
                negative_donor |= bool(np.isneginf(initial[donor].fitness))
            pair = (
                0.0 if isfinite(before.fitness) else float(before.fitness),
                0.0
                if isfinite(trial.candidate_fitness)
                else float(trial.candidate_fitness),
            )
            if pair not in required:
                continue  # Other comparisons have their existing dedicated tests.
            assert trial.candidate_variables != before.variables
            witnessed.add(pair)
            improved = trial.candidate_fitness < before.fitness
            expected_variables = (
                trial.candidate_variables if improved else before.variables
            )
            expected_fitness = trial.candidate_fitness if improved else before.fitness
            if (
                after.variables != expected_variables
                or after.fitness != expected_fitness
            ):
                violations.append(
                    f"case={case}, {key}, target={before.fitness}, "
                    + f"candidate={trial.candidate_fitness}, consolidated={after.fitness}, "
                    + f"expected_variables={expected_variables}, actual={after.variables}"
                )
    assert witnessed == required and negative_donor
    assert not violations, (
        "PRODUCTION_CONTRACT_VIOLATION [DE / -np.inf]: strict selection rejects "
        + "worse or tied candidates; -inf is not FITNESS_UNDEFINED; "
        + "; ".join(violations)
    )
