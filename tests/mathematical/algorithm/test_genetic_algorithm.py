"""Real-coded GA mathematics through Optimizer + Continuous + real Serial.

Sources: Eshelman & Schaffer (1993), BLX-alpha, doi:10.1016/B978-0-08-
094832-4.50018-0 (publisher full text unavailable); Deb & Agrawal (1995),
Simulated Binary Crossover, Complex Systems 9, 115-148 (original PDF exceeds
the browser limit). Its density and child equations were consulted in the
authors' Deb, Karthik & Okabe (2007), section 2, equations (2)-(4):
https://repository.ias.ac.in/81657/1/82-p.pdf. Deb & Deb (2014),
doi:10.1504/IJAISC.2014.059280, author's report:
https://www.egr.msu.edu/~kdeb/papers/k2012016.pdf.

Tyrannis conventions are tested separately from those publications: clipped
arithmetic extrapolation, bounded SBX with separate side normalization and a
near-parent safeguard, clipped Gaussian noise, range-normalized bounded
polynomial mutation, mutation intensity, minimization deficit weights and
infinite/plateau fallbacks, linear rank weights, synchronous candidates,
elite cap, and a no-repeat steady-state cycle with immediate spillover.
Deb & Deb's distance-scaled mutation and truncated Gaussian are NOT identical
to these Tyrannis bounded-domain conventions. Setup is actual_iter=0; genetic
generation t+1 is produced from the consolidated population at actual_iter=t.
"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from math import ceil, fsum, isfinite, pow, ulp
from typing import cast, override

import numpy as np
import pytest
from numpy.typing import NDArray

from tests._support import factories
from tests._support.assertions import (
    assert_optimizer_result_consistent,
    assert_particle_state_consistent,
)
from tests._support.numerics import BASE_SEED, STRICT_ATOL, STRICT_RTOL, seed_for
from tests._support.objectives import constant_objective, sphere
from tyrannis import Optimizer
from tyrannis.algorithm.genetic_algorithm import GAParticle, GeneticAlgorithm
from tyrannis.core.algorithm import FITNESS_UNDEFINED, AlgorithmBase
from tyrannis.processor.serial import Serial
from tyrannis.space.continuous import Continuous


@dataclass
class Selection:
    iteration: int
    target: GAParticle
    population: dict[str, GAParticle]
    parent: GAParticle
    random_key: str
    tournament_key: str


@dataclass
class Crossing:
    parents: tuple[GAParticle, GAParticle]
    result: dict[str, float]


@dataclass
class Cycle:
    active: set[str]
    queue: list[str]
    members: set[str]


class ObservedGeneticAlgorithm(GeneticAlgorithm):
    """Snapshots only; super() owns the RNG, parents and every transition."""

    def cycle_snapshot(self) -> Cycle:
        # No public accessors expose scheduling state; copy it without mutation.
        return Cycle(
            self._active_identifiers.copy(),
            self._steady_state_queue.copy(),
            self._steady_state_cycle_identifiers.copy(),
        )

    @override
    def pre_iteration(self, actual_iter: int) -> None:
        self.actual_iter: int = actual_iter
        if actual_iter == 0:
            self.initial: dict[str, GAParticle] = deepcopy(self.population)
            self.initialized: dict[str, GAParticle] = {}
            self.states: dict[int, dict[str, GAParticle]] = {}
            self.before_post: dict[int, dict[str, GAParticle]] = {}
            self.inputs: dict[tuple[int, str], dict[str, GAParticle]] = {}
            self.candidates: dict[tuple[int, str], GAParticle] = {}
            self.crossings: dict[tuple[int, str], Crossing] = {}
            self.mutations: dict[
                tuple[int, str], tuple[dict[str, float], dict[str, float]]
            ] = {}
            self.selections: list[Selection] = []
            self.probabilities: list[tuple[int, list[float]]] = []
            self.best: dict[int, GAParticle] = {}
            self.elites: dict[int, set[str]] = {}
            self.cycles: dict[int, Cycle] = {}
            self.synchronized: dict[int, Cycle] = {}
            self.started: dict[int, list[Cycle]] = {}
            self.effective_probabilities: dict[int, float | None] = {}
        super().pre_iteration(actual_iter)
        self.cycles[actual_iter] = self.cycle_snapshot()
        self.effective_probabilities[actual_iter] = self._effective_mutation_probability

    @override
    def _synchronize_steady_state_cycle(self, eligible_identifiers: list[str]) -> None:
        super()._synchronize_steady_state_cycle(eligible_identifiers)
        self.synchronized[self.actual_iter] = self.cycle_snapshot()

    @override
    def _start_steady_state_cycle(self, eligible_identifiers: list[str]) -> None:
        super()._start_steady_state_cycle(eligible_identifiers)
        self.started.setdefault(self.actual_iter, []).append(self.cycle_snapshot())

    @override
    def initialize_particle(self, identifier: str) -> GAParticle:
        particle = super().initialize_particle(identifier)
        self.initialized[identifier] = deepcopy(particle)
        return particle

    @override
    def _select_parent(
        self, particle: GAParticle, selection_random_key: str, tournament_key: str
    ) -> GAParticle:
        available = deepcopy(self.population)
        parent = super()._select_parent(particle, selection_random_key, tournament_key)
        self.selections.append(
            Selection(
                self.actual_iter,
                deepcopy(particle),
                available,
                deepcopy(parent),
                selection_random_key,
                tournament_key,
            )
        )
        return parent

    @override
    def _selection_probabilities(self) -> NDArray[np.float64]:
        # Production annotates bare ndarray; refine its numeric dtype locally.
        observe = cast(
            Callable[[], NDArray[np.float64]], super()._selection_probabilities
        )
        probabilities = observe()
        # ndarray.tolist() erases dtype in NumPy's annotation; this 1D float64
        # observation contains Python floats and does not modify the returned array.
        values = cast(list[float], probabilities.tolist())
        self.probabilities.append((self.actual_iter, values))
        return probabilities

    @override
    def _crossover(
        self, particle: GAParticle, parent_1: GAParticle, parent_2: GAParticle
    ) -> dict[str, float]:
        parents = deepcopy((parent_1, parent_2))
        result = super()._crossover(particle, parent_1, parent_2)
        self.crossings[self.actual_iter, particle.identifier] = Crossing(
            parents, result.copy()
        )
        return result

    @override
    def _mutate(
        self, variables: dict[str, float], particle: GAParticle
    ) -> dict[str, float]:
        before = variables.copy()
        result = super()._mutate(variables, particle)
        self.mutations[self.actual_iter, particle.identifier] = (before, result.copy())
        return result

    @override
    def _elite_identifiers(self) -> set[str]:
        elites = super()._elite_identifiers()
        self.elites[self.actual_iter] = elites.copy()
        return elites

    @override
    def update_particle(self, identifier: str) -> GAParticle:
        self.inputs[self.actual_iter, identifier] = deepcopy(self.population)
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


def run_ga(
    template: ObservedGeneticAlgorithm,
    bounds: dict[str, tuple[float, float]],
    *,
    objective: Callable[..., float] = sphere,
    seed: int = BASE_SEED,
    n_particles: int = 5,
    n_iterations: int = 3,
) -> ObservedGeneticAlgorithm:
    processor = Serial()
    # Shared factory omits AlgorithmBase's type parameter; its return is concrete.
    build = cast(Callable[..., Optimizer], factories.make_optimizer)
    optimizer = build(
        Continuous(bounds, cost_function=objective),
        template,
        processor=processor,
        seed=seed,
        n_particles=n_particles,
        n_iterations=n_iterations,
        history="iteration",
        fitness_failure_strategy="raise",
    ).fit()
    (executor,) = processor.processors_pool.values()
    # The public pool locates the actual replica; no public algorithm accessor.
    # ProcessorBase erases the particle generic; recover it for this GA executor.
    algorithm = cast(AlgorithmBase[GAParticle], executor._algorithm)  # pyright: ignore[reportPrivateUsage]
    assert isinstance(algorithm, ObservedGeneticAlgorithm)
    assert algorithm is not template
    assert set(algorithm.states) == set(range(n_iterations + 1))
    assert set(algorithm.candidates) == {
        (t, key) for t in range(1, n_iterations + 1) for key in algorithm.initial
    }
    history: list[GAParticle] = []
    for iteration, population in algorithm.states.items():
        assert len(population) == n_particles
        history.extend(population.values())
        for p in population.values():
            assert_particle_state_consistent(p, bounds)
            assert all(isfinite(x) for x in p.variables.values())
            assert p.fitness == pytest.approx(
                objective(**p.variables), rel=STRICT_RTOL, abs=STRICT_ATOL
            )
            assert p.candidate_variables is None and p.candidate_fitness is None
            assert not p.new_particle
        best = algorithm.best[iteration]
        assert best.fitness == min(p.fitness for p in history)
        assert any(best() == p() for p in history)
        if iteration and best.fitness == algorithm.best[iteration - 1].fitness:
            assert best() == algorithm.best[iteration - 1]()
    assert_optimizer_result_consistent(optimizer)
    assert optimizer.best_fitness == algorithm.best[n_iterations].fitness
    rows: list[list[object]] = list(optimizer.internal_history_generator_)
    assert len(rows) == n_particles * (n_iterations + 1)
    for row in rows:
        iteration, identifier = row[1], row[2]
        assert isinstance(iteration, int) and isinstance(identifier, str)
        p = algorithm.states[iteration][identifier]
        assert row[5:-1] == [p.fitness, *p.variables.values()]
    for candidate in algorithm.candidates.values():
        assert_particle_state_consistent(candidate, bounds)
        assert candidate.candidate_variables is not None
        assert candidate.candidate_fitness == pytest.approx(
            objective(**candidate.candidate_variables), rel=STRICT_RTOL, abs=STRICT_ATOL
        )
    assert_setup(algorithm, bounds)
    return algorithm


def assert_setup(
    algorithm: ObservedGeneticAlgorithm, bounds: dict[str, tuple[float, float]]
) -> None:
    """Evaluated creation positions, ready for the first genetic generation."""
    assert np.isposinf(FITNESS_UNDEFINED) and -np.inf != FITNESS_UNDEFINED
    assert algorithm.cycles[0] == Cycle(set(), [], set())
    expected_probability = (
        1 / len(bounds)
        if algorithm.mutation_probability is None
        else algorithm.mutation_probability
    )
    assert all(
        value == expected_probability
        for value in algorithm.effective_probabilities.values()
    )
    assert set(algorithm.initialized) == set(algorithm.initial)
    assert all(s.iteration > 0 for s in algorithm.selections)
    assert all(t > 0 for t, _ in algorithm.crossings)
    assert all(t > 0 for t, _ in algorithm.mutations)
    for key, created in algorithm.initial.items():
        evaluated, consolidated = algorithm.initialized[key], algorithm.states[0][key]
        assert_particle_state_consistent(created, bounds)
        assert created.fitness == FITNESS_UNDEFINED
        assert created.candidate_variables is None and created.candidate_fitness is None
        assert (
            created.variables == evaluated.candidate_variables == consolidated.variables
        )
        assert consolidated.fitness == evaluated.candidate_fitness
        assert evaluated.random_cache == consolidated.random_cache == {}
        assert algorithm.before_post[0][key]() == consolidated()
        assert algorithm.before_post[0][key].candidate_variables is None
    assert algorithm.best[0].fitness == min(
        p.fitness for p in algorithm.states[0].values()
    )


def assert_survival_and_synchrony(algorithm: ObservedGeneticAlgorithm) -> set[str]:
    """Order statistics and simultaneous replacement, without replaying a GA."""
    evidence: set[str] = set()
    for iteration in range(1, len(algorithm.states)):
        previous = algorithm.states[iteration - 1]
        elites = algorithm.elites[iteration]
        count = (
            min(algorithm.elite_count, len(previous) - 1)
            if algorithm.survival == "elitism"
            else 0
        )
        assert len(elites) == count
        assert (
            sorted(previous[key].fitness for key in elites)
            == sorted(p.fitness for p in previous.values())[:count]
        )
        for key, before in previous.items():
            snapshot = algorithm.inputs[iteration, key]
            assert {k: p() for k, p in snapshot.items()} == {
                k: p() for k, p in previous.items()
            }
            candidate = algorithm.candidates[iteration, key]
            assert candidate.random_cache == snapshot[key].random_cache
            after = algorithm.states[iteration][key]
            assert candidate() == before()
            pending = algorithm.before_post[iteration][key]
            assert pending() == before()
            assert pending.candidate_variables == candidate.candidate_variables
            assert pending.candidate_fitness == candidate.candidate_fitness
            assert (
                candidate.candidate_variables is not None
                and candidate.candidate_fitness is not None
            )
            if key in elites:
                assert after() == before()
                if candidate.candidate_variables != before.variables:
                    evidence.add("elite-candidate-discarded")
            else:
                assert after.variables == candidate.candidate_variables
                assert after.fitness == candidate.candidate_fitness
                if after.fitness > before.fitness:
                    evidence.add("worse-accepted")
            if algorithm.population_method == "generational":
                assert (iteration, key) in algorithm.crossings
                assert (iteration, key) in algorithm.mutations
        for selection in (s for s in algorithm.selections if s.iteration == iteration):
            assert {k: p() for k, p in selection.population.items()} == {
                k: p() for k, p in previous.items()
            }
            parent = selection.parent
            assert parent() == previous[parent.identifier]()
            order = list(previous)
            if order.index(parent.identifier) < order.index(
                selection.target.identifier
            ):
                earlier = algorithm.candidates[iteration, parent.identifier]
                if earlier.candidate_variables != parent.variables:
                    evidence.add("earlier-candidate-not-parent")
    return evidence


@pytest.mark.parametrize(
    "probability", [None, 0.0, 1.0], ids=["default", "never", "always"]
)
def test_setup_and_synchronous_generational_survival(
    small_bounds: dict[str, tuple[float, float]],
    probability: float | None,
) -> None:
    algorithm = run_ga(
        ObservedGeneticAlgorithm(
            population_method="generational",
            mutation_probability=probability,
        ),
        small_bounds,
    )
    assert "earlier-candidate-not-parent" in assert_survival_and_synchrony(algorithm)
    decisions = assert_mutation(algorithm, small_bounds)
    assert ("applied" in decisions) == (probability != 0)
    assert ("skipped" in decisions) == (probability != 1)
    _ = assert_crossovers(algorithm, small_bounds)
    _ = assert_selection(algorithm)


def fitness_probabilities(fitness: list[float]) -> list[float]:
    """Exact rational deficit ratios; no floating scaling algorithm is duplicated.

    Tyrannis minimization convention: w_i=max(finite f)-f_i; equal usable
    fitness and signed infinities use symmetric project fallbacks.
    """
    minimizers = [i for i, f in enumerate(fitness) if f == -np.inf]
    usable = [i for i, f in enumerate(fitness) if isfinite(f)]
    if minimizers or not usable or len({fitness[i] for i in usable}) == 1:
        support = minimizers or usable or list(range(len(fitness)))
        return [1 / len(support) if i in support else 0.0 for i in range(len(fitness))]
    worst = max(Fraction(fitness[i]) for i in usable)
    weights = [worst - Fraction(f) if isfinite(f) else Fraction(0) for f in fitness]
    return [float(w / sum(weights)) for w in weights]


def assert_selection(algorithm: ObservedGeneticAlgorithm) -> set[str]:
    evidence: set[str] = set()
    assert len(algorithm.selections) == 2 * len(algorithm.crossings)
    for index, observation in enumerate(algorithm.selections):
        assert observation.random_key == f"selection-{index % 2}"
        assert observation.tournament_key == f"tournament-{index % 2}"
        if index % 2:
            first = algorithm.selections[index - 1]
            assert (first.iteration, first.target.identifier) == (
                observation.iteration,
                observation.target.identifier,
            )
            crossing = algorithm.crossings[
                observation.iteration, observation.target.identifier
            ]
            assert [p() for p in crossing.parents] == [
                first.parent(),
                observation.parent(),
            ]
        population = list(observation.population.values())
        fitness = [float(p.fitness) for p in population]
        cache: dict[str, float | int] = observation.target.random_cache
        if algorithm.selection == "tournament":
            indices = [
                int(cache[f"{observation.tournament_key}-{i}"])
                for i in range(algorithm.tournament_size)
            ]
            assert all(0 <= i < len(population) for i in indices)
            assert observation.parent.fitness == min(fitness[i] for i in indices)
            assert observation.parent.identifier in {
                population[i].identifier for i in indices
            }
            if len(set(indices)) < len(indices):
                evidence.add("with-replacement")
            if -np.inf in (fitness[i] for i in indices):
                evidence.add("negative-infinity-wins")
            if any(isfinite(fitness[i]) for i in indices) and any(
                fitness[i] == np.inf for i in indices
            ):
                assert observation.parent.fitness < np.inf
                evidence.add("finite-beats-positive-infinity")
            continue
        iteration, observed = algorithm.probabilities[index]
        assert iteration == observation.iteration
        assert all(isfinite(p) and p >= 0 for p in observed)
        assert fsum(observed) == pytest.approx(1, rel=STRICT_RTOL, abs=STRICT_ATOL)
        if algorithm.selection == "fitness":
            expected = fitness_probabilities(fitness)
            assert observed == pytest.approx(expected, rel=STRICT_RTOL, abs=STRICT_ATOL)
        else:
            # Tied identities are unspecified: check the rank multiset within each
            # tie group, then use the observed admissible assignment for its CDF.
            total = len(population) * (len(population) + 1) / 2
            for f in set(fitness):
                better = sum(other < f for other in fitness)
                tied = sum(other == f for other in fitness)
                ranks = [(len(population) - better - j) / total for j in range(tied)]
                assert sorted(
                    observed[i] for i, value in enumerate(fitness) if value == f
                ) == pytest.approx(sorted(ranks), rel=STRICT_RTOL, abs=STRICT_ATOL)
            expected = list(observed)
        draw = float(cache[observation.random_key])
        cumulative = [fsum(expected[: i + 1]) for i in range(len(expected))]
        cumulative[-1] = 1.0
        selected = next(i for i, edge in enumerate(cumulative) if draw < edge)
        assert observation.parent() == population[selected]()
    return evidence


@pytest.mark.parametrize(
    "selection",
    ["tournament", "fitness", "ranking"],
    ids=["tournament", "fitness", "ranking"],
)
def test_parent_selection_and_finite_ties(
    small_bounds: dict[str, tuple[float, float]],
    selection: str,
) -> None:
    for objective in (sphere, constant_objective):
        algorithm = run_ga(
            ObservedGeneticAlgorithm(
                population_method="generational",
                selection=selection,
            ),
            small_bounds,
            objective=objective,
        )
        evidence = assert_selection(algorithm)
        assert "elite-candidate-discarded" in assert_survival_and_synchrony(algorithm)
        if selection == "tournament":
            assert "with-replacement" in evidence


def test_fitness_weights_preserve_exact_ratios_at_overflow_scale(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    def objective(x: float, y: float) -> float:
        del y
        return x / 5 * 1.7e308

    algorithm = run_ga(
        ObservedGeneticAlgorithm(
            population_method="generational",
            selection="fitness",
        ),
        small_bounds,
        objective=objective,
        n_iterations=1,
        seed=seed_for(0),
    )
    fitness = [float(p.fitness) for p in algorithm.states[0].values()]
    assert max(map(Fraction, fitness)) - min(map(Fraction, fitness)) > Fraction(
        float(np.finfo(float).max)
    )
    _ = assert_selection(algorithm)
    _ = assert_survival_and_synchrony(algorithm)


@pytest.mark.parametrize("selection", ["tournament", "fitness", "ranking"])
def test_signed_infinity_selection_elites_and_historical_minima(
    small_bounds: dict[str, tuple[float, float]],
    selection: str,
) -> None:
    def mixed(x: float, y: float) -> float:
        return np.inf if x > 0 else sphere(x, y)

    def plateau(x: float, y: float) -> float:
        return np.inf if x > 0 else constant_objective(x, y)

    def signed(x: float, y: float) -> float:
        return -np.inf if -1 < x < 1 else np.inf if x > 1 else sphere(x, y)

    def several_minima(x: float, y: float) -> float:
        return -np.inf if x < 0 else np.inf if x > 3 else sphere(x, y)

    def all_positive(**variables: float) -> float:
        del variables
        return np.inf

    def all_negative(**variables: float) -> float:
        del variables
        return -np.inf

    evidence: set[str] = set()
    for objective in (
        mixed,
        plateau,
        signed,
        several_minima,
        all_positive,
        all_negative,
    ):
        algorithm = run_ga(
            ObservedGeneticAlgorithm(
                population_method="generational",
                selection=selection,
                mutation="gaussian",
                mutation_probability=1,
                gaussian_sigma=0.5,
            ),
            small_bounds,
            objective=objective,
            n_iterations=2,
            seed=seed_for(5),
        )
        initial = [float(p.fitness) for p in algorithm.states[0].values()]
        if objective in (mixed, plateau):
            assert sum(f == np.inf for f in initial) >= 2
            assert any(isfinite(f) for f in initial)
        elif objective is signed:
            assert initial.count(-np.inf) == 1
            assert np.inf in initial and any(isfinite(f) for f in initial)
        elif objective is several_minima:
            assert initial.count(-np.inf) >= 2
            assert np.inf in initial and any(isfinite(f) for f in initial)
        else:
            assert set(initial) == {np.inf if objective is all_positive else -np.inf}
        evidence |= assert_selection(algorithm)
        assert "elite-candidate-discarded" in assert_survival_and_synchrony(algorithm)
        if -np.inf in initial:
            assert all(best.fitness == -np.inf for best in algorithm.best.values())
            for t, elites in algorithm.elites.items():
                assert all(
                    algorithm.states[t - 1][key].fitness == -np.inf for key in elites
                )
        for candidate in algorithm.candidates.values():
            assert candidate.candidate_variables is not None
            assert candidate.candidate_fitness == objective(
                **candidate.candidate_variables
            )
    if selection == "tournament":
        assert {
            "negative-infinity-wins",
            "finite-beats-positive-infinity",
            "with-replacement",
        } <= evidence


def assert_mutation(
    algorithm: ObservedGeneticAlgorithm,
    bounds: dict[str, tuple[float, float]],
) -> set[str]:
    """One local perturbation per observation; never advance a reference trajectory."""
    evidence: set[str] = set()
    probability = (
        1 / len(bounds)
        if algorithm.mutation_probability is None
        else algorithm.mutation_probability
    )
    for key, (before, after) in algorithm.mutations.items():
        candidate = algorithm.candidates[key]
        cache: dict[str, float] = candidate.random_cache
        assert before == algorithm.crossings[key].result
        assert after == candidate.candidate_variables
        for name, (lower, upper) in bounds.items():
            current, span = before[name], upper - lower
            if cache[f"{name}-mutation"] >= probability:
                assert after[name] == current
                evidence.add("skipped")
                continue
            evidence.add("applied")
            if algorithm.mutation == "gaussian":
                delta = (
                    algorithm.mutation_intensity
                    * algorithm.gaussian_sigma
                    * span
                    * cache[f"{name}-gaussian"]
                )
            else:
                u = cache[f"{name}-polynomial"]
                # Integrate density (1-|delta|)^eta on the chosen feasible half,
                # invert its normalized CDF. Reflect upper moves onto lower moves.
                direction = -1 if u <= 0.5 else 1
                distance = (
                    current - lower if direction < 0 else upper - current
                ) / span
                quantile = 2 * u if direction < 0 else 2 * (1 - u)
                power = algorithm.polynomial_eta + 1
                tail = pow(1 - distance, power)
                magnitude = 1 - pow(tail + quantile * (1 - tail), 1 / power)
                delta = direction * algorithm.mutation_intensity * span * magnitude
                if algorithm.polynomial_eta == 0:
                    # A constant density is uniform on the feasible chosen half.
                    assert delta == pytest.approx(
                        direction
                        * algorithm.mutation_intensity
                        * span
                        * distance
                        * (1 - quantile),
                        rel=STRICT_RTOL,
                        abs=STRICT_ATOL,
                    )
                evidence.add("lower" if direction < 0 else "upper")
                if 0 < distance < 1 and magnitude > STRICT_ATOL:
                    evidence.add(
                        "distance-lower" if direction < 0 else "distance-upper"
                    )
                if current in (lower, upper):
                    evidence.add("boundary-input")
            raw = current + delta
            # NumPy's scalar clip stub returns Any; float64 inputs preserve dtype.
            expected = cast(np.float64, np.clip(raw, lower, upper))
            assert after[name] == pytest.approx(
                expected, rel=STRICT_RTOL, abs=STRICT_ATOL
            ), (key, name, before, cache, raw)
            evidence.add("unclipped" if lower <= raw <= upper else "clipped")
            if lower < raw < upper and abs(delta) > STRICT_ATOL:
                evidence.add("nonzero-interior")
                if algorithm.mutation == "gaussian":
                    assert (after[name] - current) / span == pytest.approx(
                        algorithm.mutation_intensity
                        * algorithm.gaussian_sigma
                        * cache[f"{name}-gaussian"],
                        rel=STRICT_RTOL,
                        abs=STRICT_ATOL,
                    )
    return evidence


def assert_crossovers(
    algorithm: ObservedGeneticAlgorithm,
    bounds: dict[str, tuple[float, float]],
) -> set[str]:
    evidence: set[str] = set()
    for key, crossing in algorithm.crossings.items():
        cache: dict[str, float] = algorithm.candidates[key].random_cache
        first, second = crossing.parents
        if cache["crossover"] >= algorithm.crossover_probability:
            index = int(cache["crossover-parent"] >= 0.5)
            assert crossing.result == crossing.parents[index].variables
            if first.variables != second.variables:
                evidence.add(f"copy-parent-{index}")
            continue
        for name, (lower, upper) in bounds.items():
            p, q = first.variables[name], second.variables[name]
            if algorithm.crossover == "arithmetic":
                raw = (
                    algorithm.arithmetic_alpha * p
                    + (1 - algorithm.arithmetic_alpha) * q
                )
            elif algorithm.crossover == "blx":
                left, right = min(p, q), max(p, q)
                # Uniform quantile on the extended parental interval.
                raw = left + (
                    (1 + 2 * algorithm.blx_alpha) * cache[f"{name}-blx"]
                    - algorithm.blx_alpha
                ) * (right - left)
                if left < raw < right:
                    evidence.add("parental-interior")
                if raw < left or raw > right:
                    evidence.add("parental-extrapolation")
            else:
                assert algorithm.crossover == "sbx"
                if p == q:
                    raw = p
                    evidence.add("equal-parents")
                else:
                    # Published SBX inverse CDF, truncated at the selected side's
                    # bound. F(beta)=beta**k/2 below 1; 1-beta**(-k)/2 above.
                    midpoint, half_gap = (p + q) / 2, abs(q - p) / 2
                    sign = 1 if cache[f"{name}-sbx-child"] >= 0.5 else -1
                    limit = (
                        upper - midpoint if sign > 0 else midpoint - lower
                    ) / half_gap
                    power = algorithm.sbx_eta + 1
                    probability = cache[f"{name}-sbx"] * (1 - 0.5 * pow(limit, -power))
                    spread = (
                        pow(2 * probability, 1 / power)
                        if probability <= 0.5
                        else pow(2 * (1 - probability), -1 / power)
                    )
                    raw = midpoint + sign * half_gap * spread
                    if p in (lower, upper) or q in (lower, upper):
                        evidence.add("sbx-boundary-parent")
                    evidence.add(
                        f"sbx-{'upper' if sign > 0 else 'lower'}-{'first' if probability <= 0.5 else 'second'}"
                    )
            # Same NumPy scalar-return typing limitation as the mutation oracle.
            projected = cast(np.float64, np.clip(raw, lower, upper))
            assert crossing.result[name] == pytest.approx(
                projected, rel=STRICT_RTOL, abs=STRICT_ATOL
            ), (key, name, p, q, cache, raw)
            evidence.add("unclipped" if lower <= raw <= upper else "clipped")
            if p != q and lower < raw < upper:
                evidence.add("distinct-interior-parents")
    return evidence


def test_no_crossover_inherits_either_observed_parent(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    algorithm = run_ga(
        ObservedGeneticAlgorithm(
            population_method="generational",
            crossover_probability=0,
            mutation_probability=0,
            survival="replacement",
            selection="ranking",
        ),
        small_bounds,
        n_iterations=2,
        seed=seed_for(0),
    )
    assert {"copy-parent-0", "copy-parent-1"} <= assert_crossovers(
        algorithm, small_bounds
    )
    assert assert_mutation(algorithm, small_bounds) == {"skipped"}
    _ = assert_selection(algorithm)
    _ = assert_survival_and_synchrony(algorithm)


@pytest.mark.parametrize(
    "alpha", [0.3, 1.0, 3.0], ids=["interpolate", "endpoint", "tyrannis-extrapolate"]
)
def test_arithmetic_crossover_and_projection(
    small_bounds: dict[str, tuple[float, float]],
    alpha: float,
) -> None:
    algorithm = run_ga(
        ObservedGeneticAlgorithm(
            population_method="generational",
            crossover="arithmetic",
            crossover_probability=1,
            arithmetic_alpha=alpha,
            mutation_probability=0,
            survival="replacement",
            selection="ranking",
        ),
        small_bounds,
        n_iterations=2,
    )
    evidence = assert_crossovers(algorithm, small_bounds)
    assert {"unclipped", "distinct-interior-parents"} <= evidence
    assert ("clipped" in evidence) == (alpha > 1)
    assert assert_mutation(algorithm, small_bounds) == {"skipped"}
    _ = assert_selection(algorithm)
    _ = assert_survival_and_synchrony(algorithm)


@pytest.mark.parametrize(
    "alpha", [0.0, 2.0], ids=["parental-interval", "extended-interval"]
)
def test_blx_alpha_uniform_interval_and_projection(
    small_bounds: dict[str, tuple[float, float]],
    alpha: float,
) -> None:
    algorithm = run_ga(
        ObservedGeneticAlgorithm(
            population_method="generational",
            crossover="blx",
            blx_alpha=alpha,
            crossover_probability=1,
            mutation_probability=0,
            survival="replacement",
            selection="ranking",
        ),
        small_bounds,
        n_iterations=3,
    )
    evidence = assert_crossovers(algorithm, small_bounds)
    assert {"parental-interior", "unclipped"} <= evidence
    if alpha:
        assert {"parental-extrapolation", "clipped"} <= evidence
    else:
        assert "clipped" not in evidence and "parental-extrapolation" not in evidence
    assert assert_mutation(algorithm, small_bounds) == {"skipped"}
    _ = assert_selection(algorithm)
    _ = assert_survival_and_synchrony(algorithm)


@pytest.mark.parametrize(
    "eta", [0.0, 3.0, 20.0], ids=["eta-zero", "eta-three", "eta-default"]
)
def test_bounded_sbx_both_children_and_inverse_cdf_branches(
    small_bounds: dict[str, tuple[float, float]],
    eta: float,
) -> None:
    algorithm = run_ga(
        ObservedGeneticAlgorithm(
            population_method="generational",
            crossover="sbx",
            sbx_eta=eta,
            crossover_probability=1,
            mutation_probability=0,
            survival="replacement",
            selection="ranking",
        ),
        small_bounds,
        n_iterations=4,
    )
    evidence = assert_crossovers(algorithm, small_bounds)
    assert {
        "sbx-lower-first",
        "sbx-lower-second",
        "sbx-upper-first",
        "sbx-upper-second",
        "equal-parents",
        "distinct-interior-parents",
    } <= evidence
    assert "clipped" not in evidence
    assert assert_mutation(algorithm, small_bounds) == {"skipped"}
    _ = assert_selection(algorithm)
    _ = assert_survival_and_synchrony(algorithm)


def test_sbx_numerically_indistinguishable_parents_use_midpoint() -> None:
    # A four-ULP domain around nonzero values makes distinct drawn parents
    # numerically indistinguishable. No RNG or population is replaced.
    upper = 1.0 + 4 * ulp(1.0)
    bounds = {"x": (1.0, upper), "y": (1.0, upper)}
    algorithm = run_ga(
        ObservedGeneticAlgorithm(
            population_method="generational",
            crossover_probability=1,
            mutation_probability=0,
            selection="ranking",
        ),
        bounds,
        n_iterations=1,
    )
    distinct = False
    for crossing in algorithm.crossings.values():
        p, q = crossing.parents
        for name in bounds:
            distinct |= p.variables[name] != q.variables[name]
            assert (
                crossing.result[name]
                == 0.5 * p.variables[name] + 0.5 * q.variables[name]
            )
    assert distinct
    assert assert_mutation(algorithm, bounds) == {"skipped"}
    _ = assert_survival_and_synchrony(algorithm)


@pytest.mark.parametrize(
    "probability", [None, 0.0, 1.0], ids=["default", "never", "always"]
)
def test_gaussian_mutation_probability_range_scale_and_clipping(
    small_bounds: dict[str, tuple[float, float]],
    probability: float | None,
) -> None:
    algorithm = run_ga(
        ObservedGeneticAlgorithm(
            population_method="generational",
            crossover_probability=0,
            mutation="gaussian",
            mutation_probability=probability,
            mutation_intensity=1.7,
            gaussian_sigma=0.4,
            survival="replacement",
        ),
        small_bounds,
    )
    evidence = assert_mutation(algorithm, small_bounds)
    if probability == 0:
        assert evidence == {"skipped"}
    else:
        assert {"applied", "clipped", "unclipped", "nonzero-interior"} <= evidence
        assert ("skipped" in evidence) == (probability is None)
    _ = assert_crossovers(algorithm, small_bounds)
    _ = assert_selection(algorithm)
    _ = assert_survival_and_synchrony(algorithm)


@pytest.mark.parametrize(
    ("eta", "intensity"),
    [(0.0, 1.0), (4.0, 1.0), (0.0, 4.0), (4.0, 0.0)],
    ids=["eta-zero", "eta-four", "intensity-extension", "no-displacement"],
)
def test_polynomial_mutation_bounded_half_distributions(
    small_bounds: dict[str, tuple[float, float]],
    eta: float,
    intensity: float,
) -> None:
    algorithm = run_ga(
        ObservedGeneticAlgorithm(
            population_method="generational",
            crossover_probability=0,
            mutation="polynomial",
            mutation_probability=1,
            polynomial_eta=eta,
            mutation_intensity=intensity,
            survival="replacement",
            selection="ranking",
        ),
        small_bounds,
    )
    evidence = assert_mutation(algorithm, small_bounds)
    assert {
        "lower",
        "upper",
        "distance-lower",
        "distance-upper",
        "unclipped",
    } <= evidence
    assert ("clipped" in evidence) == (intensity > 1)
    if intensity > 1:
        assert "boundary-input" in evidence
    if intensity == 0:
        assert all(before == after for before, after in algorithm.mutations.values())
    else:
        assert "nonzero-interior" in evidence
    _ = assert_crossovers(algorithm, small_bounds)
    _ = assert_selection(algorithm)
    _ = assert_survival_and_synchrony(algorithm)


@pytest.mark.parametrize("elite_count", [1, 99], ids=["one-elite", "capped-elites"])
def test_generational_elitism_rejects_changed_elite_candidates(
    small_bounds: dict[str, tuple[float, float]],
    elite_count: int,
) -> None:
    algorithm = run_ga(
        ObservedGeneticAlgorithm(
            population_method="generational",
            elite_count=elite_count,
            mutation="gaussian",
            mutation_probability=1,
            gaussian_sigma=0.5,
        ),
        small_bounds,
    )
    assert "elite-candidate-discarded" in assert_survival_and_synchrony(algorithm)
    assert all(
        len(elites) == min(elite_count, 4) for elites in algorithm.elites.values()
    )
    assert all(
        len(elites) < len(algorithm.initial) for elites in algorithm.elites.values()
    )
    _ = assert_selection(algorithm)
    _ = assert_crossovers(algorithm, small_bounds)
    _ = assert_mutation(algorithm, small_bounds)


def test_replacement_accepts_worsening_candidates(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    algorithm = run_ga(
        ObservedGeneticAlgorithm(
            population_method="generational",
            survival="replacement",
            mutation="gaussian",
            mutation_probability=1,
            gaussian_sigma=0.5,
        ),
        small_bounds,
    )
    assert {
        "worse-accepted",
        "earlier-candidate-not-parent",
    } <= assert_survival_and_synchrony(algorithm)
    assert all(not elites for elites in algorithm.elites.values())
    _ = assert_selection(algorithm)
    assert "sbx-boundary-parent" in assert_crossovers(algorithm, small_bounds)
    _ = assert_mutation(algorithm, small_bounds)


def assert_steady_state(algorithm: ObservedGeneticAlgorithm) -> set[str]:
    """Queue conservation, coverage and cardinality on real scheduling snapshots.

    No permutations or random draws are manufactured. Within a cycle, consumed
    IDs and queued IDs partition its eligible membership. A boundary consumes
    the old remainder plus the needed prefix of a fresh permutation, excluding
    IDs already updated in that SAME iteration.
    """
    evidence: set[str] = set()
    identifiers = set(algorithm.initial)
    for iteration in range(1, len(algorithm.states)):
        elites = algorithm.elites[iteration]
        eligible = identifiers - elites
        count = min(
            max(1, ceil(algorithm.steady_state_fraction * len(identifiers))),
            len(eligible),
        )
        old, current = algorithm.cycles[iteration - 1], algorithm.cycles[iteration]
        synchronized = algorithm.synchronized[iteration]
        assert len(current.active) == count
        assert current.active <= eligible
        assert len(current.queue) == len(set(current.queue))
        assert set(current.queue) <= eligible
        # Eligibility changes can remove queued elites and insert former elites.
        newcomers = eligible - old.members
        remaining = set(old.queue) & eligible
        assert len(synchronized.queue) == len(set(synchronized.queue))
        assert set(synchronized.queue) == remaining | newcomers
        if not newcomers:
            assert synchronized.queue == [key for key in old.queue if key in eligible]
        if iteration > 1 and algorithm.elites[iteration - 1] != elites:
            evidence.add("changing-elites")
        starts = algorithm.started.get(iteration, [])
        if len(synchronized.queue) >= count:
            assert not starts
            assert current.active == set(synchronized.queue[:count])
            assert current.queue == synchronized.queue[count:]
            evidence.add("within-cycle")
        else:
            assert len(starts) == 1
            fresh = starts[0]
            assert fresh.members == eligible
            assert len(fresh.queue) == len(eligible) and set(fresh.queue) == eligible
            old_remainder = set(synchronized.queue)
            assert old_remainder <= current.active
            needed = count - len(old_remainder)
            available = [key for key in fresh.queue if key not in old_remainder]
            assert current.active - old_remainder == set(available[:needed])
            assert current.queue == [
                key for key in fresh.queue if key not in set(available[:needed])
            ]
            evidence.add("new-cycle")
            if old_remainder:
                evidence.add("spillover")
                assert len(current.active) == len(old_remainder) + needed
        assert current.members == (eligible if current.queue else set())
        # Only active particles receive operators or random values. Inactive
        # candidates preserve BOTH position and evaluated fitness exactly.
        actual_crossings = {key for t, key in algorithm.crossings if t == iteration}
        actual_mutations = {key for t, key in algorithm.mutations if t == iteration}
        assert actual_crossings == actual_mutations == current.active
        assert {
            s.target.identifier
            for s in algorithm.selections
            if s.iteration == iteration
        } == current.active
        for key in identifiers:
            candidate = algorithm.candidates[iteration, key]
            if key not in current.active:
                assert candidate.random_cache == {}
                assert (
                    candidate.candidate_variables
                    == algorithm.states[iteration - 1][key].variables
                )
                assert (
                    candidate.candidate_fitness
                    == algorithm.states[iteration - 1][key].fitness
                )
            else:
                assert candidate.random_cache
    return evidence


@pytest.mark.parametrize(
    ("fraction", "survival", "elite_count"),
    [
        (0.0, "replacement", 1),
        (0.3, "replacement", 1),
        (1.0, "replacement", 1),
        (0.5, "elitism", 1),
        (1.0, "elitism", 99),
    ],
    ids=[
        "zero-still-one",
        "ceil-and-spillover",
        "whole-population",
        "changing-elites",
        "eligible-cap",
    ],
)
def test_steady_state_cycles_spillover_and_eligible_count(
    small_bounds: dict[str, tuple[float, float]],
    fraction: float,
    survival: str,
    elite_count: int,
) -> None:
    algorithm = run_ga(
        ObservedGeneticAlgorithm(
            population_method="steady-state",
            steady_state_fraction=fraction,
            survival=survival,
            elite_count=elite_count,
            crossover="arithmetic",
            crossover_probability=1,
            mutation="gaussian",
            mutation_probability=1,
            gaussian_sigma=0.15,
        ),
        small_bounds,
        n_iterations=6,
        seed=seed_for(0),
    )
    evidence = assert_steady_state(algorithm)
    synchrony = assert_survival_and_synchrony(algorithm)
    if fraction in (0.3, 0.5):
        assert "earlier-candidate-not-parent" in synchrony
    _ = assert_selection(algorithm)
    _ = assert_crossovers(algorithm, small_bounds)
    _ = assert_mutation(algorithm, small_bounds)
    if fraction == 0.3:
        assert {"within-cycle", "spillover", "new-cycle"} <= evidence
    if survival == "elitism" and elite_count == 1:
        assert "changing-elites" in evidence
    if survival == "replacement":
        # With fixed eligibility, each complete cycle covers all IDs exactly once.
        if fraction == 0:
            assert {
                key for t in range(1, 6) for key in algorithm.cycles[t].active
            } == set(algorithm.initial)
        if fraction == 1:
            assert all(
                algorithm.cycles[t].active == set(algorithm.initial)
                for t in range(1, 7)
            )


def test_discovered_negative_infinity_remains_best_after_population_loses_it(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    def objective(x: float, y: float) -> float:
        return -np.inf if -0.4 < x < 0.4 else np.inf if x > 1 else sphere(x, y)

    algorithm = run_ga(
        ObservedGeneticAlgorithm(
            population_method="generational",
            survival="replacement",
            selection="ranking",
            mutation="gaussian",
            mutation_probability=1,
            gaussian_sigma=0.4,
        ),
        small_bounds,
        objective=objective,
        n_iterations=4,
        seed=seed_for(0),
    )
    assert isfinite(algorithm.best[0].fitness)
    discovery = min(t for t, best in algorithm.best.items() if best.fitness == -np.inf)
    assert discovery > 0
    assert any(
        t > discovery and all(p.fitness > -np.inf for p in population.values())
        for t, population in algorithm.states.items()
    )
    for t in range(discovery, len(algorithm.states)):
        assert algorithm.best[t]() == algorithm.best[discovery]()
        assert objective(**algorithm.best[t].variables) == -np.inf
    _ = assert_selection(algorithm)
    _ = assert_survival_and_synchrony(algorithm)
    _ = assert_crossovers(algorithm, small_bounds)
    _ = assert_mutation(algorithm, small_bounds)
