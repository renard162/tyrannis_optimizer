"""ABC equations and individual evaluations through the real Serial replica.

Karaboga & Basturk (2007), doi:10.1007/s10898-007-9149-x, section 2,
eqs. (2.1)-(2.2): classical probabilities, single-coordinate neighbours,
greedy selection, equally many employed/onlooker bees, one scout.
Full text: https://sci2s.ugr.es/sites/default/files/files/Teaching/
GraduatesCourses/Metaheuristicas/Bibliography/ABC-algorithm-numerical-function-2007.pdf
Page 463 inconsistently describes ties as acceptable in one paragraph and
strict improvement in another. Tyrannis explicitly requires strict improvement;
the tie tests protect that declared choice, not a silently reconciled quotation.
Karaboga & Basturk (2008), doi:10.1016/j.asoc.2007.05.007, consulted via
the authors' Erciyes publication record (abstract); full text was unavailable:
https://avesis.erciyes.edu.tr/yayin/822c3903-ba26-40a4-aaa5-02a20d96cf7f/on-the-performance-of-artificial-bee-colony-abc-algorithm
Ozturk, Karaboga & Gorkemli (2011), doi:10.3390/s110606056, eqs. (7)-(8):
https://pmc.ncbi.nlm.nih.gov/articles/PMC3231427/ . Their best-relative
probability is a later formulation, applied here to transformed ABC fitness.

The task and current Tyrannis contracts specify the two-branch nectar map,
infinity fallbacks, clipping, synchronous shared partners, sequential local
onlooker acceptance, strict trial > limit, default limit=N*D, scout replacement
at the NEXT pre_iteration, and max_scouts as an explicit extension. These
scheduling/extension choices are not attributed to the original paper.
Initialization, scout, employed and onlooker evaluations enter through
ParticleBase.update. No RNG, candidate, counter or selection is replaced.
"""

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from itertools import pairwise
from math import trunc
from typing import cast, override

import numpy as np
import pytest

from tests._support import factories
from tests._support.assertions import (
    assert_optimizer_result_consistent,
    assert_particle_state_consistent,
)
from tests._support.numerics import BASE_SEED, STRICT_ATOL, STRICT_RTOL, seed_for
from tests._support.objectives import constant_objective, sphere
from tyrannis import Optimizer
from tyrannis.algorithm.bee_colony import ABCParticle, BeeColony
from tyrannis.processor.serial import Serial
from tyrannis.space.continuous import Continuous


@dataclass
class Stage:
    iteration: int
    name: str
    identifier: str
    population: dict[str, ABCParticle]
    best: ABCParticle | None
    selected: list[str]
    probabilities: dict[str, float]
    evaluations: int
    double_check: bool


@dataclass
class Attempt:
    iteration: int
    identifier: str
    particle: ABCParticle
    base: dict[str, float]
    variable: str
    partner: ABCParticle
    phi: float
    candidate: dict[str, float]
    evaluation_index: int


@dataclass
class Evaluation:
    variables: dict[str, float]
    fitness: float
    update_index: int | None


@dataclass
class Update:
    before: ABCParticle
    after: ABCParticle
    variables: dict[str, float]
    start: int
    end: int


class ObservedBeeColony(BeeColony):
    """Snapshots at phase boundaries; all behaviour delegates to super()."""

    count_evaluations: Callable[[], int]

    def mark(self, name: str, identifier: str = "") -> None:
        self.stages.append(
            Stage(
                self.actual_iter,
                name,
                identifier,
                deepcopy(self.population),
                deepcopy(self.local_best),
                self.double_check_ids.copy(),
                self._onlooker_probabilities.copy(),
                self.count_evaluations(),
                self.double_particle_check,
            )
        )

    @override
    def pre_iteration(self, actual_iter: int) -> None:
        self.actual_iter: int = actual_iter
        if actual_iter == 0:
            self.stages: list[Stage] = []
            self.attempts: list[Attempt] = []
        self.mark("pre-before")
        super().pre_iteration(actual_iter)
        self.mark("pre-after")

    @override
    def create_random_cache(self, particle_ids: list[str], initialize: bool) -> None:
        super().create_random_cache(particle_ids, initialize)
        phase = (
            "initial"
            if initialize
            else "onlooker"
            if self._onlooker_phase
            else "employed"
        )
        self.mark(f"{phase}-cache")

    @override
    def initialize_particle(self, identifier: str) -> ABCParticle:
        self.mark("initial-before", identifier)
        particle = super().initialize_particle(identifier)
        self.mark("initial-after", identifier)
        return particle

    @override
    def update_particle(self, identifier: str) -> ABCParticle:
        self.active_identifier: str = identifier
        self.mark("employed-before", identifier)
        particle = super().update_particle(identifier)
        self.mark("employed-after", identifier)
        return particle

    @override
    def _generate_candidate(
        self,
        variables: Mapping[str, float],
        variable: str,
        partner: ABCParticle,
        phi: float,
    ) -> dict[str, float]:
        particle = deepcopy(self.population[self.active_identifier])
        base, partner_before = dict(variables), deepcopy(partner)
        candidate = super()._generate_candidate(variables, variable, partner, phi)
        self.attempts.append(
            Attempt(
                self.actual_iter,
                self.active_identifier,
                particle,
                base,
                variable,
                partner_before,
                phi,
                candidate.copy(),
                self.count_evaluations(),
            )
        )
        return candidate

    @override
    def inter_iteration(self, actual_iter: int) -> None:
        self.mark("inter-before")
        super().inter_iteration(actual_iter)
        self.mark("inter-after")

    @override
    def second_update_particle(self, identifier: str) -> ABCParticle:
        self.active_identifier = identifier
        self.mark("onlooker-before", identifier)
        particle = super().second_update_particle(identifier)
        self.mark("onlooker-after", identifier)
        return particle

    @override
    def post_iteration(self, actual_iter: int) -> None:
        self.mark("post-before")
        super().post_iteration(actual_iter)
        self.mark("post-after")


@dataclass
class Run:
    algorithm: ObservedBeeColony
    calls: list[Evaluation]
    updates: list[Update]

    def stage(self, iteration: int, name: str, identifier: str = "") -> Stage:
        matches = [
            s
            for s in self.algorithm.stages
            if (s.iteration, s.name, s.identifier) == (iteration, name, identifier)
        ]
        assert len(matches) == 1
        return matches[0]


def run_bees(
    monkeypatch: pytest.MonkeyPatch,
    bounds: dict[str, tuple[float, float]],
    *,
    objective: Callable[..., float] = sphere,
    seed: int = BASE_SEED,
    n_particles: int = 3,
    n_iterations: int = 3,
    limit: int | None = 100,
    max_scouts: int | None = 1,
    improved: bool = False,
) -> Run:
    # Functions retain their closures under deepcopy. Both logs therefore
    # observe the replica's actual evaluations, not a copied callable's counter.
    calls: list[Evaluation] = []
    updates: list[Update] = []
    active_update: int | None = None

    def recorded_objective(**variables: float) -> float:
        value = objective(**variables)
        calls.append(Evaluation(variables.copy(), value, active_update))
        return value

    real_update = ABCParticle.update

    def observed_update(
        particle: ABCParticle,
        variables: dict[str, float],
        fitness_function: Callable[[dict[str, float]], np.float64],
    ) -> None:
        nonlocal active_update
        assert active_update is None
        active_update = len(updates)
        before, start = deepcopy(particle), len(calls)
        real_update(particle, variables, fitness_function)
        updates.append(
            Update(before, deepcopy(particle), variables.copy(), start, len(calls))
        )
        active_update = None

    template = ObservedBeeColony(limit, max_scouts, improved)
    template.count_evaluations = lambda: len(calls)
    processor = Serial()
    with monkeypatch.context() as patch:
        patch.setattr(ABCParticle, "update", observed_update)
        # The shared factory annotates AlgorithmBase without its type parameter.
        # Its public return type is precise; specialize only that erased boundary.
        factory = cast(Callable[..., Optimizer], factories.make_optimizer)
        optimizer = factory(
            Continuous(bounds, cost_function=recorded_objective, use_cache=False),
            template,
            processor=processor,
            n_particles=n_particles,
            n_iterations=n_iterations,
            seed=seed,
            history="iteration",
            fitness_failure_strategy="raise",
        ).fit()
    (executor,) = processor.processors_pool.values()
    # Public pool identifies the executing replica; no public algorithm accessor.
    # ProcessorBase erases the particle generic; observe its stored algorithm
    # as object and narrow it at runtime rather than propagating Unknown.
    algorithm = cast(object, vars(executor)["_algorithm"])
    assert isinstance(algorithm, ObservedBeeColony)
    assert algorithm is not template and template.population == {}
    assert_optimizer_result_consistent(optimizer)
    assert algorithm.local_best is not None
    assert optimizer.best_fitness == algorithm.local_best.fitness
    run = Run(algorithm, calls, updates)
    assert_evaluation_accounting(run, n_particles, n_iterations)
    return run


def assert_update_bijection(run: Run) -> None:
    indexed_calls = [call for call in run.calls if call.update_index is not None]
    assert len(indexed_calls) == len(run.updates)
    assert {call.update_index for call in indexed_calls} == set(range(len(run.updates)))
    for index, update in enumerate(run.updates):
        assert update.end == update.start + 1
        call = run.calls[update.start]
        assert call.update_index == index
        assert call.variables == update.variables == update.after.candidate_variables
        assert call.fitness == update.after.candidate_fitness
        assert update.before.variables == update.after.variables
        assert update.before.fitness == update.after.fitness


def assert_evaluation_accounting(run: Run, sources: int, iterations: int) -> None:
    """Exact temporal path, per-source budget and no hidden evaluations."""
    assert_update_bijection(run)
    assert {s.iteration for s in run.algorithm.stages} == set(range(iterations + 1))
    scout_total = 0
    for t in range(iterations + 1):
        pre = run.stage(t, "pre-after")
        new_ids = [key for key, p in pre.population.items() if p.new_particle]
        scout_total += len(new_ids) if t else 0
        order = [("pre-before", ""), ("pre-after", "")]
        if new_ids:
            order += [("initial-cache", "")]
            order += [
                (name, key)
                for key in new_ids
                for name in ("initial-before", "initial-after")
            ]
        if t:
            employed = run.stage(t, "employed-cache")
            inter = run.stage(t, "inter-after")
            order += [("employed-cache", "")]
            order += [
                (name, key)
                for key in employed.population
                for name in ("employed-before", "employed-after")
            ]
            order += [("inter-before", ""), ("inter-after", ""), ("onlooker-cache", "")]
            order += [
                (name, key)
                for key in inter.selected
                for name in ("onlooker-before", "onlooker-after")
            ]
        order += [("post-before", ""), ("post-after", "")]
        stages = [s for s in run.algorithm.stages if s.iteration == t]
        assert [(s.name, s.identifier) for s in stages] == order
        assert stages[-1].evaluations - stages[0].evaluations == (
            2 * sources + len(new_ids) if t else sources
        )
        for before, after in pairwise(stages):
            expected = 0
            if before.name in ("initial-before", "employed-before"):
                expected = 1
            elif before.name == "onlooker-before":
                expected = before.population[before.identifier].onlooker_count
            assert after.evaluations - before.evaluations == expected
            if expected:
                phase_calls = run.calls[before.evaluations : after.evaluations]
                assert all(call.update_index is not None for call in phase_calls)
                assert [
                    run.updates[cast(int, call.update_index)].before.identifier
                    for call in phase_calls
                ] == [before.identifier] * expected
                if before.name == "onlooker-before":
                    assert [
                        attempt.identifier
                        for attempt in run.algorithm.attempts
                        if before.evaluations
                        <= attempt.evaluation_index
                        < after.evaluations
                    ] == [before.identifier] * expected
        if t:
            for key, particle in run.stage(t, "inter-after").population.items():
                attempts_for_source = sum(
                    attempt.identifier == key
                    and stages[0].evaluations
                    <= attempt.evaluation_index
                    < stages[-1].evaluations
                    for attempt in run.algorithm.attempts
                )
                assert attempts_for_source == 1 + particle.onlooker_count
    assert len(run.calls) == sources + 2 * sources * iterations + scout_total


def nectar(fitness: float) -> float:
    """The task's minimization-to-nectar map, including signed infinities."""
    return 1 / (1 + fitness) if fitness >= 0 else 1 + abs(fitness)


def assert_probabilities(run: Run, improved: bool) -> None:
    # Inspect evaluated +inf as well as consolidated +inf; new_particle, not
    # isinf, distinguishes unevaluated sources. -inf is always a valid optimum.
    for update in run.updates:
        evaluated = update.after
        assert evaluated.candidate_fitness is not None
        assert evaluated.abc_candidate_fitness == nectar(
            float(evaluated.candidate_fitness)
        )
    for stage in run.algorithm.stages:
        if stage.name not in ("post-after", "inter-after"):
            continue
        fitnesses = {key: float(p.fitness) for key, p in stage.population.items()}
        fits = {key: nectar(f) for key, f in fitnesses.items()}
        for key, particle in stage.population.items():
            assert particle.abc_fitness == fits[key]
            assert not np.isnan(particle.abc_fitness)
            if np.isposinf(particle.fitness):
                assert particle.abc_fitness == 0
            if np.isneginf(particle.fitness):
                assert np.isposinf(particle.abc_fitness)
        if stage.name != "inter-after":
            continue
        infinite = [key for key, f in fitnesses.items() if np.isneginf(f)]
        if infinite:
            expected = {
                key: (1.0 if key in infinite else 0.1)
                if improved
                else (1 / len(infinite) if key in infinite else 0.0)
                for key in fits
            }
        elif not any(fits.values()):
            expected = dict.fromkeys(fits, 1.0 if improved else 1 / len(fits))
        else:
            expected = {
                key: 0.9 * value / max(fits.values()) + 0.1
                if improved
                else value / sum(fits.values())
                for key, value in fits.items()
            }
        assert stage.probabilities == pytest.approx(
            expected, rel=STRICT_RTOL, abs=STRICT_ATOL
        )
        assert all(np.isfinite(p) and 0 <= p <= 1 for p in stage.probabilities.values())
        if improved:
            assert min(stage.probabilities.values()) >= 0.1
            assert max(stage.probabilities.values()) == 1.0
        else:
            assert sum(stage.probabilities.values()) == pytest.approx(
                1, rel=STRICT_RTOL, abs=STRICT_ATOL
            )


@pytest.mark.parametrize("improved", [False, True], ids=["classical", "best-relative"])
def test_finite_nectar_and_probabilities_use_post_employed_fitness(
    monkeypatch: pytest.MonkeyPatch,
    small_bounds: dict[str, tuple[float, float]],
    improved: bool,
) -> None:
    def signed_steps(**variables: float) -> float:
        return float(trunc(variables["x"]))

    # Case 5 yields zero and distinct values on both finite branches, in both modes.
    run = run_bees(
        monkeypatch,
        small_bounds,
        objective=signed_steps,
        improved=improved,
        seed=seed_for(5),
    )
    assert_probabilities(run, improved)
    _ = assert_phase_mathematics(run, small_bounds, signed_steps)
    values = sorted(
        {
            float(p.fitness)
            for s in run.algorithm.stages
            if s.name in ("post-after", "inter-after")
            for p in s.population.values()
        }
    )
    assert 0 in values
    assert sum(f < 0 for f in values) >= 2 and sum(f > 0 for f in values) >= 2
    assert all(nectar(a) > nectar(b) for a, b in pairwise(values))
    changed_probability = False
    for t in range(1, 4):
        before = run.stage(t, "employed-cache").population
        inter = run.stage(t, "inter-after")
        old = {key: nectar(float(p.fitness)) for key, p in before.items()}
        old_probabilities = {
            key: 0.9 * f / max(old.values()) + 0.1
            if improved
            else f / sum(old.values())
            for key, f in old.items()
        }
        changed_probability |= any(
            abs(inter.probabilities[key] - p) > STRICT_ATOL
            for key, p in old_probabilities.items()
        )
        pairs = sorted(
            (p.fitness, inter.probabilities[key]) for key, p in inter.population.items()
        )
        assert all(a[1] >= b[1] for a, b in pairwise(pairs))
    assert changed_probability, (
        "The pre/post-employed distinction must affect probabilities"
    )


@pytest.mark.parametrize("improved", [False, True], ids=["classical", "best-relative"])
def test_finite_ties_preserve_sources_and_count_every_failed_attempt(
    monkeypatch: pytest.MonkeyPatch,
    small_bounds: dict[str, tuple[float, float]],
    improved: bool,
) -> None:
    run = run_bees(
        monkeypatch,
        small_bounds,
        objective=constant_objective,
        improved=improved,
        n_iterations=1,
    )
    assert_probabilities(run, improved)
    witnessed = assert_phase_mathematics(run, small_bounds, constant_objective)
    assert "employed-tie" in witnessed
    inter = run.stage(1, "inter-after")
    assert set(inter.probabilities.values()) == {1.0 if improved else 1 / 3}
    for key, p in run.stage(1, "post-after").population.items():
        assert p.variables == run.stage(0, "post-after").population[key].variables
        assert p.trial_count == 1 + p.onlooker_count
        if improved:
            assert p.onlooker_count == 1


def test_setup_evaluates_each_food_source_exactly_once(
    monkeypatch: pytest.MonkeyPatch, small_bounds: dict[str, tuple[float, float]]
) -> None:
    run = run_bees(monkeypatch, small_bounds, n_iterations=1)
    assert_update_bijection(run)
    before = run.stage(0, "pre-before")
    initial = run.stage(0, "initial-cache")
    setup = run.stage(0, "post-after")
    assert before.evaluations == initial.evaluations == 0
    assert setup.evaluations == 3
    assert run.stage(0, "pre-after").double_check
    assert before.best is run.stage(0, "post-before").best is None
    assert not setup.selected and not setup.probabilities
    assert not any(a.iteration == 0 for a in run.algorithm.attempts)
    assert [s.name for s in run.algorithm.stages if s.iteration == 0] == [
        "pre-before",
        "pre-after",
        "initial-cache",
        *[name for _ in range(3) for name in ("initial-before", "initial-after")],
        "post-before",
        "post-after",
    ]
    for index, (key, particle) in enumerate(setup.population.items()):
        assert_particle_state_consistent(particle, small_bounds)
        assert (
            run.calls[index].variables
            == before.population[key].variables
            == particle.variables
        )
        assert np.isfinite(particle.fitness)
        assert particle.fitness == pytest.approx(
            sphere(**particle.variables), rel=STRICT_RTOL, abs=STRICT_ATOL
        )
        assert particle.trial_count == particle.onlooker_count == 0
        assert not particle.new_particle
        assert particle.candidate_variables is particle.candidate_fitness is None
        assert initial.population[key].random_cache == particle.random_cache == {}
    assert setup.best is not None
    assert setup.best.fitness == min(p.fitness for p in setup.population.values())


def test_cost_function_evaluation_budget_is_exact(
    monkeypatch: pytest.MonkeyPatch, small_bounds: dict[str, tuple[float, float]]
) -> None:
    run = run_bees(monkeypatch, small_bounds)
    assert_update_bijection(run)
    assert len(run.calls) == 3 + 2 * 3 * 3
    assert sum(call.update_index is not None for call in run.calls) == len(run.calls)
    assert sum(call.update_index is None for call in run.calls) == 0
    for t in range(1, 4):
        pre = run.stage(t, "pre-before")
        employed = run.stage(t, "employed-cache")
        inter = run.stage(t, "inter-before")
        onlookers = run.stage(t, "onlooker-cache")
        post = run.stage(t, "post-before")
        assert pre.evaluations == employed.evaluations
        assert inter.evaluations - employed.evaluations == 3
        assert inter.evaluations == onlookers.evaluations
        assert post.evaluations - onlookers.evaluations == 3
        assert post.evaluations == run.stage(t, "post-after").evaluations


def assert_neighbour(
    run: Run,
    attempt: Attempt,
    base: dict[str, float],
    population: dict[str, ABCParticle],
    prefix: str,
    bounds: dict[str, tuple[float, float]],
    objective: Callable[..., float],
) -> set[str]:
    """Equation (2.2), using observed draws and an independently clipped scalar."""
    cache: dict[str, object] = attempt.particle.random_cache
    variable, partner_id, phi = (
        cache[f"{prefix}-variable"],
        cache[f"{prefix}-partner"],
        cache[f"{prefix}-phi"],
    )
    assert isinstance(variable, str) and isinstance(partner_id, str)
    assert isinstance(phi, float)
    assert variable in bounds and partner_id in population
    assert partner_id != attempt.identifier
    assert -1 <= phi <= 1
    assert (attempt.variable, attempt.partner.identifier, attempt.phi) == (
        variable,
        partner_id,
        phi,
    )
    assert attempt.base == base
    assert attempt.partner.variables == population[partner_id].variables
    raw = base[variable] + phi * (
        base[variable] - population[partner_id].variables[variable]
    )
    lower, upper = bounds[variable]
    expected = {**base, variable: min(upper, max(lower, raw))}
    assert attempt.candidate == pytest.approx(
        expected, rel=STRICT_RTOL, abs=STRICT_ATOL
    )
    assert {k: v for k, v in attempt.candidate.items() if k != variable} == {
        k: v for k, v in base.items() if k != variable
    }
    call = run.calls[attempt.evaluation_index]
    assert call.variables == attempt.candidate
    assert call.fitness == pytest.approx(
        objective(**expected), rel=STRICT_RTOL, abs=STRICT_ATOL
    )
    assert call.update_index is not None
    update = run.updates[call.update_index]
    assert update.start == attempt.evaluation_index
    assert update.before.identifier == attempt.identifier
    assert update.before.trial_count == attempt.particle.trial_count + 1
    assert update.before.random_cache == update.after.random_cache == cache
    return {"clipping" if raw < lower or raw > upper else "no-clipping"}


def assert_phase_mathematics(
    run: Run,
    bounds: dict[str, tuple[float, float]],
    objective: Callable[..., float] = sphere,
) -> set[str]:
    """Check observed transitions, never advance a second ABC implementation.

    Each onlooker's expected base is the argmin of its already evaluated prefix.
    Trailing failures are the distance from the most recent strict prefix minimum.
    """
    assert_update_bijection(run)
    witnessed: set[str] = set()
    iterations = sorted({s.iteration for s in run.algorithm.stages})
    for t in iterations[1:]:
        employed = run.stage(t, "employed-cache")
        inter = run.stage(t, "inter-after")
        cache = run.stage(t, "onlooker-cache")
        post = run.stage(t, "post-after")
        # Best is historical, and can change only at consolidation boundaries.
        assert (
            employed.best is not None
            and inter.best is not None
            and post.best is not None
        )
        pending_employed = run.stage(t, "inter-before").best
        pending_onlooker = run.stage(t, "post-before").best
        assert pending_employed is not None and pending_onlooker is not None
        assert (pending_employed.variables, pending_employed.fitness) == (
            employed.best.variables,
            employed.best.fitness,
        )
        assert (pending_onlooker.variables, pending_onlooker.fitness) == (
            inter.best.variables,
            inter.best.fitness,
        )
        if inter.best.fitness < employed.best.fitness:
            witnessed.add("employed-best-improvement")
        if post.best.fitness < inter.best.fitness:
            witnessed.add("onlooker-best-improvement")
        for key, original in employed.population.items():
            before = run.stage(t, "employed-before", key)
            after = run.stage(t, "employed-after", key)
            assert {k: p.variables for k, p in before.population.items()} == {
                k: p.variables for k, p in employed.population.items()
            }
            assert after.evaluations == before.evaluations + 1
            (attempt,) = [
                a
                for a in run.algorithm.attempts
                if a.evaluation_index == before.evaluations
            ]
            witnessed |= assert_neighbour(
                run,
                attempt,
                original.variables,
                employed.population,
                "employed",
                bounds,
                objective,
            )
            earlier_partner = before.population[attempt.partner.identifier]
            if (
                earlier_partner.candidate_fitness is not None
                and earlier_partner.candidate_fitness < earlier_partner.fitness
            ):
                witnessed.add("synchronous-employed-partner")
            candidate = after.population[key]
            assert (
                candidate.candidate_variables is not None
                and candidate.candidate_fitness is not None
            )
            improved = candidate.candidate_fitness < original.fitness
            witnessed.add("employed-improvement" if improved else "employed-failure")
            if candidate.candidate_fitness == original.fitness:
                witnessed.add("employed-tie")
            accepted = inter.population[key]
            assert accepted.variables == (
                candidate.candidate_variables if improved else original.variables
            )
            assert accepted.fitness == (
                candidate.candidate_fitness if improved else original.fitness
            )
            assert (
                accepted.trial_count
                == candidate.trial_count
                == (0 if improved else original.trial_count + 1)
            )
            assert accepted.candidate_variables is accepted.candidate_fitness is None
            assert candidate.random_cache == original.random_cache
            assert (
                candidate.variables == original.variables
                and candidate.fitness == original.fitness
            )

        counts = {key: p.onlooker_count for key, p in inter.population.items()}
        assert all(isinstance(q, int) and q >= 0 for q in counts.values())
        assert sum(counts.values()) == len(counts)
        assert inter.selected == [key for key, q in counts.items() if q > 0]
        assert [
            s.identifier
            for s in run.algorithm.stages
            if s.iteration == t and s.name == "onlooker-before"
        ] == inter.selected
        for key, original in inter.population.items():
            q = counts[key]
            assert original.second_update_required == (q > 0)
            if q == 0:
                witnessed.add("no-onlooker")
                assert cache.population[key].random_cache == {}
                assert post.population[key].variables == original.variables
                assert post.population[key].fitness == original.fitness
                assert post.population[key].trial_count == original.trial_count
                continue
            if q > 1:
                witnessed.add("multiple-onlookers")
            before = run.stage(t, "onlooker-before", key)
            after = run.stage(t, "onlooker-after", key)
            assert after.evaluations - before.evaluations == q
            attempts = [
                a
                for a in run.algorithm.attempts
                if before.evaluations <= a.evaluation_index < after.evaluations
            ]
            assert len(attempts) == q
            assert {k: p.variables for k, p in before.population.items()} == {
                k: p.variables for k, p in inter.population.items()
            }
            assert set(cache.population[key].random_cache) == {
                f"onlooker-{j}-{part}"
                for j in range(q)
                for part in ("variable", "partner", "phi")
            }
            evaluations = run.calls[before.evaluations : after.evaluations]
            baseline = Evaluation(original.variables, float(original.fitness), None)
            improvements = [
                j
                for j, call in enumerate(evaluations)
                if call.fitness < min(e.fitness for e in [baseline, *evaluations[:j]])
            ]
            for j, attempt in enumerate(attempts):
                accepted = min([baseline, *evaluations[:j]], key=lambda e: e.fitness)
                witnessed |= assert_neighbour(
                    run,
                    attempt,
                    accepted.variables,
                    inter.population,
                    f"onlooker-{j}",
                    bounds,
                    objective,
                )
                previous_improvements = [i for i in improvements if i < j]
                expected_trial = (
                    j - previous_improvements[-1] - 1
                    if previous_improvements
                    else original.trial_count + j
                )
                assert attempt.particle.trial_count == expected_trial
                assert attempt.particle.variables == original.variables
                assert attempt.particle.fitness == original.fitness
                assert attempt.particle.candidate_variables == accepted.variables
                assert attempt.particle.candidate_fitness == accepted.fitness
                assert (
                    attempt.particle.random_cache == cache.population[key].random_cache
                )
                witnessed.add(
                    "onlooker-improvement" if j in improvements else "onlooker-failure"
                )
                if j not in improvements and previous_improvements:
                    witnessed.add("failure-after-improvement")
                if j > 0 and accepted.variables != original.variables:
                    witnessed.add("sequential-new-base")
                # A previous source already has a better candidate, yet the partner
                # supplied to this attempt must still be its post-employed position.
                partner = before.population[attempt.partner.identifier]
                if (
                    partner.candidate_variables is not None
                    and partner.candidate_variables != partner.variables
                ):
                    witnessed.add("synchronous-onlooker-partner")
            best = min([baseline, *evaluations], key=lambda e: e.fitness)
            candidate = after.population[key]
            final = post.population[key]
            assert candidate.candidate_variables == final.variables == best.variables
            assert candidate.candidate_fitness == final.fitness == best.fitness
            trailing = (
                q - improvements[-1] - 1 if improvements else original.trial_count + q
            )
            assert candidate.trial_count == final.trial_count == trailing
            assert candidate.variables == original.variables
            assert candidate.random_cache == cache.population[key].random_cache

    for stage in run.algorithm.stages:
        if stage.name not in ("post-after", "inter-after"):
            continue
        assert stage.best is not None
        assert stage.best.fitness == min(
            c.fitness for c in run.calls[: stage.evaluations]
        )
        assert any(
            c.variables == stage.best.variables and c.fitness == stage.best.fitness
            for c in run.calls[: stage.evaluations]
        )
        for particle in stage.population.values():
            assert_particle_state_consistent(particle, bounds)
            assert particle.candidate_variables is particle.candidate_fitness is None
            assert not particle.new_particle
            assert not np.isnan(particle.fitness) and not np.isnan(particle.abc_fitness)
            assert all(np.isfinite(v) for v in particle.variables.values())
            assert particle.fitness == objective(**particle.variables)
    return witnessed


def test_employed_phase_matches_equation_clipping_and_strict_greedy(
    monkeypatch: pytest.MonkeyPatch, small_bounds: dict[str, tuple[float, float]]
) -> None:
    # Case 3 includes clipping and a previously improved partner in the same sweep.
    run = run_bees(monkeypatch, small_bounds, seed=seed_for(3))
    witnessed = assert_phase_mathematics(run, small_bounds)
    assert {
        "clipping",
        "no-clipping",
        "employed-improvement",
        "employed-failure",
        "synchronous-employed-partner",
        "employed-best-improvement",
    } <= witnessed


def test_sequential_onlookers_use_last_accepted_state_and_shared_partners(
    monkeypatch: pytest.MonkeyPatch, small_bounds: dict[str, tuple[float, float]]
) -> None:
    # Case 3 contains improvement -> failure within one source's onlooker chain.
    run = run_bees(monkeypatch, small_bounds, seed=seed_for(3))
    witnessed = assert_phase_mathematics(run, small_bounds)
    assert {
        "no-onlooker",
        "multiple-onlookers",
        "onlooker-improvement",
        "onlooker-failure",
        "failure-after-improvement",
        "sequential-new-base",
        "synchronous-onlooker-partner",
        "onlooker-best-improvement",
    } <= witnessed


def assert_scouts(
    run: Run, bounds: dict[str, tuple[float, float]], objective: Callable[..., float]
) -> set[str]:
    """Observe replacement timing and rank inequalities, without managing scouts."""
    witnessed: set[str] = set()
    total = 0
    for before in run.algorithm.stages:
        if before.name != "pre-before" or before.iteration == 0:
            continue
        t = before.iteration
        prior = run.stage(t - 1, "post-after")
        after = run.stage(t, "pre-after")
        employed = run.stage(t, "employed-cache")
        limit = run.algorithm.limit
        if limit is None:
            limit = len(before.population) * len(bounds)
        eligible = {
            key for key, p in before.population.items() if p.trial_count > limit
        }
        replaced = {key for key, p in after.population.items() if p.new_particle}
        assert set(before.population) == set(after.population)
        assert replaced <= eligible
        maximum = run.algorithm.max_scouts
        assert len(replaced) == (
            len(eligible) if maximum is None else min(maximum, len(eligible))
        )
        if len(eligible) > 1:
            witnessed.add("multiple-eligible")
        if len(replaced) > 1:
            witnessed.add("multiple-scouts")
        if replaced and eligible - replaced:
            chosen_counts = [before.population[k].trial_count for k in replaced]
            other_counts = [
                before.population[k].trial_count for k in eligible - replaced
            ]
            assert min(chosen_counts) >= max(other_counts)
            if max(chosen_counts) > min(other_counts):
                witnessed.add("strict-priority")
        for key, old in before.population.items():
            assert old.variables == prior.population[key].variables
            assert old.fitness == prior.population[key].fitness
            assert old.trial_count == prior.population[key].trial_count
            assert not old.new_particle
            assert after.population[key].onlooker_count == 0
            if old.trial_count == limit:
                assert key not in replaced
                witnessed.add("exact-limit-kept")
            if old.trial_count == limit + 1 and key in replaced:
                witnessed.add("limit-plus-one-replaced")
                if (
                    t > 1
                    and run.stage(t - 1, "pre-before").population[key].trial_count
                    == limit
                ):
                    assert (
                        not run.stage(t - 1, "pre-after").population[key].new_particle
                    )
                    witnessed.add("exact-limit-then-plus-one")
            if key not in replaced:
                assert after.population[key].variables == old.variables
                assert after.population[key].fitness == old.fitness
                assert after.population[key].trial_count == old.trial_count
                continue
            witnessed.add("scout")
            new = after.population[key]
            assert new.identifier == old.identifier
            assert new.variables != old.variables
            assert_particle_state_consistent(new, bounds)
            assert new.trial_count == new.onlooker_count == 0
            assert np.isposinf(new.fitness)
            assert new.candidate_variables is new.candidate_fitness is None
            initialized = run.stage(t, "initial-after", key)
            call = run.calls[initialized.evaluations - 1]
            assert call.variables == new.variables
            assert call.fitness == objective(**new.variables)
            assert initialized.population[key].new_particle
            assert initialized.population[key].random_cache == {}
            ready = employed.population[key]
            assert not ready.new_particle
            assert ready.variables == new.variables and ready.fitness == call.fitness
            assert ready.trial_count == ready.onlooker_count == 0
            assert ready.candidate_variables is ready.candidate_fitness is None
            assert before.best is not None and after.best is not None
            assert before.best.variables == after.best.variables
            assert before.best.fitness == after.best.fitness
            if (
                old.variables == before.best.variables
                and call.fitness > before.best.fitness
            ):
                witnessed.add("best-source-replaced-by-worse-scout")
        total += len(replaced)
    assert total > 0
    return witnessed


@pytest.mark.parametrize(
    "limit", [None, 1], ids=["default-N-times-D", "strict-threshold"]
)
def test_scout_threshold_is_strict_and_replacement_waits_for_next_cycle(
    monkeypatch: pytest.MonkeyPatch,
    small_bounds: dict[str, tuple[float, float]],
    limit: int | None,
) -> None:
    improved = limit is None
    run = run_bees(
        monkeypatch,
        small_bounds,
        objective=constant_objective,
        limit=limit,
        max_scouts=None,
        improved=improved,
        n_iterations=5 if improved else 4,
        # Case 1 retains one source at limit, then replaces it at limit+1.
        seed=seed_for(1),
    )
    _ = assert_phase_mathematics(run, small_bounds, constant_objective)
    witnessed = assert_scouts(run, small_bounds, constant_objective)
    assert {"scout", "exact-limit-kept"} <= witnessed
    if limit is None:
        # N=3, D=2, two deterministic failures per cycle: 6 is retained at
        # pre(4); 8 triggers replacement only at pre(5), adding exactly N calls.
        assert all(
            p.trial_count == 6 for p in run.stage(4, "pre-before").population.values()
        )
        assert not any(
            p.new_particle for p in run.stage(4, "pre-after").population.values()
        )
        assert all(
            p.new_particle for p in run.stage(5, "pre-after").population.values()
        )
        assert len(run.calls) == 3 + 2 * 3 * 5 + 3
    else:
        assert {"limit-plus-one-replaced", "exact-limit-then-plus-one"} <= witnessed


@pytest.mark.parametrize(
    "max_scouts", [1, None, 10], ids=["classical-one", "all", "cap-above-population"]
)
def test_max_scouts_limits_replacement_and_prioritizes_largest_trials(
    monkeypatch: pytest.MonkeyPatch,
    small_bounds: dict[str, tuple[float, float]],
    max_scouts: int | None,
) -> None:
    run = run_bees(
        monkeypatch,
        small_bounds,
        objective=constant_objective,
        limit=1,
        max_scouts=max_scouts,
        n_iterations=3,
        seed=seed_for(3),
    )
    _ = assert_phase_mathematics(run, small_bounds, constant_objective)
    witnessed = assert_scouts(run, small_bounds, constant_objective)
    assert "multiple-eligible" in witnessed
    assert ("strict-priority" if max_scouts == 1 else "multiple-scouts") in witnessed


def signed_extremes(**variables: float) -> float:
    """Three ordered regions; finite coordinates even for infinite objectives."""
    x = variables["x"]
    return -np.inf if x < -1 else np.inf if x > 1 else sphere(**variables)


@pytest.mark.parametrize(
    ("improved", "case_id"),
    [(False, 177), (True, 85)],
    ids=["classical", "best-relative"],
)
def test_signed_infinities_preserve_order_greedy_trials_and_probabilities(
    monkeypatch: pytest.MonkeyPatch,
    small_bounds: dict[str, tuple[float, float]],
    improved: bool,
    case_id: int,
) -> None:
    # These witnesses contain all six extreme comparisons in three cycles,
    # including a finite/+inf population before any -inf is discovered.
    run = run_bees(
        monkeypatch,
        small_bounds,
        objective=signed_extremes,
        seed=seed_for(case_id),
        improved=improved,
    )
    assert_probabilities(run, improved)
    _ = assert_phase_mathematics(run, small_bounds, signed_extremes)
    transitions: set[tuple[str, str]] = set()

    def kind(fitness: float) -> str:
        return (
            "negative-infinity"
            if np.isneginf(fitness)
            else "positive-infinity"
            if np.isposinf(fitness)
            else "finite"
        )

    for attempt in run.algorithm.attempts:
        particle = attempt.particle
        current = (
            particle.fitness
            if "employed-variable" in particle.random_cache
            else particle.candidate_fitness
        )
        assert current is not None
        candidate = run.calls[attempt.evaluation_index].fitness
        transitions.add((kind(float(current)), kind(candidate)))
    assert {
        ("finite", "negative-infinity"),
        ("positive-infinity", "finite"),
        ("negative-infinity", "finite"),
        ("finite", "positive-infinity"),
        ("negative-infinity", "negative-infinity"),
        ("positive-infinity", "positive-infinity"),
    } <= transitions
    assert any(
        any(np.isposinf(p.fitness) for p in s.population.values())
        and any(np.isfinite(p.fitness) for p in s.population.values())
        and not any(np.isneginf(p.fitness) for p in s.population.values())
        for s in run.algorithm.stages
        if s.name == "inter-after"
    )
    final = run.stage(3, "post-after")
    assert final.best is not None and np.isneginf(final.best.fitness)


@pytest.mark.parametrize(
    ("fitness", "improved"),
    [(np.inf, False), (np.inf, True), (-np.inf, False), (-np.inf, True)],
    ids=[
        "positive-classical",
        "positive-best-relative",
        "negative-classical",
        "negative-best-relative",
    ],
)
def test_infinite_plateaus_have_finite_probabilities_and_strict_ties(
    monkeypatch: pytest.MonkeyPatch,
    small_bounds: dict[str, tuple[float, float]],
    fitness: float,
    improved: bool,
) -> None:
    def infinite_plateau(**variables: float) -> float:
        del variables
        return fitness

    run = run_bees(
        monkeypatch,
        small_bounds,
        objective=infinite_plateau,
        improved=improved,
        n_iterations=2,
    )
    assert_probabilities(run, improved)
    witnessed = assert_phase_mathematics(run, small_bounds, infinite_plateau)
    assert "employed-tie" in witnessed
    for t in (1, 2):
        assert set(run.stage(t, "inter-after").probabilities.values()) == {
            1.0 if improved else 1 / 3
        }
        for key, p in run.stage(t, "post-after").population.items():
            assert p.variables == run.stage(0, "post-after").population[key].variables
            assert p.trial_count == t + sum(
                run.stage(i, "inter-after").population[key].onlooker_count
                for i in range(1, t + 1)
            )
            assert p.fitness == fitness


@pytest.mark.parametrize(
    ("case_id", "optimal_sources"),
    [(0, 1), (7, 2)],
    ids=["all-onlookers-one-source", "two-infinite-minima"],
)
def test_classical_onlookers_allocate_only_to_infinite_minima(
    monkeypatch: pytest.MonkeyPatch,
    small_bounds: dict[str, tuple[float, float]],
    case_id: int,
    optimal_sources: int,
) -> None:
    run = run_bees(
        monkeypatch,
        small_bounds,
        objective=signed_extremes,
        seed=seed_for(case_id),
        n_iterations=1,
    )
    assert_probabilities(run, False)
    witnessed = assert_phase_mathematics(run, small_bounds, signed_extremes)
    inter = run.stage(1, "inter-after")
    optimal = {key for key, p in inter.population.items() if np.isneginf(p.fitness)}
    assert len(optimal) == optimal_sources
    assert inter.probabilities == {
        key: 1 / optimal_sources if key in optimal else 0 for key in inter.population
    }
    assert set(inter.selected) <= optimal
    if optimal_sources == 1:
        (key,) = optimal
        assert inter.selected == [key]
        assert inter.population[key].onlooker_count == 3
        before, after = (
            run.stage(1, "onlooker-before", key),
            run.stage(1, "onlooker-after", key),
        )
        assert after.evaluations - before.evaluations == 3
        phase_calls = run.calls[before.evaluations : after.evaluations]
        assert all(call.update_index is not None for call in phase_calls)
        assert [
            attempt.identifier
            for attempt in run.algorithm.attempts
            if before.evaluations <= attempt.evaluation_index < after.evaluations
        ] == [key] * 3
        assert {"multiple-onlookers", "no-onlooker"} <= witnessed


@pytest.mark.parametrize(
    ("improved", "case_id"), [(False, 1), (True, 2)], ids=["classical", "best-relative"]
)
def test_negative_infinite_historical_best_survives_worse_scout(
    monkeypatch: pytest.MonkeyPatch,
    small_bounds: dict[str, tuple[float, float]],
    improved: bool,
    case_id: int,
) -> None:
    run = run_bees(
        monkeypatch,
        small_bounds,
        objective=signed_extremes,
        seed=seed_for(case_id),
        improved=improved,
        limit=1,
        n_iterations=4,
    )
    assert_probabilities(run, improved)
    _ = assert_phase_mathematics(run, small_bounds, signed_extremes)
    assert "best-source-replaced-by-worse-scout" in assert_scouts(
        run, small_bounds, signed_extremes
    )
    optimum_seen = False
    for stage in run.algorithm.stages:
        if stage.best is None:
            continue
        optimum_seen |= bool(np.isneginf(stage.best.fitness))
        if optimum_seen:
            assert np.isneginf(stage.best.fitness)
    assert optimum_seen
