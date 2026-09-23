"""PSOGSA mathematics through Optimizer + Continuous + Serial.

Mirjalili & Hashim (2010), doi:10.1109/ICCIA.2010.6141614, eqs. (3)-(6),
(9)-(10), and section IV (p. 376): w is random in [0,1], c1=.5, c2=1.5.
Full text: https://www.researchgate.net/publication/261355175
Rashedi et al. (2009), doi:10.1016/j.ins.2009.03.004, eqs. (14)-(18), (21):
https://ahmetcevahircinar.com.tr/wp-content/uploads/2017/04/
GSA_A_Gravitational_Search_Algorithm.pdf

Radosavljevic (2016), doi:10.1080/08839514.2016.1185860, eq. (26), p. 454,
explicitly distinguishes r1*v + C1*r2*a + C2*r3*(gbest-x):
https://www.researchgate.net/publication/304341735
This adopted reference resolves the repeated ``rand`` notation of the original
equation. Tyrannis draws those three weights separately per agent and reuses
each scalar across dimensions. We check the equation conditioned on those
realized values; distinct draws are not a statistical proof of independence.
Lacerda Junior et al. (2019), https://arxiv.org/pdf/1911.05205, eq. (10),
also repeats the original hybrid equation without explicitly indexing rand.

Tyrannis conventions: initialization before t=0; synchronous sweeps; bounded
initial velocities; zero initial acceleration; position clipping; float64
epsilon after the distance power; inclusive ceil-rounded K-best reduction.
Equal-fitness uniform masses are a symmetric degeneracy extension, not the
published mass equation (which divides by zero). Nonfinite fitness is outside
both original papers: finite-subpopulation normalization is tested separately,
and the undocumented mixed single-level fallback must respect the documented
ordering principle that better solutions have greater mass.
"""

import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from math import ceil, exp, fsum, isfinite, sqrt
from sys import float_info
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
from tyrannis.algorithm.psogsa import PSOGSA, PSOGSAParticle
from tyrannis.core.algorithm import FITNESS_UNDEFINED
from tyrannis.processor.serial import Serial
from tyrannis.space.continuous import Continuous


@dataclass
class UpdateInputs:
    random: dict[str, float]
    positions: dict[str, dict[str, float]]
    masses: dict[str, float]
    velocity: dict[str, float]
    k_best: tuple[str, ...]
    gravity: float
    social: dict[str, float]


class ObservedPSOGSA(PSOGSA):
    """Observe mathematical inputs/candidates; super() performs every operation."""

    @override
    def pre_iteration(self, actual_iter: int) -> None:
        self.actual_iter: int = actual_iter
        if actual_iter == 0:
            # Capture creation before setup can move anything, including in pre_iteration.
            self.initial: dict[str, PSOGSAParticle] = deepcopy(self.population)
            self.inputs: dict[tuple[int, str], UpdateInputs] = {}
            self.candidates: dict[tuple[int, str], PSOGSAParticle] = {}
            self.states: dict[int, dict[str, PSOGSAParticle]] = {}
            self.best: dict[int, PSOGSAParticle] = {}
            self.k_best: dict[int, tuple[str, ...]] = {}
            self.gravity: dict[int, float] = {}
        super().pre_iteration(actual_iter)

    @override
    def update_particle(self, identifier: str) -> PSOGSAParticle:
        particle = self.population[identifier]
        assert particle.velocity is not None
        assert self.gravitational_constant is not None
        assert self.local_best is not None
        self.inputs[self.actual_iter, identifier] = UpdateInputs(
            random=particle.random_cache.copy(),
            positions={key: p.variables.copy() for key, p in self.population.items()},
            masses={key: p.mass for key, p in self.population.items()},
            velocity=particle.velocity.copy(),
            k_best=self._k_best,
            gravity=self.gravitational_constant,
            social=self.local_best.variables.copy(),
        )
        result = super().update_particle(identifier)
        self.candidates[self.actual_iter, identifier] = deepcopy(result)
        return result

    @override
    def post_iteration(self, actual_iter: int) -> None:
        super().post_iteration(actual_iter)
        assert self.local_best is not None
        assert self.gravitational_constant is not None
        self.states[actual_iter] = deepcopy(self.population)
        self.best[actual_iter] = deepcopy(self.local_best)
        self.k_best[actual_iter] = self._k_best
        self.gravity[actual_iter] = self.gravitational_constant


def run_swarm(
    algorithm: ObservedPSOGSA,
    bounds: dict[str, tuple[float, float]],
    *,
    n_iterations: int = 5,
    n_particles: int = 5,
    objective: Callable[..., float] = sphere,
    seed: int = BASE_SEED,
) -> ObservedPSOGSA:
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
    # The public pool locates the real replica; its algorithm has no public accessor.
    assert isinstance(executor._algorithm, ObservedPSOGSA)  # pyright: ignore[reportPrivateUsage]
    executed = executor._algorithm  # pyright: ignore[reportPrivateUsage]
    assert executed is not algorithm
    assert set(executed.states) == set(range(n_iterations + 1))
    assert all(len(pop) == n_particles for pop in executed.states.values())
    assert (
        set(executed.inputs)
        == set(executed.candidates)
        == {
            (iteration, key)
            for iteration in range(1, n_iterations + 1)
            for key in executed.initial
        }
    )
    rows: list[list[object]] = list(optimizer.internal_history_generator_)
    assert len(rows) == n_particles * (n_iterations + 1)
    for row in rows:
        iteration, identifier = row[1], row[2]
        assert isinstance(iteration, int) and isinstance(identifier, str)
        particle = executed.states[iteration][identifier]
        assert row[5:-1] == [particle.fitness, *particle.variables.values()]
        assert row[-1] == json.dumps(
            {
                "velocity": particle.velocity,
                "mass": particle.mass,
                "acceleration": particle.acceleration,
            }
        )
    for iteration, population in executed.states.items():
        for particle in population.values():
            assert_particle_state_consistent(particle, bounds)
            assert particle.fitness == pytest.approx(
                objective(**particle.variables), rel=STRICT_RTOL, abs=STRICT_ATOL
            )
            assert particle.velocity is not None and particle.acceleration is not None
            assert set(particle.velocity) == set(particle.acceleration) == set(bounds)
            assert all(
                isfinite(v)
                for v in (
                    *particle.variables.values(),
                    *particle.velocity.values(),
                    *particle.acceleration.values(),
                    particle.mass,
                )
            )
        history = [
            p
            for t, pop in executed.states.items()
            if t <= iteration
            for p in pop.values()
        ]
        best = executed.best[iteration]
        minimum = min(p.fitness for p in history)
        assert best.fitness == minimum
        # Ties (including all-inf) do not identify a unique mathematical best.
        assert any(
            p.fitness == minimum and p.variables == best.variables for p in history
        )
    assert_optimizer_result_consistent(optimizer)
    assert optimizer.best_fitness == executed.best[n_iterations].fitness
    return executed


def expected_masses(population: dict[str, PSOGSAParticle]) -> dict[str, float]:
    """Normalized fitness deficits; no surrogate value is assigned to infinity."""
    finite = {
        key: float(p.fitness) for key, p in population.items() if isfinite(p.fitness)
    }
    if not finite or len({p.fitness for p in population.values()}) == 1:
        # Symmetric fallback for a genuinely tied population, finite or all-inf.
        return dict.fromkeys(population, 1 / len(population))
    assert len(set(finite.values())) > 1, "Mixed single-level fitness needs a contract"
    worst = max(finite.values())
    deficits = {
        key: worst - finite[key] if key in finite else 0.0 for key in population
    }
    # (best-worst) cancels in Rashedi (15)/(16). The finite-only extension
    # discards infeasible agents; it is NOT the limit worst -> infinity.
    return {key: value / fsum(deficits.values()) for key, value in deficits.items()}


def assert_masses(population: dict[str, PSOGSAParticle]) -> None:
    masses = {key: p.mass for key, p in population.items()}
    assert masses == pytest.approx(
        expected_masses(population), rel=STRICT_RTOL, abs=STRICT_ATOL
    )
    assert all(isfinite(mass) and mass >= 0 for mass in masses.values())
    assert fsum(masses.values()) == pytest.approx(1, rel=STRICT_RTOL, abs=STRICT_ATOL)


def test_initial_state_is_mathematically_valid(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    algorithm = run_swarm(ObservedPSOGSA(), small_bounds, n_iterations=1)
    initial = algorithm.states[0]
    assert algorithm.gravity[0] == 1.0
    assert set(algorithm.k_best[0]) == set(initial)
    assert_masses(initial)
    assert len({p.fitness for p in initial.values()}) == len(initial)
    best = min(initial.values(), key=lambda p: p.fitness)
    worst = max(initial.values(), key=lambda p: p.fitness)
    assert best.mass > worst.mass == 0
    assert algorithm.best[0].variables == best.variables
    for key, particle in initial.items():
        created = algorithm.initial[key]
        assert isfinite(particle.fitness)
        assert particle.variables == created.variables
        assert particle.velocity == created.velocity
        assert particle.velocity is not None
        for name, (lower, upper) in small_bounds.items():
            assert -(upper - lower) <= particle.velocity[name] <= upper - lower
        assert particle.acceleration == dict.fromkeys(small_bounds, 0.0)
        assert particle.random_cache == {}
        assert particle.candidate_variables is None
        assert particle.candidate_fitness is None
        first = algorithm.inputs[1, key]
        assert first.positions[key] == particle.variables
        assert first.velocity == particle.velocity
        assert first.masses[key] == particle.mass
        assert first.gravity == 1.0
        assert first.social == best.variables
        assert set(first.k_best) == set(initial)


def expected_count(n: int, iterations: int, t: int, fraction: float) -> int:
    """GSA linear reduction with Tyrannis' inclusive endpoints and integer floor."""
    if iterations == 1:
        return n
    floor = max(1, ceil(n * fraction))
    return ceil(((iterations - 1 - t) * n + t * floor) / (iterations - 1))


def assert_transitions(
    algorithm: ObservedPSOGSA,
    bounds: dict[str, tuple[float, float]],
    *,
    g_zero: float = 1.0,
    alpha: float = 20.0,
    c1: float = 0.5,
    c2: float = 1.5,
    fraction: float = 1.0,
    norm: float = 2.0,
    power: float = 1.0,
) -> tuple[set[bool], bool]:
    """Check independent equations against each real candidate and consolidation."""
    states = algorithm.states
    iterations = len(states) - 1
    clipping: set[bool] = set()
    all_terms_active = False
    for population in states.values():
        assert_masses(population)
    for (iteration, identifier), observed in algorithm.inputs.items():
        t = iteration - 1
        previous = states[t]
        before, after = previous[identifier], states[iteration][identifier]
        masses = expected_masses(previous)
        gravity = g_zero * exp(-alpha * t / iterations)
        count = expected_count(len(previous), iterations, t, fraction)
        ranked = sorted(previous, key=lambda key: previous[key].fitness)
        assert len(observed.k_best) == len(set(observed.k_best)) == count
        assert set(observed.k_best) <= set(previous)
        if len({p.fitness for p in previous.values()}) == len(previous):
            assert set(observed.k_best) == set(ranked[:count])
        else:
            assert sorted(previous[key].fitness for key in observed.k_best) == [
                previous[key].fitness for key in ranked[:count]
            ]
        if fraction == 1 or t == 0:
            assert set(observed.k_best) == set(previous)
        # Input snapshots must remain identical for early AND late serial agents.
        assert observed.k_best == algorithm.k_best[t]
        assert observed.gravity == pytest.approx(
            gravity, rel=STRICT_RTOL, abs=STRICT_ATOL
        )
        assert observed.masses == pytest.approx(
            masses, rel=STRICT_RTOL, abs=STRICT_ATOL
        )
        assert observed.positions == {key: p.variables for key, p in previous.items()}
        assert observed.velocity == before.velocity
        historical = [
            p for time, pop in states.items() if time <= t for p in pop.values()
        ]
        minimum = min(p.fitness for p in historical)
        minimizers = [p for p in historical if p.fitness == minimum]
        assert any(observed.social == p.variables for p in minimizers)
        assert observed.social == algorithm.best[t].variables
        attractors = set(observed.k_best) - {identifier}
        assert set(observed.random) == {
            "inertia",
            "gravitational",
            "global-best",
            *(f"force-{key}" for key in attractors),
        }
        assert all(0 <= value < 1 for value in observed.random.values())
        candidate = algorithm.candidates[iteration, identifier]
        assert candidate.variables == before.variables
        assert candidate.fitness == before.fitness
        assert candidate.candidate_variables == after.variables
        assert candidate.candidate_fitness == after.fitness
        assert candidate.velocity == after.velocity
        assert candidate.acceleration == after.acceleration
        assert candidate.random_cache == observed.random
        distances = {
            key: fsum(
                abs(previous[key].variables[d] - before.variables[d]) ** norm
                for d in bounds
            )
            ** (1 / norm)
            for key in attractors
        }
        assert before.velocity is not None and after.velocity is not None
        assert after.acceleration is not None
        for d, (lower, upper) in bounds.items():
            # Rashedi (7)/(10)/(14)/(21): cancel passive/inertial masses,
            # retaining the continuous extension for a zero-mass target.
            acceleration = fsum(
                gravity
                * masses[key]
                * observed.random[f"force-{key}"]
                * (previous[key].variables[d] - before.variables[d])
                / (distances[key] ** power + float_info.epsilon)
                for key in attractors
            )
            persistence = observed.random["inertia"] * before.velocity[d]
            gravitational = c1 * observed.random["gravitational"] * acceleration
            social = (
                c2
                * observed.random["global-best"]
                * (observed.social[d] - before.variables[d])
            )
            velocity = fsum((persistence, gravitational, social))
            # Residuals isolate each coefficient and term, including the old velocity.
            for component, others in (
                (persistence, gravitational + social),
                (gravitational, persistence + social),
                (social, persistence + gravitational),
            ):
                assert after.velocity[d] - others == pytest.approx(
                    component, rel=STRICT_RTOL, abs=STRICT_ATOL
                )
            raw_position = before.variables[d] + velocity
            position = min(upper, max(lower, raw_position))
            assert after.acceleration[d] == pytest.approx(
                acceleration, rel=STRICT_RTOL, abs=STRICT_ATOL
            )
            assert after.velocity[d] == pytest.approx(
                velocity, rel=STRICT_RTOL, abs=STRICT_ATOL
            )
            assert after.variables[d] == pytest.approx(
                position, rel=STRICT_RTOL, abs=STRICT_ATOL
            )
            clipping.add(not lower <= raw_position <= upper)
            all_terms_active |= all(
                abs(value) > STRICT_ATOL
                for value in (
                    persistence,
                    gravitational,
                    social,
                )
            )
    return clipping, all_terms_active


def test_psogsa_transition_equation_and_temporal_alignment(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    # BASE_SEED improves on every sweep; this derived seed also preserves gbest.
    algorithm = run_swarm(ObservedPSOGSA(), small_bounds, seed=seed_for(1))
    clipping, active = assert_transitions(algorithm, small_bounds)
    assert clipping == {False, True}
    assert active, "All three hybrid terms must contribute in the same dimension/update"
    assert [algorithm.gravity[t + 1] for t in range(5)] == pytest.approx(
        [exp(-20 * t / 5) for t in range(5)], rel=STRICT_RTOL, abs=STRICT_ATOL
    )
    improvements = {
        algorithm.best[t + 1].fitness < algorithm.best[t].fitness for t in range(5)
    }
    assert improvements == {False, True}, (
        "Exercise replacement and preservation of gbest"
    )
    ids = list(algorithm.initial)
    assert any(
        algorithm.states[t + 1][key].fitness < algorithm.best[t].fitness
        for t in range(5)
        for key in ids[:-1]
    ), "A new best before the sweep ends must not leak into later social terms"
    assert any(
        algorithm.states[t][key].mass != algorithm.states[t + 1][key].mass
        for t in range(5)
        for key in ids
    ), "Changing masses distinguish adjacent time indices"
    assert all(
        len(
            {
                observed.random[name]
                for name in ("inertia", "gravitational", "global-best")
            }
        )
        == 3
        for observed in algorithm.inputs.values()
    ), "Separate realized weights distinguish accidental reuse of one cache entry"


@pytest.mark.parametrize("fraction", [0.0, 0.6, 1.0], ids=["one", "fraction", "all"])
def test_k_best_extension_respects_minimum_fraction(
    small_bounds: dict[str, tuple[float, float]],
    fraction: float,
) -> None:
    algorithm = run_swarm(ObservedPSOGSA(k_agents_percent=fraction), small_bounds)
    _ = assert_transitions(algorithm, small_bounds, fraction=fraction)
    assert all(
        len({p.fitness for p in pop.values()}) == 5 for pop in algorithm.states.values()
    ), "Exercise untied identities"
    sizes = [len(algorithm.k_best[t]) for t in range(5)]
    assert sizes == [expected_count(5, 5, t, fraction) for t in range(5)]
    assert sizes[0] == 5
    assert sizes[-1] == max(1, ceil(5 * fraction))
    assert sizes == sorted(sizes, reverse=True)
    assert any(
        sorted(algorithm.states[t], key=lambda key: algorithm.states[t][key].fitness)
        != sorted(
            algorithm.states[t + 1],
            key=lambda key: algorithm.states[t + 1][key].fitness,
        )
        for t in range(4)
    ), "Different ranks must distinguish consecutive consolidated snapshots"
    if fraction == 0:
        (attractor,) = algorithm.k_best[4]
        assert algorithm.states[5][attractor].acceleration == dict.fromkeys(
            small_bounds, 0.0
        )
        assert set(algorithm.inputs[5, attractor].random) == {
            "inertia",
            "gravitational",
            "global-best",
        }
        assert any(
            any(abs(a) > STRICT_ATOL for a in (p.acceleration or {}).values())
            for key, p in algorithm.states[5].items()
            if key != attractor
        ), "The sole attractor must exert a nonzero force on another particle"


def test_single_iteration_uses_t_zero_state(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    algorithm = run_swarm(
        ObservedPSOGSA(k_agents_percent=0),
        small_bounds,
        n_iterations=1,
        n_particles=3,
    )
    _ = assert_transitions(algorithm, small_bounds, fraction=0)
    assert algorithm.gravity == {0: 1.0, 1: 1.0}
    assert all(
        set(inputs.k_best) == set(algorithm.initial)
        for inputs in algorithm.inputs.values()
    )


def test_nondefault_distance_norm_and_power_drive_acceleration(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    algorithm = run_swarm(
        ObservedPSOGSA(r_norm=1, r_power=2),
        small_bounds,
        n_iterations=2,
        n_particles=3,
    )
    _ = assert_transitions(algorithm, small_bounds, norm=1, power=2)
    distinct_distance = False
    for (iteration, identifier), observed in algorithm.inputs.items():
        previous = algorithm.states[iteration - 1]
        for key in set(observed.k_best) - {identifier}:
            if previous[key].mass == 0:
                continue
            delta = [
                previous[key].variables[d] - previous[identifier].variables[d]
                for d in small_bounds
            ]
            l1, l2 = fsum(abs(v) for v in delta), sqrt(fsum(v * v for v in delta))
            distinct_distance |= (
                observed.random[f"force-{key}"] > 0
                and l1 != pytest.approx(l2, rel=STRICT_RTOL, abs=STRICT_ATOL)
                and l1**2 != pytest.approx(l1, rel=STRICT_RTOL, abs=STRICT_ATOL)
            )
    assert distinct_distance, "An active 2D interaction must distinguish norm AND power"


def test_zero_alpha_keeps_gravity_constant(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    algorithm = run_swarm(ObservedPSOGSA(alpha=0), small_bounds)
    _, active = assert_transitions(algorithm, small_bounds, alpha=0)
    assert active
    assert set(algorithm.gravity.values()) == {1.0}


def infinite_objective(**variables: float) -> float:
    """Every point is infeasible, without evaluation exceptions or NaN."""
    del variables
    return np.inf


def half_space_sphere(**variables: float) -> float:
    """The positive x half-space is infeasible; the other side is sphere."""
    return np.inf if variables["x"] > 0 else sphere(**variables)


def half_space_plateau(**variables: float) -> float:
    """Only one finite level, alongside strictly worse infinite fitness."""
    return np.inf if variables["x"] > 0 else constant_objective(**variables)


@pytest.mark.parametrize(
    "objective",
    [constant_objective, infinite_objective],
    ids=["finite", "all-infinite"],
)
def test_equal_fitness_uses_symmetric_uniform_mass_fallback(
    small_bounds: dict[str, tuple[float, float]],
    objective: Callable[..., float],
) -> None:
    algorithm = run_swarm(
        ObservedPSOGSA(alpha=0, k_agents_percent=0),
        small_bounds,
        objective=objective,
        n_iterations=2,
        n_particles=3,
    )
    _ = assert_transitions(algorithm, small_bounds, alpha=0, fraction=0)
    for population in algorithm.states.values():
        assert len({p.fitness for p in population.values()}) == 1
        assert {key: p.mass for key, p in population.items()} == pytest.approx(
            dict.fromkeys(population, 1 / 3),
            rel=STRICT_RTOL,
            abs=STRICT_ATOL,
        )
    assert any(
        algorithm.states[1][key].variables != p.variables
        for key, p in algorithm.states[0].items()
    ), "The fallback must permit motion"
    assert set(algorithm.k_best[0]) == set(algorithm.initial)
    assert len(algorithm.k_best[1]) == 1


@pytest.mark.parametrize(
    ("n_particles", "n_iterations"),
    [(3, 2), (5, 5)],
    ids=["one-infinite", "multiple-infinite"],
)
def test_infinite_fitness_does_not_contaminate_transitions(
    small_bounds: dict[str, tuple[float, float]],
    n_particles: int,
    n_iterations: int,
) -> None:
    algorithm = run_swarm(
        ObservedPSOGSA(alpha=0, k_agents_percent=0.6),
        small_bounds,
        objective=half_space_sphere,
        n_particles=n_particles,
        n_iterations=n_iterations,
    )
    initial = algorithm.states[0]
    assert sum(not isfinite(p.fitness) for p in initial.values()) == n_particles // 2
    assert len({p.fitness for p in initial.values() if isfinite(p.fitness)}) >= 2
    _ = assert_transitions(algorithm, small_bounds, alpha=0, fraction=0.6)
    for iteration, population in algorithm.states.items():
        assert isfinite(algorithm.best[iteration].fitness)
        finite = [p for p in population.values() if isfinite(p.fitness)]
        assert len({p.fitness for p in finite}) >= 2
        worst = max(finite, key=lambda p: p.fitness)
        best = min(finite, key=lambda p: p.fitness)
        assert best.mass > worst.mass == 0
        assert all(p.mass == 0 for p in population.values() if not isfinite(p.fitness))
    assert any(
        not isfinite(before.fitness) and isfinite(algorithm.states[t + 1][key].fitness)
        for t in range(n_iterations)
        for key, before in algorithm.states[t].items()
    ), "An initially infeasible region must not trap a particle with valid dynamics"


def test_mixed_single_finite_level_preserves_mass_ordering(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    # Minimal counterexample: two agents, one update, no synthetic state or RNG.
    algorithm = run_swarm(
        ObservedPSOGSA(),
        small_bounds,
        objective=half_space_plateau,
        n_particles=2,
        n_iterations=1,
    )
    initial = algorithm.states[0]
    (finite,) = [p for p in initial.values() if isfinite(p.fitness)]
    (infinite,) = [p for p in initial.values() if not isfinite(p.fitness)]
    assert finite.fitness == 1.0 and infinite.fitness == np.inf
    assert algorithm.best[0].variables == finite.variables
    assert set(algorithm.k_best[0]) == set(initial)
    assert fsum(p.mass for p in initial.values()) == pytest.approx(
        1,
        rel=STRICT_RTOL,
        abs=STRICT_ATOL,
    )
    assert finite.mass > infinite.mass, (
        "PRODUCTION_CONTRACT_VIOLATION [GSA mass / np.inf]: PSOGSA documents "
        "greater mass for better solutions, but finite fitness 1 and +inf "
        "receive equal mass. Neither the published finite-fitness equation "
        "nor a documented Tyrannis exception justifies this mixed fallback."
    )


def test_negative_infinite_fitness_is_best_in_both_hybrid_components(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    # MATHTESTS 34.3: both hybrid components must respect -inf < finite < +inf.
    def objective(**variables: float) -> float:
        x = variables["x"]
        return -np.inf if -1 < x < 1 else np.inf if x > 1 else sphere(**variables)

    assert np.isposinf(FITNESS_UNDEFINED) and -np.inf != FITNESS_UNDEFINED
    algorithm = run_swarm(
        ObservedPSOGSA(k_agents_percent=0),
        small_bounds,
        objective=objective,
        n_iterations=2,
        seed=seed_for(6),
    )
    initial = algorithm.states[0]
    assert sum(np.isneginf(p.fitness) for p in initial.values()) == 2
    assert sum(isfinite(p.fitness) for p in initial.values()) == 2
    assert sum(np.isposinf(p.fitness) for p in initial.values()) == 1
    for iteration, population in algorithm.states.items():
        best = algorithm.best[iteration]
        assert np.isneginf(best.fitness)
        assert best.variables == algorithm.best[0].variables
        assert all(isfinite(v) for v in best.variables.values())
        assert all(not p.new_particle for p in population.values())
        assert fsum(p.mass for p in population.values()) == pytest.approx(
            1, rel=STRICT_RTOL, abs=STRICT_ATOL
        )
        ranked = sorted(p.fitness for p in population.values())
        k_best = algorithm.k_best[iteration]
        assert [population[key].fitness for key in k_best] == ranked[: len(k_best)]

    active_social = False
    for (iteration, key), observed in algorithm.inputs.items():
        before = algorithm.states[iteration - 1][key]
        after = algorithm.states[iteration][key]
        assert observed.social == algorithm.best[iteration - 1].variables
        assert np.isneginf(objective(**observed.social))
        assert before.velocity is not None and after.velocity is not None
        assert after.acceleration is not None
        for name, (lower, upper) in small_bounds.items():
            # Isolate the published global term from the observed gravitational term.
            social = (
                1.5
                * observed.random["global-best"]
                * (observed.social[name] - before.variables[name])
            )
            active_social |= abs(social) > STRICT_ATOL
            velocity = (
                observed.random["inertia"] * before.velocity[name]
                + 0.5 * observed.random["gravitational"] * after.acceleration[name]
                + social
            )
            assert after.velocity[name] == pytest.approx(
                velocity, rel=STRICT_RTOL, abs=STRICT_ATOL
            )
            assert after.variables[name] == pytest.approx(
                min(upper, max(lower, before.variables[name] + velocity)),
                rel=STRICT_RTOL,
                abs=STRICT_ATOL,
            )
    assert active_social
    # run_swarm has checked finite mass/acceleration/velocity/position at every t.
    # Check masses last so the independent global component is exercised as well.
    for population in algorithm.states.values():
        minimizers = {key for key, p in population.items() if np.isneginf(p.fitness)}
        assert minimizers
        expected = {
            key: 1 / len(minimizers) if key in minimizers else 0.0 for key in population
        }
        assert {key: p.mass for key, p in population.items()} == pytest.approx(
            expected, rel=STRICT_RTOL, abs=STRICT_ATOL
        ), (
            "PRODUCTION_CONTRACT_VIOLATION [PSOGSA / -np.inf]: only minimizers share mass"
        )
