"""GSA equations through the real Optimizer/Continuous/Serial lifecycle.

Rashedi, Nezamabadi-pour & Saryazdi (2009), doi:10.1016/j.ins.2009.03.004:
eqs. (7)-(12), (14)-(18), (21), (28), and the linear K-best reduction in §4.
Full text: https://ahmetcevahircinar.com.tr/wp-content/uploads/2017/04/
GSA_A_Gravitational_Search_Algorithm.pdf (pp. 2235-2236, 2240).

Tyrannis contracts: setup precedes t=0; synchronous consolidated inputs;
ceil-rounded K-best interpolation to max(1, ceil(N*fraction)), inclusive
endpoints; configurable Lp distance and distance power; float64 epsilon added
AFTER that power; uniform masses for equal fitness; position-only clipping.
These choices are not all prescribed by the paper. Cancellation of equal
passive/inertial masses extends the acceleration formula to zero-mass agents.
Near-equal/nonfinite fitness fallbacks and migration are outside this suite.
"""

import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from math import ceil, exp, fsum, sqrt
from sys import float_info
from typing import override

import pytest

from tests._support import factories
from tests._support.assertions import (
    assert_optimizer_result_consistent,
    assert_particle_state_consistent,
)
from tests._support.numerics import BASE_SEED, STRICT_ATOL, STRICT_RTOL
from tests._support.objectives import constant_objective, sphere
from tyrannis.algorithm.gravitational import GSA, GSAParticle
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


class ObservedGSA(GSA):
    """Record inputs and consolidated outputs; super() performs all mathematics."""

    @override
    def pre_iteration(self, actual_iter: int) -> None:
        super().pre_iteration(actual_iter)
        self.actual_iter: int = actual_iter
        if actual_iter == 0:
            self.inputs: dict[tuple[int, str], UpdateInputs] = {}
            self.states: dict[int, dict[str, GSAParticle]] = {}

    @override
    def update_particle(self, identifier: str) -> GSAParticle:
        particle = self.population[identifier]
        assert particle.velocity is not None
        assert self.gravitational_constant is not None
        self.inputs[self.actual_iter, identifier] = UpdateInputs(
            random=particle.random_cache.copy(),
            positions={key: p.variables.copy() for key, p in self.population.items()},
            masses={key: p.mass for key, p in self.population.items()},
            velocity=particle.velocity.copy(),
            k_best=self._k_best,
            gravity=self.gravitational_constant,
        )
        return super().update_particle(identifier)

    @override
    def post_iteration(self, actual_iter: int) -> None:
        super().post_iteration(actual_iter)
        self.states[actual_iter] = deepcopy(self.population)


def run_swarm(
    algorithm: ObservedGSA,
    bounds: dict[str, tuple[float, float]],
    *,
    n_iterations: int = 5,
    objective: Callable[..., float] = sphere,
) -> ObservedGSA:
    processor = Serial()
    optimizer = factories.make_optimizer(
        Continuous(bounds, cost_function=objective),
        algorithm,
        processor=processor,
        n_particles=5,
        n_iterations=n_iterations,
        seed=BASE_SEED,
        history="iteration",
    ).fit()
    (executor,) = processor.processors_pool.values()
    # Observe the executed replica; the template does not contain this trajectory.
    assert isinstance(executor._algorithm, ObservedGSA)  # pyright: ignore[reportPrivateUsage]
    executed = executor._algorithm  # pyright: ignore[reportPrivateUsage]
    assert executed is not algorithm
    states = executed.states
    assert set(states) == set(range(n_iterations + 1))
    assert all(len(population) == 5 for population in states.values())
    assert set(executed.inputs) == {
        (iteration, key)
        for iteration, population in states.items()
        if iteration > 0
        for key in population
    }
    rows: list[list[object]] = list(optimizer.internal_history_generator_)
    for row in rows:
        iteration, identifier = row[1], row[2]
        assert isinstance(iteration, int) and isinstance(identifier, str)
        particle = states[iteration][identifier]
        assert row[5:-1] == [particle.fitness, *particle.variables.values()]
        assert row[-1] == json.dumps(
            {
                "velocity": particle.velocity,
                "mass": particle.mass,
                "acceleration": particle.acceleration,
            }
        )
    assert len(rows) == 5 * (n_iterations + 1)
    for iteration, population in states.items():
        for particle in population.values():
            assert_particle_state_consistent(particle, bounds)
            assert particle.fitness == pytest.approx(
                objective(**particle.variables), rel=STRICT_RTOL, abs=STRICT_ATOL
            )
            if iteration == 0:
                assert particle.velocity == dict.fromkeys(bounds, 0.0)
                assert particle.acceleration == dict.fromkeys(bounds, 0.0)
    assert_optimizer_result_consistent(optimizer)
    assert optimizer.best_fitness == pytest.approx(
        min(p.fitness for population in states.values() for p in population.values()),
        rel=STRICT_RTOL,
        abs=STRICT_ATOL,
    )
    return executed


def expected_masses(population: dict[str, GSAParticle]) -> dict[str, float]:
    best = min(p.fitness for p in population.values())
    worst = max(p.fitness for p in population.values())
    if best == worst:
        # Tyrannis degeneracy contract, not eq. (15), which would divide by zero.
        return dict.fromkeys(population, 1 / len(population))
    raw = {
        key: float((p.fitness - worst) / (best - worst))
        for key, p in population.items()
    }
    return {key: mass / fsum(raw.values()) for key, mass in raw.items()}


def expected_count(n: int, iterations: int, t: int, fraction: float) -> int:
    """Linear interpolation between the initial population and Tyrannis' floor."""
    minimum = max(1, ceil(n * fraction))
    if iterations == 1:
        return n  # Only t=0 exists; no terminal interpolation interval.
    return ceil(((iterations - 1 - t) * n + t * minimum) / (iterations - 1))


def assert_transitions(
    algorithm: ObservedGSA,
    bounds: dict[str, tuple[float, float]],
    *,
    g_zero: float,
    alpha: float,
    fraction: float = 0.0,
    norm: float = 2.0,
    power: float = 1.0,
) -> set[bool]:
    states = algorithm.states
    iterations = len(states) - 1
    clipping: set[bool] = set()
    for population in states.values():
        masses = {key: p.mass for key, p in population.items()}
        assert masses == pytest.approx(
            expected_masses(population), rel=STRICT_RTOL, abs=STRICT_ATOL
        )
        assert fsum(masses.values()) == pytest.approx(
            1, rel=STRICT_RTOL, abs=STRICT_ATOL
        )
        assert min(masses.values()) >= 0
        if len({p.fitness for p in population.values()}) == len(population):
            best = min(population, key=lambda key: population[key].fitness)
            worst = max(population, key=lambda key: population[key].fitness)
            assert masses[best] == max(masses.values())
            assert masses[worst] == 0

    for (iteration, identifier), observed in algorithm.inputs.items():
        t = iteration - 1
        previous = states[t]  # Same immutable mathematical state for EVERY particle.
        before, after = previous[identifier], states[iteration][identifier]
        masses = expected_masses(previous)
        gravity = g_zero * exp(-alpha * t / iterations)
        count = expected_count(len(previous), iterations, t, fraction)
        ranked = sorted(previous, key=lambda key: previous[key].fitness)
        assert len(observed.k_best) == count
        assert len(set(observed.k_best)) == count
        if len({p.fitness for p in previous.values()}) == len(previous):
            assert observed.k_best == tuple(ranked[:count])
        else:
            # Ties determine fitness ranks, never unique agent identities.
            assert [previous[key].fitness for key in observed.k_best] == [
                previous[key].fitness for key in ranked[:count]
            ]
        assert observed.gravity == pytest.approx(
            gravity, rel=STRICT_RTOL, abs=STRICT_ATOL
        )
        if t == 0:
            assert observed.gravity == g_zero
        assert observed.masses == pytest.approx(
            masses, rel=STRICT_RTOL, abs=STRICT_ATOL
        )
        for key, particle in previous.items():
            assert observed.positions[key] == particle.variables
        assert observed.velocity == before.velocity
        attractors = [key for key in observed.k_best if key != identifier]
        assert set(observed.random) == {
            "velocity",
            *(f"force-{key}" for key in attractors),
        }
        assert all(0 <= value < 1 for value in observed.random.values())
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
            # Eqs. (7), (10), (14), (21): passive/inertial masses cancel.
            acceleration = fsum(
                gravity
                * masses[key]
                * observed.random[f"force-{key}"]
                * (previous[key].variables[d] - before.variables[d])
                / (distances[key] ** power + float_info.epsilon)
                for key in attractors
            )
            velocity = observed.random["velocity"] * before.velocity[d] + acceleration
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
    return clipping


def test_gsa_transition_equations_and_temporal_alignment(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    algorithm = run_swarm(ObservedGSA(g_zero=12, alpha=2), small_bounds)
    assert assert_transitions(algorithm, small_bounds, g_zero=12, alpha=2) == {
        False,
        True,
    }
    assert any(
        iteration > 1 and any(abs(v) > STRICT_ATOL for v in observed.velocity.values())
        for (iteration, _), observed in algorithm.inputs.items()
    ), "The previous-velocity term must contribute after the first update"
    assert any(
        algorithm.states[t][key].mass != algorithm.states[t + 1][key].mass
        for t in range(5)
        for key in algorithm.population
    ), "Consecutive masses must differ to distinguish their time indices"
    assert any(
        sorted(algorithm.states[t], key=lambda key: algorithm.states[t][key].fitness)
        != sorted(
            algorithm.states[t + 1],
            key=lambda key: algorithm.states[t + 1][key].fitness,
        )
        for t in range(4)
    ), "Changing fitness ranks must distinguish the K-best input snapshots"
    ids = list(algorithm.population)
    assert any(
        algorithm.states[t + 1][key].fitness
        < min(p.fitness for p in algorithm.states[t].values())
        for t in range(5)
        for key in ids[:-1]
    ), "Exercise a new best before the serial sweep ends"
    last_inputs = {
        key: value
        for (iteration, key), value in algorithm.inputs.items()
        if iteration == 5
    }
    (sole_attractor,) = last_inputs[ids[0]].k_best
    assert last_inputs[sole_attractor].random.keys() == {"velocity"}
    assert algorithm.states[5][sole_attractor].acceleration == dict.fromkeys(
        small_bounds, 0.0
    )
    assert any(
        any(abs(a) > STRICT_ATOL for a in (p.acceleration or {}).values())
        for key, p in algorithm.states[5].items()
        if key != sole_attractor
    ), "The sole attractor must still accelerate other agents"


@pytest.mark.parametrize("fraction", [0.0, 0.6, 1.0], ids=["one", "fraction", "all"])
def test_k_best_schedule_respects_minimum_fraction(
    small_bounds: dict[str, tuple[float, float]],
    fraction: float,
) -> None:
    algorithm = run_swarm(
        ObservedGSA(g_zero=1, alpha=2, k_agents_percent=fraction), small_bounds
    )
    # Gentle motion keeps ranks unique: identity assertions must never be vacuous.
    assert all(
        len({p.fitness for p in population.values()}) == 5
        for population in algorithm.states.values()
    )
    _ = assert_transitions(
        algorithm, small_bounds, g_zero=1, alpha=2, fraction=fraction
    )
    identifier = next(iter(algorithm.population))
    sizes = [len(algorithm.inputs[t + 1, identifier].k_best) for t in range(5)]
    assert sizes == [expected_count(5, 5, t, fraction) for t in range(5)]
    assert sizes[0] == 5
    assert sizes[-1] == max(1, ceil(5 * fraction))
    assert sizes == sorted(sizes, reverse=True)
    assert min(sizes) == sizes[-1]


def test_single_iteration_uses_initial_g_and_full_initial_k_best(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    algorithm = run_swarm(ObservedGSA(g_zero=12, alpha=2), small_bounds, n_iterations=1)
    _ = assert_transitions(algorithm, small_bounds, g_zero=12, alpha=2)
    assert all(
        observed.gravity == 12 and set(observed.k_best) == set(algorithm.population)
        for observed in algorithm.inputs.values()
    )


def test_nondefault_distance_norm_and_power_drive_acceleration(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    algorithm = run_swarm(
        ObservedGSA(g_zero=12, alpha=2, r_norm=1, r_power=2), small_bounds
    )
    _ = assert_transitions(algorithm, small_bounds, g_zero=12, alpha=2, norm=1, power=2)
    distinguishable = False
    for (iteration, identifier), observed in algorithm.inputs.items():
        previous = algorithm.states[iteration - 1]
        for key in observed.k_best:
            if key == identifier or previous[key].mass == 0:
                continue
            delta = [
                previous[key].variables[d] - previous[identifier].variables[d]
                for d in small_bounds
            ]
            l1 = fsum(abs(value) for value in delta)
            l2 = sqrt(fsum(value**2 for value in delta))
            distinguishable |= (
                observed.random[f"force-{key}"] > 0
                and l1 != pytest.approx(l2, rel=STRICT_RTOL, abs=STRICT_ATOL)
                and l1**2 != pytest.approx(l1, rel=STRICT_RTOL, abs=STRICT_ATOL)
            )
    assert distinguishable, "A nonzero interaction must distinguish both norm and power"


def test_equal_fitness_produces_uniform_masses(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    algorithm = run_swarm(
        ObservedGSA(g_zero=12, alpha=2),
        small_bounds,
        objective=constant_objective,
        n_iterations=2,
    )
    _ = assert_transitions(algorithm, small_bounds, g_zero=12, alpha=2)
    for population in algorithm.states.values():
        assert {key: p.mass for key, p in population.items()} == pytest.approx(
            dict.fromkeys(population, 1 / 5), rel=STRICT_RTOL, abs=STRICT_ATOL
        )
    assert any(
        algorithm.states[1][key].variables != particle.variables
        for key, particle in algorithm.states[0].items()
    ), "Uniform masses must persist after actual movement"


def test_zero_alpha_keeps_gravitational_constant(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    algorithm = run_swarm(ObservedGSA(g_zero=1, alpha=0), small_bounds)
    _ = assert_transitions(algorithm, small_bounds, g_zero=1, alpha=0)
    assert {observed.gravity for observed in algorithm.inputs.values()} == {1}
