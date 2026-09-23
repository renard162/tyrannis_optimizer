"""PSO equations through the real Optimizer/Continuous/Serial lifecycle.

Sources: Kennedy & Eberhart (1995), doi:10.1109/ICNN.1995.488968;
Shi & Eberhart (1998), eq. (2), doi:10.1109/ICEC.1998.699146;
Clerc & Kennedy (2002), constriction, doi:10.1109/4235.985692.
Clipping, synchronous sweeps and inclusive update endpoints are Tyrannis contracts.
"""

import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import override

import numpy as np
import pytest

from tests._support import factories
from tests._support.assertions import (
    assert_optimizer_result_consistent,
    assert_particle_state_consistent,
)
from tests._support.numerics import BASE_SEED, STRICT_ATOL, STRICT_RTOL
from tests._support.objectives import sphere
from tyrannis.algorithm.pso import PSO, PSOParticle
from tyrannis.core.algorithm import FITNESS_UNDEFINED
from tyrannis.processor.serial import Serial
from tyrannis.space.continuous import Continuous


@dataclass
class UpdateInputs:
    random: dict[str, float]
    social: dict[str, float]
    inertia: float
    constriction: float


class ObservedPSO(PSO):
    """Observe inputs and typed consolidated states; super() does all the work."""

    @override
    def pre_iteration(self, actual_iter: int) -> None:
        super().pre_iteration(actual_iter)
        self.actual_iter: int = actual_iter
        if actual_iter == 0:
            self.inputs: dict[tuple[int, str], UpdateInputs] = {}
            self.states: dict[tuple[int, str], PSOParticle] = {}
            self.best: dict[int, PSOParticle] = {}

    @override
    def update_particle(self, identifier: str) -> PSOParticle:
        best = self.local_best
        assert best is not None
        self.inputs[self.actual_iter, identifier] = UpdateInputs(
            random=self.population[identifier].random_cache.copy(),
            social=best.variables.copy(),
            inertia=self._inertia,
            constriction=float(self._constriction),
        )
        return super().update_particle(identifier)

    @override
    def post_iteration(self, actual_iter: int) -> None:
        super().post_iteration(actual_iter)
        self.states.update(
            {(actual_iter, key): deepcopy(p) for key, p in self.population.items()}
        )
        assert self.local_best is not None
        self.best[actual_iter] = deepcopy(self.local_best)


def run_swarm(
    algorithm: ObservedPSO,
    bounds: dict[str, tuple[float, float]],
    *,
    n_iterations: int = 5,
    objective: Callable[..., float] = sphere,
) -> tuple[ObservedPSO, dict[tuple[int, str], PSOParticle]]:
    n_particles = 5
    processor = Serial()
    optimizer = factories.make_optimizer(
        Continuous(bounds, cost_function=objective),
        algorithm,
        processor=processor,
        n_particles=n_particles,
        n_iterations=n_iterations,
        seed=BASE_SEED,
        history="iteration",
    ).fit()
    (executor,) = processor.processors_pool.values()
    # The template has no executed population; this is the actual local replica.
    executed: object = executor._algorithm
    assert isinstance(executed, ObservedPSO)
    assert executed is not algorithm
    states = executed.states
    rows: list[list[object]] = list(optimizer.internal_history_generator_)
    for row in rows:
        iteration, identifier = row[1], row[2]
        assert isinstance(iteration, int) and isinstance(identifier, str)
        particle = states[iteration, identifier]
        assert row[5:-1] == [particle.fitness, *particle.variables.values()]
        assert row[-1] == json.dumps(
            {
                "velocity": particle.velocity,
                "personal_best_variables": particle.personal_best_variables,
                "personal_best_fitness": particle.personal_best_fitness,
            }
        )
    assert len(rows) == len(states)
    assert len(states) == n_particles * (n_iterations + 1)  # Include setup.
    assert set(executed.inputs) == {key for key in states if key[0] > 0}
    for (iteration, _), particle in states.items():
        assert_particle_state_consistent(particle, bounds)
        assert particle.fitness == pytest.approx(
            objective(**particle.variables), rel=STRICT_RTOL, abs=STRICT_ATOL
        )
        if iteration == 0:
            assert particle.personal_best_variables == particle.variables
            assert particle.personal_best_fitness == particle.fitness
    assert_optimizer_result_consistent(optimizer)
    assert optimizer.best_fitness == pytest.approx(
        min(p.fitness for p in states.values()), rel=STRICT_RTOL, abs=STRICT_ATOL
    )
    return executed, states


def assert_transitions(
    algorithm: ObservedPSO,
    states: dict[tuple[int, str], PSOParticle],
    bounds: dict[str, tuple[float, float]],
    weights: list[float],
    c1: float,
    c2: float,
    chi: float = 1.0,
) -> None:
    improvements: set[bool] = set()
    clipping: set[bool] = set()
    earlier_improvement = False
    ids = list(algorithm.population)
    for (iteration, identifier), observed in algorithm.inputs.items():
        before, after = states[iteration - 1, identifier], states[iteration, identifier]
        # Independent historical minimum, frozen before ANY update in this sweep.
        best = min(
            (p for (t, _), p in states.items() if t < iteration),
            key=lambda p: p.fitness,
        )
        assert observed.social == pytest.approx(
            best.variables, rel=STRICT_RTOL, abs=STRICT_ATOL
        )
        w = weights[iteration - 1]
        assert observed.inertia == pytest.approx(w, rel=STRICT_RTOL, abs=STRICT_ATOL)
        assert observed.constriction == pytest.approx(
            chi, rel=STRICT_RTOL, abs=STRICT_ATOL
        )
        assert before.velocity is not None and after.velocity is not None
        assert before.personal_best_variables is not None
        for name, (lower, upper) in bounds.items():
            x, v = before.variables[name], before.velocity[name]
            p, g = before.personal_best_variables[name], best.variables[name]
            r1 = observed.random[f"{name}-cognitive"]
            r2 = observed.random[f"{name}-social"]
            # Shi-Eberhart inertia / Clerc-Kennedy whole-update constriction.
            velocity = chi * (w * v + c1 * r1 * (p - x) + c2 * r2 * (g - x))
            position = min(upper, max(lower, x + velocity))
            assert after.velocity[name] == pytest.approx(
                velocity, rel=STRICT_RTOL, abs=STRICT_ATOL
            )
            assert after.variables[name] == pytest.approx(
                position, rel=STRICT_RTOL, abs=STRICT_ATOL
            )
            clipping.add(not lower <= x + velocity <= upper)
        improved = bool(after.fitness < before.personal_best_fitness)
        improvements.add(improved)
        if before.personal_best_fitness == after.fitness == np.inf:
            # An infinite tie may retain either visited position as personal best.
            assert after.personal_best_variables in (
                before.personal_best_variables,
                after.variables,
            )
        else:
            assert after.personal_best_variables == pytest.approx(
                after.variables if improved else before.personal_best_variables,
                rel=STRICT_RTOL,
                abs=STRICT_ATOL,
            )
        assert after.personal_best_fitness == pytest.approx(
            after.fitness if improved else before.personal_best_fitness,
            rel=STRICT_RTOL,
            abs=STRICT_ATOL,
        )
        earlier_improvement |= identifier != ids[-1] and after.fitness < best.fitness
    assert improvements == {False, True}, "Both personal-best branches must occur"
    assert clipping == {False, True}, "Both clipped and unclipped moves must occur"
    assert earlier_improvement, "Exercise a new best before the serial loop ends"


def test_fixed_inertia_transitions(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    algorithm, states = run_swarm(
        ObservedPSO(inertia=0.7, cognitive_coefficient=1.3, social_coefficient=1.7),
        small_bounds,
    )
    assert_transitions(algorithm, states, small_bounds, [0.7] * 5, 1.3, 1.7)


def test_linear_inertia_uses_both_endpoints_and_drives_velocity(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    algorithm, states = run_swarm(ObservedPSO(inertia=(0.4, 0.9)), small_bounds)
    # Shi-Eberhart time-decreasing inertia, with Tyrannis updates 1..5:
    # mathematical t=0..4 interpolates inclusively between max and min.
    weights = [0.9 + (0.4 - 0.9) * t / 4 for t in range(5)]
    assert_transitions(algorithm, states, small_bounds, weights, 1.5, 1.5)


def test_linear_inertia_single_iteration_uses_maximum(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    algorithm, states = run_swarm(
        ObservedPSO(inertia=(0.4, 0.9)), small_bounds, n_iterations=1
    )
    assert_transitions(algorithm, states, small_bounds, [0.9], 1.5, 1.5)


def test_clerc_kennedy_constriction_ignores_supplied_inertia(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    # Clerc-Kennedy, kappa=1 and phi=4.10: chi multiplies the WHOLE velocity.
    phi = 2.05 + 2.05
    chi = 2 / abs(2 - phi - sqrt(phi**2 - 4 * phi))
    first, states = run_swarm(
        ObservedPSO(
            inertia=0.2,
            cognitive_coefficient=2.05,
            social_coefficient=2.05,
            constriction_factor=True,
        ),
        small_bounds,
    )
    assert_transitions(first, states, small_bounds, [1.0] * 5, 2.05, 2.05, chi)
    second, other_states = run_swarm(
        ObservedPSO(
            inertia=(0.1, 0.9),
            cognitive_coefficient=2.05,
            social_coefficient=2.05,
            constriction_factor=True,
        ),
        small_bounds,
    )
    assert_transitions(second, other_states, small_bounds, [1.0] * 5, 2.05, 2.05, chi)
    assert first.inputs == second.inputs
    assert {key: p() for key, p in states.items()} == {
        key: p() for key, p in other_states.items()
    }


def test_mixed_infinite_fitness_preserves_bests_and_social_dynamics(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    def objective(**variables: float) -> float:
        return np.inf if variables["x"] > 0 else sphere(**variables)

    algorithm, states = run_swarm(
        ObservedPSO(), small_bounds, objective=objective, n_iterations=3
    )
    initial = [p for (t, _), p in states.items() if t == 0]
    assert sum(p.fitness == np.inf for p in initial) >= 2
    assert any(np.isfinite(p.fitness) for p in initial)
    assert_transitions(algorithm, states, small_bounds, [0.7] * 3, 1.5, 1.5)
    directions: set[tuple[bool, bool]] = set()
    for iteration, identifier in algorithm.inputs:
        before, after = states[iteration - 1, identifier], states[iteration, identifier]
        directions.add(
            (isfinite(before.personal_best_fitness), isfinite(after.fitness))
        )
    assert {(False, True), (True, False)} <= directions
    for iteration, best in algorithm.best.items():
        history = [p for (t, _), p in states.items() if t <= iteration]
        assert np.isfinite(best.fitness)
        assert best.fitness == min(p.fitness for p in history)
        assert any(
            p.variables == best.variables and p.fitness == best.fitness for p in history
        )
        assert all(np.isfinite(value) for value in best.variables.values())
    for particle in states.values():
        assert particle.velocity is not None
        assert particle.personal_best_variables is not None
        for vector in (
            particle.variables,
            particle.velocity,
            particle.personal_best_variables,
        ):
            assert all(np.isfinite(value) for value in vector.values())


def test_all_infinite_fitness_keeps_swarm_defined(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    def objective(**variables: float) -> float:
        del variables
        return np.inf

    algorithm, states = run_swarm(
        ObservedPSO(), small_bounds, objective=objective, n_iterations=2
    )
    for (iteration, identifier), particle in states.items():
        assert particle.fitness == particle.personal_best_fitness == np.inf
        assert particle.velocity is not None
        assert particle.personal_best_variables is not None
        for vector in (
            particle.variables,
            particle.velocity,
            particle.personal_best_variables,
        ):
            assert all(np.isfinite(value) for value in vector.values())
        assert any(
            p.variables == particle.personal_best_variables
            for (t, key), p in states.items()
            if t <= iteration and key == identifier
        )
    for iteration, best in algorithm.best.items():
        assert best.fitness == np.inf
        assert any(
            p.identifier == best.identifier and p.variables == best.variables
            for (t, _), p in states.items()
            if t <= iteration
        )
        assert all(np.isfinite(value) for value in best.variables.values())
    for (iteration, _), observed in algorithm.inputs.items():
        assert observed.social == algorithm.best[iteration - 1].variables
    assert any(
        states[1, key].variables != p.variables
        for (t, key), p in states.items()
        if t == 0
    ), "Infinite fitness must still permit actual movement"


def test_negative_infinite_fitness_is_preserved_as_best_possible(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    # MATHTESTS 34.3: -inf is an evaluated minimum, never the +inf sentinel.
    def objective(**variables: float) -> float:
        x = variables["x"]
        return -np.inf if -1 < x < 1 else np.inf if x > 1 else sphere(**variables)

    assert np.isposinf(FITNESS_UNDEFINED) and -np.inf != FITNESS_UNDEFINED
    algorithm, states = run_swarm(
        ObservedPSO(), small_bounds, objective=objective, n_iterations=3
    )
    initial = [p.fitness for (t, _), p in states.items() if t == 0]
    assert -np.inf in initial and np.inf in initial
    assert any(-np.inf < fitness < np.inf for fitness in initial)
    for iteration, best in algorithm.best.items():
        history = [p for (t, _), p in states.items() if t <= iteration]
        assert best.fitness == min(p.fitness for p in history) == -np.inf
        assert best.variables == algorithm.best[0].variables
        assert all(isfinite(value) for value in best.variables.values())

    promoted = False
    later_fitness: set[str] = set()
    violations: list[str] = []
    for (iteration, identifier), after in states.items():
        assert not after.new_particle
        assert after.velocity is not None and after.personal_best_variables is not None
        for vector in (after.variables, after.velocity, after.personal_best_variables):
            assert all(isfinite(value) for value in vector.values())
        history = [
            p for (t, key), p in states.items() if key == identifier and t <= iteration
        ]
        minimum = min(p.fitness for p in history)
        if after.personal_best_fitness != minimum or not any(
            p.fitness == minimum and p.variables == after.personal_best_variables
            for p in history
        ):
            violations.append(
                f"iteration={iteration}, particle={identifier}, expected={minimum}, "
                + f"pbest={after.personal_best_fitness}, current={after.fitness}"
            )
        if iteration == 0:
            continue
        before = states[iteration - 1, identifier]
        assert (
            before.velocity is not None and before.personal_best_variables is not None
        )
        observed = algorithm.inputs[iteration, identifier]
        assert observed.social == algorithm.best[iteration - 1].variables
        # Shi-Eberhart geometry uses positions, even when their fitness is -inf.
        for name, (lower, upper) in small_bounds.items():
            x = before.variables[name]
            velocity = (
                0.7 * before.velocity[name]
                + 1.5
                * observed.random[f"{name}-cognitive"]
                * (before.personal_best_variables[name] - x)
                + 1.5 * observed.random[f"{name}-social"] * (observed.social[name] - x)
            )
            assert after.velocity[name] == pytest.approx(
                velocity, rel=STRICT_RTOL, abs=STRICT_ATOL
            )
            assert after.variables[name] == pytest.approx(
                min(upper, max(lower, x + velocity)), rel=STRICT_RTOL, abs=STRICT_ATOL
            )
        if isfinite(before.personal_best_fitness) and np.isneginf(after.fitness):
            promoted = True
            assert np.isneginf(after.personal_best_fitness)
            assert after.personal_best_variables == after.variables
        if np.isneginf(before.personal_best_fitness):
            if isfinite(after.fitness):
                later_fitness.add("finite")
            elif np.isposinf(after.fitness):
                later_fitness.add("+inf")
    assert promoted and later_fitness == {"finite", "+inf"}
    assert not violations, (
        "PRODUCTION_CONTRACT_VIOLATION [PSO / -np.inf]: pbest must retain "
        + "the minimum of evaluated history; "
        + "; ".join(violations)
    )
