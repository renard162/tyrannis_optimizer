"""PSO equations through the real Optimizer/Continuous/Serial lifecycle.

Sources: Kennedy & Eberhart (1995), doi:10.1109/ICNN.1995.488968;
Shi & Eberhart (1998), eq. (2), doi:10.1109/ICEC.1998.699146;
Clerc & Kennedy (2002), constriction, doi:10.1109/4235.985692.
Clipping, synchronous sweeps and inclusive update endpoints are Tyrannis contracts.
"""

import json
from copy import deepcopy
from dataclasses import dataclass
from math import sqrt
from typing import override

import pytest

from tests._support import factories
from tests._support.assertions import (
    assert_optimizer_result_consistent,
    assert_particle_state_consistent,
)
from tests._support.numerics import BASE_SEED, STRICT_ATOL, STRICT_RTOL
from tests._support.objectives import sphere
from tyrannis.algorithm.pso import PSO, PSOParticle
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


def run_swarm(
    algorithm: ObservedPSO,
    bounds: dict[str, tuple[float, float]],
    *,
    n_iterations: int = 5,
) -> tuple[ObservedPSO, dict[tuple[int, str], PSOParticle]]:
    n_particles = 5
    processor = Serial()
    optimizer = factories.make_optimizer(
        Continuous(bounds, cost_function=sphere),
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
            sphere(**particle.variables), rel=STRICT_RTOL, abs=STRICT_ATOL
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
