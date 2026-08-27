from collections.abc import Callable
from copy import deepcopy

import numpy as np

from .base import (
    AlgorithmBase,
    ParticleBase,
)


class PSOParticle(ParticleBase):
    """Particle implementation for the Particle Swarm Optimization algorithm."""

    def __init__(
        self,
        identifier: str,
        fitness_function: Callable[[dict[str, float]], float],
        variables: dict[str, float],
        fitness: float | None = None,
        velocity: dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            identifier=identifier,
            fitness_function=fitness_function,
            variables=variables,
            fitness=fitness,
        )

        self._velocity = velocity

    @property
    def velocity(self) -> dict[str, float] | None:
        return self._velocity

    @velocity.setter
    def velocity(self, velocity: dict[str, float]) -> None:
        self._velocity = velocity


class PSO(AlgorithmBase):
    """Classical Particle Swarm Optimization algorithm."""

    def __init__(
        self,
        identifier: str,
        fitness_function: Callable[[dict[str, float]], float],
        boundaries: dict[str, tuple[float, float]],
        seed: int | None,
        inertia: float = 0.7,
        cognitive_coefficient: float = 1.5,
        social_coefficient: float = 1.5,
    ) -> None:
        super().__init__(
            identifier=identifier,
            fitness_function=fitness_function,
            boundaries=boundaries,
            seed=seed,
        )

        self._inertia = inertia
        self._cognitive_coefficient = cognitive_coefficient
        self._social_coefficient = social_coefficient

        self._personal_best: dict[str, ParticleBase] = {}

    def create_particle(
        self,
        identifier: str | None,
        variables: dict[str, float] | None,
        fitness: float | None,
    ) -> None:
        if identifier is None:
            raise ValueError("Particle identifier cannot be None.")

        if variables is None:
            variables = {
                name: self._rng.uniform(lower, upper)
                for name, (lower, upper) in self._boundaries.items()
            }

        velocity = {
            name: self._rng.uniform(
                -(upper - lower),
                upper - lower,
            )
            for name, (lower, upper) in self._boundaries.items()
        }

        self._population[identifier] = PSOParticle(
            identifier=identifier,
            fitness_function=self._fitness_function,
            variables=variables,
            fitness=fitness,
            velocity=velocity,
        )

    def delete_particle(self, identifier: str | None) -> None:
        if identifier is None:
            raise ValueError("Particle identifier cannot be None.")

        del self._population[identifier]

        if identifier in self._personal_best:
            del self._personal_best[identifier]

    def pre_iteration(self, actual_iter: int) -> None:
        return

    def update_particle(self, identifier: str) -> ParticleBase:
        particle = self._population[identifier]

        if not isinstance(particle, PSOParticle):
            raise TypeError(
                f"Particle '{identifier}' must be an instance of PSOParticle."
            )

        if particle.fitness is None:
            particle.update(particle.variables)
            return particle

        if self._local_best is None:
            raise RuntimeError("Local best particle has not been initialized.")

        personal_best = self._personal_best[identifier]
        global_best = self._local_best

        if particle.velocity is None:
            raise RuntimeError(f"Particle '{identifier}' does not have a velocity.")

        new_velocity = {}
        new_variables = {}

        for name, (lower, upper) in self._boundaries.items():
            current_variable = particle.variables[name]
            current_velocity = particle.velocity[name]

            personal_best_variable = personal_best.variables[name]
            global_best_variable = global_best.variables[name]

            velocity = (
                self._inertia * current_velocity
                + self._cognitive_coefficient
                * self._rng.random()
                * (personal_best_variable - current_variable)
                + self._social_coefficient
                * self._rng.random()
                * (global_best_variable - current_variable)
            )

            variable = np.clip(
                current_variable + velocity,
                lower,
                upper,
            )

            new_velocity[name] = velocity
            new_variables[name] = variable

        particle.velocity = new_velocity
        particle.update(new_variables)

        return particle

    def update_population(self, new_population: list[ParticleBase]) -> None:
        super().update_population(new_population)

    def post_iteration(self, actual_iter: int) -> None:
        if actual_iter == 0:
            for identifier, particle in self._population.items():
                particle.consolidate(new=True)
                self._personal_best[identifier] = deepcopy(particle)

            self._local_best = deepcopy(
                min(
                    self._population.values(),
                    key=lambda particle: particle.fitness,
                )
            )

            self._local_worst = deepcopy(
                max(
                    self._population.values(),
                    key=lambda particle: particle.fitness,
                )
            )

            return

        for identifier, particle in self._population.items():
            if particle.candidate_fitness is None:
                raise RuntimeError(f"Particle '{identifier}' has no candidate fitness.")

            if particle.fitness is None:
                raise RuntimeError(f"Particle '{identifier}' has no current fitness.")

            particle.consolidate(
                new=(particle.candidate_fitness < particle.fitness),
            )

            personal_best = self._personal_best[identifier]

            if (
                personal_best.fitness is None
                or particle.fitness < personal_best.fitness
            ):
                self._personal_best[identifier] = deepcopy(particle)

        best_particle = min(
            self._population.values(),
            key=lambda particle: particle.fitness,
        )

        worst_particle = max(
            self._population.values(),
            key=lambda particle: particle.fitness,
        )

        if (
            self._local_best is None
            or best_particle.fitness != self._local_best.fitness
        ):
            self._local_best = deepcopy(best_particle)

        if (
            self._local_worst is None
            or worst_particle.fitness != self._local_worst.fitness
        ):
            self._local_worst = deepcopy(worst_particle)
