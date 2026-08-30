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
        variables: dict[str, float],
        fitness: float | None = None,
        velocity: dict[str, float] | None = None,
        personal_best_variables: dict[str, float] | None = None,
        personal_best_fitness: float | None = None,
    ) -> None:
        super().__init__(
            identifier=identifier,
            variables=variables,
            fitness=fitness,
        )

        self._velocity = velocity
        self._personal_best_variables = personal_best_variables
        self._personal_best_fitness = personal_best_fitness

    @property
    def velocity(self) -> dict[str, float] | None:
        return self._velocity

    @velocity.setter
    def velocity(self, velocity: dict[str, float]) -> None:
        self._velocity = velocity

    @property
    def personal_best_variables(self) -> dict[str, float] | None:
        return self._personal_best_variables

    @property
    def personal_best_fitness(self) -> float | None:
        return self._personal_best_fitness

    def __call__(self) -> dict[str, str | dict[str, float] | float | None]:
        return {
            "identifier": self._identifier,
            "variables": self._variables,
            "fitness": self._fitness,
            "personal_best_variables": self._personal_best_variables,
            "personal_best_fitness": self._personal_best_fitness,
        }

    def update_personal_best(self) -> None:
        if self.fitness is None:
            raise RuntimeError(f"Particle '{self.identifier}' does not have a fitness.")

        if (
            self._personal_best_fitness is None
            or self.fitness < self._personal_best_fitness
        ):
            self._personal_best_variables = deepcopy(self.variables)
            self._personal_best_fitness = self.fitness


class PSO(AlgorithmBase):
    """Classical Particle Swarm Optimization algorithm."""

    def __init__(
        self,
        inertia: float = 0.7,
        cognitive_coefficient: float = 1.5,
        social_coefficient: float = 1.5,
    ) -> None:
        self._inertia = inertia
        self._cognitive_coefficient = cognitive_coefficient
        self._social_coefficient = social_coefficient

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
            variables=variables,
            fitness=fitness,
            velocity=velocity,
        )

    def delete_particle(self, identifier: str | None) -> None:
        if identifier is None:
            raise ValueError("Particle identifier cannot be None.")

        del self._population[identifier]

    def pre_iteration(self, actual_iter: int) -> None:
        return

    def initialize_particle(self, identifier: str) -> ParticleBase:
        particle = self._population[identifier]

        if not isinstance(particle, PSOParticle):
            raise TypeError(
                f"Particle '{identifier}' must be an instance of PSOParticle."
            )

        if particle.fitness is not None:
            return particle

        particle.update(
            variables=particle.variables,
            fitness_function=self._fitness_function,
        )
        particle.consolidate(consolidate_new=True)
        particle.update_personal_best()

        return particle

    def update_particle(self, identifier: str) -> ParticleBase:
        particle = self._population[identifier]

        if not isinstance(particle, PSOParticle):
            raise TypeError(
                f"Particle '{identifier}' must be an instance of PSOParticle."
            )

        if self._local_best is None:
            raise RuntimeError("Local best particle has not been initialized.")

        if particle.fitness is None:
            raise RuntimeError(f"Particle '{identifier}' does not have a fitness.")

        if particle.velocity is None:
            raise RuntimeError(f"Particle '{identifier}' does not have a velocity.")

        if particle.personal_best_variables is None:
            raise RuntimeError(
                f"Particle '{identifier}' does not have a personal best."
            )

        current_variables = particle.variables
        personal_best_variables = particle.personal_best_variables
        global_best_variables = self._local_best.variables

        new_velocity = {}
        new_variables = {}

        for name, (lower, upper) in self._boundaries.items():
            current_variable = current_variables[name]
            current_velocity = particle.velocity[name]

            personal_best_variable = personal_best_variables[name]
            global_best_variable = global_best_variables[name]

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

        particle.update(
            variables=new_variables,
            fitness_function=self._fitness_function,
        )

        return particle


def post_iteration(self, actual_iter: int) -> None:
    for particle in self._population.values():
        if not isinstance(particle, PSOParticle):
            raise TypeError(
                f"Particle '{particle.identifier}' must be an instance of PSOParticle."
            )

        if actual_iter == 0:
            continue

        if particle.candidate_fitness is None:
            raise RuntimeError(
                f"Particle '{particle.identifier}' has no candidate fitness."
            )

        if particle.fitness is None:
            raise RuntimeError(
                f"Particle '{particle.identifier}' has no current fitness."
            )

        particle.consolidate(
            consolidate_new=particle.candidate_fitness < particle.fitness,
        )
        particle.update_personal_best()

    self.update_solution_state()
