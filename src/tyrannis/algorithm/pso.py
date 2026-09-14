from collections.abc import Collection
from copy import deepcopy

import numpy as np

from ..core.algorithm import (
    FITNESS_UNDEFINED,
    AlgorithmBase,
    ParticleBase,
    Serializable,
)


class PSOParticle(ParticleBase):
    """Particle implementation for the Particle Swarm Optimization algorithm."""

    def __init__(
        self,
        identifier: str,
        variables: dict[str, float],
        fitness: np.float64 = FITNESS_UNDEFINED,
        velocity: dict[str, float] | None = None,
        personal_best_variables: dict[str, float] | None = None,
        personal_best_fitness: np.float64 = FITNESS_UNDEFINED,
    ) -> None:
        super().__init__(identifier=identifier, variables=variables, fitness=fitness)

        self._velocity = velocity
        self._personal_best_variables = personal_best_variables
        self._personal_best_fitness = personal_best_fitness

    def __call__(self) -> dict[str, Serializable]:
        return {
            "identifier": self._identifier,
            "variables": self._variables,
            "fitness": self._fitness,
            "velocity": self._velocity,
            "personal_best_variables": self._personal_best_variables,
            "personal_best_fitness": self._personal_best_fitness,
        }

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
    def personal_best_fitness(self) -> np.float64:
        return self._personal_best_fitness

    def update_personal_best(self) -> None:
        if (
            np.isinf(self._personal_best_fitness)
            or self.fitness < self._personal_best_fitness
        ):
            self._personal_best_variables = deepcopy(self.variables)
            self._personal_best_fitness = self.fitness


class PSO(AlgorithmBase):
    """Classical Particle Swarm Optimization algorithm."""

    def __init__(
        self,
        inertia: float | Collection[float] = 0.7,
        cognitive_coefficient: float = 1.5,
        social_coefficient: float = 1.5,
        constriction_factor: bool = False,
    ) -> None:
        if constriction_factor and (cognitive_coefficient + social_coefficient < 4):
            raise ValueError(
                "The sum of cognitive_coefficient and social_coefficient "
                "must be greater than or equal to 4 to use the constriction "
                "factor."
            )

        if isinstance(inertia, Collection) and not isinstance(inertia, (str, bytes)):
            if len(inertia) != 2:
                raise ValueError("Dynamic inertia must contain exactly two values.")

            inertia_values = tuple(inertia)

            if not all(
                isinstance(value, (int, float, np.number)) and np.isfinite(value)
                for value in inertia_values
            ):
                raise ValueError("Dynamic inertia values must be finite numbers.")

            if inertia_values[0] >= inertia_values[1]:
                raise ValueError(
                    "The first dynamic inertia value must be smaller than the second."
                )

            self._inertia = float(inertia_values[1])
            self._inertia_bounds = (float(inertia_values[0]), float(inertia_values[1]))
        elif isinstance(inertia, (int, float, np.number)):
            if not np.isfinite(inertia):
                raise ValueError("Inertia must be a finite number.")

            self._inertia = float(inertia)
            self._inertia_bounds = None
        else:
            raise TypeError(
                "Inertia must be a number or an ordered collection "
                "containing two numbers."
            )

        self._cognitive_coefficient = cognitive_coefficient
        self._social_coefficient = social_coefficient
        self._constriction_factor = constriction_factor

        if constriction_factor:
            phi = cognitive_coefficient + social_coefficient
            self._constriction = 2 / abs(2 - phi - np.sqrt(phi**2 - 4 * phi))
            self._inertia = 1.0
        else:
            self._constriction = 1.0

    def create_particle(
        self,
        identifier: str,
        variables: dict[str, float] | None = None,
        fitness: np.float64 = FITNESS_UNDEFINED,
        velocity: dict[str, float] | None = None,
        personal_best_variables: dict[str, float] | None = None,
        personal_best_fitness: np.float64 = FITNESS_UNDEFINED,
    ) -> None:
        if identifier is None:
            raise ValueError("Particle identifier cannot be None.")

        if variables is None:
            variables = {
                name: self._rng.uniform(lower, upper)
                for name, (lower, upper) in self._boundaries.items()
            }

        if velocity is None:
            velocity = {
                name: self._rng.uniform(-(upper - lower), upper - lower)
                for name, (lower, upper) in self._boundaries.items()
            }

        self._population[identifier] = PSOParticle(
            identifier=identifier,
            variables=variables,
            fitness=fitness,
            velocity=velocity,
            personal_best_variables=personal_best_variables,
            personal_best_fitness=personal_best_fitness,
        )

    def delete_particle(self, identifier: str | None) -> None:
        if identifier is None:
            raise ValueError("Particle identifier cannot be None.")

        del self._population[identifier]

    def pre_iteration(self, actual_iter: int) -> None:
        if self._constriction_factor or self._inertia_bounds is None:
            return

        minimum_inertia, maximum_inertia = self._inertia_bounds
        iteration = max(actual_iter, 1)

        self._inertia = maximum_inertia - (
            (maximum_inertia - minimum_inertia) * (iteration / self._max_iterations)
        )

    def create_random_cache(self, particle_ids: list[str], initialize: bool) -> None:
        for particle_id in particle_ids:
            particle = self._population[particle_id]

            if len(particle.random_cache) >= 2 * len(self._boundaries):
                continue

            particle.random_cache = []
            for _ in self._boundaries:
                cognitive_random = self._rng.random()
                social_random = self._rng.random()

                particle.random_cache.extend((cognitive_random, social_random))

    def initialize_particle(self, identifier: str) -> PSOParticle:
        particle = self._population[identifier]

        if not isinstance(particle, PSOParticle):
            raise TypeError(
                f"Particle '{identifier}' must be an instance of PSOParticle."
            )

        if np.isinf(particle.fitness):
            particle.update(
                variables=particle.variables, fitness_function=self._fitness_function
            )

        return particle

    @staticmethod
    def consolidate_new_particles(particle: PSOParticle) -> PSOParticle:
        particle.consolidate(consolidate_new=True)
        particle.update_personal_best()
        return particle

    def update_particle(self, identifier: str) -> PSOParticle:
        particle = self._population[identifier]

        if not isinstance(particle, PSOParticle):
            raise TypeError(
                f"Particle '{identifier}' must be an instance of PSOParticle."
            )

        if self._local_best is None:
            raise RuntimeError("Local best particle has not been initialized.")

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

            cognitive_random = particle.random_cache.pop(0)
            social_random = particle.random_cache.pop(0)

            velocity = self._constriction * (
                self._inertia * current_velocity
                + self._cognitive_coefficient
                * cognitive_random
                * (personal_best_variable - current_variable)
                + self._social_coefficient
                * social_random
                * (global_best_variable - current_variable)
            )

            variable = np.clip(current_variable + velocity, lower, upper)

            new_velocity[name] = velocity
            new_variables[name] = variable

        particle.velocity = new_velocity

        particle.update(
            variables=new_variables, fitness_function=self._fitness_function
        )

        return particle

    def post_iteration(self, actual_iter: int) -> None:
        for particle in self._population.values():
            if not isinstance(particle, PSOParticle):
                raise TypeError(
                    f"Particle '{particle.identifier}' must be an instance of "
                    "PSOParticle."
                )

            if actual_iter == 0:
                continue

            if particle.candidate_fitness is None:
                raise RuntimeError(
                    f"Particle '{particle.identifier}' has no candidate fitness."
                )

            particle.consolidate(
                consolidate_new=bool(particle.candidate_fitness < particle.fitness)
            )
            particle.update_personal_best()

        self.update_solution_state()
