import numpy as np

from ..core.algorithm import (
    FITNESS_UNDEFINED,
    AlgorithmBase,
    ParticleBase,
    Serializable,
)


class DEParticle(ParticleBase):
    """Particle implementation for the Differential Evolution algorithm."""


class DifferentialEvolution(AlgorithmBase[DEParticle]):
    """Differential Evolution algorithm."""

    def __init__(
        self,
        mutation_factor: float = 0.5,
        crossover_rate: float = 0.9,
    ) -> None:
        if not isinstance(mutation_factor, (int, float, np.number)):
            raise TypeError("mutation_factor must be a number.")

        if not np.isfinite(mutation_factor):
            raise ValueError("mutation_factor must be finite.")

        if mutation_factor <= 0:
            raise ValueError("mutation_factor must be greater than 0.")

        if not isinstance(crossover_rate, (int, float, np.number)):
            raise TypeError("crossover_rate must be a number.")

        if not np.isfinite(crossover_rate):
            raise ValueError("crossover_rate must be finite.")

        if not 0 <= crossover_rate <= 1:
            raise ValueError("crossover_rate must be between 0 and 1.")

        self._mutation_factor = float(mutation_factor)
        self._crossover_rate = float(crossover_rate)

    @property
    def mutation_factor(self) -> float:
        return self._mutation_factor

    @property
    def crossover_rate(self) -> float:
        return self._crossover_rate

    def create_particle(
        self,
        identifier: str,
        variables: dict[str, float] | None = None,
        fitness: np.float64 = FITNESS_UNDEFINED,
    ) -> None:
        if identifier is None:
            raise ValueError("Particle identifier cannot be None.")

        if variables is None:
            variables = {
                name: self._rng.uniform(lower, upper)
                for name, (lower, upper) in self._boundaries.items()
            }

        self._population[identifier] = DEParticle(
            identifier=identifier,
            variables=variables,
            fitness=fitness,
        )

    def delete_particle(self, identifier: str | None) -> None:
        if identifier is None:
            raise ValueError("Particle identifier cannot be None.")

        del self._population[identifier]

    def pre_iteration(self, actual_iter: int) -> None:
        if len(self._population) < 4:
            raise ValueError("Differential Evolution requires at least 4 particles.")

    def create_random_cache(
        self,
        particle_ids: list[str],
        initialize: bool,
    ) -> None:
        if initialize:
            for identifier in particle_ids:
                self._population[identifier].random_cache = {}
            return

        variables = tuple(self._boundaries)

        for identifier in particle_ids:
            donor_ids = [
                particle_id
                for particle_id in self._population
                if particle_id != identifier
            ]

            selected_donors = self._rng.choice(
                donor_ids,
                size=3,
                replace=False,
            )

            forced_variable = variables[self._rng.integers(0, len(variables))]

            cache: dict[str, Serializable] = {
                "donor-1": str(selected_donors[0]),
                "donor-2": str(selected_donors[1]),
                "donor-3": str(selected_donors[2]),
                "forced-variable": forced_variable,
            }

            for variable in variables:
                cache[f"{variable}-crossover"] = float(self._rng.random())

            self._population[identifier].random_cache = cache

    def initialize_particle(self, identifier: str) -> DEParticle:
        particle = self._population[identifier]

        if not isinstance(particle, DEParticle):
            raise TypeError(
                f"Particle '{identifier}' must be an instance of DEParticle."
            )

        if np.isinf(particle.fitness):
            particle.update(
                variables=particle.variables,
                fitness_function=self._fitness_function,
            )

        return particle

    @staticmethod
    def consolidate_new_particles(particle: DEParticle) -> DEParticle:
        particle.consolidate(consolidate_new=True)
        return particle

    def update_particle(self, identifier: str) -> DEParticle:
        particle = self._population[identifier]

        if not isinstance(particle, DEParticle):
            raise TypeError(
                f"Particle '{identifier}' must be an instance of DEParticle."
            )

        donor_1_id = particle.random_cache["donor-1"]
        donor_2_id = particle.random_cache["donor-2"]
        donor_3_id = particle.random_cache["donor-3"]
        forced_variable = particle.random_cache["forced-variable"]

        donor_1 = self._population[donor_1_id]
        donor_2 = self._population[donor_2_id]
        donor_3 = self._population[donor_3_id]

        candidate_variables = {}

        for variable, (lower, upper) in self._boundaries.items():
            crossover_random = particle.random_cache[f"{variable}-crossover"]

            if crossover_random <= self._crossover_rate or variable == forced_variable:
                candidate_variable = donor_1.variables[variable] + (
                    self._mutation_factor
                    * (donor_2.variables[variable] - donor_3.variables[variable])
                )

                candidate_variable = np.clip(
                    candidate_variable,
                    lower,
                    upper,
                )
            else:
                candidate_variable = particle.variables[variable]

            candidate_variables[variable] = candidate_variable

        particle.update(
            variables=candidate_variables,
            fitness_function=self._fitness_function,
        )

        return particle

    def post_iteration(self, actual_iter: int) -> None:
        for particle in self._population.values():
            if not isinstance(particle, DEParticle):
                raise TypeError(
                    f"Particle '{particle.identifier}' must be an instance of "
                    "DEParticle."
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

        self.update_solution_state()
