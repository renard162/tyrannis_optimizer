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
        """
        Differential Evolution algorithm (DE).

        Differential Evolution is a population-based, derivative-free optimization
        algorithm for continuous problems that generates candidate solutions from
        scaled differences between randomly selected individuals. These differential
        variations adapt naturally to the distribution of the population, allowing
        the algorithm to explore the search space while progressively concentrating
        the search as the population converges.

        Parameters
        ----------
        mutation_factor : float, default=0.5
            Scaling factor applied to the differential variation between individuals.
            Larger values increase the magnitude of the generated variations and
            generally favor exploration, while smaller values produce more conservative
            movements and favor exploitation. The value must be greater than 0 and
            less than or equal to 2.

        crossover_rate : float, default=0.9
            Probability of selecting each variable from the mutant vector during
            crossover. Higher values allow the differential mutation to affect a larger
            portion of each candidate solution, while lower values preserve more
            variables from the current solution. The value must be between 0 and 1.

        Notes
        -----
        Differential Evolution generates new candidate solutions by combining
        individuals from the current population. This implementation follows the
        DE/rand/1/bin strategy, in which three distinct individuals are randomly
        selected for each target individual. A mutant vector is generated as

            v = x_r1 + F * (x_r2 - x_r3)

        where ``x_r1`` is the randomly selected base vector, ``x_r2`` and ``x_r3``
        define the differential variation, and ``F`` is the ``mutation_factor``.

        The difference ``x_r2 - x_r3`` provides both a search direction and a scale
        derived directly from the current population. When the population is widely
        distributed, these differences tend to be larger and produce broader search
        steps. As the population becomes more concentrated, the differences generally
        decrease, naturally reducing the scale of the search.

        The mutant vector is combined with the target individual through binomial
        crossover. For each variable, the mutant value is selected with probability
        ``crossover_rate``; otherwise, the value from the target individual is
        retained. At least one variable is always selected from the mutant vector,
        ensuring that the resulting trial solution differs from the target through
        the differential mutation.

        After evaluation, the trial solution competes directly with its corresponding
        target individual. The trial solution replaces the target only when it
        provides a better objective value. Differential Evolution therefore preserves
        successful solutions while continuously generating alternatives from the
        relative positions of individuals in the population.

        The ``mutation_factor`` and ``crossover_rate`` jointly control the balance
        between exploration and exploitation. Increasing ``mutation_factor`` produces
        larger differential steps, while increasing ``crossover_rate`` allows those
        steps to affect more dimensions simultaneously. Problems with strongly
        interacting variables may benefit from higher crossover rates, whereas lower
        rates can provide more conservative changes when useful solutions depend on
        preserving individual variable values.

        The DE/rand/1 strategy requires a minimum population of four individuals:
        one target individual and three distinct individuals used to generate its
        mutant vector. Larger populations generally provide greater diversity and
        more differential directions at the cost of additional objective function
        evaluations.

        Differential Evolution is designed for continuous optimization and is
        particularly suitable for non-linear, non-convex, multi-modal, or
        non-differentiable objective functions.

        References
        ----------
        Storn, R., & Price, K. (1997). Differential Evolution - A Simple and Efficient
        Heuristic for Global Optimization over Continuous Spaces. Journal of Global
        Optimization, 11, 341-359.
        https://doi.org/10.1023/A:1008202821328

        Das, S., & Suganthan, P. N. (2011). Differential Evolution: A Survey of the
        State-of-the-Art. IEEE Transactions on Evolutionary Computation, 15(1), 4-31.
        https://doi.org/10.1109/TEVC.2010.2059031

        Zhang, M., Luo, W., & Wang, X. (2010). Two hybrid differential evolution
        algorithms for engineering design optimization. Applied Soft Computing,
        10(4), 1188-1199.
        https://doi.org/10.1016/j.asoc.2010.05.007

        Cai, H. R., Chung, C. Y., & Wong, K. P. (2008). Application of Differential
        Evolution Algorithm for Transient Stability Constrained Optimal Power Flow.
        IEEE Transactions on Power Systems, 23(2), 719-728.
        https://doi.org/10.1109/TPWRS.2008.919241
        """
        if not isinstance(mutation_factor, (int, float, np.number)):
            raise TypeError("mutation_factor must be a number.")

        if not np.isfinite(mutation_factor):
            raise ValueError("mutation_factor must be finite.")

        if not 0 < mutation_factor <= 2:
            raise ValueError("mutation_factor must be greater than 0 and at most 2.")

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
            identifier=identifier, variables=variables, fitness=fitness
        )

    def delete_particle(self, identifier: str | None) -> None:
        if identifier is None:
            raise ValueError("Particle identifier cannot be None.")

        del self._population[identifier]

    def pre_iteration(self, actual_iter: int) -> None:
        if len(self._population) < 4:
            raise ValueError("Differential Evolution requires at least 4 particles.")

    def create_random_cache(self, particle_ids: list[str], initialize: bool) -> None:
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

            selected_donors = self._rng.choice(donor_ids, size=3, replace=False)

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

        if particle.fitness == FITNESS_UNDEFINED:
            particle.update(
                variables=particle.variables, fitness_function=self._fitness_function
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

                candidate_variable = np.clip(candidate_variable, lower, upper)
            else:
                candidate_variable = particle.variables[variable]

            candidate_variables[variable] = candidate_variable

        particle.update(
            variables=candidate_variables, fitness_function=self._fitness_function
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
