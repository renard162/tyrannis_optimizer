from collections.abc import Collection
from copy import deepcopy

import numpy as np

from ..core.algorithm import (
    FITNESS_UNDEFINED,
    AlgorithmBase,
    ParticleBase,
    Serializable,
)


class ACORParticle(ParticleBase):
    """Particle implementation for the Ant Colony Optimization for Continuous Domain algorithm."""

    def __init__(
        self,
        identifier: str,
        variables: dict[str, float],
        fitness: np.float64 = FITNESS_UNDEFINED,
    ) -> None:
        super().__init__(identifier=identifier, variables=variables, fitness=fitness)

    def __call__(self) -> dict[str, Serializable]:
        return {
            "identifier": self._identifier,
            "variables": self._variables,
            "fitness": self._fitness,
        }


class AntColony(AlgorithmBase[ACORParticle]):
    """Ant Colony Optimization for Continuous Domains."""

    def __init__(
        self,
        archive_size: int = 10,
        q: float = 0.5,
        xi: float = 0.85,
    ) -> None:
        """
        Ant Colony Optimization for Continuous Domains (ACOR).

        ACOR is a continuous-domain extension of the Ant Colony Optimization (ACO)
        metaheuristic, originally developed for combinatorial optimization. Instead of
        using pheromone values associated with discrete solution components, ACOR
        represents pheromone information through an archive of promising continuous
        solutions and generates new candidate solutions by sampling probability
        distributions around them.

        Parameters
        ----------
        archive_size : int, default=10
            Number of solutions maintained in the solution archive. The archive stores
            the best solutions found by the colony and provides the probability model
            used to generate new solutions.

        q : float, default=0.5
            Controls the concentration of the probability assigned to solutions in the
            archive according to their rank. Smaller values increase the preference
            for the best solutions, while larger values distribute the probability more
            evenly across the archive.

        xi : float, default=0.85
            Controls the standard deviation of the probability distributions used to
            generate new solutions. Larger values increase the search range around
            archive solutions, while smaller values concentrate the search around them.

        Notes
        -----
        Ant Colony Optimization for Continuous Domains (ACOR) adapts the main
        principles of Ant Colony Optimization to continuous search spaces. Instead of
        associating pheromone values with discrete solution components, ACOR uses an
        archive of solutions as its pheromone representation. The archive is sorted
        according to solution quality, with better solutions receiving greater
        probability of being selected as references for generating new solutions.

        For each new solution, a solution is selected from the archive according to
        its rank-based probability. Each decision variable is then sampled from a
        normal distribution centered on the corresponding variable of the selected
        solution. The standard deviation of each distribution is calculated from the
        dispersion of that variable among the solutions in the archive and scaled by
        ``xi``. Consequently, the search distribution adapts to the concentration of
        the solutions: dispersed archive solutions produce broader exploration, while
        concentrated solutions produce more localized search.

        The parameter ``q`` controls the selection pressure applied to the archive.
        Values that concentrate probability on the best-ranked solutions increase
        exploitation, while more evenly distributed probabilities preserve greater
        exploration of the solutions stored in the archive.

        After new solutions are evaluated, they are combined with the existing archive
        and the best ``archive_size`` solutions are retained. The archive therefore
        acts as both the memory of the colony and the basis for generating subsequent
        solutions.

        The initial archive is constructed from the solutions generated during the
        algorithm initialization. When the archive is larger than the population,
        temporary particles are created only to provide the additional solutions
        required to initialize the archive. These particles are removed after the
        initialization, while the official population particles retain their original
        identifiers.

        References
        ----------
        Socha, K., & Dorigo, M. (2008). Ant colony optimization for continuous
        domains. European Journal of Operational Research, 185(3), 1155-1173.
        https://doi.org/10.1016/j.ejor.2006.06.046

        Dorigo, M., & Gambardella, L. M. (1997). Ant colony system: A cooperative
        learning approach to the traveling salesman problem. IEEE Transactions on
        Evolutionary Computation, 1(1), 53-66.
        https://doi.org/10.1109/4235.585892

        Afshar, A., & Madadgar, S. (2008). Ant Colony Optimization for Continuous
        Domains: Application to Reservoir Operation Problems. 2008 Eighth
        International Conference on Hybrid Intelligent Systems.
        https://doi.org/10.1109/HIS.2008.121

        Moradi, B., Kargar, A., & Abazari, S. (2022). Transient stability constrained
        optimal power flow solution using ant colony optimization for continuous
        domains (ACOR). IET Generation, Transmission & Distribution, 16(18),
        3734-3747.
        https://doi.org/10.1049/gtd2.12560
        """
        if not isinstance(archive_size, (int, np.integer)):
            raise TypeError("archive_size must be an integer.")

        if archive_size < 2:
            raise ValueError("archive_size must be greater than or equal to 2.")

        if not isinstance(q, (int, float, np.number)):
            raise TypeError("q must be a number.")

        if not np.isfinite(q) or q <= 0:
            raise ValueError("q must be a finite number greater than 0.")

        if not isinstance(xi, (int, float, np.number)):
            raise TypeError("xi must be a number.")

        if not np.isfinite(xi) or xi <= 0:
            raise ValueError("xi must be a finite number greater than 0.")

        self._archive_size = int(archive_size)
        self._q = float(q)
        self._xi = float(xi)

        self._solution_archive: list[tuple[dict[str, float], np.float64]] = []
        self._archive_probabilities: np.ndarray | None = None

        self._official_particle_ids: list[str] = []
        self._temporary_particle_ids: list[str] = []

    @property
    def archive_size(self) -> int:
        return self._archive_size

    @property
    def q(self) -> float:
        return self._q

    @property
    def xi(self) -> float:
        return self._xi

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

        self._population[identifier] = ACORParticle(
            identifier=identifier, variables=variables, fitness=fitness
        )

    def delete_particle(self, identifier: str | None) -> None:
        if identifier is None:
            raise ValueError("Particle identifier cannot be None.")

        del self._population[identifier]

    def _create_temporary_particle_ids(self, count: int) -> list[str]:
        identifiers = []
        index = 0

        while len(identifiers) < count:
            identifier = f"{self._identifier}|acor-initial:{index}"
            index += 1

            if identifier in self._population:
                continue

            identifiers.append(identifier)

        return identifiers

    def pre_iteration(self, actual_iter: int) -> None:
        if actual_iter == 0:
            self._official_particle_ids = list(self._population)
            missing = max(0, self._archive_size - len(self._official_particle_ids))
            self._temporary_particle_ids = self._create_temporary_particle_ids(missing)

            for identifier in self._temporary_particle_ids:
                self.create_particle(identifier=identifier)

            return

        if actual_iter == 1:
            for identifier in self._official_particle_ids:
                self.delete_particle(identifier)
                self.create_particle(identifier=identifier)

            self._archive_probabilities = self._calculate_archive_probabilities()

    def _calculate_archive_probabilities(self) -> np.ndarray:
        positions = np.arange(self._archive_size, dtype=float)
        weights = np.exp(-(positions**2) / (2 * self._q**2 * self._archive_size**2))
        return weights / np.sum(weights)

    def create_random_cache(self, particle_ids: list[str], initialize: bool) -> None:
        for identifier in particle_ids:
            particle = self._population[identifier]
            particle.random_cache = {}

            if initialize:
                continue

            if self._archive_probabilities is None:
                raise RuntimeError("Archive probabilities have not been initialized.")

            particle.random_cache["archive-index"] = int(
                self._rng.choice(self._archive_size, p=self._archive_probabilities)
            )

            for variable in self._boundaries:
                particle.random_cache[f"{variable}-normal"] = float(self._rng.normal())

    def initialize_particle(self, identifier: str) -> ACORParticle:
        particle = self._population[identifier]

        if not isinstance(particle, ACORParticle):
            raise TypeError(
                f"Particle '{identifier}' must be an instance of ACORParticle."
            )

        if np.isinf(particle.fitness):
            particle.update(
                variables=particle.variables, fitness_function=self._fitness_function
            )

        return particle

    @staticmethod
    def consolidate_new_particles(particle: ACORParticle) -> ACORParticle:
        particle.consolidate(consolidate_new=True)
        return particle

    def _calculate_sigma(self, archive_index: int, variable: str) -> float:
        reference_value = self._solution_archive[archive_index][0][variable]

        distances = sum(
            abs(solution[variable] - reference_value)
            for solution, _ in self._solution_archive
        )

        return self._xi * distances / (self._archive_size - 1)

    def update_particle(self, identifier: str) -> ACORParticle:
        particle = self._population[identifier]

        if not isinstance(particle, ACORParticle):
            raise TypeError(
                f"Particle '{identifier}' must be an instance of ACORParticle."
            )

        archive_index = particle.random_cache["archive-index"]
        reference_variables = self._solution_archive[archive_index][0]

        new_variables = {}
        for variable in self._boundaries:
            normal_value = particle.random_cache[f"{variable}-normal"]

            sigma = self._calculate_sigma(
                archive_index=archive_index, variable=variable
            )

            lower, upper = self._boundaries[variable]
            value = reference_variables[variable] + sigma * normal_value
            new_variables[variable] = float(np.clip(value, lower, upper))

        particle.update(
            variables=new_variables, fitness_function=self._fitness_function
        )

        return particle

    def _initialize_solution_archive(self, particles: Collection[ACORParticle]) -> None:
        candidates = sorted(
            (
                (deepcopy(particle.variables), particle.fitness)
                for particle in particles
            ),
            key=lambda solution: solution[1],
        )

        self._solution_archive = candidates[: self._archive_size]

    def _update_solution_archive(self, particles: Collection[ACORParticle]) -> None:
        candidates = [
            (deepcopy(particle.variables), particle.fitness) for particle in particles
        ]

        candidates.extend(self._solution_archive)
        candidates.sort(key=lambda solution: solution[1])
        self._solution_archive = candidates[: self._archive_size]

    def post_iteration(self, actual_iter: int) -> None:
        if actual_iter == 0:
            particles = list(self._population.values())

            if not all(isinstance(particle, ACORParticle) for particle in particles):
                raise TypeError("All particles must be instances of ACORParticle.")

            self._initialize_solution_archive(particles)

            for identifier in self._temporary_particle_ids:
                self.delete_particle(identifier)

            self.update_solution_state()
            return

        for particle in self._population.values():
            if not isinstance(particle, ACORParticle):
                raise TypeError(
                    f"Particle '{particle.identifier}' must be an instance of "
                    "ACORParticle."
                )

            if particle.candidate_fitness is None:
                raise RuntimeError(
                    f"Particle '{particle.identifier}' has no candidate fitness."
                )

            particle.consolidate(consolidate_new=True)

        self._update_solution_archive(list(self._population.values()))
        self.update_solution_state()
