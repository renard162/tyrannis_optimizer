import numpy as np

from ..core.algorithm import (
    FITNESS_UNDEFINED,
    AlgorithmBase,
    ParticleBase,
    Serializable,
)


class GreyWolfParticle(ParticleBase):
    """Particle implementation for the Grey Wolf Optimization algorithm."""


class GreyWolf(AlgorithmBase[GreyWolfParticle]):
    """Grey Wolf Optimization algorithm."""

    def __init__(self) -> None:
        """
        Grey Wolf Optimization (GWO).

        GWO is a population-based optimization algorithm inspired by the
        social hierarchy and hunting behavior of grey wolves. The three best
        solutions in the population are represented by alpha, beta, and delta
        wolves and guide the remaining wolves during the search.

        Notes
        -----
        The algorithm uses the positions of the alpha, beta, and delta wolves
        to generate the next position of every wolf. For each leader, the
        distance between the current wolf and the leader is calculated as

            D = |C * X_leader - X|

        and the corresponding candidate position is calculated as

            X_leader - A * D

        where ``A = 2 * a * r1 - a`` and ``C = 2 * r2``. The parameter ``a``
        decreases linearly from 2 to 0 throughout the optimization, causing
        the search to transition from exploration to exploitation.

        The final candidate position is the arithmetic mean of the three
        positions generated from alpha, beta, and delta. Candidate variables
        are clipped to the configured search boundaries before their fitness
        is evaluated.

        References
        ----------
        Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey Wolf
        Optimizer. Advances in Engineering Software, 69, 46-61.
        https://doi.org/10.1016/j.advengsoft.2013.12.007
        """
        self._a = 0.0
        self._alpha: str | None = None
        self._beta: str | None = None
        self._delta: str | None = None

    @property
    def alpha(self) -> str | None:
        return self._alpha

    @property
    def beta(self) -> str | None:
        return self._beta

    @property
    def delta(self) -> str | None:
        return self._delta

    @property
    def a(self) -> float:
        return self._a

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

        self._population[identifier] = GreyWolfParticle(
            identifier=identifier,
            variables=variables,
            fitness=fitness,
        )

    def delete_particle(self, identifier: str | None) -> None:
        if identifier is None:
            raise ValueError("Particle identifier cannot be None.")

        del self._population[identifier]

    def pre_iteration(self, actual_iter: int) -> None:
        if self._max_iterations <= 0:
            self._a = 0.0
            return

        iteration = min(max(actual_iter, 0), self._max_iterations)

        self._a = 2.0 - (2.0 * iteration / self._max_iterations)

    def create_random_cache(self, particle_ids: list[str], initialize: bool) -> None:
        for identifier in particle_ids:
            cache: dict[str, Serializable] = {}

            if not initialize:
                for variable in self._boundaries:
                    for leader in ("alpha", "beta", "delta"):
                        cache[f"{variable}-{leader}-r1"] = self._rng.random()
                        cache[f"{variable}-{leader}-r2"] = self._rng.random()

            self._population[identifier].random_cache = cache

    def initialize_particle(self, identifier: str) -> GreyWolfParticle:
        particle = self._population[identifier]

        if not isinstance(particle, GreyWolfParticle):
            raise TypeError(
                f"Particle '{identifier}' must be an instance of GreyWolfParticle."
            )

        if np.isinf(particle.fitness):
            particle.update(
                variables=particle.variables,
                fitness_function=self._fitness_function,
            )

        return particle

    @staticmethod
    def consolidate_new_particles(particle: GreyWolfParticle) -> GreyWolfParticle:
        particle.consolidate(consolidate_new=True)
        return particle

    def update_particle(self, identifier: str) -> GreyWolfParticle:
        particle = self._population[identifier]

        if not isinstance(particle, GreyWolfParticle):
            raise TypeError(
                f"Particle '{identifier}' must be an instance of GreyWolfParticle."
            )

        if self._alpha is None or self._beta is None or self._delta is None:
            raise RuntimeError("Grey wolf leaders have not been initialized.")

        leaders = {
            "alpha": self._population[self._alpha],
            "beta": self._population[self._beta],
            "delta": self._population[self._delta],
        }

        new_variables: dict[str, float] = {}

        for variable, (lower, upper) in self._boundaries.items():
            current_variable = particle.variables[variable]
            leader_positions = []

            for leader_name, leader in leaders.items():
                r1 = particle.random_cache[f"{variable}-{leader_name}-r1"]
                r2 = particle.random_cache[f"{variable}-{leader_name}-r2"]

                coefficient_a = 2.0 * self._a * r1 - self._a
                coefficient_c = 2.0 * r2

                distance = abs(
                    coefficient_c * leader.variables[variable] - current_variable
                )

                leader_position = leader.variables[variable] - coefficient_a * distance

                leader_positions.append(leader_position)

            variable_value = float(np.mean(leader_positions))

            new_variables[variable] = float(np.clip(variable_value, lower, upper))

        particle.update(
            variables=new_variables,
            fitness_function=self._fitness_function,
        )

        return particle

    def post_iteration(self, actual_iter: int) -> None:
        if actual_iter > 0:
            for particle in self._population.values():
                if not isinstance(particle, GreyWolfParticle):
                    raise TypeError(
                        f"Particle '{particle.identifier}' must be an instance "
                        "of GreyWolfParticle."
                    )

                if particle.candidate_fitness is None:
                    raise RuntimeError(
                        f"Particle '{particle.identifier}' has no candidate fitness."
                    )

                particle.consolidate(consolidate_new=True)

        self.update_solution_state()

        particles = sorted(
            self._population.values(),
            key=lambda particle: particle.fitness,
        )

        if len(particles) < 3:
            raise RuntimeError(
                "Grey Wolf Optimization requires at least three particles."
            )

        self._alpha = particles[0].identifier
        self._beta = particles[1].identifier
        self._delta = particles[2].identifier
