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

    def __init__(
        self,
        convergence_exponent: float = 1,
        exploration_enhanced: bool = True,
    ) -> None:
        """
        Grey Wolf Optimization (GWO).

        GWO is a population-based optimization algorithm inspired by the
        social hierarchy and hunting behavior of grey wolves. The three best
        solutions in the population are represented by alpha, beta, and delta
        wolves and guide the remaining wolves during the search.

        Parameters
        ----------
        convergence_exponent : float, default=1
            Exponent controlling the nonlinear convergence schedule of the
            parameter ``a``. The parameter is calculated as

                a = 2 * (1 - (t / T) ** convergence_exponent)

            where ``t`` is the current iteration and ``T`` is the maximum
            number of iterations. A value of ``1`` produces the linear
            convergence schedule of the original GWO.

        exploration_enhanced : bool, default=True
            Whether to use the exploration-enhanced position update proposed
            by EEGWO. When ``False``, the original GWO position update based
            exclusively on alpha, beta, and delta wolves is used.

        Notes
        -----
        The original GWO uses the positions of the alpha, beta, and delta
        wolves to generate the next position of every wolf. For each leader,
        the distance between the current wolf and the leader is calculated as

            D = |C * X_leader - X|

        and the corresponding candidate position is calculated as

            X_leader - A * D

        where ``A = 2 * a * r1 - a`` and ``C = 2 * r2``.

        When ``exploration_enhanced`` is enabled, the position update is
        replaced by the exploration-enhanced equation proposed by EEGWO. An
        additional individual is randomly selected from the population and
        contributes a direction based on the difference between its position
        and the current wolf position. The published EEGWO coefficients are
        ``b1 = 0.1`` and ``b2 = 0.9``.

        References
        ----------
        Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey Wolf
        Optimizer. Advances in Engineering Software, 69, 46-61.
        https://doi.org/10.1016/j.advengsoft.2013.12.007

        Long, W., Jiao, J., Liang, X., & Cai, S. (2018). An
        exploration-enhanced grey wolf optimizer to solve high-dimensional
        numerical optimization. Engineering Applications of Artificial
        Intelligence, 68, 63-80.
        https://doi.org/10.1016/j.engappai.2017.10.024
        """
        if convergence_exponent <= 0:
            raise ValueError("convergence_exponent must be greater than zero.")

        self._convergence_exponent = convergence_exponent
        self._exploration_enhanced = exploration_enhanced

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
        progress = iteration / self._max_iterations

        self._a = 2.0 * (1.0 - progress**self._convergence_exponent)

    def create_random_cache(self, particle_ids: list[str], initialize: bool) -> None:
        for identifier in particle_ids:
            cache: dict[str, Serializable] = {}

            if not initialize:
                for variable in self._boundaries:
                    for leader in ("alpha", "beta", "delta"):
                        cache[f"{variable}-{leader}-r1"] = self._rng.random()
                        cache[f"{variable}-{leader}-r2"] = self._rng.random()

                if self._exploration_enhanced:
                    cache["r3"] = self._rng.random()
                    cache["r4"] = self._rng.random()
                    cache["random_particle"] = self._rng.integers(
                        0,
                        len(self._population),
                    )

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

        random_particle = None

        if self._exploration_enhanced:
            random_index = int(particle.random_cache["random_particle"])
            random_identifier = list(self._population)[random_index]
            random_particle = self._population[random_identifier]

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

            leader_position = float(np.mean(leader_positions))

            if self._exploration_enhanced:
                if random_particle is None:
                    raise RuntimeError("Random particle has not been initialized.")

                random_direction = (
                    random_particle.variables[variable] - current_variable
                )

                r3 = particle.random_cache["r3"]
                r4 = particle.random_cache["r4"]

                variable_value = (
                    0.1 * r3 * leader_position + 0.9 * r4 * random_direction
                )
            else:
                variable_value = leader_position

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
