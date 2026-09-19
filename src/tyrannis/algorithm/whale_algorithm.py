import numpy as np

from ..core.algorithm import (
    FITNESS_UNDEFINED,
    AlgorithmBase,
    ParticleBase,
    Serializable,
)


class WhaleParticle(ParticleBase):
    """Particle for Whale Optimization Algorithm."""


class WhaleAlgorithm(AlgorithmBase[WhaleParticle]):
    """Whale Optimization Algorithm."""

    def __init__(
        self,
        spiral_coefficient: float = 1.0,
    ) -> None:
        """
        Whale Optimization Algorithm.

        WOA is a population-based, derivative-free optimization algorithm inspired
        by the hunting behavior of humpback whales. The search alternates between
        exploration around randomly selected whales, exploitation around the best
        solution, and a spiral-shaped movement that models the bubble-net feeding
        strategy.

        Parameters
        ----------
        spiral_coefficient : float, default=1.0
            Positive coefficient controlling the shape of the spiral movement used
            during the bubble-net feeding phase.

        Notes
        -----
        WOA maintains the best solution found by the population as the reference
        position of the prey. For each whale, a random value ``p`` selects between
        the shrinking-encircling/search behavior and the spiral movement.

        When ``p < 0.5``, the coefficient

            A = 2 * a * r1 - a

        determines whether the whale performs exploitation or exploration. When
        ``|A| < 1``, the whale moves toward the best solution according to

            D = |C * X_best - X|

            X(t + 1) = X_best - A * D

        where ``C = 2 * r2``. When ``|A| >= 1``, a randomly selected whale is used
        instead of the best solution:

            D = |C * X_random - X|

            X(t + 1) = X_random - A * D

        When ``p >= 0.5``, the whale follows the spiral movement:

            D = |X_best - X|

            X(t + 1) = D * exp(b * l) * cos(2 * pi * l) + X_best

        where ``b`` is ``spiral_coefficient`` and ``l`` is uniformly sampled from
        ``[-1, 1]``.

        The convergence parameter ``a`` decreases linearly from 2 to 0 over the
        optimization iterations. Iteration 0 is reserved for initialization and
        therefore uses ``a = 2``; the first optimization iteration uses the first
        point of the decreasing schedule.

        References
        ----------
        Mirjalili, S., & Lewis, A. (2016). The Whale Optimization Algorithm.
        Advances in Engineering Software, 95, 51-67.
        https://doi.org/10.1016/j.advengsoft.2016.01.008
        """
        if not isinstance(spiral_coefficient, (int, float, np.number)):
            raise TypeError("spiral_coefficient must be a number.")

        if not np.isfinite(spiral_coefficient) or spiral_coefficient <= 0:
            raise ValueError(
                "spiral_coefficient must be a finite number greater than 0."
            )

        self._spiral_coefficient = float(spiral_coefficient)
        self._a = 2.0

    @property
    def spiral_coefficient(self) -> float:
        return self._spiral_coefficient

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

        self._population[identifier] = WhaleParticle(
            identifier=identifier, variables=variables, fitness=fitness
        )

    def delete_particle(self, identifier: str | None) -> None:
        if identifier is None:
            raise ValueError("Particle identifier cannot be None.")

        del self._population[identifier]

    def pre_iteration(self, actual_iter: int) -> None:
        if actual_iter == 0:
            self._a = 2.0
            return

        if self._max_iterations <= 0:
            self._a = 0.0
            return

        iteration = min(actual_iter, self._max_iterations)

        self._a = 2.0 * (1.0 - iteration / self._max_iterations)

    def create_random_cache(self, particle_ids: list[str], initialize: bool) -> None:
        for identifier in particle_ids:
            cache: dict[str, Serializable] = {}

            if not initialize:
                cache["p"] = float(self._rng.random())
                cache["l"] = float(self._rng.uniform(-1.0, 1.0))
                cache["random_particle"] = int(
                    self._rng.integers(0, len(self._population))
                )

                for variable in self._boundaries:
                    cache[f"{variable}-r1"] = float(self._rng.random())
                    cache[f"{variable}-r2"] = float(self._rng.random())

            self._population[identifier].random_cache = cache

    def initialize_particle(self, identifier: str) -> WhaleParticle:
        particle = self._population[identifier]

        if not isinstance(particle, WhaleParticle):
            raise TypeError(
                f"Particle '{identifier}' must be an instance of WhaleParticle."
            )

        if np.isinf(particle.fitness):
            particle.update(
                variables=particle.variables, fitness_function=self._fitness_function
            )

        return particle

    @staticmethod
    def consolidate_new_particles(particle: WhaleParticle) -> WhaleParticle:
        particle.consolidate(consolidate_new=True)
        return particle

    def update_particle(self, identifier: str) -> WhaleParticle:
        particle = self._population[identifier]

        if not isinstance(particle, WhaleParticle):
            raise TypeError(
                f"Particle '{identifier}' must be an instance of WhaleParticle."
            )

        if self._local_best is None:
            raise RuntimeError("Best whale has not been initialized.")

        best_variables = self._local_best.variables

        random_index = int(particle.random_cache["random_particle"])
        random_identifier = list(self._population)[random_index]
        random_particle = self._population[random_identifier]

        p = float(particle.random_cache["p"])
        l = float(particle.random_cache["l"])

        new_variables: dict[str, float] = {}

        for variable, (lower, upper) in self._boundaries.items():
            current_variable = particle.variables[variable]

            if p < 0.5:
                r1 = float(particle.random_cache[f"{variable}-r1"])
                r2 = float(particle.random_cache[f"{variable}-r2"])

                coefficient_a = 2.0 * self._a * r1 - self._a
                coefficient_c = 2.0 * r2

                if abs(coefficient_a) < 1.0:
                    reference_variable = best_variables[variable]
                else:
                    reference_variable = random_particle.variables[variable]

                distance = abs(coefficient_c * reference_variable - current_variable)

                variable_value = reference_variable - coefficient_a * distance
            else:
                distance = abs(best_variables[variable] - current_variable)

                variable_value = (
                    distance
                    * np.exp(self._spiral_coefficient * l)
                    * np.cos(2.0 * np.pi * l)
                    + best_variables[variable]
                )

            new_variables[variable] = float(np.clip(variable_value, lower, upper))

        particle.update(
            variables=new_variables, fitness_function=self._fitness_function
        )

        return particle

    def post_iteration(self, actual_iter: int) -> None:
        if actual_iter > 0:
            for particle in self._population.values():
                if not isinstance(particle, WhaleParticle):
                    raise TypeError(
                        f"Particle '{particle.identifier}' must be an instance "
                        "of WhaleParticle."
                    )

                if particle.candidate_fitness is None:
                    raise RuntimeError(
                        f"Particle '{particle.identifier}' has no candidate fitness."
                    )

                particle.consolidate(consolidate_new=True)

        self.update_solution_state()
