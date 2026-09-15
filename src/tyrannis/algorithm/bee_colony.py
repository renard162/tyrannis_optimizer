from copy import deepcopy

import numpy as np

from ..core.algorithm import (
    FITNESS_UNDEFINED,
    AlgorithmBase,
    ParticleBase,
    Serializable,
)


class ABCParticle(ParticleBase):
    """Particle implementation for the Artificial Bee Colony algorithm."""

    def __init__(
        self,
        identifier: str,
        variables: dict[str, float],
        fitness: np.float64 = FITNESS_UNDEFINED,
    ) -> None:
        super().__init__(identifier=identifier, variables=variables, fitness=fitness)

        self._trial_count = 0

    def __call__(self) -> dict[str, Serializable]:
        return {
            "identifier": self._identifier,
            "variables": self._variables,
            "fitness": self._fitness,
        }

    @property
    def trial_count(self) -> int:
        return self._trial_count

    def increment_trial_count(self) -> None:
        self._trial_count += 1

    def reset_trial_count(self) -> None:
        self._trial_count = 0


class ABC(AlgorithmBase[ABCParticle]):
    """Classical Artificial Bee Colony algorithm for continuous optimization."""

    def __init__(
        self,
        colony_size: int = 40,
        trial_limit: int | None = None,
        scouts: float = 1.0,
        max_scouts: int | None = None,
        improved_probability: bool = True,
    ) -> None:
        """
        Artificial Bee Colony algorithm for continuous optimization.

        ABC is a population-based optimization algorithm that models the
        foraging behavior of honey bees. The population represents food sources,
        with employed and onlooker bees exploring their neighborhoods and scout
        bees replacing food sources that have not improved for a prescribed
        number of trials.

        Parameters
        ----------
        colony_size : int, default=40
            Total number of bees in the colony. The number of food sources, and
            therefore employed bees, is half of the colony size. The same number
            of onlooker bees is used, following the standard ABC formulation.

        trial_limit : int or None, default=None
            Maximum number of unsuccessful neighborhood trials tolerated by a
            food source before it becomes eligible for scout replacement. When
            ``None``, the limit is set to the number of food sources multiplied
            by the problem dimensionality.

        scouts : float, default=1.0
            Number of food sources that may be selected for scout replacement in
            each iteration. Values strictly between 0 and 1 are interpreted as
            a fraction of the number of food sources. The value 1 selects one
            scout. Values greater than 1 specify the number of scouts directly.

        max_scouts : int or None, default=None
            Maximum number of food sources that may be replaced by scouts in one
            iteration. When ``None``, all food sources that reach ``trial_limit``
            may be replaced. When a positive integer is provided, at most that
            many food sources are replaced, prioritizing the highest trial counts.

        improved_probability : bool, default=True
            Whether to use the improved onlooker probability equation. When
            enabled, probabilities are calculated as ``0.9 * fit_i / fit_max +
            0.1``. When disabled, the original ABC equation ``fit_i / sum(fit)``
            is used. A uniform fallback is used whenever the selected equation
            cannot produce a valid probability distribution.

        Notes
        -----
        The algorithm uses the standard ABC colony convention: ``colony_size``
        is the total number of bees and the population contains
        ``colony_size / 2`` food sources. Each optimization iteration performs
        one employed-bee neighborhood evaluation for every food source and a
        second set of neighborhood evaluations selected by roulette wheel for
        the onlooker bees. Because Tyrannis processes each particle once per
        iteration, repeated onlooker selections for the same particle are
        accumulated and executed sequentially within that particle's update.

        Fitness stored in particles is always the raw objective value. Whenever
        the ABC algorithm requires fitness for selection or comparison, the raw
        objective is transformed through ``_fitness_from_cost``. Subclasses may
        override this transformation to implement a different fitness mapping.
        """
        if not isinstance(colony_size, (int, np.integer)):
            raise TypeError("colony_size must be an integer.")

        colony_size = int(colony_size)

        if colony_size < 4 or colony_size % 2:
            raise ValueError(
                "colony_size must be an even integer greater than or equal to 4."
            )

        if trial_limit is not None:
            if not isinstance(trial_limit, (int, np.integer)):
                raise TypeError("trial_limit must be an integer or None.")

            if trial_limit < 1:
                raise ValueError("trial_limit must be greater than 0.")

        if not isinstance(scouts, (int, float, np.number)) or not np.isfinite(scouts):
            raise TypeError("scouts must be a finite number.")

        if scouts <= 0:
            raise ValueError("scouts must be greater than 0.")

        if max_scouts is not None:
            if not isinstance(max_scouts, (int, np.integer)):
                raise TypeError("max_scouts must be an integer or None.")

            if max_scouts <= 0:
                raise ValueError("max_scouts must be greater than 0 when provided.")

        if not isinstance(improved_probability, (bool, np.bool_)):
            raise TypeError("improved_probability must be a boolean.")

        self._colony_size = colony_size
        self._food_source_count = colony_size // 2
        self._trial_limit = trial_limit
        self._scouts = float(scouts)
        self._max_scouts = max_scouts
        self._improved_probability = bool(improved_probability)

        self._trial_limit_value: int | None = None
        self._onlooker_probabilities: dict[str, float] = {}
        self._onlooker_counts: dict[str, int] = {}

    @property
    def colony_size(self) -> int:
        return self._colony_size

    @property
    def trial_limit(self) -> int | None:
        return self._trial_limit

    @property
    def improved_probability(self) -> bool:
        return self._improved_probability

    @staticmethod
    def _fitness_from_cost(cost: np.float64) -> np.float64:
        """Transform a cost value into the maximization-oriented ABC fitness."""
        if np.isnan(cost):
            return np.float64(np.nan)

        if cost < 0:
            return np.float64(1.0 + abs(cost))

        return np.float64(1.0 / (1.0 + cost))

    def _fitness_value(self, cost: np.float64) -> np.float64:
        return self._fitness_from_cost(cost)

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

        self._population[identifier] = ABCParticle(
            identifier=identifier, variables=variables, fitness=fitness
        )

    def delete_particle(self, identifier: str | None) -> None:
        if identifier is None:
            raise ValueError("Particle identifier cannot be None.")

        del self._population[identifier]

    def _scout_count(self) -> int:
        if self._scouts == 1:
            count = 1
        elif self._scouts < 1:
            count = int(self._food_source_count * self._scouts)
        else:
            count = int(self._scouts)

        return max(count, 1)

    def _replace_scouts(self) -> None:
        eligible = [
            particle
            for particle in self._population.values()
            if particle.trial_count >= self._trial_limit_value
        ]

        if not eligible:
            return

        selected_count = min(self._scout_count(), len(eligible))

        if self._max_scouts is not None:
            selected_count = min(selected_count, self._max_scouts)

        shuffled = list(self._rng.permutation(eligible))
        shuffled.sort(key=lambda particle: particle.trial_count, reverse=True)

        selected = shuffled[:selected_count]

        for particle in selected:
            self.delete_particle(particle.identifier)
            self.create_particle(identifier=particle.identifier)

    def pre_iteration(self, actual_iter: int) -> None:
        if actual_iter == 0:
            expected_particles = self._food_source_count

            if len(self._population) != expected_particles:
                raise ValueError(
                    f"ABC requires {expected_particles} particles for a colony "
                    f"of {self._colony_size} bees, but received "
                    f"{len(self._population)}."
                )

            dimension = len(self._boundaries)

            self._trial_limit_value = (
                self._trial_limit
                if self._trial_limit is not None
                else self._food_source_count * dimension
            )

            return

        self._replace_scouts()

    def _calculate_onlooker_probabilities(self) -> dict[str, float]:
        particles = list(self._population.values())

        fitness_values = np.asarray(
            [self._fitness_value(particle.fitness) for particle in particles],
            dtype=float,
        )

        valid = np.isfinite(fitness_values) & (fitness_values >= 0)

        if not np.any(valid):
            probability = 1.0 / len(particles)

            return {particle.identifier: probability for particle in particles}

        valid_fitness = fitness_values[valid]

        if self._improved_probability:
            best_fitness = np.max(valid_fitness)

            if not np.isfinite(best_fitness) or best_fitness <= 0:
                probability = 1.0 / len(particles)

                return {particle.identifier: probability for particle in particles}

            probabilities = 0.9 * (fitness_values / best_fitness) + 0.1

        else:
            total_fitness = np.sum(valid_fitness)

            if not np.isfinite(total_fitness) or total_fitness <= 0:
                probability = 1.0 / len(particles)

                return {particle.identifier: probability for particle in particles}

            probabilities = fitness_values / total_fitness

        probabilities[~valid] = 0.0
        probabilities = np.where(np.isfinite(probabilities), probabilities, 0.0)

        total_probability = np.sum(probabilities)

        if not np.isfinite(total_probability) or total_probability <= 0:
            probability = 1.0 / len(particles)

            return {particle.identifier: probability for particle in particles}

        probabilities /= total_probability

        return {
            particle.identifier: float(probability)
            for particle, probability in zip(particles, probabilities)
        }

    def _select_onlookers(self) -> dict[str, int]:
        particle_ids = list(self._population)

        probabilities = np.asarray(
            [self._onlooker_probabilities[particle_id] for particle_id in particle_ids],
            dtype=float,
        )

        cumulative = np.cumsum(probabilities)
        cumulative[-1] = 1.0

        selections = self._rng.random(self._food_source_count)

        selected_indices = np.searchsorted(cumulative, selections, side="right")

        counts = dict.fromkeys(particle_ids, 0)

        for index in selected_indices:
            counts[particle_ids[index]] += 1

        return counts

    def _random_partner_id(self, identifier: str) -> str:
        particle_ids = [
            particle_id for particle_id in self._population if particle_id != identifier
        ]

        return particle_ids[self._rng.integers(0, len(particle_ids))]

    def create_random_cache(self, particle_ids: list[str], initialize: bool) -> None:
        if initialize:
            for particle_id in particle_ids:
                self._population[particle_id].random_cache = {}

            return

        self._onlooker_probabilities = self._calculate_onlooker_probabilities()
        self._onlooker_counts = self._select_onlookers()

        variable_names = list(self._boundaries)

        for particle_id in particle_ids:
            onlooker_count = self._onlooker_counts[particle_id]

            cache: dict[str, Serializable] = {
                "employed-variable": variable_names[
                    self._rng.integers(0, len(variable_names))
                ],
                "employed-partner": self._random_partner_id(particle_id),
                "employed-phi": float(self._rng.uniform(-1.0, 1.0)),
            }

            for index in range(onlooker_count):
                cache[f"onlooker-{index}-variable"] = variable_names[
                    self._rng.integers(0, len(variable_names))
                ]
                cache[f"onlooker-{index}-partner"] = self._random_partner_id(
                    particle_id
                )
                cache[f"onlooker-{index}-phi"] = float(self._rng.uniform(-1.0, 1.0))

            self._population[particle_id].random_cache = cache

    def initialize_particle(self, identifier: str) -> ABCParticle:
        particle = self._population[identifier]

        if not isinstance(particle, ABCParticle):
            raise TypeError(
                f"Particle '{identifier}' must be an instance of ABCParticle."
            )

        if np.isinf(particle.fitness):
            particle.update(
                variables=particle.variables, fitness_function=self._fitness_function
            )

        return particle

    @staticmethod
    def consolidate_new_particles(particle: ABCParticle) -> ABCParticle:
        particle.consolidate(consolidate_new=True)
        return particle

    def _dance(
        self,
        particle: ABCParticle,
        variable: str,
        partner_id: str,
        phi: float,
        current_variables: dict[str, float],
        current_fitness: np.float64,
    ) -> tuple[dict[str, float], np.float64]:
        partner = self._population[partner_id]

        lower, upper = self._boundaries[variable]

        candidate_variable = current_variables[variable] + phi * (
            current_variables[variable] - partner.variables[variable]
        )

        candidate_variable = float(np.clip(candidate_variable, lower, upper))

        candidate_variables = dict(current_variables)
        candidate_variables[variable] = candidate_variable

        candidate_fitness = self._fitness_function(candidate_variables)

        current_fit = self._fitness_value(current_fitness)
        candidate_fit = self._fitness_value(candidate_fitness)

        if np.isfinite(candidate_fit) and (
            not np.isfinite(current_fit) or candidate_fit > current_fit
        ):
            particle.reset_trial_count()

            return candidate_variables, candidate_fitness

        particle.increment_trial_count()

        return current_variables, current_fitness

    def update_particle(self, identifier: str) -> ABCParticle:
        particle = self._population[identifier]

        if not isinstance(particle, ABCParticle):
            raise TypeError(
                f"Particle '{identifier}' must be an instance of ABCParticle."
            )

        current_variables = dict(particle.variables)
        current_fitness = particle.fitness
        cache = particle.random_cache

        variable = cache.pop("employed-variable")
        partner_id = cache.pop("employed-partner")
        phi = cache.pop("employed-phi")

        current_variables, current_fitness = self._dance(
            particle=particle,
            variable=variable,
            partner_id=partner_id,
            phi=phi,
            current_variables=current_variables,
            current_fitness=current_fitness,
        )

        for index in range(self._onlooker_counts[identifier]):
            variable = cache.pop(f"onlooker-{index}-variable")
            partner_id = cache.pop(f"onlooker-{index}-partner")
            phi = cache.pop(f"onlooker-{index}-phi")

            current_variables, current_fitness = self._dance(
                particle=particle,
                variable=variable,
                partner_id=partner_id,
                phi=phi,
                current_variables=current_variables,
                current_fitness=current_fitness,
            )

        particle.candidate_variables = current_variables
        particle.candidate_fitness = current_fitness

        return particle

    def post_iteration(self, actual_iter: int) -> None:
        if actual_iter == 0:
            self.update_solution_state()
            return

        for particle in self._population.values():
            if not isinstance(particle, ABCParticle):
                raise TypeError(
                    f"Particle '{particle.identifier}' must be an instance "
                    "of ABCParticle."
                )

            if particle.candidate_fitness is None:
                raise RuntimeError(
                    f"Particle '{particle.identifier}' has no candidate fitness."
                )

            particle.consolidate(
                consolidate_new=bool(particle.candidate_fitness < particle.fitness)
            )

        self.update_solution_state()
