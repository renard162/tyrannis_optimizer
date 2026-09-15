from collections.abc import Mapping

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

    @staticmethod
    def _calculate_abc_fitness(fitness: np.float64) -> np.float64:
        if fitness < 0:
            return np.float64(1 + abs(fitness))

        return np.float64(1 / (1 + fitness))

    @property
    def trial_count(self) -> int:
        return self._trial_count

    @property
    def abc_fitness(self) -> np.float64:
        if np.isnan(self._fitness):
            return np.float64(np.nan)

        return self._calculate_abc_fitness(self._fitness)

    @property
    def abc_candidate_fitness(self) -> np.float64 | None:
        if self._candidate_fitness is None:
            return None

        if np.isnan(self._candidate_fitness):
            return np.float64(np.nan)

        return self._calculate_abc_fitness(self._candidate_fitness)

    def increment_trial_count(self) -> None:
        self._trial_count += 1

    def reset_trial_count(self) -> None:
        self._trial_count = 0


class ABC(AlgorithmBase):
    """Classical Artificial Bee Colony algorithm."""

    def __init__(
        self,
        colony_size: int = 40,
        limit: int | None = None,
        max_scouts: int | None = 1,
        improved_probability: bool = True,
    ) -> None:
        """
        Artificial Bee Colony algorithm for continuous optimization.

        ABC is a population-based optimization algorithm inspired by the
        foraging behavior of honey bees. The population represents food sources,
        which are explored by employed and onlooker bees, while scout bees
        replace food sources that fail to improve for a prescribed number of
        trials.

        Parameters
        ----------
        colony_size : int, default=40
            Total number of bees in the colony. Half of the colony corresponds
            to employed bees and food sources, while the other half corresponds
            to onlooker bees.

        limit : int or None, default=None
            Maximum number of unsuccessful trials allowed for a food source
            before it becomes eligible for replacement by a scout bee. When
            ``None``, the limit is set to the number of food sources multiplied
            by the problem dimensionality.

        max_scouts : int or None, default=1
            Maximum number of eligible food sources that may be replaced by
            scout bees in one iteration. When ``None``, all eligible food
            sources may be replaced. When multiple food sources are eligible
            and the value is smaller than their number, those with the highest
            trial counts are prioritized. This parameter extends the classical
            ABC behavior, in which only one scout is allowed per cycle.

        improved_probability : bool, default=True
            Whether to use the improved probability equation for selecting
            onlooker food sources. When ``True``, probabilities are calculated
            from the ratio between each food source fitness and the best
            fitness. When ``False``, the original ABC probability equation
            based on the sum of all fitness values is used.

        Notes
        -----
        The colony follows the standard ABC convention in which the number of
        food sources and employed bees is half the colony size, and the number
        of onlooker bees is equal to the number of food sources.

        The objective function is always interpreted as a cost function to be
        minimized. The raw objective value is stored in ``fitness`` and
        ``candidate_fitness``. The ABC fitness transformation is exposed by
        ``ABCParticle.abc_fitness`` and
        ``ABCParticle.abc_candidate_fitness`` and is only used when the ABC
        algorithm requires a maximization-oriented fitness value.

        Each optimization iteration performs one employed-bee neighborhood
        evaluation for every food source. Onlooker bees then select food
        sources through roulette-wheel selection, with replacement. If a food
        source is selected multiple times, its neighborhood is evaluated
        multiple times during the same particle update.

        The trial counter is an internal property of each particle and is not
        included in its serialized state. Consequently, a particle transferred
        to another execution environment starts with a new trial history.

        The classical ABC scout phase replaces at most one food source per
        cycle. ``max_scouts`` can be set to another positive value to allow
        multiple replacements as an explicit extension of that behavior.
        """
        if not isinstance(colony_size, (int, np.integer)):
            raise TypeError("colony_size must be an integer.")

        if colony_size < 4 or colony_size % 2:
            raise ValueError(
                "colony_size must be an even integer greater than or equal to 4."
            )

        if limit is not None:
            if not isinstance(limit, (int, np.integer)):
                raise TypeError("limit must be an integer or None.")

            if limit < 1:
                raise ValueError("limit must be greater than 0.")

        if max_scouts is not None:
            if not isinstance(max_scouts, (int, np.integer)):
                raise TypeError("max_scouts must be an integer or None.")

            if max_scouts < 1:
                raise ValueError("max_scouts must be greater than 0 when provided.")

        if not isinstance(improved_probability, (bool, np.bool_)):
            raise TypeError("improved_probability must be a boolean.")

        self._colony_size = int(colony_size)
        self._food_source_count = self._colony_size // 2
        self._limit = limit
        self._max_scouts = max_scouts
        self._improved_probability = bool(improved_probability)

        self._limit_value = None
        self._onlooker_counts: dict[str, int] = {}
        self._onlooker_probabilities: dict[str, float] = {}

    @property
    def colony_size(self) -> int:
        return self._colony_size

    @property
    def limit(self) -> int | None:
        return self._limit

    @property
    def max_scouts(self) -> int | None:
        return self._max_scouts

    @property
    def improved_probability(self) -> bool:
        return self._improved_probability

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

    def _select_scouts(self) -> list[str]:
        eligible = [
            particle
            for particle in self._population.values()
            if particle.trial_count >= self._limit_value
        ]

        if not eligible:
            return []

        if self._max_scouts is None:
            scout_count = len(eligible)
        else:
            scout_count = min(self._max_scouts, len(eligible))

        if scout_count == len(eligible):
            selected = eligible
        else:
            trial_counts = np.asarray(
                [particle.trial_count for particle in eligible], dtype=np.int64
            )

            order = np.argsort(-trial_counts, kind="stable")
            selected = [eligible[index] for index in order[:scout_count]]

        return [particle.identifier for particle in selected]

    def pre_iteration(self, actual_iter: int) -> None:
        if actual_iter == 0:
            if len(self._population) != self._food_source_count:
                raise ValueError(
                    f"ABC requires {self._food_source_count} particles for "
                    f"colony_size={self._colony_size}, but "
                    f"{len(self._population)} particles were provided."
                )

            dimension = len(self._boundaries)

            self._limit_value = (
                self._limit
                if self._limit is not None
                else self._food_source_count * dimension
            )

            return

        scout_ids = self._select_scouts()

        for identifier in scout_ids:
            self.delete_particle(identifier)
            self.create_particle(identifier=identifier)

    def _calculate_probabilities(self) -> dict[str, float]:
        particles = list(self._population.values())

        abc_fitness = np.asarray(
            [particle.abc_fitness for particle in particles], dtype=float
        )

        abc_fitness = np.where(
            np.isfinite(abc_fitness) & (abc_fitness > 0), abc_fitness, 0.0
        )

        if self._improved_probability:
            best_fitness = np.max(abc_fitness)

            if not np.isfinite(best_fitness) or best_fitness <= 0:
                probabilities = np.full(len(particles), 1.0 / len(particles))
            else:
                probabilities = 0.9 * (abc_fitness / best_fitness) + 0.1
        else:
            total_fitness = np.sum(abc_fitness)

            if not np.isfinite(total_fitness) or total_fitness <= 0:
                probabilities = np.full(len(particles), 1.0 / len(particles))
            else:
                probabilities = abc_fitness / total_fitness

        probabilities = np.where(
            np.isfinite(probabilities) & (probabilities >= 0), probabilities, 0.0
        )

        probability_sum = np.sum(probabilities)

        if not np.isfinite(probability_sum) or probability_sum <= 0:
            probabilities = np.full(len(particles), 1.0 / len(particles))
        else:
            probabilities /= probability_sum

        return {
            particle.identifier: float(probability)
            for particle, probability in zip(particles, probabilities)
        }

    def _select_onlookers(self) -> dict[str, int]:
        particle_ids = list(self._population)

        probabilities = np.asarray(
            [self._onlooker_probabilities[identifier] for identifier in particle_ids],
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

    def _select_partner(self, identifier: str) -> str:
        particle_ids = [
            particle_id for particle_id in self._population if particle_id != identifier
        ]

        return particle_ids[self._rng.integers(0, len(particle_ids))]

    def create_random_cache(self, particle_ids: list[str], initialize: bool) -> None:
        if initialize:
            for identifier in particle_ids:
                self._population[identifier].random_cache = {}

            return

        self._onlooker_probabilities = self._calculate_probabilities()
        self._onlooker_counts = self._select_onlookers()

        variables = tuple(self._boundaries)

        for identifier in particle_ids:
            cache: dict[str, Serializable] = {
                "employed-variable": variables[self._rng.integers(0, len(variables))],
                "employed-partner": self._select_partner(identifier),
                "employed-phi": float(self._rng.uniform(-1.0, 1.0)),
            }

            for index in range(self._onlooker_counts[identifier]):
                cache[f"onlooker-{index}-variable"] = variables[
                    self._rng.integers(0, len(variables))
                ]
                cache[f"onlooker-{index}-partner"] = self._select_partner(identifier)
                cache[f"onlooker-{index}-phi"] = float(self._rng.uniform(-1.0, 1.0))

            self._population[identifier].random_cache = cache

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

    def _generate_candidate(
        self,
        variables: Mapping[str, float],
        variable: str,
        partner: ABCParticle,
        phi: float,
    ) -> dict[str, float]:
        lower, upper = self._boundaries[variable]

        candidate_variable = variables[variable] + phi * (
            variables[variable] - partner.variables[variable]
        )

        candidate_variable = float(np.clip(candidate_variable, lower, upper))

        candidate_variables = dict(variables)
        candidate_variables[variable] = candidate_variable

        return candidate_variables

    def _attempt_update(
        self,
        particle: ABCParticle,
        variables: dict[str, float],
        fitness: np.float64,
        variable: str,
        partner_id: str,
        phi: float,
    ) -> tuple[dict[str, float], np.float64]:
        partner = self._population[partner_id]

        candidate_variables = self._generate_candidate(
            variables=variables, variable=variable, partner=partner, phi=phi
        )

        candidate_fitness = self._fitness_function(candidate_variables)

        candidate_abc_fitness = ABCParticle._calculate_abc_fitness(candidate_fitness)

        current_abc_fitness = ABCParticle._calculate_abc_fitness(fitness)

        if (
            np.isfinite(candidate_abc_fitness)
            and np.isfinite(current_abc_fitness)
            and candidate_abc_fitness > current_abc_fitness
        ):
            particle.reset_trial_count()

            return candidate_variables, candidate_fitness

        particle.increment_trial_count()

        return variables, fitness

    def update_particle(self, identifier: str) -> ABCParticle:
        particle = self._population[identifier]

        if not isinstance(particle, ABCParticle):
            raise TypeError(
                f"Particle '{identifier}' must be an instance of ABCParticle."
            )

        current_variables = dict(particle.variables)
        current_fitness = particle.fitness

        employed_variable = particle.random_cache.pop("employed-variable")
        employed_partner = particle.random_cache.pop("employed-partner")
        employed_phi = particle.random_cache.pop("employed-phi")

        current_variables, current_fitness = self._attempt_update(
            particle=particle,
            variables=current_variables,
            fitness=current_fitness,
            variable=employed_variable,
            partner_id=employed_partner,
            phi=employed_phi,
        )

        for index in range(self._onlooker_counts[identifier]):
            variable = particle.random_cache.pop(f"onlooker-{index}-variable")
            partner_id = particle.random_cache.pop(f"onlooker-{index}-partner")
            phi = particle.random_cache.pop(f"onlooker-{index}-phi")

            current_variables, current_fitness = self._attempt_update(
                particle=particle,
                variables=current_variables,
                fitness=current_fitness,
                variable=variable,
                partner_id=partner_id,
                phi=phi,
            )

        particle.candidate_variables = current_variables
        particle.candidate_fitness = current_fitness

        return particle

    def post_iteration(self, actual_iter: int) -> None:
        for particle in self._population.values():
            if not isinstance(particle, ABCParticle):
                raise TypeError(
                    f"Particle '{particle.identifier}' must be an instance of "
                    "ABCParticle."
                )

            if actual_iter == 0:
                continue

            if particle.candidate_fitness is None:
                raise RuntimeError(
                    f"Particle '{particle.identifier}' has no candidate fitness."
                )

            particle.consolidate(
                consolidate_new=bool(
                    particle.abc_candidate_fitness < particle.abc_fitness
                )
            )

        self.update_solution_state()
