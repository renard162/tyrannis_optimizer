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
        self._onlooker_count = 0

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
    def onlooker_count(self) -> int:
        return self._onlooker_count

    @property
    def second_update_required(self) -> bool:
        return self._onlooker_count > 0

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

    def set_onlooker_count(self, onlooker_count: int) -> None:
        self._onlooker_count = onlooker_count


class BeeColony(AlgorithmBase[ABCParticle]):
    """Artificial Bee Colony algorithm."""

    def __init__(
        self,
        limit: int | None = None,
        max_scouts: int | None = 1,
        improved_probability: bool = True,
    ) -> None:
        """
        Artificial Bee Colony algorithm (ABC).

        ABC is a population-based optimization algorithm inspired by the
        foraging behavior of honey bees. The population represents food sources,
        which are explored by employed and onlooker bees, while scout bees
        replace food sources that fail to improve for a prescribed number of
        trials.

        Parameters
        ----------
        limit : int or None, default=None
            Maximum number of unsuccessful trials allowed for a food source
            before it becomes eligible for replacement by a scout bee. When
            ``None``, the limit is calculated from the current number of food
            sources and the problem dimensionality.

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
        Artificial Bee Colony (ABC) is a population-based optimization
        algorithm inspired by the foraging behavior of honey bees. Employed
        bees explore the neighborhood of existing food sources, while
        onlooker bees preferentially explore promising sources through
        roulette-wheel selection. Candidate solutions are accepted only when
        they improve the current source.

        Scout bees replace food sources that have remained unsuccessful for
        a predefined number of attempts, introducing new solutions and
        preventing stagnation. By default, one scout replaces one abandoned
        food source per iteration, as in the classical ABC algorithm.
        ``max_scouts`` can be used to allow multiple replacements in the same
        iteration.

        The classical probability assigns each food source a probability
        proportional to its fitness relative to the population:

            p_i = fitness_i / sum(fitness)

        When ``improved_probability=True``, the improved formulation uses the
        best fitness as reference:

            p_i = 0.9 * (fitness_i / fitness_best) + 0.1

        This formulation maintains a minimum selection probability for less
        promising sources while still favoring better ones, increasing the
        opportunity for exploration.

        References
        ----------
        Karaboga, D., & Basturk, B. (2007). A powerful and efficient algorithm
        for numerical function optimization: Artificial Bee Colony (ABC)
        algorithm. Journal of Global Optimization, 39(3), 459-471.
        https://doi.org/10.1007/s10898-007-9149-x

        Karaboga, D., & Basturk, B. (2008). On the performance of Artificial
        Bee Colony (ABC) algorithm. Applied Soft Computing, 8(1), 687-697.
        https://doi.org/10.1016/j.asoc.2007.05.007

        Öztürk, C., Karaboga, D., & Görkemli, B. (2011). Probabilistic Dynamic
        Deployment of Wireless Sensor Networks by Artificial Bee Colony
        Algorithm. Sensors, 11(6), 6056-6065.
        https://doi.org/10.3390/s110606056

        Šarčević, T., Rocha, A. P. C., & Castro, A. J. M. (2018). Artificial
        Bee Colony Algorithm for Solving the Flight Disruption Problem.
        In Highlights of Practical Applications of Agents, Multi-Agent
        Systems, and Complexity: The PAAMS Collection. Communications in
        Computer and Information Science.
        https://doi.org/10.1007/978-3-319-94779-2_7
        """
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

        self._limit = limit
        self._max_scouts = max_scouts
        self._improved_probability = bool(improved_probability)

        self._onlooker_probabilities: dict[str, float] = {}

    @property
    def colony_size(self) -> int:
        return 2 * self._n_particles

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

    def _validate_population_size(self) -> None:
        if len(self._population) < 2:
            raise ValueError(
                "BeeColony requires at least 2 particles so each food source "
                "can select a distinct partner."
            )

    def _select_scouts(self) -> list[str]:
        limit = (
            self._limit
            if self._limit is not None
            else self._n_particles * len(self._boundaries)
        )

        eligible = [
            particle
            for particle in self._population.values()
            if particle.trial_count >= limit
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
        self._validate_population_size()

        for particle in self._population.values():
            if not isinstance(particle, ABCParticle):
                raise TypeError(
                    f"Particle '{particle.identifier}' must be an instance of "
                    "ABCParticle."
                )

            particle.set_onlooker_count(0)

        if actual_iter == 0:
            self._double_particle_check = True
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
                probabilities = np.ones(len(particles), dtype=float)
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
        probabilities = np.clip(probabilities, 0.0, 1.0)

        return {
            particle.identifier: float(probability)
            for particle, probability in zip(particles, probabilities)
        }

    def _select_onlookers(self) -> dict[str, int]:
        particle_ids = list(self._population)
        counts = dict.fromkeys(particle_ids, 0)

        selected_count = 0
        particle_index = 0
        target_count = len(particle_ids)

        while selected_count < target_count:
            identifier = particle_ids[particle_index]
            probability = self._onlooker_probabilities[identifier]

            if self._rng.random() < probability:
                counts[identifier] += 1
                selected_count += 1

            particle_index += 1

            if particle_index == len(particle_ids):
                particle_index = 0

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

        variables = tuple(self._boundaries)

        for identifier in particle_ids:
            particle = self._population[identifier]

            if not isinstance(particle, ABCParticle):
                raise TypeError(
                    f"Particle '{identifier}' must be an instance of ABCParticle."
                )

            cache: dict[str, Serializable] = {
                "employed-variable": variables[self._rng.integers(0, len(variables))],
                "employed-partner": self._select_partner(identifier),
                "employed-phi": float(self._rng.uniform(-1.0, 1.0)),
            }

            for index in range(particle.onlooker_count):
                cache[f"onlooker-{index}-variable"] = variables[
                    self._rng.integers(0, len(variables))
                ]
                cache[f"onlooker-{index}-partner"] = self._select_partner(identifier)
                cache[f"onlooker-{index}-phi"] = float(self._rng.uniform(-1.0, 1.0))

            particle.random_cache = cache

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

    @staticmethod
    def _is_improvement(
        candidate_fitness: np.float64, current_fitness: np.float64
    ) -> bool:
        candidate_abc_fitness = ABCParticle._calculate_abc_fitness(candidate_fitness)
        current_abc_fitness = ABCParticle._calculate_abc_fitness(current_fitness)

        return bool(
            np.isfinite(candidate_abc_fitness)
            and np.isfinite(current_abc_fitness)
            and candidate_abc_fitness > current_abc_fitness
        )

    def update_particle(self, identifier: str) -> ABCParticle:
        particle = self._population[identifier]

        if not isinstance(particle, ABCParticle):
            raise TypeError(
                f"Particle '{identifier}' must be an instance of ABCParticle."
            )

        variable = particle.random_cache["employed-variable"]
        partner_id = particle.random_cache["employed-partner"]
        phi = particle.random_cache["employed-phi"]
        partner = self._population[partner_id]

        if not isinstance(partner, ABCParticle):
            raise TypeError(
                f"Particle '{partner_id}' must be an instance of ABCParticle."
            )

        candidate_variables = self._generate_candidate(
            variables=particle.variables, variable=variable, partner=partner, phi=phi
        )

        particle.increment_trial_count()
        particle.update(
            variables=candidate_variables, fitness_function=self._fitness_function
        )

        if particle.candidate_fitness is None:
            raise RuntimeError(
                f"Particle '{identifier}' has no candidate fitness after update."
            )

        if self._is_improvement(
            candidate_fitness=particle.candidate_fitness,
            current_fitness=particle.fitness,
        ):
            particle.reset_trial_count()

        return particle

    def inter_iteration(self, actual_iter: int) -> None:
        for particle in self._population.values():
            if not isinstance(particle, ABCParticle):
                raise TypeError(
                    f"Particle '{particle.identifier}' must be an instance of "
                    "ABCParticle."
                )

            if particle.candidate_fitness is None:
                raise RuntimeError(
                    f"Particle '{particle.identifier}' has no candidate fitness."
                )

            improved = self._is_improvement(
                candidate_fitness=particle.candidate_fitness,
                current_fitness=particle.fitness,
            )

            particle.consolidate(consolidate_new=improved)

        self.update_solution_state()

        self._onlooker_probabilities = self._calculate_probabilities()
        onlooker_counts = self._select_onlookers()

        for identifier, onlooker_count in onlooker_counts.items():
            particle = self._population[identifier]

            if not isinstance(particle, ABCParticle):
                raise TypeError(
                    f"Particle '{identifier}' must be an instance of ABCParticle."
                )

            particle.set_onlooker_count(onlooker_count)

    def second_update_particle(self, identifier: str) -> ABCParticle:
        particle = self._population[identifier]

        if not isinstance(particle, ABCParticle):
            raise TypeError(
                f"Particle '{identifier}' must be an instance of ABCParticle."
            )

        if not particle.second_update_required:
            return particle

        current_variables = dict(particle.variables)
        current_fitness = particle.fitness

        particle.candidate_variables = current_variables
        particle.candidate_fitness = current_fitness

        for index in range(particle.onlooker_count):
            variable = particle.random_cache[f"onlooker-{index}-variable"]
            partner_id = particle.random_cache[f"onlooker-{index}-partner"]
            phi = particle.random_cache[f"onlooker-{index}-phi"]
            partner = self._population[partner_id]

            if not isinstance(partner, ABCParticle):
                raise TypeError(
                    f"Particle '{partner_id}' must be an instance of ABCParticle."
                )

            candidate_variables = self._generate_candidate(
                variables=current_variables, variable=variable, partner=partner, phi=phi
            )

            particle.increment_trial_count()
            candidate_fitness = self._fitness_function(candidate_variables)

            if np.isnan(candidate_fitness):
                particle.candidate_variables = candidate_variables
                particle.candidate_fitness = candidate_fitness
                return particle

            if self._is_improvement(
                candidate_fitness=candidate_fitness, current_fitness=current_fitness
            ):
                current_variables = candidate_variables
                current_fitness = candidate_fitness
                particle.reset_trial_count()

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
                    f"Particle '{particle.identifier}' must be an instance of "
                    "ABCParticle."
                )

            if not particle.second_update_required:
                continue

            if particle.candidate_fitness is None:
                raise RuntimeError(
                    f"Particle '{particle.identifier}' has no candidate fitness."
                )

            if particle.error_fitness is not None:
                particle.consolidate(consolidate_new=False)
                continue

            improved = self._is_improvement(
                candidate_fitness=particle.candidate_fitness,
                current_fitness=particle.fitness,
            )

            particle.consolidate(consolidate_new=improved)

        self.update_solution_state()
