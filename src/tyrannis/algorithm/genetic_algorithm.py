import numpy as np

from ..core.algorithm import (
    FITNESS_UNDEFINED,
    AlgorithmBase,
    ParticleBase,
    Serializable,
)


class GAParticle(ParticleBase):
    """Particle implementation for the Genetic Algorithm."""


class GeneticAlgorithm(AlgorithmBase[GAParticle]):
    """Real-coded Genetic Algorithm for continuous optimization."""

    def __init__(
        self,
        population_method: str = "steady-state",
        selection: str = "tournament",
        tournament_size: int = 2,
        crossover: str = "sbx",
        crossover_probability: float = 0.9,
        arithmetic_alpha: float = 0.5,
        blx_alpha: float = 0.5,
        sbx_eta: float = 20.0,
        mutation: str = "polynomial",
        mutation_probability: float | None = None,
        mutation_intensity: float = 1.0,
        gaussian_sigma: float = 0.1,
        polynomial_eta: float = 20.0,
        survival: str = "elitism",
        elite_count: int = 1,
    ) -> None:
        """
        Real-coded Genetic Algorithm for continuous optimization.

        The algorithm maintains a fixed-size population of continuous candidate
        solutions and evolves it through parent selection, crossover, mutation,
        and survivor selection. Population evolution can use either generational
        or steady-state replacement.

        Parameters
        ----------
        population_method : {"generational", "steady-state"}, default="steady-state"
            Population evolution strategy. ``"generational"`` replaces the
            population with offspring at each iteration, while ``"steady-state"``
            progressively replaces existing individuals with offspring.

        selection : {"tournament", "fitness", "ranking"}, default="tournament"
            Parent-selection strategy.

        tournament_size : int, default=2
            Number of candidates participating in each tournament when
            ``selection="tournament"``.

        crossover : {"arithmetic", "blx", "sbx"}, default="sbx"
            Crossover operator. SBX is the default real-coded crossover used
            in common continuous genetic algorithm configurations.

        crossover_probability : float, default=0.9
            Probability of applying crossover when generating an offspring.

        arithmetic_alpha : float, default=0.5
            Mixing coefficient used by arithmetic crossover.

        blx_alpha : float, default=0.5
            Extension factor used by BLX-alpha crossover.

        sbx_eta : float, default=20.0
            Distribution index used by simulated binary crossover.

        mutation : {"gaussian", "polynomial"}, default="polynomial"
            Mutation operator.

        mutation_probability : float or None, default=None
            Per-variable probability of mutation. When ``None``, the probability
            is set to ``1 / n_variables`` during context initialization.

        mutation_intensity : float, default=1.0
            Multiplicative factor applied to the magnitude produced by the
            selected mutation operator.

        gaussian_sigma : float, default=0.1
            Base standard deviation of Gaussian mutation expressed as a fraction
            of each variable's search range.

        polynomial_eta : float, default=20.0
            Distribution index used by polynomial mutation.

        survival : {"elitism", "replacement"}, default="elitism"
            Survivor-selection strategy. ``"replacement"`` always accepts the
            generated candidate, while ``"elitism"`` preserves the configured
            number of best current individuals.

        elite_count : int, default=1
            Number of current individuals protected from replacement when
            ``survival="elitism"``.

        Notes
        -----
        The real-coded representation is used directly because Tyrannis
        optimizes continuous functions of the form ``f: R^n -> R``. The
        implementation therefore does not perform binary chromosome encoding.

        SBX is based on Deb and Agrawal (1995). Polynomial mutation follows the
        real-coded mutation formulation commonly used with SBX in continuous
        evolutionary optimization.

        References
        ----------
        Deb, K., & Agrawal, R. B. (1995). Simulated binary crossover for
        continuous search space. Complex Systems, 9, 115-148.

        Eshelman, L. J., & Schaffer, J. D. (1993). Real-coded genetic
        algorithms and interval schemata. Foundations of Genetic Algorithms,
        2, 187-202.

        Deb, K. (2001). Multi-Objective Optimization using Evolutionary
        Algorithms. Wiley.
        """
        if population_method not in ("generational", "steady-state"):
            raise ValueError(
                "population_method must be either 'generational' or 'steady-state'."
            )

        if selection not in ("tournament", "fitness", "ranking"):
            raise ValueError("selection must be 'tournament', 'fitness', or 'ranking'.")

        if not isinstance(tournament_size, (int, np.integer)):
            raise TypeError("tournament_size must be an integer.")

        if tournament_size < 1:
            raise ValueError("tournament_size must be greater than 0.")

        if crossover not in ("arithmetic", "blx", "sbx"):
            raise ValueError("crossover must be 'arithmetic', 'blx', or 'sbx'.")

        self._validate_probability(crossover_probability, "crossover_probability")

        self._validate_finite_nonnegative(arithmetic_alpha, "arithmetic_alpha")
        self._validate_finite_nonnegative(blx_alpha, "blx_alpha")
        self._validate_finite_positive(sbx_eta, "sbx_eta")

        if mutation not in ("gaussian", "polynomial"):
            raise ValueError("mutation must be 'gaussian' or 'polynomial'.")

        if mutation_probability is not None:
            self._validate_probability(mutation_probability, "mutation_probability")

        self._validate_finite_nonnegative(mutation_intensity, "mutation_intensity")
        self._validate_finite_positive(gaussian_sigma, "gaussian_sigma")
        self._validate_finite_positive(polynomial_eta, "polynomial_eta")

        if survival not in ("elitism", "replacement"):
            raise ValueError("survival must be 'elitism' or 'replacement'.")

        if not isinstance(elite_count, (int, np.integer)):
            raise TypeError("elite_count must be an integer.")

        if elite_count < 1:
            raise ValueError("elite_count must be greater than 0.")

        self._population_method = population_method
        self._selection = selection
        self._tournament_size = int(tournament_size)
        self._crossover_method = crossover
        self._crossover_probability = float(crossover_probability)
        self._arithmetic_alpha = float(arithmetic_alpha)
        self._blx_alpha = float(blx_alpha)
        self._sbx_eta = float(sbx_eta)
        self._mutation = mutation
        self._mutation_probability = (
            None if mutation_probability is None else float(mutation_probability)
        )
        self._mutation_intensity = float(mutation_intensity)
        self._gaussian_sigma = float(gaussian_sigma)
        self._polynomial_eta = float(polynomial_eta)
        self._survival = survival
        self._elite_count = int(elite_count)

        self._effective_mutation_probability: float | None = None
        self._active_identifier: str | None = None

    @staticmethod
    def _validate_probability(value: float, name: str) -> None:
        if not isinstance(value, (int, float, np.number)):
            raise TypeError(f"{name} must be a number.")

        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1.")

    @staticmethod
    def _validate_finite_nonnegative(value: float, name: str) -> None:
        if not isinstance(value, (int, float, np.number)):
            raise TypeError(f"{name} must be a number.")

        if not np.isfinite(value) or value < 0:
            raise ValueError(
                f"{name} must be a finite number greater than or equal to 0."
            )

    @staticmethod
    def _validate_finite_positive(value: float, name: str) -> None:
        if not isinstance(value, (int, float, np.number)):
            raise TypeError(f"{name} must be a number.")

        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite number greater than 0.")

    @property
    def population_method(self) -> str:
        return self._population_method

    @property
    def selection(self) -> str:
        return self._selection

    @property
    def tournament_size(self) -> int:
        return self._tournament_size

    @property
    def crossover(self) -> str:
        return self._crossover_method

    @property
    def crossover_probability(self) -> float:
        return self._crossover_probability

    @property
    def arithmetic_alpha(self) -> float:
        return self._arithmetic_alpha

    @property
    def blx_alpha(self) -> float:
        return self._blx_alpha

    @property
    def sbx_eta(self) -> float:
        return self._sbx_eta

    @property
    def mutation(self) -> str:
        return self._mutation

    @property
    def mutation_probability(self) -> float | None:
        return self._mutation_probability

    @property
    def mutation_intensity(self) -> float:
        return self._mutation_intensity

    @property
    def gaussian_sigma(self) -> float:
        return self._gaussian_sigma

    @property
    def polynomial_eta(self) -> float:
        return self._polynomial_eta

    @property
    def survival(self) -> str:
        return self._survival

    @property
    def elite_count(self) -> int:
        return self._elite_count

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

        self._population[identifier] = GAParticle(
            identifier=identifier, variables=variables, fitness=fitness
        )

    def delete_particle(self, identifier: str | None) -> None:
        if identifier is None:
            raise ValueError("Particle identifier cannot be None.")

        del self._population[identifier]

    def pre_iteration(self, actual_iter: int) -> None:
        if actual_iter == 0:
            n_variables = len(self._boundaries)

            if n_variables < 1:
                raise RuntimeError("GeneticAlgorithm requires at least one variable.")

            if self._mutation_probability is None:
                self._effective_mutation_probability = 1.0 / n_variables
            else:
                self._effective_mutation_probability = self._mutation_probability

            self._active_identifier = None
            return

        if self._population_method == "steady-state":
            particle_ids = tuple(self._population)

            if not particle_ids:
                raise RuntimeError("GeneticAlgorithm population cannot be empty.")

            self._active_identifier = particle_ids[
                self._rng.integers(0, len(particle_ids))
            ]
        else:
            self._active_identifier = None

    def create_random_cache(self, particle_ids: list[str], initialize: bool) -> None:
        if initialize:
            for identifier in particle_ids:
                self._population[identifier].random_cache = {}
            return

        variables = tuple(self._boundaries)

        for identifier in particle_ids:
            if (
                self._population_method == "steady-state"
                and identifier != self._active_identifier
            ):
                self._population[identifier].random_cache = {}
                continue

            cache: dict[str, Serializable] = {"crossover": self._rng.random()}

            if self._selection == "tournament":
                for parent_index in range(2):
                    for tournament_index in range(self._tournament_size):
                        cache[f"tournament-{parent_index}-{tournament_index}"] = int(
                            self._rng.integers(0, len(self._population))
                        )
            else:
                cache["selection-0"] = self._rng.random()
                cache["selection-1"] = self._rng.random()

            for variable in variables:
                if self._crossover_method == "blx":
                    cache[f"{variable}-blx"] = self._rng.random()
                elif self._crossover_method == "sbx":
                    cache[f"{variable}-sbx"] = self._rng.random()
                    cache[f"{variable}-sbx-child"] = self._rng.random()

                cache[f"{variable}-mutation"] = self._rng.random()

                if self._mutation == "gaussian":
                    cache[f"{variable}-gaussian"] = self._rng.standard_normal()
                else:
                    cache[f"{variable}-polynomial"] = self._rng.random()

            self._population[identifier].random_cache = cache

    def initialize_particle(self, identifier: str) -> GAParticle:
        particle = self._population[identifier]

        if not isinstance(particle, GAParticle):
            raise TypeError(
                f"Particle '{identifier}' must be an instance of GAParticle."
            )

        if np.isinf(particle.fitness):
            particle.update(
                variables=particle.variables, fitness_function=self._fitness_function
            )

        return particle

    @staticmethod
    def consolidate_new_particles(particle: GAParticle) -> GAParticle:
        particle.consolidate(consolidate_new=True)
        return particle

    def _select_parent(
        self, random_value: float | None, tournament_values: list[int] | None
    ) -> GAParticle:
        particles = list(self._population.values())

        if self._selection == "tournament":
            if tournament_values is None:
                raise RuntimeError("Tournament selection values are missing.")

            candidates = [particles[index] for index in tournament_values]
            return min(candidates, key=lambda candidate: candidate.fitness)

        if random_value is None:
            raise RuntimeError("Selection random value is missing.")

        probabilities = self._selection_probabilities()
        cumulative = np.cumsum(probabilities)
        cumulative[-1] = 1.0
        index = int(np.searchsorted(cumulative, random_value, side="right"))
        return particles[index]

    def _selection_probabilities(self) -> np.ndarray:
        particles = list(self._population.values())
        fitness = np.asarray([particle.fitness for particle in particles], dtype=float)
        fitness = np.where(np.isnan(fitness), np.inf, fitness)

        if self._selection == "fitness":
            negative_infinity = np.isneginf(fitness)

            if np.any(negative_infinity):
                probabilities = negative_infinity.astype(float)
            else:
                finite = np.isfinite(fitness)

                if not np.any(finite):
                    return np.full(len(particles), 1.0 / len(particles))

                worst = np.max(fitness[finite])
                weights = np.zeros(len(particles), dtype=float)
                weights[finite] = worst - fitness[finite]
                weights[finite] += np.finfo(float).eps

                total = np.sum(weights)

                if not np.isfinite(total) or total <= 0:
                    return np.full(len(particles), 1.0 / len(particles))

                probabilities = weights
        else:
            order = np.argsort(fitness, kind="stable")
            probabilities = np.zeros(len(particles), dtype=float)
            ranks = np.arange(len(particles), 0, -1, dtype=float)
            probabilities[order] = ranks

        total = np.sum(probabilities)

        if not np.isfinite(total) or total <= 0:
            return np.full(len(particles), 1.0 / len(particles))

        return probabilities / total

    def _crossover(
        self, parent_1: GAParticle, parent_2: GAParticle, particle: GAParticle
    ) -> dict[str, float]:
        variables: dict[str, float] = {}
        crossover = float(particle.random_cache.pop("crossover"))

        if crossover >= self._crossover_probability:
            for name in self._boundaries:
                if self._crossover_method == "blx":
                    particle.random_cache.pop(f"{name}-blx")
                elif self._crossover_method == "sbx":
                    particle.random_cache.pop(f"{name}-sbx")
                    particle.random_cache.pop(f"{name}-sbx-child")

            return dict(parent_1.variables)

        for name, (lower, upper) in self._boundaries.items():
            first = parent_1.variables[name]
            second = parent_2.variables[name]

            if self._crossover_method == "arithmetic":
                alpha = self._arithmetic_alpha
                value = alpha * first + (1.0 - alpha) * second
            elif self._crossover_method == "blx":
                minimum = min(first, second)
                maximum = max(first, second)
                span = maximum - minimum
                extension = self._blx_alpha * span
                random_value = float(particle.random_cache.pop(f"{name}-blx"))
                value = (
                    random_value * (maximum + extension - minimum + extension)
                    + minimum
                    - extension
                )
            else:
                value = self._sbx_value(
                    first=first,
                    second=second,
                    lower=lower,
                    upper=upper,
                    random_value=float(particle.random_cache.pop(f"{name}-sbx")),
                    upper_child=float(particle.random_cache.pop(f"{name}-sbx-child"))
                    >= 0.5,
                )

            variables[name] = float(np.clip(value, lower, upper))

        return variables

    def _sbx_value(
        self,
        first: float,
        second: float,
        lower: float,
        upper: float,
        random_value: float,
        upper_child: bool,
    ) -> float:
        if np.isclose(first, second):
            return 0.5 * (first + second)

        first, second = sorted((first, second))
        difference = second - first

        if upper_child:
            beta = 1.0 + 2.0 * (upper - second) / difference
            alpha = 2.0 - beta ** -(self._sbx_eta + 1.0)

            if random_value <= 1.0 / alpha:
                beta_q = (random_value * alpha) ** (1.0 / (self._sbx_eta + 1.0))
            else:
                beta_q = (1.0 / (2.0 - random_value * alpha)) ** (
                    1.0 / (self._sbx_eta + 1.0)
                )

            return 0.5 * ((first + second) + beta_q * difference)

        beta = 1.0 + 2.0 * (first - lower) / difference
        alpha = 2.0 - beta ** -(self._sbx_eta + 1.0)

        if random_value <= 1.0 / alpha:
            beta_q = (random_value * alpha) ** (1.0 / (self._sbx_eta + 1.0))
        else:
            beta_q = (1.0 / (2.0 - random_value * alpha)) ** (
                1.0 / (self._sbx_eta + 1.0)
            )

        return 0.5 * ((first + second) - beta_q * difference)

    def _mutate(
        self, variables: dict[str, float], particle: GAParticle
    ) -> dict[str, float]:
        if self._effective_mutation_probability is None:
            raise RuntimeError("Mutation probability has not been initialized.")

        mutated = dict(variables)

        for name, (lower, upper) in self._boundaries.items():
            mutation = particle.random_cache.pop(f"{name}-mutation")

            if mutation >= self._effective_mutation_probability:
                if self._mutation == "gaussian":
                    particle.random_cache.pop(f"{name}-gaussian")
                else:
                    particle.random_cache.pop(f"{name}-polynomial")
                continue

            span = upper - lower

            if self._mutation == "gaussian":
                noise = particle.random_cache.pop(f"{name}-gaussian")
                delta = self._mutation_intensity * self._gaussian_sigma * span * noise
            else:
                random_value = particle.random_cache.pop(f"{name}-polynomial")

                if random_value <= 0.5:
                    delta = (2.0 * random_value) ** (
                        1.0 / (self._polynomial_eta + 1.0)
                    ) - 1.0
                else:
                    delta = 1.0 - (2.0 * (1.0 - random_value)) ** (
                        1.0 / (self._polynomial_eta + 1.0)
                    )

                delta *= self._mutation_intensity * span

            mutated[name] = float(np.clip(mutated[name] + delta, lower, upper))

        return mutated

    def update_particle(self, identifier: str) -> GAParticle:
        particle = self._population[identifier]

        if not isinstance(particle, GAParticle):
            raise TypeError(
                f"Particle '{identifier}' must be an instance of GAParticle."
            )

        if (
            self._population_method == "steady-state"
            and identifier != self._active_identifier
        ):
            particle.candidate_variables = dict(particle.variables)
            particle.candidate_fitness = particle.fitness
            return particle

        tournament_values_0 = None
        tournament_values_1 = None

        if self._selection == "tournament":
            tournament_values_0 = [
                int(particle.random_cache.pop(f"tournament-0-{index}"))
                for index in range(self._tournament_size)
            ]
            tournament_values_1 = [
                int(particle.random_cache.pop(f"tournament-1-{index}"))
                for index in range(self._tournament_size)
            ]

        selection_random_0 = None
        selection_random_1 = None

        if self._selection != "tournament":
            selection_random_0 = float(particle.random_cache.pop("selection-0"))
            selection_random_1 = float(particle.random_cache.pop("selection-1"))

        parent_1 = self._select_parent(
            random_value=selection_random_0, tournament_values=tournament_values_0
        )
        parent_2 = self._select_parent(
            random_value=selection_random_1, tournament_values=tournament_values_1
        )

        candidate_variables = self._crossover(
            parent_1=parent_1, parent_2=parent_2, particle=particle
        )
        candidate_variables = self._mutate(
            variables=candidate_variables, particle=particle
        )

        particle.update(
            variables=candidate_variables, fitness_function=self._fitness_function
        )

        return particle

    def _elite_identifiers(self) -> set[str]:
        if self._survival != "elitism":
            return set()

        elite_count = min(self._elite_count, len(self._population))
        ordered = sorted(
            self._population.values(),
            key=lambda particle: (np.isnan(particle.fitness), particle.fitness),
        )

        return {particle.identifier for particle in ordered[:elite_count]}

    def post_iteration(self, actual_iter: int) -> None:
        if actual_iter == 0:
            self.update_solution_state()
            return

        elite_identifiers = self._elite_identifiers()

        for particle in self._population.values():
            if not isinstance(particle, GAParticle):
                raise TypeError(
                    f"Particle '{particle.identifier}' must be an instance of "
                    "GAParticle."
                )

            if particle.candidate_fitness is None:
                raise RuntimeError(
                    f"Particle '{particle.identifier}' has no candidate fitness."
                )

            if particle.identifier in elite_identifiers:
                particle.candidate_variables = None
                particle.candidate_fitness = FITNESS_UNDEFINED
                continue

            particle.consolidate(consolidate_new=True)

        self.update_solution_state()
