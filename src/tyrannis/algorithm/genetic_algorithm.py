import numpy as np

from ..core.algorithm import (
    FITNESS_UNDEFINED,
    AlgorithmBase,
    ParticleBase,
)


class GAParticle(ParticleBase):
    """Particle implementation for the Genetic Algorithm."""


class GeneticAlgorithm(AlgorithmBase[GAParticle]):
    """Real-coded Genetic Algorithm for continuous optimization."""

    def __init__(
        self,
        population_method: str = "steady-state",
        steady_state_fraction: float = 0.1,
        selection: str = "tournament",
        tournament_size: int = 2,
        crossover: str = "sbx",
        crossover_probability: float = 0.9,
        arithmetic_alpha: float = 0.5,
        blx_alpha: float = 0.5,
        sbx_eta: float = 20.0,
        mutation: str = "polynomial",
        polynomial_eta: float = 20.0,
        gaussian_sigma: float = 0.1,
        mutation_probability: float | None = None,
        mutation_intensity: float = 1.0,
        survival: str = "elitism",
        elite_count: int = 1,
    ) -> None:
        """
        Real-coded Genetic Algorithm for continuous optimization.

        The Genetic Algorithm (GA) is a population-based, derivative-free optimization
        algorithm that evolves a population of candidate solutions through parent
        selection, crossover, mutation, and survivor selection. This implementation
        uses a real-valued representation designed for continuous optimization
        problems, allowing different selection, crossover, mutation, and population
        evolution strategies to be combined according to the characteristics of the
        optimization problem.

        Parameters
        ----------
        population_method : {"generational", "steady-state"}, default="steady-state"
            Defines how the population is evolved at each iteration.

            - ``"generational"``: replaces the entire population with new candidates.
            - ``"steady-state"``: progressively replaces a fraction of the population
            defined by ``steady_state_fraction``.

        steady_state_fraction : float, default=0.1
            Defines the fraction of the population modified at each steady-state
            iteration. The value must be between 0 and 1. A value of 0 deliberately
            selects exactly one particle per iteration. Selected particles are not
            repeated until the entire replacement-eligible population has been
            covered. With elitist survival, elite individuals are excluded from
            replacement and the number of updates is capped by the number of
            non-elite individuals.

        selection : {"tournament", "fitness", "ranking"}, default="tournament"
            Defines how parents are selected from the population.

            - ``"tournament"``: selects the best individual among randomly sampled
            groups of candidates.
            - ``"fitness"``: assigns selection probabilities according to fitness.
            - ``"ranking"``: assigns selection probabilities according to the relative
            ranking of individuals.

        tournament_size : int, default=2
            Defines the number of individuals competing in each tournament when
            ``selection="tournament"``.

        crossover : {"arithmetic", "blx", "sbx"}, default="sbx"
            Defines how the variables of selected parents are combined.

            - ``"arithmetic"``: combines parent variables through a weighted average.
            - ``"blx"``: samples variables from an interval surrounding the parents.
            - ``"sbx"``: generates real-valued offspring using simulated binary
            crossover.

        crossover_probability : float, default=0.9
            Defines the probability of applying crossover to a selected pair of
            parents. When crossover is not applied, the offspring inherits the
            variables of one of the selected parents.

        arithmetic_alpha : float, default=0.5
            Defines the mixing coefficient used by arithmetic crossover. The
            formulation conventionally documented in the literature uses values in
            [0, 1], where 0.5 gives equal influence to both parents. Values greater
            than 1 are supported as an extension and produce extrapolation beyond the
            parental interval before the search-space bounds are enforced.

        blx_alpha : float, default=0.5
            Defines the extension of the sampling interval used by BLX crossover.
            Larger values allow offspring to explore farther beyond the range defined
            by the parents.

        sbx_eta : float, default=20.0
            Defines the non-negative distribution index of simulated binary crossover.
            Larger values concentrate offspring closer to the parents, while smaller
            values allow broader exploration.

        mutation : {"gaussian", "polynomial"}, default="polynomial"
            Defines how variables are perturbed after crossover.

            - ``"gaussian"``: applies normally distributed perturbations.
            - ``"polynomial"``: applies bounded perturbations controlled by a
            distribution index.

        polynomial_eta : float, default=20.0
            Defines the non-negative distribution index of polynomial mutation. Larger
            values concentrate mutations closer to the current value, while smaller
            values allow larger perturbations.

        gaussian_sigma : float, default=0.1
            Defines the standard deviation of Gaussian mutation relative to the range
            of each variable.

        mutation_probability : float or None, default=None
            Defines the probability of mutating each variable. When ``None``, the
            probability is set to ``1 / n_variables``.

        mutation_intensity : float, default=1.0
            Defines a multiplicative factor controlling the magnitude of mutation.

        survival : {"elitism", "replacement"}, default="elitism"
            Defines how generated candidates are incorporated into the population.

            - ``"elitism"``: preserves the configured number of best individuals.
            - ``"replacement"``: accepts generated candidates without elitist
            protection.

        elite_count : int, default=1
            Defines the requested number of best individuals protected when
            ``survival="elitism"``. The effective elite count is dynamically limited
            to at most ``population_size - 1``, guaranteeing at least one non-elite
            individual.

        Notes
        -----
        The Genetic Algorithm maintains a population of real-valued candidate
        solutions. Unlike binary genetic algorithms, the optimization variables are
        represented directly in their continuous domain, avoiding an encoding and
        decoding step.

        At each iteration, parent solutions are selected from the current population.
        The selection strategy controls the pressure applied toward better solutions.
        Tournament selection compares randomly sampled groups of individuals and
        selects the best one from each group. Fitness selection assigns a greater
        probability to individuals with better objective values, while ranking
        selection determines selection probabilities from the relative ordering of
        the population rather than from the magnitude of the objective values.

        Selected parents are combined using the configured crossover operator.
        Arithmetic crossover conventionally generates values between the parents
        when ``arithmetic_alpha`` is in [0, 1]. Values greater than 1 are supported as
        an extension and extrapolate beyond the parental interval before the
        search-space bounds are enforced. BLX-alpha samples values from an interval
        surrounding the range defined by the parents, allowing offspring to explore
        beyond their current values. Simulated binary crossover generates real-valued
        offspring with a distribution controlled by ``sbx_eta`` and provides a flexible
        balance between exploration around and exploitation of the parent solutions.

        After crossover, mutation can modify each variable independently. Gaussian
        mutation introduces normally distributed perturbations whose scale depends
        on the variable range and ``gaussian_sigma``. Polynomial mutation produces
        bounded perturbations whose distribution is controlled by ``polynomial_eta``.
        The ``mutation_probability`` determines how frequently variables are changed,
        while ``mutation_intensity`` controls the magnitude of those changes.

        The population can evolve according to two different strategies. In
        generational evolution, the entire population is replaced by newly generated
        candidates at each iteration. In steady-state evolution, only a fraction of
        the population is modified at a time. The fraction is controlled by
        ``steady_state_fraction``. A value of 0 deliberately selects exactly one
        individual per iteration. Replacement-eligible individuals are covered
        without repetition before a new selection cycle begins; if an iteration
        crosses a cycle boundary, the next cycle immediately supplies the remaining
        updates. Newly arrived migration particles are incorporated into the current
        cycle. With elitist survival, elite individuals are excluded from replacement.

        Survivor selection determines which generated candidates become part of the
        population. With elitist survival, the best ``elite_count`` individuals are
        preserved, preventing the best solutions found so far from being replaced.
        Elites remain eligible for parent selection but are excluded from steady-state
        replacement, and the effective elite count is limited to at most one less than
        the current population size. With replacement survival, generated candidates
        are accepted without this elitist protection.

        The Genetic Algorithm is a derivative-free method and does not require
        gradient information from the objective function. Its real-valued
        representation makes it suitable for bounded continuous optimization
        problems, particularly when the objective function is nonlinear,
        non-convex, multimodal, noisy, or non-differentiable.

        The different selection, crossover, mutation, population evolution, and
        survivor strategies can be combined to adjust the balance between exploration
        and exploitation. In particular, ``crossover_probability``,
        ``mutation_probability``, ``mutation_intensity``, and the distribution
        indices of the selected operators provide direct control over the size and
        frequency of changes introduced into the population.

        References
        ----------
        Eshelman, L. J., & Schaffer, J. D. (1993). Real-coded genetic algorithms and
        interval-schemata. Foundations of Genetic Algorithms, 2, 187-202.
        https://doi.org/10.1016/B978-0-08-094832-4.50018-0

        García-Martínez, C., Lozano, M., Herrera, F., Molina, D., & Sánchez, A. M.
        (2008). Global and local real-coded genetic algorithms based on parent-centric
        crossover operators. European Journal of Operational Research, 185(3),
        1088-1113.
        https://doi.org/10.1016/j.ejor.2006.06.043

        Deep, K., & Thakur, M. (2007). A new crossover operator for real coded genetic
        algorithms. Applied Mathematics and Computation, 188(1), 895-911.
        https://doi.org/10.1016/j.amc.2006.10.047

        Subbaraj, P., & Rajnarayanan, P. N. (2009). Optimal reactive power dispatch
        using self-adaptive real coded genetic algorithm. Electric Power Systems
        Research, 79(2), 374-381.
        https://doi.org/10.1016/j.epsr.2008.07.008

        Singh, V., Sharma, S. K., & Vaibhav, S. (2016). Transport aircraft conceptual
        design optimization using real coded genetic algorithm. International Journal
        of Aerospace Engineering, 2016, 2813541.
        https://doi.org/10.1155/2016/2813541

        Pal, P., Das, C. B., Panda, A., & Bhunia, A. K. (2005). An application of
        real-coded genetic algorithm for mixed integer non-linear programming in an
        optimal two-warehouse inventory policy for deteriorating items with a linear
        trend in demand and a fixed planning horizon. International Journal of
        Computer Mathematics, 82(2), 163-175.
        https://doi.org/10.1080/00207160412331296733

        Deb, K., & Agrawal, R. B. (1995). Simulated binary crossover for continuous
        search space. Complex Systems, 9, 115-148.

        Deb, K., & Deb, D. (2014). Analysing mutation schemes for real-parameter
        genetic algorithms. International Journal of Artificial Intelligence and Soft
        Computing, 4(1), 1-28.
        https://doi.org/10.1504/IJAISC.2014.059280

        Yang, J.-M., & Kao, C.-Y. (1996). Combined evolutionary algorithm for real
        parameters optimization. Proceedings of the 1996 IEEE International
        Conference on Evolutionary Computation, 732-737.
        https://doi.org/10.1109/ICEC.1996.542693

        Blanco, A., Delgado, M., & Pegalajar, M. C. (2001). A real-coded genetic
        algorithm for training recurrent neural networks. Neural Networks, 14(1),
        93-105.
        https://doi.org/10.1016/S0893-6080(00)00081-2

        Wang, J., Cheng, Z., Ersoy, O. K., Zhang, P., Dai, W., & Dong, Z. (2018).
        Improvement analysis and application of real-coded genetic algorithm for
        solving constrained optimization problems. Mathematical Problems in
        Engineering, 2018, 5760841.
        https://doi.org/10.1155/2018/5760841
        """
        if population_method not in ("generational", "steady-state"):
            raise ValueError(
                "population_method must be either 'generational' or 'steady-state'."
            )

        self._validate_probability(steady_state_fraction, "steady_state_fraction")

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
        self._validate_finite_nonnegative(sbx_eta, "sbx_eta")

        if mutation not in ("gaussian", "polynomial"):
            raise ValueError("mutation must be 'gaussian' or 'polynomial'.")

        if mutation_probability is not None:
            self._validate_probability(mutation_probability, "mutation_probability")

        self._validate_finite_nonnegative(mutation_intensity, "mutation_intensity")
        self._validate_finite_positive(gaussian_sigma, "gaussian_sigma")
        self._validate_finite_nonnegative(polynomial_eta, "polynomial_eta")

        if survival not in ("elitism", "replacement"):
            raise ValueError("survival must be 'elitism' or 'replacement'.")

        if not isinstance(elite_count, (int, np.integer)):
            raise TypeError("elite_count must be an integer.")

        if elite_count < 1:
            raise ValueError("elite_count must be greater than 0.")

        self._population_method = population_method
        self._steady_state_fraction = float(steady_state_fraction)
        self._selection = selection
        self._tournament_size = int(tournament_size)

        self._crossover_method = crossover
        self._crossover_probability = float(crossover_probability)
        self._arithmetic_alpha = float(arithmetic_alpha)
        self._blx_alpha = float(blx_alpha)
        self._sbx_eta = float(sbx_eta)

        self._mutation = mutation
        self._mutation_probability = mutation_probability
        self._mutation_intensity = float(mutation_intensity)
        self._gaussian_sigma = float(gaussian_sigma)
        self._polynomial_eta = float(polynomial_eta)

        self._survival = survival
        self._elite_count = int(elite_count)

        self._effective_mutation_probability: float | None = None
        self._active_identifiers: set[str] = set()
        self._steady_state_queue: list[str] = []
        self._steady_state_cycle_identifiers: set[str] = set()

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

    @staticmethod
    def _next_power_of_ten(value: float) -> float:
        if value <= 0:
            return float(np.nextafter(0.0, 1.0))

        exponent = int(np.ceil(np.log10(value)))
        power = float(10.0**exponent)

        if power == 0.0:
            return float(np.nextafter(0.0, 1.0))

        return power

    @staticmethod
    def _ulp_toward_zero(value: float) -> float:
        if value == 0.0:
            return float(np.nextafter(0.0, 1.0))

        adjacent = np.nextafter(value, 0.0)
        return float(abs(value - adjacent))

    def _sbx_tolerance(
        self, first: float, second: float, lower: float, upper: float
    ) -> float:
        epsilon_margin = float(10.0 * np.finfo(float).eps)

        if np.signbit(lower) != np.signbit(upper):
            domain_tolerance = epsilon_margin * abs(lower) + epsilon_margin * abs(upper)
        else:
            domain_tolerance = epsilon_margin * abs(upper - lower)

        ulp_tolerance = 4.0 * max(
            self._ulp_toward_zero(first), self._ulp_toward_zero(second)
        )

        return self._next_power_of_ten(max(domain_tolerance, ulp_tolerance))

    def _start_steady_state_cycle(self, eligible_identifiers: list[str]) -> None:
        self._steady_state_cycle_identifiers = set(eligible_identifiers)
        self._steady_state_queue = eligible_identifiers.copy()
        self._rng.shuffle(self._steady_state_queue)

    def _synchronize_steady_state_cycle(self, eligible_identifiers: list[str]) -> None:
        eligible_identifier_set = set(eligible_identifiers)

        self._steady_state_queue = [
            identifier
            for identifier in self._steady_state_queue
            if identifier in eligible_identifier_set
        ]
        self._steady_state_cycle_identifiers.intersection_update(
            eligible_identifier_set
        )

        new_identifiers = [
            identifier
            for identifier in eligible_identifiers
            if identifier not in self._steady_state_cycle_identifiers
        ]

        if new_identifiers:
            self._steady_state_queue.extend(new_identifiers)
            self._steady_state_cycle_identifiers.update(new_identifiers)
            self._rng.shuffle(self._steady_state_queue)

        if not self._steady_state_queue:
            self._steady_state_cycle_identifiers = set()

    @property
    def population_method(self) -> str:
        return self._population_method

    @property
    def steady_state_fraction(self) -> float:
        return self._steady_state_fraction

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

            self._active_identifiers = set()
            self._steady_state_queue = []
            self._steady_state_cycle_identifiers = set()
            return

        if self._population_method != "steady-state":
            self._active_identifiers = set()
            return

        particle_ids = list(self._population)
        elite_identifiers = self._elite_identifiers()
        eligible_identifiers = [
            identifier
            for identifier in particle_ids
            if identifier not in elite_identifiers
        ]

        if not eligible_identifiers:
            self._active_identifiers = set()
            self._steady_state_queue = []
            self._steady_state_cycle_identifiers = set()
            return

        self._synchronize_steady_state_cycle(eligible_identifiers=eligible_identifiers)

        n_updates = max(
            1, int(np.ceil(self._steady_state_fraction * len(particle_ids)))
        )
        n_updates = min(n_updates, len(eligible_identifiers))

        active_identifiers: set[str] = set()

        while len(active_identifiers) < n_updates:
            if not self._steady_state_queue:
                self._start_steady_state_cycle(
                    eligible_identifiers=eligible_identifiers
                )

            selected_index = next(
                index
                for index, identifier in enumerate(self._steady_state_queue)
                if identifier not in active_identifiers
            )
            identifier = self._steady_state_queue[selected_index]
            self._steady_state_queue = (
                self._steady_state_queue[:selected_index]
                + self._steady_state_queue[selected_index + 1 :]
            )
            active_identifiers.add(identifier)

            if not self._steady_state_queue:
                self._steady_state_cycle_identifiers = set()

        self._active_identifiers = active_identifiers

    def create_random_cache(self, particle_ids: list[str], initialize: bool) -> None:
        for identifier in particle_ids:
            particle = self._population[identifier]
            particle.random_cache = {}

            if initialize:
                continue

            if (
                self._population_method == "steady-state"
                and identifier not in self._active_identifiers
            ):
                continue

            particle.random_cache["crossover"] = self._rng.random()
            particle.random_cache["crossover-parent"] = self._rng.random()

            if self._crossover_method != "arithmetic":
                for name in self._boundaries:
                    if self._crossover_method == "blx":
                        particle.random_cache[f"{name}-blx"] = self._rng.random()
                    else:
                        particle.random_cache[f"{name}-sbx"] = self._rng.random()
                        particle.random_cache[f"{name}-sbx-child"] = self._rng.random()

            if self._selection == "tournament":
                for parent_index in range(2):
                    for tournament_index in range(self._tournament_size):
                        particle.random_cache[
                            f"tournament-{parent_index}-{tournament_index}"
                        ] = int(self._rng.integers(0, len(self._population)))
            else:
                particle.random_cache["selection-0"] = self._rng.random()
                particle.random_cache["selection-1"] = self._rng.random()

            for name in self._boundaries:
                particle.random_cache[f"{name}-mutation"] = self._rng.random()

                if self._mutation == "gaussian":
                    particle.random_cache[f"{name}-gaussian"] = (
                        self._rng.standard_normal()
                    )
                else:
                    particle.random_cache[f"{name}-polynomial"] = self._rng.random()

    def initialize_particle(self, identifier: str) -> GAParticle:
        particle = self._population[identifier]

        if not isinstance(particle, GAParticle):
            raise TypeError(
                f"Particle '{identifier}' must be an instance of GAParticle."
            )

        if particle.fitness == FITNESS_UNDEFINED:
            particle.update(
                variables=particle.variables, fitness_function=self._fitness_function
            )

        return particle

    @staticmethod
    def consolidate_new_particles(particle: GAParticle) -> GAParticle:
        particle.consolidate(consolidate_new=True)
        return particle

    def _select_parent(
        self, particle: GAParticle, selection_random_key: str, tournament_key: str
    ) -> GAParticle:
        particles = list(self._population.values())

        if self._selection == "tournament":
            indices = [
                particle.random_cache[f"{tournament_key}-{index}"]
                for index in range(self._tournament_size)
            ]

            candidates = [particles[int(index)] for index in indices]

            return min(candidates, key=lambda candidate: candidate.fitness)

        random_value = particle.random_cache[selection_random_key]
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
            negative_infinite = np.isneginf(fitness)

            if np.any(negative_infinite):
                probabilities = negative_infinite.astype(float)
                return probabilities / np.sum(probabilities)

            finite = np.isfinite(fitness)

            if not np.any(finite):
                return np.full(len(particles), 1.0 / len(particles))

            finite_fitness = fitness[finite]
            maximum_magnitude = np.max(np.abs(finite_fitness))

            if maximum_magnitude == 0.0:
                scaled_fitness = finite_fitness
            else:
                _, exponent = np.frexp(maximum_magnitude)
                scaled_fitness = np.ldexp(finite_fitness, -int(exponent))

            worst = np.max(scaled_fitness)
            probabilities = np.zeros(len(particles), dtype=float)
            probabilities[finite] = worst - scaled_fitness

            total = np.sum(probabilities)

            if total == 0.0:
                probabilities[finite] = 1.0
                return probabilities / np.sum(probabilities)

            if not np.isfinite(total):
                raise RuntimeError(
                    "Fitness-proportionate selection produced non-finite weights."
                )

            return probabilities / total

        order = np.argsort(fitness, kind="stable")

        probabilities = np.zeros(len(particles), dtype=float)
        probabilities[order] = np.arange(len(particles), 0, -1, dtype=float)

        return probabilities / np.sum(probabilities)

    def _crossover(
        self, particle: GAParticle, parent_1: GAParticle, parent_2: GAParticle
    ) -> dict[str, float]:
        if particle.random_cache["crossover"] >= self._crossover_probability:
            if particle.random_cache["crossover-parent"] < 0.5:
                return dict(parent_1.variables)

            return dict(parent_2.variables)

        variables: dict[str, float] = {}

        for name, (lower, upper) in self._boundaries.items():
            first = parent_1.variables[name]
            second = parent_2.variables[name]

            if self._crossover_method == "arithmetic":
                value = (
                    self._arithmetic_alpha * first
                    + (1.0 - self._arithmetic_alpha) * second
                )
            elif self._crossover_method == "blx":
                minimum = min(first, second)
                maximum = max(first, second)
                span = maximum - minimum
                extension = self._blx_alpha * span
                random_value = particle.random_cache[f"{name}-blx"]

                value = minimum - extension + random_value * (span + 2.0 * extension)
            else:
                value = self._sbx_value(
                    first=first,
                    second=second,
                    lower=lower,
                    upper=upper,
                    random_value=particle.random_cache[f"{name}-sbx"],
                    upper_child=(particle.random_cache[f"{name}-sbx-child"] >= 0.5),
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
        tolerance = self._sbx_tolerance(
            first=first, second=second, lower=lower, upper=upper
        )

        if np.isclose(first, second, rtol=0.0, atol=tolerance):
            return 0.5 * first + 0.5 * second

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
            if (
                particle.random_cache[f"{name}-mutation"]
                >= self._effective_mutation_probability
            ):
                continue

            span = upper - lower

            if self._mutation == "gaussian":
                noise = particle.random_cache[f"{name}-gaussian"]

                delta = self._mutation_intensity * self._gaussian_sigma * span * noise
            else:
                random_value = particle.random_cache[f"{name}-polynomial"]
                current = mutated[name]

                delta_1 = (current - lower) / span
                delta_2 = (upper - current) / span
                mut_pow = 1.0 / (self._polynomial_eta + 1.0)

                if random_value <= 0.5:
                    xy = 1.0 - delta_1
                    value = 2.0 * random_value + (1.0 - 2.0 * random_value) * xy ** (
                        self._polynomial_eta + 1.0
                    )
                    delta = value**mut_pow - 1.0
                else:
                    xy = 1.0 - delta_2
                    value = 2.0 * (1.0 - random_value) + 2.0 * (
                        random_value - 0.5
                    ) * xy ** (self._polynomial_eta + 1.0)
                    delta = 1.0 - value**mut_pow

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
            and identifier not in self._active_identifiers
        ):
            particle.candidate_variables = dict(particle.variables)
            particle.candidate_fitness = particle.fitness
            return particle

        parent_1 = self._select_parent(
            particle=particle,
            selection_random_key="selection-0",
            tournament_key="tournament-0",
        )

        parent_2 = self._select_parent(
            particle=particle,
            selection_random_key="selection-1",
            tournament_key="tournament-1",
        )

        candidate_variables = self._crossover(
            particle=particle, parent_1=parent_1, parent_2=parent_2
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

        elite_count = min(self._elite_count, max(len(self._population) - 1, 0))

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

            if self._survival == "elitism" and particle.identifier in elite_identifiers:
                particle.consolidate(consolidate_new=False)
            else:
                particle.consolidate(consolidate_new=True)

        self.update_solution_state()
