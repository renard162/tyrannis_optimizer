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
        convergence_exponent: float = 1.0,
    ) -> None:
        """
        Whale Optimization Algorithm (WOA).

        WOA is a population-based, derivative-free optimization algorithm inspired by
        the bubble-net hunting behavior of humpback whales. The algorithm combines
        global exploration through searches around randomly selected solutions with
        local exploitation around the best solution found so far, while also using a
        spiral movement to model the bubble-net feeding strategy. A nonlinear
        convergence factor controls the transition between exploration and exploitation,
        allowing the duration of the exploratory phase to be adjusted through a
        convergence exponent.

        Parameters
        ----------
        spiral_coefficient : float, default=1.0
            Positive coefficient controlling the shape of the logarithmic spiral used
            during the bubble-net feeding phase. Larger values produce stronger radial
            expansion or contraction around the best solution, while smaller values
            produce a tighter spiral. A value of 1.0 is the conventional choice in WOA
            formulations. The value must also keep the exponential term of the spiral
            representable by the floating-point type used by Tyrannis.

        convergence_exponent : float, default=1.0
            Positive exponent controlling the nonlinear decrease of the convergence
            factor. A value of 1.0 reproduces the linear convergence schedule of the
            original WOA. Values greater than 1.0 keep the convergence factor larger
            for a greater portion of the optimization, extending the exploratory phase
            before stronger exploitation begins. Values between 0 and 1.0 cause the
            convergence factor to decrease more rapidly, favoring exploitation earlier
            in the optimization.

        Notes
        -----
        WOA models each candidate solution as a whale and treats the best solution
        found by the population as the estimated position of the prey. At each
        iteration, whales update their positions according to one of three hunting
        behaviors: shrinking encircling, spiral bubble-net feeding, or search for prey.

        The balance between exploration and exploitation is controlled by the
        convergence factor ``a`` and the coefficient vector

            A = 2 * a * r1 - a

        where ``r1`` is a random vector in ``[0, 1]`` with one independently sampled
        value per search dimension. A second coefficient vector is defined as

            C = 2 * r2

        where ``r2`` is another independently sampled random vector in ``[0, 1]``.

        When the randomly selected probability ``p`` is smaller than 0.5, the whale
        uses either the encircling or search behavior. A single behavior is selected
        for the whole whale from the infinity norm of ``A``,

            ||A||_inf = max_j |A_j|.

        If ``||A||_inf < 1``, the current whale moves toward the best solution
        according to

            D = |C * X_best - X|

            X(t + 1) = X_best - A * D

        This shrinking-encircling behavior progressively concentrates the population
        around promising regions of the search space as ``a`` decreases.

        When ``p < 0.5`` and ``||A||_inf >= 1``, the whale instead uses one randomly
        selected whale as its reference vector:

            D = |C * X_random - X|

            X(t + 1) = X_random - A * D

        The same randomly selected whale is used for all dimensions of the update.
        This prevents the search from being restricted to the current best solution
        and provides the main exploration mechanism of WOA.

        When ``p >= 0.5``, the whale performs the spiral bubble-net movement around
        the best solution. The distance to the best solution is calculated as

            D' = |X_best - X|

        and the new position is obtained from

            X(t + 1) = D' * exp(b * l) * cos(2 * pi * l) + X_best

        where ``b`` is ``spiral_coefficient``. Following the MATLAB implementation
        released by the original author, an auxiliary parameter ``a2`` decreases
        linearly from -1 toward -2 over the optimization updates,

            a2(t) = -1 - t / T,

        and ``l`` is sampled from ``[a2(t), 1]``. This differs from the fixed
        ``[-1, 1]`` interval stated in the original article and follows the behavior
        implemented and documented in the author's source code.

        The convergence factor is defined by the nonlinear schedule

            a(t) = 2 * (1 - (t / T) ** p)

        where ``T`` is the total number of optimization updates, ``p`` is
        ``convergence_exponent``, and ``t`` is the zero-based optimization-update
        index. Tyrannis reserves ``actual_iter = 0`` for population initialization,
        so the first actual particle update, at ``actual_iter = 1``, uses ``t = 0``.
        With ``p = 1``, the schedule is linear and corresponds to the original WOA
        implementation. Over the executed updates, ``t`` ranges from 0 to ``T - 1``;
        the continuous schedule reaches zero at ``t = T`` after the final update.

        With ``p > 1``, the factor remains relatively large during the early and
        middle stages of the optimization and decreases more sharply toward the end,
        extending the exploratory regime. With ``0 < p < 1``, the factor decreases
        more rapidly at the beginning and remains closer to zero later, shifting the
        exploration-exploitation transition toward earlier iterations.

        The choice of ``convergence_exponent`` therefore affects the temporal
        distribution of the search effort rather than the dimensional scale of the
        decision variables. Values greater than 1 can be useful when maintaining
        population diversity and global exploration for longer is desirable, whereas
        values below 1 can be used when faster concentration around promising
        solutions is preferred. The default ``p = 1`` preserves the behavior of the
        classical WOA and provides a natural baseline for tuning the nonlinear
        schedule.

        The best solution is updated both before and after each optimization update.
        Updating it before particle movement allows already evaluated particles received
        through migration to influence the current iteration immediately, while the
        post-iteration update incorporates improvements generated by the current
        particle movements.

        The algorithm is intended for continuous optimization problems and does not
        require gradient information. The population size determines the diversity
        available to the exploration mechanism, while the number of iterations
        determines the duration over which the convergence factor transitions from
        exploration toward exploitation. Larger populations can provide broader
        coverage of high-dimensional or multimodal search spaces, while additional
        iterations allow the nonlinear convergence schedule to complete its transition.

        References
        ----------
        Mirjalili, S., & Lewis, A. (2016). The Whale Optimization Algorithm.
        Advances in Engineering Software, 95, 51-67.
        https://doi.org/10.1016/j.advengsoft.2016.01.008

        Mirjalili, S. (2018). The Whale Optimization Algorithm (WOA) source code.
        MATLAB Central File Exchange, version 1.0.0.0.
        https://www.mathworks.com/matlabcentral/fileexchange/55667-the-whale-optimization-algorithm/files/WOA/WOA.m

        Yang, Q., Li, X., Yang, T., Wu, H., & Zhang, L. (2025). An Improved Whale
        Optimization Algorithm for the Clean Production Transformation of Automotive
        Body Painting. Biomimetics, 10(5), 273.
        https://doi.org/10.3390/biomimetics10050273

        Zhong, M., & Long, W. (2017). Whale optimization algorithm with nonlinear
        control parameter. MATEC Web of Conferences, 139, 00157.
        https://doi.org/10.1051/matecconf/201713900157

        Hussien, A. G., et al. (2019). A comprehensive survey: Whale Optimization
        Algorithm and its applications. Swarm and Evolutionary Computation, 48, 1-24.
        https://doi.org/10.1016/j.swevo.2019.03.004

        Yuan, X., Miao, Z., Liu, Z., Yan, Z., & Zhou, F. (2020). Multi-Strategy
        Ensemble Whale Optimization Algorithm and Its Application to Analog Circuits
        Intelligent Fault Diagnosis. Applied Sciences, 10(11), 3667.
        https://doi.org/10.3390/app10113667

        Liu, L., & Zhang, R. (2022). Multistrategy Improved Whale Optimization
        Algorithm and Its Application. Computational Intelligence and Neuroscience,
        2022, 3418269.
        https://doi.org/10.1155/2022/3418269
        """
        if not isinstance(spiral_coefficient, (int, float, np.number)):
            raise TypeError("spiral_coefficient must be a number.")

        try:
            spiral_coefficient_value = float(spiral_coefficient)
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError(
                "spiral_coefficient must be representable as a finite float."
            ) from error

        if not np.isfinite(spiral_coefficient_value) or spiral_coefficient_value <= 0:
            raise ValueError(
                "spiral_coefficient must be a finite number greater than 0."
            )

        max_spiral_coefficient = float(np.log(np.finfo(float).max))
        if spiral_coefficient_value > max_spiral_coefficient:
            raise ValueError(
                "spiral_coefficient is too large to keep the exponential spiral "
                "term finite."
            )

        if not isinstance(convergence_exponent, (int, float, np.number)):
            raise TypeError("convergence_exponent must be a number.")

        if not np.isfinite(convergence_exponent) or convergence_exponent <= 0:
            raise ValueError(
                "convergence_exponent must be a finite number greater than 0."
            )

        self._spiral_coefficient = spiral_coefficient_value
        self._convergence_exponent = float(convergence_exponent)
        self._a = 2.0
        self._a2 = -1.0

    @property
    def spiral_coefficient(self) -> float:
        return self._spiral_coefficient

    @property
    def convergence_exponent(self) -> float:
        return self._convergence_exponent

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
            self._a2 = -1.0
            return

        self.update_solution_state()

        if self._max_iterations <= 0:
            self._a = 0.0
            self._a2 = -2.0
            return

        iteration = min(max(actual_iter - 1, 0), self._max_iterations - 1)
        progress = iteration / self._max_iterations

        self._a = 2.0 * (1.0 - progress**self._convergence_exponent)
        self._a2 = -1.0 - progress

    def create_random_cache(self, particle_ids: list[str], initialize: bool) -> None:
        for identifier in particle_ids:
            cache: dict[str, Serializable] = {}

            if not initialize:
                cache["p"] = float(self._rng.random())
                cache["l"] = float(self._rng.uniform(self._a2, 1.0))
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

        if particle.fitness == FITNESS_UNDEFINED:
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

        coefficient_a: dict[str, float] = {}
        coefficient_c: dict[str, float] = {}

        if p < 0.5:
            for variable in self._boundaries:
                r1 = float(particle.random_cache[f"{variable}-r1"])
                r2 = float(particle.random_cache[f"{variable}-r2"])

                coefficient_a[variable] = 2.0 * self._a * r1 - self._a
                coefficient_c[variable] = 2.0 * r2

            a_infinity_norm = max(abs(value) for value in coefficient_a.values())
            use_best_reference = a_infinity_norm < 1.0
        else:
            use_best_reference = True

        new_variables: dict[str, float] = {}

        for variable, (lower, upper) in self._boundaries.items():
            current_variable = particle.variables[variable]

            if p < 0.5:
                if use_best_reference:
                    reference_variable = best_variables[variable]
                else:
                    reference_variable = random_particle.variables[variable]

                distance = abs(
                    coefficient_c[variable] * reference_variable - current_variable
                )

                variable_value = reference_variable - coefficient_a[variable] * distance
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
