import warnings
from copy import deepcopy

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
        exploration_enhanced: bool = False,
    ) -> None:
        """
        Grey Wolf Optimization (GWO).

        GWO is a population-based, derivative-free optimization algorithm inspired
        by the social hierarchy and hunting behavior of grey wolves. The search is
        guided by the three best solutions found during the optimization, represented
        by the alpha, beta, and delta wolves, which collectively direct the remaining
        wolves toward promising regions of the search space. The algorithm uses a
        time-varying convergence parameter to control the balance between exploration
        and exploitation and is designed for continuous optimization problems.

        Parameters
        ----------
        convergence_exponent : float, default=1
            Positive finite exponent controlling the nonlinear schedule of the
            convergence parameter ``a``. With ``exploration_enhanced=False``, a value
            of ``1`` produces the linear decay used by the original GWO. Values greater
            than ``1`` maintain larger values of ``a`` for a greater portion of the
            optimization, while values between ``0`` and ``1`` accelerate the initial
            decay. With ``exploration_enhanced=True``, the EEGWO convergence schedule
            is used instead. The published EEGWO formulation uses
            ``convergence_exponent=1.5``; other values are accepted as configurable
            extensions and produce a warning indicating the published value.

        exploration_enhanced : bool, default=False
            Whether to use the exploration-enhanced grey wolf optimizer (EEGWO).
            When ``False``, the original GWO convergence schedule and position update
            are used. When ``True``, both the nonlinear EEGWO convergence schedule and
            its exploration-enhanced position update are enabled. The EEGWO update
            introduces a randomly selected individual distinct from the current wolf
            in addition to the alpha, beta, and delta guidance.

        Notes
        -----
        GWO models the hunting behavior of a grey wolf pack through a hierarchy in
        which the alpha, beta, and delta wolves represent the three best historical
        solutions and guide the remaining wolves. For each leading wolf, the distance
        from the current wolf is calculated as

            D = |C * X_leader - X|

        and the corresponding candidate position is calculated as

            X_leader - A * D

        where

            A = 2 * a * r1 - a
            C = 2 * r2

        with ``r1`` and ``r2`` uniformly distributed random values in ``[0, 1]`` for
        each search dimension and leader.

        The parameter ``a`` controls the transition between exploration and
        exploitation. For the standard GWO mode, this implementation generalizes the
        original linear schedule using

            a = 2 * (1 - (t / T) ** convergence_exponent)

        where ``t`` is the zero-based optimization-update index and ``T`` is the
        maximum number of optimization updates. Thus, the first actual particle
        update uses ``t = 0`` even though Tyrannis reserves ``actual_iter = 0`` for
        particle initialization. With ``convergence_exponent = 1``, this reproduces
        the linear convergence schedule of the original GWO implementation.

        Increasing the exponent above 1 slows the decay of ``a`` during the early and
        intermediate stages of the GWO optimization, preserving larger movement
        coefficients and extending the exploratory phase. Values between 0 and 1
        produce the opposite effect, causing ``a`` to decrease more rapidly at the
        beginning of the optimization.

        For each wolf, the three candidate positions generated from the alpha, beta,
        and delta wolves are combined to obtain the next position. The three leaders
        are maintained as historical snapshots independently from the active
        population, so movement or migration of the particles that originated those
        solutions does not discard the leadership information. The resulting position
        is constrained to the configured search boundaries.

        When ``exploration_enhanced`` is enabled, the algorithm uses EEGWO. Its
        convergence parameter follows

            a = 2 * (1 - ((T - t) / T) ** convergence_exponent)

        and its published formulation uses ``convergence_exponent = 1.5``. The next
        position is calculated from the mean of the three GWO candidate positions and
        a direction defined by another randomly selected wolf:

            X(t + 1) = b1 * r3 * X_mean + b2 * r4 * (X_random - X)

        where ``b1 = 0.1``, ``b2 = 0.9``, ``r3`` and ``r4`` are independently sampled
        in ``[0, 1]`` for each search dimension, and ``X_random`` is different from
        the wolf being updated. This additional direction introduces information from
        outside the three historical leaders and increases population diversity.

        This Tyrannis implementation requires at least five active particles. The
        original GWO formulation is structurally defined by three distinct leaders;
        the higher minimum used here is an architectural constraint of Tyrannis that
        guarantees the three leaders together with at least two additional population
        members, providing a fixed minimum compatible with population migration and
        the EEGWO extension.

        GWO is intended for continuous optimization and can be applied to non-linear,
        non-convex, multi-modal, noisy, and non-differentiable objective functions.
        The ``convergence_exponent`` should be selected according to the desired
        exploration-exploitation schedule rather than the dimensionality of the
        optimization problem, while ``exploration_enhanced`` is a structural choice
        between the original GWO and EEGWO formulations.

        References
        ----------
        Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey Wolf Optimizer.
        Advances in Engineering Software, 69, 46-61.
        https://doi.org/10.1016/j.advengsoft.2013.12.007

        Huang, Y., Liu, Q., Song, H., Han, T., & Li, T. (2024). CMGWO: Grey wolf
        optimizer for fusion cell-like P systems. Heliyon, 10(14), e34496.
        https://doi.org/10.1016/j.heliyon.2024.e34496

        Long, W., Jiao, J., Liang, X., & Cai, S. (2018). An exploration-enhanced
        grey wolf optimizer to solve high-dimensional numerical optimization.
        Engineering Applications of Artificial Intelligence, 68, 63-80.
        https://doi.org/10.1016/j.engappai.2017.10.024

        Faris, H., Aljarah, I., Al-Betar, M. A., & Mirjalili, S. (2018). Grey wolf
        optimizer: a review of recent variants and applications. Neural Computing
        and Applications, 30, 413-435.
        https://doi.org/10.1007/s00521-017-3272-5

        Nadimi-Shahraki, M. H., Taghian, S., & Mirjalili, S. (2021). An improved
        grey wolf optimizer for solving engineering problems. Expert Systems with
        Applications, 166, 113917.
        https://doi.org/10.1016/j.eswa.2020.113917

        Khan, A. S., et al. (2018). A new Non-Dominated Sorting Grey Wolf Optimizer
        (NS-GWO) algorithm: Development and application to solve engineering
        designs and economic constrained emission dispatch problem with integration
        of wind power. Engineering Applications of Artificial Intelligence, 72,
        449-467.
        https://doi.org/10.1016/j.engappai.2018.04.018
        """
        if not isinstance(convergence_exponent, (int, float, np.number)):
            raise TypeError("convergence_exponent must be a number.")

        if not np.isfinite(convergence_exponent) or convergence_exponent <= 0:
            raise ValueError(
                "convergence_exponent must be a finite number greater than 0."
            )

        if not isinstance(exploration_enhanced, (bool, np.bool_)):
            raise TypeError("exploration_enhanced must be a boolean.")

        self._convergence_exponent = float(convergence_exponent)
        self._exploration_enhanced = bool(exploration_enhanced)

        if self._exploration_enhanced and self._convergence_exponent != 1.5:
            warnings.warn(
                "exploration_enhanced=True enables EEGWO mode, whose published "
                "formulation uses convergence_exponent=1.5; received "
                f"convergence_exponent={self._convergence_exponent}.",
                category=UserWarning,
                stacklevel=2,
            )

        self._a = 0.0
        self._alpha: GreyWolfParticle | None = None
        self._beta: GreyWolfParticle | None = None
        self._delta: GreyWolfParticle | None = None

    @property
    def alpha(self) -> str | None:
        if self._alpha is None:
            return None
        return self._alpha.identifier

    @property
    def beta(self) -> str | None:
        if self._beta is None:
            return None
        return self._beta.identifier

    @property
    def delta(self) -> str | None:
        if self._delta is None:
            return None
        return self._delta.identifier

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
            identifier=identifier, variables=variables, fitness=fitness
        )

    def delete_particle(self, identifier: str | None) -> None:
        if identifier is None:
            raise ValueError("Particle identifier cannot be None.")

        del self._population[identifier]

    def _update_leaders(self) -> None:
        candidates: dict[str, GreyWolfParticle] = {}

        for leader in (self._alpha, self._beta, self._delta):
            if leader is not None:
                candidates[leader.identifier] = deepcopy(leader)

        for particle in self._population.values():
            if not isinstance(particle, GreyWolfParticle):
                raise TypeError(
                    f"Particle '{particle.identifier}' must be an instance "
                    "of GreyWolfParticle."
                )

            historical_particle = candidates.get(particle.identifier)
            if (
                historical_particle is None
                or particle.fitness < historical_particle.fitness
            ):
                candidates[particle.identifier] = deepcopy(particle)

        leaders = sorted(candidates.values(), key=lambda particle: particle.fitness)

        if len(leaders) < 3:
            raise RuntimeError(
                "Grey Wolf Optimization requires three distinct leaders."
            )

        self._alpha = leaders[0]
        self._beta = leaders[1]
        self._delta = leaders[2]

    def pre_iteration(self, actual_iter: int) -> None:
        if len(self._population) < 5:
            raise ValueError(
                "Grey Wolf Optimization requires at least 5 active particles "
                "in Tyrannis."
            )

        if actual_iter == 0:
            self._a = 0.0
            return

        self._update_leaders()

        if self._max_iterations <= 0:
            self._a = 0.0
            return

        iteration = min(max(actual_iter - 1, 0), self._max_iterations - 1)
        progress = iteration / self._max_iterations

        if self._exploration_enhanced:
            self._a = 2.0 * (1.0 - (1.0 - progress) ** self._convergence_exponent)
        else:
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
                        cache[f"{variable}-r3"] = self._rng.random()
                        cache[f"{variable}-r4"] = self._rng.random()

                if self._exploration_enhanced:
                    random_identifiers = [
                        particle_id
                        for particle_id in self._population
                        if particle_id != identifier
                    ]
                    random_index = int(self._rng.integers(0, len(random_identifiers)))
                    cache["random_particle"] = random_identifiers[random_index]

            self._population[identifier].random_cache = cache

    def initialize_particle(self, identifier: str) -> GreyWolfParticle:
        particle = self._population[identifier]

        if not isinstance(particle, GreyWolfParticle):
            raise TypeError(
                f"Particle '{identifier}' must be an instance of GreyWolfParticle."
            )

        if particle.fitness == FITNESS_UNDEFINED:
            particle.update(
                variables=particle.variables, fitness_function=self._fitness_function
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

        leaders = {"alpha": self._alpha, "beta": self._beta, "delta": self._delta}

        random_particle = None

        if self._exploration_enhanced:
            random_identifier = str(particle.random_cache["random_particle"])

            if random_identifier == identifier:
                raise RuntimeError(
                    "EEGWO random particle must differ from the particle being updated."
                )

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

                r3 = particle.random_cache[f"{variable}-r3"]
                r4 = particle.random_cache[f"{variable}-r4"]

                variable_value = (
                    0.1 * r3 * leader_position + 0.9 * r4 * random_direction
                )
            else:
                variable_value = leader_position

            new_variables[variable] = float(np.clip(variable_value, lower, upper))

        particle.update(
            variables=new_variables, fitness_function=self._fitness_function
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
