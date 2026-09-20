from collections.abc import Collection
from copy import deepcopy

import numpy as np

from ..core.algorithm import (
    FITNESS_UNDEFINED,
    AlgorithmBase,
    ParticleBase,
    Serializable,
)


class PSOParticle(ParticleBase):
    """Particle implementation for the Particle Swarm Optimization algorithm."""

    def __init__(
        self,
        identifier: str,
        variables: dict[str, float],
        fitness: np.float64 = FITNESS_UNDEFINED,
        velocity: dict[str, float] | None = None,
        personal_best_variables: dict[str, float] | None = None,
        personal_best_fitness: np.float64 = FITNESS_UNDEFINED,
    ) -> None:
        super().__init__(identifier=identifier, variables=variables, fitness=fitness)

        self._velocity = velocity
        self._personal_best_variables = personal_best_variables
        self._personal_best_fitness = personal_best_fitness

    def __call__(self) -> dict[str, Serializable]:
        return {
            "identifier": self._identifier,
            "variables": self._variables,
            "fitness": self._fitness,
            "velocity": self._velocity,
            "personal_best_variables": self._personal_best_variables,
            "personal_best_fitness": self._personal_best_fitness,
        }

    @property
    def velocity(self) -> dict[str, float] | None:
        return self._velocity

    @velocity.setter
    def velocity(self, velocity: dict[str, float]) -> None:
        self._velocity = velocity

    @property
    def personal_best_variables(self) -> dict[str, float] | None:
        return self._personal_best_variables

    @property
    def personal_best_fitness(self) -> np.float64:
        return self._personal_best_fitness

    def update_personal_best(self) -> None:
        if (
            np.isinf(self._personal_best_fitness)
            or self.fitness < self._personal_best_fitness
        ):
            self._personal_best_variables = deepcopy(self.variables)
            self._personal_best_fitness = self.fitness


class PSO(AlgorithmBase[PSOParticle]):
    """Classical Particle Swarm Optimization algorithm."""

    def __init__(
        self,
        inertia: float | Collection[float] = 0.7,
        cognitive_coefficient: float = 1.5,
        social_coefficient: float = 1.5,
        constriction_factor: bool = False,
    ) -> None:
        """
        Particle Swarm Optimization algorithm.

        PSO is a population-based optimization algorithm in which each particle
        adjusts its velocity according to its previous velocity, its own best-known
        position, and the best-known position of the swarm. The cognitive and
        social coefficients control the influence of the particle's personal
        experience and the swarm's experience, respectively.

        See the `Particle swarm optimization
        <https://en.wikipedia.org/wiki/Particle_swarm_optimization>`_ article
        on Wikipedia for an overview of the algorithm and its development.

        Parameters
        ----------
        inertia : float or Collection[float], default=0.7
            Inertia weight applied to the particle velocity. When a single numeric
            value is provided, the inertia weight remains constant throughout the
            optimization. When a collection containing exactly two numbers is
            provided, the smaller and larger values define the minimum and maximum
            inertia weights, respectively, and the inertia weight is decreased
            linearly throughout the optimization.

        cognitive_coefficient : float, default=1.5
            Coefficient controlling the influence of each particle's personal best
            position on its velocity update. It must be a finite number greater
            than 0.

        social_coefficient : float, default=1.5
            Coefficient controlling the influence of the swarm's best-known
            position on each particle's velocity update. It must be a finite
            number greater than 0.

        constriction_factor : bool, default=False
            Whether to use the PSO constriction factor formulation. When enabled,
            the constriction factor is calculated from the sum of the cognitive
            and social coefficients and applied to the complete velocity update.
            The inertia weight is set to 1 and its provided value or dynamic range
            is completely ignored, since the constriction factor itself controls
            the contraction of the velocity update in this formulation.

            This option requires the sum of ``cognitive_coefficient`` and
            ``social_coefficient`` to be strictly greater than 4. A commonly
            used configuration is ``cognitive_coefficient=2.05`` and
            ``social_coefficient=2.05``, resulting in ``phi=4.10`` and a
            constriction factor of approximately 0.7298.

        Notes
        -----
        In the standard PSO formulation, each particle updates its velocity from
        three components: its previous velocity, the displacement toward its own
        best-known position, and the displacement toward the best-known position
        of the swarm. The resulting velocity is then used to update the particle
        position:

        ``v(t+1) = w*v(t) + c1*r1*(p(t)-x(t)) + c2*r2*(g(t)-x(t))``

        ``x(t+1) = x(t) + v(t+1)``

        where ``w`` is the inertia weight, ``c1`` and ``c2`` are the cognitive and
        social coefficients, ``r1`` and ``r2`` are random values, ``p(t)`` is the
        particle's best-known position, and ``g(t)`` is the best-known position of
        the swarm. The personal best is updated only when the particle reaches a
        position with a better objective value.

        When dynamic inertia is used, the inertia weight decreases linearly from
        the maximum specified value at the beginning of the optimization to the
        minimum specified value at the final iteration. Larger inertia values
        generally favor broader exploration of the search space, while smaller
        values progressively emphasize local refinement.

        When ``constriction_factor`` is enabled, the constriction coefficient is
        calculated as

        ``chi = 2 / abs(2 - phi - sqrt(phi**2 - 4 * phi))``

        where ``phi`` is the sum of the cognitive and social coefficients and must
        be strictly greater than 4. The coefficient is then applied to the complete
        velocity update:

        ``v(t+1) = chi * [v(t) + c1*r1*(p(t)-x(t)) + c2*r2*(g(t)-x(t))]``

        In this formulation, ``chi`` controls the overall contraction of the
        particle velocity and provides the stability mechanism derived from the
        dynamical analysis of PSO. Consequently, the inertia weight is fixed at
        1 so that it does not introduce an additional contraction factor. Any
        value or dynamic range supplied through ``inertia`` is therefore ignored
        when the constriction factor is enabled.

        The PSO formulation is based on Kennedy and Eberhart (1995). The inertia
        weight was introduced by Shi and Eberhart (1998), who studied its effect
        on the exploration and convergence behavior of PSO. The linear
        time-varying inertia strategy implemented here follows the commonly used
        time-decreasing inertia-weight approach from this line of research. The
        constriction-factor formulation is based on the stability analysis of
        Clerc and Kennedy (2002).

        References
        ----------
        Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.
        Proceedings of ICNN'95 - International Conference on Neural Networks,
        1942-1948. https://doi.org/10.1109/ICNN.1995.488968

        Shi, Y., & Eberhart, R. C. (1998). A modified particle swarm optimizer.
        Proceedings of the 1998 IEEE International Conference on Evolutionary
        Computation, 69-73. https://doi.org/10.1109/ICEC.1998.699146

        Clerc, M., & Kennedy, J. (2002). The particle swarm - explosion, stability,
        and convergence in a multidimensional complex space. IEEE Transactions
        on Evolutionary Computation, 6(1), 58-73.
        https://doi.org/10.1109/4235.985692
        """
        for name, value in (
            ("cognitive_coefficient", cognitive_coefficient),
            ("social_coefficient", social_coefficient),
        ):
            if not isinstance(value, (int, float, np.number)):
                raise TypeError(f"{name} must be a number.")

            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")

            if value <= 0:
                raise ValueError(f"{name} must be greater than 0.")

        if constriction_factor and (cognitive_coefficient + social_coefficient <= 4):
            raise ValueError(
                "The sum of cognitive_coefficient and social_coefficient "
                "must be greater than 4 to use the constriction factor."
            )

        if isinstance(inertia, Collection) and not isinstance(inertia, (str, bytes)):
            if len(inertia) != 2:
                raise ValueError("Dynamic inertia must contain exactly two values.")

            inertia_values = tuple(sorted(inertia))

            if not all(
                isinstance(value, (int, float, np.number)) and np.isfinite(value)
                for value in inertia_values
            ):
                raise ValueError("Dynamic inertia values must be finite numbers.")

            self._inertia = float(inertia_values[1])
            self._inertia_bounds = (float(inertia_values[0]), float(inertia_values[1]))
        elif isinstance(inertia, (int, float, np.number)):
            if not np.isfinite(inertia):
                raise ValueError("Inertia must be a finite number.")

            self._inertia = float(inertia)
            self._inertia_bounds = None
        else:
            raise TypeError(
                "Inertia must be a number or an ordered collection "
                "containing two numbers."
            )

        self._cognitive_coefficient = cognitive_coefficient
        self._social_coefficient = social_coefficient
        self._constriction_factor = constriction_factor

        if constriction_factor:
            phi = cognitive_coefficient + social_coefficient
            self._constriction = 2 / abs(2 - phi - np.sqrt(phi**2 - 4 * phi))
            self._inertia = 1.0
        else:
            self._constriction = 1.0

    def create_particle(
        self,
        identifier: str,
        variables: dict[str, float] | None = None,
        fitness: np.float64 = FITNESS_UNDEFINED,
        velocity: dict[str, float] | None = None,
        personal_best_variables: dict[str, float] | None = None,
        personal_best_fitness: np.float64 = FITNESS_UNDEFINED,
    ) -> None:
        if identifier is None:
            raise ValueError("Particle identifier cannot be None.")

        if variables is None:
            variables = {
                name: self._rng.uniform(lower, upper)
                for name, (lower, upper) in self._boundaries.items()
            }

        if velocity is None:
            velocity = {
                name: self._rng.uniform(-(upper - lower), upper - lower)
                for name, (lower, upper) in self._boundaries.items()
            }

        self._population[identifier] = PSOParticle(
            identifier=identifier,
            variables=variables,
            fitness=fitness,
            velocity=velocity,
            personal_best_variables=personal_best_variables,
            personal_best_fitness=personal_best_fitness,
        )

    def delete_particle(self, identifier: str | None) -> None:
        if identifier is None:
            raise ValueError("Particle identifier cannot be None.")

        del self._population[identifier]

    def pre_iteration(self, actual_iter: int) -> None:
        if self._constriction_factor or self._inertia_bounds is None:
            return

        minimum_inertia, maximum_inertia = self._inertia_bounds

        if actual_iter == 0 or self._max_iterations == 1:
            self._inertia = maximum_inertia
            return

        progress = (actual_iter - 1) / (self._max_iterations - 1)
        self._inertia = maximum_inertia - (
            (maximum_inertia - minimum_inertia) * progress
        )

    def create_random_cache(self, particle_ids: list[str], initialize: bool) -> None:
        for particle_id in particle_ids:
            particle = self._population[particle_id]

            particle.random_cache = {}
            for variable in self._boundaries:
                particle.random_cache[f"{variable}-cognitive"] = self._rng.random()
                particle.random_cache[f"{variable}-social"] = self._rng.random()

    def initialize_particle(self, identifier: str) -> PSOParticle:
        particle = self._population[identifier]

        if not isinstance(particle, PSOParticle):
            raise TypeError(
                f"Particle '{identifier}' must be an instance of PSOParticle."
            )

        if np.isinf(particle.fitness):
            particle.update(
                variables=particle.variables, fitness_function=self._fitness_function
            )

        return particle

    @staticmethod
    def consolidate_new_particles(particle: PSOParticle) -> PSOParticle:
        particle.consolidate(consolidate_new=True)
        particle.update_personal_best()
        return particle

    def update_particle(self, identifier: str) -> PSOParticle:
        particle = self._population[identifier]

        if not isinstance(particle, PSOParticle):
            raise TypeError(
                f"Particle '{identifier}' must be an instance of PSOParticle."
            )

        if self._local_best is None:
            raise RuntimeError("Local best particle has not been initialized.")

        if particle.velocity is None:
            raise RuntimeError(f"Particle '{identifier}' does not have a velocity.")

        if particle.personal_best_variables is None:
            raise RuntimeError(
                f"Particle '{identifier}' does not have a personal best."
            )

        current_variables = particle.variables
        personal_best_variables = particle.personal_best_variables
        global_best_variables = self._local_best.variables

        new_velocity = {}
        new_variables = {}

        for name, (lower, upper) in self._boundaries.items():
            current_variable = current_variables[name]
            current_velocity = particle.velocity[name]

            personal_best_variable = personal_best_variables[name]
            global_best_variable = global_best_variables[name]

            cognitive_random = particle.random_cache[(f"{name}-cognitive")]
            social_random = particle.random_cache[(f"{name}-social")]

            velocity = self._constriction * (
                self._inertia * current_velocity
                + self._cognitive_coefficient
                * cognitive_random
                * (personal_best_variable - current_variable)
                + self._social_coefficient
                * social_random
                * (global_best_variable - current_variable)
            )

            variable = np.clip(current_variable + velocity, lower, upper)

            new_velocity[name] = velocity
            new_variables[name] = variable

        particle.velocity = new_velocity

        particle.update(
            variables=new_variables, fitness_function=self._fitness_function
        )

        return particle

    def post_iteration(self, actual_iter: int) -> None:
        for particle in self._population.values():
            if not isinstance(particle, PSOParticle):
                raise TypeError(
                    f"Particle '{particle.identifier}' must be an instance of "
                    "PSOParticle."
                )

            if actual_iter == 0:
                continue

            if particle.candidate_fitness is None:
                raise RuntimeError(
                    f"Particle '{particle.identifier}' has no candidate fitness."
                )

            particle.consolidate(consolidate_new=True)
            particle.update_personal_best()

        self.update_solution_state()
