import numpy as np

from ..core.algorithm import (
    FITNESS_UNDEFINED,
    AlgorithmBase,
    ParticleBase,
    Serializable,
)


class PSOGSAParticle(ParticleBase):
    """Particle implementation for the Particle Swarm Optimization and Gravitational Search Algorithm."""

    def __init__(
        self,
        identifier: str,
        variables: dict[str, float],
        fitness: np.float64 = FITNESS_UNDEFINED,
        velocity: dict[str, float] | None = None,
        mass: float = 0.0,
        acceleration: dict[str, float] | None = None,
    ) -> None:
        super().__init__(identifier=identifier, variables=variables, fitness=fitness)

        self._velocity = velocity
        self._mass = mass
        self._acceleration = acceleration

    def __call__(self) -> dict[str, Serializable]:
        return {
            "identifier": self._identifier,
            "variables": self._variables,
            "fitness": self._fitness,
            "velocity": self._velocity,
            "mass": self._mass,
            "acceleration": self._acceleration,
        }

    @property
    def velocity(self) -> dict[str, float] | None:
        return self._velocity

    @velocity.setter
    def velocity(self, velocity: dict[str, float]) -> None:
        self._velocity = velocity

    @property
    def mass(self) -> float:
        return self._mass

    @mass.setter
    def mass(self, mass: float) -> None:
        self._mass = mass

    @property
    def acceleration(self) -> dict[str, float] | None:
        return self._acceleration

    @acceleration.setter
    def acceleration(self, acceleration: dict[str, float]) -> None:
        self._acceleration = acceleration


class PSOGSA(AlgorithmBase[PSOGSAParticle]):
    """Particle Swarm Optimization and Gravitational Search Algorithm."""

    def __init__(
        self,
        c1: float = 0.5,
        c2: float = 1.5,
        g_zero: float = 1.0,
        alpha: float = 20.0,
        r_norm: float = 2.0,
        r_power: float = 1.0,
        k_agents_percent: float = 0.0,
    ) -> None:
        """
        Particle Swarm Optimization and Gravitational Search Algorithm (PSOGSA).

        PSOGSA combines Particle Swarm Optimization with the
        Gravitational Search Algorithm. Particle velocities are updated using
        the previous velocity, the gravitational acceleration produced by the
        population, and the direction toward the best solution found so far.

        Parameters
        ----------
        c1 : float, default=0.5
            Coefficient controlling the influence of the gravitational acceleration
            on the particle velocity.

        c2 : float, default=1.5
            Coefficient controlling the influence of the global best solution on the
            particle velocity.

        g_zero : float, default=1.0
            Initial value of the gravitational constant.

        alpha : float, default=20.0
            Descending coefficient of the gravitational constant.

        r_norm : float, default=2.0
            Order of the norm used to calculate the distance between particles.
            The default is the Euclidean norm.

        r_power : float, default=1.0
            Power applied to the inter-particle distance in the gravitational force
            denominator.

        k_agents_percent : float, default=0.0
            Minimum fraction of the population considered in the gravitational
            interaction. The effective minimum is always one particle.

        Notes
        -----
        The gravitational force is calculated from the masses of the particles,
        the gravitational constant, the distance between particles, and a random
        factor. The resulting acceleration is combined with the previous velocity
        and the global-best attraction to update each particle.

        The gravitational constant decreases exponentially during the optimization.

        The algorithm minimizes the fitness function: lower fitness values represent
        better solutions.

        References
        ----------
        Mirjalili, S., & Hashim, S. Z. M. (2010). A new hybrid PSOGSA algorithm
        for function optimization. Proceedings of ICCIA 2010, 374-377.
        https://doi.org/10.1109/ICCIA.2010.6141614

        Rashedi, E., Nezamabadi-pour, H., & Saryazdi, S. (2009). GSA:
        A Gravitational Search Algorithm. Information Sciences, 179(13), 2232-2248.
        https://doi.org/10.1016/j.ins.2009.03.004
        """
        for name, value in (
            ("c1", c1),
            ("c2", c2),
            ("g_zero", g_zero),
            ("alpha", alpha),
            ("r_norm", r_norm),
            ("r_power", r_power),
            ("k_agents_percent", k_agents_percent),
        ):
            if not isinstance(value, (int, float, np.number)):
                raise TypeError(f"{name} must be a number.")

            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")

        if g_zero <= 0:
            raise ValueError("g_zero must be greater than 0.")

        if alpha < 0:
            raise ValueError("alpha must be greater than or equal to 0.")

        if r_norm == 0:
            raise ValueError("r_norm cannot be zero.")

        if r_power <= 0:
            raise ValueError("r_power must be greater than 0.")

        if not 0 <= k_agents_percent <= 1:
            raise ValueError("k_agents_percent must be between 0 and 1.")

        self._c1 = float(c1)
        self._c2 = float(c2)
        self._g_zero = float(g_zero)
        self._alpha = float(alpha)
        self._r_norm = float(r_norm)
        self._r_power = float(r_power)
        self._k_agents_percent = float(k_agents_percent)

        self._gravitational_constant: float | None = None
        self._variable_names: tuple[str, ...] = ()

    @property
    def c1(self) -> float:
        return self._c1

    @property
    def c2(self) -> float:
        return self._c2

    @property
    def g_zero(self) -> float:
        return self._g_zero

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def r_norm(self) -> float:
        return self._r_norm

    @property
    def r_power(self) -> float:
        return self._r_power

    @property
    def k_agents_percent(self) -> float:
        return self._k_agents_percent

    @property
    def gravitational_constant(self) -> float | None:
        return self._gravitational_constant

    def create_particle(
        self,
        identifier: str,
        variables: dict[str, float] | None = None,
        fitness: np.float64 = FITNESS_UNDEFINED,
        velocity: dict[str, float] | None = None,
        mass: float = 0.0,
        acceleration: dict[str, float] | None = None,
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

        if acceleration is None:
            acceleration = {name: 0.0 for name in self._boundaries}

        self._population[identifier] = PSOGSAParticle(
            identifier=identifier,
            variables=variables,
            fitness=fitness,
            velocity=velocity,
            mass=mass,
            acceleration=acceleration,
        )

    def delete_particle(self, identifier: str | None) -> None:
        if identifier is None:
            raise ValueError("Particle identifier cannot be None.")

        del self._population[identifier]

    def pre_iteration(self, actual_iter: int) -> None:
        if actual_iter == 0:
            self._variable_names = tuple(self._boundaries)

            if not self._variable_names:
                raise ValueError("PSOGSA requires at least one optimization variable.")

            self._gravitational_constant = self._g_zero
            return

        self._gravitational_constant = self._g_zero * np.exp(
            -self._alpha * actual_iter / self._max_iterations
        )

    def create_random_cache(
        self,
        particle_ids: list[str],
        initialize: bool,
    ) -> None:
        for particle_id in particle_ids:
            cache: dict[str, Serializable] = {}

            if not initialize:
                for other_id in self._population:
                    if other_id != particle_id:
                        cache[f"force-{other_id}"] = float(self._rng.random())

                cache["inertia"] = float(self._rng.random())
                cache["gravitational"] = float(self._rng.random())
                cache["global-best"] = float(self._rng.random())

            self._population[particle_id].random_cache = cache

    def initialize_particle(self, identifier: str) -> PSOGSAParticle:
        particle = self._population[identifier]

        if not isinstance(particle, PSOGSAParticle):
            raise TypeError(
                f"Particle '{identifier}' must be an instance of PSOGSAParticle."
            )

        if np.isinf(particle.fitness):
            particle.update(
                variables=particle.variables,
                fitness_function=self._fitness_function,
            )

        return particle

    @staticmethod
    def consolidate_new_particles(particle: PSOGSAParticle) -> PSOGSAParticle:
        particle.consolidate(consolidate_new=True)
        return particle

    def _calculate_acceleration(
        self,
        particle: PSOGSAParticle,
    ) -> dict[str, float]:
        if self._gravitational_constant is None:
            raise RuntimeError("The gravitational constant has not been initialized.")

        acceleration = {name: 0.0 for name in self._variable_names}

        epsilon = np.finfo(float).eps

        for other_id, other_particle in self._population.items():
            if other_id == particle.identifier:
                continue

            distance = np.linalg.norm(
                np.asarray(
                    [
                        particle.variables[name] - other_particle.variables[name]
                        for name in self._variable_names
                    ],
                    dtype=float,
                ),
                ord=self._r_norm,
            )

            denominator = distance**self._r_power + epsilon
            random_factor = float(particle.random_cache[f"force-{other_id}"])

            force_factor = (
                self._gravitational_constant
                * particle.mass
                * other_particle.mass
                * random_factor
                / denominator
            )

            for name in self._variable_names:
                acceleration[name] += force_factor * (
                    other_particle.variables[name] - particle.variables[name]
                )

        if particle.mass > 0:
            for name in self._variable_names:
                acceleration[name] /= particle.mass

        return acceleration

    def update_particle(self, identifier: str) -> PSOGSAParticle:
        particle = self._population[identifier]

        if not isinstance(particle, PSOGSAParticle):
            raise TypeError(
                f"Particle '{identifier}' must be an instance of PSOGSAParticle."
            )

        if particle.velocity is None:
            raise RuntimeError(f"Particle '{identifier}' does not have a velocity.")

        if self._local_best is None:
            raise RuntimeError("Local best particle has not been initialized.")

        acceleration = self._calculate_acceleration(particle)

        inertia_random = float(particle.random_cache["inertia"])
        gravitational_random = float(particle.random_cache["gravitational"])
        global_best_random = float(particle.random_cache["global-best"])

        global_best_variables = self._local_best.variables

        new_velocity = {}
        new_variables = {}

        for name, (lower, upper) in self._boundaries.items():
            current_variable = particle.variables[name]
            current_velocity = particle.velocity[name]

            velocity = (
                inertia_random * current_velocity
                + self._c1 * gravitational_random * acceleration[name]
                + self._c2
                * global_best_random
                * (global_best_variables[name] - current_variable)
            )

            variable = float(np.clip(current_variable + velocity, lower, upper))

            new_velocity[name] = velocity
            new_variables[name] = variable

        particle.velocity = new_velocity
        particle.acceleration = acceleration

        particle.update(
            variables=new_variables,
            fitness_function=self._fitness_function,
        )

        return particle

    def _update_masses(self) -> None:
        particles = list(self._population.values())

        if not particles:
            return

        fitness = np.asarray(
            [particle.fitness for particle in particles],
            dtype=float,
        )

        finite = np.isfinite(fitness)

        if not np.any(finite):
            masses = np.full(
                len(particles),
                1.0 / len(particles),
            )
        else:
            finite_fitness = fitness[finite]
            best_fitness = np.min(finite_fitness)
            worst_fitness = np.max(finite_fitness)

            if np.isclose(best_fitness, worst_fitness):
                masses = np.full(
                    len(particles),
                    1.0 / len(particles),
                )
            else:
                adjusted_fitness = fitness.copy()

                scale = max(
                    abs(worst_fitness),
                    abs(best_fitness),
                    1.0,
                )

                adjusted_fitness[~finite] = worst_fitness + scale

                masses = (adjusted_fitness - worst_fitness) / (
                    best_fitness - worst_fitness
                )

                masses = np.maximum(masses, 0.0)

                total_mass = np.sum(masses)

                if not np.isfinite(total_mass) or total_mass <= 0:
                    masses = np.full(
                        len(particles),
                        1.0 / len(particles),
                    )
                else:
                    masses /= total_mass

        for particle, mass in zip(particles, masses):
            particle.mass = float(mass)

    def post_iteration(self, actual_iter: int) -> None:
        if actual_iter == 0:
            self._update_masses()
            self.update_solution_state()
            return

        for particle in self._population.values():
            if not isinstance(particle, PSOGSAParticle):
                raise TypeError(
                    f"Particle '{particle.identifier}' must be an instance of "
                    "PSOGSAParticle."
                )

            if particle.candidate_fitness is None:
                raise RuntimeError(
                    f"Particle '{particle.identifier}' has no candidate fitness."
                )

            particle.consolidate(consolidate_new=True)

        self._update_masses()
        self.update_solution_state()
