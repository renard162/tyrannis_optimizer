import numpy as np

from ..core.algorithm import (
    FITNESS_UNDEFINED,
    AlgorithmBase,
    ParticleBase,
    Serializable,
)


class GSAParticle(ParticleBase):
    """Particle implementation for the Gravitational Search Algorithm."""

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


class GSA(AlgorithmBase[GSAParticle]):
    """Gravitational Search Algorithm."""

    def __init__(
        self,
        g_zero: float = 100.0,
        alpha: float = 20.0,
        r_norm: float = 2.0,
        r_power: float = 1.0,
        k_agents_percent: float = 0.0,
    ) -> None:
        """
        Gravitational Search Algorithm (GSA).

        GSA is a population-based, derivative-free optimization algorithm inspired by
        gravitational interactions between masses. Candidate solutions are modeled as
        agents whose masses are determined by their objective values, causing better
        solutions to exert stronger attraction on the population. The resulting
        accelerations guide the agents through the search space while the gravitational
        interaction progressively decreases over the optimization process.

        Parameters
        ----------
        g_zero : float, default=100.0
            Initial gravitational constant controlling the overall strength of the
            attraction between agents. Larger values produce stronger accelerations and
            broader movement, while smaller values result in more conservative search
            trajectories.

        alpha : float, default=20.0
            Decay coefficient of the gravitational constant. Larger values reduce the
            gravitational interaction more rapidly, favoring earlier exploitation,
            while smaller values preserve stronger interactions for more iterations.

        r_norm : float, default=2.0
            Norm used to measure the distance between agents. It must be greater than
            or equal to 1. The default value uses Euclidean distance, while other valid
            values change how separation between solutions is measured across the
            search dimensions.

        r_power : float, default=1.0
            Exponent applied to the distance in the gravitational interaction. Larger
            values reduce the influence of distant agents more strongly, concentrating
            the search around nearby solutions.

        k_agents_percent : float, default=0.0
            Minimum fraction of the population retained as gravitational attractors as
            the search progresses. A value of ``0.0`` allows the set of attractors to
            decrease to a single best agent, while larger values preserve influence
            from a greater fraction of the population.

        Notes
        -----
        GSA represents each candidate solution as a mass moving through the search
        space under the gravitational influence of other solutions. Better objective
        values correspond to larger normalized masses, causing better solutions to
        exert greater influence on the movement of the population.

        For a minimization problem, the mass assigned to an agent is derived from its
        fitness relative to the best and worst solutions in the current population:

            m_i(t) = (fitness_i(t) - worst(t)) / (best(t) - worst(t))

            M_i(t) = m_i(t) / sum_j(m_j(t))

        The gravitational interaction is controlled by a time-dependent constant
        ``G`` that decreases exponentially throughout the optimization:

            G(t) = G0 * exp(-alpha * t / T)

        where ``G0`` is controlled by ``g_zero``, ``alpha`` determines the decay rate,
        ``t`` is the current iteration, and ``T`` is the total number of iterations.
        This decay allows stronger movement during the early search and progressively
        reduces the magnitude of the gravitational interactions as the optimization
        advances.

        Each agent is attracted toward a subset of the best agents in the population.
        The contribution of an attracting agent depends on its mass, its distance from
        the current agent, and a random factor. The distance contribution is controlled
        by ``r_norm`` and ``r_power``, allowing the geometry and distance sensitivity
        of the gravitational interaction to be adjusted.

        The resulting gravitational forces determine the acceleration of each agent.
        Its velocity combines this acceleration with a randomized contribution from
        its previous velocity, and the new position is obtained by adding the resulting
        velocity to the current position. Consequently, the population moves toward
        regions associated with heavier, better-performing agents while retaining
        stochastic variation in its trajectories.

        The set of agents capable of exerting gravitational attraction is progressively
        reduced during the search. Initially, the entire population can contribute to
        the gravitational interaction. As the optimization advances, this set decreases
        toward the fraction specified by ``k_agents_percent``. The default value of
        ``0.0`` allows the search to converge toward influence from only the best agent,
        while larger values retain a broader set of attractors and therefore preserve
        more population-wide interaction near the end of the optimization.

        The parameters ``g_zero`` and ``alpha`` primarily control the magnitude and
        persistence of movement. Increasing ``g_zero`` strengthens the initial search,
        whereas increasing ``alpha`` causes that influence to disappear more rapidly.
        ``r_norm`` and ``r_power`` determine how the spatial separation between agents
        affects their interaction, and ``k_agents_percent`` controls how concentrated
        the gravitational influence becomes toward the end of the search.

        References
        ----------
        Rashedi, E., Nezamabadi-pour, H., & Saryazdi, S. (2009). GSA: A Gravitational
        Search Algorithm. Information Sciences, 179(13), 2232-2248.
        https://doi.org/10.1016/j.ins.2009.03.004

        Rashedi, E., Rashedi, E., & Nezamabadi-pour, H. (2018). A comprehensive survey
        on gravitational search algorithm. Swarm and Evolutionary Computation, 41,
        141-158.
        https://doi.org/10.1016/j.swevo.2018.02.018

        Saha, S. K., Kar, R., Mandal, D., & Ghoshal, S. P. (2014). Gravitation search
        algorithm: Application to the optimal IIR filter design. Journal of King Saud
        University - Engineering Sciences, 26(1), 69-81.
        https://doi.org/10.1016/j.jksues.2012.12.003

        Han, X., Chang, X., Quan, L., Xiong, X., Li, J., Zhang, Z., & Liu, Y. (2014).
        Feature subset selection by gravitational search algorithm optimization.
        Information Sciences, 281, 128-146.
        https://doi.org/10.1016/j.ins.2014.05.030
        """
        for name, value in (
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

        if r_norm < 1:
            raise ValueError("r_norm must be greater than or equal to 1.")

        if r_power <= 0:
            raise ValueError("r_power must be greater than 0.")

        if not 0 <= k_agents_percent <= 1:
            raise ValueError("k_agents_percent must be between 0 and 1.")

        self._g_zero = float(g_zero)
        self._alpha = float(alpha)
        self._r_norm = float(r_norm)
        self._r_power = float(r_power)
        self._k_agents_percent = float(k_agents_percent)

        self._gravitational_constant: float | None = None
        self._variable_names: tuple[str, ...] = ()
        self._k_best: tuple[str, ...] = ()

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
            velocity = {name: 0.0 for name in self._boundaries}

        if acceleration is None:
            acceleration = {name: 0.0 for name in self._boundaries}

        self._population[identifier] = GSAParticle(
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
                raise ValueError("GSA requires at least one optimization variable.")

            self._gravitational_constant = self._g_zero
            return

        algorithm_iter = actual_iter - 1
        self._gravitational_constant = self._g_zero * np.exp(
            -self._alpha * algorithm_iter / self._max_iterations
        )

    def create_random_cache(
        self,
        particle_ids: list[str],
        initialize: bool,
    ) -> None:
        for particle_id in particle_ids:
            cache: dict[str, Serializable] = {}

            if not initialize:
                for other_id in self._k_best:
                    if other_id != particle_id and other_id in self._population:
                        cache[f"force-{other_id}"] = float(self._rng.random())

                cache["velocity"] = float(self._rng.random())

            self._population[particle_id].random_cache = cache

    def initialize_particle(self, identifier: str) -> GSAParticle:
        particle = self._population[identifier]

        if not isinstance(particle, GSAParticle):
            raise TypeError(
                f"Particle '{identifier}' must be an instance of GSAParticle."
            )

        if particle.fitness == FITNESS_UNDEFINED:
            particle.update(
                variables=particle.variables,
                fitness_function=self._fitness_function,
            )

        return particle

    @staticmethod
    def consolidate_new_particles(particle: GSAParticle) -> GSAParticle:
        particle.consolidate(consolidate_new=True)
        return particle

    def _calculate_acceleration(
        self,
        particle: GSAParticle,
    ) -> dict[str, float]:
        if self._gravitational_constant is None:
            raise RuntimeError("The gravitational constant has not been initialized.")

        acceleration = {name: 0.0 for name in self._variable_names}
        epsilon = np.finfo(float).eps

        for other_id in self._k_best:
            if other_id == particle.identifier or other_id not in self._population:
                continue

            other_particle = self._population[other_id]
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
            acceleration_factor = (
                self._gravitational_constant
                * other_particle.mass
                * random_factor
                / denominator
            )

            for name in self._variable_names:
                acceleration[name] += float(
                    acceleration_factor
                    * (other_particle.variables[name] - particle.variables[name])
                )

        return acceleration

    def update_particle(self, identifier: str) -> GSAParticle:
        particle = self._population[identifier]

        if not isinstance(particle, GSAParticle):
            raise TypeError(
                f"Particle '{identifier}' must be an instance of GSAParticle."
            )

        if particle.velocity is None:
            raise RuntimeError(f"Particle '{identifier}' does not have a velocity.")

        acceleration = self._calculate_acceleration(particle)
        velocity_random = float(particle.random_cache["velocity"])

        new_velocity = {}
        new_variables = {}

        for name, (lower, upper) in self._boundaries.items():
            velocity = velocity_random * particle.velocity[name] + acceleration[name]
            variable = float(np.clip(particle.variables[name] + velocity, lower, upper))

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
        negative_infinite = np.isneginf(fitness)
        finite = np.isfinite(fitness)

        if np.any(negative_infinite):
            masses = negative_infinite.astype(float)
            masses /= np.sum(masses)
        elif not np.any(finite):
            masses = np.full(len(particles), 1.0 / len(particles))
        else:
            finite_fitness = fitness[finite]
            best_fitness = np.min(finite_fitness)
            worst_fitness = np.max(finite_fitness)
            masses = np.zeros(len(particles), dtype=float)

            if np.isclose(best_fitness, worst_fitness):
                masses[finite] = 1.0 / np.count_nonzero(finite)
            else:
                masses[finite] = (finite_fitness - worst_fitness) / (
                    best_fitness - worst_fitness
                )
                masses = np.maximum(masses, 0.0)
                total_mass = np.sum(masses)

                if not np.isfinite(total_mass) or total_mass <= 0:
                    masses[finite] = 1.0 / np.count_nonzero(finite)
                else:
                    masses /= total_mass

        for particle, mass in zip(particles, masses):
            particle.mass = float(mass)

    def _update_k_best(self, next_iter: int) -> None:
        if not self._population:
            self._k_best = ()
            return

        population_size = len(self._population)
        minimum_agents = max(
            1,
            int(np.ceil(population_size * self._k_agents_percent)),
        )

        last_algorithm_iter = max(self._max_iterations - 1, 1)
        progress = min(next_iter, self._max_iterations - 1) / last_algorithm_iter

        n_agents = int(
            np.ceil(population_size - (population_size - minimum_agents) * progress)
        )
        n_agents = max(minimum_agents, min(population_size, n_agents))

        ranked_particles = sorted(
            self._population.values(),
            key=lambda particle: particle.fitness,
        )
        self._k_best = tuple(
            particle.identifier for particle in ranked_particles[:n_agents]
        )

    def post_iteration(self, actual_iter: int) -> None:
        if actual_iter > 0:
            for particle in self._population.values():
                if not isinstance(particle, GSAParticle):
                    raise TypeError(
                        f"Particle '{particle.identifier}' must be an instance of "
                        "GSAParticle."
                    )

                if particle.candidate_fitness is None:
                    raise RuntimeError(
                        f"Particle '{particle.identifier}' has no candidate fitness."
                    )

                particle.consolidate(consolidate_new=True)

        self._update_masses()
        self._update_k_best(actual_iter)
        self.update_solution_state()
