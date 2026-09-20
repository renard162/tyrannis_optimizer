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
        k_agents_percent: float = 1.0,
    ) -> None:
        """
        Particle Swarm Optimization and Gravitational Search Algorithm (PSOGSA).

        PSOGSA is a population-based hybrid optimization algorithm that combines
        the exploitation capability of Particle Swarm Optimization with the
        exploration capability of the Gravitational Search Algorithm. The
        gravitational component promotes interaction between candidate solutions,
        while the PSO component guides the search toward the best solution found
        so far, providing a balance between exploration and exploitation.

        Parameters
        ----------
        c1 : float, default=0.5
            Coefficient controlling the influence of the gravitational acceleration
            on the particle velocity. Larger values increase the contribution of
            the gravitational search to the movement of particles. It must be a
            finite number greater than 0.

        c2 : float, default=1.5
            Coefficient controlling the influence of the best solution found so
            far on the particle velocity. Larger values increase the attraction
            toward the best-known solution. It must be a finite number greater
            than 0.

        g_zero : float, default=1.0
            Initial value of the gravitational constant. Larger values increase
            the initial influence of gravitational interactions and promote
            broader exploration.

        alpha : float, default=20.0
            Rate controlling the decrease of the gravitational constant during
            the optimization. Larger values cause the gravitational influence
            to decrease more rapidly.

        r_norm : float, default=2.0
            Order of the norm used to calculate the distance between particles.
            The default value of 2 corresponds to the Euclidean norm.

        r_power : float, default=1.0
            Power applied to the distance in the gravitational interaction.
            Larger values cause the influence of distant particles to decrease
            more rapidly.

        k_agents_percent : float, default=1.0
            Minimum fraction of the population retained as gravitational attractors
            as the search progresses. The value must be between 0 and 1. The
            default value of 1 keeps the entire population participating in the
            gravitational interaction, reproducing the original PSOGSA formulation.
            Lower values progressively restrict the interaction to the best agents,
            extending the original formulation with the Kbest strategy of GSA.

        Notes
        -----
        PSOGSA combines the main search mechanisms of Particle Swarm Optimization
        and the Gravitational Search Algorithm. Each candidate solution is
        influenced by the gravitational attraction of other solutions and by the
        best solution found by the population.

        In the gravitational component, candidate solutions are assigned masses
        according to their fitness. Better solutions receive greater mass and
        therefore exert a stronger influence on the search. The gravitational
        constant controls the overall strength of these interactions and
        decreases during the optimization, reducing the gravitational influence
        as the search progresses.

        The velocity of each particle is updated from three components: its
        previous velocity, the gravitational acceleration produced by the
        population, and the direction toward the best solution found so far.
        The coefficients ``c1`` and ``c2`` control the relative influence of the
        gravitational and global-best components, respectively.

        The parameters ``g_zero`` and ``alpha`` control the temporal behavior of
        the gravitational search. ``g_zero`` determines its initial strength,
        while ``alpha`` determines how rapidly that strength decreases. Larger
        values of ``g_zero`` increase the initial gravitational influence, whereas
        larger values of ``alpha`` shift the search more rapidly toward the
        global-best component.

        The parameters ``r_norm`` and ``r_power`` control how the distance between
        solutions affects gravitational interactions. ``r_norm`` defines the
        distance metric, with ``r_norm=2`` corresponding to the Euclidean distance.
        ``r_power`` determines how strongly the interaction decreases with distance.
        The conventional configuration uses ``r_norm=2`` and ``r_power=1``.

        In the original PSOGSA formulation, all particles contribute to the
        gravitational interaction throughout the optimization. This behavior is
        preserved by the default ``k_agents_percent=1``. Lower values enable a
        GSA-inspired Kbest strategy in which the set of gravitational attractors
        progressively decreases toward the specified minimum population fraction,
        concentrating the gravitational interaction on the best solutions.

        PSOGSA is designed for continuous optimization and can be applied to
        nonlinear, non-convex, multimodal, and derivative-free objective
        functions. Larger populations generally provide broader exploration,
        while additional iterations allow the search to progressively refine
        promising regions of the search space. The population size should
        therefore be increased for higher-dimensional or more complex problems.

        References
        ----------
        Mirjalili, S., & Hashim, S. Z. M. (2010). A new hybrid PSOGSA algorithm
        for function optimization. Proceedings of ICCIA 2010, 374-377.
        https://doi.org/10.1109/ICCIA.2010.6141614

        Mirjalili, S., Hashim, S. Z. M., & Sardroudi, H. M. (2012). Training
        feedforward neural networks using hybrid particle swarm optimization and
        gravitational search algorithm. Applied Mathematics and Computation,
        218(22), 11125-11137.
        https://doi.org/10.1016/j.amc.2012.04.069

        Jayaprakasam, S., Abdul Rahim, S. K., & Leow, C. Y. (2015). PSOGSA-Explore:
        A new hybrid metaheuristic approach for beampattern optimization in
        collaborative beamforming. Applied Soft Computing, 30, 229-237.
        https://doi.org/10.1016/j.asoc.2015.01.024

        Radosavljević, J. (2016). A solution to the combined economic and emission
        dispatch using hybrid PSOGSA algorithm. Applied Artificial Intelligence,
        30(5), 445-474.
        https://doi.org/10.1080/08839514.2016.1185860

        Lacerda Junior, W. R., Martins, S. A. M., & Nepomuceno, E. G. (2019).
        Identificação de Sistemas Não Lineares Utilizando o Algoritmo Híbrido e
        Binário de Otimização por Enxame de Partículas e Busca Gravitacional.
        Anais do 14º Simpósio Brasileiro de Automação Inteligente.
        https://doi.org/10.17648/sbai-2019-111317
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

        if c1 <= 0:
            raise ValueError("c1 must be greater than 0.")

        if c2 <= 0:
            raise ValueError("c2 must be greater than 0.")

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

        self._c1 = float(c1)
        self._c2 = float(c2)
        self._g_zero = float(g_zero)
        self._alpha = float(alpha)
        self._r_norm = float(r_norm)
        self._r_power = float(r_power)
        self._k_agents_percent = float(k_agents_percent)

        self._gravitational_constant: float | None = None
        self._variable_names: tuple[str, ...] = ()
        self._k_best: tuple[str, ...] = ()

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

        algorithm_iter = actual_iter - 1
        self._gravitational_constant = self._g_zero * np.exp(
            -self._alpha * algorithm_iter / self._max_iterations
        )

    def create_random_cache(self, particle_ids: list[str], initialize: bool) -> None:
        for particle_id in particle_ids:
            cache: dict[str, Serializable] = {}

            if not initialize:
                for other_id in self._k_best:
                    if other_id != particle_id and other_id in self._population:
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
            masses = np.full(len(particles), 1.0 / len(particles))
        else:
            finite_fitness = fitness[finite]
            best_fitness = np.min(finite_fitness)
            worst_fitness = np.max(finite_fitness)

            if np.isclose(best_fitness, worst_fitness):
                masses = np.full(len(particles), 1.0 / len(particles))
            else:
                adjusted_fitness = fitness.copy()

                scale = max(abs(worst_fitness), abs(best_fitness), 1.0)

                adjusted_fitness[~finite] = worst_fitness + scale

                masses = (adjusted_fitness - worst_fitness) / (
                    best_fitness - worst_fitness
                )

                masses = np.maximum(masses, 0.0)

                total_mass = np.sum(masses)

                if not np.isfinite(total_mass) or total_mass <= 0:
                    masses = np.full(len(particles), 1.0 / len(particles))
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
        self._update_k_best(actual_iter)
        self.update_solution_state()
