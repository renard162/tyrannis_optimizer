import numpy as np

from ..core.algorithm import FITNESS_UNDEFINED, AlgorithmBase, ParticleBase


class CMAESCandidateSolution(ParticleBase):
    """Candidate solution for Covariance Matrix Adaptation Evolution Strategy algorithm."""

    def __init__(
        self,
        identifier: str,
        variables: dict[str, float],
        fitness: np.float64 = FITNESS_UNDEFINED,
    ) -> None:
        super().__init__(identifier=identifier, variables=variables, fitness=fitness)
        self._raw_candidate_variables: dict[str, float] | None = None

    @property
    def raw_candidate_variables(self) -> dict[str, float] | None:
        return self._raw_candidate_variables

    @raw_candidate_variables.setter
    def raw_candidate_variables(self, variables: dict[str, float] | None) -> None:
        self._raw_candidate_variables = variables


class CMAES(AlgorithmBase[CMAESCandidateSolution]):
    """Covariance Matrix Adaptation Evolution Strategy algorithm."""

    _MAX_RESAMPLING_ATTEMPTS = 100_000

    def __init__(
        self,
        sigma: float = 0.3,
        boundary_handling: str = "adaptive-penalty",
    ) -> None:
        """
        Covariance Matrix Adaptation Evolution Strategy (CMA-ES).

        CMA-ES is a population-based, derivative-free optimization algorithm for
        continuous problems that adapts a multivariate normal search distribution to
        the objective landscape. Candidate solutions are sampled from the distribution
        and evaluated, and the best candidates are used to update its mean, covariance
        matrix, and step size, allowing the search to progressively adapt its location,
        scale, and orientation to promising regions of the search space.

        Parameters
        ----------
        sigma : float, default=0.3
            Initial global step size expressed as a fraction of the average variable
            range. Larger values produce broader initial exploration, while smaller
            values concentrate the initial search around the initial mean.

        boundary_handling : {"adaptive-penalty", "reflection", "resampling", "clip"}, default="adaptive-penalty"
            Defines how samples outside the optimization boundaries are handled
            before evaluation.

            - ``"adaptive-penalty"``: evaluates the closest feasible point and
              ranks the original Gaussian sample using an adaptive quadratic
              boundary penalty.
            - ``"reflection"``: reflects the sample into the feasible domain while
              preserving the original Gaussian sample for CMA-ES adaptation.
            - ``"resampling"``: rejects infeasible samples and draws again until
              a feasible sample is obtained.
            - ``"clip"``: clips the sample to the nearest feasible point and uses
              the clipped point both for evaluation and CMA-ES adaptation.

        Notes
        -----
        CMA-ES maintains a multivariate normal search distribution defined by a
        mean vector ``m``, a covariance matrix ``C``, and a global step size
        ``sigma``. Together, these parameters determine where candidate
        solutions are generated, the directions in which the search is
        expanded or contracted, and the overall scale of the search:

            x ~ N(m, sigma² C)

        At each iteration, a population of candidate solutions is sampled from
        the current distribution. The candidates are evaluated and ranked
        according to their objective values, and the best ``mu`` candidates are
        selected to update the distribution. CMA-ES requires at least two
        particles, including after population changes during optimization.

        The new mean is calculated as a weighted combination of the selected
        candidates, with better-ranked candidates receiving greater weights.
        This moves the search distribution toward regions associated with
        better objective values.

        The covariance matrix is adapted from the selected candidates and the
        accumulated search path. The rank-mu update captures the distribution
        of the selected candidates in the current iteration, while the rank-one
        update uses the covariance evolution path to accumulate information
        about consistent search directions across iterations. This allows
        CMA-ES to learn correlations between variables and orient its search
        distribution according to the geometry of the objective function.

        The global step size is adapted independently through cumulative
        step-size adaptation. The evolution path used for this adaptation
        measures the consistency of successive changes to the mean, allowing
        the algorithm to increase the search scale when progress occurs in a
        consistent direction and decrease it when the search becomes less
        directional.

        The algorithm therefore adapts three complementary aspects of the
        search: the mean determines where to search, the covariance matrix
        determines the shape and orientation of the search distribution, and
        the step size determines its overall scale.

        Boundary handling never evaluates the objective function outside the
        configured boundaries. With ``"adaptive-penalty"``, the objective is
        evaluated at the closest feasible point, while the original Gaussian
        sample is preserved for distribution adaptation. The artificial ranking
        value is the evaluated fitness plus a coordinate-wise adaptive quadratic
        penalty. Non-finite fitness values never enter penalty arithmetic. In the
        special case of ``-np.inf``, the best possible fitness is preserved and
        the feasible evaluated point, rather than the infeasible Gaussian sample,
        is used in the CMA-ES distribution update.

        With ``"reflection"``, samples are reflected into the feasible domain for
        evaluation while their original Gaussian coordinates are retained for
        distribution adaptation. With ``"resampling"``, infeasible samples are
        rejected before evaluation, so the evaluated point and the point used by
        CMA-ES adaptation are identical.

        ``"clip"`` preserves the legacy CMA-ES boundary behavior for compatibility.
        Samples outside the boundaries are clipped to the nearest feasible point,
        which is then used both for evaluation and distribution adaptation. This
        approach can distort the learned search distribution because different
        Gaussian samples may collapse to the same boundary point, so the adaptation
        vectors no longer necessarily represent the originally sampled Gaussian
        steps. For this reason, ``"clip"`` is retained for compatibility but is not
        the recommended boundary handling strategy.

        CMA-ES is designed for continuous optimization problems and is
        particularly suitable for non-linear, non-convex, multi-modal, noisy,
        or non-differentiable objective functions. It is also well suited to
        problems where the optimization variables have different scales or
        strong correlations.

        The initial value of ``sigma`` controls the scale of the initial
        search. Larger values favor broader exploration, while smaller values
        produce a more concentrated initial search.

        References
        ----------
        Akimoto, Y., & Hansen, N. (2020). Diagonal Acceleration for Covariance Matrix
        Adaptation Evolution Strategies. Evolutionary Computation, 28(3), 405-435.
        https://doi.org/10.1162/evco_a_00260

        Hansen, N., & Ostermeier, A. (2001). Completely derandomized self-adaptation
        in evolution strategies. Evolutionary Computation, 9(2), 159-195.
        https://doi.org/10.1162/106365601750190398

        Hansen, N. (2006). The CMA Evolution Strategy: A Comparing Review. In
        Towards a New Evolutionary Computation, 75-102.
        https://doi.org/10.1007/3-540-32494-1_4

        Hansen, N., Niederberger, A. S. P., Guzzella, L., & Koumoutsakos, P. (2009).
        A Method for Handling Uncertainty in Evolutionary Optimization With an
        Application to Feedback Control of Combustion. IEEE Transactions on
        Evolutionary Computation, 13(1), 180-197.
        https://doi.org/10.1109/TEVC.2008.924423

        Biedrzycki, R. (2020). Handling bound constraints in CMA-ES: An experimental
        study. Swarm and Evolutionary Computation, 52, 100627.
        https://doi.org/10.1016/j.swevo.2019.100627

        Chocat, R., Brevault, L., Balesdent, M., & Defoort, S. (2015). Modified
        Covariance Matrix Adaptation - Evolution Strategy algorithm for constrained
        optimization under uncertainty, application to rocket design. International
        Journal for Simulation and Multidisciplinary Design Optimization, 6, A1.
        https://doi.org/10.1051/smdo/2015001

        Dang, V.-H., Vien, N. A., & Chung, T. (2019). A covariance matrix adaptation
        evolution strategy in reproducing kernel Hilbert space. Genetic Programming
        and Evolvable Machines, 20, 479-501.
        https://doi.org/10.1007/s10710-019-09357-1
        """
        if not isinstance(sigma, (int, float, np.number)):
            raise TypeError("sigma must be a number.")

        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be a finite number greater than 0.")

        if not isinstance(boundary_handling, str):
            raise TypeError("boundary_handling must be a string.")

        allowed_boundary_handling = (
            "adaptive-penalty",
            "reflection",
            "resampling",
            "clip",
        )

        if boundary_handling not in allowed_boundary_handling:
            raise ValueError(
                "boundary_handling must be 'adaptive-penalty', 'reflection', "
                "'resampling', or 'clip'."
            )

        self._initial_sigma = float(sigma)
        self._boundary_handling = boundary_handling

        self._mean: np.ndarray | None = None
        self._covariance: np.ndarray | None = None
        self._step_size: float | None = None
        self._p_sigma: np.ndarray | None = None
        self._p_c: np.ndarray | None = None
        self._gamma_sigma: float = 0.0
        self._gamma_c: float = 0.0
        self._weights: np.ndarray | None = None
        self._mu: int | None = None
        self._mu_eff: float | None = None
        self._cc: float | None = None
        self._cs: float | None = None
        self._c1: float | None = None
        self._cmu: float | None = None
        self._damps: float | None = None
        self._chi_n: float | None = None
        self._dimension: int | None = None
        self._variable_names: tuple[str, ...] = ()
        self._lower_bounds: np.ndarray | None = None
        self._upper_bounds: np.ndarray | None = None
        self._injected_particle_ids: set[str] = set()
        self._penalty_gamma: np.ndarray | None = None
        self._penalty_history: list[float] = []
        self._penalty_weights_initialized = False

    @property
    def sigma(self) -> float:
        if self._step_size is None:
            return self._initial_sigma
        return self._step_size

    @property
    def boundary_handling(self) -> str:
        return self._boundary_handling

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

        self._population[identifier] = CMAESCandidateSolution(
            identifier=identifier, variables=variables, fitness=fitness
        )

    def delete_particle(self, identifier: str | None) -> None:
        if identifier is None:
            raise ValueError("Particle identifier cannot be None.")

        del self._population[identifier]

    def _initialize_parameters(self) -> None:
        self._variable_names = tuple(self._boundaries)
        self._dimension = len(self._variable_names)

        if self._dimension == 0:
            raise ValueError("CMA-ES requires at least one optimization variable.")

        n = self._dimension

        self._chi_n = float(np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n**2)))
        self._covariance = np.eye(n)
        self._p_sigma = np.zeros(n)
        self._p_c = np.zeros(n)
        self._gamma_sigma = 0.0
        self._gamma_c = 0.0
        self._lower_bounds = np.asarray(
            [lower for lower, _ in self._boundaries.values()], dtype=float
        )
        self._upper_bounds = np.asarray(
            [upper for _, upper in self._boundaries.values()], dtype=float
        )
        self._penalty_gamma = np.zeros(n)
        self._penalty_history = []
        self._penalty_weights_initialized = False

        ranges = self._upper_bounds - self._lower_bounds
        self._step_size = self._initial_sigma * float(np.mean(ranges))

    def _update_strategy_parameters(self) -> None:
        if self._dimension is None:
            raise RuntimeError("CMA-ES dimension has not been initialized.")

        population_size = len(self._population)

        if population_size < 2:
            raise ValueError("CMA-ES requires at least 2 particles.")

        mu = population_size // 2
        indices = np.arange(1, mu + 1, dtype=float)
        weights = np.log((population_size + 1.0) / 2.0) - np.log(indices)
        weights = weights / np.sum(weights)
        mu_eff = float(1.0 / np.sum(weights**2))

        n = self._dimension
        matrix_degrees = n * (n + 1.0) / 2.0

        cs = (mu_eff + 2.0) / (n + mu_eff + 5.0)
        damps = 1.0 + cs + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0)

        c1 = 1.0 / (2.0 * (matrix_degrees / n + 1.0) * (n + 1.0) ** 0.75 + mu_eff / 2.0)

        mu_prime = (
            mu_eff
            + 1.0 / mu_eff
            - 2.0
            + 0.5 * population_size / (population_size + 5.0)
        )

        cmu = min(mu_prime * c1, 1.0 - c1)
        cc = np.sqrt(mu_eff * c1) / 2.0

        self._mu = mu
        self._weights = weights
        self._mu_eff = mu_eff
        self._cc = float(cc)
        self._cs = float(cs)
        self._c1 = float(c1)
        self._cmu = float(cmu)
        self._damps = float(damps)

    def _variables_to_array(self, variables: dict[str, float]) -> np.ndarray:
        return np.asarray(
            [variables[name] for name in self._variable_names], dtype=float
        )

    def _array_to_variables(self, values: np.ndarray) -> dict[str, float]:
        return {name: float(value) for name, value in zip(self._variable_names, values)}

    def _project_to_boundaries(self, values: np.ndarray) -> np.ndarray:
        if self._lower_bounds is None or self._upper_bounds is None:
            raise RuntimeError("CMA-ES boundaries have not been initialized.")

        return np.clip(values, self._lower_bounds, self._upper_bounds)

    def _is_within_boundaries(self, values: np.ndarray) -> bool:
        if self._lower_bounds is None or self._upper_bounds is None:
            raise RuntimeError("CMA-ES boundaries have not been initialized.")

        return bool(
            np.all(np.isfinite(values))
            and np.all(values >= self._lower_bounds)
            and np.all(values <= self._upper_bounds)
        )

    def _reflect_to_boundaries(self, values: np.ndarray) -> np.ndarray:
        if self._lower_bounds is None or self._upper_bounds is None:
            raise RuntimeError("CMA-ES boundaries have not been initialized.")

        widths = self._upper_bounds - self._lower_bounds
        period = 2.0 * widths
        shifted = np.mod(values - self._lower_bounds, period)
        reflected = self._lower_bounds + np.where(
            shifted <= widths, shifted, period - shifted
        )

        return np.clip(reflected, self._lower_bounds, self._upper_bounds)

    @staticmethod
    def _sampling_transform(covariance: np.ndarray) -> np.ndarray:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, np.finfo(float).eps)

        return eigenvectors @ np.diag(np.sqrt(eigenvalues))

    def _raw_candidate_from_z(self, z: np.ndarray, transform: np.ndarray) -> np.ndarray:
        if self._mean is None or self._step_size is None:
            raise RuntimeError("CMA-ES state has not been initialized.")

        return self._mean + self._step_size * (transform @ z)

    def _resample_candidate(
        self, particle: CMAESCandidateSolution, transform: np.ndarray
    ) -> np.ndarray:
        local_rng: object = particle.random_cache["cmaes-rng"]

        if not isinstance(local_rng, np.random.Generator):
            raise TypeError("CMA-ES resampling cache must contain a local Generator.")

        dimension = self._dimension

        if dimension is None:
            raise RuntimeError("CMA-ES dimension has not been initialized.")

        for _ in range(self._MAX_RESAMPLING_ATTEMPTS):
            z = local_rng.standard_normal(dimension)
            candidate = self._raw_candidate_from_z(z=z, transform=transform)

            if self._is_within_boundaries(candidate):
                return candidate

        raise RuntimeError(
            "CMA-ES resampling could not generate a feasible candidate after "
            f"{self._MAX_RESAMPLING_ATTEMPTS} attempts."
        )

    def _update_boundary_penalty(
        self,
        actual_iter: int,
        particles: list[CMAESCandidateSolution],
        mean: np.ndarray,
        covariance: np.ndarray,
        step_size: float,
        mu_eff: float,
        dimension: int,
    ) -> None:
        if self._penalty_gamma is None:
            raise RuntimeError("CMA-ES boundary penalty has not been initialized.")

        finite_fitness = np.asarray(
            [
                float(particle.candidate_fitness)
                for particle in particles
                if particle.candidate_fitness is not None
                and np.isfinite(particle.candidate_fitness)
            ],
            dtype=float,
        )

        with np.errstate(over="ignore", invalid="ignore"):
            variance_scale = float(
                np.float64(step_size)
                * np.float64(step_size)
                * np.trace(covariance)
                / dimension
            )

        if (
            len(finite_fitness) >= 2
            and np.isfinite(variance_scale)
            and variance_scale > 0
        ):
            lower_quartile = float(np.quantile(finite_fitness, 0.25))
            upper_quartile = float(np.quantile(finite_fitness, 0.75))
            delta_f = (upper_quartile - lower_quartile) / variance_scale

            if np.isfinite(delta_f) and delta_f > 0:
                self._penalty_history.insert(0, delta_f)

        history_limit = 20 + int(np.floor(3.0 * dimension / len(particles)))

        if len(self._penalty_history) > history_limit:
            del self._penalty_history[history_limit:]

        if not self._penalty_history:
            return

        penalty_history = np.asarray(self._penalty_history, dtype=float)
        delta_fit = float(np.median(penalty_history))

        if not np.isfinite(delta_fit) or delta_fit <= 0:
            return

        projected_mean = self._project_to_boundaries(mean)
        mean_violation = mean - projected_mean
        mean_is_infeasible = bool(np.any(mean_violation != 0.0))

        if mean_is_infeasible and (
            not self._penalty_weights_initialized or actual_iter == 2
        ):
            self._penalty_gamma.fill(2.0 * delta_fit)
            self._penalty_weights_initialized = True

        if not self._penalty_weights_initialized:
            return

        diagonal = np.maximum(np.diag(covariance), np.finfo(float).eps)
        coordinate_std = step_size * np.sqrt(diagonal)
        normalized_distance = np.abs(mean_violation) / coordinate_std
        threshold = 3.0 * max(1.0, np.sqrt(dimension) / mu_eff)
        damping = min(1.0, mu_eff / (10.0 * dimension))
        excess = np.maximum(0.0, normalized_distance - threshold)
        increase = np.exp((damping / 2.0) * np.tanh(excess / 3.0))

        with np.errstate(over="ignore"):
            self._penalty_gamma *= increase

        maximum_float = np.finfo(float).max
        self._penalty_gamma = np.minimum(self._penalty_gamma, maximum_float)

        decrease_mask = self._penalty_gamma > 5.0 * delta_fit
        self._penalty_gamma[decrease_mask] *= np.exp(-damping / 3.0)

    def _boundary_penalty_score(self, particle: CMAESCandidateSolution) -> float:
        if particle.candidate_fitness is None:
            raise RuntimeError(
                f"Particle '{particle.identifier}' has no candidate fitness."
            )

        fitness = float(particle.candidate_fitness)

        if not np.isfinite(fitness):
            return fitness

        if particle.candidate_variables is None:
            raise RuntimeError(
                f"Particle '{particle.identifier}' has no candidate variables."
            )

        raw_variables = particle.raw_candidate_variables

        if raw_variables is None:
            return fitness

        if self._penalty_gamma is None or self._dimension is None:
            raise RuntimeError("CMA-ES boundary penalty has not been initialized.")

        raw_candidate = self._variables_to_array(raw_variables)
        feasible_candidate = self._variables_to_array(particle.candidate_variables)
        violation = raw_candidate - feasible_candidate

        with np.errstate(over="ignore", invalid="ignore"):
            penalty = float(
                np.sum(self._penalty_gamma * violation**2) / self._dimension
            )

        if np.isnan(penalty):
            raise RuntimeError("CMA-ES boundary penalty produced NaN.")

        return fitness + penalty

    def pre_iteration(self, actual_iter: int) -> None:
        if len(self._population) < 2:
            raise ValueError("CMA-ES requires at least 2 particles.")

        if actual_iter == 0:
            self._initialize_parameters()

            initial_values = np.asarray(
                [
                    self._variables_to_array(particle.variables)
                    for particle in self._population.values()
                ]
            )

            self._mean = np.mean(initial_values, axis=0)
            self._injected_particle_ids = set()
        else:
            self._injected_particle_ids = set(self.new_particles_id)

        self._update_strategy_parameters()

        if self._mean is None:
            raise RuntimeError("CMA-ES mean has not been initialized.")

    def create_random_cache(self, particle_ids: list[str], initialize: bool) -> None:
        dimension = self._dimension

        for particle_id in particle_ids:
            cache = {}

            if not initialize and particle_id not in self._injected_particle_ids:
                if dimension is None:
                    raise RuntimeError("CMA-ES dimension has not been initialized.")

                if self._boundary_handling == "resampling":
                    cache["cmaes-rng"] = np.random.default_rng(
                        self._rng.integers(0, 2**32, size=4, dtype=np.uint32)
                    )
                else:
                    cache["cmaes-z"] = self._rng.standard_normal(dimension)

            self._population[particle_id].random_cache = cache

    def initialize_particle(self, identifier: str) -> CMAESCandidateSolution:
        particle = self._population[identifier]

        if not isinstance(particle, CMAESCandidateSolution):
            raise TypeError(
                f"Particle '{identifier}' must be an instance of "
                "CMAESCandidateSolution."
            )

        if particle.fitness == FITNESS_UNDEFINED:
            initial_values = self._variables_to_array(particle.variables)

            if not self._is_within_boundaries(initial_values):
                raise RuntimeError(
                    f"Particle '{identifier}' cannot be evaluated "
                    "outside CMA-ES boundaries."
                )

            particle.update(
                variables=particle.variables, fitness_function=self._fitness_function
            )

        return particle

    @staticmethod
    def consolidate_new_particles(
        particle: CMAESCandidateSolution,
    ) -> CMAESCandidateSolution:
        particle.consolidate(consolidate_new=True)
        return particle

    def update_particle(self, identifier: str) -> CMAESCandidateSolution:
        particle = self._population[identifier]

        if not isinstance(particle, CMAESCandidateSolution):
            raise TypeError(
                f"Particle '{identifier}' must be an instance of "
                "CMAESCandidateSolution."
            )

        particle.raw_candidate_variables = None

        if identifier in self._injected_particle_ids:
            if particle.candidate_variables is None:
                particle.candidate_variables = dict(particle.variables)

            if particle.candidate_fitness is None:
                particle.candidate_fitness = particle.fitness

            return particle

        if self._mean is None or self._covariance is None or self._step_size is None:
            raise RuntimeError("CMA-ES state has not been initialized.")

        transform = self._sampling_transform(self._covariance)

        if self._boundary_handling == "resampling":
            raw_candidate = self._resample_candidate(
                particle=particle, transform=transform
            )
            evaluated_candidate = raw_candidate
        else:
            z = particle.random_cache["cmaes-z"]
            raw_candidate = self._raw_candidate_from_z(z=z, transform=transform)

            if not np.all(np.isfinite(raw_candidate)):
                raise RuntimeError("CMA-ES generated a non-finite candidate solution.")

            if self._boundary_handling in ("adaptive-penalty", "clip"):
                evaluated_candidate = self._project_to_boundaries(raw_candidate)
            else:
                evaluated_candidate = self._reflect_to_boundaries(raw_candidate)

        if not self._is_within_boundaries(evaluated_candidate):
            raise RuntimeError(
                "CMA-ES boundary handling produced an infeasible candidate."
            )

        particle.raw_candidate_variables = self._array_to_variables(raw_candidate)
        particle.update(
            variables=self._array_to_variables(evaluated_candidate),
            fitness_function=self._fitness_function,
        )

        return particle

    def post_iteration(self, actual_iter: int) -> None:
        if actual_iter == 0:
            self._injected_particle_ids = set()
            self.update_solution_state()
            return

        if (
            self._mean is None
            or self._covariance is None
            or self._step_size is None
            or self._p_sigma is None
            or self._p_c is None
            or self._weights is None
            or self._mu is None
            or self._mu_eff is None
            or self._cc is None
            or self._cs is None
            or self._c1 is None
            or self._cmu is None
            or self._damps is None
            or self._chi_n is None
            or self._dimension is None
        ):
            raise RuntimeError("CMA-ES state has not been initialized.")

        mean = self._mean
        covariance = self._covariance
        step_size = self._step_size
        p_sigma = self._p_sigma
        p_c = self._p_c
        gamma_sigma = self._gamma_sigma
        gamma_c = self._gamma_c
        weights = self._weights
        mu = self._mu
        mu_eff = self._mu_eff
        cc = self._cc
        cs = self._cs
        c1 = self._c1
        cmu = self._cmu
        damps = self._damps
        chi_n = self._chi_n
        dimension = self._dimension

        particles = list(self._population.values())

        for particle in particles:
            if particle.candidate_fitness is None:
                raise RuntimeError(
                    f"Particle '{particle.identifier}' has no candidate fitness."
                )

        if self._boundary_handling == "adaptive-penalty":
            self._update_boundary_penalty(
                actual_iter=actual_iter,
                particles=particles,
                mean=mean,
                covariance=covariance,
                step_size=step_size,
                mu_eff=mu_eff,
                dimension=dimension,
            )
            particles.sort(key=self._boundary_penalty_score)
        else:

            def fitness_key(particle: CMAESCandidateSolution) -> float:
                if particle.candidate_fitness is None:
                    raise RuntimeError(
                        f"Particle '{particle.identifier}' has no candidate fitness."
                    )

                return float(particle.candidate_fitness)

            particles.sort(key=fitness_key)

        selected = particles[:mu]
        old_mean = mean.copy()
        selected_variables = []

        for particle in selected:
            if particle.candidate_variables is None:
                raise RuntimeError(
                    f"Particle '{particle.identifier}' has no candidate variables."
                )

            if (
                particle.identifier in self._injected_particle_ids
                or self._boundary_handling == "clip"
                or (
                    self._boundary_handling == "adaptive-penalty"
                    and particle.candidate_fitness == -np.inf
                )
            ):
                adaptation_variables = particle.candidate_variables
            else:
                adaptation_variables = particle.raw_candidate_variables

                if adaptation_variables is None:
                    raise RuntimeError(
                        f"Particle '{particle.identifier}' "
                        "has no raw candidate variables."
                    )

            selected_variables.append(self._variables_to_array(adaptation_variables))

        selected_candidates = np.asarray(selected_variables)
        y_selected = (selected_candidates - old_mean) / step_size

        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, np.finfo(float).eps)

        inv_sqrt_covariance = (
            eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
        )

        injection_limit = np.sqrt(dimension) + 2.0 * dimension / (dimension + 2.0)

        for index, particle in enumerate(selected):
            if particle.identifier not in self._injected_particle_ids:
                continue

            mahalanobis_norm = np.linalg.norm(inv_sqrt_covariance @ y_selected[index])

            if mahalanobis_norm > injection_limit:
                y_selected[index] *= injection_limit / mahalanobis_norm

        y_w = np.sum(weights[:, np.newaxis] * y_selected, axis=0)
        new_mean = old_mean + step_size * y_w

        p_sigma = (1.0 - cs) * p_sigma + np.sqrt(cs * (2.0 - cs) * mu_eff) * (
            inv_sqrt_covariance @ y_w
        )

        gamma_sigma = (1.0 - cs) ** 2 * gamma_sigma + cs * (2.0 - cs)

        p_sigma_norm = np.linalg.norm(p_sigma)

        h_sigma = int(
            p_sigma_norm**2 / gamma_sigma < (2.0 + 4.0 / (dimension + 1.0)) * dimension
        )

        p_c = (1.0 - cc) * p_c + h_sigma * np.sqrt(cc * (2.0 - cc) * mu_eff) * y_w

        gamma_c = (1.0 - cc) ** 2 * gamma_c + h_sigma * cc * (2.0 - cc)

        rank_mu = np.zeros_like(covariance)

        for weight, y in zip(weights, y_selected):
            rank_mu += weight * np.outer(y, y)

        covariance = (
            (1.0 - c1 * gamma_c - cmu) * covariance
            + c1 * np.outer(p_c, p_c)
            + cmu * rank_mu
        )

        covariance = (covariance + covariance.T) / 2.0

        step_size *= np.exp(
            (cs / damps) * (p_sigma_norm / chi_n - np.sqrt(gamma_sigma))
        )

        self._mean = new_mean
        self._covariance = covariance
        self._step_size = step_size
        self._p_sigma = p_sigma
        self._p_c = p_c
        self._gamma_sigma = gamma_sigma
        self._gamma_c = gamma_c

        for particle in self._population.values():
            if particle.candidate_fitness is None:
                raise RuntimeError(
                    f"Particle '{particle.identifier}' has no candidate fitness."
                )

            particle.consolidate(consolidate_new=True)
            particle.raw_candidate_variables = None

        self._injected_particle_ids = set()
        self.update_solution_state()
