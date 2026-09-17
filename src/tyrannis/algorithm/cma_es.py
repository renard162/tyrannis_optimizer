import numpy as np
from copy import deepcopy

from ..core.algorithm import (
    FITNESS_UNDEFINED,
    AlgorithmBase,
    ParticleBase,
)


class CMAESCandidateSolution(ParticleBase):
    """Candidate solution for Covariance Matrix Adaptation Evolution Strategy algorithm."""


class CMAES(AlgorithmBase[CMAESCandidateSolution]):
    """Covariance Matrix Adaptation Evolution Strategy algorithm."""

    def __init__(
        self,
        sigma: float = 0.3,
    ) -> None:
        """
        Covariance Matrix Adaptation Evolution Strategy algorithm.

        CMA-ES is a population-based optimization algorithm for continuous
        optimization that adapts a multivariate normal distribution to the
        geometry of the search landscape. Candidate solutions are sampled from
        the current distribution, and the distribution's mean, covariance
        matrix, and step size are updated from the best candidates.

        Parameters
        ----------
        sigma : float, default=0.3
            Initial global step size expressed as a fraction of the average
            variable range. The value must be greater than zero.

        Notes
        -----
        The population size is obtained from the optimization context and is
        not an algorithm constructor parameter. The initial mean is calculated
        from the initial population created by the processor. From the first
        optimization iteration onward, new candidate solutions are sampled
        from the adapted distribution.

        The mean is represented by a temporary particle so that its cost can
        be evaluated through the particle execution contract. This particle
        does not belong to the CMA-ES population and is removed before the
        iteration is consolidated.

        References
        ----------
        Hansen, N., & Ostermeier, A. (2001). Completely Derandomized Self-
        Adaptation in Evolution Strategies. Evolutionary Computation, 9(2),
        159-195. https://doi.org/10.1162/106365601750190398
        """
        if not isinstance(sigma, (int, float, np.number)):
            raise TypeError("sigma must be a number.")

        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be a finite number greater than 0.")

        self._initial_sigma = float(sigma)

        self._mean_particle: CMAESCandidateSolution | None = None
        self._mean: np.ndarray | None = None
        self._covariance: np.ndarray | None = None
        self._step_size: float | None = None
        self._p_sigma: np.ndarray | None = None
        self._p_c: np.ndarray | None = None
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

    @property
    def sigma(self) -> float:
        if self._step_size is None:
            return self._initial_sigma
        return self._step_size

    @property
    def mean_particle(self) -> CMAESCandidateSolution | None:
        return self._mean_particle

    @property
    def _center_identifier(self) -> str:
        return f"{self._identifier}|center"

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

        self._mu = max(1, self._n_particles // 2)

        indices = np.arange(1, self._mu + 1, dtype=float)
        weights = np.log(self._mu + 0.5) - np.log(indices)
        self._weights = weights / np.sum(weights)
        self._mu_eff = 1.0 / np.sum(self._weights**2)

        n = self._dimension
        mu_eff = self._mu_eff

        self._cc = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)

        self._cs = (mu_eff + 2.0) / (n + mu_eff + 5.0)

        self._c1 = 2.0 / ((n + np.sqrt(2.0)) ** 2 + mu_eff)

        self._cmu = min(
            1.0 - self._c1,
            2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff),
        )

        self._damps = (
            1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + self._cs
        )

        self._chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n**2))

        self._covariance = np.eye(n)
        self._p_sigma = np.zeros(n)
        self._p_c = np.zeros(n)

        ranges = np.asarray(
            [upper - lower for lower, upper in self._boundaries.values()], dtype=float
        )

        self._step_size = self._initial_sigma * float(np.mean(ranges))

    def _variables_to_array(self, variables: dict[str, float]) -> np.ndarray:
        return np.asarray(
            [variables[name] for name in self._variable_names], dtype=float
        )

    def _array_to_variables(self, values: np.ndarray) -> dict[str, float]:
        return {name: float(value) for name, value in zip(self._variable_names, values)}

    def pre_iteration(self, actual_iter: int) -> None:
        center_identifier = self._center_identifier

        if actual_iter == 0:
            self._initialize_parameters()

            initial_values = np.asarray(
                [
                    self._variables_to_array(particle.variables)
                    for particle in self._population.values()
                ]
            )

            self._mean = np.mean(initial_values, axis=0)

        if self._mean is None:
            raise RuntimeError("CMA-ES mean has not been initialized.")

        self.create_particle(
            identifier=center_identifier, variables=self._array_to_variables(self._mean)
        )

    def create_random_cache(self, particle_ids: list[str], initialize: bool) -> None:
        for particle_id in particle_ids:
            cache = {}

            if not initialize and particle_id != self._center_identifier:
                cache["cmaes-z"] = self._rng.standard_normal(self._dimension)

            self._population[particle_id].random_cache = cache

    def initialize_particle(self, identifier: str) -> CMAESCandidateSolution:
        particle = self._population[identifier]

        if not isinstance(particle, CMAESCandidateSolution):
            raise TypeError(
                f"Particle '{identifier}' must be an instance of "
                "CMAESCandidateSolution."
            )

        if np.isinf(particle.fitness):
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

        if identifier == self._center_identifier:
            return particle

        if self._mean is None or self._covariance is None or self._step_size is None:
            raise RuntimeError("CMA-ES state has not been initialized.")

        eigenvalues, eigenvectors = np.linalg.eigh(self._covariance)
        eigenvalues = np.maximum(eigenvalues, np.finfo(float).eps)
        transform = eigenvectors @ np.diag(np.sqrt(eigenvalues))
        z = particle.random_cache["cmaes-z"]
        candidate = self._mean + self._step_size * (transform @ z)

        for index, name in enumerate(self._variable_names):
            lower, upper = self._boundaries[name]
            candidate[index] = np.clip(candidate[index], lower, upper)

        particle.update(
            variables=self._array_to_variables(candidate),
            fitness_function=self._fitness_function,
        )

        return particle

    def post_iteration(self, actual_iter: int) -> None:
        center_identifier = self._center_identifier

        if actual_iter == 0:
            center = self._population[center_identifier]
            self._mean_particle = deepcopy(center)

            self.delete_particle(center_identifier)
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

        particles = [
            particle
            for particle_id, particle in self._population.items()
            if particle_id != center_identifier
        ]

        for particle in particles:
            if particle.candidate_fitness is None:
                raise RuntimeError(
                    f"Particle '{particle.identifier}' has no candidate fitness."
                )

        particles.sort(key=lambda particle: particle.candidate_fitness)
        selected = particles[: self._mu]
        old_mean = self._mean.copy()

        selected_candidates = np.asarray(
            [
                self._variables_to_array(particle.candidate_variables)
                for particle in selected
            ]
        )

        new_mean = np.sum(self._weights[:, np.newaxis] * selected_candidates, axis=0)
        y_w = (new_mean - old_mean) / self._step_size
        eigenvalues, eigenvectors = np.linalg.eigh(self._covariance)
        eigenvalues = np.maximum(eigenvalues, np.finfo(float).eps)

        inv_sqrt_covariance = (
            eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
        )

        self._p_sigma = (1.0 - self._cs) * self._p_sigma + np.sqrt(
            self._cs * (2.0 - self._cs) * self._mu_eff
        ) * (inv_sqrt_covariance @ y_w)

        p_sigma_norm = np.linalg.norm(self._p_sigma)

        h_sigma = int(
            p_sigma_norm
            / np.sqrt(1.0 - (1.0 - self._cs) ** (2.0 * (actual_iter + 1)))
            / self._chi_n
            < 1.4 + 2.0 / (self._dimension + 1.0)
        )

        self._p_c = (1.0 - self._cc) * self._p_c + h_sigma * np.sqrt(
            self._cc * (2.0 - self._cc) * self._mu_eff
        ) * y_w

        y_selected = (selected_candidates - old_mean) / self._step_size
        rank_mu = np.zeros_like(self._covariance)

        for weight, y in zip(self._weights, y_selected):
            rank_mu += weight * np.outer(y, y)

        self._covariance = (
            (1.0 - self._c1 - self._cmu) * self._covariance
            + self._c1
            * (
                np.outer(self._p_c, self._p_c)
                + (1.0 - h_sigma) * self._cc * (2.0 - self._cc) * self._covariance
            )
            + self._cmu * rank_mu
        )

        self._covariance = (self._covariance + self._covariance.T) / 2.0

        self._step_size *= np.exp(
            (self._cs / self._damps) * (p_sigma_norm / self._chi_n - 1.0)
        )

        self._mean = new_mean
        center = self._population[center_identifier]
        self._mean_particle = deepcopy(center)
        self.delete_particle(center_identifier)

        for particle in particles:
            particle.consolidate(consolidate_new=True)

        self.update_solution_state()
