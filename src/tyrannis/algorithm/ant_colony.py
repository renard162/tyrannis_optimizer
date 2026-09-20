import warnings
from collections.abc import Collection
from copy import deepcopy
from typing import cast

import numpy as np
from scipy.linalg import qr

from ..core.algorithm import FITNESS_UNDEFINED, AlgorithmBase, ParticleBase


class ACORParticle(ParticleBase):
    """Particle implementation for the Ant Colony Optimization for Continuous Domain algorithm."""


class AntColony(AlgorithmBase[ACORParticle]):
    """Ant Colony Optimization for Continuous Domains."""

    def __init__(
        self,
        archive_size: int | str = 20,
        q: float = 0.25,
        xi: float = 0.85,
        separable: bool = True,
    ) -> None:
        """
        Ant Colony Optimization for Continuous Domains (ACOR).

        ACOR is a continuous-domain extension of the Ant Colony Optimization (ACO)
        metaheuristic, originally developed for combinatorial optimization. Instead of
        using pheromone values associated with discrete solution components, ACOR
        represents pheromone information through an archive of promising continuous
        solutions and generates new candidate solutions by sampling probability
        distributions around them.

        By default, this implementation uses Sep-ACOR, the separable variant of ACOR.
        Sep-ACOR samples decision variables independently in the original coordinate
        system, avoiding the coordinate-system adaptation required by the original
        ACOR formulation. Setting ``separable=False`` enables the original ACOR
        correlation-handling mechanism, which constructs an adaptive orthonormal
        coordinate system so that dependencies between decision variables can influence
        the generation of new solutions.

        Parameters
        ----------
        archive_size : int or "auto", default=20
            Number of solutions maintained in the solution archive. Integer values are
            used exactly as provided and must be at least 2. With the default Sep-ACOR
            formulation, this lower limit is independent of problem dimensionality.
            When ``separable=False``, an integer archive size must additionally be at
            least the number of encoded decision variables used internally by the search
            space.

            The value ``"auto"`` is available only when ``separable=False``. In this
            mode, the archive size is determined after the search space has been
            configured and is set to the number of encoded decision variables required
            by the original ACOR correlation-handling mechanism. The universal minimum
            archive size of 2 is retained for one-dimensional encoded spaces. A warning
            reports the resulting archive size.

        q : float, default=0.25
            Controls the concentration of the probability assigned to solutions in the
            archive according to their rank. Smaller values increase the preference
            for the best solutions, while larger values distribute the probability more
            evenly across the archive. Its effect depends on ``archive_size``, since the
            width of the rank-based distribution is proportional to their product.

        xi : float, default=0.85
            Controls the standard deviation of the probability distributions used to
            generate new solutions. Larger values increase the search range around
            archive solutions, while smaller values concentrate the search around them.

        separable : bool, default=True
            Whether to use Sep-ACOR, the separable variant of ACOR. The default value
            ``True`` samples each decision variable independently in the original
            coordinate system. This reduces computational cost, avoids the
            orthogonalization required by the original ACOR formulation, and only
            requires an integer ``archive_size >= 2``. Automatic archive sizing is not
            available in this mode because its minimum archive size is fixed and does
            not depend on problem dimensionality.

            When ``False``, the original ACOR formulation is used. An adaptive
            orthonormal coordinate system is constructed from the solution archive so
            that correlations between decision variables can influence the search.
            This formulation is computationally more demanding and requires
            ``archive_size`` to be at least the number of encoded decision variables.
            ``archive_size="auto"`` can be used to determine this minimum-compatible
            archive size automatically after the search space has been configured.

        Notes
        -----
        Ant Colony Optimization for Continuous Domains (ACOR) adapts the main
        principles of Ant Colony Optimization to continuous search spaces. Instead of
        associating pheromone values with discrete solution components, ACOR uses an
        archive of solutions as its pheromone representation. The archive is sorted
        according to solution quality, with better solutions receiving greater
        probability of being selected as references for generating new solutions.

        For each new solution, one archive solution is selected according to its
        rank-based probability. The default Sep-ACOR formulation treats the search
        dimensions as separable: each decision variable is sampled independently from
        a normal distribution centered on the corresponding value of the selected
        archive solution. The standard deviation is calculated from the dispersion of
        that variable among the archived solutions and scaled by ``xi``. This
        formulation has lower computational cost and imposes fewer requirements on
        the archive size, making it suitable as the general-purpose default.

        Setting ``separable=False`` enables the correlation-handling mechanism of the
        original ACOR formulation. Instead of sampling directly in the original
        coordinate system, ACOR progressively constructs an adaptive orthonormal
        coordinate system from directions represented by the solution archive.
        Directions farther from the selected reference in the remaining search
        subspace are more likely to define subsequent axes. Gaussian sampling is then
        performed in this temporary coordinate system before the generated solution
        is transformed back to the original variables. This allows correlations
        between decision variables to influence the search, at the cost of additional
        computation and the requirement that the archive contain at least as many
        solutions as there are encoded decision variables. When this encoded
        dimensionality is not known in advance, ``archive_size="auto"`` derives the
        required archive size from the configured search space and reports the
        resulting value with a warning.

        The parameter ``q`` controls the selection pressure applied to the archive.
        Values that concentrate probability on the best-ranked solutions increase
        exploitation, while more evenly distributed probabilities preserve greater
        exploration of the solutions stored in the archive.

        After new solutions are evaluated, they are combined with the existing archive
        and the best ``archive_size`` solutions are retained. Ties between solutions
        with the same objective value are resolved randomly. The archive therefore
        acts as both the memory of the colony and the basis for generating subsequent
        solutions.

        References
        ----------
        Socha, K., & Dorigo, M. (2008). Ant colony optimization for continuous
        domains. European Journal of Operational Research, 185(3), 1155-1173.
        https://doi.org/10.1016/j.ejor.2006.06.046

        Liao, T., Molina, D., Stützle, T., Montes de Oca, M. A., & Dorigo, M. (2012).
        An ACO algorithm benchmarked on the BBOB noiseless function testbed.
        Proceedings of the 14th Annual Conference Companion on Genetic and
        Evolutionary Computation, 159-166.
        https://doi.org/10.1145/2330784.2330809

        Afshar, A., & Madadgar, S. (2008). Ant Colony Optimization for Continuous
        Domains: Application to Reservoir Operation Problems. 2008 Eighth
        International Conference on Hybrid Intelligent Systems.
        https://doi.org/10.1109/HIS.2008.121

        Moradi, B., Kargar, A., & Abazari, S. (2022). Transient stability constrained
        optimal power flow solution using ant colony optimization for continuous
        domains (ACOR). IET Generation, Transmission & Distribution, 16(18),
        3734-3747.
        https://doi.org/10.1049/gtd2.12560
        """
        if not isinstance(separable, (bool, np.bool_)):
            raise TypeError("separable must be a boolean.")

        if isinstance(archive_size, (bool, np.bool_)):
            raise TypeError("archive_size must be an integer or 'auto'.")

        if isinstance(archive_size, (int, np.integer)):
            if archive_size < 2:
                raise ValueError("archive_size must be greater than or equal to 2.")
        elif isinstance(archive_size, str):
            if archive_size != "auto":
                raise ValueError("archive_size must be an integer or 'auto'.")

            if separable:
                raise ValueError(
                    "archive_size='auto' is only available when separable=False. "
                    "Sep-ACOR requires an integer archive_size greater than or equal to 2."
                )
        else:
            raise TypeError("archive_size must be an integer or 'auto'.")

        if not isinstance(q, (int, float, np.number)):
            raise TypeError("q must be a number.")

        if not np.isfinite(q) or q <= 0:
            raise ValueError("q must be a finite number greater than 0.")

        if not isinstance(xi, (int, float, np.number)):
            raise TypeError("xi must be a number.")

        if not np.isfinite(xi) or xi <= 0:
            raise ValueError("xi must be a finite number greater than 0.")

        self._automatic_archive_size = archive_size == "auto"
        self._archive_size = 2 if self._automatic_archive_size else int(archive_size)
        self._archive_size_resolved = not self._automatic_archive_size
        self._q = float(q)
        self._xi = float(xi)
        self._separable = bool(separable)

        self._solution_archive: list[tuple[dict[str, float], np.float64]] = []
        self._archive_probabilities: np.ndarray | None = None

        self._initial_archive_particle_ids: list[str] = []
        self._temporary_particle_ids: list[str] = []

    @property
    def archive_size(self) -> int | str:
        if self._automatic_archive_size and not self._archive_size_resolved:
            return "auto"

        return self._archive_size

    @property
    def q(self) -> float:
        return self._q

    @property
    def xi(self) -> float:
        return self._xi

    @property
    def separable(self) -> bool:
        return self._separable

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

        self._population[identifier] = ACORParticle(
            identifier=identifier, variables=variables, fitness=fitness
        )

    def delete_particle(self, identifier: str | None) -> None:
        if identifier is None:
            raise ValueError("Particle identifier cannot be None.")

        del self._population[identifier]

    def _create_temporary_particle_ids(self, count: int) -> list[str]:
        identifiers = []
        index = 0

        while len(identifiers) < count:
            identifier = f"{self._identifier}|acor-initial:{index}"
            index += 1

            if identifier in self._population:
                continue

            identifiers.append(identifier)

        return identifiers

    def _validate_archive_size(self) -> None:
        if self._separable:
            return

        n_dimensions = len(self._boundaries)

        if self._automatic_archive_size:
            if not self._archive_size_resolved:
                self._archive_size = max(2, n_dimensions)
                self._archive_size_resolved = True

                warnings.warn(
                    f"archive_size='auto' was resolved to {self._archive_size} because "
                    f"ACOR is operating on {n_dimensions} encoded decision variables.",
                    UserWarning,
                    stacklevel=2,
                )

            return

        if self._archive_size < n_dimensions:
            raise ValueError(
                f"archive_size ({self._archive_size}) must be greater than or equal "
                f"to the number of encoded decision variables ({n_dimensions}) when "
                "separable=False. Set archive_size='auto' to use the minimum valid "
                "archive size automatically, or use separable=True, which requires "
                "only archive_size >= 2."
            )

    def pre_iteration(self, actual_iter: int) -> None:
        self._validate_archive_size()

        if actual_iter == 0:
            population_ids = list(self._population)

            if len(population_ids) >= self._archive_size:
                selected = self._rng.choice(
                    population_ids, size=self._archive_size, replace=False
                )

                self._initial_archive_particle_ids = [
                    str(identifier) for identifier in selected
                ]
                self._temporary_particle_ids = []
            else:
                missing = self._archive_size - len(population_ids)

                self._temporary_particle_ids = self._create_temporary_particle_ids(
                    missing
                )
                self._initial_archive_particle_ids = [
                    *population_ids,
                    *self._temporary_particle_ids,
                ]

                for identifier in self._temporary_particle_ids:
                    self.create_particle(identifier=identifier)

            return

        migrated_particles = [
            particle
            for particle in self._population.values()
            if particle.new_particle and np.isfinite(particle.fitness)
        ]

        if migrated_particles:
            self._update_solution_archive(migrated_particles)
            self.update_solution_state()

        if self._archive_probabilities is None:
            self._archive_probabilities = self._calculate_archive_probabilities()

    def _calculate_archive_probabilities(self) -> np.ndarray:
        positions = np.arange(self._archive_size, dtype=float)

        weights = np.exp(-(positions**2) / (2 * self._q**2 * self._archive_size**2))

        return weights / np.sum(weights)

    def create_random_cache(self, particle_ids: list[str], initialize: bool) -> None:
        variables = tuple(self._boundaries)

        for identifier in particle_ids:
            particle = self._population[identifier]
            particle.random_cache = {}

            if initialize:
                continue

            if self._archive_probabilities is None:
                raise RuntimeError("Archive probabilities have not been initialized.")

            particle.random_cache["archive-index"] = int(
                self._rng.choice(self._archive_size, p=self._archive_probabilities)
            )

            for variable in variables:
                particle.random_cache[f"{variable}-normal"] = float(self._rng.normal())

            if self._separable:
                continue

            for step in range(len(variables)):
                particle.random_cache[f"direction-{step}-uniform"] = float(
                    self._rng.random()
                )

                for variable in variables:
                    particle.random_cache[f"fallback-{step}-{variable}"] = float(
                        self._rng.normal()
                    )

    def initialize_particle(self, identifier: str) -> ACORParticle:
        particle = self._population[identifier]

        if not isinstance(particle, ACORParticle):
            raise TypeError(
                f"Particle '{identifier}' must be an instance of ACORParticle."
            )

        if np.isinf(particle.fitness):
            particle.update(
                variables=particle.variables, fitness_function=self._fitness_function
            )

        return particle

    @staticmethod
    def consolidate_new_particles(particle: ACORParticle) -> ACORParticle:
        particle.consolidate(consolidate_new=True)
        return particle

    def _calculate_sigma(self, archive_index: int, variable: str) -> float:
        reference_value = self._solution_archive[archive_index][0][variable]

        distances = sum(
            abs(solution[variable] - reference_value)
            for solution, _ in self._solution_archive
        )

        return self._xi * distances / (self._archive_size - 1)

    def _generate_separable_solution(self, particle: ACORParticle) -> dict[str, float]:
        archive_index = particle.random_cache["archive-index"]
        reference_variables = self._solution_archive[archive_index][0]

        new_variables = {}

        for variable, (lower, upper) in self._boundaries.items():
            normal_value = particle.random_cache[f"{variable}-normal"]

            sigma = self._calculate_sigma(
                archive_index=archive_index, variable=variable
            )

            value = reference_variables[variable] + sigma * normal_value

            new_variables[variable] = float(np.clip(value, lower, upper))

        return new_variables

    @staticmethod
    def _project_to_remaining_subspace(
        vectors: np.ndarray, basis: np.ndarray
    ) -> np.ndarray:
        if basis.shape[1] == 0:
            return vectors.copy()

        return vectors - (vectors @ basis) @ basis.T

    @staticmethod
    def _orthonormalize(basis: np.ndarray, direction: np.ndarray) -> np.ndarray:
        if basis.shape[1] == 0:
            input_matrix = direction[:, np.newaxis]
        else:
            input_matrix = np.column_stack((basis, direction))

        q_matrix, r_matrix = cast(
            tuple[np.ndarray, np.ndarray],
            qr(input_matrix, mode="economic", pivoting=False, check_finite=False),
        )

        diagonal = np.diag(r_matrix)
        signs = np.where(diagonal < 0, -1.0, 1.0)

        return q_matrix * signs[np.newaxis, :]

    def _fallback_direction(
        self,
        particle: ACORParticle,
        step: int,
        variables: tuple[str, ...],
        basis: np.ndarray,
    ) -> np.ndarray:
        random_direction = np.asarray(
            [
                particle.random_cache[f"fallback-{step}-{variable}"]
                for variable in variables
            ],
            dtype=float,
        )

        projected = self._project_to_remaining_subspace(
            random_direction[np.newaxis, :], basis
        )[0]

        tolerance = np.finfo(float).eps * max(1, len(variables)) * 100

        if np.linalg.norm(projected) > tolerance:
            return projected

        for index in range(len(variables)):
            canonical = np.zeros(len(variables), dtype=float)
            canonical[index] = 1.0

            projected = self._project_to_remaining_subspace(
                canonical[np.newaxis, :], basis
            )[0]

            if np.linalg.norm(projected) > tolerance:
                return projected

        raise RuntimeError("Unable to construct an ACOR coordinate direction.")

    @staticmethod
    def _select_direction_index(
        distances: np.ndarray, random_value: float
    ) -> int | None:
        max_distance = float(np.max(distances))

        if max_distance <= 0 or not np.isfinite(max_distance):
            return None

        scaled_distances = distances / max_distance
        weights = scaled_distances**4
        total_weight = float(np.sum(weights))

        if total_weight <= 0 or not np.isfinite(total_weight):
            return None

        cumulative = np.cumsum(weights / total_weight)

        index = int(np.searchsorted(cumulative, random_value, side="right"))

        return min(index, len(distances) - 1)

    def _generate_correlated_solution(self, particle: ACORParticle) -> dict[str, float]:
        variables = tuple(self._boundaries)
        n_dimensions = len(variables)

        archive_index = particle.random_cache["archive-index"]

        archive = np.asarray(
            [
                [solution[variable] for variable in variables]
                for solution, _ in self._solution_archive
            ],
            dtype=float,
        )

        reference = archive[archive_index]
        differences = archive - reference

        basis = np.empty((n_dimensions, 0), dtype=float)
        sampled_coordinates = np.empty(n_dimensions, dtype=float)

        tolerance = np.finfo(float).eps * max(1, n_dimensions) * 100

        for step, variable in enumerate(variables):
            residuals = self._project_to_remaining_subspace(differences, basis)

            distances = np.linalg.norm(residuals, axis=1)

            direction_index = self._select_direction_index(
                distances=distances,
                random_value=particle.random_cache[f"direction-{step}-uniform"],
            )

            if direction_index is None:
                direction = self._fallback_direction(
                    particle=particle, step=step, variables=variables, basis=basis
                )
            else:
                direction = residuals[direction_index]

            if np.linalg.norm(direction) <= tolerance:
                direction = self._fallback_direction(
                    particle=particle, step=step, variables=variables, basis=basis
                )

            basis = self._orthonormalize(basis=basis, direction=direction)

            axis = basis[:, step]

            archive_coordinates = archive @ axis
            reference_coordinate = float(reference @ axis)

            sigma = (
                self._xi
                * float(np.sum(np.abs(archive_coordinates - reference_coordinate)))
                / (self._archive_size - 1)
            )

            sampled_coordinates[step] = (
                reference_coordinate
                + sigma * particle.random_cache[f"{variable}-normal"]
            )

        new_position = basis @ sampled_coordinates

        return {
            variable: float(np.clip(new_position[index], lower, upper))
            for index, (variable, (lower, upper)) in enumerate(self._boundaries.items())
        }

    def update_particle(self, identifier: str) -> ACORParticle:
        particle = self._population[identifier]

        if not isinstance(particle, ACORParticle):
            raise TypeError(
                f"Particle '{identifier}' must be an instance of ACORParticle."
            )

        if self._separable:
            new_variables = self._generate_separable_solution(particle)
        else:
            new_variables = self._generate_correlated_solution(particle)

        particle.update(
            variables=new_variables, fitness_function=self._fitness_function
        )

        return particle

    @staticmethod
    def _fitness_sort_key(solution: tuple[dict[str, float], np.float64]) -> float:
        fitness = solution[1]

        if np.isnan(fitness):
            return np.inf

        return float(fitness)

    def _rank_solutions(
        self, candidates: list[tuple[dict[str, float], np.float64]]
    ) -> list[tuple[dict[str, float], np.float64]]:
        if not candidates:
            return []

        permutation = self._rng.permutation(len(candidates))

        randomized = [candidates[index] for index in permutation]

        randomized.sort(key=self._fitness_sort_key)

        return randomized

    def _initialize_solution_archive(self, particles: Collection[ACORParticle]) -> None:
        candidates = [
            (deepcopy(particle.variables), particle.fitness) for particle in particles
        ]

        if len(candidates) != self._archive_size:
            raise RuntimeError(
                "Initial ACOR solution archive must contain exactly "
                f"{self._archive_size} solutions."
            )

        self._solution_archive = self._rank_solutions(candidates)

    def _update_solution_archive(self, particles: Collection[ACORParticle]) -> None:
        candidates = [
            (deepcopy(particle.variables), particle.fitness) for particle in particles
        ]

        candidates.extend(self._solution_archive)

        self._solution_archive = self._rank_solutions(candidates)[: self._archive_size]

    def post_iteration(self, actual_iter: int) -> None:
        if actual_iter == 0:
            particles = [
                self._population[identifier]
                for identifier in self._initial_archive_particle_ids
            ]

            if not all(isinstance(particle, ACORParticle) for particle in particles):
                raise TypeError("All particles must be instances of ACORParticle.")

            self._initialize_solution_archive(particles)

            self._archive_probabilities = self._calculate_archive_probabilities()

            self.update_solution_state()

            for identifier in self._temporary_particle_ids:
                self.delete_particle(identifier)

            if self._temporary_particle_ids:
                self.update_solution_state()

            self._temporary_particle_ids = []
            return

        for particle in self._population.values():
            if not isinstance(particle, ACORParticle):
                raise TypeError(
                    f"Particle '{particle.identifier}' must be an instance of "
                    "ACORParticle."
                )

            if particle.candidate_fitness is None:
                raise RuntimeError(
                    f"Particle '{particle.identifier}' has no candidate fitness."
                )

            particle.consolidate(consolidate_new=True)

        self._update_solution_archive(list(self._population.values()))

        self.update_solution_state()
