from collections.abc import Collection
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.special import gammaln
from scipy.stats import chi2, truncnorm

from ..core.algorithm import FITNESS_UNDEFINED, AlgorithmBase, ParticleBase


@dataclass(frozen=True)
class _ConditionalNormal:
    variable_index: int
    parent_indices: tuple[int, ...]
    mean: float
    parent_means: np.ndarray
    coefficients: np.ndarray
    standard_deviation: float


@dataclass(frozen=True)
class _BGeContext:
    sample_count: int
    dimension: int
    alpha_mu: float
    alpha_w: float
    prior_scale: np.ndarray
    posterior_scale: np.ndarray


class EGNAParticle(ParticleBase):
    """Particle implementation for the Estimation of Gaussian Networks Algorithm."""


class EGNA(AlgorithmBase[EGNAParticle]):
    """Estimation of Gaussian Networks Algorithm."""

    def __init__(
        self,
        selection_fraction: float = 0.5,
        structure_learning: str = "bic",
        edge_exclusion_alpha: float = 0.05,
        bge_alpha_mu: float = 1.0,
        bge_alpha_w: float | None = None,
    ) -> None:
        """
        Estimation of Gaussian Networks Algorithm (EGNA).

        EGNA is a multivariate continuous Estimation of Distribution Algorithm
        (EDA). At each optimization iteration, truncation selection identifies a
        set of promising solutions, a Gaussian Bayesian network is learned from
        those solutions, and the resulting joint Gaussian model is sampled to
        generate the next population. Dependencies between encoded decision
        variables are represented explicitly by the directed acyclic graph of the
        Gaussian network.

        The public ``structure_learning`` argument exposes the three classical
        EGNA structure-learning variants. ``"bic"`` uses penalized maximum
        likelihood with the Bayesian Information Criterion, ``"bge"`` uses the
        score-equivalent Bayesian Gaussian equivalent metric, and
        ``"edge-exclusion"`` tests edges in the saturated Gaussian graphical model
        and converts the retained dependencies to an ordered Gaussian network.

        Parameters
        ----------
        selection_fraction : float, default=0.5
            Fraction of the current population selected by truncation selection to
            learn the Gaussian network. The selected count is
            ``floor(selection_fraction * population_size)``. Because EGNA allows
            unrestricted multivariate dependencies, the selected population must
            contain more observations than encoded variables so that the Gaussian
            conditionals are identifiable.

        structure_learning : {"bic", "bge", "edge-exclusion"}, default="bic"
            Defines how the Gaussian network structure is learned.

            - ``"bic"``: uses the Bayesian Information Criterion to score candidate
            network structures during structure learning.
            - ``"bge"``: uses the Bayesian Gaussian equivalent score to evaluate
            candidate network structures.
            - ``"edge-exclusion"``: learns the network through conditional independence
            tests that determine which edges should be excluded.

        edge_exclusion_alpha : float, default=0.05
            Significance level used only when
            ``structure_learning="edge-exclusion"``. An edge is retained when the
            null hypothesis of conditional independence is rejected at this level.
            The likelihood-ratio edge-exclusion statistic is evaluated with the
            finite-sample correction described by Smith and Whittaker.

        bge_alpha_mu : float, default=1.0
            Positive equivalent sample size for the mean component of the
            normal-Wishart prior used only when ``structure_learning="bge"``.
            The default follows a standard weak BGe prior specification.

        bge_alpha_w : float or None, default=None
            Wishart equivalent sample size used only when
            ``structure_learning="bge"``. ``None`` resolves to ``d + 2``, where
            ``d`` is the number of encoded variables, following the conventional
            weak proper BGe prior. An explicitly supplied value must be greater
            than ``d + 1`` because the associated isotropic prior scale must be
            strictly positive.

        Notes
        -----
        For a selected population ``S``, EGNA models the joint density as a
        Gaussian Bayesian network

        ``p(X) = product_i p(X_i | Pa(X_i))``.

        Each local conditional is linear Gaussian,

        ``X_i = mu_i + beta_i^T (Pa(X_i) - mu_Pa(i)) + epsilon_i``,

        with ``epsilon_i`` Gaussian and independent of its parents. The network is
        sampled in topological order so that every parent value is available before
        its descendants are generated.

        With ``structure_learning="bic"``, network structures are compared using
        the sum of local Gaussian maximum-likelihood scores penalized by the
        Bayesian Information Criterion,

        ``BIC = log L_hat - 0.5 * k * log(N)``.

        This implementation builds the first structure with greedy forward search
        from the empty graph. Later generations use the previously learned structure
        as the starting point of a local search that considers one-edge additions
        and deletions and accepts the modification giving the largest strict BIC
        improvement.

        With ``structure_learning="bge"``, the same search policy is used but the
        local score is the Bayesian Gaussian equivalent marginal likelihood. A
        uniform prior over DAG structures is used, so structural comparisons are
        determined by the BGe marginal likelihood. EGNA uses a score-equivalent
        normal-Wishart prior. The mean prior equivalent sample size is
        ``bge_alpha_mu``; the Wishart equivalent sample size is
        ``bge_alpha_w`` or ``d + 2`` when automatic. In the normalized search
        coordinates, the prior mean is fixed at the center of the encoded box,
        ``0.5`` in every dimension, and the Wishart scale is isotropic with

        ``t = alpha_mu * (alpha_w - d - 1) / (alpha_mu + 1)``.

        With ``structure_learning="edge-exclusion"``, edge-exclusion tests are
        applied to the saturated Gaussian graphical model. For variables ``X_i``
        and ``X_j``, the sample partial correlation conditioned on all remaining
        variables is obtained from the precision matrix. The likelihood-ratio
        statistic is

        ``T = -N * log(1 - r_ij|rest**2)``.

        Edges whose conditional-independence null hypothesis is not rejected at
        ``edge_exclusion_alpha`` are removed. Retained undirected dependencies are
        represented as directed edges from earlier to later variables in the stable
        encoded-variable order, producing an acyclic network suitable for Gaussian
        ancestral sampling.

        EGNA is defined for continuous Gaussian models with unrestricted
        dependencies. This implementation therefore requires the selected sample
        count to exceed the number of encoded variables. Exact numerical
        degeneracies are handled at machine precision; they do not introduce a
        user-tunable regularization strength.

        Tyrannis optimizes over finite encoded boundaries. Model fitting is carried
        out after an affine normalization of every encoded variable to ``[0, 1]``.
        Sampling uses the corresponding conditional Gaussian truncated to that
        interval, which preserves valid encoded candidates without placing clipped
        probability mass directly on the boundaries.

        References
        ----------
        Larrañaga, P., Etxeberria, R., Lozano, J. A., & Peña, J. M. (2000).
        Optimization in Continuous Domains by Learning and Simulation of Gaussian
        Networks. Workshop on Optimization by Building and Using Probabilistic
        Models, GECCO 2000, 201-204.

        Bengoetxea, E., Miquélez, T., Larrañaga, P., & Lozano, J. A. (2002).
        Experimental Results in Function Optimization with EDAs in Continuous
        Domain. In Estimation of Distribution Algorithms: A New Tool for
        Evolutionary Computation, 181-194. Springer.
        https://doi.org/10.1007/978-1-4615-1539-5_8

        Geiger, D., & Heckerman, D. (1994). Learning Gaussian Networks.
        Uncertainty Proceedings 1994, 235-243.
        https://doi.org/10.1016/B978-1-55860-332-5.50035-3

        Smith, P. W. F., & Whittaker, J. (1998). Edge Exclusion Tests for
        Graphical Gaussian Models. In M. I. Jordan (Ed.), Learning in Graphical
        Models, 555-574. Springer.
        https://doi.org/10.1007/978-94-011-5014-9_21

        Soloviev, V. P., Bielza, C., & Larrañaga, P. (2024).
        Semiparametric Estimation of Distribution Algorithms for Continuous
        Optimization. IEEE Transactions on Evolutionary Computation, 28(4),
        1069-1083.
        https://doi.org/10.1109/TEVC.2023.3290670
        """
        if not isinstance(selection_fraction, (int, float, np.number)) or isinstance(
            selection_fraction, (bool, np.bool_)
        ):
            raise TypeError("selection_fraction must be a number.")

        if not np.isfinite(selection_fraction) or not 0 < selection_fraction <= 1:
            raise ValueError(
                "selection_fraction must be finite and greater than 0 and less "
                "than or equal to 1."
            )

        if not isinstance(structure_learning, str):
            raise TypeError("structure_learning must be a string.")

        allowed_structure_learning = ("bic", "bge", "edge-exclusion")

        if structure_learning not in allowed_structure_learning:
            raise ValueError(
                "structure_learning must be 'bic', 'bge', or 'edge-exclusion'."
            )

        if not isinstance(edge_exclusion_alpha, (int, float, np.number)) or isinstance(
            edge_exclusion_alpha, (bool, np.bool_)
        ):
            raise TypeError("edge_exclusion_alpha must be a number.")

        if not np.isfinite(edge_exclusion_alpha) or not 0 < edge_exclusion_alpha < 1:
            raise ValueError(
                "edge_exclusion_alpha must be finite and strictly between 0 and 1."
            )

        if not isinstance(bge_alpha_mu, (int, float, np.number)) or isinstance(
            bge_alpha_mu, (bool, np.bool_)
        ):
            raise TypeError("bge_alpha_mu must be a number.")

        if not np.isfinite(bge_alpha_mu) or bge_alpha_mu <= 0:
            raise ValueError("bge_alpha_mu must be a finite number greater than 0.")

        if bge_alpha_w is not None:
            if not isinstance(bge_alpha_w, (int, float, np.number)) or isinstance(
                bge_alpha_w, (bool, np.bool_)
            ):
                raise TypeError("bge_alpha_w must be a number or None.")

            if not np.isfinite(bge_alpha_w):
                raise ValueError("bge_alpha_w must be finite when provided.")

        self._selection_fraction = float(selection_fraction)
        self._structure_learning = structure_learning
        self._edge_exclusion_alpha = float(edge_exclusion_alpha)
        self._bge_alpha_mu = float(bge_alpha_mu)
        self._bge_alpha_w = None if bge_alpha_w is None else float(bge_alpha_w)

        self._variable_names: tuple[str, ...] = ()
        self._lower_bounds: np.ndarray | None = None
        self._spans: np.ndarray | None = None
        self._parents: tuple[tuple[int, ...], ...] = ()
        self._topological_order: tuple[int, ...] = ()
        self._conditionals: tuple[_ConditionalNormal, ...] = ()
        self._active_particle_ids: set[str] = set()
        self._variance_floor = np.finfo(float).eps

    @property
    def selection_fraction(self) -> float:
        return self._selection_fraction

    @property
    def structure_learning(self) -> str:
        return self._structure_learning

    @property
    def edge_exclusion_alpha(self) -> float:
        return self._edge_exclusion_alpha

    @property
    def bge_alpha_mu(self) -> float:
        return self._bge_alpha_mu

    @property
    def bge_alpha_w(self) -> float | None:
        return self._bge_alpha_w

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
                name: float(self._rng.uniform(lower, upper))
                for name, (lower, upper) in self._boundaries.items()
            }
        else:
            variables = deepcopy(variables)

        self._population[identifier] = EGNAParticle(
            identifier=identifier, variables=variables, fitness=fitness
        )

    def delete_particle(self, identifier: str | None) -> None:
        if identifier is None:
            raise ValueError("Particle identifier cannot be None.")

        del self._population[identifier]

    def _initialize_model_context(self) -> None:
        self._variable_names = tuple(self._boundaries)

        if not self._variable_names:
            raise ValueError("EGNA requires at least one optimization variable.")

        bounds = np.asarray(
            [self._boundaries[name] for name in self._variable_names], dtype=float
        )

        if not np.all(np.isfinite(bounds)):
            raise ValueError("EGNA requires finite encoded variable boundaries.")

        lower_bounds = bounds[:, 0]
        upper_bounds = bounds[:, 1]
        spans = upper_bounds - lower_bounds

        if np.any(spans <= 0):
            raise ValueError(
                "EGNA requires every encoded variable to have a positive span."
            )

        if self._bge_alpha_w is not None:
            minimum_alpha_w = len(self._variable_names) + 1

            if self._bge_alpha_w <= minimum_alpha_w:
                raise ValueError(
                    "bge_alpha_w must be greater than the number of encoded "
                    "variables plus 1."
                )

        self._lower_bounds = lower_bounds
        self._spans = spans
        self._parents = ()
        self._topological_order = ()
        self._conditionals = ()

    @staticmethod
    def _fitness_sort_key(fitness: np.float64) -> tuple[bool, float]:
        value = float(fitness)
        return (np.isnan(value), value)

    def _rank_particles(self, particle_ids: Collection[str]) -> list[str]:
        identifiers = list(particle_ids)

        if not identifiers:
            return []

        permutation = self._rng.permutation(len(identifiers))
        randomized = [identifiers[int(index)] for index in permutation]

        return sorted(
            randomized,
            key=lambda identifier: self._fitness_sort_key(
                self._population[identifier].fitness
            ),
        )

    def _variables_to_normalized_array(self, variables: dict[str, float]) -> np.ndarray:
        if self._lower_bounds is None or self._spans is None:
            raise RuntimeError("EGNA model context has not been initialized.")

        values = np.asarray(
            [variables[name] for name in self._variable_names], dtype=float
        )
        normalized = (values - self._lower_bounds) / self._spans
        return np.clip(normalized, 0.0, 1.0)

    def _normalized_array_to_variables(self, values: np.ndarray) -> dict[str, float]:
        if self._lower_bounds is None or self._spans is None:
            raise RuntimeError("EGNA model context has not been initialized.")

        bounded = np.clip(values, 0.0, 1.0)
        encoded = self._lower_bounds + bounded * self._spans

        return {
            name: float(value)
            for name, value in zip(self._variable_names, encoded, strict=True)
        }

    def _selected_matrix(self) -> np.ndarray:
        population_ids = list(self._population)
        population_size = len(population_ids)

        if population_size == 0:
            raise ValueError("EGNA requires at least one particle.")

        selected_count = int(np.floor(self._selection_fraction * population_size))
        n_variables = len(self._variable_names)

        if selected_count <= n_variables:
            raise ValueError(
                "EGNA requires the selected population to contain more solutions "
                "than encoded variables. Increase the population size or "
                "selection_fraction."
            )

        selected_ids = self._rank_particles(population_ids)[:selected_count]

        return np.asarray(
            [
                self._variables_to_normalized_array(
                    self._population[identifier].variables
                )
                for identifier in selected_ids
            ],
            dtype=float,
        )

    @staticmethod
    def _adjacency_from_parents(
        parents: list[set[int]] | tuple[tuple[int, ...], ...],
    ) -> np.ndarray:
        n_variables = len(parents)
        adjacency = np.zeros((n_variables, n_variables), dtype=np.uint8)

        for child, parent_set in enumerate(parents):
            for parent in parent_set:
                adjacency[parent, child] = 1

        return adjacency

    @staticmethod
    def _topological_sort(parents: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
        n_variables = len(parents)
        children: list[set[int]] = [set() for _ in range(n_variables)]
        indegree = [len(parent_set) for parent_set in parents]

        for child, parent_set in enumerate(parents):
            for parent in parent_set:
                children[parent].add(child)

        available = {index for index, degree in enumerate(indegree) if degree == 0}
        order: list[int] = []

        while available:
            node = min(available)
            available.remove(node)
            order.append(node)

            for child in sorted(children[node]):
                indegree[child] -= 1

                if indegree[child] == 0:
                    available.add(child)

        if len(order) != n_variables:
            raise RuntimeError("The learned EGNA Gaussian network contains a cycle.")

        return tuple(order)

    def _local_bic_score(
        self, data: np.ndarray, child: int, parents: tuple[int, ...]
    ) -> float:
        y = data[:, child]
        n_samples = len(y)

        if parents:
            parent_values = data[:, parents]
            design = np.column_stack((np.ones(n_samples), parent_values))

            if np.linalg.matrix_rank(design) < design.shape[1]:
                return -np.inf

            coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
            residuals = y - design @ coefficients
        else:
            residuals = y - float(np.mean(y))

        variance = max(float(np.mean(np.square(residuals))), self._variance_floor)
        log_likelihood = -0.5 * n_samples * (np.log(2.0 * np.pi * variance) + 1.0)
        n_parameters = len(parents) + 2
        penalty = 0.5 * n_parameters * np.log(n_samples)
        return float(log_likelihood - penalty)

    def _resolve_bge_alpha_w(self) -> float:
        dimension = len(self._variable_names)

        if self._bge_alpha_w is None:
            return float(dimension + 2)

        return self._bge_alpha_w

    def _prepare_bge_context(self, data: np.ndarray) -> _BGeContext:
        n_samples, dimension = data.shape
        alpha_mu = self._bge_alpha_mu
        alpha_w = self._resolve_bge_alpha_w()
        prior_scale_value = alpha_mu * (alpha_w - dimension - 1.0) / (alpha_mu + 1.0)

        if prior_scale_value <= 0 or not np.isfinite(prior_scale_value):
            raise ValueError(
                "The configured BGe hyperparameters do not define a positive "
                "normal-Wishart prior scale."
            )

        prior_scale = np.eye(dimension, dtype=float) * prior_scale_value
        sample_mean = np.mean(data, axis=0)
        centered = data - sample_mean
        scatter = centered.T @ centered
        prior_mean = np.full(dimension, 0.5, dtype=float)
        mean_difference = sample_mean - prior_mean
        posterior_scale = (
            prior_scale
            + scatter
            + (n_samples * alpha_mu / (n_samples + alpha_mu))
            * np.outer(mean_difference, mean_difference)
        )

        return _BGeContext(
            sample_count=n_samples,
            dimension=dimension,
            alpha_mu=alpha_mu,
            alpha_w=alpha_w,
            prior_scale=prior_scale,
            posterior_scale=posterior_scale,
        )

    @staticmethod
    def _submatrix_logdet(matrix: np.ndarray, indices: tuple[int, ...]) -> float:
        if not indices:
            return 0.0

        submatrix = matrix[np.ix_(indices, indices)]
        sign, logdet = np.linalg.slogdet(submatrix)

        if sign <= 0 or not np.isfinite(logdet):
            raise RuntimeError("BGe scale submatrix is not positive definite.")

        return float(logdet)

    def _local_bge_score(
        self, context: _BGeContext, child: int, parents: tuple[int, ...]
    ) -> float:
        parent_indices = tuple(sorted(parents))
        family_indices = tuple(sorted((*parent_indices, child)))
        n_samples = context.sample_count
        dimension = context.dimension
        parent_count = len(parent_indices)
        alpha_mu = context.alpha_mu
        alpha_w = context.alpha_w
        base = alpha_w - dimension + parent_count

        logdet_t_family = self._submatrix_logdet(context.prior_scale, family_indices)
        logdet_r_family = self._submatrix_logdet(
            context.posterior_scale, family_indices
        )
        logdet_t_parents = self._submatrix_logdet(context.prior_scale, parent_indices)
        logdet_r_parents = self._submatrix_logdet(
            context.posterior_scale, parent_indices
        )

        score = 0.5 * np.log(alpha_mu / (n_samples + alpha_mu))
        score += gammaln((n_samples + base + 1.0) / 2.0)
        score -= gammaln((base + 1.0) / 2.0)
        score -= 0.5 * n_samples * np.log(np.pi)
        score += 0.5 * (base + 1.0) * logdet_t_family
        score -= 0.5 * base * logdet_t_parents
        score += 0.5 * (n_samples + base) * logdet_r_parents
        score -= 0.5 * (n_samples + base + 1.0) * logdet_r_family
        return float(score)

    def _score_search_structure(
        self, data: np.ndarray, metric: str
    ) -> tuple[tuple[int, ...], ...]:
        n_variables = data.shape[1]
        use_previous_structure = len(self._parents) == n_variables

        if use_previous_structure:
            parents = [set(parent_set) for parent_set in self._parents]
        else:
            parents = [set() for _ in range(n_variables)]

        bge_context = self._prepare_bge_context(data) if metric == "bge" else None
        score_cache: dict[tuple[int, tuple[int, ...]], float] = {}

        def score(child: int, parent_set: set[int]) -> float:
            parent_tuple = tuple(sorted(parent_set))
            key = (child, parent_tuple)

            if key in score_cache:
                return score_cache[key]

            if metric == "bic":
                value = self._local_bic_score(
                    data=data, child=child, parents=parent_tuple
                )
            else:
                if bge_context is None:
                    raise RuntimeError("BGe score context has not been initialized.")

                value = self._local_bge_score(
                    context=bge_context, child=child, parents=parent_tuple
                )

            score_cache[key] = value
            return value

        current_scores = [score(child, parents[child]) for child in range(n_variables)]
        allow_deletions = use_previous_structure

        while True:
            adjacency = self._adjacency_from_parents(parents)
            reachability = np.isfinite(
                shortest_path(adjacency, directed=True, unweighted=True, method="D")
            )
            operations: list[tuple[str, int, int]] = []

            for parent in range(n_variables):
                for child in range(n_variables):
                    if parent == child:
                        continue

                    if parent in parents[child]:
                        if allow_deletions:
                            operations.append(("delete", parent, child))
                        continue

                    if reachability[child, parent]:
                        continue

                    operations.append(("add", parent, child))

            if not operations:
                break

            operation_order = self._rng.permutation(len(operations))
            best_operation: tuple[str, int, int] | None = None
            best_score = 0.0
            best_delta = 0.0

            for operation_index in operation_order:
                action, parent, child = operations[int(operation_index)]
                candidate_parents = set(parents[child])

                if action == "add":
                    candidate_parents.add(parent)
                else:
                    candidate_parents.remove(parent)

                candidate_score = score(child, candidate_parents)
                current_score = current_scores[child]
                delta = candidate_score - current_score

                if np.isnan(delta):
                    continue

                finite_values = [
                    abs(value)
                    for value in (candidate_score, current_score)
                    if np.isfinite(value)
                ]
                scale = max((1.0, *finite_values))
                tolerance = np.finfo(float).eps * scale * 100.0

                if delta <= tolerance:
                    continue

                if best_operation is None or delta > best_delta + tolerance:
                    best_operation = (action, parent, child)
                    best_score = candidate_score
                    best_delta = delta

            if best_operation is None:
                break

            action, parent, child = best_operation

            if action == "add":
                parents[child].add(parent)
            else:
                parents[child].remove(parent)

            current_scores[child] = best_score

        return tuple(tuple(sorted(parent_set)) for parent_set in parents)

    def _regularized_precision(self, data: np.ndarray) -> np.ndarray:
        centered = data - np.mean(data, axis=0)
        covariance = centered.T @ centered / len(data)
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues = np.linalg.eigvalsh(covariance)
        scale = max(1.0, float(np.max(np.abs(np.diag(covariance)))))
        minimum_eigenvalue = np.finfo(float).eps * len(self._variable_names) * scale
        smallest = float(np.min(eigenvalues))

        if smallest < minimum_eigenvalue:
            covariance = covariance + np.eye(covariance.shape[0]) * (
                minimum_eigenvalue - smallest
            )

        try:
            return np.linalg.inv(covariance)
        except np.linalg.LinAlgError as error:
            raise RuntimeError(
                "Unable to invert the EGNA covariance matrix for edge-exclusion "
                "structure learning."
            ) from error

    def _edge_exclusion_structure(
        self, data: np.ndarray
    ) -> tuple[tuple[int, ...], ...]:
        n_samples, n_variables = data.shape
        precision = self._regularized_precision(data)
        parents: list[set[int]] = [set() for _ in range(n_variables)]

        for child in range(n_variables):
            for parent in range(child):
                denominator = float(
                    np.sqrt(precision[parent, parent] * precision[child, child])
                )

                if not np.isfinite(denominator) or denominator <= 0:
                    raise RuntimeError(
                        "Invalid precision-matrix diagonal encountered during "
                        "edge-exclusion structure learning."
                    )

                partial_correlation = float(-precision[parent, child] / denominator)
                partial_correlation = float(
                    np.clip(
                        partial_correlation,
                        -1.0 + np.finfo(float).eps,
                        1.0 - np.finfo(float).eps,
                    )
                )
                statistic = -n_samples * np.log1p(
                    -(partial_correlation * partial_correlation)
                )
                cdf = float(chi2.cdf(statistic, df=1))
                correction = (
                    0.5
                    * (2 * n_variables + 1)
                    * statistic
                    * float(chi2.pdf(statistic, df=1))
                    / n_samples
                )
                corrected_cdf = float(np.clip(cdf - correction, 0.0, 1.0))
                p_value = 1.0 - corrected_cdf

                if p_value < self._edge_exclusion_alpha:
                    parents[child].add(parent)

        return tuple(tuple(sorted(parent_set)) for parent_set in parents)

    def _learn_structure(self, data: np.ndarray) -> tuple[tuple[int, ...], ...]:
        if self._structure_learning == "edge-exclusion":
            return self._edge_exclusion_structure(data)

        return self._score_search_structure(data=data, metric=self._structure_learning)

    def _fit_conditionals(
        self, data: np.ndarray, parents: tuple[tuple[int, ...], ...]
    ) -> tuple[_ConditionalNormal, ...]:
        means = np.mean(data, axis=0)
        conditionals: list[_ConditionalNormal] = []

        for variable, parent_indices in enumerate(parents):
            y = data[:, variable]
            mean = float(means[variable])

            if parent_indices:
                parent_values = data[:, parent_indices]
                parent_means = means[np.asarray(parent_indices, dtype=int)]
                centered_parents = parent_values - parent_means
                centered_y = y - mean
                coefficients, _, _, _ = np.linalg.lstsq(
                    centered_parents, centered_y, rcond=None
                )
                residuals = centered_y - centered_parents @ coefficients
            else:
                parent_means = np.empty(0, dtype=float)
                coefficients = np.empty(0, dtype=float)
                residuals = y - mean

            variance = max(float(np.mean(np.square(residuals))), 0.0)
            standard_deviation = float(np.sqrt(variance))

            if standard_deviation <= np.sqrt(self._variance_floor):
                standard_deviation = 0.0

            conditionals.append(
                _ConditionalNormal(
                    variable_index=variable,
                    parent_indices=parent_indices,
                    mean=mean,
                    parent_means=np.asarray(parent_means, dtype=float),
                    coefficients=np.asarray(coefficients, dtype=float),
                    standard_deviation=standard_deviation,
                )
            )

        return tuple(conditionals)

    def _learn_probability_model(self, selected_data: np.ndarray) -> None:
        parents = self._learn_structure(selected_data)
        topological_order = self._topological_sort(parents)
        conditionals = self._fit_conditionals(data=selected_data, parents=parents)

        self._parents = parents
        self._topological_order = topological_order
        self._conditionals = conditionals

    def pre_iteration(self, actual_iter: int) -> None:
        if actual_iter == 0:
            self._initialize_model_context()
            self._active_particle_ids = set()
            return

        if not self._variable_names:
            raise RuntimeError("EGNA model context has not been initialized.")

        if not self._population:
            raise ValueError("EGNA requires at least one particle.")

        selected_data = self._selected_matrix()
        self._learn_probability_model(selected_data)

        self._active_particle_ids = set(self._population)

    def create_random_cache(self, particle_ids: list[str], initialize: bool) -> None:
        for identifier in particle_ids:
            particle = self._population[identifier]
            particle.random_cache = {}

            if initialize or identifier not in self._active_particle_ids:
                continue

            particle.random_cache = {
                "variable-uniforms": self._rng.random(len(self._variable_names))
            }

    def initialize_particle(self, identifier: str) -> EGNAParticle:
        particle = self._population[identifier]

        if not isinstance(particle, EGNAParticle):
            raise TypeError(
                f"Particle '{identifier}' must be an instance of EGNAParticle."
            )

        if particle.fitness == FITNESS_UNDEFINED:
            particle.update(
                variables=particle.variables, fitness_function=self._fitness_function
            )

        return particle

    @staticmethod
    def consolidate_new_particles(particle: EGNAParticle) -> EGNAParticle:
        particle.consolidate(consolidate_new=True)
        return particle

    @staticmethod
    def _sample_truncated_normal(
        mean: float, standard_deviation: float, random_value: float
    ) -> float:
        if standard_deviation == 0.0:
            return float(np.clip(mean, 0.0, 1.0))

        epsilon = np.finfo(float).eps
        probability = float(np.clip(random_value, epsilon, 1.0 - epsilon))
        lower = (0.0 - mean) / standard_deviation
        upper = (1.0 - mean) / standard_deviation
        value = float(
            truncnorm.ppf(probability, lower, upper, loc=mean, scale=standard_deviation)
        )

        if not np.isfinite(value):
            raise RuntimeError(
                "EGNA truncated Gaussian sampling produced a non-finite value."
            )

        return float(np.clip(value, 0.0, 1.0))

    def _sample_candidate(self, particle: EGNAParticle) -> dict[str, float]:
        if not self._conditionals or not self._topological_order:
            raise RuntimeError("EGNA probability model has not been initialized.")

        random_values = np.asarray(
            particle.random_cache["variable-uniforms"], dtype=float
        )

        if random_values.shape != (len(self._variable_names),):
            raise RuntimeError("EGNA random cache has an invalid sampling shape.")

        candidate = np.full(len(self._variable_names), np.nan, dtype=float)

        for variable in self._topological_order:
            conditional = self._conditionals[variable]

            if conditional.parent_indices:
                parent_values = candidate[
                    np.asarray(conditional.parent_indices, dtype=int)
                ]

                if np.any(np.isnan(parent_values)):
                    raise RuntimeError(
                        "EGNA attempted to sample a variable before one of its parents."
                    )

                conditional_mean = float(
                    conditional.mean
                    + conditional.coefficients
                    @ (parent_values - conditional.parent_means)
                )
            else:
                conditional_mean = conditional.mean

            candidate[variable] = self._sample_truncated_normal(
                mean=conditional_mean,
                standard_deviation=conditional.standard_deviation,
                random_value=float(random_values[variable]),
            )

        if np.any(np.isnan(candidate)):
            raise RuntimeError("EGNA produced an incomplete sampled candidate.")

        return self._normalized_array_to_variables(candidate)

    def update_particle(self, identifier: str) -> EGNAParticle:
        particle = self._population[identifier]

        if not isinstance(particle, EGNAParticle):
            raise TypeError(
                f"Particle '{identifier}' must be an instance of EGNAParticle."
            )

        if identifier not in self._active_particle_ids:
            return particle

        candidate_variables = self._sample_candidate(particle)
        particle.update(
            variables=candidate_variables, fitness_function=self._fitness_function
        )
        return particle

    def post_iteration(self, actual_iter: int) -> None:
        if actual_iter == 0:
            self.update_solution_state()
            return

        for identifier in self._active_particle_ids:
            particle = self._population[identifier]

            if (
                particle.candidate_variables is None
                or particle.candidate_fitness is None
            ):
                raise RuntimeError(
                    f"EGNA particle '{identifier}' has no evaluated candidate."
                )

            particle.consolidate(consolidate_new=True)

        self._active_particle_ids = set()
        self.update_solution_state()
