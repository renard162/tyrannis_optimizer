from collections.abc import Collection
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from scipy.cluster.vq import ClusterError, kmeans2, whiten
from scipy.linalg import solve
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.stats import multivariate_normal, norm, truncnorm

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
class _ClusterModel:
    weight: float
    conditionals: tuple[_ConditionalNormal, ...]


@dataclass(frozen=True)
class _SubproblemModel:
    variable_indices: tuple[int, ...]
    clusters: tuple[_ClusterModel, ...]
    cumulative_weights: np.ndarray


@dataclass(frozen=True)
class _NetworkModel:
    weight: float
    parents: tuple[tuple[int, ...], ...]
    topological_order: tuple[int, ...]
    effective_max_parents: int
    subproblem_models: tuple[_SubproblemModel, ...]


class RBOAParticle(ParticleBase):
    """Particle implementation for the Improved Real-Coded Bayesian
    Optimization Algorithm.

    """


class RBOA(AlgorithmBase[RBOAParticle]):
    """Improved Real-Coded Bayesian Optimization Algorithm (IrBOA)."""

    def __init__(
        self,
        selection_fraction: float = 0.5,
        offspring_size: int | None = None,
        bic_penalty: float = 0.5,
        model_selection_clusters: int = 1,
        adaptive_initial_clusters: int | None = None,
        adaptive_clustering_penalty: float = 0.3,
        clustering_threshold: float = 0.3,
        max_parents: int | None = None,
        covariance_floor: float = 1e-12,
    ) -> None:
        """
        Improved Real-Coded Bayesian Optimization Algorithm (IrBOA).

        This implementation keeps the public ``RBOA`` class name while implementing
        the improved real-coded Bayesian optimization algorithm (IrBOA). IrBOA
        extends rBOA by adaptively partitioning the selected population and learning
        one Bayesian network for each retained region. This preserves distinct
        dependency structures while diversity is high and allows the number of
        regional models to decrease automatically as the population converges.

        Inside every regional Bayesian network, the original rBOA probabilistic
        building-block model is retained. Maximal connected components define
        disjoint subproblems. Each subproblem is independently clustered by BEND and
        represented by a mixture of local Gaussian distributions. Offspring
        generation first selects one regional Bayesian network and then performs
        probabilistic building-block crossover using that network's own subproblems.

        Parameters
        ----------
        selection_fraction : float, default=0.5
            Fraction of the current persistent population selected by truncation
            selection to learn the probabilistic models. The selected count is
            ``floor(selection_fraction * population_size)``. At least two selected
            solutions are required.

        offspring_size : int or None, default=None
            Number of offspring sampled from the learned models at each iteration.
            When ``None``, the value is resolved dynamically as
            ``population_size - floor(selection_fraction * population_size)``.
            Values greater than the persistent population size are supported by
            creating temporary particles that are initialized directly as IrBOA
            offspring and removed after survivor selection.

        bic_penalty : float, default=0.5
            Non-negative coefficient multiplying the BIC model-complexity term.
            Structure search inside each adaptive region minimizes the rBOA
            mixture-BIC form: component-weighted conditional Gaussian entropies plus
            ``K * bic_penalty * log(n_region) * n_parameters``.

        model_selection_clusters : int, default=1
            Number ``K`` of fixed K-means mixture components used only while learning
            the structure of each regional Bayesian network. The components inside
            one region share that region's graph and contribute jointly to its
            mixture BIC. ``1`` retains the simplified structure-learning setting used
            in the original GECCO 2004 rBOA experiments. Every retained adaptive
            region must contain at least ``K`` selected observations and ``K``
            distinct variance-scaled observations. This parameter is distinct from
            ``adaptive_initial_clusters``.

        adaptive_initial_clusters : int or None, default=None
            Initial upper bound for the adaptive k-prime-means partition that creates
            the set of Bayesian networks. ``None`` resolves to the number of encoded
            decision variables, following the IrBOA configuration that begins from
            an overestimate and removes unnecessary clusters. For the automatic
            setting, the initial count is bounded by the numbers of selected and
            distinct solutions because every initial K-means cluster must contain at
            least one sample.

        adaptive_clustering_penalty : float, default=0.3
            Strictly positive penalty ``E`` in the k-prime-means discrepancy used by
            IrBOA, ``||x - c_j||^2 - E * log2(p_j)``, where ``p_j`` is the
            current relative size of adaptive cluster ``j``. The literature value
            ``0.3`` is used by default. Smaller clusters receive a larger penalty and
            can disappear during adaptive reassignment.

        clustering_threshold : float, default=0.3
            BEND leader clustering threshold in the normalized distance interval
            ``[0, 1]``. A projected selected solution joins its nearest leader when
            its BEND distance is strictly smaller than this threshold; otherwise a
            new cluster is created. BEND acts inside one Bayesian-network subproblem
            and is distinct from the top-level adaptive clustering.

        max_parents : int or None, default=None
            Maximum number of parents allowed for one node in each learned Bayesian
            network. ``None`` requests the unrestricted ``n_variables - 1`` setting
            used in rBOA, subject to the observations available for identifiable
            local Gaussian regressions in the smallest fixed model-selection mixture
            component of each adaptive region. An explicitly supplied limit that is
            not identifiable raises an exception.

        covariance_floor : float, default=1e-12
            Strictly positive lower eigenvalue used when regularizing local cluster
            covariance matrices and conditional Gaussian variances. Modeling and
            adaptive clustering are performed in an internal affine normalization of
            every encoded variable to ``[0, 1]``, so this value is dimensionless.

        Notes
        -----
        Let ``S`` denote the selected population. IrBOA first partitions ``S`` into
        adaptive regions ``A_j``. If ``p_j = |A_j| / |S|`` and ``c_j`` is the
        centroid of adaptive region ``j``, reassignment uses the k-prime-means
        discrepancy

        ``d_E(x, c_j) = ||x - c_j||^2 - E * log2(p_j)``.

        The squared-distance term favors geometrically nearby regions, while the
        frequency term penalizes low-occupancy regions. Regions that lose every
        solution are removed, so the retained cluster count is inferred from the
        selected population rather than held fixed. Top-level adaptive clustering
        operates on coordinates normalized to the fixed encoded boundaries.
        Therefore, when population diversity decreases, the squared-distance term
        contracts instead of being restored by per-generation scaling, making the
        occupancy penalty relatively more influential. This is the mechanism by
        which IrBOA can automatically reduce the number of regional models during
        convergence.

        Each retained region learns its own Gaussian Bayesian network, with distinct
        structure and parameters. For network ``j``,

        ``p_j(X) = product_i p_j(X_i | Pa_j(X_i))``,

        and the probability of choosing that network during offspring generation is

        ``B_j = |A_j| / sum_h |A_h|``.

        The top-level adaptive regions are not the mixture components controlled by
        ``model_selection_clusters``. The latter are an rBOA structure-learning
        device local to one adaptive region: when ``K > 1``, that region is
        partitioned into ``K`` fixed K-means components whose conditional Gaussian
        contributions are combined in the mixture BIC, while all ``K`` components
        still share one regional Bayesian-network structure. With the default
        ``K = 1``, structure learning uses one component per adaptive region.

        Structure learning starts from an empty graph and proceeds by greedy
        incremental edge addition. At each step, the admissible directed edge that
        gives the largest strict improvement in mixture BIC is added. Edges that
        create a cycle or violate the effective parent limit are excluded. The search
        terminates when no admissible edge improves the score.

        After a regional Bayesian network is learned, edge directions are ignored
        only to obtain its maximal connected components. These connected components
        are disjoint by construction and define that network's probabilistic building
        blocks, or subproblems. Dependencies that share a variable therefore belong
        to the same connected component; they are not represented as overlapping
        building blocks.

        Each subproblem is then clustered independently using BEND leader clustering.
        If ``C_ijk`` denotes BEND cluster ``k`` of subproblem ``Z_ij`` in regional
        network ``j``, its local mixture weight is

        ``beta_ijk = |C_ijk| / sum_l |C_ijl|``.

        A multivariate Gaussian distribution is estimated by maximum likelihood for
        every BEND cluster. Within each Gaussian component, the corresponding
        regional Bayesian-network structure factorizes the joint distribution into
        univariate conditional Gaussian distributions sampled in topological order.
        BEND therefore models local nonlinearity inside one disjoint subproblem. It
        must not be confused with the top-level adaptive clustering, whose clusters
        are alternative full Bayesian-network models over the same complete decision
        vector.

        Offspring generation first chooses a regional network according to ``B_j``.
        Probabilistic building-block crossover is then performed within that network:
        one BEND Gaussian component is selected independently for each disjoint
        subproblem, the corresponding variables are sampled, and the sampled
        subproblems are combined into a complete candidate. Multiple regional
        networks are therefore alternative dependency models, not overlapping
        building blocks.

        Let ``n`` denote the persistent population size and ``q`` the number of
        offspring generated in one iteration. Under partial renewal with ``q < n``,
        the best ``n - q`` parents are retained and the ``q`` newly generated
        offspring complete the next population. When ``q >= n``, the next population
        is selected from the generated offspring so that its size remains ``n``.

        References
        ----------
        Ahn, C. W., Ramakrishna, R. S., & Goldberg, D. E. (2004). Real-Coded
        Bayesian Optimization Algorithm: Bringing the Strength of BOA into the
        Continuous World. Genetic and Evolutionary Computation - GECCO 2004,
        840-851. https://doi.org/10.1007/978-3-540-24854-5_86

        Moradabadi, B., Beigy, H., & Ahn, C. W. (2013). An Improved Real-Coded
        Bayesian Optimization Algorithm for Continuous Global Optimization.
        International Journal of Innovative Computing, Information and Control,
        9(6), 2505-2519.
        https://www.researchgate.net/publication/286944613_An_improved_real-coded_bayesian_optimization_algorithm_for_continuous_global_optimization

        Ahn, C. W., Ramakrishna, R. S., & Goldberg, D. E. (2006). Real-Coded
        Bayesian Optimization Algorithm. Towards a New Evolutionary Computation:
        Advances on Estimation of Distribution Algorithms, 51-73.
        https://doi.org/10.1007/11543138_5

        Žalik, K. R. (2008). An Efficient k'-Means Clustering Algorithm. Pattern
        Recognition Letters, 29(9), 1385-1391.
        https://doi.org/10.1016/j.patrec.2008.02.014

        Ahn, C. W., & Ramakrishna, R. S. (2008). On the Scalability of Real-Coded
        Bayesian Optimization Algorithm. IEEE Transactions on Evolutionary
        Computation, 12(3), 307-322.
        https://doi.org/10.1109/TEVC.2007.902856
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

        if offspring_size is not None:
            if isinstance(offspring_size, (bool, np.bool_)) or not isinstance(
                offspring_size, (int, np.integer)
            ):
                raise TypeError("offspring_size must be an integer or None.")

            if offspring_size < 1:
                raise ValueError("offspring_size must be greater than or equal to 1.")

        if not isinstance(bic_penalty, (int, float, np.number)) or isinstance(
            bic_penalty, (bool, np.bool_)
        ):
            raise TypeError("bic_penalty must be a number.")

        if not np.isfinite(bic_penalty) or bic_penalty < 0:
            raise ValueError(
                "bic_penalty must be a finite number greater than or equal to 0."
            )

        if isinstance(model_selection_clusters, (bool, np.bool_)) or not isinstance(
            model_selection_clusters, (int, np.integer)
        ):
            raise TypeError("model_selection_clusters must be an integer.")

        if model_selection_clusters < 1:
            raise ValueError(
                "model_selection_clusters must be greater than or equal to 1."
            )

        if adaptive_initial_clusters is not None:
            if isinstance(
                adaptive_initial_clusters, (bool, np.bool_)
            ) or not isinstance(adaptive_initial_clusters, (int, np.integer)):
                raise TypeError("adaptive_initial_clusters must be an integer or None.")

            if adaptive_initial_clusters < 1:
                raise ValueError(
                    "adaptive_initial_clusters must be greater than or equal to 1."
                )

        if not isinstance(
            adaptive_clustering_penalty, (int, float, np.number)
        ) or isinstance(adaptive_clustering_penalty, (bool, np.bool_)):
            raise TypeError("adaptive_clustering_penalty must be a number.")

        if (
            not np.isfinite(adaptive_clustering_penalty)
            or adaptive_clustering_penalty <= 0
        ):
            raise ValueError(
                "adaptive_clustering_penalty must be a finite number greater than 0."
            )

        if not isinstance(clustering_threshold, (int, float, np.number)) or isinstance(
            clustering_threshold, (bool, np.bool_)
        ):
            raise TypeError("clustering_threshold must be a number.")

        if not np.isfinite(clustering_threshold) or not 0 <= clustering_threshold <= 1:
            raise ValueError(
                "clustering_threshold must be finite and between 0 and 1 inclusive."
            )

        if max_parents is not None:
            if isinstance(max_parents, (bool, np.bool_)) or not isinstance(
                max_parents, (int, np.integer)
            ):
                raise TypeError("max_parents must be an integer or None.")

            if max_parents < 0:
                raise ValueError("max_parents must be greater than or equal to 0.")

        if not isinstance(covariance_floor, (int, float, np.number)) or isinstance(
            covariance_floor, (bool, np.bool_)
        ):
            raise TypeError("covariance_floor must be a number.")

        if not np.isfinite(covariance_floor) or covariance_floor <= 0:
            raise ValueError("covariance_floor must be a finite number greater than 0.")

        self._selection_fraction = float(selection_fraction)
        self._offspring_size = None if offspring_size is None else int(offspring_size)
        self._bic_penalty = float(bic_penalty)
        self._model_selection_clusters = int(model_selection_clusters)
        self._adaptive_initial_clusters = (
            None
            if adaptive_initial_clusters is None
            else int(adaptive_initial_clusters)
        )
        self._adaptive_clustering_penalty = float(adaptive_clustering_penalty)
        self._clustering_threshold = float(clustering_threshold)
        self._max_parents = None if max_parents is None else int(max_parents)
        self._covariance_floor = float(covariance_floor)

        self._variable_names: tuple[str, ...] = ()
        self._lower_bounds: np.ndarray | None = None
        self._spans: np.ndarray | None = None
        self._network_models: tuple[_NetworkModel, ...] = ()
        self._network_cumulative_weights = np.empty(0, dtype=float)
        self._parent_particle_ids: list[str] = []
        self._active_parent_ids: set[str] = set()
        self._temporary_particle_ids: set[str] = set()
        self._temporary_index = 0
        self._iteration_offspring_size = 0

    @property
    def selection_fraction(self) -> float:
        return self._selection_fraction

    @property
    def offspring_size(self) -> int | None:
        return self._offspring_size

    @property
    def bic_penalty(self) -> float:
        return self._bic_penalty

    @property
    def model_selection_clusters(self) -> int:
        return self._model_selection_clusters

    @property
    def adaptive_initial_clusters(self) -> int | None:
        return self._adaptive_initial_clusters

    @property
    def adaptive_clustering_penalty(self) -> float:
        return self._adaptive_clustering_penalty

    @property
    def clustering_threshold(self) -> float:
        return self._clustering_threshold

    @property
    def max_parents(self) -> int | None:
        return self._max_parents

    @property
    def covariance_floor(self) -> float:
        return self._covariance_floor

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

        self._population[identifier] = RBOAParticle(
            identifier=identifier, variables=variables, fitness=fitness
        )

    def delete_particle(self, identifier: str | None) -> None:
        if identifier is None:
            raise ValueError("Particle identifier cannot be None.")

        del self._population[identifier]

    def _initialize_model_context(self) -> None:
        self._variable_names = tuple(self._boundaries)

        if not self._variable_names:
            raise ValueError("RBOA requires at least one optimization variable.")

        bounds = np.asarray(
            [self._boundaries[name] for name in self._variable_names], dtype=float
        )

        if not np.all(np.isfinite(bounds)):
            raise ValueError("RBOA requires finite encoded variable boundaries.")

        lower_bounds = bounds[:, 0]
        upper_bounds = bounds[:, 1]
        spans = upper_bounds - lower_bounds

        if np.any(spans <= 0):
            raise ValueError(
                "RBOA requires every encoded variable to have a positive span."
            )

        if (
            self._max_parents is not None
            and self._max_parents > len(self._variable_names) - 1
        ):
            raise ValueError(
                "max_parents cannot exceed the number of encoded variables minus 1."
            )

        self._lower_bounds = lower_bounds
        self._spans = spans

    def _resolve_effective_max_parents(self, minimum_cluster_size: int) -> int:
        n_variables = len(self._variable_names)
        identifiable = max(minimum_cluster_size - 2, 0)

        if self._max_parents is None:
            return min(n_variables - 1, identifiable)

        if self._max_parents > identifiable:
            raise ValueError(
                "The smallest rBOA model-selection mixture component inside an "
                "IrBOA adaptive region is too small for the configured max_parents. "
                "Each local Gaussian conditional requires at least max_parents + 2 "
                "samples. Increase the population or selection_fraction, reduce "
                "model_selection_clusters or adaptive_initial_clusters, or reduce "
                "max_parents."
            )

        return self._max_parents

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
            raise RuntimeError("RBOA model context has not been initialized.")

        values = np.asarray(
            [variables[name] for name in self._variable_names], dtype=float
        )
        normalized = (values - self._lower_bounds) / self._spans
        return np.clip(normalized, 0.0, 1.0)

    def _normalized_array_to_variables(self, values: np.ndarray) -> dict[str, float]:
        if self._lower_bounds is None or self._spans is None:
            raise RuntimeError("RBOA model context has not been initialized.")

        bounded = np.clip(values, 0.0, 1.0)
        encoded = self._lower_bounds + bounded * self._spans

        return {
            name: float(value)
            for name, value in zip(self._variable_names, encoded, strict=True)
        }

    def _selected_matrix(self) -> np.ndarray:
        population_size = len(self._parent_particle_ids)
        selected_count = int(np.floor(self._selection_fraction * population_size))

        if selected_count < 2:
            raise ValueError(
                "RBOA requires at least two selected solutions. Increase the "
                "population size or selection_fraction."
            )

        selected_ids = self._rank_particles(self._parent_particle_ids)[:selected_count]

        return np.asarray(
            [
                self._variables_to_normalized_array(
                    self._population[identifier].variables
                )
                for identifier in selected_ids
            ],
            dtype=float,
        )

    def _initial_adaptive_cluster_count(self, data: np.ndarray) -> int:
        n_samples = len(data)
        distinct_samples = len(np.unique(data, axis=0))

        if self._adaptive_initial_clusters is None:
            return min(len(self._variable_names), n_samples, distinct_samples)

        if self._adaptive_initial_clusters > n_samples:
            raise ValueError(
                "adaptive_initial_clusters cannot exceed the number of selected "
                "IrBOA solutions."
            )

        if self._adaptive_initial_clusters > distinct_samples:
            raise ValueError(
                "adaptive_initial_clusters cannot exceed the number of distinct "
                "selected IrBOA solutions."
            )

        return self._adaptive_initial_clusters

    def _initial_adaptive_kmeans_labels(
        self, data: np.ndarray, n_clusters: int
    ) -> np.ndarray:
        if n_clusters == 1:
            return np.zeros(len(data), dtype=int)

        labels: np.ndarray | None = None
        last_error: ClusterError | None = None

        for _ in range(10):
            try:
                _, labels = kmeans2(
                    data,
                    n_clusters,
                    iter=100,
                    minit="points",
                    missing="raise",
                    check_finite=False,
                    rng=self._rng,
                )
                break
            except ClusterError as error:
                last_error = error

        if labels is None:
            raise RuntimeError(
                "SciPy K-means repeatedly produced an empty IrBOA initial "
                "adaptive cluster."
            ) from last_error

        return np.asarray(labels, dtype=int)

    def _adaptive_clusters(self, data: np.ndarray) -> tuple[np.ndarray, ...]:
        n_samples = len(data)

        if n_samples == 0:
            raise ValueError(
                "IrBOA adaptive clustering requires at least one selected solution."
            )

        n_clusters = self._initial_adaptive_cluster_count(data)

        if n_clusters <= 1:
            return (np.arange(n_samples, dtype=int),)

        labels = self._initial_adaptive_kmeans_labels(data=data, n_clusters=n_clusters)

        for _ in range(1000):
            active_labels = np.unique(labels)

            if len(active_labels) == 1:
                return (np.arange(n_samples, dtype=int),)

            clusters = [np.flatnonzero(labels == label) for label in active_labels]
            centers = np.asarray(
                [np.mean(data[indices], axis=0) for indices in clusters], dtype=float
            )
            probabilities = np.asarray(
                [len(indices) / n_samples for indices in clusters], dtype=float
            )
            squared_distances = np.sum(
                np.square(data[:, None, :] - centers[None, :, :]), axis=2
            )
            discrepancy = (
                squared_distances
                - self._adaptive_clustering_penalty * (np.log2(probabilities)[None, :])
            )

            cluster_order = self._rng.permutation(len(active_labels))
            ordered_discrepancy = discrepancy[:, cluster_order]
            selected_positions = np.argmin(ordered_discrepancy, axis=1)
            new_labels = cluster_order[selected_positions].astype(int)

            if np.array_equal(new_labels, labels):
                final_labels = np.unique(new_labels)
                return tuple(
                    np.flatnonzero(new_labels == label).astype(int)
                    for label in final_labels
                )

            labels = new_labels

        raise RuntimeError(
            "IrBOA adaptive clustering did not converge within 1000 iterations."
        )

    def _kmeans_clusters(self, data: np.ndarray) -> tuple[np.ndarray, ...]:
        n_samples = len(data)
        n_clusters = self._model_selection_clusters

        if n_clusters > n_samples:
            raise ValueError(
                "model_selection_clusters cannot exceed the number of solutions "
                "in an IrBOA adaptive region."
            )

        if n_clusters == 1:
            return (np.arange(n_samples, dtype=int),)

        standard_deviations = np.std(data, axis=0)
        active_dimensions = standard_deviations > 0
        metric_data = np.zeros_like(data, dtype=float)

        if np.any(active_dimensions):
            metric_data[:, active_dimensions] = whiten(
                data[:, active_dimensions],
                check_finite=False,
            )

        unique_observations = len(np.unique(metric_data, axis=0))

        if unique_observations < n_clusters:
            raise ValueError(
                "model_selection_clusters cannot exceed the number of distinct "
                "observations in an IrBOA adaptive region after variance scaling."
            )

        labels: np.ndarray | None = None
        last_error: ClusterError | None = None

        for _ in range(10):
            try:
                _, labels = kmeans2(
                    metric_data,
                    n_clusters,
                    iter=100,
                    minit="points",
                    missing="raise",
                    check_finite=False,
                    rng=self._rng,
                )
                break
            except ClusterError as error:
                last_error = error

        if labels is None:
            raise RuntimeError(
                "SciPy K-means repeatedly produced an empty rBOA model-selection "
                "cluster inside an IrBOA adaptive region."
            ) from last_error

        return tuple(
            np.flatnonzero(labels == cluster).astype(int)
            for cluster in range(n_clusters)
        )

    def _local_bic_score(
        self,
        data: np.ndarray,
        mixture_clusters: tuple[np.ndarray, ...],
        child: int,
        parents: tuple[int, ...],
    ) -> float:
        weighted_entropy = 0.0

        for indices in mixture_clusters:
            component = data[indices]
            y = component[:, child]
            n_samples = len(y)

            if parents:
                parent_values = component[:, parents]
                design = np.column_stack((np.ones(n_samples), parent_values))
                coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
                residuals = y - design @ coefficients
            else:
                residuals = y - float(np.mean(y))

            variance = max(
                float(np.mean(np.square(residuals))),
                self._covariance_floor,
            )
            conditional_entropy = float(norm.entropy(scale=np.sqrt(variance)))
            weighted_entropy += n_samples * conditional_entropy

        n_parameters = len(parents) + 2
        complexity = (
            len(mixture_clusters) * self._bic_penalty * np.log(len(data)) * n_parameters
        )

        return float(weighted_entropy + complexity)

    def _learn_structure(
        self,
        data: np.ndarray,
        mixture_clusters: tuple[np.ndarray, ...],
        effective_max_parents: int,
    ) -> tuple[tuple[int, ...], ...]:
        n_variables = data.shape[1]
        parents: list[set[int]] = [set() for _ in range(n_variables)]
        adjacency = np.zeros((n_variables, n_variables), dtype=np.uint8)
        score_cache: dict[tuple[int, tuple[int, ...]], float] = {}

        def score(child: int, parent_set: set[int]) -> float:
            parent_tuple = tuple(sorted(parent_set))
            key = (child, parent_tuple)

            if key not in score_cache:
                score_cache[key] = self._local_bic_score(
                    data=data,
                    mixture_clusters=mixture_clusters,
                    child=child,
                    parents=parent_tuple,
                )

            return score_cache[key]

        current_scores = [score(child, set()) for child in range(n_variables)]

        while True:
            reachability = np.isfinite(
                shortest_path(
                    adjacency,
                    directed=True,
                    unweighted=True,
                    method="D",
                )
            )
            best_edge: tuple[int, int] | None = None
            best_new_score = 0.0
            best_delta = 0.0

            for parent in range(n_variables):
                for child in range(n_variables):
                    if parent == child or parent in parents[child]:
                        continue

                    if len(parents[child]) >= effective_max_parents:
                        continue

                    if reachability[child, parent]:
                        continue

                    candidate_parents = set(parents[child])
                    candidate_parents.add(parent)
                    candidate_score = score(child, candidate_parents)
                    delta = candidate_score - current_scores[child]

                    tolerance = (
                        np.finfo(float).eps
                        * max(1.0, abs(candidate_score), abs(current_scores[child]))
                        * 100.0
                    )

                    if delta < best_delta - tolerance:
                        best_delta = delta
                        best_edge = (parent, child)
                        best_new_score = candidate_score

            if best_edge is None:
                break

            parent, child = best_edge
            parents[child].add(parent)
            adjacency[parent, child] = 1
            current_scores[child] = best_new_score

        return tuple(tuple(sorted(parent_set)) for parent_set in parents)

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
            raise RuntimeError("The learned rBOA Bayesian network contains a cycle.")

        return tuple(order)

    @staticmethod
    def _connected_components(
        parents: tuple[tuple[int, ...], ...], topological_order: tuple[int, ...]
    ) -> tuple[tuple[int, ...], ...]:
        n_variables = len(parents)
        adjacency = np.zeros((n_variables, n_variables), dtype=np.uint8)

        for child, parent_set in enumerate(parents):
            for parent in parent_set:
                adjacency[parent, child] = 1

        n_components, labels = connected_components(
            adjacency,
            directed=False,
            return_labels=True,
        )
        topological_position = {
            variable: position for position, variable in enumerate(topological_order)
        }
        components = [
            tuple(
                sorted(
                    np.flatnonzero(labels == component).tolist(),
                    key=lambda variable: topological_position[variable],
                )
            )
            for component in range(n_components)
        ]
        components.sort(
            key=lambda component: min(
                topological_position[variable] for variable in component
            )
        )

        return tuple(components)

    def _bend_clusters(self, data: np.ndarray) -> list[np.ndarray]:
        n_samples, n_dimensions = data.shape

        if n_samples == 0:
            raise ValueError("BEND clustering requires at least one selected solution.")

        minimum = np.min(data, axis=0)
        ranges = np.ptp(data, axis=0)
        active_dimensions = ranges > 0
        scaled = np.zeros_like(data, dtype=float)

        if np.any(active_dimensions):
            scaled[:, active_dimensions] = (
                data[:, active_dimensions] - minimum[active_dimensions]
            ) / ranges[active_dimensions]

        lower_corner = np.zeros(n_dimensions, dtype=float)
        upper_corner = np.zeros(n_dimensions, dtype=float)
        upper_corner[active_dimensions] = 1.0

        lower_distances = np.linalg.norm(scaled - lower_corner, axis=1)
        upper_distances = np.linalg.norm(scaled - upper_corner, axis=1)
        approximate_maximum_distance = float(
            max(np.max(lower_distances), np.max(upper_distances))
        )

        if approximate_maximum_distance <= 0:
            return [np.arange(n_samples, dtype=int)]

        sample_order = self._rng.permutation(n_samples)
        leaders: list[int] = []
        clusters: list[list[int]] = []

        for sample_value in sample_order:
            sample_index = int(sample_value)

            if not leaders:
                leaders.append(sample_index)
                clusters.append([sample_index])
                continue

            cluster_order = self._rng.permutation(len(leaders))
            best_cluster_index: int | None = None
            best_distance = np.inf

            for cluster_value in cluster_order:
                cluster_index = int(cluster_value)
                leader_index = leaders[cluster_index]
                distance = float(
                    np.linalg.norm(scaled[sample_index] - scaled[leader_index])
                    / approximate_maximum_distance
                )

                if distance < best_distance:
                    best_distance = distance
                    best_cluster_index = cluster_index

            if best_cluster_index is None:
                raise RuntimeError("Unable to select a BEND leader cluster.")

            if best_distance < self._clustering_threshold:
                clusters[best_cluster_index].append(sample_index)
            else:
                leaders.append(sample_index)
                clusters.append([sample_index])

        return [np.asarray(cluster, dtype=int) for cluster in clusters]

    def _regularize_covariance(self, covariance: np.ndarray) -> np.ndarray:
        covariance = np.asarray(covariance, dtype=float)
        covariance = 0.5 * (covariance + covariance.T)
        minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(covariance)))

        if minimum_eigenvalue < self._covariance_floor:
            covariance = covariance + np.eye(covariance.shape[0]) * (
                self._covariance_floor - minimum_eigenvalue
            )

        return covariance

    def _fit_cluster_model(
        self,
        cluster_data: np.ndarray,
        component: tuple[int, ...],
        parents: tuple[tuple[int, ...], ...],
    ) -> _ClusterModel:
        mean, covariance = multivariate_normal.fit(cluster_data)
        mean = np.asarray(mean, dtype=float)
        covariance = self._regularize_covariance(
            np.atleast_2d(np.asarray(covariance, dtype=float))
        )
        local_index = {variable: index for index, variable in enumerate(component)}
        conditionals: list[_ConditionalNormal] = []

        for variable in component:
            variable_local_index = local_index[variable]
            parent_indices = tuple(
                parent for parent in parents[variable] if parent in local_index
            )

            if parent_indices:
                parent_local_indices = tuple(
                    local_index[parent] for parent in parent_indices
                )
                covariance_pp = covariance[
                    np.ix_(parent_local_indices, parent_local_indices)
                ]
                covariance_py = covariance[
                    np.asarray(parent_local_indices, dtype=int), variable_local_index
                ]
                coefficients = solve(
                    covariance_pp,
                    covariance_py,
                    assume_a="pos",
                    check_finite=False,
                )
                covariance_yp = covariance[
                    variable_local_index, np.asarray(parent_local_indices, dtype=int)
                ]
                variance = float(
                    covariance[variable_local_index, variable_local_index]
                    - covariance_yp @ coefficients
                )
                parent_means = mean[np.asarray(parent_local_indices, dtype=int)]
            else:
                coefficients = np.empty(0, dtype=float)
                parent_means = np.empty(0, dtype=float)
                variance = float(covariance[variable_local_index, variable_local_index])

            variance = max(variance, self._covariance_floor)

            conditionals.append(
                _ConditionalNormal(
                    variable_index=variable,
                    parent_indices=parent_indices,
                    mean=float(mean[variable_local_index]),
                    parent_means=np.asarray(parent_means, dtype=float),
                    coefficients=np.asarray(coefficients, dtype=float),
                    standard_deviation=float(np.sqrt(variance)),
                )
            )

        return _ClusterModel(
            weight=float(len(cluster_data)), conditionals=tuple(conditionals)
        )

    def _fit_subproblem_models(
        self,
        selected_data: np.ndarray,
        components: tuple[tuple[int, ...], ...],
        parents: tuple[tuple[int, ...], ...],
    ) -> tuple[_SubproblemModel, ...]:
        models: list[_SubproblemModel] = []

        for component in components:
            projected = selected_data[:, component]
            cluster_indices = self._bend_clusters(projected)
            clusters: list[_ClusterModel] = []

            for indices in cluster_indices:
                cluster = self._fit_cluster_model(
                    cluster_data=projected[indices],
                    component=component,
                    parents=parents,
                )
                clusters.append(cluster)

            weights = np.asarray([cluster.weight for cluster in clusters], dtype=float)
            weights /= np.sum(weights)
            cumulative_weights = np.cumsum(weights)
            cumulative_weights[-1] = 1.0

            normalized_clusters = tuple(
                _ClusterModel(weight=float(weight), conditionals=cluster.conditionals)
                for cluster, weight in zip(clusters, weights, strict=True)
            )

            models.append(
                _SubproblemModel(
                    variable_indices=component,
                    clusters=normalized_clusters,
                    cumulative_weights=cumulative_weights,
                )
            )

        return tuple(models)

    def _learn_probability_model(self, selected_data: np.ndarray) -> None:
        adaptive_clusters = self._adaptive_clusters(selected_data)
        network_models: list[_NetworkModel] = []

        for indices in adaptive_clusters:
            region_data = selected_data[indices]
            mixture_clusters = self._kmeans_clusters(region_data)
            minimum_cluster_size = min(
                len(cluster_indices) for cluster_indices in mixture_clusters
            )
            effective_max_parents = self._resolve_effective_max_parents(
                minimum_cluster_size=minimum_cluster_size
            )
            parents = self._learn_structure(
                data=region_data,
                mixture_clusters=mixture_clusters,
                effective_max_parents=effective_max_parents,
            )
            topological_order = self._topological_sort(parents)
            components = self._connected_components(
                parents=parents, topological_order=topological_order
            )
            subproblem_models = self._fit_subproblem_models(
                selected_data=region_data, components=components, parents=parents
            )
            network_models.append(
                _NetworkModel(
                    weight=float(len(region_data)),
                    parents=parents,
                    topological_order=topological_order,
                    effective_max_parents=effective_max_parents,
                    subproblem_models=subproblem_models,
                )
            )

        weights = np.asarray([model.weight for model in network_models], dtype=float)
        weights /= np.sum(weights)
        cumulative_weights = np.cumsum(weights)
        cumulative_weights[-1] = 1.0

        self._network_models = tuple(
            _NetworkModel(
                weight=float(weight),
                parents=model.parents,
                topological_order=model.topological_order,
                effective_max_parents=model.effective_max_parents,
                subproblem_models=model.subproblem_models,
            )
            for model, weight in zip(network_models, weights, strict=True)
        )
        self._network_cumulative_weights = cumulative_weights

    def _create_temporary_identifier(self) -> str:
        while True:
            identifier = f"{self._identifier}|rboa-offspring:{self._temporary_index}"
            self._temporary_index += 1

            if identifier not in self._population:
                return identifier

    def pre_iteration(self, actual_iter: int) -> None:
        if actual_iter == 0:
            self._initialize_model_context()
            self._parent_particle_ids = list(self._population)
            self._active_parent_ids = set()
            self._temporary_particle_ids = set()
            self._iteration_offspring_size = 0
            return

        if not self._variable_names:
            raise RuntimeError("RBOA model context has not been initialized.")

        self._parent_particle_ids = list(self._population)

        if not self._parent_particle_ids:
            raise ValueError("RBOA requires at least one persistent particle.")

        selected_data = self._selected_matrix()
        self._learn_probability_model(selected_data)

        population_size = len(self._parent_particle_ids)
        offspring_size = (
            population_size - len(selected_data)
            if self._offspring_size is None
            else self._offspring_size
        )
        self._iteration_offspring_size = offspring_size
        active_count = min(offspring_size, population_size)
        self._active_parent_ids = set(self._parent_particle_ids[:active_count])
        self._temporary_particle_ids = set()

        temporary_count = max(0, offspring_size - population_size)

        if temporary_count == 0:
            return

        reference_variables = deepcopy(
            self._population[self._parent_particle_ids[0]].variables
        )

        for _ in range(temporary_count):
            identifier = self._create_temporary_identifier()
            self._temporary_particle_ids.add(identifier)
            self.create_particle(
                identifier=identifier, variables=deepcopy(reference_variables)
            )

    def _cache_offspring_random_values(self, particle: RBOAParticle) -> None:
        cache = {"network-uniform": float(self._rng.random())}

        for network_index, network in enumerate(self._network_models):
            for subproblem_index in range(len(network.subproblem_models)):
                cache[
                    f"network-{network_index}-subproblem-{subproblem_index}-uniform"
                ] = float(self._rng.random())

        for variable_index in range(len(self._variable_names)):
            cache[f"variable-{variable_index}-uniform"] = float(self._rng.random())

        particle.random_cache = cache

    def create_random_cache(self, particle_ids: list[str], initialize: bool) -> None:
        for identifier in particle_ids:
            particle = self._population[identifier]
            particle.random_cache = {}

            should_sample = (
                initialize and identifier in self._temporary_particle_ids
            ) or (not initialize and identifier in self._active_parent_ids)

            if should_sample:
                self._cache_offspring_random_values(particle)

    def initialize_particle(self, identifier: str) -> RBOAParticle:
        particle = self._population[identifier]

        if not isinstance(particle, RBOAParticle):
            raise TypeError(
                f"Particle '{identifier}' must be an instance of RBOAParticle."
            )

        if identifier in self._temporary_particle_ids:
            candidate_variables = self._sample_offspring(particle)
            particle.update(
                variables=candidate_variables, fitness_function=self._fitness_function
            )
            return particle

        if particle.fitness == FITNESS_UNDEFINED:
            particle.update(
                variables=particle.variables, fitness_function=self._fitness_function
            )

        return particle

    @staticmethod
    def consolidate_new_particles(particle: RBOAParticle) -> RBOAParticle:
        particle.consolidate(consolidate_new=True)
        return particle

    def _select_network(self, random_value: float) -> tuple[int, _NetworkModel]:
        network_index = int(
            np.searchsorted(
                self._network_cumulative_weights, random_value, side="right"
            )
        )
        network_index = min(network_index, len(self._network_models) - 1)
        return network_index, self._network_models[network_index]

    @staticmethod
    def _select_cluster(model: _SubproblemModel, random_value: float) -> _ClusterModel:
        cluster_index = int(
            np.searchsorted(model.cumulative_weights, random_value, side="right")
        )
        cluster_index = min(cluster_index, len(model.clusters) - 1)
        return model.clusters[cluster_index]

    @staticmethod
    def _sample_truncated_standardized_normal(
        mean: float, standard_deviation: float, random_value: float
    ) -> float:
        epsilon = np.finfo(float).eps
        probability = float(np.clip(random_value, epsilon, 1.0 - epsilon))
        lower = (0.0 - mean) / standard_deviation
        upper = (1.0 - mean) / standard_deviation
        value = float(
            truncnorm.ppf(probability, lower, upper, loc=mean, scale=standard_deviation)
        )

        if not np.isfinite(value):
            value = float(np.clip(mean, 0.0, 1.0))

        return float(np.clip(value, 0.0, 1.0))

    def _sample_offspring(self, particle: RBOAParticle) -> dict[str, float]:
        if not self._network_models:
            raise RuntimeError("IrBOA probability model has not been initialized.")

        network_value = float(particle.random_cache["network-uniform"])
        network_index, network = self._select_network(network_value)
        candidate = np.full(len(self._variable_names), np.nan, dtype=float)

        for subproblem_index, model in enumerate(network.subproblem_models):
            mixture_value = float(
                particle.random_cache[
                    f"network-{network_index}-subproblem-{subproblem_index}-uniform"
                ]
            )
            cluster = self._select_cluster(model=model, random_value=mixture_value)

            for conditional in cluster.conditionals:
                if conditional.parent_indices:
                    parent_values = candidate[
                        np.asarray(conditional.parent_indices, dtype=int)
                    ]

                    if np.any(np.isnan(parent_values)):
                        raise RuntimeError(
                            "IrBOA attempted to sample a variable before one of its "
                            "Bayesian-network parents."
                        )

                    conditional_mean = conditional.mean + float(
                        conditional.coefficients
                        @ (parent_values - conditional.parent_means)
                    )
                else:
                    conditional_mean = conditional.mean

                random_value = float(
                    particle.random_cache[
                        f"variable-{conditional.variable_index}-uniform"
                    ]
                )
                candidate[conditional.variable_index] = (
                    self._sample_truncated_standardized_normal(
                        mean=conditional_mean,
                        standard_deviation=conditional.standard_deviation,
                        random_value=random_value,
                    )
                )

        if np.any(np.isnan(candidate)):
            raise RuntimeError("IrBOA did not sample every encoded decision variable.")

        return self._normalized_array_to_variables(candidate)

    def update_particle(self, identifier: str) -> RBOAParticle:
        particle = self._population[identifier]

        if not isinstance(particle, RBOAParticle):
            raise TypeError(
                f"Particle '{identifier}' must be an instance of RBOAParticle."
            )

        if identifier in self._temporary_particle_ids:
            return particle

        if identifier not in self._active_parent_ids:
            return particle

        candidate_variables = self._sample_offspring(particle)
        particle.update(
            variables=candidate_variables, fitness_function=self._fitness_function
        )

        return particle

    def _offspring_solutions(self) -> list[tuple[dict[str, float], np.float64]]:
        offspring: list[tuple[dict[str, float], np.float64]] = []

        for identifier in self._parent_particle_ids:
            if identifier not in self._active_parent_ids:
                continue

            particle = self._population[identifier]

            if (
                particle.candidate_variables is None
                or particle.candidate_fitness is None
            ):
                raise RuntimeError(
                    f"RBOA particle '{identifier}' has no evaluated offspring "
                    "candidate."
                )

            offspring.append(
                (
                    deepcopy(particle.candidate_variables),
                    np.float64(particle.candidate_fitness),
                )
            )

        for identifier in self._temporary_particle_ids:
            particle = self._population[identifier]
            offspring.append(
                (deepcopy(particle.variables), np.float64(particle.fitness))
            )

        return offspring

    def _parent_solutions(self) -> list[tuple[dict[str, float], np.float64]]:
        return [
            (
                deepcopy(self._population[identifier].variables),
                np.float64(self._population[identifier].fitness),
            )
            for identifier in self._parent_particle_ids
        ]

    def _rank_solutions(
        self, solutions: Collection[tuple[dict[str, float], np.float64]]
    ) -> list[tuple[dict[str, float], np.float64]]:
        candidates = list(solutions)

        if not candidates:
            return []

        permutation = self._rng.permutation(len(candidates))
        randomized = [candidates[int(index)] for index in permutation]
        randomized.sort(key=lambda solution: self._fitness_sort_key(solution[1]))
        return randomized

    def _renew_population(
        self,
        parents: Collection[tuple[dict[str, float], np.float64]],
        offspring: Collection[tuple[dict[str, float], np.float64]],
    ) -> list[tuple[dict[str, float], np.float64]]:
        parent_list = list(parents)
        offspring_list = list(offspring)
        population_size = len(parent_list)
        offspring_size = len(offspring_list)

        parent_survivor_count = max(population_size - offspring_size, 0)
        offspring_survivor_count = population_size - parent_survivor_count

        retained_parents = self._rank_solutions(parent_list)[:parent_survivor_count]

        if offspring_survivor_count == offspring_size:
            retained_offspring = offspring_list
        else:
            retained_offspring = self._rank_solutions(offspring_list)[
                :offspring_survivor_count
            ]

        survivors = [*retained_parents, *retained_offspring]

        if len(survivors) != population_size:
            raise RuntimeError(
                "RBOA renewal did not preserve the persistent population size."
            )

        return survivors

    def _apply_survivors(
        self, survivors: Collection[tuple[dict[str, float], np.float64]]
    ) -> None:
        survivor_list = list(survivors)

        if len(survivor_list) != len(self._parent_particle_ids):
            raise RuntimeError(
                "RBOA survivor count must equal the persistent population size."
            )

        for identifier, (variables, fitness) in zip(
            self._parent_particle_ids, survivor_list, strict=True
        ):
            particle = self._population[identifier]
            particle.candidate_variables = deepcopy(variables)
            particle.candidate_fitness = np.float64(fitness)
            particle.consolidate(consolidate_new=True)

    def _clear_temporary_particles(self) -> None:
        for identifier in tuple(self._temporary_particle_ids):
            if identifier in self._population:
                self.delete_particle(identifier)

        self._temporary_particle_ids = set()

    def post_iteration(self, actual_iter: int) -> None:
        if actual_iter == 0:
            self.update_solution_state()
            return

        offspring = self._offspring_solutions()
        expected_offspring = self._iteration_offspring_size

        if len(offspring) != expected_offspring:
            raise RuntimeError(
                "RBOA generated an unexpected number of evaluated offspring."
            )

        parents = self._parent_solutions()
        survivors = self._renew_population(parents=parents, offspring=offspring)
        self._apply_survivors(survivors)
        self._clear_temporary_particles()
        self._active_parent_ids = set()
        self._iteration_offspring_size = 0
        self._parent_particle_ids = list(self._population)
        self.update_solution_state()
