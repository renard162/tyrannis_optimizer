from collections.abc import Collection
from copy import deepcopy
from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy.linalg import eigvalsh, inv, lstsq, solve
from scipy.sparse.csgraph import shortest_path
from scipy.special import gammaln
from scipy.stats import chi2

from ..core.algorithm import FITNESS_UNDEFINED, AlgorithmBase, ParticleBase

_FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]


@dataclass(frozen=True)
class _ConditionalNormal:
    variable_index: int
    parent_indices: tuple[int, ...]
    mean: float
    parent_means: _FloatArray
    coefficients: _FloatArray
    standard_deviation: float


@dataclass(frozen=True)
class _BGeContext:
    sample_count: int
    dimension: int
    alpha_mu: float
    alpha_w: float
    prior_scale: _FloatArray
    posterior_scale: _FloatArray


class EGNAParticle(ParticleBase):
    """Particle implementation for the Estimation of Gaussian Networks Algorithm."""


class EGNA(AlgorithmBase[EGNAParticle]):
    """Estimation of Gaussian Networks Algorithm."""

    def __init__(
        self,
        selection_fraction: float = 0.5,
        structure_learning: str = "bic",
        variance_scaling: str = "sdr-avs",
        edge_exclusion_alpha: float = 0.05,
        bge_alpha_mu: float = 1.0,
        bge_alpha_w: float | None = None,
        sdr_avs_eta_decay: float = 0.9,
        sdr_avs_threshold: float = 1.0
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
        and fits a Gaussian concentration model to the retained dependencies.

        The ``variance_scaling`` argument controls whether the learned Gaussian
        network is sampled without variance scaling or with one of two published
        adaptive variance-scaling mechanisms. ``"no-scaling"`` follows the
        original EGNA formulation without adaptive variance scaling,
        ``"sdr-avs"`` applies Standard-Deviation Ratio triggered Adaptive Variance
        Scaling, and ``"ce-avs"`` applies two-stage Cross-Entropy Adaptive
        Variance Scaling.

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
            - ``"edge-exclusion"``: learns the network through conditional
              independence tests that determine which edges should be excluded.

        variance_scaling : {"no-scaling", "sdr-avs", "ce-avs"}, default="sdr-avs"
            Defines how the variance of the learned Gaussian network is treated
            before offspring sampling.

            - ``"no-scaling"``: applies no variance-scaling mechanism and samples
              directly from the Gaussian network estimated by EGNA.
            - ``"sdr-avs"``: applies Standard-Deviation Ratio triggered Adaptive
              Variance Scaling.
            - ``"ce-avs"``: applies two-stage Cross-Entropy Adaptive Variance
              Scaling.

        edge_exclusion_alpha : float, default=0.05
            Significance level used only when
            ``structure_learning="edge-exclusion"``. An edge is retained when the
            null hypothesis of conditional independence is rejected at this level.
            The likelihood-ratio edge-exclusion statistic is evaluated with the
            finite-sample correction described by Smith and Whittaker.

        bge_alpha_mu : float, default=1.0
            Positive equivalent sample size for the mean component of the
            normal-Wishart prior used only when ``structure_learning="bge"``.
            The default follows a conventional normal-Wishart BGe prior specification.

        bge_alpha_w : float or None, default=None
            Wishart equivalent sample size used only when
            ``structure_learning="bge"``. ``None`` resolves to ``d + 2``, where
            ``d`` is the number of encoded variables, following the conventional
            proper BGe prior. An explicitly supplied value must be greater
            than ``d + 1`` because the associated isotropic prior scale must be
            strictly positive.

        sdr_avs_eta_decay : float, default=0.9
            Multiplicative decay factor used only when
            ``variance_scaling="sdr-avs"``. When no improving offspring is found,
            the variance multiplier is multiplied by this value. When the SDR
            trigger requests additional exploration, the multiplier is increased
            by the reciprocal factor. The default follows Bosman, Grahl, and
            Rothlauf.

        sdr_avs_threshold : float, default=1.0
            Standard-Deviation Ratio threshold used only when
            ``variance_scaling="sdr-avs"``. Additional variance enlargement is
            triggered when the largest conditional SDR of the centroid of
            improving offspring exceeds this value. The default follows Bosman,
            Grahl, and Rothlauf.

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
        improvement. If an inherited family becomes rank deficient, redundant
        parent columns are removed while preserving its fitted projection before
        resuming score comparisons. A machine-precision variance floor makes the
        likelihood finite for exactly fitted data; there is no finite unrestricted
        Gaussian maximum likelihood estimate in that case.

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

        The fixed box-centered mean and unit prior-predictive marginal variance
        are prior choices in normalized coordinates, not universal EGNA defaults.

        With ``structure_learning="edge-exclusion"``, edge-exclusion tests are
        applied to the saturated Gaussian graphical model. For variables ``X_i``
        and ``X_j``, the sample partial correlation conditioned on all remaining
        variables is obtained from the precision matrix. The likelihood-ratio
        statistic is

        ``T = -N * log(1 - r_ij|rest**2)``.

        Edges whose conditional-independence null hypothesis is not rejected at
        ``edge_exclusion_alpha`` are removed. Retained undirected dependencies are
        constraints on the precision matrix, not as directed edges. The covariance
        is fitted by constrained Gaussian maximum likelihood, then factored using
        all predecessors in encoded-variable order. This chain-rule factorization
        preserves the fitted joint distribution, including for nonchordal graphs.
        The p-value uses the first-order finite-sample CDF correction; on singular
        data, machine-precision covariance stabilization is a numerical extension,
        not an exact significance test for the original degenerate sample.

        With ``variance_scaling="no-scaling"``, the learned conditional standard
        deviations are sampled without adaptive scaling and the complete offspring
        population replaces the previous generation, preserving the original EGNA
        behavior.

        With ``variance_scaling="sdr-avs"``, Tyrannis follows the SDR-AVS
        generational loop for a Bayesian-factorized normal EDA. The variance
        multiplier ``c_AVS`` starts at 1 and the sampling covariance is
        ``c_AVS * Sigma``. Improving offspring are those whose fitness is better
        than the worst member of the selected set used to estimate the model. The
        centroid of those improvements is evaluated in each conditional Gaussian;
        the largest conditional standard-deviation ratio is compared with
        ``sdr_avs_threshold``. If it exceeds the threshold, ``c_AVS`` is multiplied
        by ``1 / sdr_avs_eta_decay``. If no improvement is found, it is multiplied
        by ``sdr_avs_eta_decay``. The multiplier is never allowed below 1. The
        selected set is retained and offspring replace the unselected population.

        With ``variance_scaling="ce-avs"``, Tyrannis uses the two-stage compositive
        CE-AVS factor. If the previous model is ``N(mu_M, Gamma_M)``, the survivor
        population after truncate replacement is modeled as
        ``N(mu_P, Gamma_P)``, and the newly learned model mean is ``mu_M_new``,
        then

        ``a1 = sqrt(trace(Gamma_M^-1 * (Gamma_P + d_P d_P^T)) / d)``,

        ``a2 = sqrt(trace(Gamma_M^-1 * (d_M d_M^T)) / d)``,

        where ``d_P = mu_M - mu_P`` and ``d_M = mu_M - mu_M_new``. Sampling uses
        ``N(mu_M_new, a^2 * Gamma_M_new)`` with ``a = a1 + a2``. The CE-AVS branch
        performs truncate replacement over the union of parents and offspring, as
        required for the survivor-distribution comparison used by CE-AVS.

        Global variance scaling leaves the learned regression coefficients
        unchanged and scales every conditional residual standard deviation by the
        same standard-deviation factor. This is equivalent to multiplying the full
        Gaussian-network covariance by the corresponding scalar. For adaptive
        scaling branches, residual variances are kept strictly positive at machine
        precision so that variance enlargement remains mathematically possible and
        the CE-AVS linear systems remain nonsingular.

        EGNA is defined for continuous Gaussian models with unrestricted
        dependencies. This implementation therefore requires the selected sample
        count to exceed the number of encoded variables. Exact numerical
        degeneracies are handled at machine precision; they do not introduce a
        user-tunable regularization strength.

        Tyrannis optimizes over finite encoded boundaries. Model fitting is carried
        out after an affine normalization of every encoded variable to ``[0, 1]``.
        Sampling generates the complete unbounded Gaussian network and rejects
        the whole vector if any coordinate is outside that interval. Accepted
        vectors therefore follow the joint Gaussian conditioned on the box.
        Variance scaling applies to the Gaussian proposal; the covariance after
        conditioning generally differs from that proposal covariance. CE-AVS keeps
        its published unbounded-Gaussian adaptation rule, not a new truncated-KL
        minimizer. After 100,000 unsuccessful proposals an explicit sampling error
        is raised rather than substituting another distribution.

        References
        ----------
        Larrañaga, P., Etxeberria, R., Lozano, J. A., & Peña, J. M. (2000).
        Optimization in Continuous Domains by Learning and Simulation of Gaussian
        Networks. Workshop on Optimization by Building and Using Probabilistic
        Models, GECCO 2000, 201-204.
        https://www.researchgate.net/publication/290129690_Optimization_in_continuous_domains_by_learning_and_simulation_of_Gaussian_networks

        Bengoetxea, E., Miquélez, T., Larrañaga, P., & Lozano, J. A. (2002).
        Experimental Results in Function Optimization with EDAs in Continuous
        Domain. In Estimation of Distribution Algorithms: A New Tool for
        Evolutionary Computation, 181-194. Springer.
        https://doi.org/10.1007/978-1-4615-1539-5_8

        Geiger, D., & Heckerman, D. (1994). Learning Gaussian Networks.
        Uncertainty Proceedings 1994, 235-243.
        https://doi.org/10.1016/B978-1-55860-332-5.50035-3

        Kuipers, J., Moffa, G., & Heckerman, D. (2014). Addendum on the
        Scoring of Gaussian Directed Acyclic Graphical Models.
        Annals of Statistics, 42(4), 1689-1691.
        https://doi.org/10.1214/14-AOS1217

        Dempster, A. P. (1972). Covariance Selection. Biometrics, 28(1), 157-175.
        https://doi.org/10.2307/2528966

        Smith, P. W. F., & Whittaker, J. (1998). Edge Exclusion Tests for
        Graphical Gaussian Models. In M. I. Jordan (Ed.), Learning in Graphical
        Models, 555-574. Springer.
        https://doi.org/10.1007/978-94-011-5014-9_21

        Bosman, P. A. N., Grahl, J., & Rothlauf, F. (2007). SDR: A Better
        Trigger for Adaptive Variance Scaling in Normal EDAs. Proceedings of the
        Genetic and Evolutionary Computation Conference, 492-499.
        https://doi.org/10.1145/1276958.1277066

        Cai, Y., Sun, X., Xu, H., & Jia, P. (2007). Cross Entropy and Adaptive
        Variance Scaling in Continuous EDA. Proceedings of the Genetic and
        Evolutionary Computation Conference, 609-616.
        https://doi.org/10.1145/1276958.1277081

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

        if not isinstance(variance_scaling, str):
            raise TypeError("variance_scaling must be a string.")

        allowed_variance_scaling = ("no-scaling", "sdr-avs", "ce-avs")

        if variance_scaling not in allowed_variance_scaling:
            raise ValueError(
                "variance_scaling must be 'no-scaling', 'sdr-avs', or 'ce-avs'."
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

        if not isinstance(sdr_avs_eta_decay, (int, float, np.number)) or isinstance(
            sdr_avs_eta_decay, (bool, np.bool_)
        ):
            raise TypeError("sdr_avs_eta_decay must be a number.")

        if (
            not np.isfinite(sdr_avs_eta_decay)
            or sdr_avs_eta_decay <= 0
            or sdr_avs_eta_decay > 1
        ):
            raise ValueError(
                "sdr_avs_eta_decay must be finite and greater than 0 and less "
                "than or equal to 1."
            )

        if not isinstance(sdr_avs_threshold, (int, float, np.number)) or isinstance(
            sdr_avs_threshold, (bool, np.bool_)
        ):
            raise TypeError("sdr_avs_threshold must be a number.")

        if not np.isfinite(sdr_avs_threshold) or sdr_avs_threshold < 0:
            raise ValueError(
                "sdr_avs_threshold must be finite and greater than or equal to 0."
            )

        self._selection_fraction = float(selection_fraction)
        self._structure_learning = structure_learning
        self._variance_scaling = variance_scaling
        self._edge_exclusion_alpha = float(edge_exclusion_alpha)
        self._bge_alpha_mu = float(bge_alpha_mu)
        self._bge_alpha_w = None if bge_alpha_w is None else float(bge_alpha_w)
        self._sdr_avs_eta_decay = float(sdr_avs_eta_decay)
        self._sdr_avs_threshold = float(sdr_avs_threshold)

        self._variable_names: tuple[str, ...] = ()
        self._lower_bounds: _FloatArray | None = None
        self._spans: _FloatArray | None = None
        self._parents: tuple[tuple[int, ...], ...] = ()
        self._topological_order: tuple[int, ...] = ()
        self._conditionals: tuple[_ConditionalNormal, ...] = ()
        self._active_particle_ids: set[str] = set()
        self._selected_particle_ids: tuple[str, ...] = ()
        self._selected_fitness_threshold: float | None = None
        self._variance_floor = np.finfo(float).eps
        self._sampling_standard_deviation_scale = 1.0
        self._sdr_variance_multiplier = 1.0
        self._ce_previous_model_mean: _FloatArray | None = None
        self._ce_previous_model_covariance: _FloatArray | None = None

    @property
    def selection_fraction(self) -> float:
        return self._selection_fraction

    @property
    def structure_learning(self) -> str:
        return self._structure_learning

    @property
    def variance_scaling(self) -> str:
        return self._variance_scaling

    @property
    def edge_exclusion_alpha(self) -> float:
        return self._edge_exclusion_alpha

    @property
    def bge_alpha_mu(self) -> float:
        return self._bge_alpha_mu

    @property
    def bge_alpha_w(self) -> float | None:
        return self._bge_alpha_w

    @property
    def sdr_avs_eta_decay(self) -> float:
        return self._sdr_avs_eta_decay

    @property
    def sdr_avs_threshold(self) -> float:
        return self._sdr_avs_threshold

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
        self._active_particle_ids = set()
        self._selected_particle_ids = ()
        self._selected_fitness_threshold = None
        self._sampling_standard_deviation_scale = 1.0
        self._sdr_variance_multiplier = 1.0
        self._ce_previous_model_mean = None
        self._ce_previous_model_covariance = None

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

    def _variables_to_normalized_array(
        self, variables: dict[str, float]
    ) -> _FloatArray:
        if self._lower_bounds is None or self._spans is None:
            raise RuntimeError("EGNA model context has not been initialized.")

        values = np.asarray(
            [variables[name] for name in self._variable_names], dtype=float
        )
        normalized = (values - self._lower_bounds) / self._spans
        return np.clip(normalized, 0.0, 1.0)

    def _normalized_array_to_variables(self, values: _FloatArray) -> dict[str, float]:
        if self._lower_bounds is None or self._spans is None:
            raise RuntimeError("EGNA model context has not been initialized.")

        bounded = np.clip(values, 0.0, 1.0)
        encoded = self._lower_bounds + bounded * self._spans

        return {
            # Only floating-point affine roundoff is clamped: the proposal was
            # already accepted inside the normalized box before this conversion.
            name: float(np.clip(value, *self._boundaries[name]))
            for name, value in zip(self._variable_names, encoded, strict=True)
        }

    def _selected_matrix(self) -> _FloatArray:
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
        self._selected_particle_ids = tuple(selected_ids)
        self._selected_fitness_threshold = float(
            self._population[selected_ids[-1]].fitness
        )

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
        parents: list[set[int]] | tuple[tuple[int, ...], ...]
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.uint8]]:
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
        self, data: _FloatArray, child: int, parents: tuple[int, ...]
    ) -> float:
        y = data[:, child]
        n_samples = len(y)

        if parents:
            parent_values = data[:, parents]
            design = np.column_stack((np.ones(n_samples), parent_values))

            if np.linalg.matrix_rank(design) < design.shape[1]:
                return -np.inf

            least_squares = lstsq(
                design,
                y,
                cond=np.finfo(float).eps * max(design.shape),
                lapack_driver="gelsd",
                check_finite=False
            )
            # SciPy's untyped dispatcher includes None; gelsd returns four values.
            coefficients, _, _, _ = cast(
                tuple[_FloatArray, object, int, _FloatArray], least_squares
            )
            residuals = y - design @ coefficients
        else:
            residuals = y - float(np.mean(y))

        residual_variance = float(np.mean(np.square(residuals)))
        variance = max(residual_variance, self._variance_floor)
        log_likelihood = (
            -0.5
            * n_samples
            * (np.log(2.0 * np.pi * variance) + residual_variance / variance)
        )
        n_parameters = len(parents) + 2
        penalty = 0.5 * n_parameters * np.log(n_samples)
        return float(log_likelihood - penalty)

    def _resolve_bge_alpha_w(self) -> float:
        dimension = len(self._variable_names)

        if self._bge_alpha_w is None:
            return float(dimension + 2)

        return self._bge_alpha_w

    def _prepare_bge_context(self, data: _FloatArray) -> _BGeContext:
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
    def _submatrix_logdet(matrix: _FloatArray, indices: tuple[int, ...]) -> float:
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
        self, data: _FloatArray, metric: str
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

        if metric == "bic" and use_previous_structure:
            for child in range(n_variables):
                if np.isfinite(score(child, parents[child])):
                    continue

                # A basis of the inherited design preserves its projection and RSS.
                # Removing dependent columns also reduces the parameter penalty.
                design = np.ones((len(data), 1))
                independent: set[int] = set()
                for parent in sorted(parents[child]):
                    candidate_design = np.column_stack((design, data[:, parent]))
                    if np.linalg.matrix_rank(candidate_design) == design.shape[1] + 1:
                        independent.add(parent)
                        design = candidate_design
                parents[child] = independent

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

    def _regularized_precision(self, data: _FloatArray) -> _FloatArray:
        centered = data - np.mean(data, axis=0)
        covariance = centered.T @ centered / len(data)
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues = eigvalsh(covariance, driver="evd", check_finite=False)
        scale = max(1.0, float(np.max(np.abs(np.diag(covariance)))))
        minimum_eigenvalue = np.finfo(float).eps * len(self._variable_names) * scale
        smallest = float(np.min(eigenvalues))

        if smallest < minimum_eigenvalue:
            covariance = covariance + np.eye(covariance.shape[0]) * (
                minimum_eigenvalue - smallest
            )

        try:
            return inv(covariance, check_finite=False)
        except np.linalg.LinAlgError as error:
            raise RuntimeError(
                "Unable to invert the EGNA covariance matrix for edge-exclusion "
                "structure learning."
            ) from error

    def _edge_exclusion_structure(
        self, data: _FloatArray
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
                statistic_density_product = np.sqrt(statistic / (2.0 * np.pi)) * np.exp(
                    -0.5 * statistic
                )
                correction = (
                    0.5 * (2 * n_variables + 1) * statistic_density_product / n_samples
                )
                p_value = float(
                    np.clip(chi2.sf(statistic, df=1) + correction, 0.0, 1.0)
                )

                if p_value < self._edge_exclusion_alpha:
                    parents[child].add(parent)
                    parents[parent].add(child)

        return tuple(tuple(sorted(parent_set)) for parent_set in parents)

    def _learn_structure(self, data: _FloatArray) -> tuple[tuple[int, ...], ...]:
        if self._structure_learning == "edge-exclusion":
            return self._edge_exclusion_structure(data)

        return self._score_search_structure(data=data, metric=self._structure_learning)

    def _fit_conditionals(
        self, data: _FloatArray, parents: tuple[tuple[int, ...], ...]
    ) -> tuple[_ConditionalNormal, ...]:
        means = np.asarray(np.mean(data, axis=0), dtype=np.float64)
        conditionals: list[_ConditionalNormal] = []

        for variable, parent_indices in enumerate(parents):
            y = data[:, variable]
            mean = float(means[variable])

            if parent_indices:
                parent_values = data[:, parent_indices]
                parent_means = means[np.asarray(parent_indices, dtype=int)]
                centered_parents = parent_values - parent_means
                centered_y = y - mean
                least_squares = lstsq(
                    centered_parents,
                    centered_y,
                    cond=np.finfo(float).eps * max(centered_parents.shape),
                    lapack_driver="gelsd",
                    check_finite=False
                )
                # SciPy's untyped dispatcher includes None; gelsd returns four values.
                coefficients, _, _, _ = cast(
                    tuple[_FloatArray, object, int, _FloatArray], least_squares
                )
                residuals = centered_y - centered_parents @ coefficients
            else:
                parent_means = np.empty(0, dtype=float)
                coefficients = np.empty(0, dtype=float)
                residuals = y - mean

            variance = max(float(np.mean(np.square(residuals))), 0.0)

            if self._variance_scaling == "no-scaling":
                standard_deviation = float(np.sqrt(variance))
            else:
                variance = max(variance, self._variance_floor)
                standard_deviation = float(np.sqrt(variance))

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

    def _fit_concentration_covariance(
        self, data: _FloatArray, neighbors: tuple[tuple[int, ...], ...]
    ) -> _FloatArray:
        """Maximize logdet(Sigma) with empirical diagonal/edge entries fixed."""
        centered = data - np.asarray(np.mean(data, axis=0), dtype=np.float64)
        empirical = self._regularize_covariance_matrix(
            centered.T @ centered / len(data)
        )
        deviations = np.sqrt(np.diag(empirical))
        empirical = empirical / np.outer(deviations, deviations)
        covariance = empirical.copy()
        dimension = len(neighbors)
        missing = np.ones((dimension, dimension), dtype=bool)
        np.fill_diagonal(missing, False)
        for node, adjacent in enumerate(neighbors):
            missing[node, list(adjacent)] = False

        if not np.any(missing):
            return covariance * np.outer(deviations, deviations)

        for _ in range(10_000):
            for node, adjacent in enumerate(neighbors):
                other = [index for index in range(dimension) if index != node]
                if adjacent:
                    coefficients = solve(
                        covariance[np.ix_(adjacent, adjacent)],
                        empirical[list(adjacent), node],
                        assume_a="pos",
                        check_finite=False
                    )
                    column = covariance[np.ix_(other, adjacent)] @ coefficients
                else:
                    column = np.zeros(dimension - 1)
                covariance[other, node] = column
                covariance[node, other] = column

            precision = inv(covariance, check_finite=False)
            partial = precision / np.sqrt(
                np.outer(np.diag(precision), np.diag(precision))
            )
            if np.max(np.abs(partial[missing])) <= 1e-10:
                return covariance * np.outer(deviations, deviations)

        raise RuntimeError("EGNA concentration-model likelihood fit did not converge.")

    def _concentration_conditionals(
        self, data: _FloatArray, neighbors: tuple[tuple[int, ...], ...]
    ) -> tuple[_ConditionalNormal, ...]:
        covariance = self._fit_concentration_covariance(data, neighbors)
        means = np.asarray(np.mean(data, axis=0), dtype=np.float64)
        conditionals: list[_ConditionalNormal] = []
        for variable in range(data.shape[1]):
            coefficients = solve(
                covariance[:variable, :variable],
                covariance[:variable, variable],
                assume_a="pos",
                check_finite=False
            )
            # NumPy scalar indexing/matmul erases the known float64 scalar dtype.
            variance = float(
                cast(
                    np.float64,
                    covariance[variable, variable]
                    - covariance[variable, :variable] @ coefficients
                )
            )
            if variance <= 0 or not np.isfinite(variance):
                raise RuntimeError(
                    "EGNA concentration covariance is not positive definite."
                )
            conditionals.append(
                _ConditionalNormal(
                    variable_index=variable,
                    parent_indices=tuple(range(variable)),
                    mean=means.item(variable),
                    parent_means=means[:variable].copy(),
                    coefficients=coefficients,
                    standard_deviation=float(np.sqrt(variance))
                )
            )
        return tuple(conditionals)

    def _learn_probability_model(self, selected_data: _FloatArray) -> None:
        parents = self._learn_structure(selected_data)
        if self._structure_learning == "edge-exclusion":
            conditionals = self._concentration_conditionals(selected_data, parents)
            parents = tuple(conditional.parent_indices for conditional in conditionals)
        else:
            conditionals = self._fit_conditionals(data=selected_data, parents=parents)
        topological_order = self._topological_sort(parents)

        self._parents = parents
        self._topological_order = topological_order
        self._conditionals = conditionals

    def _regularize_covariance_matrix(self, covariance: _FloatArray) -> _FloatArray:
        covariance = np.asarray(covariance, dtype=float)
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues = eigvalsh(covariance, driver="evd", check_finite=False)
        scale = max(1.0, float(np.max(np.abs(np.diag(covariance)))))
        minimum_eigenvalue = np.finfo(float).eps * len(self._variable_names) * scale
        smallest = float(np.min(eigenvalues))

        if smallest < minimum_eigenvalue:
            covariance = covariance + np.eye(covariance.shape[0]) * (
                minimum_eigenvalue - smallest
            )

        return covariance

    def _model_joint_moments(self) -> tuple[_FloatArray, _FloatArray]:
        n_variables = len(self._variable_names)

        if len(self._conditionals) != n_variables:
            raise RuntimeError("EGNA probability model has not been initialized.")

        means = np.empty(n_variables, dtype=float)
        regression = np.zeros((n_variables, n_variables), dtype=float)
        residual_variances = np.empty(n_variables, dtype=float)

        for conditional in self._conditionals:
            variable = conditional.variable_index
            means[variable] = conditional.mean
            residual_variances[variable] = max(
                conditional.standard_deviation**2, self._variance_floor
            )

            for parent, coefficient in zip(
                conditional.parent_indices, conditional.coefficients, strict=True
            ):
                regression[variable, parent] = float(coefficient)

        system = np.eye(n_variables, dtype=float) - regression

        try:
            transition = inv(system, check_finite=False)
        except np.linalg.LinAlgError as error:
            raise RuntimeError(
                "Unable to reconstruct the EGNA joint Gaussian covariance."
            ) from error

        covariance = transition @ np.diag(residual_variances) @ transition.T
        covariance = self._regularize_covariance_matrix(covariance)
        return means, covariance

    def _population_moments(self) -> tuple[_FloatArray, _FloatArray]:
        if not self._population:
            raise ValueError("EGNA requires at least one particle.")

        data = np.asarray(
            [
                self._variables_to_normalized_array(particle.variables)
                for particle in self._population.values()
            ],
            dtype=float,
        )
        mean = np.mean(data, axis=0)
        centered = data - mean
        covariance = centered.T @ centered / len(data)
        covariance = self._regularize_covariance_matrix(covariance)
        return np.asarray(mean, dtype=float), covariance

    @staticmethod
    def _solve_positive_definite(
        covariance: _FloatArray, value: _FloatArray
    ) -> _FloatArray:
        try:
            return solve(covariance, value, assume_a="pos", check_finite=False)
        except np.linalg.LinAlgError as error:
            raise RuntimeError(
                "Unable to solve an EGNA variance-scaling covariance system."
            ) from error

    def _ce_avs_scale(
        self,
        previous_model_mean: _FloatArray,
        previous_model_covariance: _FloatArray,
        population_mean: _FloatArray,
        population_covariance: _FloatArray,
        current_model_mean: _FloatArray
    ) -> float:
        dimension = len(previous_model_mean)
        population_shift = previous_model_mean - population_mean
        model_shift = previous_model_mean - current_model_mean

        covariance_term = float(
            np.trace(
                self._solve_positive_definite(
                    previous_model_covariance, population_covariance
                )
            )
        )
        population_shift_term = float(
            population_shift
            @ self._solve_positive_definite(previous_model_covariance, population_shift)
        )
        model_shift_term = float(
            model_shift
            @ self._solve_positive_definite(previous_model_covariance, model_shift)
        )

        a1_squared = max((covariance_term + population_shift_term) / dimension, 0.0)
        a2_squared = max(model_shift_term / dimension, 0.0)
        scale = float(np.sqrt(a1_squared) + np.sqrt(a2_squared))

        if not np.isfinite(scale) or scale <= 0:
            raise RuntimeError(
                "CE-AVS produced an invalid EGNA standard-deviation scale."
            )

        return scale

    def _prepare_variance_scaling(self) -> None:
        if self._variance_scaling == "no-scaling":
            self._sampling_standard_deviation_scale = 1.0
            return

        if self._variance_scaling == "sdr-avs":
            self._sampling_standard_deviation_scale = float(
                np.sqrt(self._sdr_variance_multiplier)
            )
            return

        current_model_mean, current_model_covariance = self._model_joint_moments()

        if (
            self._ce_previous_model_mean is None
            or self._ce_previous_model_covariance is None
        ):
            scale = 1.0
        else:
            population_mean, population_covariance = self._population_moments()
            scale = self._ce_avs_scale(
                previous_model_mean=self._ce_previous_model_mean,
                previous_model_covariance=self._ce_previous_model_covariance,
                population_mean=population_mean,
                population_covariance=population_covariance,
                current_model_mean=current_model_mean,
            )

        self._sampling_standard_deviation_scale = scale
        self._ce_previous_model_mean = current_model_mean.copy()
        self._ce_previous_model_covariance = current_model_covariance.copy()

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
        self._prepare_variance_scaling()

        if self._variance_scaling == "sdr-avs":
            selected = set(self._selected_particle_ids)
            active = set(self._population) - selected

            if not active:
                raise ValueError(
                    "SDR-AVS requires at least one offspring per generation. "
                    "Use selection_fraction less than 1."
                )

            self._active_particle_ids = active
        else:
            self._active_particle_ids = set(self._population)

    def create_random_cache(self, particle_ids: list[str], initialize: bool) -> None:
        for identifier in particle_ids:
            particle = self._population[identifier]
            particle.random_cache = {}

            if initialize or identifier not in self._active_particle_ids:
                continue

            particle.random_cache = {
                "random_generator": np.random.default_rng(
                    self._rng.integers(0, 2**32, size=4, dtype=np.uint32)
                )
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

    def _sample_candidate(self, particle: EGNAParticle) -> dict[str, float]:
        if not self._conditionals or not self._topological_order:
            raise RuntimeError("EGNA probability model has not been initialized.")

        local_rng: object = particle.random_cache["random_generator"]
        if not isinstance(local_rng, np.random.Generator):
            raise TypeError("EGNA sampling cache must contain a local Generator.")
        for _ in range(100_000):
            residuals = local_rng.standard_normal(len(self._variable_names))
            candidate = np.full(len(self._variable_names), np.nan, dtype=float)
            for variable in self._topological_order:
                conditional = self._conditionals[variable]
                parent_values = candidate[list(conditional.parent_indices)]
                conditional_mean = float(
                    conditional.mean
                    + conditional.coefficients
                    @ (parent_values - conditional.parent_means)
                )
                candidate[variable] = (
                    conditional_mean
                    + conditional.standard_deviation
                    * self._sampling_standard_deviation_scale
                    * residuals[variable]
                )

            if not np.all(np.isfinite(candidate)):
                raise RuntimeError("EGNA produced a non-finite Gaussian candidate.")
            if np.all((candidate >= 0.0) & (candidate <= 1.0)):
                return self._normalized_array_to_variables(candidate)

        raise RuntimeError(
            "EGNA joint Gaussian rejection sampling exhausted 100,000 attempts."
        )

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

    def _validate_evaluated_candidates(self) -> None:
        for identifier in self._active_particle_ids:
            particle = self._population[identifier]

            if (
                particle.candidate_variables is None
                or particle.candidate_fitness is None
            ):
                raise RuntimeError(
                    f"EGNA particle '{identifier}' has no evaluated candidate."
                )

    def _sdr_conditional_ratio(self, centroid: _FloatArray) -> float:
        ratios: list[float] = []
        standard_deviation_scale = float(np.sqrt(self._sdr_variance_multiplier))

        for variable in self._topological_order:
            conditional = self._conditionals[variable]

            if conditional.parent_indices:
                parent_values = centroid[
                    np.asarray(conditional.parent_indices, dtype=int)
                ]
                conditional_mean = float(
                    conditional.mean
                    + conditional.coefficients
                    @ (parent_values - conditional.parent_means)
                )
            else:
                conditional_mean = conditional.mean

            standard_deviation = (
                conditional.standard_deviation * standard_deviation_scale
            )

            if standard_deviation <= 0 or not np.isfinite(standard_deviation):
                raise RuntimeError(
                    "SDR-AVS requires positive finite conditional standard deviations."
                )

            ratios.append(
                abs(float(centroid[variable]) - conditional_mean) / standard_deviation
            )

        return max(ratios)

    def _update_sdr_variance_multiplier(self) -> None:
        threshold = self._selected_fitness_threshold

        if threshold is None:
            raise RuntimeError("SDR-AVS selected-fitness threshold is unavailable.")

        improvements: list[_FloatArray] = []

        for identifier in self._population:
            if identifier not in self._active_particle_ids:
                continue
            particle = self._population[identifier]
            candidate_fitness = particle.candidate_fitness
            candidate_variables = particle.candidate_variables

            if candidate_fitness is None or candidate_variables is None:
                raise RuntimeError(
                    f"EGNA particle '{identifier}' has no evaluated candidate."
                )

            if float(candidate_fitness) < threshold:
                improvements.append(
                    self._variables_to_normalized_array(candidate_variables)
                )

        if not improvements:
            self._sdr_variance_multiplier *= self._sdr_avs_eta_decay
        else:
            centroid = np.mean(np.asarray(improvements, dtype=float), axis=0)
            sdr = self._sdr_conditional_ratio(centroid)

            if sdr > self._sdr_avs_threshold:
                self._sdr_variance_multiplier /= self._sdr_avs_eta_decay

        self._sdr_variance_multiplier = max(1.0, self._sdr_variance_multiplier)

    def _parent_solutions(self) -> list[tuple[dict[str, float], np.float64]]:
        return [
            (deepcopy(particle.variables), np.float64(particle.fitness))
            for particle in self._population.values()
        ]

    def _offspring_solutions(self) -> list[tuple[dict[str, float], np.float64]]:
        offspring: list[tuple[dict[str, float], np.float64]] = []

        for identifier in self._population:
            if identifier not in self._active_particle_ids:
                continue
            particle = self._population[identifier]

            if (
                particle.candidate_variables is None
                or particle.candidate_fitness is None
            ):
                raise RuntimeError(
                    f"EGNA particle '{identifier}' has no evaluated candidate."
                )

            offspring.append(
                (
                    deepcopy(particle.candidate_variables),
                    np.float64(particle.candidate_fitness),
                )
            )

        return offspring

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

    def _apply_survivors(
        self, survivors: Collection[tuple[dict[str, float], np.float64]]
    ) -> None:
        survivor_list = list(survivors)
        identifiers = list(self._population)

        if len(survivor_list) != len(identifiers):
            raise RuntimeError(
                "EGNA survivor count must equal the persistent population size."
            )

        for identifier, (variables, fitness) in zip(
            identifiers, survivor_list, strict=True
        ):
            particle = self._population[identifier]
            particle.candidate_variables = deepcopy(variables)
            particle.candidate_fitness = np.float64(fitness)
            particle.consolidate(consolidate_new=True)

    def _ce_avs_replacement(self) -> None:
        parents = self._parent_solutions()
        offspring = self._offspring_solutions()
        population_size = len(parents)

        if len(offspring) != population_size:
            raise RuntimeError(
                "CE-AVS requires one evaluated offspring per persistent particle."
            )

        survivors = self._rank_solutions([*parents, *offspring])[:population_size]
        self._apply_survivors(survivors)

    def post_iteration(self, actual_iter: int) -> None:
        if actual_iter == 0:
            self.update_solution_state()
            return

        self._validate_evaluated_candidates()

        if self._variance_scaling == "ce-avs":
            self._ce_avs_replacement()
        else:
            if self._variance_scaling == "sdr-avs":
                self._update_sdr_variance_multiplier()

            for identifier in self._active_particle_ids:
                self._population[identifier].consolidate(consolidate_new=True)

        self._active_particle_ids = set()
        self.update_solution_state()
