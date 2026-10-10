"""EGNA: Gaussian-network EDA mathematics through the real Serial replica.

Bengoetxea et al. (2002), doi:10.1007/978-1-4615-1539-5_8, identifies
EGNABIC, EGNABGe and edge-exclusion; its accessible abstract was consulted.
Geiger & Heckerman (1994), sections 3.1-3.2, gives normal-Wishart updating
and score equivalence: https://dang.cs.technion.ac.il/conf_papers/GeigerH94.pdf
Smith & Whittaker (1998), doi:10.1007/978-94-011-5014-9_21, is the cited
edge-test source; its full text and Larranaga et al. (2000) were unavailable.
The precise finite-sample correction below therefore tests the supplied
Tyrannis contract, without claiming independent bibliographic verification.

Linear Gaussian conditionals model selected P_(t-1). Tyrannis specifies affine
[0,1] coordinates, truncated-normal ancestral sampling, prior settings,
structure reuse/addition/deletion tolerance, precision stabilization and
random caches. Its full generational replacement gives sampled P_t; t=0 is
setup. Numerical degeneracies are unit-test territory unless naturally reached.
"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from math import exp, floor, lgamma, log, pi, sqrt
from statistics import NormalDist
from typing import cast, override

import numpy as np
import pytest

from tests._support import factories
from tests._support.assertions import (
    assert_valid_permutation,
    assert_variables_within_bounds,
)
from tests._support.numerics import (
    BASE_SEED,
    LINALG_ATOL,
    LINALG_RTOL,
    STRICT_ATOL,
    STRICT_RTOL,
    seed_for,
)
from tests._support.objectives import constant_objective, sphere
from tyrannis import Optimizer
from tyrannis.algorithm.egna import EGNA, EGNAParticle

# Private model records have no public observational equivalent.
from tyrannis.algorithm.egna import (
    _BGeContext as BGeContext,  # pyright: ignore[reportPrivateUsage]
)
from tyrannis.algorithm.egna import (
    _ConditionalNormal as ConditionalNormal,  # pyright: ignore[reportPrivateUsage]
)
from tyrannis.core.algorithm import FITNESS_UNDEFINED
from tyrannis.processor.serial import Serial
from tyrannis.space.continuous import Continuous

Array = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
Parents = tuple[tuple[int, ...], ...]
Population = dict[str, EGNAParticle]


@dataclass
class Sample:
    identifier: str
    before: Population
    parents: Parents
    order: tuple[int, ...]
    conditionals: tuple[ConditionalNormal, ...]
    cache: dict[str, object]
    values: dict[str, float]


@dataclass
class Generation:
    before: Population
    previous: Parents
    selected: Array = field(default_factory=lambda: np.empty((0, 0)))
    parents: Parents = ()
    order: tuple[int, ...] = ()
    conditionals: tuple[ConditionalNormal, ...] = ()
    scores: dict[tuple[int, tuple[int, ...]], float] = field(default_factory=dict)
    context: BGeContext | None = None
    precision: Array | None = None
    fits: int = 0
    active: set[str] = field(default_factory=set)
    samples: list[Sample] = field(default_factory=list)
    caches: list[tuple[bool, dict[str, dict[str, object]]]] = field(
        default_factory=list
    )
    processed: list[tuple[str, dict[str, object], EGNAParticle]] = field(
        default_factory=list
    )
    before_post: Population = field(default_factory=dict)
    after: Population = field(default_factory=dict)


class ObservedEGNA(EGNA):
    """Delegate every hook unchanged; copy only inputs, outputs and model state."""

    # Production uses bare ndarray annotations; specialize observation boundaries.
    _lower_bounds: Array | None
    _spans: Array | None

    @override
    def pre_iteration(self, actual_iter: int) -> None:
        self.iteration: int = actual_iter
        if actual_iter == 0:
            self.generations: dict[int, Generation] = {}
        self.generations[actual_iter] = Generation(
            deepcopy(self.population), self._parents
        )
        super().pre_iteration(actual_iter)
        g = self.generations[actual_iter]
        g.parents, g.order = self._parents, self._topological_order
        g.conditionals = deepcopy(self._conditionals)
        g.active = self._active_particle_ids.copy()
        if actual_iter == 0:
            self.names: tuple[str, ...] = self._variable_names
            self.lower: Array | None = deepcopy(self._lower_bounds)
            self.spans: Array | None = deepcopy(self._spans)

    @override
    def _selected_matrix(self) -> Array:
        result = cast(Callable[[], Array], super()._selected_matrix)()
        self.generations[self.iteration].selected = result.copy()
        return result

    @override
    def _learn_probability_model(self, selected_data: Array) -> None:
        cast(Callable[[Array], None], super()._learn_probability_model)(selected_data)
        self.generations[self.iteration].fits += 1

    @override
    def _local_bic_score(
        self, data: Array, child: int, parents: tuple[int, ...]
    ) -> float:
        result = cast(
            Callable[[Array, int, tuple[int, ...]], float], super()._local_bic_score
        )(data, child, parents)
        self.generations[self.iteration].scores[child, parents] = result
        return result

    @override
    def _prepare_bge_context(self, data: Array) -> BGeContext:
        result = cast(Callable[[Array], BGeContext], super()._prepare_bge_context)(data)
        self.generations[self.iteration].context = deepcopy(result)
        return result

    @override
    def _local_bge_score(
        self, context: BGeContext, child: int, parents: tuple[int, ...]
    ) -> float:
        result = super()._local_bge_score(context, child, parents)
        self.generations[self.iteration].scores[child, parents] = result
        return result

    @override
    def _regularized_precision(self, data: Array) -> Array:
        result = cast(Callable[[Array], Array], super()._regularized_precision)(data)
        self.generations[self.iteration].precision = result.copy()
        return result

    @override
    def create_random_cache(self, particle_ids: list[str], initialize: bool) -> None:
        super().create_random_cache(particle_ids, initialize)
        self.generations[self.iteration].caches.append(
            (
                initialize,
                {
                    key: deepcopy(self.population[key].random_cache)
                    for key in particle_ids
                },
            )
        )

    @override
    def initialize_particle(self, identifier: str) -> EGNAParticle:
        cache = deepcopy(self.population[identifier].random_cache)
        result = super().initialize_particle(identifier)
        self.generations[self.iteration].processed.append(
            (identifier, cache, deepcopy(result))
        )
        return result

    @override
    def update_particle(self, identifier: str) -> EGNAParticle:
        cache = deepcopy(self.population[identifier].random_cache)
        result = super().update_particle(identifier)
        self.generations[self.iteration].processed.append(
            (identifier, cache, deepcopy(result))
        )
        return result

    @override
    def _sample_candidate(self, particle: EGNAParticle) -> dict[str, float]:
        before = deepcopy(self.population)
        parents, order = self._parents, self._topological_order
        conditionals = deepcopy(self._conditionals)
        cache = deepcopy(particle.random_cache)
        result = super()._sample_candidate(particle)
        self.generations[self.iteration].samples.append(
            Sample(
                particle.identifier,
                before,
                parents,
                order,
                conditionals,
                cache,
                result.copy(),
            )
        )
        return result

    @override
    def post_iteration(self, actual_iter: int) -> None:
        g = self.generations[actual_iter]
        g.before_post = deepcopy(self.population)
        super().post_iteration(actual_iter)
        g.after = deepcopy(self.population)


def run_egna(
    algorithm: ObservedEGNA,
    bounds: dict[str, tuple[float, float]],
    *,
    objective: Callable[..., float] = sphere,
    iterations: int = 2,
    seed: int = BASE_SEED,
) -> ObservedEGNA:
    processor = Serial()
    # Shared factory erases the generic particle parameter in AlgorithmBase.
    factory = cast(Callable[..., Optimizer], factories.make_optimizer)
    _ = factory(
        Continuous(bounds, cost_function=objective, use_cache=False),
        algorithm,
        processor=processor,
        n_particles=8,
        n_iterations=iterations,
        seed=seed,
        fitness_failure_strategy="raise",
    ).fit()
    (executor,) = processor.processors_pool.values()
    executed = cast(object, vars(executor)["_algorithm"])
    assert isinstance(executed, ObservedEGNA) and executed is not algorithm
    assert set(executed.generations) == set(range(iterations + 1))
    for g in executed.generations.values():
        assert set(g.after) == set(g.before) and len(g.after) == 8
        for p in g.after.values():
            assert_variables_within_bounds(p.variables, bounds)
            assert p.fitness == pytest.approx(
                objective(**p.variables), rel=STRICT_RTOL, abs=STRICT_ATOL
            )
            assert p.candidate_variables is None and p.candidate_fitness is None
    return executed


@pytest.fixture
def bounds() -> dict[str, tuple[float, float]]:
    return {"x": (-7.0, 5.0), "y": (20.0, 80.0)}


@pytest.fixture
def trajectory(bounds: dict[str, tuple[float, float]]) -> ObservedEGNA:
    return run_egna(ObservedEGNA(selection_fraction=0.6), bounds)


def assert_selection(a: ObservedEGNA, bounds: dict[str, tuple[float, float]]) -> None:
    lower = np.array([b[0] for b in bounds.values()])
    spans = np.array([b[1] - b[0] for b in bounds.values()])
    for t, g in a.generations.items():
        if t == 0:
            continue
        previous = a.generations[t - 1].after
        assert [(p.variables, p.fitness) for p in g.before.values()] == [
            (p.variables, p.fitness) for p in previous.values()
        ]
        count = floor(a.selection_fraction * len(previous))
        assert count > len(bounds)
        assert g.selected.shape == (count, len(bounds))
        remaining = list(previous.values())
        fitness: list[float] = []
        for index in range(len(g.selected)):
            # NumPy iteration/indexing erases the concrete float64 dtype.
            row = cast(Array, g.selected[index])
            matches = [
                p
                for p in remaining
                if np.allclose(
                    row,
                    (np.array(list(p.variables.values())) - lower) / spans,
                    rtol=STRICT_RTOL,
                    atol=STRICT_ATOL,
                )
            ]
            assert matches
            p = matches[0]
            remaining.remove(p)
            fitness.append(float(p.fitness))
        assert fitness == sorted(float(p.fitness) for p in previous.values())[:count]
        assert np.all(np.isfinite(g.selected))


def test_setup_affine_normalization_and_truncation_selection(
    trajectory: ObservedEGNA,
    bounds: dict[str, tuple[float, float]],
) -> None:
    a = trajectory
    g = a.generations[0]
    assert a.names == tuple(bounds)
    assert a.lower is not None and a.spans is not None
    np.testing.assert_array_equal(a.lower, [b[0] for b in bounds.values()])
    np.testing.assert_array_equal(a.spans, [b[1] - b[0] for b in bounds.values()])
    assert g.parents == g.order == g.conditionals == ()
    assert not g.active and not g.samples and g.fits == 0
    assert all(p.fitness == FITNESS_UNDEFINED for p in g.before.values())
    assert len(g.processed) == len(g.before)
    for identifier, cache, p in g.processed:
        assert cache == p.random_cache == {}
        assert p.candidate_variables == g.before[identifier].variables
        assert p.candidate_fitness == pytest.approx(sphere(**p.variables))
    assert len(g.caches) == 1
    initialize, caches = g.caches[0]
    assert initialize and set(caches) == set(g.before)
    assert all(cache == {} for cache in caches.values())
    assert_selection(a, bounds)


def bic(data: Array, child: int, parents: tuple[int, ...]) -> float:
    """Gaussian ML via an orthogonal projection, then k=|Pa|+2 BIC penalty."""
    n = len(data)
    design = cast(Array, np.column_stack((np.ones(n), data[:, parents])))
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return -np.inf
    q, _ = np.linalg.qr(design, mode="reduced")
    residual = data[:, child] - q @ (q.T @ data[:, child])
    variance = max(float(residual @ residual / n), np.finfo(float).eps)
    return -n / 2 * (log(2 * pi * variance) + 1) - (len(parents) + 2) / 2 * log(n)


def bge_scales(data: Array, alpha_mu: float, alpha_w: float) -> tuple[Array, Array]:
    n, d = data.shape
    prior = np.eye(d) * (alpha_mu * (alpha_w - d - 1) / (alpha_mu + 1))
    mean = cast(Array, data.mean(axis=0))
    scatter = (data - mean).T @ (data - mean)
    posterior = (
        prior
        + scatter
        + n * alpha_mu / (n + alpha_mu) * np.outer(mean - 0.5, mean - 0.5)
    )
    return prior, posterior


def bge(
    data: Array, child: int, parents: tuple[int, ...], alpha_mu: float, alpha_w: float
) -> float:
    """Log evidence(family) - log evidence(parents), integrating normal-Wishart.

    Multivariate-gamma ratios telescope to the local univariate-gamma formula;
    this independently checks all determinant exponents, including empty sets.
    """
    n, d = data.shape
    prior, posterior = bge_scales(data, alpha_mu, alpha_w)

    def evidence(indices: tuple[int, ...]) -> float:
        size = len(indices)
        if size == 0:
            return 0.0
        degrees = alpha_w - d + size
        gamma_ratio = sum(
            lgamma((degrees + n - j) / 2) - lgamma((degrees - j) / 2)
            for j in range(size)
        )
        t = float(cast(np.float64, np.linalg.det(prior[np.ix_(indices, indices)])))
        r = float(cast(np.float64, np.linalg.det(posterior[np.ix_(indices, indices)])))
        return (
            size / 2 * log(alpha_mu / (alpha_mu + n))
            - n * size / 2 * log(pi)
            + gamma_ratio
            + degrees / 2 * log(t)
            - (degrees + n) / 2 * log(r)
        )

    return evidence((*parents, child)) - evidence(parents)


def assert_dag(g: Generation) -> None:
    assert_valid_permutation(g.order, list(range(len(g.parents))))
    for child, parents in enumerate(g.parents):
        assert all(g.order.index(parent) < g.order.index(child) for parent in parents)
    # Choosing the smallest currently available node is a Tyrannis convention.
    for i, node in enumerate(g.order):
        available = [v for v in g.order[i:] if set(g.parents[v]) <= set(g.order[:i])]
        assert node == min(available)


def assert_two_node_search(
    g: Generation, score: Callable[[int, tuple[int, ...]], float]
) -> None:
    """There are exactly three two-node DAGs; no greedy-search replica is needed."""
    empty: Parents = ((), ())
    alternatives: tuple[Parents, ...] = (((1,), ()), ((), (0,)))
    start = g.previous or empty
    # The first score requests reveal the actual starting graph without RNG replay.
    assert list(g.scores)[:2] == [(i, start[i]) for i in range(2)]
    assert_dag(g)
    for (child, parents), observed in g.scores.items():
        assert observed == pytest.approx(
            score(child, parents), rel=LINALG_RTOL, abs=LINALG_ATOL
        )
    if start == empty:
        deltas = [score(i, (1 - i,)) - score(i, ()) for i in range(2)]
        tolerances = [
            np.finfo(float).eps
            * 100
            * max(1.0, abs(score(i, ())), abs(score(i, (1 - i,))))
            for i in range(2)
        ]
        if all(delta <= tol for delta, tol in zip(deltas, tolerances, strict=True)):
            assert g.parents == empty
        else:
            assert g.parents in alternatives
            child = next(i for i in range(2) if g.parents[i])
            assert deltas[child] > tolerances[child]
            assert deltas[child] == pytest.approx(
                max(deltas), rel=LINALG_RTOL, abs=LINALG_ATOL
            )
            # Reverse addition would form a cycle; no two-parent reciprocal graph.
            assert sum(map(len, g.parents)) == 1
    else:
        child = next(i for i in range(2) if start[i])
        current, candidate = score(child, start[child]), score(child, ())
        tolerance = np.finfo(float).eps * 100 * max(1.0, abs(current), abs(candidate))
        delta = candidate - current
        assert g.parents == (empty if delta > tolerance else start)


def test_bic_local_scores_and_first_forward_search(
    bounds: dict[str, tuple[float, float]],
) -> None:
    learned: set[bool] = set()
    # Case 4 has an improving dependency; case 0 rejects both initial additions.
    for case in (4, 0):
        a = run_egna(
            ObservedEGNA(selection_fraction=0.6),
            bounds,
            iterations=1,
            seed=seed_for(case),
        )
        g = a.generations[1]
        assert g.previous == ()
        assert set(g.scores) == {(0, ()), (1, ()), (0, (1,)), (1, (0,))}
        assert_two_node_search(g, partial(bic, g.selected))
        learned.add(any(g.parents))
    assert learned == {False, True}


def test_bic_reuses_previous_structure_and_accepts_improving_deletion(
    bounds: dict[str, tuple[float, float]],
) -> None:
    a = run_egna(
        ObservedEGNA(selection_fraction=0.6), bounds, iterations=3, seed=seed_for(11)
    )
    additions = deletions = 0
    for t in range(1, 4):
        g = a.generations[t]
        assert g.previous == a.generations[t - 1].parents
        assert_two_node_search(g, partial(bic, g.selected))
        old, new = sum(map(len, g.previous)), sum(map(len, g.parents))
        additions += new > old
        deletions += new < old
    assert additions and deletions
    assert_selection(a, bounds)


def test_bge_default_and_explicit_priors_local_evidence_and_search(
    bounds: dict[str, tuple[float, float]],
) -> None:
    scores: list[float] = []
    selected: list[Array] = []
    for alpha_mu, configured_w in ((1.0, None), (1.0, 6.5), (2.5, 6.5)):
        a = run_egna(
            ObservedEGNA(
                selection_fraction=0.75,
                structure_learning="bge",
                bge_alpha_mu=alpha_mu,
                bge_alpha_w=configured_w,
            ),
            bounds,
            iterations=1,
            seed=seed_for(0),
        )
        g = a.generations[1]
        alpha_w = float(len(bounds) + 2) if configured_w is None else configured_w
        context = g.context
        assert context is not None
        assert context.alpha_w == alpha_w and context.alpha_mu == alpha_mu
        assert context.sample_count == len(g.selected) and context.dimension == 2
        prior, posterior = bge_scales(g.selected, alpha_mu, alpha_w)
        # Production records have unparameterized ndarray fields.
        np.testing.assert_allclose(
            cast(Array, context.prior_scale), prior, rtol=LINALG_RTOL, atol=LINALG_ATOL
        )
        np.testing.assert_allclose(
            cast(Array, context.posterior_scale),
            posterior,
            rtol=LINALG_RTOL,
            atol=LINALG_ATOL,
        )
        score = partial(bge, g.selected, alpha_mu=alpha_mu, alpha_w=alpha_w)
        assert_two_node_search(g, score)
        assert any(g.parents)
        assert score(0, ()) + score(1, (0,)) == pytest.approx(
            score(1, ()) + score(0, (1,)), rel=LINALG_RTOL, abs=LINALG_ATOL
        )
        scores.append(score(0, ()))
        selected.append(g.selected)
    for data in selected[1:]:
        np.testing.assert_array_equal(data, selected[0])
    assert len(set(scores)) == 3


def test_edge_exclusion_partial_correlations_and_finite_sample_correction(
    bounds: dict[str, tuple[float, float]],
) -> None:
    bounds = dict(bounds, z=(-2.0, 6.0))
    a = run_egna(
        ObservedEGNA(
            selection_fraction=0.75,
            structure_learning="edge-exclusion",
            edge_exclusion_alpha=0.2,
        ),
        bounds,
        iterations=1,
        seed=seed_for(0),
    )
    g = a.generations[1]
    n, d = g.selected.shape
    covariance = cast(Array, np.cov(g.selected, rowvar=False, bias=True))
    minimum = (
        np.finfo(float).eps
        * d
        * max(1.0, float(cast(np.float64, np.max(np.abs(np.diag(covariance))))))
    )
    # This ordinary trajectory needs no precision stabilization.
    assert np.min(np.linalg.eigvalsh(covariance)) > minimum
    precision = np.linalg.solve(covariance, np.eye(d))
    assert g.precision is not None
    np.testing.assert_allclose(
        g.precision, precision, rtol=LINALG_RTOL, atol=LINALG_ATOL
    )
    expected: set[tuple[int, int]] = set()
    correction_changes_decision = False
    conditional_differs_from_marginal = False
    normal = NormalDist()
    # NumPy scalar indexing returns Any; tolist has the known real matrix shape.
    omega = cast(list[list[float]], precision.tolist())
    cov = cast(list[list[float]], covariance.tolist())
    for i, j in ((0, 1), (0, 2), (1, 2)):
        r = -omega[i][j] / sqrt(omega[i][i] * omega[j][j])
        r = min(1 - float(np.finfo(float).eps), max(-1 + float(np.finfo(float).eps), r))
        statistic = -n * log(1 - r * r)
        # Chi-square(1) is Z^2: independent of scipy.stats.chi2 in production.
        cdf = 2 * normal.cdf(sqrt(statistic)) - 1
        # T*f_chi2(T) has this continuous expression even at T=0.
        correction = (
            (2 * d + 1) / (2 * n) * sqrt(statistic / (2 * pi)) * exp(-statistic / 2)
        )
        p = 1 - min(1.0, max(0.0, cdf - correction))
        if p < a.edge_exclusion_alpha:
            expected.add((i, j))
        correction_changes_decision |= (1 - cdf < a.edge_exclusion_alpha) != (
            p < a.edge_exclusion_alpha
        )
        marginal = cov[i][j] / sqrt(cov[i][i] * cov[j][j])
        conditional_differs_from_marginal |= abs(r - marginal) > 0.1
    assert 0 < len(expected) < 3
    assert correction_changes_decision and conditional_differs_from_marginal
    assert {
        (parent, child) for child, parents in enumerate(g.parents) for parent in parents
    } == expected
    assert g.order == (0, 1, 2)  # Stable encoded order, not a universal GBN property.
    assert_dag(g)
    assert_conditionals(g)  # Also checks a child with two simultaneous parents.
    assert_selection(a, bounds)


def conditional(
    data: Array, child: int, parents: tuple[int, ...]
) -> tuple[float, Array, Array, float]:
    """Centered least squares via normal equations on these well-conditioned data."""
    means = cast(Array, np.mean(data, axis=0))
    mean = float(cast(np.float64, means[child]))
    y = cast(Array, data[:, child]) - mean
    parent_means = means[list(parents)]
    if parents:
        x = data[:, parents] - parent_means
        beta = np.linalg.solve(x.T @ x, x.T @ y)
        residual = y - x @ beta
    else:
        beta = np.empty(0)
        residual = y
    sd = sqrt(float(residual @ residual / len(data)))
    return mean, parent_means, beta, sd


def assert_conditionals(g: Generation) -> None:
    assert len(g.conditionals) == len(g.parents)
    assert_dag(g)
    for child, observed in enumerate(g.conditionals):
        mean, parent_means, beta, sd = conditional(g.selected, child, g.parents[child])
        assert (
            observed.variable_index == child
            and observed.parent_indices == g.parents[child]
        )
        assert observed.mean == pytest.approx(mean, rel=LINALG_RTOL, abs=LINALG_ATOL)
        np.testing.assert_allclose(
            cast(Array, observed.parent_means),
            parent_means,
            rtol=LINALG_RTOL,
            atol=LINALG_ATOL,
        )
        np.testing.assert_allclose(
            cast(Array, observed.coefficients), beta, rtol=LINALG_RTOL, atol=LINALG_ATOL
        )
        expected_sd = 0.0 if sd <= sqrt(np.finfo(float).eps) else sd
        assert observed.standard_deviation == pytest.approx(
            expected_sd, rel=LINALG_RTOL, abs=LINALG_ATOL
        )
        assert observed.standard_deviation**2 == pytest.approx(
            expected_sd**2, rel=LINALG_RTOL, abs=LINALG_ATOL
        )


def test_gaussian_roots_children_and_topological_order(
    trajectory: ObservedEGNA,
) -> None:
    roots = children = 0
    for t, g in trajectory.generations.items():
        if t:
            assert_conditionals(g)
            roots += sum(not parents for parents in g.parents)
            children += sum(bool(parents) for parents in g.parents)
    assert roots and children


def truncated_quantile(mean: float, sd: float, u: float) -> float:
    if sd <= sqrt(np.finfo(float).eps):
        return min(1.0, max(0.0, mean))
    normal = NormalDist(mean, sd)
    lower, upper = normal.cdf(0.0), normal.cdf(1.0)
    return normal.inv_cdf(lower + u * (upper - lower))


def test_cached_uniforms_exact_ancestral_sampling_and_truncation(
    bounds: dict[str, tuple[float, float]],
) -> None:
    # This small real trajectory includes substantial mass outside the box.
    trajectory = run_egna(
        ObservedEGNA(selection_fraction=0.6), bounds, seed=seed_for(3)
    )
    lower = np.array([b[0] for b in bounds.values()])
    spans = np.array([b[1] - b[0] for b in bounds.values()])
    distinguishes_clipping = distinguishes_old_parent = distinguishes_mean = (
        distinguishes_other
    ) = False
    for t, g in trajectory.generations.items():
        if not t:
            continue
        assert_conditionals(g)
        assert len(g.caches) == 1
        initialize, caches = g.caches[0]
        assert not initialize and set(caches) == set(g.before)
        processed = {identifier: (cache, p) for identifier, cache, p in g.processed}
        for sample in g.samples:
            assert set(sample.cache) == {"variable-uniforms"}
            uniforms = cast(
                Array, np.asarray(sample.cache["variable-uniforms"], dtype=float)
            )
            assert uniforms.shape == (len(bounds),)
            assert np.all((0 <= uniforms) & (uniforms < 1))
            cache, p = processed[sample.identifier]
            assert set(cache) == set(p.random_cache) == set(sample.cache)
            for saved in (cache, p.random_cache, caches[sample.identifier]):
                np.testing.assert_array_equal(
                    uniforms, np.asarray(saved["variable-uniforms"], dtype=float)
                )
            expected: Array = np.full(len(bounds), np.nan)
            for child in g.order:
                parents = g.parents[child]
                mean, parent_means, beta, sd = conditional(g.selected, child, parents)
                mu = mean + float(beta @ (expected[list(parents)] - parent_means))
                u = float(cast(np.float64, uniforms[child]))
                value = truncated_quantile(mu, sd, u)
                expected[child] = value
                if sd > sqrt(np.finfo(float).eps):
                    unbounded = NormalDist(mu, sd)
                    outside = unbounded.cdf(0.0) + 1 - unbounded.cdf(1.0)
                    clipped = min(1.0, max(0.0, unbounded.inv_cdf(u)))
                    distinguishes_clipping |= (
                        outside > 0.05 and abs(value - clipped) > 0.01
                    )
                if parents:
                    old = (
                        np.array(list(g.before[sample.identifier].variables.values()))
                        - lower
                    ) / spans
                    other = next(
                        s for s in g.samples if s.identifier != sample.identifier
                    )
                    other_values = (
                        np.array(list(other.values.values())) - lower
                    ) / spans
                    wrong_old = mean + float(beta @ (old[list(parents)] - parent_means))
                    wrong_other = mean + float(
                        beta @ (other_values[list(parents)] - parent_means)
                    )
                    distinguishes_old_parent |= (
                        abs(value - truncated_quantile(wrong_old, sd, u)) > 0.001
                    )
                    distinguishes_mean |= (
                        abs(value - truncated_quantile(mean, sd, u)) > 0.001
                    )
                    distinguishes_other |= (
                        abs(value - truncated_quantile(wrong_other, sd, u)) > 0.001
                    )
            actual = np.array(list(sample.values.values()))
            np.testing.assert_allclose(
                (actual - lower) / spans, expected, rtol=LINALG_RTOL, atol=LINALG_ATOL
            )
            np.testing.assert_allclose(
                actual, lower + spans * expected, rtol=LINALG_RTOL, atol=LINALG_ATOL
            )
            assert p.candidate_variables == sample.values
            assert p.candidate_fitness == pytest.approx(
                sphere(**sample.values), rel=STRICT_RTOL, abs=STRICT_ATOL
            )
    assert distinguishes_clipping
    assert distinguishes_old_parent and distinguishes_mean and distinguishes_other


def test_affine_model_invariance(bounds: dict[str, tuple[float, float]]) -> None:
    unit_bounds = {name: (0.0, 1.0) for name in bounds}
    runs = [
        run_egna(
            ObservedEGNA(selection_fraction=0.75, structure_learning="bge"),
            box,
            objective=constant_objective,
            iterations=1,
        )
        for box in (unit_bounds, bounds)
    ]
    unit, scaled = [a.generations[1] for a in runs]
    np.testing.assert_allclose(
        unit.selected, scaled.selected, rtol=STRICT_RTOL, atol=STRICT_ATOL
    )
    assert unit.parents == scaled.parents
    for key in unit.scores:
        assert unit.scores[key] == pytest.approx(
            scaled.scores[key], rel=LINALG_RTOL, abs=LINALG_ATOL
        )
    for a, box in zip(runs, (unit_bounds, bounds), strict=True):
        assert_selection(a, box)
        assert_conditionals(a.generations[1])
    lower = np.array([b[0] for b in bounds.values()])
    spans = np.array([b[1] - b[0] for b in bounds.values()])
    for u, s in zip(unit.samples, scaled.samples, strict=True):
        np.testing.assert_allclose(
            np.array(list(u.values.values())),
            (np.array(list(s.values.values())) - lower) / spans,
            rtol=LINALG_RTOL,
            atol=LINALG_ATOL,
        )


def test_one_synchronous_model_precedes_the_entire_sampling_sweep(
    trajectory: ObservedEGNA,
) -> None:
    for t, g in trajectory.generations.items():
        if not t:
            continue
        assert g.fits == 1
        assert g.active == set(g.before)
        assert_valid_permutation([s.identifier for s in g.samples], list(g.before))
        assert_conditionals(g)
        for index, sample in enumerate(g.samples):
            assert sample.parents == g.parents and sample.order == g.order
            for actual, frozen in zip(sample.conditionals, g.conditionals, strict=True):
                assert actual.mean == frozen.mean
                assert actual.standard_deviation == frozen.standard_deviation
                assert actual.parent_indices == frozen.parent_indices
                np.testing.assert_array_equal(
                    cast(Array, actual.coefficients), cast(Array, frozen.coefficients)
                )
                np.testing.assert_array_equal(
                    cast(Array, actual.parent_means), cast(Array, frozen.parent_means)
                )
            # Earlier candidates already exist during Serial's sweep, but neither
            # they nor their fitness have entered the training population/model.
            assert (
                sum(p.candidate_variables is not None for p in sample.before.values())
                == index
            )
            for identifier, p in sample.before.items():
                assert p.variables == g.before[identifier].variables
                assert p.fitness == g.before[identifier].fitness


def test_full_generational_replacement_including_worse_candidates(
    trajectory: ObservedEGNA,
) -> None:
    worse = 0
    for t, g in trajectory.generations.items():
        if not t:
            continue
        assert set(g.before) == set(g.before_post) == set(g.after) == g.active
        assert_valid_permutation(
            [identifier for identifier, _, _ in g.processed], list(g.before)
        )
        for sample in g.samples:
            identifier = sample.identifier
            pending = g.before_post[identifier]
            after = g.after[identifier]
            assert pending.candidate_variables == sample.values
            assert after.variables == pending.candidate_variables
            assert after.fitness == pending.candidate_fitness
            worse += after.fitness > g.before[identifier].fitness
    assert worse > 0  # An elitist min(parent, offspring) policy would fail here.


def signed_plateaus(x: float, y: float) -> float:
    """Ordered infinite regions exercise selection without nonfinite coordinates."""
    if x < -0.3:
        return -np.inf
    if x > 0.3:
        return np.inf
    return sphere(x=x, y=y)


def test_infinite_fitness_ordering_and_ties_keep_finite_models() -> None:
    bounds = {"x": (-1.0, 1.0), "y": (-2.0, 2.0)}
    a = run_egna(
        ObservedEGNA(selection_fraction=0.75),
        bounds,
        objective=signed_plateaus,
        seed=seed_for(0),
    )
    fitness = [float(p.fitness) for p in a.generations[0].after.values()]
    assert -np.inf in fitness and np.inf in fitness and any(np.isfinite(fitness))
    assert FITNESS_UNDEFINED == np.inf and -np.inf < min(
        f for f in fitness if np.isfinite(f)
    )
    assert_selection(a, bounds)  # Matches ordered fitness, never tied identifiers.
    tied = run_egna(
        ObservedEGNA(selection_fraction=0.6),
        bounds,
        objective=constant_objective,
        iterations=1,
    )
    assert_selection(tied, bounds)
    for run in (a, tied):
        for t, g in run.generations.items():
            if t:
                assert_conditionals(g)
                assert np.all(np.isfinite(g.selected))
                assert all(np.isfinite(score) for score in g.scores.values())
                assert all(
                    np.all(np.isfinite(list(s.values.values()))) for s in g.samples
                )
