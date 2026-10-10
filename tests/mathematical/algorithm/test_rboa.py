"""rBOA mathematics through Optimizer + Continuous + the real Serial replica.

Ahn et al. (2004), doi:10.1007/978-3-540-24854-5_86, eqs. (4)-(7):
https://www.cs.york.ac.uk/rts/docs/GECCO_2004/Conference%20proceedings/papers/3102/31020840.pdf
Bosman & Thierens (2001), sections 3-5, mixture/factorization metrics and BEND:
https://homepages.cwi.nl/~bosman/publications/2001_advancingcontinuousideas.pdf
K=1 is Ahn's experimental simplification, not a restriction of mixture-BIC.
Tyrannis adds affine [0,1] modeling, covariance floors, bounded truncated-normal
sampling of Gaussian conditionals, q>=n renewal and a scalar random-cache
lifecycle. Iteration 0 is setup; iteration t>0 models parents P_(t-1), samples
offspring O_t, then renews P_t. No production helper supplies an oracle.
"""

from collections import Counter
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from math import floor, log, pi
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
from tyrannis.algorithm.rboa import RBOA, RBOAParticle

# The probability model is private and has no public observational equivalent.
from tyrannis.algorithm.rboa import (
    _SubproblemModel as SubproblemModel,  # pyright: ignore[reportPrivateUsage]
)
from tyrannis.core.algorithm import FITNESS_UNDEFINED
from tyrannis.processor.serial import Serial
from tyrannis.space.continuous import Continuous

Array = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
Indices = np.ndarray[tuple[int, ...], np.dtype[np.int_]]
Population = dict[str, RBOAParticle]
Parents = tuple[tuple[int, ...], ...]


@dataclass
class Sample:
    identifier: str
    before: Population
    models: tuple[SubproblemModel, ...]
    parents: Parents
    cache: dict[str, object]
    values: dict[str, float]


@dataclass
class Generation:
    before: Population
    selected: Array = field(default_factory=lambda: np.empty((0, 2)))
    kmeans: tuple[Indices, ...] = ()
    scores: dict[tuple[int, tuple[int, ...]], float] = field(default_factory=dict)
    parents: Parents = ()
    order: tuple[int, ...] = ()
    limit: int = 0
    active: set[str] = field(default_factory=set)
    temporary: set[str] = field(default_factory=set)
    q: int = 0
    before_post: Population = field(default_factory=dict)
    after: Population = field(default_factory=dict)
    models: tuple[SubproblemModel, ...] = ()
    bend: list[tuple[Array, list[Indices]]] = field(default_factory=list)
    covariance: list[tuple[Array, Array]] = field(default_factory=list)
    samples: list[Sample] = field(default_factory=list)
    processed: list[tuple[bool, str, dict[str, object], RBOAParticle]] = field(
        default_factory=list
    )
    caches: list[tuple[bool, dict[str, dict[str, object]]]] = field(
        default_factory=list
    )


class ObservedRBOA(RBOA):
    """Only copy observations; every overridden hook delegates to super()."""

    # Production's bare ndarray annotations erase dtype; specialize only the
    # observation boundary. Casts below likewise recover actual array dtypes.
    _lower_bounds: Array | None
    _spans: Array | None

    @override
    def pre_iteration(self, actual_iter: int) -> None:
        self.iteration: int = actual_iter
        if actual_iter == 0:
            self.generations: dict[int, Generation] = {}
            self.initialized: dict[tuple[int, str], RBOAParticle] = {}
        self.generations[actual_iter] = Generation(deepcopy(self.population))
        super().pre_iteration(actual_iter)
        g = self.generations[actual_iter]
        g.parents, g.order = self._parents, self._topological_order
        g.limit = self._effective_max_parents
        g.active = self._active_parent_ids.copy()
        g.temporary = self._temporary_particle_ids.copy()
        g.q = self._iteration_offspring_size
        g.models = deepcopy(self._subproblem_models)
        if actual_iter == 0:
            self.names: tuple[str, ...] = self._variable_names
            self.lower: Array | None = deepcopy(self._lower_bounds)
            self.spans: Array | None = deepcopy(self._spans)
            self.parent_ids: list[str] = self._parent_particle_ids.copy()

    @override
    def _selected_matrix(self) -> Array:
        result = cast(Callable[[], Array], super()._selected_matrix)()
        self.generations[self.iteration].selected = result.copy()
        return result

    @override
    def _kmeans_clusters(self, data: Array) -> tuple[Indices, ...]:
        result = cast(Callable[[Array], tuple[Indices, ...]], super()._kmeans_clusters)(
            data
        )
        self.generations[self.iteration].kmeans = deepcopy(result)
        return result

    @override
    def _local_bic_score(
        self,
        data: Array,
        mixture_clusters: tuple[Indices, ...],
        child: int,
        parents: tuple[int, ...],
    ) -> float:
        score = cast(
            Callable[[Array, tuple[Indices, ...], int, tuple[int, ...]], float],
            super()._local_bic_score,
        )
        result = score(data, mixture_clusters, child, parents)
        self.generations[self.iteration].scores[child, parents] = result
        return result

    @override
    def initialize_particle(self, identifier: str) -> RBOAParticle:
        cache = self.population[identifier].random_cache.copy()
        particle = super().initialize_particle(identifier)
        self.initialized[self.iteration, identifier] = deepcopy(particle)
        self.generations[self.iteration].processed.append(
            (True, identifier, cache, deepcopy(particle))
        )
        return particle

    @override
    def update_particle(self, identifier: str) -> RBOAParticle:
        cache = self.population[identifier].random_cache.copy()
        particle = super().update_particle(identifier)
        self.generations[self.iteration].processed.append(
            (False, identifier, cache, deepcopy(particle))
        )
        return particle

    @override
    def create_random_cache(self, particle_ids: list[str], initialize: bool) -> None:
        super().create_random_cache(particle_ids, initialize)
        self.generations[self.iteration].caches.append(
            (
                initialize,
                {key: self.population[key].random_cache.copy() for key in particle_ids},
            )
        )

    @override
    def _bend_clusters(self, data: Array) -> list[Indices]:
        result = cast(Callable[[Array], list[Indices]], super()._bend_clusters)(data)
        self.generations[self.iteration].bend.append((data.copy(), deepcopy(result)))
        return result

    @override
    def _regularize_covariance(self, covariance: Array) -> Array:
        before = covariance.copy()
        result = cast(Callable[[Array], Array], super()._regularize_covariance)(
            covariance
        )
        self.generations[self.iteration].covariance.append((before, result.copy()))
        return result

    @override
    def _sample_offspring(self, particle: RBOAParticle) -> dict[str, float]:
        before = deepcopy(self.population)
        models = deepcopy(self._subproblem_models)
        parents = self._parents
        cache = particle.random_cache.copy()
        result = super()._sample_offspring(particle)
        self.generations[self.iteration].samples.append(
            Sample(
                particle.identifier,
                before,
                models,
                parents,
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


def run_rboa(
    algorithm: ObservedRBOA,
    bounds: dict[str, tuple[float, float]],
    *,
    objective: Callable[..., float] = sphere,
    n: int = 8,
    iterations: int = 2,
    seed: int = BASE_SEED,
) -> ObservedRBOA:
    processor = Serial()
    # The shared factory erases AlgorithmBase's generic particle parameter.
    factory = cast(Callable[..., Optimizer], factories.make_optimizer)
    _ = factory(
        Continuous(bounds, cost_function=objective, use_cache=False),
        algorithm,
        processor=processor,
        n_particles=n,
        n_iterations=iterations,
        seed=seed,
        fitness_failure_strategy="raise",
    ).fit()
    (executor,) = processor.processors_pool.values()
    # Serial's public pool exposes replicas, but its algorithm has no accessor.
    executed = cast(object, vars(executor)["_algorithm"])
    assert isinstance(executed, ObservedRBOA) and executed is not algorithm
    assert set(executed.generations) == set(range(iterations + 1))
    for g in executed.generations.values():
        assert len(g.after) == n
        for p in g.after.values():
            assert_variables_within_bounds(p.variables, bounds)
            assert p.fitness == pytest.approx(
                objective(**p.variables), rel=STRICT_RTOL, abs=STRICT_ATOL
            )
            assert p.candidate_variables is None and p.candidate_fitness is None
    return executed


@pytest.fixture
def bounds() -> dict[str, tuple[float, float]]:
    # Offset and unequal spans are part of the affine-model contract.
    return {"x": (-7.0, 5.0), "y": (20.0, 80.0)}


@pytest.fixture
def trajectory(bounds: dict[str, tuple[float, float]]) -> ObservedRBOA:
    return run_rboa(ObservedRBOA(selection_fraction=0.6), bounds)


def assert_selection(a: ObservedRBOA, bounds: dict[str, tuple[float, float]]) -> None:
    lower = np.array([b[0] for b in bounds.values()])
    spans = np.array([b[1] - b[0] for b in bounds.values()])
    for t, g in a.generations.items():
        if t == 0:
            continue
        previous = a.generations[t - 1].after
        assert [(p.variables, p.fitness) for p in g.before.values()] == [
            (p.variables, p.fitness) for p in previous.values()
        ]
        selected_count = floor(a.selection_fraction * len(previous))
        assert g.selected.shape == (selected_count, len(bounds))
        # Match selected rows to real parents without imposing an order on ties.
        remaining = list(previous.values())
        fitness: list[float] = []
        for index in range(len(g.selected)):
            # NumPy integer/fancy indexing loses shape typing at this boundary.
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
        assert (
            fitness
            == sorted(float(p.fitness) for p in previous.values())[:selected_count]
        )
        assert np.all(np.isfinite(g.selected))
        assert np.all((g.selected >= 0) & (g.selected <= 1))


def test_setup_affine_normalization_and_truncation_selection(
    trajectory: ObservedRBOA,
    bounds: dict[str, tuple[float, float]],
) -> None:
    a = trajectory
    setup = a.generations[0]
    assert a.names == tuple(bounds)
    assert a.lower is not None and a.spans is not None
    np.testing.assert_array_equal(a.lower, [-7.0, 20.0])
    np.testing.assert_array_equal(a.spans, [12.0, 60.0])
    assert a.parent_ids == list(setup.before)
    assert not setup.active and not setup.temporary and setup.q == 0
    assert not setup.parents and not setup.order and not setup.scores
    assert not setup.models
    for key, p in setup.after.items():
        assert setup.before[key].fitness == FITNESS_UNDEFINED == np.inf
        assert p.variables == setup.before[key].variables
        assert a.initialized[0, key].candidate_variables == p.variables
        assert a.initialized[0, key].candidate_fitness == p.fitness
        assert p.random_cache == {}
    assert_selection(a, bounds)


def bic(
    data: Array,
    clusters: tuple[Indices, ...],
    child: int,
    parents: tuple[int, ...],
    penalty: float,
    variance_floor: float,
) -> float:
    """Ahn eq. (4); centered scalar regression suffices for these 2D cases."""
    entropy = 0.0
    for indices in clusters:
        component = cast(Array, data[indices])
        mean = cast(Array, component.mean(axis=0))
        centered: Array = component - mean
        residual: Array = centered[:, child]
        if parents:
            (parent,) = parents
            x: Array = centered[:, parent]
            residual = residual - x * float(x @ residual) / float(x @ x)
        variance = max(float(residual @ residual) / len(indices), variance_floor)
        entropy += len(indices) * 0.5 * log(2 * pi * np.e * variance)
    return entropy + len(clusters) * penalty * log(len(data)) * (len(parents) + 2)


def test_local_bic_greedy_edge_and_parent_limit(
    bounds: dict[str, tuple[float, float]],
) -> None:
    for limit in (0, 1):
        a = run_rboa(ObservedRBOA(bic_penalty=0, max_parents=limit), bounds)
        for t, g in a.generations.items():
            if t == 0:
                continue
            assert len(g.kmeans) == 1
            np.testing.assert_array_equal(g.kmeans[0], np.arange(len(g.selected)))
            for (child, parents), score in g.scores.items():
                assert score == pytest.approx(
                    bic(
                        g.selected,
                        g.kmeans,
                        child,
                        parents,
                        0,
                        a.covariance_floor,
                    ),
                    rel=STRICT_RTOL,
                    abs=STRICT_ATOL,
                )
            assert_valid_permutation(g.order, [0, 1])
            edges = [(p, c) for c, ps in enumerate(g.parents) for p in ps]
            assert all(len(ps) <= limit for ps in g.parents)
            assert all(g.order.index(p) < g.order.index(c) for p, c in edges)
            if limit == 0:
                assert g.parents == ((), ())
            else:
                deltas = {
                    c: bic(g.selected, g.kmeans, c, (1 - c,), 0, a.covariance_floor)
                    - bic(g.selected, g.kmeans, c, (), 0, a.covariance_floor)
                    for c in (0, 1)
                }
                assert min(deltas.values()) < -STRICT_ATOL
                assert len(edges) == 1  # A second edge in 2D would create a cycle.
                # Equivalent Gaussian directions can tie; no rounding-based tie oracle.
                assert deltas[edges[0][1]] == pytest.approx(
                    min(deltas.values()),
                    rel=STRICT_RTOL,
                    abs=STRICT_ATOL,
                )


@pytest.mark.parametrize(
    ("n", "case", "limit"),
    [(12, 0, 1), (8, 1, 0)],
    ids=["regressions-identifiable", "smallest-cluster-limits-parents"],
)
def test_two_component_bic_and_identifiable_parent_limit(
    bounds: dict[str, tuple[float, float]],
    n: int,
    case: int,
    limit: int,
) -> None:
    # Two real K-means components need more selected observations than K=1.
    a = run_rboa(
        ObservedRBOA(selection_fraction=0.75, model_selection_clusters=2),
        bounds,
        n=n,
        iterations=1,
        seed=seed_for(case),
    )
    g = a.generations[1]
    assert len(g.kmeans) == 2
    assert_valid_permutation(
        cast(list[int], np.concatenate(g.kmeans).tolist()), list(range(len(g.selected)))
    )
    assert g.limit == max(0, min(1, min(map(len, g.kmeans)) - 2))
    assert g.limit == limit
    assert set(g.scores) == {(0, ()), (1, ())} | (
        {(0, (1,)), (1, (0,))} if limit else set()
    )
    assert all(len(ps) <= g.limit for ps in g.parents)
    for (child, parents), score in g.scores.items():
        entropy = bic(g.selected, g.kmeans, child, parents, 0, a.covariance_floor)
        complexity = 2 * a.bic_penalty * log(len(g.selected)) * (len(parents) + 2)
        assert score == pytest.approx(
            entropy + complexity, rel=STRICT_RTOL, abs=STRICT_ATOL
        )
    # Check the actual greedy outcome using only the two possible edges.
    if g.limit:
        deltas = [
            bic(g.selected, g.kmeans, c, (1 - c,), a.bic_penalty, a.covariance_floor)
            - bic(g.selected, g.kmeans, c, (), a.bic_penalty, a.covariance_floor)
            for c in (0, 1)
        ]
        edges = [(p, c) for c, ps in enumerate(g.parents) for p in ps]
        if min(deltas) < -STRICT_ATOL:
            assert len(edges) == 1
            assert deltas[edges[0][1]] == pytest.approx(min(deltas), abs=STRICT_ATOL)
        else:
            assert not edges
    assert_selection(a, bounds)


def gaussian(
    data: Array, component: tuple[int, ...], parents: Parents, variance_floor: float
) -> tuple[Array, dict[int, tuple[float, Array, Array, float]]]:
    """ML moments and Schur complements, independent of scipy's fitting/solve."""
    mean = cast(Array, np.mean(data, axis=0))
    centered = data - mean
    covariance: Array = centered.T @ centered / len(data)
    shift = max(0.0, variance_floor - cast(float, np.linalg.eigvalsh(covariance).min()))
    covariance = covariance + shift * np.eye(len(component))
    conditionals: dict[int, tuple[float, Array, Array, float]] = {}
    for i, variable in enumerate(component):
        ps = [component.index(p) for p in parents[variable]]
        pp: Array = covariance[np.ix_(ps, ps)]
        py: Array = covariance[ps, i]
        beta: Array = np.linalg.inv(pp) @ py if ps else np.empty(0)
        variance = max(
            variance_floor, cast(float, covariance.item(i, i)) - float(py @ beta)
        )
        conditionals[variable] = (
            cast(float, mean.item(i)),
            mean[ps],
            beta,
            variance**0.5,
        )
    return covariance, conditionals


def assert_models(
    g: Generation, models: tuple[SubproblemModel, ...], variance_floor: float
) -> None:
    # In 2D there are exactly two possible undirected partitions.
    expected_components = (
        {frozenset((0, 1))} if any(g.parents) else {frozenset((0,)), frozenset((1,))}
    )
    assert {frozenset(m.variable_indices) for m in models} == expected_components
    assert_valid_permutation([v for m in models for v in m.variable_indices], [0, 1])
    for model, (data, clusters) in zip(models, g.bend, strict=True):
        component = model.variable_indices
        assert component == tuple(v for v in g.order if v in component)
        np.testing.assert_array_equal(data, g.selected[:, component])
        weights = np.array([len(indices) / len(data) for indices in clusters])
        assert np.all(weights > 0)
        assert sum(weights) == pytest.approx(1, abs=STRICT_ATOL)
        np.testing.assert_allclose(
            [c.weight for c in model.clusters],
            weights,
            rtol=STRICT_RTOL,
            atol=STRICT_ATOL,
        )
        cumulative = cast(Array, model.cumulative_weights)
        np.testing.assert_allclose(
            cumulative, np.cumsum(weights), rtol=STRICT_RTOL, atol=STRICT_ATOL
        )
        assert cumulative[-1] == 1
        for fitted, indices in zip(model.clusters, clusters, strict=True):
            _, expected = gaussian(
                cast(Array, data[indices]), component, g.parents, variance_floor
            )
            assert tuple(c.variable_index for c in fitted.conditionals) == component
            for conditional in fitted.conditionals:
                v = conditional.variable_index
                mean, parent_means, beta, sigma = expected[v]
                assert conditional.parent_indices == g.parents[v]
                assert conditional.mean == pytest.approx(mean, abs=STRICT_ATOL)
                np.testing.assert_allclose(
                    cast(Array, conditional.parent_means),
                    parent_means,
                    rtol=LINALG_RTOL,
                    atol=LINALG_ATOL,
                )
                np.testing.assert_allclose(
                    cast(Array, conditional.coefficients),
                    beta,
                    rtol=LINALG_RTOL,
                    atol=LINALG_ATOL,
                )
                assert conditional.standard_deviation == pytest.approx(
                    sigma, rel=LINALG_RTOL, abs=LINALG_ATOL
                )
                assert np.isfinite(sigma) and sigma > 0
                assert sigma**2 >= variance_floor - LINALG_ATOL


def test_bend_components_weights_and_covariance_floor(
    trajectory: ObservedRBOA,
    bounds: dict[str, tuple[float, float]],
) -> None:
    singleton = run_rboa(
        ObservedRBOA(clustering_threshold=0, max_parents=0), bounds, iterations=1
    )
    joined = run_rboa(
        ObservedRBOA(bic_penalty=0, clustering_threshold=1), bounds, iterations=1
    )
    # In 1D the two extrema have BEND distance exactly 1. Case 0 makes an
    # extremum a leader, so equality must create another cluster (strict <).
    endpoint = run_rboa(
        ObservedRBOA(clustering_threshold=1, max_parents=0),
        bounds,
        iterations=1,
        seed=seed_for(0),
    )
    branches: set[str] = set()
    for a in (trajectory, singleton, joined, endpoint):
        for t, g in a.generations.items():
            if t == 0:
                continue
            assert_models(g, g.models, a.covariance_floor)
            for data, clusters in g.bend:
                assert_valid_permutation(
                    cast(list[int], np.concatenate(clusters).tolist()),
                    list(range(len(data))),
                )
                minimum = cast(Array, data.min(axis=0))
                span = cast(Array, np.ptp(data, axis=0))
                scaled: Array = np.divide(
                    data - minimum, span, out=np.zeros_like(data), where=span > 0
                )
                upper: Array = (span > 0).astype(float)
                diameter = max(
                    cast(float, cast(Array, np.linalg.norm(scaled, axis=1)).max()),
                    cast(
                        float, cast(Array, np.linalg.norm(scaled - upper, axis=1)).max()
                    ),
                )
                assert diameter > 0
                leaders = [cast(int, indices.item(0)) for indices in clusters]
                for k, indices in enumerate(clusters):
                    leader = leaders[k]
                    for previous in leaders[:k]:
                        distance = (
                            float(
                                np.linalg.norm(
                                    cast(Array, scaled[leader] - scaled[previous])
                                )
                            )
                            / diameter
                        )
                        assert distance >= a.clustering_threshold
                        branches.add("new")
                        if distance == a.clustering_threshold:
                            branches.add("strict threshold")
                    for offset in range(1, len(indices)):
                        member = cast(int, indices.item(offset))
                        distance = (
                            float(
                                np.linalg.norm(
                                    cast(Array, scaled[member] - scaled[leader])
                                )
                            )
                            / diameter
                        )
                        assert distance < a.clustering_threshold
                        # All earlier-created leaders existed when this member joined.
                        assert all(
                            distance
                            <= float(
                                np.linalg.norm(cast(Array, scaled[member] - scaled[p]))
                            )
                            / diameter
                            + STRICT_ATOL
                            for p in leaders[:k]
                        )
                        branches.add("join")
                if a.clustering_threshold == 0:
                    assert all(len(c) == 1 for c in clusters)
            cluster_inputs = [
                (cast(Array, data[indices]), m.variable_indices)
                for m, (data, clusters) in zip(g.models, g.bend, strict=True)
                for indices in clusters
            ]
            for (data, component), (raw, regularized) in zip(
                cluster_inputs, g.covariance, strict=True
            ):
                mean = cast(Array, data.mean(axis=0))
                centered = data - mean
                np.testing.assert_allclose(
                    raw,
                    centered.T @ centered / len(data),
                    rtol=LINALG_RTOL,
                    atol=LINALG_ATOL,
                )
                expected, _ = gaussian(data, component, g.parents, a.covariance_floor)
                np.testing.assert_allclose(
                    regularized, expected, rtol=LINALG_RTOL, atol=LINALG_ATOL
                )
                assert (
                    np.linalg.eigvalsh(regularized).min()
                    >= a.covariance_floor - LINALG_ATOL
                )
                assert np.linalg.eigvalsh(regularized).min() > 0
                if len(data) == 1:
                    np.testing.assert_array_equal(
                        regularized, np.eye(len(component)) * a.covariance_floor
                    )
                    branches.add("floor")
                if np.linalg.eigvalsh(raw).min() >= a.covariance_floor:
                    np.testing.assert_array_equal(raw, regularized)
                    branches.add("unchanged")
    assert branches == {"new", "join", "floor", "unchanged", "strict threshold"}


def assert_sampling(
    a: ObservedRBOA,
    bounds: dict[str, tuple[float, float]],
    objective: Callable[..., float] = sphere,
) -> set[str]:
    witnessed: set[str] = set()
    lower = np.array([b[0] for b in bounds.values()])
    spans = np.array([b[1] - b[0] for b in bounds.values()])
    for t, g in a.generations.items():
        for initialize, caches in g.caches:
            active = g.temporary if initialize else g.active
            keys = {f"subproblem-{i}-uniform" for i in range(len(g.models))} | {
                f"variable-{j}-uniform" for j in range(2)
            }
            for identifier, cache in caches.items():
                assert set(cache) == (keys if identifier in active else set())
                assert all(isinstance(u, float) and 0 <= u < 1 for u in cache.values())
        for _, _, before, particle in g.processed:
            assert (
                particle.random_cache == before
            )  # Processing never consumes the cache.
        assert len(g.samples) == g.q
        for sample in g.samples:
            assert t > 0 and sample.parents == g.parents
            # Each offspring still reads P_(t-1) and the same learned parameters.
            for key, parent in g.before.items():
                observed = sample.before[key]
                assert (observed.variables, observed.fitness) == (
                    parent.variables,
                    parent.fitness,
                )
            assert_models(g, sample.models, a.covariance_floor)
            expected = np.full(2, np.nan)
            for i, (model, (data, clusters)) in enumerate(
                zip(g.models, g.bend, strict=True)
            ):
                u = sample.cache[f"subproblem-{i}-uniform"]
                assert isinstance(u, float)
                # CDF from observed cluster cardinalities, never fitted weights.
                cumulative = np.cumsum([len(c) / len(data) for c in clusters])
                k = next(k for k in range(len(clusters)) if u < cumulative[k])
                if k > 0:
                    witnessed.add("nonfirst mixture")
                _, conditionals = gaussian(
                    cast(Array, data[clusters[k]]),
                    model.variable_indices,
                    g.parents,
                    a.covariance_floor,
                )
                for v in model.variable_indices:
                    mean, parent_means, beta, sigma = conditionals[v]
                    ps = list(g.parents[v])
                    sampled_parents: Array = expected[ps]
                    assert np.all(np.isfinite(sampled_parents))
                    conditional_mean = mean + float(
                        beta @ (sampled_parents - parent_means)
                    )
                    if ps and abs(conditional_mean - mean) > STRICT_ATOL:
                        witnessed.add("conditional")
                        old = (
                            np.array(
                                list(
                                    sample.before[sample.identifier].variables.values()
                                )
                            )
                            - lower
                        ) / spans
                        old_parents: Array = old[ps]
                        stale_mean = mean + float(beta @ (old_parents - parent_means))
                        if abs(conditional_mean - stale_mean) > STRICT_ATOL:
                            witnessed.add("fresh parent")
                    u = sample.cache[f"variable-{v}-uniform"]
                    assert isinstance(u, float)
                    # Independent inverse CDF; production uses scipy.stats.truncnorm.ppf.
                    normal = NormalDist()
                    pa = normal.cdf(-conditional_mean / sigma)
                    pb = normal.cdf((1 - conditional_mean) / sigma)
                    expected[v] = conditional_mean + sigma * normal.inv_cdf(
                        pa + u * (pb - pa)
                    )
                    unrestricted = conditional_mean + sigma * normal.inv_cdf(u)
                    if (
                        abs(
                            cast(float, expected.item(v))
                            - min(1.0, max(0.0, unrestricted))
                        )
                        > 1e-5
                    ):
                        witnessed.add("truncation")
                    if unrestricted < 0 or unrestricted > 1:
                        witnessed.add("outside")
                    else:
                        witnessed.add("inside")
            actual = np.array(list(sample.values.values()))
            np.testing.assert_allclose(
                (actual - lower) / spans, expected, rtol=LINALG_RTOL, atol=LINALG_ATOL
            )
            np.testing.assert_allclose(
                actual, lower + spans * expected, rtol=LINALG_RTOL, atol=LINALG_ATOL
            )
            initialize = sample.identifier in g.temporary
            matching = [
                p
                for init, key, _, p in g.processed
                if init == initialize and key == sample.identifier
            ]
            (evaluated,) = matching
            assert evaluated.candidate_variables == sample.values
            assert evaluated.candidate_fitness == pytest.approx(
                objective(**sample.values), rel=STRICT_RTOL, abs=STRICT_ATOL
            )
            assert evaluated.random_cache == sample.cache
    return witnessed


def test_conditional_sampling_uses_cached_uniforms_and_synchronous_model(
    bounds: dict[str, tuple[float, float]],
) -> None:
    # Case 0 gives a child dependency and an unrestricted draw outside [0,1].
    a = run_rboa(
        ObservedRBOA(bic_penalty=0, clustering_threshold=1),
        bounds,
        iterations=2,
        seed=seed_for(0),
    )
    witnessed = assert_sampling(a, bounds)
    assert {
        "conditional",
        "fresh parent",
        "truncation",
        "outside",
        "inside",
    } <= witnessed
    mixture = run_rboa(ObservedRBOA(max_parents=0, covariance_floor=0.01), bounds)
    assert "nonfirst mixture" in assert_sampling(mixture, bounds)


def solution(
    p: RBOAParticle, *, candidate: bool = False
) -> tuple[tuple[float, ...], float]:
    values = p.candidate_variables if candidate else p.variables
    fitness = p.candidate_fitness if candidate else p.fitness
    assert values is not None and fitness is not None
    return tuple(values.values()), float(fitness)


def assert_renewal(a: ObservedRBOA) -> bool:
    """Compare multisets and fitness order; tied identities have no fixed order."""
    worse_child_entered = False
    for t, g in a.generations.items():
        if t == 0:
            continue
        n = len(g.before)
        assert g.q == (
            n - floor(a.selection_fraction * n)
            if a.offspring_size is None
            else a.offspring_size
        )
        assert len(g.temporary) == max(0, g.q - n)
        assert len(g.active) == min(g.q, n)
        assert set(g.before_post) == set(g.before) | g.temporary
        assert set(g.after) == set(g.before)
        parents = Counter(solution(p) for p in g.before.values())
        children = Counter(
            [solution(g.before_post[key], candidate=True) for key in g.active]
            + [solution(g.before_post[key]) for key in g.temporary]
        )
        assert sum(children.values()) == g.q
        survivors = Counter(solution(p) for p in g.after.values())
        if g.q < n:
            assert children <= survivors  # Every child enters, even a worse one.
            retained = survivors - children
            assert retained <= parents
            assert (
                sorted(f for _, f in retained.elements())
                == sorted(f for _, f in parents.elements())[: n - g.q]
            )
            discarded = parents - retained
            if discarded and children:
                worse_child_entered |= max(f for _, f in children) > min(
                    f for _, f in discarded
                )
        else:
            assert survivors <= children
            assert (
                sorted(f for _, f in survivors.elements())
                == sorted(f for _, f in children.elements())[:n]
            )
        for key in g.before:
            assert solution(g.before_post[key]) == solution(g.before[key])
    return worse_child_entered


def test_default_renewal_retains_best_parents_and_all_offspring(
    trajectory: ObservedRBOA,
    bounds: dict[str, tuple[float, float]],
) -> None:
    assert trajectory.offspring_size is None
    assert all(g.q < len(g.before) for g in trajectory.generations.values())
    _ = assert_renewal(trajectory)
    # A wider Gaussian makes accepting a child worse than a discarded parent
    # observable; the expected renewal is still derived from actual fitness.
    a = run_rboa(ObservedRBOA(covariance_floor=0.1, max_parents=0), bounds)
    assert assert_renewal(a)
    _ = assert_sampling(a, bounds)


def test_excess_offspring_renewal_includes_temporary_particles(
    bounds: dict[str, tuple[float, float]],
) -> None:
    a = run_rboa(
        ObservedRBOA(offspring_size=11, max_parents=0, covariance_floor=0.01), bounds
    )
    _ = assert_renewal(a)
    _ = assert_sampling(a, bounds)
    for t, g in a.generations.items():
        if t == 0:
            continue
        survivors = {solution(p) for p in g.after.values()}
        assert any(solution(g.before_post[key]) in survivors for key in g.temporary)
        assert any(
            solution(g.before_post[key], candidate=True) in survivors
            for key in g.active
        )
        assert {s.identifier for s in g.samples} == g.active | g.temporary
        assert all((t, key) in a.initialized for key in g.temporary)
        assert not (set(g.after) & g.temporary)


def test_full_selection_learns_without_producing_offspring(
    bounds: dict[str, tuple[float, float]],
) -> None:
    a = run_rboa(ObservedRBOA(selection_fraction=1), bounds)
    assert_selection(a, bounds)
    _ = assert_renewal(a)
    _ = assert_sampling(a, bounds)
    for t, g in a.generations.items():
        if t > 0:
            assert g.models and g.scores
            assert len(g.selected) == len(g.before)
            assert g.q == 0 and not g.samples and not g.temporary and not g.active
            assert Counter(solution(p) for p in g.before.values()) == Counter(
                solution(p) for p in g.after.values()
            )
            assert_models(g, g.models, a.covariance_floor)


def regional_fitness(x: float, y: float) -> float:
    """Three ordered fitness regimes on the unit square, without NaNs."""
    if x < 1 / 3:
        return -np.inf
    if x > 2 / 3:
        return np.inf
    return sphere(x, y)


def test_infinite_fitness_order_and_ties_keep_models_defined() -> None:
    bounds = {"x": (0.0, 1.0), "y": (0.0, 1.0)}
    for objective, q in (
        (regional_fitness, None),
        (regional_fitness, 11),
        (constant_objective, None),
    ):
        a = run_rboa(
            ObservedRBOA(max_parents=0, covariance_floor=0.01, offspring_size=q),
            bounds,
            objective=objective,
        )
        initial = [float(p.fitness) for p in a.generations[0].after.values()]
        if objective is regional_fitness:
            assert -np.inf in initial and np.inf in initial
            assert any(np.isfinite(f) for f in initial)
            assert FITNESS_UNDEFINED == np.inf and FITNESS_UNDEFINED != -np.inf
        else:
            assert set(initial) == {1.0}
        assert_selection(a, bounds)
        _ = assert_renewal(a)
        _ = assert_sampling(a, bounds, objective)


def test_affine_rescaling_preserves_probability_geometry(
    bounds: dict[str, tuple[float, float]],
) -> None:
    def normalized_sphere(x: float, y: float) -> float:
        return sphere((x + 7) / 12, (y - 20) / 60)

    unit_bounds = {"x": (0.0, 1.0), "y": (0.0, 1.0)}
    unit = run_rboa(ObservedRBOA(max_parents=0), unit_bounds)
    scaled = run_rboa(ObservedRBOA(max_parents=0), bounds, objective=normalized_sphere)
    assert_selection(unit, unit_bounds)
    assert_selection(scaled, bounds)
    for t in range(1, 3):
        left, right = unit.generations[t], scaled.generations[t]
        np.testing.assert_allclose(
            left.selected, right.selected, rtol=STRICT_RTOL, atol=STRICT_ATOL
        )
        assert left.parents == right.parents
        for unit_sample, scaled_sample in zip(left.samples, right.samples, strict=True):
            assert unit_sample.cache == scaled_sample.cache
            normalized = [
                (scaled_sample.values["x"] + 7) / 12,
                (scaled_sample.values["y"] - 20) / 60,
            ]
            np.testing.assert_allclose(
                list(unit_sample.values.values()),
                normalized,
                rtol=STRICT_RTOL,
                atol=STRICT_ATOL,
            )
    _ = assert_sampling(unit, unit_bounds)
    _ = assert_sampling(scaled, bounds, normalized_sphere)
