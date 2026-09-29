"""CMA-ES equations through Optimizer, Continuous and the real Serial replica.

Sources (equation numbers below refer to AH unless stated otherwise):
HO: Hansen & Ostermeier (2001), section 5, cumulation and CSA:
https://www.cmap.polytechnique.fr/~nikolaus.hansen/cmaartic.pdf
HMK: Hansen, Mueller & Koumoutsakos (2003), rank-mu adaptation:
https://www.cmap.polytechnique.fr/~nikolaus.hansen/evco_11_1_1_0.pdf
H: Hansen (2006), comparing review, Gaussian sampling / covariance factors:
https://www.cmap.polytechnique.fr/~nikolaus.hansen/hansenedacomparing.pdf
AH: Akimoto & Hansen (2020), doi:10.1162/evco_a_00260 (author preprint):
https://arxiv.org/html/1905.05885v1
AH section 2.1, eqs. (2)-(11), and section 5, eqs. (36)-(38) identify
the positive-weight CMA variant here, with fixed diagonal decoding D=I.
The revised learning rates do NOT imply that Tyrannis implements dd-CMA or
active negative weights. Sampling uses B diag(sqrt(eigenvalues)), a factor
AA.T=C, not the symmetric square root used for whitening.

Tyrannis/task extensions: initial population centroid, range-scaled sigma,
coordinate clipping BEFORE selection, evaluated temporary center, historical
population best, and arbitrary tie order. AH's footnote 2 instead averages
tied weights; we deliberately test the task's tie contract, not that rule.
actual_iter=0 is setup; actual_iter=k>0 maps distribution t=k-1 to t+1.

Injection scope: Optimizer's default Local forces IslandIsolation and supplies
NoCommunicationDriver for one island. Real arrival uses migration_control ->
ProcessorBase._insert_arrival_particle -> create_particle -> new_particles_id.
The available multi-island backends require MPI/Spark; exercising this here
would require a distributed execution, not a small Local+Serial trajectory.
A dedicated migration mathematical test must check preserved external positions,
absent z, selected-step Mahalanobis limiting, mean/rank-mu contributions and
parameter recalculation after population changes. Hansen (2011), Fig. 1 eq. (3)
and Table 1 give c_y=sqrt(n)+2*n/(n+2): https://arxiv.org/pdf/1110.4181 .
No injected flags or covariance eigenvalue safeguards are forced artificially.
"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from itertools import pairwise, permutations
from math import log, sqrt
from typing import cast, override

import numpy as np
import pytest

from tests._support import factories
from tests._support.assertions import assert_particle_state_consistent
from tests._support.numerics import (
    BASE_SEED,
    LINALG_ATOL,
    LINALG_RTOL,
    STRICT_ATOL,
    STRICT_RTOL,
    seed_for,
)
from tests._support.objectives import CountingObjective, constant_objective, sphere
from tyrannis import Optimizer
from tyrannis.algorithm.cma_es import CMAES, CMAESCandidateSolution
from tyrannis.core.algorithm import ParticleBase
from tyrannis.processor.serial import Serial
from tyrannis.space.continuous import Continuous

Vector = np.ndarray[tuple[int, ...], np.dtype[np.float64]]


@dataclass
class Distribution:
    mean: Vector
    covariance: Vector
    sigma: float
    p_sigma: Vector
    p_c: Vector
    gamma_sigma: float
    gamma_c: float
    weights: Vector
    mu: int
    mu_eff: float
    cc: float
    cs: float
    c1: float
    cmu: float
    damps: float
    chi_n: float


@dataclass
class Stage:
    iteration: int
    phase: str
    identifier: str
    distribution: Distribution
    population: dict[str, CMAESCandidateSolution]
    center: CMAESCandidateSolution | None
    best: CMAESCandidateSolution | None
    double_ids: list[str]
    evaluations: int


class ObservedCMAES(CMAES):
    """Only snapshot state; every lifecycle hook delegates to production."""

    count_evaluations: Callable[[], int]
    # Production uses unparameterized ndarray annotations. These are its actual
    # float64 arrays; specializing them changes no runtime state.
    _mean: Vector | None
    _covariance: Vector | None
    _p_sigma: Vector | None
    _p_c: Vector | None
    _weights: Vector | None

    def distribution(self) -> Distribution:
        # Protected state is observable here because no public equivalent exists.
        assert self._mean is not None and self._covariance is not None
        assert self._p_sigma is not None and self._p_c is not None
        assert self._weights is not None and self._mu is not None
        assert self._mu_eff is not None and self._cc is not None
        assert self._cs is not None and self._c1 is not None
        assert self._cmu is not None and self._damps is not None
        assert self._chi_n is not None
        return Distribution(
            self._mean.copy(),
            self._covariance.copy(),
            self.sigma,
            self._p_sigma.copy(),
            self._p_c.copy(),
            self._gamma_sigma,
            self._gamma_c,
            self._weights.copy(),
            self._mu,
            self._mu_eff,
            self._cc,
            self._cs,
            self._c1,
            self._cmu,
            self._damps,
            self._chi_n,
        )

    def mark(self, phase: str, identifier: str = "") -> None:
        self.stages.append(
            Stage(
                self.actual_iter,
                phase,
                identifier,
                self.distribution(),
                deepcopy(self.population),
                deepcopy(self.mean_particle),
                deepcopy(self.local_best),
                self.double_check_ids.copy(),
                self.count_evaluations(),
            )
        )

    @override
    def pre_iteration(self, actual_iter: int) -> None:
        self.actual_iter: int = actual_iter
        if actual_iter == 0:
            self.initial: dict[str, CMAESCandidateSolution] = deepcopy(self.population)
            self.stages: list[Stage] = []
            self.rng_unchanged: list[bool] = []
            self.center_independent: list[bool] = []
            self.cache_requests: list[tuple[int, bool, list[str]]] = []
            self.empty_cache_rng_unchanged: list[bool] = []
        super().pre_iteration(actual_iter)
        self.mark("pre")

    @override
    def create_random_cache(self, particle_ids: list[str], initialize: bool) -> None:
        rng_state = deepcopy(self._rng.bit_generator.state)
        super().create_random_cache(particle_ids, initialize)
        if initialize or particle_ids == [self._center_identifier]:
            self.empty_cache_rng_unchanged.append(
                rng_state == self._rng.bit_generator.state
            )
        self.cache_requests.append((self.actual_iter, initialize, particle_ids.copy()))
        self.mark("initial-cache" if initialize else "cache")

    @override
    def initialize_particle(self, identifier: str) -> CMAESCandidateSolution:
        self.mark("initialize-before", identifier)
        particle = super().initialize_particle(identifier)
        self.mark("initialize-after", identifier)
        return particle

    @override
    def update_particle(self, identifier: str) -> CMAESCandidateSolution:
        self.mark("sample-before", identifier)
        rng_state = deepcopy(self._rng.bit_generator.state)
        particle = super().update_particle(identifier)
        self.rng_unchanged.append(rng_state == self._rng.bit_generator.state)
        self.mark("sample-after", identifier)
        return particle

    @override
    def inter_iteration(self, actual_iter: int) -> None:
        self.mark("inter-before")
        super().inter_iteration(actual_iter)
        self.mark("inter-after")

    @override
    def second_update_particle(self, identifier: str) -> CMAESCandidateSolution:
        self.mark("second-before", identifier)
        rng_state = deepcopy(self._rng.bit_generator.state)
        particle = super().second_update_particle(identifier)
        self.rng_unchanged.append(rng_state == self._rng.bit_generator.state)
        self.mark("second-after", identifier)
        return particle

    @override
    def post_iteration(self, actual_iter: int) -> None:
        self.mark("post-before")
        center = self.population[self._center_identifier]
        super().post_iteration(actual_iter)
        snapshot = self.mean_particle
        assert snapshot is not None
        self.center_independent.append(
            snapshot is not center and snapshot.variables is not center.variables
        )
        self.mark("post-after")


@dataclass
class Evaluation:
    identifier: str | None
    variables: dict[str, float]
    fitness: float


@dataclass
class Run:
    algorithm: ObservedCMAES
    calls: list[Evaluation]
    updates: list[tuple[str, int, int]]

    def stage(self, iteration: int, phase: str, identifier: str = "") -> Stage:
        matches = [
            s
            for s in self.algorithm.stages
            if (s.iteration, s.phase, s.identifier) == (iteration, phase, identifier)
        ]
        assert len(matches) == 1
        return matches[0]


def run_cma(
    monkeypatch: pytest.MonkeyPatch,
    bounds: dict[str, tuple[float, float]],
    *,
    objective: Callable[..., float] = sphere,
    sigma: float = 0.3,
    seed: int = BASE_SEED,
    iterations: int = 4,
) -> Run:
    calls: list[Evaluation] = []
    updates: list[tuple[str, int, int]] = []
    active_identifier: str | None = None

    def recorded_objective(**variables: float) -> float:
        value = objective(**variables)
        calls.append(Evaluation(active_identifier, variables.copy(), value))
        return value

    real_update = ParticleBase.update

    def observed_update(
        particle: ParticleBase,
        variables: dict[str, float],
        fitness_function: Callable[[dict[str, float]], np.float64],
    ) -> None:
        nonlocal active_identifier
        active_identifier = particle.identifier
        start = len(calls)
        real_update(particle, variables, fitness_function)
        updates.append((particle.identifier, start, len(calls)))
        active_identifier = None

    template = ObservedCMAES(sigma)
    template.count_evaluations = lambda: len(calls)
    processor = Serial()
    with monkeypatch.context() as patch:
        patch.setattr(ParticleBase, "update", observed_update)
        # Shared factory erases AlgorithmBase's particle type; retain its real
        # public return type at this existing untyped boundary.
        factory = cast(Callable[..., Optimizer], factories.make_optimizer)
        optimizer = factory(
            Continuous(bounds, cost_function=recorded_objective, use_cache=False),
            template,
            processor=processor,
            n_particles=5,
            n_iterations=iterations,
            seed=seed,
            fitness_failure_strategy="raise",
        ).fit()
    (executor,) = processor.processors_pool.values()
    # Serial exposes its replicas publicly, but not their algorithm accessor.
    # Processor's stored AlgorithmBase also erases the particle generic.
    algorithm = cast(object, vars(executor)["_algorithm"])
    assert isinstance(algorithm, ObservedCMAES)
    assert algorithm is not template and not template.population
    assert len(calls) == len(updates)
    for index, (identifier, start, end) in enumerate(updates):
        assert (start, end) == (index, index + 1)
        assert calls[index].identifier == identifier
    assert algorithm.local_best is not None
    assert optimizer.best_fitness == algorithm.local_best.fitness
    for stage in algorithm.stages:
        assert_finite_geometry(stage)
    return Run(algorithm, calls, updates)


def coordinates(particle: CMAESCandidateSolution, *, candidate: bool = False) -> Vector:
    values = particle.candidate_variables if candidate else particle.variables
    assert values is not None
    return np.array([values["x"], values["y"]], dtype=float)


def weights_for(population_size: int) -> Vector:
    """AH section 2.1: normalize positive logarithmic rank weights."""
    raw = [
        log((population_size + 1) / (2 * i)) for i in range(1, population_size // 2 + 1)
    ]
    return np.array(raw) / sum(raw)


@pytest.fixture
def trajectory(
    monkeypatch: pytest.MonkeyPatch, small_bounds: dict[str, tuple[float, float]]
) -> Run:
    # case 1 selects a clipped candidate and exercises both signs of CSA.
    return run_cma(monkeypatch, small_bounds, seed=seed_for(1))


def test_setup_initializes_distribution_from_population_mathematically(
    trajectory: Run,
) -> None:
    pre = trajectory.stage(0, "pre")
    post = trajectory.stage(0, "post-after")
    state = pre.distribution
    centroid = (
        sum(
            (coordinates(p) for p in trajectory.algorithm.initial.values()), np.zeros(2)
        )
        / 5
    )
    np.testing.assert_allclose(state.mean, centroid, rtol=STRICT_RTOL, atol=STRICT_ATOL)
    np.testing.assert_array_equal(state.covariance, np.eye(2))
    np.testing.assert_array_equal(state.p_sigma, np.zeros(2))
    np.testing.assert_array_equal(state.p_c, np.zeros(2))
    assert state.gamma_sigma == state.gamma_c == 0
    assert state.sigma == pytest.approx(
        0.3 * (10 + 6) / 2, rel=STRICT_RTOL, abs=STRICT_ATOL
    )
    assert state.chi_n == pytest.approx(
        sqrt(2) * (1 - 1 / 8 + 1 / 84), rel=STRICT_RTOL, abs=STRICT_ATOL
    )
    assert pre.evaluations == 0 and post.evaluations == 6
    assert pre.double_ids == post.double_ids == []
    assert post.center is not None
    np.testing.assert_allclose(
        coordinates(pre.population[post.center.identifier]),
        centroid,
        rtol=STRICT_RTOL,
        atol=STRICT_ATOL,
    )
    np.testing.assert_allclose(
        coordinates(post.center), centroid, rtol=STRICT_RTOL, atol=STRICT_ATOL
    )
    assert post.center.fitness == pytest.approx(
        sphere(**post.center.variables), rel=STRICT_RTOL, abs=STRICT_ATOL
    )
    assert post.center.identifier not in post.population
    assert trajectory.algorithm.center_independent[0]
    assert_distribution_equal(state, post.distribution)
    setup = [s for s in trajectory.algorithm.stages if s.iteration == 0]
    assert not any(s.phase.startswith(("sample", "inter", "second")) for s in setup)
    assert all(not p.random_cache for s in setup for p in s.population.values())


def test_strategy_parameters_follow_revised_published_defaults(trajectory: Run) -> None:
    # AH eqs. (6),(7),(36)-(38), full symmetric C has n(n+1)/2 degrees.
    n, population_size = 2, 5
    weights = weights_for(population_size)
    mass = 1 / float(weights @ weights)
    cs = (mass + 2) / (n + mass + 5)
    c1 = 1 / ((n + 3) * (n + 1) ** 0.75 + mass / 2)
    cmu = min(
        (mass + 1 / mass - 2 + population_size / (2 * (population_size + 5))) * c1,
        1 - c1,
    )
    cc = sqrt(mass * c1) / 2
    damps = 1 + cs + 2 * max(0, sqrt((mass - 1) / (n + 1)) - 1)
    for stage in trajectory.algorithm.stages:
        state = stage.distribution
        assert state.mu == population_size // 2 == len(state.weights)
        assert np.all(state.weights > 0) and np.all(np.diff(state.weights) < 0)
        assert sum(state.weights) == pytest.approx(1, rel=STRICT_RTOL, abs=STRICT_ATOL)
        np.testing.assert_allclose(
            state.weights, weights, rtol=STRICT_RTOL, atol=STRICT_ATOL
        )
        assert 1 <= state.mu_eff <= state.mu
        np.testing.assert_allclose(
            [state.mu_eff, state.cs, state.damps, state.cc, state.c1, state.cmu],
            [mass, cs, damps, cc, c1, cmu],
            rtol=STRICT_RTOL,
            atol=STRICT_ATOL,
        )


def assert_distribution_equal(first: Distribution, second: Distribution) -> None:
    for left, right in (
        (first.mean, second.mean),
        (first.covariance, second.covariance),
        (first.p_sigma, second.p_sigma),
        (first.p_c, second.p_c),
        (first.weights, second.weights),
    ):
        np.testing.assert_array_equal(left, right)
    assert (
        first.sigma,
        first.gamma_sigma,
        first.gamma_c,
        first.mu,
        first.mu_eff,
        first.cc,
        first.cs,
        first.c1,
        first.cmu,
        first.damps,
        first.chi_n,
    ) == (
        second.sigma,
        second.gamma_sigma,
        second.gamma_c,
        second.mu,
        second.mu_eff,
        second.cc,
        second.cs,
        second.c1,
        second.cmu,
        second.damps,
        second.chi_n,
    )


def cached_z(particle: CMAESCandidateSolution) -> Vector:
    # ParticleBase.random_cache erases value types. CMA's entry is an ndarray;
    # specialize this dependency boundary and validate dtype/shape explicitly.
    z = cast(Vector, particle.random_cache["cmaes-z"])
    assert z.shape == (2,) and z.dtype == np.float64
    return z


def candidates_at(run: Run, iteration: int) -> list[CMAESCandidateSolution]:
    stage = run.stage(iteration, "inter-before")
    return [stage.population[key] for key in run.algorithm.initial]


def candidate_fitness(particle: CMAESCandidateSolution) -> float:
    assert particle.candidate_fitness is not None
    return float(particle.candidate_fitness)


def selected_at(run: Run, iteration: int) -> tuple[CMAESCandidateSolution, ...]:
    """Identify selection through its weighted mean, without prescribing tie order.

    With only five candidates/two parents, the admissible recombinations are
    small. Strict ranks admit exactly one ordered pair. Ties admit any pair
    with the best fitness ranks; production's mean must belong to that set.
    No internals, RNG or observed weight is used to define the admissible set.
    """
    candidates = candidates_at(run, iteration)
    weights = weights_for(len(candidates))
    best_scores = sorted(candidate_fitness(p) for p in candidates)[: len(weights)]
    new_mean = run.stage(iteration, "inter-after").distribution.mean
    matches: list[tuple[CMAESCandidateSolution, ...]] = []
    for selected in permutations(candidates, len(weights)):
        if [candidate_fitness(p) for p in selected] != best_scores:
            continue
        recombination = weights @ np.stack(
            [coordinates(p, candidate=True) for p in selected]
        )
        if np.allclose(recombination, new_mean, rtol=LINALG_RTOL, atol=LINALG_ATOL):
            matches.append(selected)
    assert matches, "Mean must recombine exactly mu current, best-ranked candidates"
    return matches[0]


def normalized_steps(run: Run, iteration: int) -> Vector:
    old = run.stage(iteration, "pre").distribution
    return (
        np.stack([coordinates(p, candidate=True) for p in selected_at(run, iteration)])
        - old.mean
    ) / old.sigma


def whiten(covariance: Vector) -> Vector:
    """Symmetric C^(-1/2), not the nonsymmetric sampling factor (H; AH eq. 3)."""
    eigenvalues, basis = np.linalg.eigh(covariance)
    return (basis * (eigenvalues**-0.5)) @ basis.T


def sigma_path(old: Distribution, weighted_step: Vector) -> Vector:
    """AH eq. (3), with D=I and the covariance from before sampling."""
    return (1 - old.cs) * old.p_sigma + sqrt(old.cs * (2 - old.cs) * old.mu_eff) * (
        whiten(old.covariance) @ weighted_step
    )


def path_variance(previous: float, rate: float, gate: int = 1) -> float:
    """AH eqs. (4),(10): variance accumulated by an exponentially decayed path."""
    return (1 - rate) ** 2 * previous + gate * (1 - (1 - rate) ** 2)


def path_gate(path: Vector, variance: float) -> int:
    """AH eq. (11), dimension two in these trajectories."""
    return int(float(path @ path) < variance * 2 * (2 + 4 / 3))


def test_sampling_uses_cached_gaussians_and_synchronous_old_distribution(
    trajectory: Run,
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    clipped: set[bool] = set()
    lower = np.array([small_bounds[name][0] for name in ("x", "y")])
    upper = np.array([small_bounds[name][1] for name in ("x", "y")])
    for iteration in range(1, 5):
        old = trajectory.stage(iteration, "pre").distribution
        previous = trajectory.stage(iteration - 1, "post-after").distribution
        assert_distribution_equal(old, previous)
        eigenvalues, basis = np.linalg.eigh(old.covariance)
        factor = basis * np.sqrt(eigenvalues)
        np.testing.assert_allclose(
            factor @ factor.T, old.covariance, rtol=LINALG_RTOL, atol=LINALG_ATOL
        )
        if iteration == 1:
            np.testing.assert_array_equal(factor, np.eye(2))
        for key in trajectory.algorithm.initial:
            before = trajectory.stage(iteration, "sample-before", key)
            after = trajectory.stage(iteration, "sample-after", key)
            assert_distribution_equal(before.distribution, old)
            assert_distribution_equal(after.distribution, old)
            particle = after.population[key]
            assert (
                set(before.population[key].random_cache)
                == set(particle.random_cache)
                == {"cmaes-z"}
            )
            z = cached_z(before.population[key])
            cache_stage = next(
                s
                for s in trajectory.algorithm.stages
                if s.iteration == iteration and s.phase == "cache"
            )
            np.testing.assert_array_equal(z, cached_z(cache_stage.population[key]))
            np.testing.assert_array_equal(cached_z(particle), z)
            raw = old.mean + old.sigma * (factor @ z)
            expected = np.clip(raw, lower, upper)
            clipped.add(not np.array_equal(raw, expected))
            np.testing.assert_allclose(
                coordinates(particle, candidate=True),
                expected,
                rtol=LINALG_RTOL,
                atol=LINALG_ATOL,
            )
            assert particle.candidate_variables is not None
            assert candidate_fitness(particle) == pytest.approx(
                sphere(**particle.candidate_variables), rel=STRICT_RTOL, abs=STRICT_ATOL
            )
            assert after.evaluations == before.evaluations + 1
    assert clipped == {False, True}
    assert trajectory.algorithm.rng_unchanged and all(
        trajectory.algorithm.rng_unchanged
    )
    # Caches are fresh per generation, yet unchanged during each update.
    for key in trajectory.algorithm.initial:
        draws = [
            cached_z(trajectory.stage(i, "sample-before", key).population[key])
            for i in range(1, 5)
        ]
        assert all(not np.array_equal(a, b) for a, b in pairwise(draws))


def test_selection_recombines_evaluated_candidates_not_previous_fitness(
    trajectory: Run,
) -> None:
    changed_ranking = False
    selected_clipped = False
    for iteration in range(1, 5):
        old = trajectory.stage(iteration, "pre").distribution
        candidates = candidates_at(trajectory, iteration)
        ranked = sorted(candidates, key=candidate_fitness)
        assert len({candidate_fitness(p) for p in ranked}) == 5
        selected = selected_at(trajectory, iteration)
        assert [p.identifier for p in selected] == [p.identifier for p in ranked[:2]]
        old_order = [p.identifier for p in sorted(candidates, key=lambda p: p.fitness)]
        changed_ranking |= old_order[:2] != [p.identifier for p in selected]
        steps = normalized_steps(trajectory, iteration)
        weighted = weights_for(5) @ steps
        new_mean = trajectory.stage(iteration, "inter-after").distribution.mean
        np.testing.assert_allclose(
            new_mean,
            old.mean + old.sigma * weighted,
            rtol=LINALG_RTOL,
            atol=LINALG_ATOL,
        )
        eigenvalues, basis = np.linalg.eigh(old.covariance)
        for particle in selected:
            z = cached_z(
                trajectory.stage(
                    iteration, "sample-before", particle.identifier
                ).population[particle.identifier]
            )
            raw_step = (basis * np.sqrt(eigenvalues)) @ z
            evaluated_step = (
                coordinates(particle, candidate=True) - old.mean
            ) / old.sigma
            selected_clipped |= not np.allclose(
                raw_step, evaluated_step, rtol=LINALG_RTOL, atol=LINALG_ATOL
            )
    assert changed_ranking
    assert selected_clipped, "A selected clipped point must actually enter adaptation"


@pytest.mark.parametrize("stall", [False, True], ids=["ordinary", "h-sigma-zero"])
def test_paths_and_covariance_follow_published_recurrences(
    trajectory: Run,
    monkeypatch: pytest.MonkeyPatch,
    stall: bool,
) -> None:
    if stall:
        # A small initial scale far from sphere's optimum generates sustained
        # directional progress. case 2 stalls the path in generation 3.
        trajectory = run_cma(
            monkeypatch,
            {"x": (20.0, 30.0), "y": (10.0, 16.0)},
            sigma=0.02,
            seed=seed_for(2),
        )
    gates: set[int] = set()
    for iteration in range(1, 5):
        old = trajectory.stage(iteration, "pre").distribution
        new = trajectory.stage(iteration, "inter-after").distribution
        steps = normalized_steps(trajectory, iteration)
        weighted = weights_for(5) @ steps
        inverse = whiten(old.covariance)
        np.testing.assert_allclose(
            inverse @ old.covariance @ inverse,
            np.eye(2),
            rtol=LINALG_RTOL,
            atol=LINALG_ATOL,
        )
        expected_path = sigma_path(old, weighted)
        expected_gamma = path_variance(old.gamma_sigma, old.cs)
        gate = path_gate(expected_path, expected_gamma)
        gates.add(gate)
        np.testing.assert_allclose(
            new.p_sigma, expected_path, rtol=LINALG_RTOL, atol=LINALG_ATOL
        )
        assert new.gamma_sigma == pytest.approx(
            expected_gamma, rel=STRICT_RTOL, abs=STRICT_ATOL
        )
        # Constant c_sigma also gives the independent closed form 1-(1-cs)^(2k).
        assert new.gamma_sigma == pytest.approx(
            1 - (1 - old.cs) ** (2 * iteration), rel=STRICT_RTOL, abs=STRICT_ATOL
        )
        pc = (1 - old.cc) * old.p_c + gate * sqrt(
            old.cc * (2 - old.cc) * old.mu_eff
        ) * weighted
        gc = path_variance(old.gamma_c, old.cc, gate)
        np.testing.assert_allclose(new.p_c, pc, rtol=LINALG_RTOL, atol=LINALG_ATOL)
        if gate == 0:
            assert np.linalg.norm(old.p_c) > LINALG_ATOL
            np.testing.assert_allclose(
                new.p_c, (1 - old.cc) * old.p_c, rtol=LINALG_RTOL, atol=LINALG_ATOL
            )
        assert new.gamma_c == pytest.approx(gc, rel=STRICT_RTOL, abs=STRICT_ATOL)
        # AH (8): expose the three terms and recover each contribution separately.
        rank_one = np.outer(pc, pc)
        rank_mu = steps.T @ np.diag(weights_for(5)) @ steps
        decay = (1 - old.c1 * gc - old.cmu) * old.covariance
        np.testing.assert_allclose(
            new.covariance,
            decay + old.c1 * rank_one + old.cmu * rank_mu,
            rtol=LINALG_RTOL,
            atol=LINALG_ATOL,
        )
        np.testing.assert_allclose(
            (new.covariance - decay - old.c1 * rank_one) / old.cmu,
            rank_mu,
            rtol=LINALG_RTOL,
            atol=LINALG_ATOL,
        )
        np.testing.assert_allclose(
            (new.covariance - decay - old.cmu * rank_mu) / old.c1,
            rank_one,
            rtol=LINALG_RTOL,
            atol=LINALG_ATOL,
        )
        assert np.linalg.norm(rank_mu - np.diag(np.diag(rank_mu))) > LINALG_ATOL
        assert np.linalg.norm(rank_one - np.diag(np.diag(rank_one))) > LINALG_ATOL
    assert gates == ({0, 1} if stall else {1})


def test_covariance_learns_correlation_and_remains_positive(trajectory: Run) -> None:
    correlated = False
    for iteration in range(1, 5):
        covariance = trajectory.stage(iteration, "post-after").distribution.covariance
        assert np.all(np.isfinite(covariance))
        np.testing.assert_allclose(
            covariance, covariance.T, rtol=LINALG_RTOL, atol=LINALG_ATOL
        )
        assert np.all(np.linalg.eigvalsh(covariance) > 0)
        correlated |= bool(
            np.linalg.norm(covariance - np.diag(np.diag(covariance))) > LINALG_ATOL
        )
    assert correlated


def test_cumulative_step_size_has_both_growth_and_shrinkage(trajectory: Run) -> None:
    directions: set[bool] = set()
    for iteration in range(1, 5):
        old = trajectory.stage(iteration, "pre").distribution
        new = trajectory.stage(iteration, "inter-after").distribution
        weighted = weights_for(5) @ normalized_steps(trajectory, iteration)
        expected_path = sigma_path(old, weighted)
        gamma = path_variance(old.gamma_sigma, old.cs)
        # AH (5) is an additive change of log(sigma); do not reuse new.sigma,
        # new.p_sigma or a freshly adapted covariance to derive the increment.
        increment = (
            old.cs
            / old.damps
            * (float(np.linalg.norm(expected_path)) / old.chi_n - sqrt(gamma))
        )
        assert log(new.sigma / old.sigma) == pytest.approx(
            increment, rel=LINALG_RTOL, abs=LINALG_ATOL
        )
        assert new.sigma > 0 and np.isfinite(new.sigma)
        directions.add(new.sigma > old.sigma)
    assert directions == {False, True}


def test_center_is_evaluated_after_adaptation_with_exact_budget(
    trajectory: Run,
    monkeypatch: pytest.MonkeyPatch,
    small_bounds: dict[str, tuple[float, float]],
    counting_objective: CountingObjective,
) -> None:
    center_id = f"{trajectory.algorithm.identifier}|center"
    assert all(trajectory.algorithm.center_independent)
    assert trajectory.algorithm.empty_cache_rng_unchanged and all(
        trajectory.algorithm.empty_cache_rng_unchanged
    )
    assert len(trajectory.calls) == (5 + 1) * (4 + 1) == 30
    for iteration in range(5):
        pre = trajectory.stage(iteration, "pre")
        post = trajectory.stage(iteration, "post-after")
        assert post.evaluations - pre.evaluations == 6
        block = trajectory.calls[pre.evaluations : post.evaluations]
        assert [call.identifier for call in block] == [
            *trajectory.algorithm.initial,
            center_id,
        ]
        assert post.center is not None
        assert len(post.population) == 5 and center_id not in post.population
        assert post.double_ids == []
        np.testing.assert_allclose(
            coordinates(post.center),
            post.distribution.mean,
            rtol=LINALG_RTOL,
            atol=LINALG_ATOL,
        )
        assert post.center.variables == block[-1].variables
        assert post.center.fitness == block[-1].fitness
        assert post.center.fitness == pytest.approx(
            sphere(**post.center.variables), rel=STRICT_RTOL, abs=STRICT_ATOL
        )
        if iteration == 0:
            continue
        inter_before = trajectory.stage(iteration, "inter-before")
        inter_after = trajectory.stage(iteration, "inter-after")
        second_before = trajectory.stage(iteration, "second-before", center_id)
        second_after = trajectory.stage(iteration, "second-after", center_id)
        assert inter_after.double_ids == [center_id]
        assert (
            inter_before.evaluations
            == inter_after.evaluations
            == second_before.evaluations
            == pre.evaluations + 5
        )
        assert second_after.evaluations == post.evaluations
        assert_distribution_equal(inter_after.distribution, second_before.distribution)
        assert_distribution_equal(inter_after.distribution, post.distribution)
        assert (
            second_after.population[center_id].candidate_variables
            == post.center.variables
        )
        assert not second_before.population[center_id].random_cache
        assert not second_after.population[center_id].random_cache
        first_center = trajectory.stage(iteration, "sample-before", center_id)
        assert not first_center.population[center_id].random_cache
        assert (
            trajectory.stage(iteration, "sample-after", center_id).evaluations
            == first_center.evaluations
        )
        # Only the center participates in the second phase/cache request.
        second_ids = [
            s.identifier
            for s in trajectory.algorithm.stages
            if s.iteration == iteration and s.phase == "second-before"
        ]
        assert second_ids == [center_id]
        requests = [
            ids
            for i, initialize, ids in trajectory.algorithm.cache_requests
            if i == iteration and not initialize
        ]
        assert requests == [[*trajectory.algorithm.initial, center_id], [center_id]]
    # The shared counter is enclosed by a function, preserving its identity
    # when the processor deep-copies the template.
    counted = run_cma(monkeypatch, small_bounds, objective=counting_objective)
    assert counting_objective.calls == len(counted.calls) == 30


def test_replacement_is_generational_and_local_best_is_historical(
    trajectory: Run,
) -> None:
    improvements = deteriorations = historical_retained = center_better = False
    historical = np.inf
    for iteration in range(5):
        post = trajectory.stage(iteration, "post-after")
        generation_best = min(p.fitness for p in post.population.values())
        historical = min(historical, float(generation_best))
        assert post.best is not None and post.center is not None
        assert post.best.fitness == historical
        assert post.best.identifier != post.center.identifier
        historical_retained |= generation_best > historical
        center_better |= post.center.fitness < generation_best
        if iteration == 0:
            for key, particle in post.population.items():
                assert particle.variables == trajectory.algorithm.initial[key].variables
                assert particle.fitness == pytest.approx(
                    sphere(**particle.variables), rel=STRICT_RTOL, abs=STRICT_ATOL
                )
            continue
        before = trajectory.stage(iteration, "inter-before")
        for key, particle in post.population.items():
            candidate = before.population[key]
            improvements |= candidate_fitness(candidate) < candidate.fitness
            deteriorations |= candidate_fitness(candidate) > candidate.fitness
            assert particle.variables == candidate.candidate_variables
            assert particle.fitness == candidate.candidate_fitness
            assert (
                particle.candidate_variables is None
                and particle.candidate_fitness is None
            )
        previous = trajectory.stage(iteration - 1, "post-after").best
        assert previous is not None
        if generation_best >= previous.fitness:
            assert post.best.variables == previous.variables
            assert post.best.identifier == previous.identifier
    assert improvements and deteriorations and historical_retained and center_better


def positive_infinite_region(**variables: float) -> float:
    """A small feasible half-space forces +inf into selection when mu needs it."""
    return np.inf if variables["x"] > -2 else sphere(**variables)


def mixed_infinite_regions(**variables: float) -> float:
    """Three ordered regions, with a strictly best -inf half-space."""
    if variables["x"] < 0:
        return -np.inf
    return np.inf if variables["x"] > 2 else sphere(**variables)


def assert_finite_geometry(stage: Stage) -> None:
    state = stage.distribution
    for array in (
        state.mean,
        state.covariance,
        state.p_sigma,
        state.p_c,
        state.weights,
    ):
        assert np.all(np.isfinite(array))
    assert np.all(np.isfinite([state.sigma, state.gamma_sigma, state.gamma_c]))
    assert state.sigma > 0
    for particle in stage.population.values():
        assert np.all(np.isfinite(coordinates(particle)))
        if particle.candidate_variables is not None:
            assert np.all(np.isfinite(coordinates(particle, candidate=True)))
    if stage.center is not None:
        assert np.all(np.isfinite(coordinates(stage.center)))


@pytest.mark.parametrize(
    "mixed", [False, True], ids=["finite-and-positive-inf", "negative-finite-positive"]
)
def test_extreme_fitness_preserves_order_and_rank_based_geometry(
    monkeypatch: pytest.MonkeyPatch,
    small_bounds: dict[str, tuple[float, float]],
    mixed: bool,
) -> None:
    objective = mixed_infinite_regions if mixed else positive_infinite_region
    seed = seed_for(1 if mixed else 3)
    run = run_cma(monkeypatch, small_bounds, objective=objective, seed=seed)

    def finite_ranks(**variables: float) -> float:
        # Sphere <= 34 on these bounds. Replacing infinities with +/-100
        # preserves every strict rank AND every tie, independently of CMA.
        value = objective(**variables)
        return -100.0 if value == -np.inf else 100.0 if value == np.inf else value

    finite_run = run_cma(monkeypatch, small_bounds, objective=finite_ranks, seed=seed)
    counts: list[tuple[int, int, int]] = []
    positive_selected: set[bool] = set()
    for iteration in range(1, 5):
        candidates = candidates_at(run, iteration)
        fitnesses = [candidate_fitness(p) for p in candidates]
        negative = sum(f == -np.inf for f in fitnesses)
        positive = sum(f == np.inf for f in fitnesses)
        finite = 5 - negative - positive
        counts.append((negative, finite, positive))
        selected = selected_at(run, iteration)
        selected_scores = [candidate_fitness(p) for p in selected]
        assert selected_scores == sorted(fitnesses)[:2]
        assert selected_scores.count(np.inf) == max(0, 2 - finite - negative)
        assert selected_scores.count(-np.inf) == min(negative, 2)
        positive_selected.add(np.inf in selected_scores)
        steps = normalized_steps(run, iteration)
        assert np.all(np.isfinite(steps))
        assert np.all(np.isfinite(weights_for(5) @ steps))
        for phase in ("pre", "inter-before", "inter-after", "post-after"):
            stage = run.stage(iteration, phase)
            assert_finite_geometry(stage)
            assert_distribution_equal(
                stage.distribution, finite_run.stage(iteration, phase).distribution
            )
        post = run.stage(iteration, "post-after")
        assert post.best is not None
        if mixed:
            assert post.best.fitness == -np.inf
        for particle in post.population.values():
            assert_particle_state_consistent(particle, small_bounds)
            assert particle.fitness == objective(**particle.variables)
    assert any(finite and positive for _, finite, positive in counts)
    assert any(positive == 1 for _, _, positive in counts)
    assert any(positive > 1 for _, _, positive in counts)
    if mixed:
        assert any(
            negative and finite and positive for negative, finite, positive in counts
        )
        assert any(negative == 1 for negative, _, _ in counts)
        assert any(negative > 1 for negative, _, _ in counts)
    else:
        assert positive_selected == {False, True}


@pytest.mark.parametrize(
    "fitness",
    [1.0, np.inf, -np.inf],
    ids=["finite-ties", "all-positive-inf", "all-negative-inf"],
)
def test_tied_fitness_has_defined_geometry_and_center_sentinel_accounting(
    monkeypatch: pytest.MonkeyPatch,
    small_bounds: dict[str, tuple[float, float]],
    counting_objective: CountingObjective,
    fitness: float,
) -> None:
    counting_objective.return_value = fitness
    run = run_cma(monkeypatch, small_bounds, objective=counting_objective)
    # Cache-free all-+inf has ONE extra old-center initialization per generation:
    # its legitimate fitness equals FITNESS_UNDEFINED. This is sentinel
    # reevaluation, not another adaptation or an extra offspring sample.
    assert (
        counting_objective.calls
        == len(run.calls)
        == 30 + (4 if fitness == np.inf else 0)
    )
    finite_run = run_cma(monkeypatch, small_bounds, objective=constant_objective)
    center_id = f"{run.algorithm.identifier}|center"
    assert run.stage(0, "post-after").evaluations == 6
    for iteration in range(1, 5):
        selected = selected_at(run, iteration)
        assert len(selected) == 2 and all(
            candidate_fitness(p) == fitness for p in selected
        )
        assert np.all(np.isfinite(normalized_steps(run, iteration)))
        assert np.all(np.isfinite(weights_for(5) @ normalized_steps(run, iteration)))
        for phase in ("pre", "inter-before", "inter-after", "post-after"):
            stage = run.stage(iteration, phase)
            assert_finite_geometry(stage)
            assert_distribution_equal(
                stage.distribution, finite_run.stage(iteration, phase).distribution
            )
        pre = run.stage(iteration, "pre")
        post = run.stage(iteration, "post-after")
        assert post.best is not None and post.best.fitness == fitness
        assert post.center is not None and post.center.fitness == fitness
        block = run.calls[pre.evaluations : post.evaluations]
        assert [call.identifier for call in block] == (
            [center_id] if fitness == np.inf else []
        ) + [*run.algorithm.initial, center_id]
        initial_before = run.stage(iteration, "initialize-before", center_id)
        initial_after = run.stage(iteration, "initialize-after", center_id)
        assert initial_after.evaluations - initial_before.evaluations == int(
            fitness == np.inf
        )
        assert_distribution_equal(
            initial_before.distribution, initial_after.distribution
        )
        if fitness == np.inf:
            np.testing.assert_allclose(
                np.array(list(block[0].variables.values())),
                pre.distribution.mean,
                rtol=LINALG_RTOL,
                atol=LINALG_ATOL,
            )
        assert all(p.fitness == fitness for p in post.population.values())
    assert all(run.algorithm.rng_unchanged)
