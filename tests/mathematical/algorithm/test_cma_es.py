"""Positive CMA through Optimizer + Continuous + the real Serial replica.

AH: Akimoto & Hansen (2020), section 2.1, equations (1)-(11), (36)-(38):
https://arxiv.org/html/1905.05885v1 (fixed D=I, no active or diagonal update).
HP: Hansen et al. (2009), IV.B, equation (12), with the 2011 addendum:
https://www.cmap.polytechnique.fr/~nikolaus.hansen/TEC2009online.pdf
Tyrannis: initial centroid, range-scaled sigma, four boundary policies,
average tied weights including zero ranks, and feasible geometry for -inf.
Iteration zero is setup; iteration k>0 executes mathematical t=k-1.

Tie-dependent path normalization is a pending design decision. Tests of ties
protect recombination and rank-mu, without freezing that normalization choice.
"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from math import exp, log, sqrt
from typing import cast, override

import numpy as np
import pytest

from tests._support import factories
from tests._support.assertions import assert_variables_within_bounds
from tests._support.numerics import BASE_SEED, LINALG_ATOL, LINALG_RTOL, seed_for
from tests._support.objectives import CountingObjective, constant_objective, sphere
from tyrannis import Optimizer
from tyrannis.algorithm.cma_es import CMAES, CMAESCandidateSolution
from tyrannis.core.algorithm import ParticleBase
from tyrannis.processor.serial import Serial
from tyrannis.space.continuous import Continuous

Array = np.ndarray[tuple[int, ...], np.dtype[np.float64]]


@dataclass
class Distribution:
    mean: Array
    covariance: Array
    sigma: float
    p_sigma: Array
    p_c: Array
    gamma_sigma: float
    gamma_c: float
    weights: Array
    mu_eff: float
    cc: float
    cs: float
    c1: float
    cmu: float
    damps: float
    chi_n: float
    penalty: Array
    history: list[float]


@dataclass
class Generation:
    before: Distribution
    after: Distribution
    candidates: list[CMAESCandidateSolution]
    scores: Array


class ObservedCMAES(CMAES):
    """Record actual inputs/results without replacing equations or random draws."""

    def snapshot(self) -> Distribution:
        assert self._mean is not None and self._covariance is not None
        assert self._p_sigma is not None and self._p_c is not None
        assert self._weights is not None and self._mu_eff is not None
        assert self._cc is not None and self._cs is not None
        assert self._c1 is not None and self._cmu is not None
        assert self._damps is not None and self._chi_n is not None
        assert self._penalty_gamma is not None
        return Distribution(
            self._mean.copy(),
            self._covariance.copy(),
            self.sigma,
            self._p_sigma.copy(),
            self._p_c.copy(),
            self._gamma_sigma,
            self._gamma_c,
            self._weights.copy(),
            self._mu_eff,
            self._cc,
            self._cs,
            self._c1,
            self._cmu,
            self._damps,
            self._chi_n,
            self._penalty_gamma.copy(),
            self._penalty_history.copy(),
        )

    @override
    def pre_iteration(self, actual_iter: int) -> None:
        super().pre_iteration(actual_iter)
        if actual_iter == 0:
            self.initial: Array = np.array(
                [list(p.variables.values()) for p in self.population.values()]
            )
            self.setup: Distribution = self.snapshot()
            self.generations: list[Generation] = []
            self.draws: list[Array] = []
            self.samples: list[tuple[Array, list[Array]]] = []
            self.cache_preserved: list[bool] = []
            self.rng_preserved: list[bool] = []
            self.scores: dict[str, float] = {}

    @override
    def _raw_candidate_from_z(self, z: Array, transform: Array) -> Array:
        result = super()._raw_candidate_from_z(z, transform)
        self.draws.append(z.copy())
        return result

    @override
    def update_particle(self, identifier: str) -> CMAESCandidateSolution:
        state = deepcopy(self._rng.bit_generator.state)
        keys = set(self.population[identifier].random_cache)
        self.draws = []
        result = super().update_particle(identifier)
        raw = result.raw_candidate_variables
        if raw is not None:
            self.samples.append((np.array(list(raw.values())), self.draws.copy()))
        self.cache_preserved.append(keys == set(result.random_cache))
        self.rng_preserved.append(state == self._rng.bit_generator.state)
        return result

    @override
    def _boundary_penalty_score(self, particle: CMAESCandidateSolution) -> float:
        score = super()._boundary_penalty_score(particle)
        self.scores[particle.identifier] = score
        return score

    @override
    def post_iteration(self, actual_iter: int) -> None:
        before = self.snapshot()
        candidates = deepcopy(list(self.population.values()))
        super().post_iteration(actual_iter)
        if actual_iter:
            scores = np.array(
                [
                    self.scores[p.identifier]
                    if self.boundary_handling == "adaptive-penalty"
                    else fitness(p)
                    for p in candidates
                ]
            )
            self.generations.append(
                Generation(before, self.snapshot(), candidates, scores)
            )
        else:
            np.testing.assert_array_equal(self.setup.mean, self.snapshot().mean)
            assert self.snapshot().gamma_sigma == self.snapshot().gamma_c == 0
        assert not self.double_particle_check
        assert all(
            not p.random_cache and p.raw_candidate_variables is None
            for p in self.population.values()
        )


def run_cma(
    monkeypatch: pytest.MonkeyPatch,
    bounds: dict[str, tuple[float, float]],
    *,
    objective: Callable[..., float] = sphere,
    sigma: float = 0.3,
    policy: str = "adaptive-penalty",
    seed: int = BASE_SEED,
    iterations: int = 4,
    population: int = 5,
) -> ObservedCMAES:
    calls: list[dict[str, float]] = []
    updates: list[dict[str, float]] = []

    def recorded(**variables: float) -> float:
        assert_variables_within_bounds(variables, bounds)
        calls.append(variables.copy())
        return objective(**variables)

    real_update = ParticleBase.update

    def observed_update(
        particle: ParticleBase,
        variables: dict[str, float],
        fitness_function: Callable[[dict[str, float]], np.float64],
    ) -> None:
        updates.append(variables.copy())
        real_update(particle, variables, fitness_function)

    processor = Serial()
    with monkeypatch.context() as patch:
        patch.setattr(ParticleBase, "update", observed_update)
        # Existing shared factory has unparameterized AlgorithmBase annotations.
        factory = cast(Callable[..., Optimizer], factories.make_optimizer)
        _ = factory(
            Continuous(bounds, cost_function=recorded, use_cache=False),
            ObservedCMAES(sigma=sigma, boundary_handling=policy),
            processor=processor,
            n_particles=population,
            n_iterations=iterations,
            seed=seed,
            fitness_failure_strategy="raise",
        ).fit()
    (executor,) = processor.processors_pool.values()
    # Replica's algorithm accessor is private and its generic type is erased.
    algorithm = cast(object, vars(executor)["_algorithm"])
    assert isinstance(algorithm, ObservedCMAES)
    assert calls == updates and len(calls) == population * (iterations + 1)
    assert all(algorithm.cache_preserved) and all(algorithm.rng_preserved)
    return algorithm


def fitness(particle: CMAESCandidateSolution) -> float:
    assert particle.candidate_fitness is not None
    return float(particle.candidate_fitness)


def point(particle: CMAESCandidateSolution, *, raw: bool = False) -> Array:
    values = particle.raw_candidate_variables if raw else particle.candidate_variables
    assert values is not None
    return np.array(list(values.values()))


def weights_for(size: int) -> Array:
    """AH positive logarithmic weights, padded with zero ranks."""
    ranks = np.maximum(0, [log((size + 1) / (2 * i)) for i in range(1, size + 1)])
    return ranks / sum(ranks)


def tied_weights(scores: Array) -> Array:
    """Footnote 2: average over the entire occupied rank interval, order-free."""
    ranks = weights_for(len(scores))
    values = list(map(float, scores))
    return np.array(
        [
            np.mean(
                ranks[sum(v < score for v in values) : sum(v <= score for v in values)]
            )
            for score in values
        ]
    )


def close(actual: Array | float, expected: Array | float) -> None:
    np.testing.assert_allclose(actual, expected, rtol=LINALG_RTOL, atol=LINALG_ATOL)


def geometry(generation: Generation, policy: str) -> tuple[Array, Array]:
    positions = np.stack(
        [
            point(
                p,
                raw=policy != "clip"
                and not (policy == "adaptive-penalty" and fitness(p) == -np.inf),
            )
            for p in generation.candidates
        ]
    )
    return (positions - generation.before.mean) / generation.before.sigma, tied_weights(
        generation.scores
    )


def assert_recombination(generation: Generation, policy: str) -> None:
    old, new = generation.before, generation.after
    steps, weights = geometry(generation, policy)
    close(new.mean, old.mean + old.sigma * (weights @ steps))
    # Isolate rank-mu using the observed path, so this does not decide how
    # tied weights should normalize the paths (an explicitly pending decision).
    expected = (1 - old.c1 * new.gamma_c - old.cmu) * old.covariance
    expected += old.c1 * np.outer(new.p_c, new.p_c)
    expected += old.cmu * ((steps.T * weights) @ steps)
    close(new.covariance, expected)


def test_setup_parameters_and_sampling(
    monkeypatch: pytest.MonkeyPatch, small_bounds: dict[str, tuple[float, float]]
) -> None:
    run = run_cma(monkeypatch, small_bounds, policy="reflection")
    initial = run.setup
    centroid = np.ones(len(run.initial)) @ run.initial / len(run.initial)
    close(initial.mean, centroid)
    close(initial.covariance, np.eye(2))
    close(initial.p_sigma, np.zeros(2))
    close(initial.p_c, np.zeros(2))
    close(initial.sigma, 0.3 * (10 + 6) / 2)
    nominal = weights_for(5)[:2]
    mass = 1 / float(nominal @ nominal)
    cs = (mass + 2) / (mass + 7)
    c1 = 1 / (5 * 3**0.75 + mass / 2)
    close(initial.weights, nominal)
    close(
        np.array(
            [
                initial.mu_eff,
                initial.cs,
                initial.damps,
                initial.cc,
                initial.c1,
                initial.cmu,
                initial.chi_n,
            ]
        ),
        np.array(
            [
                mass,
                cs,
                1 + cs,
                sqrt(mass * c1) / 2,
                c1,
                min((mass + 1 / mass - 2 + 0.25) * c1, 1 - c1),
                sqrt(2) * (1 - 1 / 8 + 1 / 84),
            ]
        ),
    )
    for index, generation in enumerate(run.generations):
        old = generation.before
        eigenvalues, basis = np.linalg.eigh(old.covariance)
        factor = basis * np.sqrt(eigenvalues)
        close(factor @ factor.T, old.covariance)
        for raw, draws in run.samples[index * 5 : (index + 1) * 5]:
            assert len(draws) == 1
            close(raw, old.mean + old.sigma * (factor @ draws[0]))


@pytest.mark.parametrize("stall", [False, True], ids=["ordinary", "h-sigma-zero"])
def test_mean_csa_and_covariance_transitions(
    monkeypatch: pytest.MonkeyPatch,
    small_bounds: dict[str, tuple[float, float]],
    stall: bool,
) -> None:
    bounds = {"x": (20.0, 30.0), "y": (10.0, 16.0)} if stall else small_bounds
    run = run_cma(
        monkeypatch,
        bounds,
        sigma=0.02 if stall else 0.3,
        policy="reflection",
        seed=seed_for(2) if stall else BASE_SEED,
    )
    gates: set[int] = set()
    for k, generation in enumerate(run.generations, 1):
        old, new = generation.before, generation.after
        assert len(set(generation.scores)) == 5
        steps, weights = geometry(generation, "reflection")
        weighted = weights @ steps
        eigenvalues, basis = np.linalg.eigh(old.covariance)
        inverse = (basis * eigenvalues**-0.5) @ basis.T
        ps = (1 - old.cs) * old.p_sigma + sqrt(old.cs * (2 - old.cs) * old.mu_eff) * (
            inverse @ weighted
        )
        gs = 1 - (1 - old.cs) ** (2 * k)
        gate = int(float(ps @ ps) < gs * 2 * (2 + 4 / 3))
        gates.add(gate)
        pc = (1 - old.cc) * old.p_c + gate * sqrt(
            old.cc * (2 - old.cc) * old.mu_eff
        ) * weighted
        gc = (1 - old.cc) ** 2 * old.gamma_c + gate * (1 - (1 - old.cc) ** 2)
        close(new.p_sigma, ps)
        close(new.gamma_sigma, gs)
        close(new.p_c, pc)
        close(new.gamma_c, gc)
        close(
            log(new.sigma / old.sigma),
            old.cs / old.damps * (float(np.linalg.norm(ps)) / old.chi_n - sqrt(gs)),
        )
        assert_recombination(generation, "reflection")
        close(new.covariance, new.covariance.T)
        assert np.all(np.linalg.eigvalsh(new.covariance) > 0)
    assert gates == ({0, 1} if stall else {1})


@pytest.mark.parametrize(
    "policy", ["adaptive-penalty", "reflection", "resampling", "clip"]
)
def test_boundary_evaluation_and_adaptation(
    monkeypatch: pytest.MonkeyPatch,
    small_bounds: dict[str, tuple[float, float]],
    policy: str,
) -> None:
    run = run_cma(monkeypatch, small_bounds, sigma=2.0, policy=policy, iterations=3)
    lower = np.array([limits[0] for limits in small_bounds.values()])
    upper = np.array([limits[1] for limits in small_bounds.values()])
    repairs = rejected = multiple_widths = 0
    for k, generation in enumerate(run.generations):
        old = generation.before
        eigenvalues, basis = np.linalg.eigh(old.covariance)
        factor = basis * np.sqrt(eigenvalues)
        for particle, (raw, draws) in zip(
            generation.candidates, run.samples[5 * k : 5 * (k + 1)], strict=True
        ):
            evaluated = point(particle)
            close(raw, old.mean + old.sigma * (factor @ draws[-1]))
            repairs += int(not np.array_equal(raw, evaluated))
            multiple_widths += int(np.any(np.abs(raw - evaluated) > upper - lower))
            if policy == "reflection":
                # Repeated mirror operations are an independent oracle for the
                # periodic folding formula, including multi-width excursions.
                expected = raw.copy()
                while np.any(expected < lower) or np.any(expected > upper):
                    expected = np.where(
                        expected < lower, 2 * lower - expected, expected
                    )
                    expected = np.where(
                        expected > upper, 2 * upper - expected, expected
                    )
            elif policy == "resampling":
                expected = raw
                for z in draws[:-1]:
                    proposal = old.mean + old.sigma * (factor @ z)
                    assert np.any(proposal < lower) or np.any(proposal > upper)
                    rejected += 1
            else:
                expected = np.clip(raw, lower, upper)
            close(evaluated, expected)
            close(
                fitness(particle),
                sphere(**dict(zip(small_bounds, map(float, evaluated), strict=True))),
            )
        assert_recombination(generation, policy)
    if policy == "resampling":
        assert rejected > 0 and repairs == 0
        repeated = run_cma(
            monkeypatch, small_bounds, sigma=2.0, policy=policy, iterations=3
        )
        for first, second in zip(run.generations, repeated.generations, strict=True):
            np.testing.assert_array_equal(first.after.mean, second.after.mean)
    else:
        assert repairs > 0 and multiple_widths > 0


def test_ties_preserve_all_rank_mass_and_ignore_candidate_order(
    monkeypatch: pytest.MonkeyPatch, small_bounds: dict[str, tuple[float, float]]
) -> None:
    def staircase(**variables: float) -> float:
        return float(variables["x"] >= -2)

    run = run_cma(
        monkeypatch, small_bounds, objective=staircase, policy="clip", population=8
    )
    blocks: set[str] = set()
    for generation in run.generations:
        scores = generation.scores
        for value in set(map(float, scores)):
            first, last = (
                int(np.count_nonzero(scores < value)),
                int(np.count_nonzero(scores <= value)),
            )
            if last - first > 1:
                blocks.add(
                    "above" if first >= 4 else "inside" if last <= 4 else "crossing"
                )
        weights = tied_weights(scores)
        close(float(np.sum(weights)), 1.0)
        positions = np.stack([point(p) for p in generation.candidates])
        # Both permutations must describe the same production mean.
        close(generation.after.mean, tied_weights(scores[::-1]) @ positions[::-1])
        assert_recombination(generation, "clip")
    assert blocks == {"above", "inside", "crossing"}


def test_adaptive_penalty_matches_published_scale_and_coefficients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bounds = {"x": (0.1, 2.0), "y": (-2.0, 2.0)}
    run = run_cma(monkeypatch, bounds, sigma=1.0, iterations=8)
    lower = np.array([limits[0] for limits in bounds.values()])
    upper = np.array([limits[1] for limits in bounds.values()])
    activated = penalized = False
    history: list[float] = []
    for k, generation in enumerate(run.generations, 1):
        old, new = generation.before, generation.after
        values = np.array([fitness(p) for p in generation.candidates])
        iqr = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
        scale = iqr / (old.sigma**2 * sum(map(float, np.diag(old.covariance))) / 2)
        if scale > 0:
            history.insert(0, scale)
        close(np.array(new.history), np.array(history))
        typical = float(np.median(history))
        violation = np.abs(old.mean - np.clip(old.mean, lower, upper))
        gamma = old.penalty.copy()
        if np.any(violation) and (not activated or k == 2):
            gamma = np.full(2, 2 * typical)
            activated = True
        if activated:
            distance = violation / (old.sigma * np.sqrt(np.diag(old.covariance)))
            excess = np.maximum(0, distance - 3 * max(1, sqrt(2) / old.mu_eff))
            damping = min(1, old.mu_eff / 20)
            gamma *= np.exp(damping * np.tanh(excess / 3) / 2)
            gamma = np.where(gamma > 5 * typical, gamma * exp(-damping / 3), gamma)
        close(new.penalty, gamma)
        penalties = np.array(
            [
                float(gamma @ (point(p, raw=True) - point(p)) ** 2) / 2
                for p in generation.candidates
            ]
        )
        close(generation.scores, values + penalties)
        penalized |= bool(np.any(penalties > 0))
        assert_recombination(generation, "adaptive-penalty")
    assert activated and penalized


@pytest.mark.parametrize(
    "value", [1.0, np.inf, -np.inf], ids=["flat", "positive-inf", "negative-inf"]
)
def test_full_ties_and_infinite_fitness_have_finite_geometry(
    monkeypatch: pytest.MonkeyPatch,
    small_bounds: dict[str, tuple[float, float]],
    counting_objective: CountingObjective,
    value: float,
) -> None:
    counting_objective.return_value = value
    run = run_cma(
        monkeypatch, small_bounds, objective=counting_objective, sigma=2.0, iterations=2
    )
    assert counting_objective.calls == 15
    for generation in run.generations:
        np.testing.assert_array_equal(generation.scores, np.full(5, value))
        np.testing.assert_array_equal(generation.after.penalty, np.zeros(2))
        assert generation.after.history == []
        assert_recombination(generation, "adaptive-penalty")
        assert np.all(np.isfinite(generation.after.covariance))
        assert np.isfinite(generation.after.sigma) and generation.after.sigma > 0


def test_mixed_infinities_use_feasible_negative_infinity_geometry(
    monkeypatch: pytest.MonkeyPatch, small_bounds: dict[str, tuple[float, float]]
) -> None:
    def regions(**variables: float) -> float:
        if variables["x"] < 0:
            return -np.inf
        return np.inf if variables["x"] > 2 else sphere(**variables)

    run = run_cma(monkeypatch, small_bounds, objective=regions, sigma=1.0, iterations=4)
    saw_finite = saw_positive = repaired_negative = False
    for generation in run.generations:
        assert_recombination(generation, "adaptive-penalty")
        for p, score in zip(
            generation.candidates, map(float, generation.scores), strict=True
        ):
            value = fitness(p)
            if np.isfinite(value):
                saw_finite = True
                assert score >= value
            else:
                assert score == value
                saw_positive |= value == np.inf
                repaired_negative |= value == -np.inf and not np.array_equal(
                    point(p), point(p, raw=True)
                )
        assert all(np.isfinite(item) and item > 0 for item in generation.after.history)
    assert saw_finite and saw_positive and repaired_negative


def test_penalty_arithmetic_preserves_representable_scale_equivalence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ordinary = run_cma(
        monkeypatch,
        {"x": (-1.0, 1.0), "y": (-1.0, 1.0)},
        objective=constant_objective,
        sigma=2.0,
        iterations=1,
    )
    scale = 1e160
    extreme = run_cma(
        monkeypatch,
        {"x": (-scale, scale), "y": (-scale, scale)},
        objective=constant_objective,
        sigma=2.0,
        iterations=1,
    )
    close(extreme.generations[0].after.mean / scale, ordinary.generations[0].after.mean)
    close(
        extreme.generations[0].after.covariance,
        ordinary.generations[0].after.covariance,
    )
    np.testing.assert_array_equal(extreme.generations[0].scores, np.ones(5))

    # HP's normalized spread must survive sigma**2 overflow as well. Scale
    # objective values with the coordinates so gamma itself remains normal.
    scale = 1e160
    bounds = {"x": (0.1, 2.0), "y": (-2.0, 2.0)}
    reference = run_cma(monkeypatch, bounds, sigma=1.0, iterations=8)

    def scaled_sphere(**variables: float) -> float:
        return scale * sphere(
            **{name: value / scale for name, value in variables.items()}
        )

    scaled = run_cma(
        monkeypatch,
        {name: (lo * scale, hi * scale) for name, (lo, hi) in bounds.items()},
        objective=scaled_sphere,
        sigma=1.0,
        iterations=8,
    )
    assert any(np.any(g.after.penalty > 0) for g in reference.generations)
    for expected, actual in zip(reference.generations, scaled.generations, strict=True):
        close(actual.after.penalty * scale, expected.after.penalty)
        close(np.array(actual.after.history) * scale, np.array(expected.after.history))
        close(actual.scores / scale, expected.scores)
        close(actual.after.mean / scale, expected.after.mean)
        close(actual.after.covariance, expected.after.covariance)
