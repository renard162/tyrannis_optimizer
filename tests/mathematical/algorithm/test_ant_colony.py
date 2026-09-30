"""ACOR equations through Optimizer + Continuous + the actual Serial replica.

Socha & Dorigo (2008), section 3.3, eqs. (7)-(9), and Appendix A, eq. (A.1):
https://iridia.ulb.ac.be/~mdorigo/Published_papers/All_Dorigo_papers/SocDor2008ejor.pdf
Liao, Molina, Stutzle, Montes de Oca & Dorigo (2012), sections 1-3:
https://doi.org/10.1145/2330784.2330809
Author-deposited text consulted:
https://www.researchgate.net/publication/254462256
The latter calls the unrotated variant ACOR and the rotated variant ACOR-vch;
Tyrannis calls the former Sep-ACOR. Its equations (1)-(3) give the same kernel.

actual_iter=0 builds A_0; iteration i>0 samples from A_(i-1), then builds A_i.
Unconditional current-particle replacement, temporary initialization, random
tie order and historical local_best follow the documented Tyrannis lifecycle.
Clamping follows Liao section 3. Gaussian completion of a degenerate basis is
the project's numerical realization of Appendix A's random missing directions.

Migration scope: Local uses one island with NoCommunicationDriver; Optimizer
defaults to IslandIsolation. Real arrival requires the migration backend and
ProcessorBase._insert_arrival_particle. AntColony.pre_iteration admits evaluated
new particles when fitness != FITNESS_UNDEFINED, hence includes -inf and excludes
the +inf sentinel. No artificial arrivals/flags are injected here; transport and
the pre-sampling admission of an externally evaluated -inf belong to migration
tests. Canonical fallback after a degenerate random vector is a unit safeguard.
"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from inspect import currentframe
from math import exp, pi, sqrt
from typing import cast, override

import numpy as np
import pytest

from tests._support import factories
from tests._support.assertions import assert_variables_within_bounds
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
from tyrannis.algorithm.ant_colony import ACORParticle, AntColony
from tyrannis.core.algorithm import FITNESS_UNDEFINED, ParticleBase
from tyrannis.processor.serial import Serial
from tyrannis.space.continuous import Continuous

Array = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
Archive = list[tuple[dict[str, float], np.float64]]


@dataclass
class Axis:
    distances: Array
    uniform: float
    selected: int | None
    partial: Array
    direction: Array
    basis: Array


@dataclass
class Sample:
    iteration: int
    identifier: str
    archive: Archive
    probabilities: Array
    population: dict[str, ACORParticle]
    cache: dict[str, int | float]
    candidate: ACORParticle
    raw: Array
    sigma: dict[str, float]
    axes: list[Axis]
    fallbacks: dict[int, Array]


class ObservedAntColony(AntColony):
    """Only copy observations; production super() owns every transition."""

    _archive_probabilities: Array | None

    @override
    def pre_iteration(self, actual_iter: int) -> None:
        self.actual_iter: int = actual_iter
        if actual_iter == 0:
            self.initial: dict[str, ACORParticle] = deepcopy(self.population)
            self.initialized: dict[str, ACORParticle] = {}
            self.samples: list[Sample] = []
            self.archives: dict[int, Archive] = {}
            self.archive_refs: list[Archive] = []
            self.probabilities: dict[int, Array] = {}
            self.states: dict[int, dict[str, ACORParticle]] = {}
            self.bests: dict[int, ACORParticle] = {}
            self.independent: list[bool] = []
            self.sampling_rng_unchanged: list[bool] = []
        super().pre_iteration(actual_iter)
        if actual_iter == 0:
            self.setup_population: dict[str, ACORParticle] = deepcopy(self.population)
            self.temporary_ids: list[str] = self._temporary_particle_ids.copy()
            self.selected_ids: list[str] = self._initial_archive_particle_ids.copy()
            self.archive_before_setup: Archive = deepcopy(self._solution_archive)

    @override
    def initialize_particle(self, identifier: str) -> ACORParticle:
        particle = super().initialize_particle(identifier)
        self.initialized[identifier] = deepcopy(particle)
        return particle

    @override
    def update_particle(self, identifier: str) -> ACORParticle:
        assert self._archive_probabilities is not None
        archive = deepcopy(self._solution_archive)
        probabilities = self._archive_probabilities.copy()
        population = deepcopy(self.population)
        cache = self.population[identifier].random_cache.copy()
        rng = deepcopy(self._rng.bit_generator.state)
        self.observed_sigmas: dict[str, float] = {}
        self.observed_axes: list[Axis] = []
        self.observed_fallbacks: dict[int, Array] = {}
        raw: list[float] = []
        real_clip = np.clip

        def observed_clip(a: float, a_min: float, a_max: float) -> np.float64:
            raw.append(float(a))
            # NumPy's scalar clip overload is Any; these float64 scalar inputs
            # return a float64 scalar. Preserve the actual returned object.
            return cast(np.float64, real_clip(a, a_min, a_max))

        # A scoped forwarding spy observes the otherwise inaccessible raw sample.
        # It calls the original clip unchanged; no supplied draw/state is replaced.
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(np, "clip", observed_clip)
            particle = super().update_particle(identifier)
        self.sampling_rng_unchanged.append(rng == self._rng.bit_generator.state)
        self.samples.append(
            Sample(
                self.actual_iter,
                identifier,
                archive,
                probabilities,
                population,
                cache,
                deepcopy(particle),
                np.array(raw),
                self.observed_sigmas.copy(),
                self.observed_axes.copy(),
                self.observed_fallbacks.copy(),
            )
        )
        return particle

    @override
    def _calculate_sigma(self, archive_index: int, variable: str) -> float:
        sigma = super()._calculate_sigma(archive_index, variable)
        self.observed_sigmas[variable] = sigma
        return sigma

    @override
    def _generate_correlated_solution(self, particle: ACORParticle) -> dict[str, float]:
        # Static numerical helpers cannot write instance observations themselves.
        # These scoped spies forward to super's actual helpers, without changing
        # arguments, results, RNG state, or their call order.
        # Production's bare ndarray annotations erase dtype/shape. These casts
        # specialize that existing boundary to its actual float64 arrays.
        select = cast(
            Callable[[Array, float], int | None], super()._select_direction_index
        )
        orthonormalize = cast(Callable[[Array, Array], Array], super()._orthonormalize)
        choices: list[tuple[Array, float, int | None]] = []

        def observed_select(distances: Array, random_value: float) -> int | None:
            selected = select(distances, random_value)
            choices.append((distances.copy(), random_value, selected))
            return selected

        def observed_basis(basis: Array, direction: Array) -> Array:
            result: Array = orthonormalize(basis, direction)
            distances, uniform, selected = choices[-1]
            self.observed_axes.append(
                Axis(
                    distances,
                    uniform,
                    selected,
                    basis.copy(),
                    direction.copy(),
                    result.copy(),
                )
            )
            return result

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                ObservedAntColony,
                "_select_direction_index",
                staticmethod(observed_select),
            )
            patch.setattr(
                ObservedAntColony, "_orthonormalize", staticmethod(observed_basis)
            )
            return super()._generate_correlated_solution(particle)

    @override
    def _fallback_direction(
        self,
        particle: ACORParticle,
        step: int,
        variables: tuple[str, ...],
        basis: Array,
    ) -> Array:
        # Same unparameterized production-array boundary as the static helpers.
        fallback = cast(
            Callable[[ACORParticle, int, tuple[str, ...], Array], Array],
            super()._fallback_direction,
        )
        result = fallback(particle, step, variables, basis)
        self.observed_fallbacks[step] = result.copy()
        return result

    @override
    def post_iteration(self, actual_iter: int) -> None:
        before = list(self.population.values())
        super().post_iteration(actual_iter)
        self.independent.append(
            all(
                variables is not p.variables
                for variables, _ in self._solution_archive
                for p in before
            )
        )
        self.archives[actual_iter] = deepcopy(self._solution_archive)
        # Keep live references too: later production moves must not mutate them.
        self.archive_refs.append(self._solution_archive)
        assert self._archive_probabilities is not None
        self.probabilities[actual_iter] = self._archive_probabilities.copy()
        self.states[actual_iter] = deepcopy(self.population)
        assert self.local_best is not None
        self.bests[actual_iter] = deepcopy(self.local_best)


@dataclass
class Run:
    algorithm: ObservedAntColony
    evaluations: list[tuple[dict[str, float], float]]


def run_ants(
    bounds: dict[str, tuple[float, float]],
    *,
    objective: Callable[..., float] = sphere,
    archive_size: int | str = 4,
    q: float = 0.25,
    xi: float = 0.85,
    separable: bool = True,
    particles: int = 4,
    iterations: int = 3,
    seed: int = BASE_SEED,
) -> Run:
    evaluations: list[tuple[dict[str, float], float]] = []
    evaluation_ids: list[str] = []

    def recorded_objective(**variables: float) -> float:
        # Read the real evaluation frame; ParticleBase.update is never replaced.
        # Introspection's locals are untyped, so narrow the observed object.
        frame = currentframe()
        try:
            while (
                frame is not None and frame.f_code is not ParticleBase.update.__code__
            ):
                frame = frame.f_back
            assert frame is not None, "Evaluation bypassed ParticleBase.update"
            particle = cast(object, frame.f_locals["self"])
            assert isinstance(particle, ACORParticle)
            evaluation_ids.append(particle.identifier)
        finally:
            del frame
        fitness = objective(**variables)
        evaluations.append((variables.copy(), fitness))
        return fitness

    template = ObservedAntColony(archive_size, q, xi, separable)
    processor = Serial()
    # Shared factory erases the particle generic; retain its actual return type.
    factory = cast(Callable[..., Optimizer], factories.make_optimizer)
    optimizer = factory(
        Continuous(bounds, cost_function=recorded_objective, use_cache=False),
        template,
        processor=processor,
        n_particles=particles,
        n_iterations=iterations,
        seed=seed,
        fitness_failure_strategy="raise",
    ).fit()
    (executor,) = processor.processors_pool.values()
    # The public pool exposes the replica, but its algorithm has no accessor.
    # The stored AlgorithmBase also erases its particle generic at this boundary.
    algorithm = cast(object, vars(executor)["_algorithm"])
    assert isinstance(algorithm, ObservedAntColony)
    assert algorithm is not template and not template.population
    assert set(algorithm.states) == set(range(iterations + 1))
    assert len(algorithm.samples) == particles * iterations
    assert all(sample.iteration > 0 for sample in algorithm.samples)
    assert all(algorithm.independent)
    assert all(algorithm.sampling_rng_unchanged)
    for i, archive in enumerate(algorithm.archive_refs):
        assert archive == algorithm.archives[i]
    assert algorithm.local_best is not None
    assert optimizer.best_fitness == algorithm.local_best.fitness
    assert len(evaluation_ids) == len(evaluations)
    assert isinstance(algorithm.archive_size, int)
    k = algorithm.archive_size
    initial_count = max(k, particles)
    assert len(evaluations) == initial_count + particles * iterations
    assert evaluation_ids[:initial_count] == list(algorithm.initialized)
    for i, sample in enumerate(algorithm.samples):
        assert sample.archive == algorithm.archives[sample.iteration - 1]
        np.testing.assert_array_equal(
            sample.probabilities, algorithm.probabilities[sample.iteration - 1]
        )
        assert evaluation_ids[initial_count + i] == sample.identifier
        candidate = sample.candidate
        assert candidate.candidate_variables is not None
        assert candidate.candidate_fitness is not None
        assert (
            candidate.candidate_variables,
            candidate.candidate_fitness,
        ) == evaluations[initial_count + i]
        current = algorithm.states[sample.iteration][sample.identifier]
        assert (current.variables, current.fitness) == (
            candidate.candidate_variables,
            candidate.candidate_fitness,
        )
        assert not current.new_particle
    for iteration, archive in algorithm.archives.items():
        state = algorithm.states[iteration]
        assert len(state) == particles
        assert len(archive) == k
        for variables, fitness in archive:
            assert_variables_within_bounds(variables, bounds)
            assert np.all(np.isfinite(list(variables.values())))
            assert not np.isnan(fitness)
            assert fitness == pytest.approx(
                objective(**variables), rel=STRICT_RTOL, abs=STRICT_ATOL
            )
        np.testing.assert_allclose(
            algorithm.probabilities[iteration],
            rank_probabilities(k, q),
            rtol=STRICT_RTOL,
            atol=STRICT_ATOL,
        )
        seen = evaluations[: initial_count + iteration * particles]
        assert algorithm.bests[iteration].fitness == min(f for _, f in seen)
        assert (
            algorithm.bests[iteration].variables,
            algorithm.bests[iteration].fitness,
        ) in seen
        if iteration:
            assert_archive_union(
                archive,
                algorithm.archives[iteration - 1]
                + [(p.variables, p.fitness) for p in state.values()],
                k,
            )
    return Run(algorithm, evaluations)


def rank_probabilities(k: int, q: float) -> Array:
    """Socha (7)-(8): the common normal density prefactor cancels."""
    weights = [
        exp(-0.5 * (rank / (q * k)) ** 2) / (q * k * sqrt(2 * pi)) for rank in range(k)
    ]
    return np.array([weight / sum(weights) for weight in weights])


def assert_setup(run: Run, k: int, particles: int) -> None:
    ants = run.algorithm
    assert ants.archive_before_setup == []
    assert len(ants.selected_ids) == len(set(ants.selected_ids)) == k
    assert set(ants.initialized) == set(ants.setup_population)
    assert len(run.evaluations) == max(k, particles) + len(ants.samples)
    initial_solutions: Archive = []
    for identifier, evaluated in ants.initialized.items():
        assert evaluated.candidate_variables is not None
        assert (
            evaluated.candidate_variables == ants.setup_population[identifier].variables
        )
        assert evaluated.candidate_fitness is not None
        assert evaluated.random_cache == {}
        assert (
            evaluated.candidate_variables,
            evaluated.candidate_fitness,
        ) in run.evaluations[: max(k, particles)]
        if identifier in ants.selected_ids:
            initial_solutions.append(
                (evaluated.candidate_variables, evaluated.candidate_fitness)
            )
    archive = ants.archives[0]
    assert len(archive) == k
    assert sorted(f for _, f in initial_solutions) == [f for _, f in archive]
    assert all(solution in initial_solutions for solution in archive)
    assert set(ants.states[0]) == set(ants.initial)
    assert len(ants.states[0]) == particles
    np.testing.assert_allclose(
        ants.probabilities[0],
        rank_probabilities(k, ants.q),
        rtol=STRICT_RTOL,
        atol=STRICT_ATOL,
    )
    assert sum(ants.probabilities[0]) == pytest.approx(
        1, rel=STRICT_RTOL, abs=STRICT_ATOL
    )
    assert np.all(np.diff(ants.probabilities[0]) < 0)


def test_setup_builds_ranked_snapshots_and_normalized_gaussian_weights(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    run = run_ants(small_bounds, iterations=1)
    assert_setup(run, k=4, particles=4)
    for variables, fitness in run.algorithm.archives[0]:
        assert fitness == pytest.approx(
            sphere(**variables), rel=STRICT_RTOL, abs=STRICT_ATOL
        )
    assert run.algorithm.bests[0].fitness == min(
        f for _, f in run.algorithm.archives[0]
    )


def test_nondefault_q_changes_gaussian_rank_pressure(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    run = run_ants(small_bounds, q=0.6, iterations=1)
    assert_setup(run, k=4, particles=4)
    assert run.algorithm.probabilities[0][0] < rank_probabilities(4, 0.25)[0]


def assert_separable_sample(
    sample: Sample,
    bounds: dict[str, tuple[float, float]],
    xi: float,
) -> Array:
    """Socha (9) and Liao (3), then the affine standard-normal sample."""
    index = sample.cache["archive-index"]
    assert isinstance(index, int) and 0 <= index < len(sample.archive)
    reference = sample.archive[index][0]
    sigmas: list[float] = []
    expected_raw: list[float] = []
    for name in bounds:
        sigma = (
            xi
            * sum(abs(x[name] - reference[name]) for x, _ in sample.archive)
            / (len(sample.archive) - 1)
        )
        sigmas.append(sigma)
        assert sample.sigma[name] == pytest.approx(
            sigma, rel=STRICT_RTOL, abs=STRICT_ATOL
        )
        expected_raw.append(reference[name] + sigma * sample.cache[f"{name}-normal"])
    np.testing.assert_allclose(
        sample.raw, expected_raw, rtol=STRICT_RTOL, atol=STRICT_ATOL
    )
    assert_clipped_sample(sample, bounds)
    return np.array(sigmas)


def assert_clipped_sample(
    sample: Sample, bounds: dict[str, tuple[float, float]]
) -> None:
    candidate = sample.candidate.candidate_variables
    assert candidate is not None
    expected = [
        min(upper, max(lower, float(raw)))
        for raw, (lower, upper) in zip(sample.raw.flat, bounds.values(), strict=True)
    ]
    np.testing.assert_allclose(
        list(candidate.values()), expected, rtol=STRICT_RTOL, atol=STRICT_ATOL
    )
    assert np.all(np.isfinite(sample.raw))
    assert_variables_within_bounds(candidate, bounds)


@pytest.fixture
def separable_run(small_bounds: dict[str, tuple[float, float]]) -> Run:
    # This case witnesses both replacement of local_best and later preservation.
    return run_ants(small_bounds, q=0.6, xi=1.7, iterations=4, seed=seed_for(0))


def test_separable_sampling_uses_one_reference_and_dimensionwise_sigma(
    separable_run: Run,
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    ants = separable_run.algorithm
    clipped = interior = distinct_draws = unequal_sigmas = False
    nonbest_reference = False
    for sample in ants.samples:
        sigmas = assert_separable_sample(sample, small_bounds, xi=1.7)
        unequal_sigmas |= not np.isclose(
            sigmas.flat[0], sigmas.flat[1], rtol=STRICT_RTOL, atol=STRICT_ATOL
        )
        distinct_draws |= sample.cache["x-normal"] != sample.cache["y-normal"]
        nonbest_reference |= sample.cache["archive-index"] != 0
        for raw, (lower, upper) in zip(
            sample.raw.flat, small_bounds.values(), strict=True
        ):
            clipped |= raw < lower or raw > upper
            interior |= lower < raw < upper
        candidate = sample.candidate
        assert candidate.candidate_variables is not None
        assert candidate.candidate_fitness == pytest.approx(
            sphere(**candidate.candidate_variables), rel=STRICT_RTOL, abs=STRICT_ATOL
        )
    assert (
        clipped and interior and distinct_draws and unequal_sigmas and nonbest_reference
    )


def test_sampling_keeps_archive_and_probabilities_frozen_through_each_sweep(
    separable_run: Run,
) -> None:
    ants = separable_run.algorithm
    for sample in ants.samples:
        previous = sample.iteration - 1
        assert sample.archive == ants.archives[previous]
        np.testing.assert_array_equal(
            sample.probabilities, ants.probabilities[previous]
        )
        np.testing.assert_allclose(
            sample.probabilities,
            rank_probabilities(4, 0.6),
            rtol=STRICT_RTOL,
            atol=STRICT_ATOL,
        )
        for identifier, particle in sample.population.items():
            assert particle.variables == ants.states[previous][identifier].variables
            assert particle.fitness == ants.states[previous][identifier].fitness
    assert any(ants.archives[i] != ants.archives[i - 1] for i in range(1, 5))


def assert_archive_union(archive: Archive, combined: Archive, k: int) -> None:
    """Best-k fitness multiset plus admissible membership, even across ties."""
    assert len(archive) == k
    assert [f for _, f in archive] == sorted(f for _, f in combined)[:k]
    remaining = combined.copy()
    for solution in archive:
        assert solution in remaining
        remaining.remove(solution)


def test_population_accepts_worse_candidates_but_archive_is_elitist(
    separable_run: Run,
) -> None:
    ants = separable_run.algorithm
    improved = worse_rejected = admitted = removed = preserved = historical = False
    improved_admitted = updated_best = False
    best = min(f for _, f in ants.archives[0])
    for iteration in range(1, 5):
        previous, archive = ants.archives[iteration - 1], ants.archives[iteration]
        candidates: Archive = []
        for sample in (s for s in ants.samples if s.iteration == iteration):
            pending = sample.candidate
            assert pending.candidate_variables is not None
            assert pending.candidate_fitness is not None
            current = ants.states[iteration][sample.identifier]
            assert current.variables == pending.candidate_variables
            assert current.fitness == pending.candidate_fitness
            assert (
                current.candidate_variables is None
                and current.candidate_fitness is None
            )
            solution = (current.variables, current.fitness)
            candidates.append(solution)
            old = ants.states[iteration - 1][sample.identifier]
            improved |= current.fitness < old.fitness
            improved_admitted |= (
                current.fitness < old.fitness
                and solution in archive
                and solution not in previous
            )
            worse_rejected |= (
                current.fitness > old.fitness
                and solution not in archive
                and current.variables != old.variables
            )
        combined = previous + candidates
        assert_archive_union(archive, combined, 4)
        # With distinct fitness, identities as well as ranks are uniquely determined.
        assert len({f for _, f in combined}) == len(combined)
        assert archive == sorted(combined, key=lambda s: s[1])[:4]
        admitted |= any(s in archive and s not in previous for s in candidates)
        removed |= any(s not in archive for s in previous)
        preserved |= any(s in archive for s in previous)
        new_best = min(best, *(f for _, f in candidates))
        updated_best |= new_best < best
        best = new_best
        assert ants.bests[iteration].fitness == best
        historical |= best < min(p.fitness for p in ants.states[iteration].values())
    assert improved and worse_rejected and admitted and removed and preserved
    assert historical and improved_admitted and updated_best


def test_temporary_solutions_are_evaluated_archived_and_keep_historical_best(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    run = run_ants(
        small_bounds, particles=2, archive_size=4, iterations=3, seed=seed_for(0)
    )
    assert_setup(run, k=4, particles=2)
    ants = run.algorithm
    assert len(ants.temporary_ids) == 2
    temporary = [ants.initialized[key] for key in ants.temporary_ids]
    temporary_best = min(
        p.candidate_fitness for p in temporary if p.candidate_fitness is not None
    )
    assert temporary_best < min(p.fitness for p in ants.states[0].values())
    assert ants.bests[0].fitness == temporary_best
    for particle in temporary:
        assert particle.candidate_variables is not None
        assert particle.candidate_fitness == pytest.approx(
            sphere(**particle.candidate_variables), rel=STRICT_RTOL, abs=STRICT_ATOL
        )
        assert (
            particle.candidate_variables,
            particle.candidate_fitness,
        ) in ants.archives[0]
    for iteration, state in ants.states.items():
        assert len(state) == 2 and not set(state).intersection(ants.temporary_ids)
        assert ants.bests[iteration].fitness <= temporary_best


def assert_correlated_sample(
    sample: Sample,
    bounds: dict[str, tuple[float, float]],
    xi: float,
) -> Array:
    """Appendix A: projected d^4 roulette, Gram-Schmidt, then eq. (9)."""
    names = tuple(bounds)
    dimensions = len(names)
    archive = np.array([[x[name] for name in names] for x, _ in sample.archive])
    index = sample.cache["archive-index"]
    assert isinstance(index, int) and 0 <= index < len(sample.archive)
    reference = np.array([sample.archive[index][0][name] for name in names])
    differences = archive - reference
    assert len(sample.axes) == dimensions
    expected_axes: list[Array] = []
    coordinates: list[float] = []
    sigmas: list[float] = []
    for step, (name, observed) in enumerate(zip(names, sample.axes, strict=True)):
        partial = (
            np.column_stack(expected_axes)
            if expected_axes
            else np.empty((dimensions, 0))
        )
        # Independent orthogonal projector, with archive solutions as rows.
        projector = np.eye(dimensions) - partial @ partial.T
        residuals = differences @ projector
        distances = np.asarray(np.linalg.norm(residuals, axis=1), dtype=float)
        np.testing.assert_allclose(
            observed.partial, partial, rtol=LINALG_RTOL, atol=LINALG_ATOL
        )
        np.testing.assert_allclose(
            observed.distances, distances, rtol=LINALG_RTOL, atol=LINALG_ATOL
        )
        assert observed.uniform == sample.cache[f"direction-{step}-uniform"]
        assert 0 <= observed.uniform < 1

        # Check A.1 locally even at numerical rank loss, using the actual norms
        # as observed inputs. The geometric choice below is independently
        # reconstructed too whenever its relative CDF is numerically stable.
        if np.max(observed.distances) > 0:
            actual_weights = observed.distances**4
            actual_cdf = np.cumsum(actual_weights / np.sum(actual_weights))
            expected_index = next(
                i for i, edge in enumerate(actual_cdf.flat) if observed.uniform < edge
            )
            assert observed.selected == expected_index
        else:
            assert observed.selected is None
        if np.max(distances) > LINALG_ATOL:
            weights = distances**4
            cdf = np.cumsum(weights / np.sum(weights))
            selected = next(
                i for i, edge in enumerate(cdf.flat) if observed.uniform < edge
            )
            assert observed.selected == selected
            direction = np.asarray(residuals[selected], dtype=float)
            assert step not in sample.fallbacks
        else:
            # Mathematical rank loss, rather than copying the production's eps
            # threshold: every residual must vanish to the shared linear tolerance.
            np.testing.assert_allclose(residuals, 0, rtol=LINALG_RTOL, atol=LINALG_ATOL)
            assert step in sample.fallbacks
            raw_fallback = np.array(
                [sample.cache[f"fallback-{step}-{n}"] for n in names]
            )
            direction = projector @ raw_fallback
            np.testing.assert_allclose(
                sample.fallbacks[step], direction, rtol=LINALG_RTOL, atol=LINALG_ATOL
            )
        np.testing.assert_allclose(
            observed.direction, direction, rtol=LINALG_RTOL, atol=LINALG_ATOL
        )
        # Gram-Schmidt preserves the sign of the selected/fallback direction.
        orthogonal = projector @ direction
        assert np.linalg.norm(orthogonal) > LINALG_ATOL
        axis = orthogonal / np.linalg.norm(orthogonal)
        expected_axes.append(axis)
        basis = np.column_stack(expected_axes)
        np.testing.assert_allclose(
            observed.basis, basis, rtol=LINALG_RTOL, atol=LINALG_ATOL
        )
        np.testing.assert_allclose(
            observed.basis.T @ observed.basis,
            np.eye(step + 1),
            rtol=LINALG_RTOL,
            atol=LINALG_ATOL,
        )
        transformed = archive @ axis
        center = float(reference @ axis)
        sigma = (
            xi
            * sum(abs(float(z) - center) for z in transformed.flat)
            / (len(archive) - 1)
        )
        sigmas.append(sigma)
        coordinates.append(center + sigma * sample.cache[f"{name}-normal"])
    expected_raw = sum(
        (
            coordinate * axis
            for coordinate, axis in zip(coordinates, expected_axes, strict=True)
        ),
        start=np.zeros(dimensions),
    )
    np.testing.assert_allclose(
        sample.raw, expected_raw, rtol=LINALG_RTOL, atol=LINALG_ATOL
    )
    assert_clipped_sample(sample, bounds)
    return np.array(sigmas)


def test_correlated_minimal_archive_rotates_and_completes_degenerate_basis(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    run = run_ants(
        small_bounds,
        separable=False,
        archive_size=2,
        particles=2,
        iterations=3,
        xi=1.7,
        seed=seed_for(0),
    )
    assert_setup(run, k=2, particles=2)
    clipped = interior = rotated = fallback = False
    for sample in run.algorithm.samples:
        sigmas = assert_correlated_sample(sample, small_bounds, xi=1.7)
        rotated |= np.all(np.abs(sample.axes[0].basis) > LINALG_ATOL)
        fallback |= 1 in sample.fallbacks
        # Two points have no spread in the perpendicular direction.
        assert sigmas.flat[1] == pytest.approx(0, abs=LINALG_ATOL)
        for raw, (lower, upper) in zip(
            sample.raw.flat, small_bounds.values(), strict=True
        ):
            clipped |= raw < lower or raw > upper
            interior |= lower < raw < upper
    assert clipped and interior and rotated and fallback


def test_correlated_direction_roulette_uses_fourth_power_and_transformed_spread(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    # k=2 alone cannot distinguish d^4 from d or a hardcoded sole direction.
    run = run_ants(
        small_bounds,
        separable=False,
        archive_size=4,
        particles=4,
        iterations=3,
        xi=1.7,
        q=0.6,
    )
    discriminating = transformed_spread = False
    for sample in run.algorithm.samples:
        sigmas = assert_correlated_sample(sample, small_bounds, xi=1.7)
        first = sample.axes[0]
        linear_cdf = np.cumsum(first.distances / np.sum(first.distances))
        linear_choice = next(
            i for i, edge in enumerate(linear_cdf.flat) if first.uniform < edge
        )
        discriminating |= linear_choice != first.selected
        transformed_spread |= (
            sigmas.flat[0] > LINALG_ATOL and sigmas.flat[1] > LINALG_ATOL
        )
    assert discriminating and transformed_spread


def test_automatic_archive_size_supplies_geometry_for_three_dimensions(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    bounds = {**small_bounds, "z": (-2.0, 2.0)}

    with pytest.warns(UserWarning):
        run = run_ants(
            bounds,
            separable=False,
            archive_size="auto",
            particles=2,
            iterations=1,
        )

    assert_setup(run, k=3, particles=2)
    assert run.algorithm.archive_size == 3

    for sample in run.algorithm.samples:
        sigmas = assert_correlated_sample(sample, bounds, xi=0.85)
        assert 2 in sample.fallbacks
        assert sigmas.flat[2] == pytest.approx(0, abs=LINALG_ATOL)


@pytest.mark.parametrize(
    "fitness",
    [1.0, np.inf, -np.inf],
    ids=["finite-tie", "positive-infinity-tie", "negative-infinity-tie"],
)
def test_tied_fitness_preserves_rank_probabilities_and_coordinate_only_sampling(
    small_bounds: dict[str, tuple[float, float]],
    counting_objective: CountingObjective,
    fitness: float,
) -> None:
    counting_objective.return_value = fitness
    objective = constant_objective if np.isfinite(fitness) else counting_objective
    run = run_ants(small_bounds, objective=objective, iterations=3)
    assert_setup(run, k=4, particles=4)
    ants = run.algorithm
    # run_ants re-evaluates archive entries for the independent fitness assertion;
    # the recorded execution evaluations remain exactly N*(1+T).
    if not np.isfinite(fitness):
        assert counting_objective.calls == len(run.evaluations) + sum(
            map(len, ants.archives.values())
        )
    assert all(f == fitness for _, f in run.evaluations)
    assert np.isposinf(FITNESS_UNDEFINED) and not np.isneginf(FITNESS_UNDEFINED)
    for sample in ants.samples:
        _ = assert_separable_sample(sample, small_bounds, xi=0.85)
        assert len({f for _, f in sample.archive}) == 1
        assert np.all(np.isfinite(sample.probabilities))
    if np.isneginf(fitness):
        assert all(np.isneginf(best.fitness) for best in ants.bests.values())
        assert all(
            p.fitness != FITNESS_UNDEFINED
            for state in ants.states.values()
            for p in state.values()
        )
    elif np.isposinf(fitness):
        assert all(np.isposinf(best.fitness) for best in ants.bests.values())
    # Equal fitness does not mean coincident positions or a zero-width kernel.
    assert any(s.sigma["x"] > 0 and s.sigma["y"] > 0 for s in ants.samples)


def partially_infeasible(**variables: float) -> float:
    return np.inf if variables["x"] > 0 else sphere(**variables)


def signed_extremes(**variables: float) -> float:
    if variables["x"] < -1:
        return -np.inf
    return np.inf if variables["x"] > 1 else sphere(**variables)


@pytest.mark.parametrize(
    "objective",
    [partially_infeasible, signed_extremes],
    ids=["finite-positive", "negative-finite-positive"],
)
def test_mixed_infinities_are_ranked_by_sign_and_excluded_from_geometry(
    small_bounds: dict[str, tuple[float, float]],
    objective: Callable[..., float],
) -> None:
    run = run_ants(
        small_bounds,
        objective=objective,
        separable=False,
        archive_size=4,
        particles=4,
        iterations=3,
        q=0.6,
        seed=seed_for(0),
    )
    assert_setup(run, k=4, particles=4)
    ants = run.algorithm
    fitnesses = [f for _, f in ants.archives[0]]
    assert any(np.isfinite(f) for f in fitnesses)
    assert any(np.isposinf(f) for f in fitnesses)
    assert np.isposinf(fitnesses[-1])
    for sample in ants.samples:
        _ = assert_correlated_sample(sample, small_bounds, xi=0.85)
        assert np.all(np.isfinite(sample.probabilities))
    if objective is signed_extremes:
        assert np.isneginf(fitnesses[0])
        assert np.isposinf(FITNESS_UNDEFINED)
        for iteration, archive in ants.archives.items():
            assert np.isneginf(archive[0][1])
            assert np.isneginf(ants.bests[iteration].fitness)
            assert ants.bests[iteration].fitness != FITNESS_UNDEFINED
            # Once admitted, no finite/+inf candidate may displace the -inf tier.
            if iteration:
                old_count = sum(np.isneginf(f) for _, f in ants.archives[iteration - 1])
                assert sum(np.isneginf(f) for _, f in archive) >= old_count
