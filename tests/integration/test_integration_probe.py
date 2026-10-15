"""Unit contracts of the shared lifecycle probe, without a processor double."""

import json
import pickle
from copy import deepcopy
from typing import Literal

import numpy as np
import pytest
from _support.assertions import assert_particle_state_consistent

# The shared factory omits AlgorithmBase type parameters. Keep the concrete
# probe reference below; its return annotation is an existing helper limitation.
from _support.factories import (
    configure_algorithm,  # pyright: ignore[reportUnknownVariableType]
)
from _support.integration_probe import (
    IntegrationProbeAlgorithm,
    IntegrationProbeParticle,
)
from _support.numerics import BASE_SEED
from _support.objectives import CountingObjective

from tyrannis.core.algorithm import CostFunctionWrapperBase


def _initialize(algorithm: IntegrationProbeAlgorithm) -> None:
    algorithm.pre_iteration(0)
    for identifier in ("p0", "p1", "p2", "p3"):
        algorithm.create_particle(identifier)
    algorithm.create_random_cache(list(algorithm.population), initialize=True)
    particles = [
        algorithm.consolidate_new_particles(algorithm.initialize_particle(identifier))
        for identifier in algorithm.population
    ]
    algorithm.update_population(particles)
    algorithm.post_iteration(0)


def _update(algorithm: IntegrationProbeAlgorithm, iteration: int) -> None:
    algorithm.pre_iteration(iteration)
    algorithm.create_random_cache(list(algorithm.population), initialize=False)
    algorithm.update_population(
        [algorithm.update_particle(identifier) for identifier in algorithm.population]
    )
    if algorithm.double_particle_check:
        algorithm.inter_iteration(iteration)
        algorithm.create_random_cache(algorithm.double_check_ids, initialize=False)
        algorithm.update_population(
            [
                algorithm.second_update_particle(identifier)
                for identifier in algorithm.double_check_ids
            ]
        )
    algorithm.post_iteration(iteration)


@pytest.mark.parametrize("double_eval", [False, True], ids=["single", "double"])
def test_probe_records_only_selected_phases_and_evaluation_requests(
    small_bounds: dict[str, tuple[float, float]],
    counting_objective: CountingObjective,
    double_eval: bool,
) -> None:
    algorithm = IntegrationProbeAlgorithm(double_eval=double_eval)
    assert (
        configure_algorithm(
            algorithm, small_bounds, fitness_function=counting_objective
        )
        is algorithm
    )
    _initialize(algorithm)
    _update(algorithm, 1)

    assert counting_objective.calls == (10 if double_eval else 8)
    assert not algorithm.force_stop
    assert algorithm.double_check_ids == (["p0", "p2"] if double_eval else [])
    for identifier, particle in algorithm.population.items():
        phases = ["initialize", "update"]
        iterations = [0, 1]
        if double_eval and identifier in ("p0", "p2"):
            phases.append("second_update")
            iterations.append(1)
        assert [event["phase"] for event in particle.trace] == phases
        assert [event["iteration"] for event in particle.trace] == iterations
        assert [event["evaluation_count"] for event in particle.trace] == list(
            range(1, len(phases) + 1)
        )
        assert particle.evaluation_count == len(phases)
        for event in particle.trace:
            assert event["particle_id"] == event["cache_particle_id"] == identifier
            assert event["phase"] == event["cache_phase"]
            assert event["iteration"] == event["cache_iteration"]
        assert_particle_state_consistent(particle, small_bounds)
        assert particle.candidate_variables is None
        assert not particle.new_particle
        assert particle.random_cache == {}


@pytest.mark.parametrize("phase", ["initialize", "update", "second_update"])
def test_probe_processing_consumes_only_its_preserved_cache(
    small_bounds: dict[str, tuple[float, float]],
    monkeypatch: pytest.MonkeyPatch,
    phase: Literal["initialize", "update", "second_update"],
) -> None:
    algorithm = IntegrationProbeAlgorithm(double_eval=True)
    assert configure_algorithm(algorithm, small_bounds) is algorithm
    if phase == "initialize":
        algorithm.pre_iteration(0)
        for identifier in ("p0", "p1"):
            algorithm.create_particle(identifier)
    else:
        _initialize(algorithm)
        algorithm.pre_iteration(1)
        if phase == "second_update":
            algorithm.create_random_cache(list(algorithm.population), initialize=False)
            algorithm.update_population(
                [
                    algorithm.update_particle(identifier)
                    for identifier in algorithm.population
                ]
            )
            algorithm.inter_iteration(1)
    identifiers = (
        algorithm.double_check_ids
        if phase == "second_update"
        else list(algorithm.population)
    )
    algorithm.create_random_cache(identifiers, initialize=phase == "initialize")
    caches: list[dict[str, object]] = [
        algorithm.population[identifier].random_cache for identifier in identifiers
    ]
    generators = [cache["generator"] for cache in caches]
    assert len({id(cache) for cache in caches}) == len(caches)
    assert len({id(generator) for generator in generators}) == len(caches)
    assert len({id(cache["array"]) for cache in caches}) == len(caches)
    snapshots = [deepcopy(cache) for cache in caches]
    # pickle has an untyped return; narrow it before exercising the cached RNG.
    worker: object = pickle.loads(pickle.dumps(algorithm))  # pyright: ignore[reportAny]
    assert isinstance(worker, IntegrationProbeAlgorithm)
    # Workers must need neither the controller RNG nor a newly constructed RNG.
    monkeypatch.setattr(algorithm, "_rng", None)
    monkeypatch.setattr(worker, "_rng", None)
    monkeypatch.setattr(np.random, "default_rng", None)
    global_state = pickle.dumps(np.random.get_state())
    operations = {
        "initialize": (algorithm.initialize_particle, worker.initialize_particle),
        "update": (algorithm.update_particle, worker.update_particle),
        "second_update": (
            algorithm.second_update_particle,
            worker.second_update_particle,
        ),
    }
    process, process_worker = operations[phase]
    for identifier, cache, snapshot in zip(identifiers, caches, snapshots, strict=True):
        scalar = snapshot["scalar"]
        expected_generator = snapshot["generator"]
        actual_generator = cache["generator"]
        assert isinstance(scalar, float)
        assert isinstance(expected_generator, np.random.Generator)
        assert isinstance(actual_generator, np.random.Generator)
        # Snapshot all particles before any processing: a shared BitGenerator
        # would advance later particles too and fail the continuation check.
        draws = [float(expected_generator.random()) for _ in range(1 + int(3 * scalar))]
        updated = process(identifier)
        returned = process_worker(identifier)
        assert returned.trace == updated.trace
        assert returned.candidate_variables == updated.candidate_variables
        assert updated.random_cache is cache
        assert cache["generator"] is actual_generator
        assert cache.keys() == snapshot.keys()
        assert cache["scalar"] == scalar
        array = cache["array"]
        original_array = snapshot["array"]
        assert isinstance(array, np.ndarray)
        assert isinstance(original_array, np.ndarray)
        np.testing.assert_array_equal(array, original_array)
        assert updated.trace[-1]["draws"] == draws
        assert updated.trace[-1]["scalar"] == scalar
        assert updated.trace[-1]["array"] == list(map(float, original_array))
        np.testing.assert_array_equal(
            deepcopy(actual_generator).random(4), expected_generator.random(4)
        )
        assert_particle_state_consistent(updated, small_bounds)
    assert pickle.dumps(np.random.get_state()) == global_state


def test_probe_refreshes_generator_state_for_each_particle_phase_and_iteration(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    algorithm = IntegrationProbeAlgorithm(double_eval=True)
    assert configure_algorithm(algorithm, small_bounds) is algorithm
    for identifier in ("p0", "p1", "p2", "p3"):
        algorithm.create_particle(identifier)
    caches: list[dict[str, object]] = []
    states: list[bytes] = []
    for iteration, phase in (
        (0, "initialize"),
        (1, "update"),
        (1, "second_update"),
        (2, "update"),
    ):
        if phase == "second_update":
            algorithm.inter_iteration(iteration)
            identifiers = algorithm.double_check_ids
        else:
            algorithm.pre_iteration(iteration)
            identifiers = list(algorithm.population)
        algorithm.create_random_cache(identifiers, initialize=phase == "initialize")
        for identifier in identifiers:
            cache: dict[str, object] = algorithm.population[identifier].random_cache
            generator = cache["generator"]
            assert isinstance(generator, np.random.Generator)
            assert all(cache is not previous for previous in caches)
            assert all(generator is not previous["generator"] for previous in caches)
            # Fixed suite seed: compare full initial states, not a single draw
            # or literal NumPy outputs. This catches reseeding/reusing a stream.
            state = pickle.dumps(generator.bit_generator.state)
            assert state not in states
            caches.append(cache)
            states.append(state)
        # Cache generation itself must advance the parent even without draws
        # in workers. Retained references detect reuse after framework clearing.
        algorithm.update_population([])


def test_probe_honors_independent_spawned_algorithm_seeds(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    runs: list[list[bytes]] = []
    for _ in range(2):
        run: list[bytes] = []
        for child in np.random.SeedSequence(BASE_SEED).spawn(2):
            algorithm = IntegrationProbeAlgorithm()
            assert configure_algorithm(algorithm, small_bounds) is algorithm
            # Configure is the same public entry point used on island replicas.
            # The helper establishes context; this configuration supplies the
            # actual child SeedSequence under test, without a distributed runtime.
            algorithm.configure("probe", CostFunctionWrapperBase, child)
            algorithm.create_particle("p0")
            algorithm.pre_iteration(0)
            algorithm.create_random_cache(["p0"], initialize=True)
            cache: dict[str, object] = algorithm.population["p0"].random_cache
            generator = cache["generator"]
            assert isinstance(generator, np.random.Generator)
            run.append(pickle.dumps(generator.bit_generator.state))
        runs.append(run)
    assert runs[0] == runs[1]
    assert runs[0][0] != runs[0][1]


def test_probe_reproduces_trace_after_seed_reset_and_pickle_transport(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    first = IntegrationProbeAlgorithm(double_eval=True)
    repeated = IntegrationProbeAlgorithm(double_eval=True)
    for algorithm in (first, repeated):
        assert configure_algorithm(algorithm, small_bounds) is algorithm
        _initialize(algorithm)
    # pickle has an untyped return; narrow the deserialized object immediately.
    transported: object = pickle.loads(pickle.dumps(first))  # pyright: ignore[reportAny]
    assert isinstance(transported, IntegrationProbeAlgorithm)
    for algorithm in (first, repeated, transported):
        _update(algorithm, 1)
        _update(algorithm, 2)
    expected = [particle() for particle in first.population.values()]
    assert [particle() for particle in repeated.population.values()] == expected
    assert [particle() for particle in transported.population.values()] == expected


def test_probe_reconstruction_preserves_trace_and_known_fitness(
    small_bounds: dict[str, tuple[float, float]],
    counting_objective: CountingObjective,
) -> None:
    algorithm = IntegrationProbeAlgorithm()
    assert (
        configure_algorithm(
            algorithm, small_bounds, fitness_function=counting_objective
        )
        is algorithm
    )
    _initialize(algorithm)
    original = algorithm.population["p0"]
    assert json.loads(original.dump()) == original()
    algorithm.create_particle(
        original.identifier,
        original.variables,
        original.fitness,
        trace=original.trace,
        evaluation_count=original.evaluation_count,
    )
    restored = algorithm.population["p0"]
    assert restored() == original()
    assert restored.trace is not original.trace
    algorithm.create_random_cache(["p0"], initialize=True)
    _ = algorithm.consolidate_new_particles(algorithm.initialize_particle("p0"))
    assert counting_objective.calls == 4
    assert restored.evaluation_count == original.evaluation_count
    assert restored.variables == original.variables
    assert restored.fitness == original.fitness
    assert restored.trace[-1]["phase"] == "initialize"
    # pickle has an untyped return; narrow the deserialized object immediately.
    copied: object = pickle.loads(pickle.dumps(restored))  # pyright: ignore[reportAny]
    assert isinstance(copied, IntegrationProbeParticle)
    assert copied() == restored()


def test_probe_force_stop_is_requested_after_consolidation_and_reset_by_context(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    algorithm = IntegrationProbeAlgorithm(force_stop_at_iteration=1)
    assert configure_algorithm(algorithm, small_bounds) is algorithm
    _initialize(algorithm)
    assert not algorithm.force_stop
    with pytest.warns(UserWarning):
        _update(algorithm, 1)
    assert algorithm.force_stop
    assert not algorithm.double_particle_check
    assert all(
        particle.evaluation_count == 2 for particle in algorithm.population.values()
    )
    assert all(
        particle.candidate_variables is None
        for particle in algorithm.population.values()
    )
    assert configure_algorithm(algorithm, small_bounds) is algorithm
    assert not algorithm.force_stop
