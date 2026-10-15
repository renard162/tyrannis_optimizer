"""Unit contracts of the shared lifecycle probe, without a processor double."""

import json
import pickle
from copy import deepcopy

import numpy as np
import pytest

from tests._support.assertions import assert_particle_state_consistent

# The shared factory omits AlgorithmBase type parameters. Keep the concrete
# probe reference below; its return annotation is an existing helper limitation.
from tests._support.factories import (
    configure_algorithm,  # pyright: ignore[reportUnknownVariableType]
)
from tests._support.integration_probe import (
    IntegrationProbeAlgorithm,
    IntegrationProbeParticle,
)
from tests._support.objectives import CountingObjective


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


def test_probe_consumes_cache_without_removing_or_mutating_materialized_values(
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    algorithm = IntegrationProbeAlgorithm()
    assert configure_algorithm(algorithm, small_bounds) is algorithm
    _initialize(algorithm)
    algorithm.pre_iteration(1)
    algorithm.create_random_cache(list(algorithm.population), initialize=False)
    generators: list[object] = [
        particle.random_cache["generator"] for particle in algorithm.population.values()
    ]
    assert len({id(generator) for generator in generators}) == 4
    # pickle has an untyped return; narrow it before exercising the cached RNG.
    worker: object = pickle.loads(pickle.dumps(algorithm))  # pyright: ignore[reportAny]
    assert isinstance(worker, IntegrationProbeAlgorithm)
    snapshots: list[str] = []
    for identifier, particle in algorithm.population.items():
        cache: dict[str, object] = particle.random_cache
        snapshot = deepcopy(cache)
        scalar = snapshot["scalar"]
        generator = snapshot["generator"]
        assert isinstance(scalar, float)
        assert isinstance(generator, np.random.Generator)
        draws = [float(generator.random()) for _ in range(1 + int(3 * scalar))]
        updated = algorithm.update_particle(identifier)
        returned = worker.update_particle(identifier)
        assert returned.trace == updated.trace
        assert returned.candidate_variables == updated.candidate_variables
        assert updated.random_cache is cache
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
        assert_particle_state_consistent(updated, small_bounds)
        snapshots.append(json.dumps(updated.trace[-1]["draws"]))
    assert len(set(snapshots)) == 4


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
    events = [
        event for particle in first.population.values() for event in particle.trace
    ]
    assert len({json.dumps(event["array"]) for event in events}) == len(events)
    assert len({json.dumps(event["draws"]) for event in events}) == len(events)


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
