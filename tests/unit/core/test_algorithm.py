"""Concrete state management shared by Tyrannis algorithms and particles."""

from collections.abc import Callable
from operator import methodcaller
from typing import override

import numpy as np
import pytest

from tests._support.numerics import BASE_SEED
from tests._support.objectives import encoded_sphere
from tyrannis.core.algorithm import (
    FITNESS_UNDEFINED,
    AlgorithmBase,
    CostFunctionWrapperBase,
    ParticleBase,
)


class _CostFunctionWrapper(CostFunctionWrapperBase):
    pass


class _Particle(ParticleBase):
    pass


class _Algorithm(AlgorithmBase[_Particle]):
    @override
    def __init__(self) -> None:
        pass

    @property
    def context_size(self) -> tuple[int, int]:
        return self._max_iterations, self._n_particles

    @property
    def configured_fitness(self) -> Callable[[dict[str, float]], np.float64]:
        return self._fitness_function

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    @override
    def create_particle(
        self,
        identifier: str,
        variables: dict[str, float] | None = None,
        fitness: np.float64 = FITNESS_UNDEFINED,
        *args: object,
        **kwargs: object,
    ) -> None:
        raise NotImplementedError

    @override
    def delete_particle(self, identifier: str | None) -> None:
        raise NotImplementedError

    @override
    def pre_iteration(self, actual_iter: int) -> None:
        raise NotImplementedError

    @override
    def create_random_cache(self, particle_ids: list[str], initialize: bool) -> None:
        raise NotImplementedError

    @override
    def initialize_particle(self, identifier: str) -> _Particle:
        raise NotImplementedError

    @staticmethod
    @override
    def consolidate_new_particles(particle: _Particle) -> _Particle:
        raise NotImplementedError

    @override
    def update_particle(self, identifier: str) -> _Particle:
        raise NotImplementedError

    @override
    def post_iteration(self, actual_iter: int) -> None:
        raise NotImplementedError


def _fitness(variables: dict[str, float]) -> np.float64:
    return np.float64(encoded_sphere(variables))


def _initialized_algorithm() -> _Algorithm:
    algorithm = _Algorithm()
    algorithm.initialize_context(_fitness, {"x": (-5.0, 5.0)}, 5, 2)
    return algorithm


def test_cost_wrapper_forwards_arguments_and_result() -> None:
    expected = np.float64(5.0)

    def objective(value: float, *, offset: float) -> np.float64:
        assert value == 2.0
        assert offset == 3.0
        return expected

    result = _CostFunctionWrapper(objective)(2.0, offset=3.0)

    assert result is expected


def test_particle_construction_distinguishes_unevaluated_and_restored_state() -> None:
    unevaluated = _Particle("new", {"x": 1.0})
    restored = _Particle("restored", {"x": 2.0}, np.float64(4.0))

    assert unevaluated.identifier == "new"
    assert unevaluated.variables == {"x": 1.0}
    assert unevaluated.fitness == FITNESS_UNDEFINED
    assert unevaluated.new_particle
    assert unevaluated.candidate_variables is None
    assert unevaluated.candidate_fitness is None
    assert restored.variables == restored.candidate_variables == {"x": 2.0}
    assert restored.fitness == restored.candidate_fitness == np.float64(4.0)
    assert restored.new_particle


def test_particle_exposes_serializable_current_state() -> None:
    particle = _Particle("p", {"x": 2.0}, np.float64(3.0))

    assert particle() == {
        "identifier": "p",
        "variables": {"x": 2.0},
        "fitness": np.float64(3.0),
    }
    assert particle.dump() == (
        '{"identifier": "p", "variables": {"x": 2.0}, "fitness": 3.0}'
    )


def test_particle_update_evaluates_candidate_without_changing_current_state() -> None:
    particle = _Particle("p", {"x": 1.0}, np.float64(1.0))
    candidate = {"x": 3.0}

    particle.update(candidate, _fitness)

    assert particle.candidate_variables == candidate
    assert particle.candidate_fitness == np.float64(9.0)
    assert particle.variables == {"x": 1.0}
    assert particle.fitness == np.float64(1.0)


def test_particle_update_rejects_missing_variables() -> None:
    particle = _Particle("p", {"x": 1.0})

    with pytest.raises(ValueError, match="Variables cannot be None"):
        methodcaller("update", None, np.float64)(particle)


def test_particle_consolidation_requires_complete_candidate() -> None:
    particle = _Particle("p", {"x": 1.0})

    with pytest.raises(RuntimeError, match="No candidate solution"):
        particle.consolidate(consolidate_new=True)


def test_unevaluated_new_particle_consolidates_candidate_without_selection() -> None:
    particle = _Particle("p", {"x": 1.0})
    particle.update({"x": 2.0}, _fitness)
    particle.error_fitness = np.float64(8.0)

    particle.consolidate(consolidate_new=False)

    assert particle.variables == {"x": 2.0}
    assert particle.fitness == np.float64(4.0)
    assert not particle.new_particle
    assert particle.candidate_variables is None
    assert particle.candidate_fitness is None
    assert particle.error_fitness is None


def test_restored_new_particle_preserves_current_state_on_consolidation() -> None:
    particle = _Particle("p", {"x": 1.0}, np.float64(1.0))
    particle.candidate_variables = {"x": 2.0}
    particle.candidate_fitness = np.float64(4.0)
    particle.error_fitness = np.float64(8.0)

    particle.consolidate(consolidate_new=True)

    assert particle.variables == {"x": 1.0}
    assert particle.fitness == np.float64(1.0)
    assert not particle.new_particle
    assert particle.candidate_variables is None
    assert particle.candidate_fitness is None
    assert particle.error_fitness is None


@pytest.mark.parametrize("consolidate_new", [False, True], ids=["keep", "replace"])
def test_existing_particle_replaces_state_only_when_selected(
    consolidate_new: bool,
) -> None:
    particle = _Particle("p", {"x": 1.0})
    particle.update({"x": 1.0}, _fitness)
    particle.consolidate(consolidate_new=True)
    particle.update({"x": 2.0}, _fitness)
    particle.error_fitness = np.float64(8.0)

    particle.consolidate(consolidate_new=consolidate_new)

    expected = 2.0 if consolidate_new else 1.0
    assert particle.variables == {"x": expected}
    assert particle.fitness == np.float64(expected**2)
    assert not particle.new_particle
    assert particle.candidate_variables is None
    assert particle.candidate_fitness is None
    assert particle.error_fitness is None


def test_algorithm_context_starts_with_empty_iteration_state() -> None:
    algorithm = _Algorithm()

    algorithm.initialize_context(_fitness, {"x": (-5.0, 5.0)}, 7, 2)

    assert algorithm.population == {}
    assert algorithm.local_best is None
    assert algorithm.iter_best is None
    assert algorithm.iter_worst is None
    assert not algorithm.double_particle_check
    assert algorithm.double_check_ids == []
    assert algorithm.context_size == (7, 2)


def test_algorithm_configuration_wraps_fitness_and_propagates_seed() -> None:
    first = _initialized_algorithm()
    second = _initialized_algorithm()

    for algorithm in (first, second):
        algorithm.configure("algorithm", _CostFunctionWrapper, BASE_SEED)

    assert first.identifier == second.identifier == "algorithm"
    wrapped_fitness = first.configured_fitness
    assert isinstance(wrapped_fitness, _CostFunctionWrapper)
    assert wrapped_fitness({"x": 2.0}) == np.float64(4.0)
    assert first.rng.random(3).tolist() == second.rng.random(3).tolist()


def test_algorithm_population_updates_membership_count_and_random_caches() -> None:
    algorithm = _initialized_algorithm()
    first = _Particle("first", {"x": 1.0})
    second = _Particle("second", {"x": 2.0})
    existing = _Particle("existing", {"x": 3.0})
    existing.update({"x": 3.0}, _fitness)
    existing.consolidate(consolidate_new=True)
    algorithm.population.update(
        {particle.identifier: particle for particle in (first, second, existing)}
    )

    assert algorithm.new_particles_id == ["first", "second"]
    algorithm.update_n_particles()
    assert algorithm.context_size == (5, 3)

    for particle in algorithm.population.values():
        particle.random_cache = {"sample": 1.0}
    replacement = _Particle("first", {"x": 4.0})
    replacement.random_cache = {"sample": 2.0}
    algorithm.update_population([replacement])

    assert algorithm.population == {
        "first": replacement,
        "second": second,
        "existing": existing,
    }
    assert algorithm.new_particles_id == ["first", "second"]
    assert all(not particle.random_cache for particle in algorithm.population.values())


def test_algorithm_default_second_phase_preserves_population_state() -> None:
    algorithm = _initialized_algorithm()
    particle = _Particle("p", {"x": 1.0})
    algorithm.population[particle.identifier] = particle

    assert not algorithm.double_particle_check
    algorithm.double_check_ids = ["p"]
    assert algorithm.double_check_ids == ["p"]
    assert algorithm.second_update_particle("p") is particle
    algorithm.inter_iteration(1)
    assert algorithm.double_check_ids == ["p"]
    assert algorithm.population["p"] is particle


def test_algorithm_solution_state_tracks_iteration_extrema_and_copied_best() -> None:
    algorithm = _initialized_algorithm()
    first = _Particle("first", {"x": 1.0}, np.float64(1.0))
    second = _Particle("second", {"x": 3.0}, np.float64(9.0))
    algorithm.population.update({"first": first, "second": second})

    algorithm.update_solution_state()

    assert algorithm.iter_best is first
    assert algorithm.iter_worst is second
    assert algorithm.local_best is not first
    assert algorithm.local_best is not None
    assert algorithm.local_best.variables == {"x": 1.0}
    first.variables["x"] = 2.0
    assert algorithm.local_best.variables == {"x": 1.0}

    improved = _Particle("improved", {"x": 0.5}, np.float64(0.25))
    algorithm.population["improved"] = improved
    algorithm.update_solution_state()
    assert algorithm.iter_best is improved
    assert algorithm.iter_worst is second
    assert algorithm.local_best is not improved
    assert algorithm.local_best is not None
    assert algorithm.local_best.fitness == np.float64(0.25)

    del algorithm.population["improved"]
    algorithm.update_solution_state()
    assert algorithm.iter_best is first
    assert algorithm.iter_worst is second
    assert algorithm.local_best is not None
    assert algorithm.local_best.fitness == np.float64(0.25)
