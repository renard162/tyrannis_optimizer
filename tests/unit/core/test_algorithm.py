from __future__ import annotations

import json

import numpy as np
import pytest

from tyrannis.core.algorithm import (
    AlgorithmBase,
    CostFunctionWrapperBase,
    ParticleBase,
)

# ============================================================================
# CostFunctionWrapperBase
# ============================================================================


class TestCostFunctionWrapperBase:
    """Tests for CostFunctionWrapperBase."""

    def test_initializes_with_function(self) -> None:
        def function(value: float) -> float:
            return value**2

        wrapper = CostFunctionWrapperBase(function)

        assert wrapper._function is function

    def test_call_forwards_positional_arguments(self) -> None:
        def function(x: float, y: float) -> float:
            return x + y

        wrapper = CostFunctionWrapperBase(function)

        assert wrapper(2.0, 3.0) == 5.0

    def test_call_forwards_keyword_arguments(self) -> None:
        def function(x: float, y: float) -> float:
            return x * y

        wrapper = CostFunctionWrapperBase(function)

        assert wrapper(x=2.0, y=3.0) == 6.0

    def test_call_returns_function_result(self) -> None:
        expected = 42.5

        def function(_: float) -> float:
            return expected

        wrapper = CostFunctionWrapperBase(function)

        assert wrapper(10.0) == expected

    def test_call_propagates_function_exception(self) -> None:
        def function(_: float) -> float:
            raise RuntimeError("test error")

        wrapper = CostFunctionWrapperBase(function)

        with pytest.raises(RuntimeError, match="test error"):
            wrapper(1.0)


# ============================================================================
# ParticleBase
# ============================================================================


class TestParticleBase:
    """Tests for ParticleBase."""

    def test_initial_state(self, particle: ParticleBase) -> None:
        assert particle.identifier == "particle:0"
        assert particle.variables == {
            "x": 1.0,
            "y": 2.0,
        }
        assert particle.fitness == 5.0
        assert particle.new_particle is True
        assert particle.candidate_variables is None
        assert particle.candidate_fitness is None

    def test_call_returns_constructor_state(
        self,
        particle: ParticleBase,
    ) -> None:
        assert particle() == {
            "identifier": "particle:0",
            "variables": {
                "x": 1.0,
                "y": 2.0,
            },
            "fitness": 5.0,
        }

    def test_call_does_not_include_candidate_state(
        self,
        particle: ParticleBase,
    ) -> None:
        particle.candidate_variables = {
            "x": 3.0,
            "y": 4.0,
        }
        particle.candidate_fitness = 25.0

        assert particle() == {
            "identifier": "particle:0",
            "variables": {
                "x": 1.0,
                "y": 2.0,
            },
            "fitness": 5.0,
        }

    def test_identifier_is_read_only(
        self,
        particle: ParticleBase,
    ) -> None:
        with pytest.raises(AttributeError):
            particle.identifier = "particle:1"  # type: ignore[misc]

    def test_variables_is_read_only(
        self,
        particle: ParticleBase,
    ) -> None:
        with pytest.raises(AttributeError):
            particle.variables = {"x": 10.0, "y": 10.0}  # type: ignore[misc]

    def test_fitness_is_read_only(
        self,
        particle: ParticleBase,
    ) -> None:
        with pytest.raises(AttributeError):
            particle.fitness = 100.0  # type: ignore[misc]

    def test_candidate_variables_can_be_set(
        self,
        particle: ParticleBase,
    ) -> None:
        variables = {
            "x": 7.0,
            "y": 8.0,
        }

        particle.candidate_variables = variables

        assert particle.candidate_variables is variables

    def test_candidate_variables_can_be_cleared(
        self,
        particle: ParticleBase,
    ) -> None:
        particle.candidate_variables = {
            "x": 7.0,
            "y": 8.0,
        }

        particle.candidate_variables = None  # type: ignore[assignment]

        assert particle.candidate_variables is None

    def test_candidate_fitness_can_be_set(
        self,
        particle: ParticleBase,
    ) -> None:
        particle.candidate_fitness = 123.0

        assert particle.candidate_fitness == 123.0

    def test_candidate_fitness_can_be_cleared(
        self,
        particle: ParticleBase,
    ) -> None:
        particle.candidate_fitness = 123.0
        particle.candidate_fitness = None

        assert particle.candidate_fitness is None

    def test_dump_returns_json(
        self,
        particle: ParticleBase,
    ) -> None:
        dumped = particle.dump()

        assert isinstance(dumped, str)

        assert json.loads(dumped) == {
            "identifier": "particle:0",
            "variables": {
                "x": 1.0,
                "y": 2.0,
            },
            "fitness": 5.0,
        }

    def test_dump_contains_current_state_only(
        self,
        particle: ParticleBase,
    ) -> None:
        particle.candidate_variables = {
            "x": 3.0,
            "y": 4.0,
        }
        particle.candidate_fitness = 25.0

        assert json.loads(particle.dump()) == {
            "identifier": "particle:0",
            "variables": {
                "x": 1.0,
                "y": 2.0,
            },
            "fitness": 5.0,
        }

    def test_update_creates_candidate(
        self,
        particle: ParticleBase,
        fitness_function,
    ) -> None:
        variables = {
            "x": 3.0,
            "y": 4.0,
        }

        particle.update(
            variables=variables,
            fitness_function=fitness_function,
        )

        assert particle.variables == {
            "x": 1.0,
            "y": 2.0,
        }
        assert particle.fitness == 5.0
        assert particle.candidate_variables is variables
        assert particle.candidate_fitness == 25.0
        assert particle.new_particle is True

    def test_update_evaluates_fitness_once(
        self,
        particle: ParticleBase,
    ) -> None:
        calls: list[dict[str, float]] = []

        def fitness_function(
            variables: dict[str, float],
        ) -> float:
            calls.append(variables)
            return sum(value**2 for value in variables.values())

        variables = {
            "x": 3.0,
            "y": 4.0,
        }

        particle.update(
            variables=variables,
            fitness_function=fitness_function,
        )

        assert calls == [variables]
        assert particle.candidate_fitness == 25.0

    def test_update_replaces_existing_candidate(
        self,
        particle: ParticleBase,
        fitness_function,
    ) -> None:
        particle.update(
            variables={
                "x": 3.0,
                "y": 4.0,
            },
            fitness_function=fitness_function,
        )

        particle.update(
            variables={
                "x": 6.0,
                "y": 8.0,
            },
            fitness_function=fitness_function,
        )

        assert particle.variables == {
            "x": 1.0,
            "y": 2.0,
        }
        assert particle.fitness == 5.0
        assert particle.candidate_variables == {
            "x": 6.0,
            "y": 8.0,
        }
        assert particle.candidate_fitness == 100.0

    def test_update_rejects_none_variables(
        self,
        particle: ParticleBase,
    ) -> None:
        with pytest.raises(
            ValueError,
            match="Variables cannot be None.",
        ):
            particle.update(
                variables=None,  # type: ignore[arg-type]
                fitness_function=lambda _: 0.0,
            )

    def test_update_does_not_modify_current_state(
        self,
        particle: ParticleBase,
        fitness_function,
    ) -> None:
        particle.update(
            variables={
                "x": 9.0,
                "y": 8.0,
            },
            fitness_function=fitness_function,
        )

        assert particle.variables == {
            "x": 1.0,
            "y": 2.0,
        }
        assert particle.fitness == 5.0

    def test_recreate_replaces_current_state(
        self,
        particle: ParticleBase,
    ) -> None:
        particle.recreate(
            variables={
                "x": 5.0,
                "y": 6.0,
            },
            fitness=61.0,
        )

        assert particle.identifier == "particle:0"
        assert particle.variables == {
            "x": 5.0,
            "y": 6.0,
        }
        assert particle.fitness == 61.0

    def test_recreate_marks_particle_as_new(
        self,
        particle: ParticleBase,
    ) -> None:
        particle.consolidate(consolidate_new=True)

        assert particle.new_particle is False

        particle.recreate(
            variables={
                "x": 5.0,
                "y": 6.0,
            },
            fitness=61.0,
        )

        assert particle.new_particle is True

    def test_recreate_clears_candidate_state(
        self,
        particle: ParticleBase,
    ) -> None:
        particle.update(
            variables={
                "x": 3.0,
                "y": 4.0,
            },
            fitness_function=lambda variables: sum(
                value**2 for value in variables.values()
            ),
        )

        particle.recreate(
            variables={
                "x": 5.0,
                "y": 6.0,
            },
            fitness=61.0,
        )

        assert particle.candidate_variables is None
        assert particle.candidate_fitness is None

    def test_recreate_preserves_identifier(
        self,
        particle: ParticleBase,
    ) -> None:
        particle.recreate(
            variables={
                "x": 5.0,
                "y": 6.0,
            },
            fitness=61.0,
        )

        assert particle.identifier == "particle:0"

    def test_consolidate_true_accepts_candidate(
        self,
        particle: ParticleBase,
        fitness_function,
    ) -> None:
        particle.update(
            variables={
                "x": 3.0,
                "y": 4.0,
            },
            fitness_function=fitness_function,
        )

        particle.consolidate(consolidate_new=True)

        assert particle.variables == {
            "x": 3.0,
            "y": 4.0,
        }
        assert particle.fitness == 25.0
        assert particle.new_particle is False
        assert particle.candidate_variables is None
        assert particle.candidate_fitness is None

    def test_consolidate_false_rejects_candidate_as_current_state(
        self,
        particle: ParticleBase,
        fitness_function,
    ) -> None:
        particle.update(
            variables={
                "x": 3.0,
                "y": 4.0,
            },
            fitness_function=fitness_function,
        )

        particle.consolidate(consolidate_new=False)

        assert particle.variables == {
            "x": 1.0,
            "y": 2.0,
        }
        assert particle.fitness == 5.0
        assert particle.new_particle is False
        assert particle.candidate_variables is None
        assert particle.candidate_fitness is None

    def test_consolidate_false_initializes_particle_without_fitness(
        self,
        fitness_function,
    ) -> None:
        particle = ParticleBase(
            identifier="particle:0",
            variables={
                "x": 1.0,
            },
        )

        particle.update(
            variables={
                "x": 2.0,
            },
            fitness_function=fitness_function,
        )

        particle.consolidate(consolidate_new=False)

        assert particle.variables == {
            "x": 2.0,
        }
        assert particle.fitness == 4.0
        assert particle.new_particle is False
        assert particle.candidate_variables is None
        assert particle.candidate_fitness is None

    def test_consolidate_true_initializes_particle_without_fitness(
        self,
        fitness_function,
    ) -> None:
        particle = ParticleBase(
            identifier="particle:0",
            variables={
                "x": 1.0,
            },
        )

        particle.update(
            variables={
                "x": 2.0,
            },
            fitness_function=fitness_function,
        )

        particle.consolidate(consolidate_new=True)

        assert particle.variables == {
            "x": 2.0,
        }
        assert particle.fitness == 4.0
        assert particle.new_particle is False

    def test_consolidate_new_particle_with_existing_fitness(
        self,
        particle: ParticleBase,
    ) -> None:
        particle.consolidate(consolidate_new=True)

        assert particle.variables == {
            "x": 1.0,
            "y": 2.0,
        }
        assert particle.fitness == 5.0
        assert particle.new_particle is False

    def test_consolidate_without_candidate_raises(
        self,
        particle: ParticleBase,
    ) -> None:
        particle._new_particle = False

        with pytest.raises(
            RuntimeError,
            match="No candidate solution available for consolidation.",
        ):
            particle.consolidate(consolidate_new=True)

    def test_consolidate_new_particle_without_fitness_raises(self) -> None:
        particle = ParticleBase(
            identifier="particle:0",
            variables={
                "x": 1.0,
            },
        )

        with pytest.raises(
            RuntimeError,
            match="No candidate solution available for consolidation.",
        ):
            particle.consolidate(consolidate_new=True)

    def test_consolidate_clears_candidate_after_accepting(
        self,
        particle: ParticleBase,
        fitness_function,
    ) -> None:
        particle.update(
            variables={
                "x": 3.0,
                "y": 4.0,
            },
            fitness_function=fitness_function,
        )

        particle.consolidate(consolidate_new=True)

        assert particle.candidate_variables is None
        assert particle.candidate_fitness is None

    def test_consolidate_clears_candidate_after_rejecting(
        self,
        particle: ParticleBase,
        fitness_function,
    ) -> None:
        particle.update(
            variables={
                "x": 3.0,
                "y": 4.0,
            },
            fitness_function=fitness_function,
        )

        particle.consolidate(consolidate_new=False)

        assert particle.candidate_variables is None
        assert particle.candidate_fitness is None


# ============================================================================
# AlgorithmBase
# ============================================================================


class TestAlgorithmBase:
    """Tests for concrete behavior implemented by AlgorithmBase."""

    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            AlgorithmBase()  # pyright: ignore[reportAbstractUsage]

    def test_initialize_context(
        self,
        algorithm,
        fitness_function,
        boundaries,
    ) -> None:
        algorithm.initialize_context(
            fitness_function=fitness_function,
            boundaries=boundaries,
        )

        assert algorithm.population == {}
        assert algorithm.local_best is None
        assert algorithm.iter_best is None
        assert algorithm.iter_worst is None

    def test_initialize_context_resets_population(
        self,
        algorithm,
    ) -> None:
        algorithm.create_particle(
            identifier="particle:0",
            variables={"x": 1.0, "y": 2.0},
            fitness=5.0,
        )

        algorithm.initialize_context(
            fitness_function=lambda _: 0.0,
            boundaries={
                "x": (-1.0, 1.0),
            },
        )

        assert algorithm.population == {}

    def test_initialize_context_resets_solution_state(
        self,
        algorithm,
    ) -> None:
        algorithm.create_particle(
            identifier="particle:0",
            variables={"x": 1.0, "y": 2.0},
            fitness=5.0,
        )
        algorithm.update_solution_state()

        assert algorithm.local_best is not None
        assert algorithm.iter_best is not None
        assert algorithm.iter_worst is not None

        algorithm.initialize_context(
            fitness_function=lambda _: 0.0,
            boundaries={
                "x": (-1.0, 1.0),
            },
        )

        assert algorithm.local_best is None
        assert algorithm.iter_best is None
        assert algorithm.iter_worst is None

    def test_configure_sets_identifier(
        self,
        algorithm,
        cost_function_wrapper,
    ) -> None:
        algorithm.configure(
            identifier="test-algorithm",
            cost_function_wrapper=cost_function_wrapper,
            seed=42,
        )

        assert algorithm.identifier == "test-algorithm"

    def test_configure_wraps_fitness_function(
        self,
        algorithm,
        cost_function_wrapper,
    ) -> None:
        original_function = algorithm._fitness_function

        algorithm.configure(
            identifier="test-algorithm",
            cost_function_wrapper=cost_function_wrapper,
            seed=42,
        )

        assert algorithm._fitness_function is not original_function
        assert (
            algorithm._fitness_function(
                {"x": 1.0, "y": 2.0},
            )
            == 5.0
        )

    def test_configure_initializes_random_generator(
        self,
        algorithm,
        cost_function_wrapper,
    ) -> None:
        algorithm.configure(
            identifier="test-algorithm",
            cost_function_wrapper=cost_function_wrapper,
            seed=42,
        )

        assert isinstance(
            algorithm._rng,
            np.random.Generator,
        )

    def test_configure_seed_is_reproducible(
        self,
        algorithm,
        cost_function_wrapper,
    ) -> None:
        algorithm.configure(
            identifier="test-algorithm",
            cost_function_wrapper=cost_function_wrapper,
            seed=42,
        )
        first = algorithm._rng.random()

        algorithm.configure(
            identifier="test-algorithm",
            cost_function_wrapper=cost_function_wrapper,
            seed=42,
        )
        second = algorithm._rng.random()

        assert first == second

    def test_population_is_accessible(
        self,
        algorithm,
    ) -> None:
        assert algorithm.population == {}

    def test_new_particles_id_returns_new_particles(
        self,
        algorithm,
    ) -> None:
        algorithm.create_particle(
            identifier="particle:0",
            variables={"x": 1.0, "y": 2.0},
            fitness=5.0,
        )
        algorithm.create_particle(
            identifier="particle:1",
            variables={"x": 3.0, "y": 4.0},
            fitness=25.0,
        )

        assert algorithm.new_particles_id == [
            "particle:0",
            "particle:1",
        ]

    def test_new_particles_id_excludes_consolidated_particles(
        self,
        algorithm,
    ) -> None:
        algorithm.create_particle(
            identifier="particle:0",
            variables={"x": 1.0, "y": 2.0},
            fitness=5.0,
        )
        algorithm.create_particle(
            identifier="particle:1",
            variables={"x": 3.0, "y": 4.0},
            fitness=25.0,
        )

        algorithm.population["particle:0"].consolidate(
            consolidate_new=True,
        )

        assert algorithm.new_particles_id == ["particle:1"]

    def test_get_unmodified_particle_returns_population_particle(
        self,
        algorithm,
    ) -> None:
        algorithm.create_particle(
            identifier="particle:0",
            variables={"x": 1.0, "y": 2.0},
            fitness=5.0,
        )

        particle = algorithm.get_unmodified_particle("particle:0")

        assert particle is algorithm.population["particle:0"]

    def test_get_unmodified_particle_copies_current_state_to_candidate(
        self,
        algorithm,
    ) -> None:
        algorithm.create_particle(
            identifier="particle:0",
            variables={"x": 1.0, "y": 2.0},
            fitness=5.0,
        )

        particle = algorithm.get_unmodified_particle("particle:0")

        assert particle.candidate_variables == {
            "x": 1.0,
            "y": 2.0,
        }
        assert particle.candidate_fitness == 5.0

    def test_get_unmodified_particle_raises_for_unknown_identifier(
        self,
        algorithm,
    ) -> None:
        with pytest.raises(KeyError):
            algorithm.get_unmodified_particle("unknown")

    def test_update_population_adds_particles(
        self,
        algorithm,
    ) -> None:
        particles = [
            ParticleBase(
                identifier="particle:0",
                variables={"x": 1.0},
                fitness=1.0,
            ),
            ParticleBase(
                identifier="particle:1",
                variables={"x": 2.0},
                fitness=4.0,
            ),
        ]

        algorithm.update_population(particles)

        assert algorithm.population == {
            "particle:0": particles[0],
            "particle:1": particles[1],
        }

    def test_update_population_replaces_same_identifier(
        self,
        algorithm,
    ) -> None:
        original = ParticleBase(
            identifier="particle:0",
            variables={"x": 1.0},
            fitness=1.0,
        )
        replacement = ParticleBase(
            identifier="particle:0",
            variables={"x": 5.0},
            fitness=25.0,
        )

        algorithm.update_population([original])
        algorithm.update_population([replacement])

        assert algorithm.population["particle:0"] is replacement

    def test_update_population_accepts_any_iterable(
        self,
        algorithm,
    ) -> None:
        particles = (
            ParticleBase(
                identifier=f"particle:{index}",
                variables={"x": float(index)},
                fitness=float(index),
            )
            for index in range(3)
        )

        algorithm.update_population(particles)

        assert list(algorithm.population) == [
            "particle:0",
            "particle:1",
            "particle:2",
        ]

    def test_update_solution_state_sets_iteration_best(
        self,
        algorithm,
    ) -> None:
        algorithm.create_particle(
            identifier="particle:0",
            variables={"x": 1.0},
            fitness=10.0,
        )
        algorithm.create_particle(
            identifier="particle:1",
            variables={"x": 2.0},
            fitness=2.0,
        )
        algorithm.create_particle(
            identifier="particle:2",
            variables={"x": 3.0},
            fitness=20.0,
        )

        algorithm.update_solution_state()

        assert algorithm.iter_best is algorithm.population["particle:1"]

    def test_update_solution_state_sets_iteration_worst(
        self,
        algorithm,
    ) -> None:
        algorithm.create_particle(
            identifier="particle:0",
            variables={"x": 1.0},
            fitness=10.0,
        )
        algorithm.create_particle(
            identifier="particle:1",
            variables={"x": 2.0},
            fitness=2.0,
        )
        algorithm.create_particle(
            identifier="particle:2",
            variables={"x": 3.0},
            fitness=20.0,
        )

        algorithm.update_solution_state()

        assert algorithm.iter_worst is algorithm.population["particle:2"]

    def test_update_solution_state_sets_local_best(
        self,
        algorithm,
    ) -> None:
        algorithm.create_particle(
            identifier="particle:0",
            variables={"x": 1.0},
            fitness=10.0,
        )
        algorithm.create_particle(
            identifier="particle:1",
            variables={"x": 2.0},
            fitness=2.0,
        )

        algorithm.update_solution_state()

        assert algorithm.local_best is not None
        assert algorithm.local_best.identifier == "particle:1"
        assert algorithm.local_best.fitness == 2.0

    def test_local_best_is_a_copy_of_iteration_best(
        self,
        algorithm,
    ) -> None:
        algorithm.create_particle(
            identifier="particle:0",
            variables={"x": 1.0},
            fitness=2.0,
        )

        algorithm.update_solution_state()

        assert algorithm.local_best is not algorithm.population["particle:0"]
        assert algorithm.local_best() == algorithm.population["particle:0"]()

    def test_local_best_is_not_affected_by_population_particle_recreation(
        self,
        algorithm,
    ) -> None:
        algorithm.create_particle(
            identifier="particle:0",
            variables={"x": 1.0},
            fitness=2.0,
        )

        algorithm.update_solution_state()

        algorithm.population["particle:0"].recreate(
            variables={"x": 100.0},
            fitness=10000.0,
        )

        assert algorithm.local_best is not None
        assert algorithm.local_best.variables == {"x": 1.0}
        assert algorithm.local_best.fitness == 2.0

    def test_update_solution_state_replaces_local_best_when_new_best_is_better(
        self,
        algorithm,
    ) -> None:
        algorithm.create_particle(
            identifier="particle:0",
            variables={"x": 1.0},
            fitness=10.0,
        )

        algorithm.update_solution_state()

        algorithm.create_particle(
            identifier="particle:1",
            variables={"x": 2.0},
            fitness=2.0,
        )

        algorithm.update_solution_state()

        assert algorithm.local_best is not None
        assert algorithm.local_best.identifier == "particle:1"
        assert algorithm.local_best.fitness == 2.0

    def test_update_solution_state_preserves_local_best_when_new_best_is_worse(
        self,
        algorithm,
    ) -> None:
        algorithm.create_particle(
            identifier="particle:0",
            variables={"x": 1.0},
            fitness=2.0,
        )

        algorithm.update_solution_state()

        local_best = algorithm.local_best

        algorithm.create_particle(
            identifier="particle:1",
            variables={"x": 2.0},
            fitness=10.0,
        )

        algorithm.update_solution_state()

        assert algorithm.local_best is local_best
        assert algorithm.local_best is not None
        assert algorithm.local_best.identifier == "particle:0"
        assert algorithm.local_best.fitness == 2.0

    def test_update_solution_state_raises_for_empty_population(
        self,
        algorithm,
    ) -> None:
        with pytest.raises(ValueError):
            algorithm.update_solution_state()

    def test_get_fitness_returns_particle_fitness(
        self,
        particle: ParticleBase,
    ) -> None:
        assert AlgorithmBase.get_fitness(particle) == 5.0

    def test_get_fitness_raises_when_fitness_is_none(self) -> None:
        particle = ParticleBase(
            identifier="particle:0",
            variables={"x": 1.0},
            fitness=None,
        )

        with pytest.raises(
            RuntimeError,
            match="Particle 'particle:0' does not have a fitness.",
        ):
            AlgorithmBase.get_fitness(particle)

    def test_get_fitness_can_be_used_as_key_function(self) -> None:
        particles = [
            ParticleBase(
                identifier="a",
                variables={"x": 1.0},
                fitness=5.0,
            ),
            ParticleBase(
                identifier="b",
                variables={"x": 2.0},
                fitness=2.0,
            ),
            ParticleBase(
                identifier="c",
                variables={"x": 3.0},
                fitness=8.0,
            ),
        ]

        assert (
            min(
                particles,
                key=AlgorithmBase.get_fitness,
            ).identifier
            == "b"
        )

        assert (
            max(
                particles,
                key=AlgorithmBase.get_fitness,
            ).identifier
            == "c"
        )

    def test_iter_best_is_none_before_update_solution_state(
        self,
        algorithm,
    ) -> None:
        assert algorithm.iter_best is None

    def test_iter_worst_is_none_before_update_solution_state(
        self,
        algorithm,
    ) -> None:
        assert algorithm.iter_worst is None

    def test_iter_best_returns_current_population_particle(
        self,
        algorithm,
    ) -> None:
        algorithm.create_particle(
            identifier="particle:0",
            variables={"x": 1.0},
            fitness=2.0,
        )
        algorithm.create_particle(
            identifier="particle:1",
            variables={"x": 2.0},
            fitness=10.0,
        )

        algorithm.update_solution_state()

        assert algorithm.iter_best is algorithm.population["particle:0"]

    def test_iter_worst_returns_current_population_particle(
        self,
        algorithm,
    ) -> None:
        algorithm.create_particle(
            identifier="particle:0",
            variables={"x": 1.0},
            fitness=2.0,
        )
        algorithm.create_particle(
            identifier="particle:1",
            variables={"x": 2.0},
            fitness=10.0,
        )

        algorithm.update_solution_state()

        assert algorithm.iter_worst is algorithm.population["particle:1"]
