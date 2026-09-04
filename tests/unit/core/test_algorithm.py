# pyright: reportUndefinedVariable=false
# ruff: noqa: F821

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

    def test_initial_state(
        self,
        particle: ParticleBase,
    ) -> None:
        assert particle.identifier == "particle:0"
        assert particle.variables == {"x": 1.0, "y": 2.0}
        assert particle.fitness == 5.0

        assert particle.new_particle is True
        assert particle.candidate_variables is None
        assert particle.candidate_fitness is None

    def test_call_returns_constructor_state(
        self,
        particle: ParticleBase,
    ) -> None:
        state = particle()

        assert state == {
            "identifier": "particle:0",
            "variables": {
                "x": 1.0,
                "y": 2.0,
            },
            "fitness": 5.0,
        }

    def test_call_returns_current_state_after_update(
        self,
        particle: ParticleBase,
    ) -> None:
        particle.update(
            variables={"x": 3.0, "y": 4.0},
            fitness_function=lambda variables: sum(
                value**2 for value in variables.values()
            ),
        )

        state = particle()

        # update() creates a candidate and must not change the current state.
        assert state == {
            "identifier": "particle:0",
            "variables": {
                "x": 1.0,
                "y": 2.0,
            },
            "fitness": 5.0,
        }

    def test_recreate_resets_particle_state(
        self,
        particle: ParticleBase,
    ) -> None:
        particle.update(
            variables={"x": 3.0, "y": 4.0},
            fitness_function=lambda variables: sum(
                value**2 for value in variables.values()
            ),
        )

        particle.recreate(
            variables={"x": 5.0, "y": 6.0},
            fitness=61.0,
        )

        assert particle.identifier == "particle:0"
        assert particle.variables == {"x": 5.0, "y": 6.0}
        assert particle.fitness == 61.0
        assert particle.new_particle is True
        assert particle.candidate_variables is None
        assert particle.candidate_fitness is None

    def test_recreate_preserves_identifier(
        self,
        particle: ParticleBase,
    ) -> None:
        particle.recreate(
            variables={"x": 5.0, "y": 6.0},
            fitness=61.0,
        )

        assert particle.identifier == "particle:0"

    def test_candidate_variables_setter(
        self,
        particle: ParticleBase,
    ) -> None:
        variables = {"x": 7.0, "y": 8.0}

        particle.candidate_variables = variables

        assert particle.candidate_variables is variables

    def test_candidate_fitness_setter(
        self,
        particle: ParticleBase,
    ) -> None:
        particle.candidate_fitness = 123.0

        assert particle.candidate_fitness == 123.0

    def test_candidate_values_can_be_reset(
        self,
        particle: ParticleBase,
    ) -> None:
        particle.candidate_variables = {"x": 3.0}
        particle.candidate_fitness = 9.0

        particle.candidate_variables = None
        particle.candidate_fitness = None

        assert particle.candidate_variables is None
        assert particle.candidate_fitness is None

    def test_dump_returns_valid_json(
        self,
        particle: ParticleBase,
    ) -> None:
        dumped = particle.dump()

        assert isinstance(dumped, str)

        state = json.loads(dumped)

        assert state == {
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
    ) -> None:
        variables = {"x": 3.0, "y": 4.0}

        particle.update(
            variables=variables,
            fitness_function=lambda values: sum(value**2 for value in values.values()),
        )

        assert particle.variables == {"x": 1.0, "y": 2.0}
        assert particle.fitness == 5.0

        assert particle.candidate_variables is variables
        assert particle.candidate_fitness == 25.0

        assert particle.new_particle is True

    def test_update_replaces_previous_candidate(
        self,
        particle: ParticleBase,
    ) -> None:
        particle.update(
            variables={"x": 3.0, "y": 4.0},
            fitness_function=lambda values: sum(value**2 for value in values.values()),
        )

        particle.update(
            variables={"x": 6.0, "y": 8.0},
            fitness_function=lambda values: sum(value**2 for value in values.values()),
        )

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

    def test_consolidate_new_accepts_candidate(
        self,
        particle: ParticleBase,
    ) -> None:
        particle.update(
            variables={"x": 3.0, "y": 4.0},
            fitness_function=lambda values: sum(value**2 for value in values.values()),
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

    def test_consolidate_without_new_keeps_current_solution(
        self,
        particle: ParticleBase,
    ) -> None:
        particle.update(
            variables={"x": 3.0, "y": 4.0},
            fitness_function=lambda values: sum(value**2 for value in values.values()),
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

    def test_consolidate_without_new_accepts_candidate_when_current_fitness_is_none(
        self,
    ) -> None:
        particle = ParticleBase(
            identifier="particle:0",
            variables={"x": 1.0},
            fitness=None,
        )

        particle.update(
            variables={"x": 2.0},
            fitness_function=lambda values: values["x"] ** 2,
        )

        particle.consolidate(consolidate_new=False)

        assert particle.variables == {"x": 2.0}
        assert particle.fitness == 4.0
        assert particle.new_particle is False
        assert particle.candidate_variables is None
        assert particle.candidate_fitness is None

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

    def test_consolidate_new_particle_with_existing_fitness_without_candidate(
        self,
        particle: ParticleBase,
    ) -> None:
        particle.consolidate(consolidate_new=True)

        assert particle.new_particle is False
        assert particle.variables == {
            "x": 1.0,
            "y": 2.0,
        }
        assert particle.fitness == 5.0

    def test_consolidate_new_particle_without_fitness_raises(self) -> None:
        particle = ParticleBase(
            identifier="particle:0",
            variables={"x": 1.0},
            fitness=None,
        )

        with pytest.raises(
            RuntimeError,
            match="No candidate solution available for consolidation.",
        ):
            particle.consolidate(consolidate_new=True)


# ============================================================================
# AlgorithmBase
# ============================================================================


class TestAlgorithmBase:
    """Tests for the concrete behavior implemented by AlgorithmBase."""

    def test_initialize_context(
        self,
        fitness_function,
        boundaries,
    ) -> None:
        algorithm = DummyAlgorithm()

        algorithm.initialize_context(
            fitness_function=fitness_function,
            boundaries=boundaries,
        )

        assert algorithm._fitness_function is fitness_function
        assert algorithm._boundaries is boundaries
        assert algorithm.population == {}
        assert algorithm.local_best is None
        assert algorithm.iter_best is None
        assert algorithm.iter_worst is None

    def test_initialize_context_resets_execution_state(
        self,
        algorithm,
    ) -> None:
        algorithm.create_particle(
            identifier="particle:0",
            variables={"x": 1.0, "y": 2.0},
            fitness=5.0,
        )

        algorithm._local_best = algorithm.population["particle:0"]
        algorithm._iter_best = "particle:0"
        algorithm._iter_worst = "particle:0"

        algorithm.initialize_context(
            fitness_function=algorithm._fitness_function,
            boundaries=algorithm._boundaries,
        )

        assert algorithm.population == {}
        assert algorithm.local_best is None
        assert algorithm.iter_best is None
        assert algorithm.iter_worst is None

    def test_configure_sets_identifier(
        self,
        algorithm,
    ) -> None:
        algorithm.configure(
            identifier="test-algorithm",
            cost_function_wrapper=DummyCostFunctionWrapper,
            seed=42,
        )

        assert algorithm.identifier == "test-algorithm"

    def test_configure_wraps_fitness_function(
        self,
        algorithm,
    ) -> None:
        original_function = algorithm._fitness_function

        algorithm.configure(
            identifier="test-algorithm",
            cost_function_wrapper=DummyCostFunctionWrapper,
            seed=42,
        )

        assert isinstance(
            algorithm._fitness_function,
            DummyCostFunctionWrapper,
        )
        assert algorithm._fitness_function is not original_function

        assert algorithm._fitness_function({"x": 1.0, "y": 2.0}) == 5.0

    def test_configure_creates_rng(
        self,
        algorithm,
    ) -> None:
        algorithm.configure(
            identifier="test-algorithm",
            cost_function_wrapper=DummyCostFunctionWrapper,
            seed=42,
        )

        assert isinstance(algorithm._rng, np.random.Generator)

    def test_configure_uses_seed(
        self,
        algorithm,
    ) -> None:
        algorithm.configure(
            identifier="test-algorithm",
            cost_function_wrapper=DummyCostFunctionWrapper,
            seed=42,
        )

        first = algorithm._rng.random()

        algorithm.configure(
            identifier="test-algorithm",
            cost_function_wrapper=DummyCostFunctionWrapper,
            seed=42,
        )

        second = algorithm._rng.random()

        assert first == second

    def test_population_returns_internal_population(
        self,
        algorithm,
    ) -> None:
        assert algorithm.population is algorithm._population

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

        algorithm.population["particle:0"].consolidate(consolidate_new=True)

        assert algorithm.new_particles_id == ["particle:1"]

    def test_get_unmodified_particle_creates_candidate_from_current_state(
        self,
        algorithm,
    ) -> None:
        algorithm.create_particle(
            identifier="particle:0",
            variables={"x": 1.0, "y": 2.0},
            fitness=5.0,
        )

        particle = algorithm.get_unmodified_particle("particle:0")

        assert particle.variables == {
            "x": 1.0,
            "y": 2.0,
        }
        assert particle.fitness == 5.0

        assert particle.candidate_variables == particle.variables
        assert particle.candidate_fitness == particle.fitness

    def test_get_unmodified_particle_returns_same_particle(
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
            DummyParticle(
                identifier="particle:0",
                variables={"x": 1.0},
                fitness=1.0,
            ),
            DummyParticle(
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

    def test_update_population_replaces_particle_with_same_identifier(
        self,
        algorithm,
    ) -> None:
        original = DummyParticle(
            identifier="particle:0",
            variables={"x": 1.0},
            fitness=1.0,
        )

        replacement = DummyParticle(
            identifier="particle:0",
            variables={"x": 5.0},
            fitness=25.0,
        )

        algorithm.update_population([original])
        algorithm.update_population([replacement])

        assert algorithm.population["particle:0"] is replacement
        assert algorithm.population["particle:0"].fitness == 25.0

    def test_update_solution_state_sets_iteration_best_and_worst(
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

        assert algorithm.iter_best.identifier == "particle:1"
        assert algorithm.iter_worst.identifier == "particle:2"

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

    def test_update_solution_state_local_best_is_independent_copy(
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

        algorithm.population["particle:0"].recreate(
            variables={"x": 100.0},
            fitness=10000.0,
        )

        assert algorithm.local_best.fitness == 2.0
        assert algorithm.local_best.variables == {"x": 1.0}

    def test_update_solution_state_does_not_replace_better_local_best(
        self,
        algorithm,
    ) -> None:
        algorithm.create_particle(
            identifier="particle:0",
            variables={"x": 1.0},
            fitness=2.0,
        )

        algorithm.update_solution_state()

        first_local_best = algorithm.local_best

        algorithm.create_particle(
            identifier="particle:1",
            variables={"x": 2.0},
            fitness=10.0,
        )

        algorithm.update_solution_state()

        assert algorithm.local_best is first_local_best
        assert algorithm.local_best.identifier == "particle:0"
        assert algorithm.local_best.fitness == 2.0

    def test_update_solution_state_replaces_worse_local_best(
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

    @pytest.mark.parametrize(
        "fitness",
        [None, np.nan],
    )
    def test_get_fitness_rejects_missing_fitness(
        self,
        fitness,
    ) -> None:
        particle = DummyParticle(
            identifier="particle:0",
            variables={"x": 1.0},
            fitness=fitness,
        )

        if fitness is None:
            with pytest.raises(
                RuntimeError,
                match="does not have a fitness",
            ):
                AlgorithmBase.get_fitness(particle)
        else:
            # NaN is a valid float from the perspective of get_fitness().
            assert np.isnan(AlgorithmBase.get_fitness(particle))

    def test_get_fitness_returns_fitness(
        self,
        particle,
    ) -> None:
        assert AlgorithmBase.get_fitness(particle) == 5.0

    def test_iter_best_returns_none_before_solution_state_update(
        self,
        algorithm,
    ) -> None:
        assert algorithm.iter_best is None

    def test_iter_worst_returns_none_before_solution_state_update(
        self,
        algorithm,
    ) -> None:
        assert algorithm.iter_worst is None

    def test_iter_best_returns_particle_after_solution_state_update(
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

    def test_iter_worst_returns_particle_after_solution_state_update(
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

    def test_get_fitness_can_be_used_as_key_function(self) -> None:
        particles = [
            DummyParticle("a", {"x": 1.0}, 5.0),
            DummyParticle("b", {"x": 2.0}, 2.0),
            DummyParticle("c", {"x": 3.0}, 8.0),
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
