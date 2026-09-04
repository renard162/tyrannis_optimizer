import inspect

import numpy as np
import pytest
from doubles.algorithm import (
    DummyAlgorithm,
    DummyCostFunctionWrapper,
    DummyParticle,
)

from tyrannis.core.algorithm import AlgorithmBase


def test_initialize_context_sets_initial_state() -> None:
    algorithm = DummyAlgorithm()

    fitness = lambda variables: sum(variables.values())
    boundaries = {
        "x": (-1.0, 1.0),
        "y": (-2.0, 2.0),
    }

    algorithm.initialize_context(
        fitness_function=fitness,
        boundaries=boundaries,
    )

    assert algorithm._fitness_function is fitness
    assert algorithm._boundaries == boundaries
    assert algorithm.population == {}
    assert algorithm.local_best is None
    assert algorithm.iter_best is None
    assert algorithm.iter_worst is None


def test_configure_sets_identifier_wrapper_and_random_generator() -> None:
    algorithm = DummyAlgorithm()

    fitness = lambda variables: sum(variables.values())

    algorithm.initialize_context(
        fitness_function=fitness,
        boundaries={"x": (-1.0, 1.0)},
    )

    algorithm.configure(
        identifier="algorithm:0",
        cost_function_wrapper=DummyCostFunctionWrapper,
        seed=42,
    )

    assert algorithm.identifier == "algorithm:0"
    assert isinstance(
        algorithm._fitness_function,
        DummyCostFunctionWrapper,
    )
    assert isinstance(algorithm._rng, np.random.Generator)


def test_properties_return_current_algorithm_state() -> None:
    algorithm = DummyAlgorithm()

    algorithm.initialize_context(
        fitness_function=lambda variables: sum(variables.values()),
        boundaries={"x": (-1.0, 1.0)},
    )

    algorithm.configure(
        identifier="algorithm:0",
        cost_function_wrapper=DummyCostFunctionWrapper,
        seed=42,
    )

    best = DummyParticle(
        identifier="particle:0",
        variables={"x": 1.0},
        fitness=1.0,
    )
    worst = DummyParticle(
        identifier="particle:1",
        variables={"x": 2.0},
        fitness=4.0,
    )

    algorithm.update_population([best, worst])
    algorithm._iter_best = "particle:0"
    algorithm._iter_worst = "particle:1"

    assert algorithm.identifier == "algorithm:0"
    assert algorithm.population == {
        "particle:0": best,
        "particle:1": worst,
    }
    assert algorithm.iter_best is best
    assert algorithm.iter_worst is worst
    assert algorithm.local_best is None
    assert algorithm.new_particles_id == [
        "particle:0",
        "particle:1",
    ]


def test_get_unmodified_particle_creates_candidate_from_current_state() -> None:
    algorithm = DummyAlgorithm()

    algorithm.initialize_context(
        fitness_function=lambda variables: sum(variables.values()),
        boundaries={"x": (-1.0, 1.0)},
    )

    particle = DummyParticle(
        identifier="particle:0",
        variables={"x": 2.0},
        fitness=4.0,
    )

    algorithm.update_population([particle])

    result = algorithm.get_unmodified_particle("particle:0")

    assert result is particle
    assert particle.candidate_variables == particle.variables
    assert particle.candidate_fitness == particle.fitness


def test_update_population_adds_and_replaces_particles() -> None:
    algorithm = DummyAlgorithm()

    algorithm.initialize_context(
        fitness_function=lambda variables: sum(variables.values()),
        boundaries={"x": (-1.0, 1.0)},
    )

    first = DummyParticle(
        identifier="particle:0",
        variables={"x": 1.0},
        fitness=1.0,
    )
    replacement = DummyParticle(
        identifier="particle:0",
        variables={"x": 2.0},
        fitness=4.0,
    )
    second = DummyParticle(
        identifier="particle:1",
        variables={"x": 3.0},
        fitness=9.0,
    )

    algorithm.update_population([first])
    algorithm.update_population([replacement, second])

    assert algorithm.population == {
        "particle:0": replacement,
        "particle:1": second,
    }


def test_update_solution_state_selects_best_worst_and_local_best() -> None:
    algorithm = DummyAlgorithm()

    algorithm.initialize_context(
        fitness_function=lambda variables: sum(variables.values()),
        boundaries={"x": (-1.0, 1.0)},
    )

    best = DummyParticle(
        identifier="particle:0",
        variables={"x": 1.0},
        fitness=1.0,
    )
    worst = DummyParticle(
        identifier="particle:1",
        variables={"x": 2.0},
        fitness=4.0,
    )

    algorithm.update_population([best, worst])
    algorithm.update_solution_state()

    assert algorithm.iter_best is best
    assert algorithm.iter_worst is worst
    assert algorithm.local_best is not best
    assert algorithm.local_best() == best()  # type: ignore

    worse = DummyParticle(
        identifier="particle:2",
        variables={"x": 3.0},
        fitness=9.0,
    )

    algorithm.update_population([worse])
    algorithm.update_solution_state()

    assert algorithm.iter_best is best
    assert algorithm.iter_worst is worse
    assert algorithm.local_best() == best()  # type: ignore


def test_get_fitness_returns_fitness_and_rejects_undefined_fitness() -> None:
    particle = DummyParticle(
        identifier="particle:0",
        variables={"x": 1.0},
        fitness=2.0,
    )

    assert AlgorithmBase.get_fitness(particle) == 2.0

    particle.recreate(
        variables={"x": 1.0},
        fitness=None,
    )

    with pytest.raises(
        RuntimeError,
        match="does not have a fitness",
    ):
        AlgorithmBase.get_fitness(particle)


def test_is_abstract() -> None:
    assert inspect.isabstract(AlgorithmBase)
