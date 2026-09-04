import json

import pytest
from doubles.algorithm import DummyParticle


def test_particle_state_and_properties() -> None:
    particle = DummyParticle(
        identifier="particle:0",
        variables={"x": 1.0, "y": 2.0},
        fitness=5.0,
    )

    assert particle.identifier == "particle:0"
    assert particle.variables == {"x": 1.0, "y": 2.0}
    assert particle.fitness == 5.0
    assert particle.new_particle is True
    assert particle.candidate_variables is None
    assert particle.candidate_fitness is None


def test_call_returns_serializable_state() -> None:
    particle = DummyParticle(
        identifier="particle:0",
        variables={"x": 1.0},
        fitness=2.0,
    )

    state = particle()

    assert state == {
        "identifier": "particle:0",
        "variables": {"x": 1.0},
        "fitness": 2.0,
    }


def test_dump_serializes_current_state() -> None:
    particle = DummyParticle(
        identifier="particle:0",
        variables={"x": 1.0},
        fitness=2.0,
    )

    assert json.loads(particle.dump()) == particle()


def test_recreate_resets_particle_state() -> None:
    particle = DummyParticle(
        identifier="particle:0",
        variables={"x": 1.0},
        fitness=2.0,
    )

    particle.candidate_variables = {"x": 3.0}
    particle.candidate_fitness = 9.0
    particle.recreate(
        variables={"x": 5.0},
        fitness=25.0,
    )

    assert particle.identifier == "particle:0"
    assert particle.variables == {"x": 5.0}
    assert particle.fitness == 25.0
    assert particle.new_particle is True
    assert particle.candidate_variables is None
    assert particle.candidate_fitness is None


def test_update_creates_candidate_solution() -> None:
    particle = DummyParticle(
        identifier="particle:0",
        variables={"x": 1.0},
        fitness=1.0,
    )

    particle.update(
        variables={"x": 3.0},
        fitness_function=lambda variables: variables["x"] ** 2,
    )

    assert particle.variables == {"x": 1.0}
    assert particle.fitness == 1.0
    assert particle.candidate_variables == {"x": 3.0}
    assert particle.candidate_fitness == 9.0


def test_update_rejects_none_variables() -> None:
    particle = DummyParticle(
        identifier="particle:0",
        variables={"x": 1.0},
        fitness=1.0,
    )

    with pytest.raises(ValueError, match="Variables cannot be None"):
        particle.update(
            variables=None,  # type: ignore[arg-type]
            fitness_function=lambda variables: 0.0,
        )


def test_consolidate_applies_candidate_when_requested() -> None:
    particle = DummyParticle(
        identifier="particle:0",
        variables={"x": 1.0},
        fitness=1.0,
    )

    particle.update(
        variables={"x": 3.0},
        fitness_function=lambda variables: variables["x"] ** 2,
    )

    particle.consolidate(consolidate_new=True)

    assert particle.variables == {"x": 3.0}
    assert particle.fitness == 9.0
    assert particle.new_particle is False
    assert particle.candidate_variables is None
    assert particle.candidate_fitness is None


def test_consolidate_rejects_missing_candidate() -> None:
    particle = DummyParticle(
        identifier="particle:0",
        variables={"x": 1.0},
        fitness=1.0,
    )

    particle._new_particle = False

    with pytest.raises(
        RuntimeError,
        match="No candidate solution available",
    ):
        particle.consolidate(consolidate_new=True)


def test_consolidate_can_reject_candidate() -> None:
    particle = DummyParticle(
        identifier="particle:0",
        variables={"x": 1.0},
        fitness=1.0,
    )

    particle.update(
        variables={"x": 3.0},
        fitness_function=lambda variables: variables["x"] ** 2,
    )

    particle.consolidate(consolidate_new=False)

    assert particle.variables == {"x": 1.0}
    assert particle.fitness == 1.0
    assert particle.new_particle is False
    assert particle.candidate_variables is None
    assert particle.candidate_fitness is None


def test_consolidate_new_particle_with_fitness() -> None:
    particle = DummyParticle(
        identifier="particle",
        variables={"x": 1.0},
        fitness=10.0,
    )

    assert particle.new_particle

    particle.consolidate(consolidate_new=True)

    assert not particle.new_particle
    assert particle.variables == {"x": 1.0}
    assert particle.fitness == 10.0
    assert particle.candidate_variables is None
    assert particle.candidate_fitness is None
