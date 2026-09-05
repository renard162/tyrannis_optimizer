import numpy as np
import pytest
from doubles.algorithm import DummyCostFunctionWrapper

from tyrannis.algorithm.pso import PSO, PSOParticle


def create_algorithm() -> PSO:
    algorithm = PSO()

    algorithm.initialize_context(
        fitness_function=lambda variables: sum(
            value**2 for value in variables.values()
        ),
        boundaries={
            "x": (-5.0, 5.0),
            "y": (-5.0, 5.0),
        },
    )

    algorithm.configure(
        identifier="test",
        cost_function_wrapper=DummyCostFunctionWrapper,
        seed=42,
    )

    return algorithm


def test_particle_properties() -> None:
    particle = PSOParticle(
        identifier="particle:0",
        variables={"x": 1.0},
        fitness=2.0,
        velocity={"x": 0.5},
        personal_best_variables={"x": 1.0},
        personal_best_fitness=2.0,
    )

    assert particle.identifier == "particle:0"
    assert particle.variables == {"x": 1.0}

    assert particle.fitness is not None
    np.testing.assert_allclose(particle.fitness, 2.0)

    assert particle.velocity is not None
    np.testing.assert_allclose(particle.velocity["x"], 0.5)

    assert particle.personal_best_variables is not None
    np.testing.assert_allclose(
        particle.personal_best_variables["x"],
        1.0,
    )

    assert particle.personal_best_fitness is not None
    np.testing.assert_allclose(
        particle.personal_best_fitness,
        2.0,
    )


def test_particle_recreate() -> None:
    particle = PSOParticle(
        identifier="particle:0",
        variables={"x": 1.0},
        fitness=2.0,
        velocity={"x": 0.5},
        personal_best_variables={"x": 1.0},
        personal_best_fitness=2.0,
    )

    particle.recreate(
        variables={"x": 2.0},
        fitness=4.0,
        velocity={"x": -0.5},
        personal_best_variables={"x": 0.5},
        personal_best_fitness=0.25,
    )

    assert particle.new_particle
    assert particle.variables == {"x": 2.0}
    assert particle.velocity == {"x": -0.5}
    assert particle.personal_best_variables == {"x": 0.5}

    assert particle.fitness is not None
    np.testing.assert_allclose(particle.fitness, 4.0)

    assert particle.personal_best_fitness is not None
    np.testing.assert_allclose(
        particle.personal_best_fitness,
        0.25,
    )

    assert particle.candidate_variables is None
    assert particle.candidate_fitness is None


def test_update_personal_best() -> None:
    particle = PSOParticle(
        identifier="particle:0",
        variables={"x": 2.0},
        fitness=4.0,
        velocity={"x": 0.0},
        personal_best_variables={"x": 3.0},
        personal_best_fitness=9.0,
    )

    particle.update_personal_best()

    assert particle.personal_best_variables == {"x": 2.0}
    assert particle.personal_best_fitness is not None
    np.testing.assert_allclose(
        particle.personal_best_fitness,
        4.0,
    )


def test_update_personal_best_keeps_better_solution() -> None:
    particle = PSOParticle(
        identifier="particle:0",
        variables={"x": 2.0},
        fitness=4.0,
        velocity={"x": 0.0},
        personal_best_variables={"x": 1.0},
        personal_best_fitness=1.0,
    )

    particle.update_personal_best()

    assert particle.personal_best_variables == {"x": 1.0}
    assert particle.personal_best_fitness is not None
    np.testing.assert_allclose(
        particle.personal_best_fitness,
        1.0,
    )


def test_update_personal_best_without_fitness() -> None:
    particle = PSOParticle(
        identifier="particle:0",
        variables={"x": 2.0},
    )

    with pytest.raises(
        RuntimeError,
        match="does not have a fitness",
    ):
        particle.update_personal_best()


def test_create_particle() -> None:
    algorithm = create_algorithm()

    algorithm.create_particle("particle:0")

    particle = algorithm.population["particle:0"]

    assert isinstance(particle, PSOParticle)
    assert set(particle.variables) == {"x", "y"}

    assert particle.velocity is not None
    assert set(particle.velocity) == {"x", "y"}

    for name, (lower, upper) in algorithm._boundaries.items():
        assert lower <= particle.variables[name] <= upper
        assert -(upper - lower) <= particle.velocity[name] <= upper - lower


def test_create_particle_with_values() -> None:
    algorithm = create_algorithm()

    algorithm.create_particle(
        identifier="particle:0",
        variables={"x": 1.0, "y": 2.0},
        fitness=5.0,
        velocity={"x": 0.1, "y": 0.2},
        personal_best_variables={"x": 1.0, "y": 2.0},
        personal_best_fitness=5.0,
    )

    particle = algorithm.population["particle:0"]

    assert isinstance(particle, PSOParticle)
    assert particle.variables == {"x": 1.0, "y": 2.0}
    assert particle.velocity == {"x": 0.1, "y": 0.2}
    assert particle.personal_best_variables == {"x": 1.0, "y": 2.0}

    assert particle.fitness is not None
    np.testing.assert_allclose(particle.fitness, 5.0)

    assert particle.personal_best_fitness is not None
    np.testing.assert_allclose(
        particle.personal_best_fitness,
        5.0,
    )


def test_create_particle_without_identifier() -> None:
    algorithm = create_algorithm()

    with pytest.raises(
        ValueError,
        match="Particle identifier cannot be None",
    ):
        algorithm.create_particle(None)  # pyright: ignore[reportArgumentType]


def test_delete_particle() -> None:
    algorithm = create_algorithm()

    algorithm.create_particle("particle:0")
    algorithm.delete_particle("particle:0")

    assert algorithm.population == {}


def test_delete_particle_without_identifier() -> None:
    algorithm = create_algorithm()

    with pytest.raises(
        ValueError,
        match="Particle identifier cannot be None",
    ):
        algorithm.delete_particle(None)


def test_initialize_particle() -> None:
    algorithm = create_algorithm()

    algorithm.create_particle(
        identifier="particle:0",
        variables={"x": 2.0, "y": 1.0},
    )

    particle = algorithm.initialize_particle("particle:0")

    assert isinstance(particle, PSOParticle)
    assert not particle.new_particle
    assert particle.personal_best_variables == {
        "x": 2.0,
        "y": 1.0,
    }

    assert particle.fitness is not None
    np.testing.assert_allclose(particle.fitness, 5.0)

    assert particle.personal_best_fitness is not None
    np.testing.assert_allclose(
        particle.personal_best_fitness,
        5.0,
    )


def test_initialize_particle_rejects_wrong_particle_type() -> None:
    algorithm = create_algorithm()

    algorithm._population["particle:0"] = object()  # type: ignore[assignment]

    with pytest.raises(
        TypeError,
        match="must be an instance of PSOParticle",
    ):
        algorithm.initialize_particle("particle:0")


def test_update_particle() -> None:
    algorithm = create_algorithm()

    particle = PSOParticle(
        identifier="particle:0",
        variables={"x": 1.0, "y": 1.0},
        fitness=2.0,
        velocity={"x": 0.0, "y": 0.0},
        personal_best_variables={"x": 1.0, "y": 1.0},
        personal_best_fitness=2.0,
    )

    algorithm._population[particle.identifier] = particle
    algorithm._local_best = PSOParticle(
        identifier="best",
        variables={"x": 0.0, "y": 0.0},
        fitness=0.0,
    )

    result = algorithm.update_particle("particle:0")

    assert result is particle
    assert particle.candidate_variables is not None
    assert particle.candidate_fitness is not None
    assert particle.velocity is not None


def test_update_particle_without_local_best() -> None:
    algorithm = create_algorithm()

    algorithm.create_particle(
        identifier="particle:0",
        variables={"x": 1.0, "y": 1.0},
        fitness=2.0,
        velocity={"x": 0.0, "y": 0.0},
        personal_best_variables={"x": 1.0, "y": 1.0},
        personal_best_fitness=2.0,
    )

    with pytest.raises(
        RuntimeError,
        match="Local best particle has not been initialized",
    ):
        algorithm.update_particle("particle:0")


def test_update_particle_without_fitness() -> None:
    algorithm = create_algorithm()

    particle = PSOParticle(
        identifier="particle:0",
        variables={"x": 1.0, "y": 1.0},
        fitness=None,
        velocity={"x": 0.0, "y": 0.0},
        personal_best_variables={"x": 1.0, "y": 1.0},
        personal_best_fitness=2.0,
    )

    algorithm._population[particle.identifier] = particle
    algorithm._local_best = particle

    with pytest.raises(
        RuntimeError,
        match="does not have a fitness",
    ):
        algorithm.update_particle("particle:0")


def test_update_particle_without_velocity() -> None:
    algorithm = create_algorithm()

    particle = PSOParticle(
        identifier="particle:0",
        variables={"x": 1.0, "y": 1.0},
        fitness=2.0,
        velocity=None,
        personal_best_variables={"x": 1.0, "y": 1.0},
        personal_best_fitness=2.0,
    )

    algorithm._population[particle.identifier] = particle
    algorithm._local_best = particle

    with pytest.raises(
        RuntimeError,
        match="does not have a velocity",
    ):
        algorithm.update_particle("particle:0")


def test_update_particle_without_personal_best() -> None:
    algorithm = create_algorithm()

    particle = PSOParticle(
        identifier="particle:0",
        variables={"x": 1.0, "y": 1.0},
        fitness=2.0,
        velocity={"x": 0.0, "y": 0.0},
        personal_best_variables=None,
        personal_best_fitness=None,
    )

    algorithm._population[particle.identifier] = particle
    algorithm._local_best = particle

    with pytest.raises(
        RuntimeError,
        match="does not have a personal best",
    ):
        algorithm.update_particle("particle:0")


def test_post_iteration() -> None:
    algorithm = create_algorithm()

    particle = PSOParticle(
        identifier="particle:0",
        variables={"x": 1.0, "y": 1.0},
        fitness=2.0,
        velocity={"x": 0.0, "y": 0.0},
        personal_best_variables={"x": 1.0, "y": 1.0},
        personal_best_fitness=2.0,
    )

    algorithm._population[particle.identifier] = particle

    algorithm.post_iteration(0)

    assert particle.fitness is not None
    np.testing.assert_allclose(particle.fitness, 2.0)
    assert particle.candidate_fitness is None

    particle.update(
        variables={"x": 0.5, "y": 0.5},
        fitness_function=algorithm._fitness_function,
    )

    algorithm.post_iteration(1)

    assert particle.fitness is not None
    np.testing.assert_allclose(particle.fitness, 0.5)

    assert particle.personal_best_fitness is not None
    np.testing.assert_allclose(
        particle.personal_best_fitness,
        0.5,
    )

    assert algorithm.local_best is not None


def test_post_iteration_keeps_current_solution_when_candidate_is_worse() -> None:
    algorithm = create_algorithm()

    particle = PSOParticle(
        identifier="particle:0",
        variables={"x": 0.5, "y": 0.5},
        fitness=0.5,
        velocity={"x": 0.0, "y": 0.0},
        personal_best_variables={"x": 0.5, "y": 0.5},
        personal_best_fitness=0.5,
    )

    algorithm._population[particle.identifier] = particle
    algorithm._local_best = particle

    particle.update(
        variables={"x": 1.0, "y": 1.0},
        fitness_function=algorithm._fitness_function,
    )

    algorithm.post_iteration(1)

    assert particle.fitness is not None
    np.testing.assert_allclose(particle.fitness, 0.5)

    assert particle.personal_best_fitness is not None
    np.testing.assert_allclose(
        particle.personal_best_fitness,
        0.5,
    )
