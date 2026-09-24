"""Smoke-test every optimization algorithm exposed by the public package API."""

from collections.abc import Callable

import numpy as np
import pytest
from _support.assertions import (
    assert_optimizer_result_consistent,
    assert_variables_within_bounds,
)
from _support.factories import make_optimizer
from _support.numerics import BASE_SEED
from _support.objectives import constant_objective, sphere

import tyrannis.algorithm as algorithm_module
from tyrannis.core.algorithm import AlgorithmBase
from tyrannis.space.continuous import Continuous

N_PARTICLES = 20
N_ITERATIONS = 3


def _registered_algorithms() -> tuple[type[AlgorithmBase], ...]:
    """Return every public algorithm registered by ``tyrannis.algorithm``."""
    algorithms: list[type[AlgorithmBase]] = []

    for name in algorithm_module.__all__:
        algorithm_class = getattr(algorithm_module, name)

        if not isinstance(algorithm_class, type) or not issubclass(
            algorithm_class, AlgorithmBase
        ):
            raise TypeError(
                f"tyrannis.algorithm.__all__ entry {name!r} is not an AlgorithmBase "
                "subclass."
            )

        algorithms.append(algorithm_class)

    return tuple(algorithms)


def all_positive_infinity(**variables: float) -> float:
    """Represent a search space in which every evaluated solution is infeasible."""
    del variables
    return np.inf


def all_negative_infinity(**variables: float) -> float:
    """Represent a search space in which every evaluated solution is optimal."""
    del variables
    return -np.inf


ALGORITHMS = _registered_algorithms()


@pytest.mark.parametrize(
    "algorithm_class",
    ALGORITHMS,
    ids=lambda algorithm_class: algorithm_class.__name__,
)
@pytest.mark.parametrize(
    "objective",
    (
        pytest.param(sphere, id="sphere"),
        pytest.param(constant_objective, id="constant"),
        pytest.param(all_positive_infinity, id="positive-infinity"),
        pytest.param(all_negative_infinity, id="negative-infinity"),
    ),
)
def test_public_algorithm_smoke(
    algorithm_class: type[AlgorithmBase],
    objective: Callable[..., float],
    small_bounds: dict[str, tuple[float, float]],
) -> None:
    """Run a minimal public optimization and validate its returned result."""
    optimizer = make_optimizer(
        Continuous(small_bounds, cost_function=objective),
        algorithm_class(),
        n_iterations=N_ITERATIONS,
        n_particles=N_PARTICLES,
        seed=BASE_SEED,
    ).fit()

    assert_optimizer_result_consistent(optimizer)

    best_solution = optimizer.best_solution
    assert isinstance(best_solution, dict)
    assert_variables_within_bounds(best_solution, small_bounds)

    expected_fitness = objective(**best_solution)

    assert not np.isnan(optimizer.best_fitness)
    assert optimizer.best_fitness == expected_fitness
