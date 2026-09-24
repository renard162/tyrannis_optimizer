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
from tyrannis import Optimizer
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


def partially_infeasible(**variables: float) -> float:
    """Return finite fitness on one half-space and +inf on the other."""
    if variables["x"] > 0:
        return np.inf

    return sphere(**variables)


def mixed_extreme_fitness(**variables: float) -> float:
    """Expose -inf, finite, and +inf regions in the same search space."""
    x = variables["x"]

    if x < 0:
        return -np.inf

    if x > 2:
        return np.inf

    return sphere(**variables)


ALGORITHMS = _registered_algorithms()


def _run_algorithm(
    algorithm_class: type[AlgorithmBase],
    objective: Callable[..., float],
    small_bounds: dict[str, tuple[float, float]],
) -> Optimizer:
    """Run one public optimization using only default algorithm parameters."""
    return make_optimizer(
        space=Continuous(boundaries=small_bounds, cost_function=objective),
        algorithm=algorithm_class(),
        n_iterations=N_ITERATIONS,
        n_particles=N_PARTICLES,
        seed=BASE_SEED,
    ).fit()


def _assert_common_result(
    optimizer: Optimizer,
    objective: Callable[..., float],
    small_bounds: dict[str, tuple[float, float]],
) -> dict[str, float]:
    """Assert public result consistency shared by every smoke scenario."""
    assert_optimizer_result_consistent(optimizer)

    best_solution = optimizer.best_solution
    assert isinstance(best_solution, dict)

    assert_variables_within_bounds(variables=best_solution, boundaries=small_bounds)

    expected_fitness = objective(**best_solution)

    assert not np.isnan(optimizer.best_fitness)
    assert optimizer.best_fitness == expected_fitness

    return best_solution


@pytest.mark.parametrize(
    "algorithm_class", ALGORITHMS, ids=lambda algorithm_class: algorithm_class.__name__
)
def test_finite_objective_smoke(
    algorithm_class: type[AlgorithmBase], small_bounds: dict[str, tuple[float, float]]
) -> None:
    """Every public algorithm must optimize a regular finite objective."""
    optimizer = _run_algorithm(
        algorithm_class=algorithm_class, objective=sphere, small_bounds=small_bounds
    )
    _assert_common_result(
        optimizer=optimizer, objective=sphere, small_bounds=small_bounds
    )

    assert np.isfinite(optimizer.best_fitness)
    assert optimizer.best_fitness >= 0


@pytest.mark.parametrize(
    "algorithm_class", ALGORITHMS, ids=lambda algorithm_class: algorithm_class.__name__
)
def test_constant_objective_smoke(
    algorithm_class: type[AlgorithmBase], small_bounds: dict[str, tuple[float, float]]
) -> None:
    """Every public algorithm must tolerate a fully tied finite population."""
    optimizer = _run_algorithm(
        algorithm_class=algorithm_class,
        objective=constant_objective,
        small_bounds=small_bounds,
    )
    _assert_common_result(
        optimizer=optimizer, objective=constant_objective, small_bounds=small_bounds
    )

    assert optimizer.best_fitness == 1.0


@pytest.mark.parametrize(
    "algorithm_class", ALGORITHMS, ids=lambda algorithm_class: algorithm_class.__name__
)
def test_partially_infeasible_objective_smoke(
    algorithm_class: type[AlgorithmBase], small_bounds: dict[str, tuple[float, float]]
) -> None:
    """Every public algorithm must operate with finite and +inf fitness together."""
    optimizer = _run_algorithm(
        algorithm_class=algorithm_class,
        objective=partially_infeasible,
        small_bounds=small_bounds,
    )
    best_solution = _assert_common_result(
        optimizer=optimizer, objective=partially_infeasible, small_bounds=small_bounds
    )

    assert np.isfinite(optimizer.best_fitness)
    assert best_solution["x"] <= 0


@pytest.mark.parametrize(
    "algorithm_class", ALGORITHMS, ids=lambda algorithm_class: algorithm_class.__name__
)
def test_mixed_extreme_fitness_smoke(
    algorithm_class: type[AlgorithmBase], small_bounds: dict[str, tuple[float, float]]
) -> None:
    """Every public algorithm must preserve -inf as the best mixed fitness."""
    optimizer = _run_algorithm(
        algorithm_class=algorithm_class,
        objective=mixed_extreme_fitness,
        small_bounds=small_bounds,
    )
    best_solution = _assert_common_result(
        optimizer=optimizer, objective=mixed_extreme_fitness, small_bounds=small_bounds
    )

    assert np.isneginf(optimizer.best_fitness)
    assert best_solution["x"] < 0


@pytest.mark.parametrize(
    "algorithm_class", ALGORITHMS, ids=lambda algorithm_class: algorithm_class.__name__
)
def test_all_positive_infinity_smoke(
    algorithm_class: type[AlgorithmBase], small_bounds: dict[str, tuple[float, float]]
) -> None:
    """Every public algorithm must tolerate an entirely +inf objective."""
    optimizer = _run_algorithm(
        algorithm_class=algorithm_class,
        objective=all_positive_infinity,
        small_bounds=small_bounds,
    )
    _assert_common_result(
        optimizer=optimizer, objective=all_positive_infinity, small_bounds=small_bounds
    )

    assert np.isposinf(optimizer.best_fitness)


@pytest.mark.parametrize(
    "algorithm_class", ALGORITHMS, ids=lambda algorithm_class: algorithm_class.__name__
)
def test_all_negative_infinity_smoke(
    algorithm_class: type[AlgorithmBase], small_bounds: dict[str, tuple[float, float]]
) -> None:
    """Every public algorithm must preserve an entirely -inf objective."""
    optimizer = _run_algorithm(
        algorithm_class=algorithm_class,
        objective=all_negative_infinity,
        small_bounds=small_bounds,
    )
    _assert_common_result(
        optimizer=optimizer, objective=all_negative_infinity, small_bounds=small_bounds
    )

    assert np.isneginf(optimizer.best_fitness)
