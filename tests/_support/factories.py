"""Explicit setup helpers shared by tests without hiding test assertions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from _support.numerics import BASE_SEED
from _support.objectives import encoded_sphere
from tyrannis import Optimizer
from tyrannis.core.algorithm import AlgorithmBase, CostFunctionWrapperBase
from tyrannis.core.space import SpaceBase


def configure_algorithm(
    algorithm: AlgorithmBase,
    boundaries: Mapping[str, tuple[float, float]],
    *,
    n_iterations: int = 5,
    n_particles: int = 8,
    seed: int = BASE_SEED,
    identifier: str = "test|algorithm",
    fitness_function: Callable[[dict[str, float]], float] = encoded_sphere,
) -> AlgorithmBase:
    """Initialize a concrete algorithm for direct unit or mathematical tests."""
    algorithm.initialize_context(
        fitness_function=fitness_function,
        boundaries=dict(boundaries),
        n_iter=n_iterations,
        n_particles=n_particles,
    )
    algorithm.configure(
        identifier=identifier,
        cost_function_wrapper=CostFunctionWrapperBase,
        seed=seed,
    )
    return algorithm


def make_optimizer(
    space: SpaceBase,
    algorithm: AlgorithmBase,
    *,
    n_iterations: int = 5,
    n_particles: int = 8,
    seed: int = BASE_SEED,
    **kwargs: Any,
) -> Optimizer:
    """Build a small public optimizer configuration for integration tests."""
    return Optimizer(
        space=space,
        algorithm=algorithm,
        n_iterations=n_iterations,
        n_particles=n_particles,
        seed=seed,
        **kwargs,
    )
