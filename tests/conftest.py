from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from tyrannis.core.algorithm import (
    AlgorithmBase,
    CostFunctionWrapperBase,
    ParticleBase,
)

# ============================================================================
# Test implementation of abstract classes
# ============================================================================


class DummyAlgorithm(AlgorithmBase):
    """Minimal concrete implementation of AlgorithmBase for testing."""

    def __init__(self) -> None:
        pass

    def create_particle(
        self,
        identifier: str,
        variables: dict[str, float] | None = None,
        fitness: float | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if variables is None:
            variables = {name: 0.0 for name in self._boundaries}

        self._population[identifier] = ParticleBase(
            identifier=identifier,
            variables=variables,
            fitness=fitness,
        )

    def delete_particle(self, identifier: str | None) -> None:
        if identifier is None:
            raise ValueError("Particle identifier cannot be None.")

        del self._population[identifier]

    def pre_iteration(self, actual_iter: int) -> None:
        raise NotImplementedError

    def initialize_particle(self, identifier: str) -> ParticleBase:
        raise NotImplementedError

    def update_particle(self, identifier: str) -> ParticleBase:
        raise NotImplementedError

    def post_iteration(self, actual_iter: int) -> None:
        raise NotImplementedError


# ============================================================================
# Generic fixtures
# ============================================================================


@pytest.fixture
def fitness_function() -> Callable[[dict[str, float]], float]:
    """Deterministic fitness function for core tests."""

    def sphere(variables: dict[str, float]) -> float:
        return sum(value**2 for value in variables.values())

    return sphere


@pytest.fixture
def boundaries() -> dict[str, tuple[float, float]]:
    """Generic search-space boundaries."""

    return {
        "x": (-10.0, 10.0),
        "y": (-10.0, 10.0),
    }


@pytest.fixture
def particle() -> ParticleBase:
    """Create a particle with a deterministic initial state."""

    return ParticleBase(
        identifier="particle:0",
        variables={
            "x": 1.0,
            "y": 2.0,
        },
        fitness=5.0,
    )


@pytest.fixture
def algorithm(
    fitness_function: Callable[[dict[str, float]], float],
    boundaries: dict[str, tuple[float, float]],
) -> DummyAlgorithm:
    """Create an AlgorithmBase test instance with its common context initialized."""

    algorithm = DummyAlgorithm()

    algorithm.initialize_context(
        fitness_function=fitness_function,
        boundaries=boundaries,
    )

    return algorithm


@pytest.fixture
def cost_function_wrapper() -> type[CostFunctionWrapperBase]:
    """Return the neutral cost-function wrapper used by core tests."""

    return CostFunctionWrapperBase


@pytest.fixture
def dummy_algorithm_class() -> type[DummyAlgorithm]:
    """Return the minimal concrete implementation of AlgorithmBase."""

    return DummyAlgorithm
