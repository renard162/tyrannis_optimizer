"""Reusable contract assertions for Tyrannis tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from tyrannis import Optimizer
from tyrannis.core.algorithm import ParticleBase


def assert_variables_within_bounds(
    variables: Mapping[str, float],
    boundaries: Mapping[str, tuple[float, float]],
) -> None:
    """Assert that variables have exactly the expected names and valid values."""
    assert set(variables) == set(boundaries)
    for name, (lower, upper) in boundaries.items():
        assert lower <= variables[name] <= upper, name


def assert_valid_permutation(
    values: Sequence[Any],
    choices: Sequence[Any],
) -> None:
    """Assert permutation equality without requiring choices to be hashable."""
    assert len(values) == len(choices)
    remaining = list(choices)
    for value in values:
        for index, expected in enumerate(remaining):
            if value == expected:
                remaining.pop(index)
                break
        else:
            raise AssertionError(f"Unexpected permutation value: {value!r}")
    assert not remaining


def assert_particle_state_consistent(
    particle: ParticleBase,
    boundaries: Mapping[str, tuple[float, float]],
) -> None:
    """Assert common particle-state invariants without algorithm-specific details."""
    assert_variables_within_bounds(particle.variables, boundaries)
    if particle.candidate_variables is not None:
        assert_variables_within_bounds(particle.candidate_variables, boundaries)


def assert_optimizer_result_consistent(optimizer: Optimizer) -> None:
    """Assert consistency among the public result, solution, and fitness APIs."""
    result = optimizer.result_
    assert result is not None
    assert optimizer.best_solution == result["variables"]
    assert optimizer.best_fitness == result["fitness"]


def assert_reproducible(
    optimizer_factory: Callable[[], Optimizer],
) -> None:
    """Run two fresh optimizers and assert identical public outputs and history."""
    first = optimizer_factory().fit()
    second = optimizer_factory().fit()

    assert first.result_ == second.result_
    assert list(first.history_generator) == list(second.history_generator)
