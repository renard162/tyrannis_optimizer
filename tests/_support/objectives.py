"""Small, serializable objectives shared by tests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


def encoded_sphere(variables: Mapping[str, float]) -> float:
    """Evaluate the sum of squares of an algorithm's encoded variables."""
    return float(sum(float(value) ** 2 for value in variables.values()))


def sphere(*values: float, **named_values: float) -> float:
    """Evaluate a positional or keyword sum-of-squares objective."""
    inputs = named_values.values() if named_values else values
    return float(sum(float(value) ** 2 for value in inputs))


def constant_objective(*values: Any, **named_values: Any) -> float:
    """Return a deterministic constant regardless of the decoded inputs."""
    del values, named_values
    return 1.0


def fails_for_positive(*values: float, **named_values: float) -> float:
    """Raise predictably when at least one decoded numeric input is positive."""
    inputs = named_values.values() if named_values else values
    if any(float(value) > 0.0 for value in inputs):
        raise ValueError("positive values are rejected by this test objective")
    return 0.0


@dataclass
class CountingObjective:
    """Callable objective used to observe cache evaluation behavior."""

    return_value: float = 1.0
    calls: int = 0

    def __call__(self, *values: Any, **named_values: Any) -> float:
        del values, named_values
        self.calls += 1
        return self.return_value
