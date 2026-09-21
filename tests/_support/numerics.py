"""Shared numeric and reproducibility policy for tests."""

from __future__ import annotations

from math import sqrt
from numbers import Integral, Real

import numpy as np

BASE_SEED = 24_301
STRICT_RTOL = 1e-12
STRICT_ATOL = 1e-12
LINALG_RTOL = 1e-10
LINALG_ATOL = 1e-12


def seed_for(case_id: int) -> int:
    """Derive a stable integer seed for an explicitly identified test case."""
    if not isinstance(case_id, Integral) or isinstance(case_id, bool):
        raise TypeError("case_id must be an integer.")
    if case_id < 0:
        raise ValueError("case_id must be non-negative.")

    sequence = np.random.SeedSequence([BASE_SEED, int(case_id)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def frequency_tolerance(probability: float, sample_size: int, *, floor: float = 0.015) -> float:
    """Return a conservative six-sigma bound for an empirical frequency."""
    if not isinstance(probability, Real) or not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be finite and between zero and one.")
    if not isinstance(sample_size, Integral) or isinstance(sample_size, bool):
        raise TypeError("sample_size must be an integer.")
    if sample_size <= 0:
        raise ValueError("sample_size must be greater than zero.")
    if not isinstance(floor, Real) or floor < 0.0:
        raise ValueError("floor must be non-negative.")

    standard_error = sqrt(float(probability) * (1.0 - float(probability)) / sample_size)
    return max(float(floor), 6.0 * standard_error)
