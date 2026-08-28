import numpy as np


def booth(x: dict[str, float]) -> float:
    """Booth function with a plate-shaped surface.

    Number of inputs: 2 (x1, x2).
    Recommended domain: x1, x2 ∈ [-10, 10].
    Global minimum: f(1, 3) = 0 at x = (1, 3).
    """
    x1, x2 = np.fromiter(x.values(), dtype=float)

    return float((x1 + 2 * x2 - 7) ** 2 + (2 * x1 + x2 - 5) ** 2)


def matyas(x: dict[str, float]) -> float:
    """Matyas function with a plate-shaped surface.

    The function has no local minima other than the global minimum.

    Number of inputs: 2 (x1, x2).
    Recommended domain: x1, x2 ∈ [-10, 10].
    Global minimum: f(0, 0) = 0 at x = (0, 0).
    """
    x1, x2 = np.fromiter(x.values(), dtype=float)

    return float(0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2)


def mccormick(x: dict[str, float]) -> float:
    """McCormick function with a plate-shaped surface.

    Number of inputs: 2 (x1, x2).
    Recommended domain: x1 ∈ [-1.5, 4], x2 ∈ [-3, 4].
    Global minimum: f(-0.54719, -1.54719) ≈ -1.9133.
    """
    x1, x2 = np.fromiter(x.values(), dtype=float)

    return float(np.sin(x1 + x2) + (x1 - x2) ** 2 - 1.5 * x1 + 2.5 * x2 + 1)


def power_sum(x: dict[str, float]) -> float:
    """Power Sum function with a plate-shaped surface.

    The function uses the standard recommended parameter vector
    b = (8, 18, 44, 114) for d = 4.

    Number of inputs: Arbitrary (d >= 1).
    Recommended domain: xi ∈ [0, d].
    Global minimum: f(1, 2, 3, 4) = 0 for d = 4.
    """
    values = np.fromiter(x.values(), dtype=float)
    dimension = values.size

    b = np.array([8, 18, 44, 114], dtype=float)

    if dimension != b.size:
        raise ValueError("The standard Power Sum benchmark uses exactly 4 inputs.")

    powers = np.arange(1, dimension + 1)

    return float(np.sum((values**powers - b) ** 2))


def zakharov(x: dict[str, float]) -> float:
    """Zakharov function with a plate-shaped surface.

    The function has no local minima other than the global minimum.

    Number of inputs: Arbitrary (d >= 1).
    Recommended domain: xi ∈ [-5, 10].
    Global minimum: f(0, ..., 0) = 0 at x = (0, ..., 0).
    """
    values = np.fromiter(x.values(), dtype=float)
    indices = np.arange(1, values.size + 1)

    linear_term = 0.5 * np.sum(indices * values)

    return float(np.sum(values**2) + linear_term**2 + linear_term**4)
