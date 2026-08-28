import numpy as np


def dejong_5(x: dict[str, float]) -> float:
    """De Jong function N. 5 with very sharp drops on a flat surface.

    www.sfu.ca/~ssurjano/optimization.html

    Number of inputs: 2 (x1, x2).
    Recommended domain: x1, x2 ∈ [-65.536, 65.536].
    Global minimum: f(-32, -32) ≈ 0.998004.
    """
    x1, x2 = np.fromiter(x.values(), dtype=float)

    a = np.array([-32, -16, 0, 16, 32], dtype=float)

    grid_x1 = np.tile(a, 5)
    grid_x2 = np.repeat(a, 5)

    indices = np.arange(1, 26, dtype=float)

    denominator = indices + (x1 - grid_x1) ** 6 + (x2 - grid_x2) ** 6

    return float(1 / (0.002 + np.sum(1 / denominator)))


def easom(x: dict[str, float]) -> float:
    """Easom function with a narrow global minimum.

    The function is unimodal, although the global minimum occupies
    a very small region relative to the search domain.

    www.sfu.ca/~ssurjano/optimization.html

    Number of inputs: 2 (x1, x2).
    Recommended domain: x1, x2 ∈ [-100, 100].
    Global minimum: f(π, π) = -1 at x = (π, π).
    """
    x1, x2 = np.fromiter(x.values(), dtype=float)

    return float(
        -np.cos(x1) * np.cos(x2) * np.exp(-((x1 - np.pi) ** 2) - (x2 - np.pi) ** 2)
    )


def michalewicz(x: dict[str, float]) -> float:
    """Michalewicz function with steep valleys and ridges.

    The function has d! local minima. The parameter m controls the
    steepness of the valleys and ridges; the recommended value is m=10.

    www.sfu.ca/~ssurjano/optimization.html

    Number of inputs: Arbitrary (d >= 1).
    Recommended domain: xi ∈ [0, π].
    Default parameter: m = 10.

    Global minimum depends on the dimension. For d=2, the global
    minimum is approximately f(2.20, 1.57) ≈ -1.8013.
    """
    values = np.fromiter(x.values(), dtype=float)
    indices = np.arange(1, values.size + 1, dtype=float)
    m = 10

    return float(
        -np.sum(np.sin(values) * np.sin(indices * values**2 / np.pi) ** (2 * m))
    )
