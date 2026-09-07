import numpy as np


def three_hump_camel(x1: float, x2: float) -> float:
    """Three-Hump Camel function with a valley-shaped surface.

    www.sfu.ca/~ssurjano/optimization.html

    Number of inputs: 2 (x1, x2).
    Recommended domain: x1, x2 ∈ [-5, 5].
    Global minimum: f(0, 0) = 0 at x = (0, 0).
    """
    return float(2 * x1**2 - 1.05 * x1**4 + x1**6 / 6 + x1 * x2 + x2**2)


def six_hump_camel(x1: float, x2: float) -> float:
    """Six-Hump Camel function with a valley-shaped surface.

    The function has six local minima, two of which are global.

    www.sfu.ca/~ssurjano/optimization.html

    Number of inputs: 2 (x1, x2).
    Recommended domain: x1 ∈ [-3, 3], x2 ∈ [-2, 2].
    Global minima: f ≈ -1.031628 at
    (±0.089842, ∓0.712656).
    """
    return float(
        (4 - 2.1 * x1**2 + x1**4 / 3) * x1**2 + x1 * x2 + (-4 + 4 * x2**2) * x2**2
    )


def dixon_price(*x: list[float]) -> float:
    """Dixon-Price function with a valley-shaped surface.

    The function is unimodal and has a single global minimum.

    www.sfu.ca/~ssurjano/optimization.html

    Number of inputs: Arbitrary (d >= 1).
    Recommended domain: xi ∈ [-10, 10].
    Global minimum: f(x*) = 0, where
    x*_i = 2^(-(2^i - 2) / 2^i), for i = 1, ..., d.
    """
    values = np.array(x, dtype=float)
    indices = np.arange(2, values.size + 1)

    return float(
        (values[0] - 1) ** 2
        + np.sum(indices * (2 * values[1:] ** 2 - values[:-1]) ** 2)
    )


def rosenbrock(*x: list[float]) -> float:
    """Rosenbrock function, also known as the Valley or Banana function.

    The function is unimodal, with a narrow parabolic valley leading
    to the global minimum.

    www.sfu.ca/~ssurjano/optimization.html

    Number of inputs: Arbitrary (d >= 2).
    Recommended domain: xi ∈ [-5, 10].
    Alternative restricted domain: xi ∈ [-2.048, 2.048].
    Global minimum: f(1, ..., 1) = 0 at x = (1, ..., 1).
    """
    values = np.array(x, dtype=float)

    return float(
        np.sum(100 * (values[1:] - values[:-1] ** 2) ** 2 + (values[:-1] - 1) ** 2)
    )
