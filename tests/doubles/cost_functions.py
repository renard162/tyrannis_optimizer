import numpy as np


def sphere(x: dict[str, float]) -> float:
    """Sphere function with a simple bowl-shaped surface.

    The function is continuous, convex, and unimodal.

    www.sfu.ca/~ssurjano/optimization.html

    Number of inputs: Arbitrary (d >= 1).
    Recommended domain: xi ∈ [-5.12, 5.12].
    Global minimum: f(0, ..., 0) = 0 at x = (0, ..., 0).
    """
    values = np.fromiter(x.values(), dtype=float)

    return float(np.sum(values**2))
