import numpy as np


def bohachevsky_1(x: dict[str, float]) -> float:
    """Bohachevsky function N. 1 with a bowl-shaped surface.

    Number of inputs: 2 (x1, x2).
    Recommended domain: x1, x2 ∈ [-100, 100].
    Global minimum: f(0, 0) = 0 at x = (0, 0).
    """
    x1, x2 = np.fromiter(x.values(), dtype=float)

    return float(
        x1**2
        + 2 * x2**2
        - 0.3 * np.cos(3 * np.pi * x1)
        - 0.4 * np.cos(4 * np.pi * x2)
        + 0.7
    )


def bohachevsky_2(x: dict[str, float]) -> float:
    """Bohachevsky function N. 2 with a bowl-shaped surface.

    Number of inputs: 2 (x1, x2).
    Recommended domain: x1, x2 ∈ [-100, 100].
    Global minimum: f(0, 0) = 0 at x = (0, 0).
    """
    x1, x2 = np.fromiter(x.values(), dtype=float)

    return float(
        x1**2 + 2 * x2**2 - 0.3 * np.cos(3 * np.pi * x1) * np.cos(4 * np.pi * x2) + 0.3
    )


def bohachevsky_3(x: dict[str, float]) -> float:
    """Bohachevsky function N. 3 with a bowl-shaped surface.

    Number of inputs: 2 (x1, x2).
    Recommended domain: x1, x2 ∈ [-100, 100].
    Global minimum: f(0, 0) = 0 at x = (0, 0).
    """
    x1, x2 = np.fromiter(x.values(), dtype=float)

    return float(
        x1**2 + 2 * x2**2 - 0.3 * np.cos(3 * np.pi * x1 + 4 * np.pi * x2) + 0.3
    )


def perm_0_d_beta(x: dict[str, float]) -> float:
    """Perm function 0,d,beta with a bowl-shaped surface.

    Number of inputs: Arbitrary (d >= 1).
    Recommended domain: xi ∈ [-d, d].
    Default parameter: beta = 10.
    Global minimum: f(1, 1/2, ..., 1/d) = 0 at
    x_i = 1/i for i = 1, ..., d.
    """
    values = np.fromiter(x.values(), dtype=float)
    dimension = values.size
    beta = 10.0

    indices = np.arange(1, dimension + 1, dtype=float)

    powers = indices[:, np.newaxis]
    values_matrix = values[np.newaxis, :]
    indices_matrix = indices[np.newaxis, :]

    inner = np.sum(
        (indices_matrix + beta) * (values_matrix**powers - indices_matrix ** (-powers)),
        axis=1,
    )

    return float(np.sum(inner**2))


def rotated_hyper_ellipsoid(x: dict[str, float]) -> float:
    """Rotated Hyper-Ellipsoid function.

    The function is continuous, convex, and unimodal. It extends
    the Axis Parallel Hyper-Ellipsoid (Sum Squares) function by
    coupling the variables through cumulative sums.

    Number of inputs: Arbitrary (d >= 1).
    Recommended domain: xi ∈ [-65.536, 65.536].
    Global minimum: f(0, ..., 0) = 0 at x = (0, ..., 0).
    """
    values = np.fromiter(x.values(), dtype=float)

    cumulative_sum = np.cumsum(values)

    return float(np.sum(cumulative_sum**2))


def sphere(x: dict[str, float]) -> float:
    """Sphere function with a simple bowl-shaped surface.

    The function is continuous, convex, and unimodal.

    Number of inputs: Arbitrary (d >= 1).
    Recommended domain: xi ∈ [-5.12, 5.12].
    Global minimum: f(0, ..., 0) = 0 at x = (0, ..., 0).
    """
    values = np.fromiter(x.values(), dtype=float)

    return float(np.sum(values**2))


def sum_of_different_powers(x: dict[str, float]) -> float:
    """Sum of Different Powers function.

    The function is unimodal and has increasing powers for successive
    variables.

    Number of inputs: Arbitrary (d >= 1).
    Recommended domain: xi ∈ [-1, 1].
    Global minimum: f(0, ..., 0) = 0 at x = (0, ..., 0).
    """
    values = np.fromiter(x.values(), dtype=float)
    powers = np.arange(2, values.size + 2, dtype=float)

    return float(np.sum(np.abs(values) ** powers))


def sum_squares(x: dict[str, float]) -> float:
    """Sum Squares function, also known as the Axis Parallel
    Hyper-Ellipsoid function.

    The function is continuous, convex, and unimodal, with no local
    minimum other than the global minimum.

    Number of inputs: Arbitrary (d >= 1).
    Recommended domain: xi ∈ [-10, 10].
    Global minimum: f(0, ..., 0) = 0 at x = (0, ..., 0).
    """
    values = np.fromiter(x.values(), dtype=float)
    indices = np.arange(1, values.size + 1, dtype=float)

    return float(np.sum(indices * values**2))


def trid(x: dict[str, float]) -> float:
    """Trid function with a single global minimum.

    Number of inputs: Arbitrary (d >= 2).
    Recommended domain: xi ∈ [-d², d²].
    Global minimum: f(x*) = -d(d + 4)(d - 1) / 6, where
    x_i = i(d + 1 - i) for i = 1, ..., d.
    """
    values = np.fromiter(x.values(), dtype=float)

    sum_squared = np.sum((values - 1) ** 2)
    adjacent_product = np.sum(values[1:] * values[:-1])

    return float(sum_squared - adjacent_product)
