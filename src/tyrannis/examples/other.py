import numpy as np


def beale(x: dict[str, float]) -> float:
    """Beale function with sharp peaks near the domain boundaries.

    www.sfu.ca/~ssurjano/optimization.html

    Number of inputs: 2 (x1, x2).
    Recommended domain: x1, x2 ∈ [-4.5, 4.5].
    Global minimum: f(3, 0.5) = 0 at x = (3, 0.5).
    """
    x1, x2 = np.fromiter(x.values(), dtype=float)

    return float(
        (1.5 - x1 + x1 * x2) ** 2
        + (2.25 - x1 + x1 * x2**2) ** 2
        + (2.625 - x1 + x1 * x2**3) ** 2
    )


def branin(x: dict[str, float]) -> float:
    """Branin-Hoo function with three global minima.

    www.sfu.ca/~ssurjano/optimization.html

    Number of inputs: 2 (x1, x2).
    Recommended domain: x1 ∈ [-5, 10], x2 ∈ [0, 15].
    Global minima: f ≈ 0.397887 at
    (-π, 12.275), (π, 2.275), and (9.42478, 2.475).
    """
    x1, x2 = np.fromiter(x.values(), dtype=float)

    a = 1.0
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6.0
    s = 10.0
    t = 1 / (8 * np.pi)

    return float(a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s)


def colville(x: dict[str, float]) -> float:
    """Colville function with strong variable interactions.

    www.sfu.ca/~ssurjano/optimization.html

    Number of inputs: 4 (x1, x2, x3, x4).
    Recommended domain: xi ∈ [-10, 10].
    Global minimum: f(1, 1, 1, 1) = 0 at x = (1, 1, 1, 1).
    """
    x1, x2, x3, x4 = np.fromiter(x.values(), dtype=float)

    return float(
        100 * (x1**2 - x2) ** 2
        + (x1 - 1) ** 2
        + (x3 - 1) ** 2
        + 90 * (x3**2 - x4) ** 2
        + 10.1 * ((x2 - 1) ** 2 + (x4 - 1) ** 2)
        + 19.8 * (x2 - 1) * (x4 - 1)
    )


def forrester(x: dict[str, float]) -> float:
    """Forrester et al. (2008) function with one global minimum,
    one local minimum, and a zero-gradient inflection point.

    www.sfu.ca/~ssurjano/optimization.html

    Number of inputs: 1 (x).
    Recommended domain: x ∈ [0, 1].
    Global minimum: f(0.757249) ≈ -6.020740.
    """
    value = next(iter(x.values()))

    return float((6 * value - 2) ** 2 * np.sin(12 * value - 4))


def goldstein_price(x: dict[str, float]) -> float:
    """Goldstein-Price function with several local minima.

    www.sfu.ca/~ssurjano/optimization.html

    Number of inputs: 2 (x1, x2).
    Recommended domain: x1, x2 ∈ [-2, 2].
    Global minimum: f(0, -1) = 3 at x = (0, -1).
    """
    x1, x2 = np.fromiter(x.values(), dtype=float)

    factor_1 = 1 + (x1 + x2 + 1) ** 2 * (
        19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2
    )

    factor_2 = 30 + (2 * x1 - 3 * x2) ** 2 * (
        18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2
    )

    return float(factor_1 * factor_2)


def hartmann_3(x: dict[str, float]) -> float:
    """Hartmann 3-dimensional function with four local minima.

    www.sfu.ca/~ssurjano/optimization.html

    Number of inputs: 3 (x1, x2, x3).
    Recommended domain: xi ∈ (0, 1).
    Global minimum: f ≈ -3.86278 at approximately
    x = (0.114614, 0.555649, 0.852547).
    """
    values = np.fromiter(x.values(), dtype=float)

    alpha = np.array([1.0, 1.2, 3.0, 3.2])

    a = np.array(
        [
            [3.0, 10.0, 30.0],
            [0.1, 10.0, 35.0],
            [3.0, 10.0, 30.0],
            [0.1, 10.0, 35.0],
        ]
    )

    p = 1e-4 * np.array(
        [
            [3689, 1170, 2673],
            [4699, 4387, 7470],
            [1091, 8732, 5547],
            [381, 5743, 8828],
        ]
    )

    distance = np.sum(
        a * (values - p) ** 2,
        axis=1,
    )

    return float(-np.sum(alpha * np.exp(-distance)))


def hartmann_4(x: dict[str, float]) -> float:
    """Hartmann 4-dimensional function.

    www.sfu.ca/~ssurjano/optimization.html

    Number of inputs: 4 (x1, x2, x3, x4).
    Recommended domain: xi ∈ [0, 1].
    Global minimum: f ≈ -3.13547 at approximately
    x = (0.1873, 0.1909, 0.5563, 0.2645).
    """
    values = np.fromiter(x.values(), dtype=float)

    alpha = np.array([1.0, 1.2, 3.0, 3.2])

    a = np.array(
        [
            [10.0, 3.0, 17.0, 3.5],
            [0.05, 10.0, 17.0, 0.1],
            [3.0, 3.5, 1.7, 10.0],
            [17.0, 8.0, 0.05, 10.0],
        ]
    )

    p = 1e-4 * np.array(
        [
            [1312, 1696, 5569, 124],
            [2329, 4135, 8307, 3736],
            [2348, 1451, 3522, 2883],
            [4047, 8828, 8732, 5743],
        ]
    )

    distance = np.sum(
        a * (values - p) ** 2,
        axis=1,
    )

    outer = np.sum(alpha * np.exp(-distance))

    return float((1.1 - outer) / 0.839)


def hartmann_6(x: dict[str, float]) -> float:
    """Hartmann 6-dimensional function with six local minima.

    www.sfu.ca/~ssurjano/optimization.html

    Number of inputs: 6 (x1, x2, x3, x4, x5, x6).
    Recommended domain: xi ∈ (0, 1).
    Global minimum: f ≈ -3.04246 at approximately
    x = (0.20169, 0.15001, 0.47687, 0.27533, 0.31165, 0.65730).
    """
    values = np.fromiter(x.values(), dtype=float)

    alpha = np.array([1.0, 1.2, 3.0, 3.2])

    a = np.array(
        [
            [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
            [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
            [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
            [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
        ]
    )

    p = 1e-4 * np.array(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ]
    )

    distance = np.sum(
        a * (values - p) ** 2,
        axis=1,
    )

    outer = np.sum(alpha * np.exp(-distance))

    return float(-(2.58 + outer) / 1.94)


def perm_d_beta(x: dict[str, float]) -> float:
    """Perm function d, beta.

    www.sfu.ca/~ssurjano/optimization.html

    Number of inputs: Arbitrary (d >= 1).
    Recommended domain: xi ∈ [-d, d].
    Default parameter: beta = 0.5.
    Global minimum: f(1, ..., 1) = 0 at x = (1, ..., 1).
    """
    values = np.fromiter(x.values(), dtype=float)
    dimension = values.size
    beta = 0.5

    indices = np.arange(1, dimension + 1, dtype=float)

    powers = indices[:, np.newaxis]
    indices_matrix = indices[np.newaxis, :]
    values_matrix = values[np.newaxis, :]

    inner = np.sum(
        (indices_matrix**powers + beta)
        * ((values_matrix / indices_matrix) ** powers - 1),
        axis=1,
    )

    return float(np.sum(inner**2))


def powell(x: dict[str, float]) -> float:
    """Powell function with groups of four interacting variables.

    www.sfu.ca/~ssurjano/optimization.html

    Number of inputs: Arbitrary multiples of 4 (d = 4k).
    Recommended domain: xi ∈ [-4, 5].
    Global minimum: f(0, ..., 0) = 0 at x = (0, ..., 0).
    """
    values = np.fromiter(x.values(), dtype=float)

    if values.size % 4 != 0:
        raise ValueError(
            "The Powell function requires a number of inputs that is a multiple of 4."
        )

    groups = values.reshape(-1, 4)

    return float(
        np.sum(
            (groups[:, 0] + 10 * groups[:, 1]) ** 2
            + 5 * (groups[:, 2] - groups[:, 3]) ** 2
            + (groups[:, 1] - 2 * groups[:, 2]) ** 4
            + 10 * (groups[:, 0] - groups[:, 3]) ** 4
        )
    )


def shekel(x: dict[str, float]) -> float:
    """Shekel function with multiple local minima.

    www.sfu.ca/~ssurjano/optimization.html

    Number of inputs: 4 (x1, x2, x3, x4).
    Recommended domain: xi ∈ [0, 10].
    Default parameters: m = 10.
    Global minimum: f(4, 4, 4, 4) ≈ -10.5364.
    """
    values = np.fromiter(x.values(), dtype=float)

    b = 0.1 * np.array(
        [1, 2, 2, 4, 4, 6, 3, 7, 5, 5],
        dtype=float,
    )

    c = np.array(
        [
            [4.0, 1.0, 8.0, 6.0],
            [4.0, 1.0, 8.0, 6.0],
            [4.0, 1.0, 8.0, 6.0],
            [4.0, 1.0, 8.0, 6.0],
            [3.0, 7.0, 3.0, 1.0],
            [2.0, 9.0, 2.0, 3.0],
            [5.0, 3.0, 5.0, 8.0],
            [8.0, 1.0, 1.0, 9.0],
            [6.0, 2.0, 6.0, 3.0],
            [7.0, 3.6, 7.0, 2.0],
        ]
    )

    distance = np.sum(
        (values - c) ** 2,
        axis=1,
    )

    return float(-np.sum(1 / (distance + b)))


def styblinski_tang(x: dict[str, float]) -> float:
    """Styblinski-Tang function with multiple local minima.

    www.sfu.ca/~ssurjano/optimization.html

    Number of inputs: Arbitrary (d >= 1).
    Recommended domain: xi ∈ [-5, 5].
    Global minimum: f(x*) ≈ -39.16617 * d, where
    x_i ≈ -2.903534 for every i.
    """
    values = np.fromiter(x.values(), dtype=float)

    return float(0.5 * np.sum(values**4 - 16 * values**2 + 5 * values))
