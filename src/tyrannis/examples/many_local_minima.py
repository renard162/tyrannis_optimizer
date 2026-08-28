import numpy as np


def ackley(x: dict[str, float]) -> float:
    """Ackley function with many local minima.

    www.sfu.ca/~ssurjano/optimization.html

    Number of inputs: Arbitrary (n >= 1).
    Recommended domain: xi ∈ [-32.768, 32.768].
    Global minimum: f(0, ..., 0) = 0 at x = (0, ..., 0).
    """
    values = np.fromiter(x.values(), dtype=float)
    dimension = values.size

    sum_squared = np.sum(values**2)
    sum_cosine = np.sum(np.cos(2 * np.pi * values))

    return float(
        -20 * np.exp(-0.2 * np.sqrt(sum_squared / dimension))
        - np.exp(sum_cosine / dimension)
        + 20
        + np.e
    )


def bukin_6(x: dict[str, float]) -> float:
    """Bukin function N. 6 with a narrow, curved ridge.

    www.sfu.ca/~ssurjano/optimization.html

    Number of inputs: 2 (x1, x2).
    Recommended domain: x1 ∈ [-15, -5], x2 ∈ [-3, 3].
    Global minimum: f(-10, 1) = 0 at x = (-10, 1).
    """
    x1, x2 = np.fromiter(x.values(), dtype=float)

    return float(100 * np.sqrt(np.abs(x2 - 0.01 * x1**2)) + 0.01 * np.abs(x1 + 10))


def cross_in_tray(x: dict[str, float]) -> float:
    """Cross-in-Tray function with four global minima.

    www.sfu.ca/~ssurjano/optimization.html

    Number of inputs: 2 (x1, x2).
    Recommended domain: x1, x2 ∈ [-10, 10].
    Global minima: f ≈ -2.06261 at
    (±1.34941, ±1.34941).
    """
    x1, x2 = np.fromiter(x.values(), dtype=float)

    radius = np.sqrt(x1**2 + x2**2)

    value = np.sin(x1) * np.sin(x2) * np.exp(np.abs(100 - radius / np.pi))

    return float(-0.0001 * (np.abs(value) + 1) ** 0.1)


def drop_wave(x: dict[str, float]) -> float:
    """Drop-Wave function with many local minima.

    www.sfu.ca/~ssurjano/optimization.html

    Number of inputs: 2 (x1, x2).
    Recommended domain: x1, x2 ∈ [-5.12, 5.12].
    Global minimum: f(0, 0) = -1 at x = (0, 0).
    """
    x1, x2 = np.fromiter(x.values(), dtype=float)

    radius_squared = x1**2 + x2**2

    return float(
        -(1 + np.cos(12 * np.sqrt(radius_squared))) / (0.5 * radius_squared + 2)
    )


def eggholder(x: dict[str, float]) -> float:
    """Eggholder function with many local minima.

    www.sfu.ca/~ssurjano/optimization.html

    Number of inputs: 2 (x1, x2).
    Recommended domain: x1, x2 ∈ [-512, 512].
    Global minimum: f(512, 404.2319) ≈ -959.6407.
    """
    x1, x2 = np.fromiter(x.values(), dtype=float)

    return float(
        -(x2 + 47) * np.sin(np.sqrt(np.abs(x2 + x1 / 2 + 47)))
        - x1 * np.sin(np.sqrt(np.abs(x1 - x2 - 47)))
    )


def gramacy_lee(x: dict[str, float]) -> float:
    """Gramacy & Lee (2012) function with multiple local minima.

    www.sfu.ca/~ssurjano/optimization.html

    Number of inputs: 1 (x).
    Recommended domain: x ∈ [0.5, 2.5].
    Global minimum: f(0.548563) ≈ -0.869011.
    """
    value = next(iter(x.values()))

    return float(np.sin(10 * np.pi * value) / (2 * value) + (value - 1) ** 4)


def griewank(x: dict[str, float]) -> float:
    """Griewank function with many widespread local minima.

    www.sfu.ca/~ssurjano/optimization.html

    Number of inputs: Arbitrary (n >= 1).
    Recommended domain: xi ∈ [-600, 600].
    Global minimum: f(0, ..., 0) = 0 at x = (0, ..., 0).
    """
    values = np.fromiter(x.values(), dtype=float)
    indices = np.arange(1, values.size + 1)

    sum_term = np.sum(values**2) / 4000
    product_term = np.prod(np.cos(values / np.sqrt(indices)))

    return float(sum_term - product_term + 1)


def holder_table(x: dict[str, float]) -> float:
    """Holder Table function with four global minima.

    www.sfu.ca/~ssurjano/optimization.html

    Number of inputs: 2 (x1, x2).
    Recommended domain: x1, x2 ∈ [-10, 10].
    Global minima: f ≈ -19.2085 at
    (±8.05502, ±9.66459).
    """
    x1, x2 = np.fromiter(x.values(), dtype=float)

    value = np.sin(x1) * np.cos(x2) * np.exp(np.abs(1 - np.sqrt(x1**2 + x2**2) / np.pi))

    return float(-np.abs(value))


def langermann(x: dict[str, float]) -> float:
    """Langermann function with several unevenly distributed local minima.

    www.sfu.ca/~ssurjano/optimization.html

    Number of inputs: 2 (x1, x2).
    Recommended domain: x1, x2 ∈ [0, 10].
    Uses the standard parameters m=5, c=(1, 2, 5, 2, 3).
    Global minimum: approximately
    f(2.002992, 1.006096) ≈ -1.306.
    """
    x1, x2 = np.fromiter(x.values(), dtype=float)

    a = np.array([3, 5, 2, 1, 7], dtype=float)
    b = np.array([5, 2, 1, 4, 9], dtype=float)
    c = np.array([1, 2, 5, 2, 3], dtype=float)

    distance = (x1 - a) ** 2 + (x2 - b) ** 2

    return float(-np.sum(c * np.exp(-distance / np.pi) * np.cos(np.pi * distance)))


def levy(x: dict[str, float]) -> float:
    """Levy function with many local minima.

    www.sfu.ca/~ssurjano/optimization.html

    Number of inputs: Arbitrary (n >= 1).
    Recommended domain: xi ∈ [-10, 10].
    Global minimum: f(1, ..., 1) = 0 at x = (1, ..., 1).
    """
    values = np.fromiter(x.values(), dtype=float)
    w = 1 + (values - 1) / 4

    term_1 = np.sin(np.pi * w[0]) ** 2

    term_2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))

    term_3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)

    return float(term_1 + term_2 + term_3)


def levy_13(x: dict[str, float]) -> float:
    """Levy function N. 13 with multiple local minima.

    www.sfu.ca/~ssurjano/optimization.html

    Number of inputs: 2 (x1, x2).
    Recommended domain: x1, x2 ∈ [-10, 10].
    Global minimum: f(1, 1) = 0 at x = (1, 1).
    """
    x1, x2 = np.fromiter(x.values(), dtype=float)

    return float(
        np.sin(3 * np.pi * x1) ** 2
        + (x1 - 1) ** 2 * (1 + np.sin(3 * np.pi * x2) ** 2)
        + (x2 - 1) ** 2 * (1 + np.sin(2 * np.pi * x2) ** 2)
    )


def rastrigin(x: dict[str, float]) -> float:
    """Rastrigin function with regularly distributed local minima.

    www.sfu.ca/~ssurjano/optimization.html

    Number of inputs: Arbitrary (n >= 1).
    Recommended domain: xi ∈ [-5.12, 5.12].
    Global minimum: f(0, ..., 0) = 0 at x = (0, ..., 0).
    """
    values = np.fromiter(x.values(), dtype=float)
    dimension = values.size

    return float(10 * dimension + np.sum(values**2 - 10 * np.cos(2 * np.pi * values)))


def schaffer_2(x: dict[str, float]) -> float:
    """Schaffer function N. 2 with multiple local minima.

    www.sfu.ca/~ssurjano/optimization.html

    Number of inputs: 2 (x1, x2).
    Recommended domain: x1, x2 ∈ [-100, 100].
    Global minimum: f(0, 0) = 0 at x = (0, 0).
    """
    x1, x2 = np.fromiter(x.values(), dtype=float)

    difference = x1**2 - x2**2
    radius_squared = x1**2 + x2**2

    return float(
        0.5 + (np.sin(difference) ** 2 - 0.5) / (1 + 0.001 * radius_squared) ** 2
    )


def schaffer_4(x: dict[str, float]) -> float:
    """Schaffer function N. 4 with multiple local minima.

    www.sfu.ca/~ssurjano/optimization.html

    Number of inputs: 2 (x1, x2).
    Recommended domain: x1, x2 ∈ [-100, 100].
    Global minima: f ≈ 0.292579 at
    (0, ±1.253115) and (±1.253115, 0).
    """
    x1, x2 = np.fromiter(x.values(), dtype=float)

    difference = np.abs(x1**2 - x2**2)
    radius_squared = x1**2 + x2**2

    return float(
        0.5
        + (np.cos(np.sin(difference)) ** 2 - 0.5) / (1 + 0.001 * radius_squared) ** 2
    )


def schwefel(x: dict[str, float]) -> float:
    """Schwefel function with many local minima.

    www.sfu.ca/~ssurjano/optimization.html

    Number of inputs: Arbitrary (n >= 1).
    Recommended domain: xi ∈ [-500, 500].
    Global minimum: f(420.968746, ..., 420.968746) ≈ 0.
    """
    values = np.fromiter(x.values(), dtype=float)
    dimension = values.size

    return float(
        418.9829 * dimension - np.sum(values * np.sin(np.sqrt(np.abs(values))))
    )


def shubert(x: dict[str, float]) -> float:
    """Shubert function with multiple local minima.

    www.sfu.ca/~ssurjano/optimization.html

    Number of inputs: 2 (x1, x2).
    Recommended domain: x1, x2 ∈ [-10, 10].
    Global minima: 18 global minima with f ≈ -186.7309.
    """
    x1, x2 = np.fromiter(x.values(), dtype=float)

    indices = np.arange(1, 6)

    sum_x1 = np.sum(indices * np.cos((indices + 1) * x1 + indices))

    sum_x2 = np.sum(indices * np.cos((indices + 1) * x2 + indices))

    return float(sum_x1 * sum_x2)
