"""Exercise the installed base distribution from outside the source checkout."""

import sys
from importlib.metadata import distribution
from math import isfinite
from pathlib import Path

import tyrannis
from tyrannis import Optimizer
from tyrannis.algorithm import PSO
from tyrannis.space import Continuous


def sphere(x: float) -> float:
    """Provide a small objective using the documented keyword-based API."""
    return x * x


def main() -> None:
    """Check installation provenance and run a minimal public-API optimization."""
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Package smoke test must run in a clean virtual environment")
    if tyrannis.__file__ is None:
        raise RuntimeError("Imported tyrannis has no module file")

    installed = distribution("tyrannis")
    module_path = Path(tyrannis.__file__).resolve()
    expected_path = Path(installed.locate_file("tyrannis/__init__.py")).resolve()
    if module_path != expected_path or not module_path.is_relative_to(
        Path(sys.prefix).resolve()
    ):
        raise RuntimeError(
            f"Imported tyrannis is not the installed package: {module_path}"
        )
    files = installed.files
    if files is None or "tyrannis/__init__.py" not in {str(file) for file in files}:
        raise RuntimeError("Distribution metadata does not own the imported module")

    optimizer = Optimizer(
        space=Continuous(boundaries={"x": (-1.0, 1.0)}, cost_function=sphere),
        algorithm=PSO(),
        n_particles=4,
        n_iterations=2,
        seed=42,
    ).fit()
    fitness = optimizer.best_fitness
    if not isfinite(fitness) or not 0.0 <= fitness <= 1.0:
        raise RuntimeError(f"Optimization produced invalid fitness: {fitness}")
    print(f"tyrannis {installed.version}: {module_path}; best fitness={fitness}")


if __name__ == "__main__":
    main()
