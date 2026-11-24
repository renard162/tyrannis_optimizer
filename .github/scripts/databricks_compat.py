"""Validate equivalent CPU dependencies and local Spark, not Databricks itself.

Usage: databricks_compat.py requirements RUNTIME
       databricks_compat.py test RUNTIME
"""

import os
import platform
import subprocess
import sys
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory

from check_junit import check_report

# Only the relevant CPU packages from the official ML release notes are pinned.
# Keep installation and verification driven by the same version mapping.
RUNTIMES: dict[str, dict[str, str]] = {
    "17.3": {
        "pyspark": "4.0.0",
        "numpy": "2.1.3",
        "scipy": "1.15.1",
        "cloudpickle": "3.0.0",
        "joblib": "1.4.2",
        "pandas": "2.2.3",
        "pyarrow": "19.0.1",
    },
    "18": {
        "pyspark": "4.1.0",
        "numpy": "2.1.3",
        "scipy": "1.15.3",
        "cloudpickle": "3.0.0",
        "joblib": "1.4.2",
        "pandas": "2.2.3",
        "pyarrow": "21.0.0",
    },
}
TESTS = (
    "tests/unit/backend/test_spark_parallel.py",
    "tests/unit/backend/test_spark_distributed.py",
    "tests/unit/backend/test_spark_communication.py",
    "tests/integration/backend/test_spark_parallel.py",
    "tests/integration/backend/test_spark_distributed.py",
)


def installed_versions(expected: dict[str, str]) -> dict[str, str | None]:
    """Collect actual metadata even when an installation is incomplete."""
    installed: dict[str, str | None] = {}
    for package in expected:
        try:
            installed[package] = version(package)
        except PackageNotFoundError:
            installed[package] = None
    return installed


def validate_environment(
    expected: dict[str, str], installed: dict[str, str | None]
) -> None:
    """Reject wrong versions, external imports and inconsistent dependencies."""
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Expected Python 3.12, got {platform.python_version()}")
    prefix = Path(sys.prefix).resolve()
    if (
        sys.prefix == sys.base_prefix
        or prefix != Path(os.environ["DATABRICKS_ENV"]).resolve()
    ):
        raise RuntimeError("Must use the dedicated Databricks virtual environment")
    for variable in ("PYSPARK_PYTHON", "PYSPARK_DRIVER_PYTHON"):
        # Do not resolve the interpreter symlink: venv identity is significant.
        if Path(os.environ[variable]).absolute() != Path(sys.executable).absolute():
            raise RuntimeError(f"{variable} does not select the test interpreter")
    if installed != expected:
        raise RuntimeError(
            f"Dependency versions differ: {installed}; expected {expected}"
        )
    for package in (*expected, "tyrannis", "pytest", "pytest_timeout"):
        module = import_module(package)
        if module.__file__ is None or not Path(
            module.__file__
        ).resolve().is_relative_to(prefix):
            raise RuntimeError(f"{package} was imported from outside {prefix}")
    # uv pip does not synchronize the project or consult its conventional lock.
    _ = subprocess.run(["uv", "pip", "check", "--python", sys.executable], check=True)


def run_tests(runtime: str) -> None:
    """Validate the isolated environment and require successful, unskipped tests."""
    expected = RUNTIMES[runtime]
    validate_environment(expected, installed_versions(expected))
    with TemporaryDirectory(prefix="databricks-") as directory:
        junit = Path(directory) / "spark.xml"
        _ = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                *TESTS,
                f"--junitxml={junit}",
                "-q",
                "-rs",
            ],
            check=True,
        )
        count = check_report(junit)
        print(f"Verified {count} executed tests with no skips, failures, or errors")


def main() -> None:
    """Emit exact requirements or run validation with the isolated interpreter."""
    if len(sys.argv) < 3 or sys.argv[2] not in RUNTIMES:
        raise SystemExit("Supply requirements/test and a known runtime: 17.3 or 18")
    command, runtime = sys.argv[1:3]
    if command == "requirements" and len(sys.argv) == 3:
        for package, required in RUNTIMES[runtime].items():
            print(f"{package}=={required}")
    elif command == "test" and len(sys.argv) == 3:
        run_tests(runtime)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
