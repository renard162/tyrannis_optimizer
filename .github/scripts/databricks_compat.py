"""Validate equivalent CPU dependencies and local Spark, not Databricks itself.

Usage: databricks_compat.py requirements RUNTIME
       databricks_compat.py test RUNTIME REPORT_DIRECTORY
See ../DATABRICKS_COMPATIBILITY.md for the evidence schema and support policy.
"""

import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import import_module
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from xml.etree import ElementTree

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


def run_tests(runtime: str, directory: Path) -> None:
    """Write final evidence on success or failure, preserving a failing exit."""
    directory.mkdir(parents=True, exist_ok=True)
    junit = directory / "spark.xml"
    output = directory / "compatibility.json"
    # A rerun must never consume evidence left by a previous invocation.
    junit.unlink(missing_ok=True)
    output.unlink(missing_ok=True)
    expected = RUNTIMES[runtime]
    installed = installed_versions(expected)
    report: dict[str, object] = {
        "schema_version": 1,
        "runtime": f"{runtime} LTS ML",
        "validation_scope": "equivalent-cpu-dependencies-local-spark",
        "python_version": platform.python_version(),
        "expected_python": "3.12",
        "expected_dependencies": expected,
        "installed_dependencies": installed,
        "commit_sha": os.environ.get("GITHUB_SHA"),
        "actions_run_id": os.environ.get("GITHUB_RUN_ID"),
        "actions_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "result": "failed",
        "dependency_check": "not_passed",
        "pytest_exit_code": None,
        "tests_executed": None,
        "tests_skipped": None,
        "test_targets": TESTS,
    }
    try:
        validate_environment(expected, installed)
        report["dependency_check"] = "passed"
        report["tyrannis_version"] = distribution("tyrannis").version
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                *TESTS,
                f"--junitxml={junit}",
                "-q",
                "-rs",
            ],
            check=False,
        )
        report["pytest_exit_code"] = result.returncode
        # Preserve counts on test failures/skips when pytest produced readable XML.
        root = ElementTree.parse(junit).getroot()
        cases = list(root.iter("testcase"))
        skipped = sum(case.find("skipped") is not None for case in cases)
        report["tests_executed"] = len(cases) - skipped
        report["tests_skipped"] = skipped
        count = check_report(junit)
        result.check_returncode()
        report["tests_executed"] = count
        report["result"] = "passed"
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        report["validated_at"] = datetime.now(UTC).isoformat()
        _ = output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Emit exact requirements or run validation with the isolated interpreter."""
    if len(sys.argv) < 3 or sys.argv[2] not in RUNTIMES:
        raise SystemExit("Supply requirements/test and a known runtime: 17.3 or 18")
    command, runtime = sys.argv[1:3]
    if command == "requirements" and len(sys.argv) == 3:
        for package, required in RUNTIMES[runtime].items():
            print(f"{package}=={required}")
    elif command == "test" and len(sys.argv) == 4:
        run_tests(runtime, Path(sys.argv[3]))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
