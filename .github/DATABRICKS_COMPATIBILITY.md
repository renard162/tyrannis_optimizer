# Databricks LTS ML compatibility evidence

Official Tyrannis support is limited to Databricks **LTS ML** versions that pass
the dedicated CI matrix. Non-LTS versions may work but are neither systematically
tested nor advertised as officially supported. A matrix entry is a validation
target, not a declaration that the runtime has passed.

This job checks equivalent open-source CPU dependencies and real local Spark
execution. It does **not** certify execution in a Databricks workspace or reproduce
GPU libraries, proprietary components, or the complete Databricks distribution.

## Version sources

Versions were checked against the official CPU ML release notes on 2026-11-23:

| Dependency | 17.3 LTS ML | 18 LTS ML |
| --- | --- | --- |
| Python | 3.12 | 3.12 |
| PySpark | 4.0.0 | 4.1.0 |
| NumPy | 2.1.3 | 2.1.3 |
| SciPy | 1.15.1 | 1.15.3 |
| cloudpickle | 3.0.0 | 3.0.0 |
| joblib | 1.4.2 | 1.4.2 |
| pandas | 2.2.3 | 2.2.3 |
| PyArrow | 19.0.1 | 21.0.0 |

- [17.3 LTS ML CPU libraries](https://docs.databricks.com/aws/en/release-notes/runtime/17.3lts-ml)
- [17.3 LTS base runtime](https://docs.databricks.com/aws/en/release-notes/runtime/17.3lts)
- [18 LTS ML CPU libraries](https://docs.databricks.com/aws/en/release-notes/runtime/18ml)
- [18 LTS base runtime](https://docs.databricks.com/aws/en/release-notes/runtime/18)

The PyPI PySpark versions match the Apache Spark versions listed for the base
runtimes; they do not contain Databricks patches. Python is constrained to the
3.12 minor series, and evidence records the actual patch version. Java 17 is used
on Ubuntu 24.04. The runtime 18 notes now list Java 21 as the default and explicitly
allow falling back to Java 17; this job does not claim to test Java 21.

## Installation and execution

`scripts/databricks_compat.py` owns the package version mappings. Its `requirements`
command emits exact pins used by a **single** `uv pip install` resolution together
with `.[spark]`, pytest 9.1.1 and pytest-timeout 2.4.0. Test tool pins match the
conventional lock at implementation time but are not Databricks runtime pins.
Each matrix leg creates a separate environment under `runner.temp`. No `uv sync`,
`uv run`, lockfile modification, dependency override or editable installation is
used. Transitive dependencies are resolved subject to the package requirements.

The `test` command runs through that environment's Python. It verifies Python,
every pinned version, the virtual environment, driver/worker interpreter settings,
import provenance (including Tyrannis and pytest), and `uv pip check` before
starting pytest with the same interpreter. It uses the three existing Spark unit
test files and both existing integration files. The communication unit tests cover
TCP transport contracts without requiring a Spark cluster. The integration tests
use actual `local[2]` Spark sessions and existing shared probes; their generic skip
behavior remains unchanged.

Both pytest exit status and `check_junit.check_report` must succeed. Missing or
invalid XML, no tests, collection errors, failures and any skips fail the leg.
Spark startup/worker skips therefore cannot produce successful evidence. An
existing report is removed before each invocation so a rerun cannot reuse it.

The CI Gate requires the aggregate `databricks-lts-ml-tests` result along with the
six existing jobs. With `fail-fast: false`, both legs can finish independently;
any failed, cancelled, skipped or missing required job result prevents the gate
from passing. The existing coverage and conventional Spark jobs are unchanged.

## Evidence schema (version 1)

Each artifact `databricks-<runtime>-lts-ml` contains the available files from
`reports/databricks-<runtime>/`:

- `compatibility.json`: final validation outcome, including failure diagnostics.
- `spark.xml`: original pytest JUnit report, when pytest produced one.

JSON fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | Integer `1`. |
| `runtime` | `17.3 LTS ML` or `18 LTS ML`. |
| `validation_scope` | `equivalent-cpu-dependencies-local-spark`. |
| `expected_python`, `python_version` | Required minor series and actual full version. |
| `expected_dependencies` | Package name to exact required version. |
| `installed_dependencies` | Package name to actual version; `null` if absent. |
| `commit_sha` | `GITHUB_SHA`: the checked-out commit, including PR merge commits. |
| `actions_run_id`, `actions_run_attempt` | GitHub execution identifiers, strings. |
| `validated_at` | UTC ISO 8601 timestamp when the final record was written. |
| `result` | `passed` only after all checks; otherwise `failed`. |
| `dependency_check` | `passed` or `not_passed`. |
| `pytest_exit_code` | Actual subprocess status, or `null` if not run. |
| `tests_executed`, `tests_skipped` | JUnit case counts when readable; otherwise `null`. Executed excludes skips. |
| `test_targets` | Exact existing test files selected. |
| `tyrannis_version` | Installed distribution version, after environment validation. |
| `error` | Diagnostic on failure; absent on success. |

Local executions have `null` Actions identifiers. The commit can be supplied as
`GITHUB_SHA` for local provenance, without inventing an Actions execution ID.
Failed/error cases count as executed; collection errors do not imply that a test
body ran. Successful evidence requires a nonempty, internally consistent report
without errors or skips.

The JSON is written at the end, including on ordinary validation/test failures,
and exceptions retain a nonzero process exit. Installation failure or abrupt
termination can prevent JSON/JUnit creation. The artifact step always attempts
upload of available evidence; absence is never a positive result and the original
failure remains. Consumers must require successful CI execution as well as
`result: passed`, and match the commit/run/attempt before declaring support.
No public publishing, badge generation or write permission is added here.

## Adding another LTS ML target

Confirm the official CPU and base runtime notes, add its exact package mapping to
`RUNTIMES` in the helper, add its key to `strategy.matrix.runtime` in `ci.yml`, and
update this source table. Run the isolated installation and real Spark validation.
Do not change Tyrannis requirements or substitute pins to force a passing result.
The gate's job count stays seven when adding legs to the existing matrix job.
