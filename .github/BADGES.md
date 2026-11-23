# Offline badge generation (Task 3A)

`scripts/generate_badges.py` consumes the existing Task 1 coverage.py format 3
JSON and Task 2 `compatibility.json` / `spark.xml`. It uses only the Python
standard library and the existing `databricks_compat.RUNTIMES`, `TESTS` and
`check_junit.check_report`. It does not run tests, access services or publish.

## Invocation

From the repository root, with Python 3.12 or later:

```powershell
.venv/Scripts/python.exe .github/scripts/generate_badges.py `
  --coverage reports/coverage.json `
  --metadata badge-inputs/metadata.json `
  --databricks 17.3 reports/databricks-17.3/compatibility.json reports/databricks-17.3/spark.xml `
  --databricks 18 reports/databricks-18/compatibility.json reports/databricks-18/spark.xml `
  --output badge-output
```

On Linux use `python` and shell-appropriate continuations. All paths are explicit
and relative to the working directory or absolute. Repeat `--databricks` for each
runtime. Unprovided matrix entries are recorded as unverified. A supplied runtime
must exist in the canonical mapping; duplicates are errors. Adding future LTS ML
entries to the Task 2 mapping requires no generator logic change.

The output directory must **not exist** and must be independent of the input
directories (including metadata). The entire bundle is serialized and staged in
the output parent's filesystem, then renamed into place. Validation failures do
not create output; existing output is never overwritten. Use a fresh destination
for each invocation. This also prevents a failed rerun from silently replacing
previous evidence; selecting and expiring published bundles belongs to Task 3B.

## Metadata input

Supply a UTF-8 JSON object via `--metadata`. Supported fields:

| Field | Contract |
| --- | --- |
| `commit_sha` | Full lowercase 40-character hexadecimal SHA |
| `actions_run_id` | Positive decimal string |
| `actions_run_attempt` | Positive decimal string; prevents mixing reruns |
| `actions_run_url` | `https://github.com/OWNER/REPO/actions/runs/ID`, matching run ID |
| `consolidated_at` | Explicit ISO 8601 timestamp with timezone |
| `coverage_scope` | Object containing `python`, `os`, `suite` |

The exact affirmative coverage scope is
`{"python": "3.12", "os": "ubuntu-24.04", "suite": "not optional"}`.
This is a **caller declaration**: coverage.py does not record the OS, Python
version, pytest selection, commit or Actions execution. The generator cannot
authenticate artifact origins offline. Task 3B must bind the coverage artifact to
the correct job/run/attempt/commit and provide truthful metadata. A missing or
different scope produces `not verified`, preserving the measured value only in
provenance. Do not label local Windows coverage as Ubuntu evidence.

Missing execution metadata is permitted for local inspection, explicitly listed
under `publication.missing_metadata`, and yields exit 1. Databricks reports without
SHA/run/attempt are separately listed under `reports_without_execution_identity`;
caller-supplied values do not retroactively prove their identity. Non-null
SHA/run/attempt values in **all** reports must agree with each other and supplied
metadata. Invalid identifiers, mismatched URLs and conflicting identities are
errors. No timestamp or execution identifier is generated or inferred from the
environment. Use `{}` when no metadata is known.

## Validation and states

Coverage uses `totals.percent_covered` without summing file metrics. It requires
branch coverage, files exclusively under `src/tyrannis/`, numeric finite percentages
in [0, 100], nonnegative integer counters, and consistent aggregate totals.
Optional line/branch percentages must also agree with their totals. Boolean
metrics are rejected. The measured float is preserved; only the message rounds
to the nearest integer (Python's ties-to-even formatting). Colors use the
**unrounded** value: [90,100] brightgreen; [80,90) green; [70,80) yellowgreen;
[60,70) yellow; [0,60) red. These are presentation bands, not passing thresholds:
valid 0% coverage is an approved measurement.

Databricks approval requires schema 1, exact runtime identity and validation
scope, Python 3.12, the canonical expected dependency pins, matching installed
versions, `dependency_check: passed`, the canonical test targets, a validation
timestamp, zero pytest exit status, positive execution, no skips/errors/failures,
consistent JSON/JUnit counts and `result: passed` without an error. The existing
JUnit validator is the approval gate; testcase classnames must also demonstrate
each required Spark test module. A textual success cannot override missing or
contradictory evidence.

These are the Task 2 **equivalent CPU dependencies and local Spark** checks. They
do not certify proprietary Databricks components, GPU libraries or execution in a
Databricks workspace. Only approved LTS ML matrix entries appear in the badge,
sorted numerically; no non-LTS or unvalidated version is added manually.

- `approved`: sufficient positive evidence.
- `failed`: dependency validation, test failures/errors/skips, or a final negative
  validation result. Failed entries are excluded from the approved list.
- `unverified`: absent reports, missing fields, zero executed tests, incomplete
  module evidence, or JUnit validation that cannot establish a valid execution.
- Invalid JSON/XML, invalid metrics, contradictory success, duplicate reports,
  conflicting identities or unsupported contracts cause an explicit error.

Missing paths are recognized as missing evidence, never zero coverage. Parsing
errors are not suppressed. Diagnostic reasons and original compatibility records
are retained in provenance. Overall evidence is failed if any indicator fails,
otherwise unverified if any is unverified, otherwise approved.

## Output and exit status

Exactly three UTF-8 JSON files, stable sorted keys, indentation and trailing newline:

| File | Fields |
| --- | --- |
| `coverage.json` | `schemaVersion: 1`, `label: "coverage"`, `message`, `color` |
| `databricks-lts-ml.json` | `schemaVersion: 1`, `label: "Databricks LTS ML"`, `message`, `color` |
| `provenance.json` | `schema_version: 1`, `metadata`, `observed_identifiers`, `coverage`, `databricks`, `evidence_state`, `publication` |

Coverage messages are percentages or `not verified` (lightgrey). Databricks
messages list approved versions, e.g. `17.3 | 18` (green). If none passed, the
message is `failed` (red) when a runtime failed, otherwise `not verified`
(lightgrey). Partial approval lists only approved versions; provenance exposes
the other outcomes. Shields endpoints contain no provenance fields.

| Exit | Meaning |
| --- | --- |
| 0 | All evidence approved and required execution metadata present/consistent |
| 1 | Bundle generated, but evidence failed/unverified or publication metadata incomplete |
| 2 | Invalid inputs or output/write error; no new bundle committed |

Exit 0 is not authorization to publish. `requires_workflow_verification` is always
true: Task 3B must verify CI success and artifact ownership, decide publication of
negative/partial outcomes, prevent stale results and configure GitHub Pages.
No workflow, permissions, README, production code, pins or tool configuration is
changed by Task 3A.

## Tests

`tests/unit/test_generate_badges.py` exercises the CLI using synthetic reports.
They prove generator contracts, not actual runtime compatibility:

```powershell
.venv/Scripts/python.exe -m pytest tests/unit/test_generate_badges.py -q
ruff check .github/scripts/generate_badges.py tests/unit/test_generate_badges.py
ruff format --check .github/scripts/generate_badges.py tests/unit/test_generate_badges.py
basedpyright --pythonpath .venv/Scripts/python.exe --baselinemode=lock .github/scripts/generate_badges.py tests/unit/test_generate_badges.py
```
