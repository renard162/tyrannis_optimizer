# Continuous Integration, Documentation, and Continuous Deployment

Tyrannis uses three GitHub Actions workflows to validate changes, publish the package, and deploy the documentation website. This guide explains how to prepare changes, satisfy the required checks, publish a release, and recover from failures.

## 1. Workflow overview

| Workflow | File | Responsibility |
| --- | --- | --- |
| `CI` | [ci.yml](../.github/workflows/ci.yml) | Code quality, compatibility tests, and package validation |
| `Controlled package release` | [release.yml](../.github/workflows/release.yml) | Package publication, verification, and recovery |
| `Documentation` | [pages.yml](../.github/workflows/pages.yml) | GitHub Pages deployment |

The release workflow (CD) calls the full CI workflow and uses the distributions it approves. **Documentation is a separate workflow:** it neither requires CI or CD approval nor contributes to their acceptance criteria.

PyPI and TestPyPI publication are independent. Neither requires publication on the other index. However, installation verification for both destinations obtains dependencies and the sdist build backend from PyPI.

```text
CI (independent)
  push / pull request / manual dispatch
    -> seven required job groups -> CI Gate

Controlled package release
  published GitHub Release / manual dispatch
    -> authorize -> ci (reusable CI, including CI Gate)
                     |
                     +-> package-validate [validation only]
                     |
                     +-> bind-publication [publication requested]
                           -> publish-pypi OR publish-testpypi
                           -> verify-pypi OR verify-testpypi

Documentation (independent)
  filtered master push / manual dispatch
    -> build -> deploy (artifact availability, freshness, Pages)
```

Both publishing jobs require successful CI and publication binding. Each verifier also requires its publisher to succeed, although an error from the upload action itself can be reconciled during verification.

**In this guide:**

- [Prepare changes and pass CI](#2-continuous-integration)
- [Run a validation or publish a package](#3-continuous-deployment)
- [Understand publication controls and artifact identity](#4-publication-controls)
- [Understand post-publication checks and retries](#5-post-publication-verification)
- [Diagnose failures and recover safely](#6-publication-recovery)
- [Publish the documentation website](#7-documentation-github-pages)
- [Understand guarantees and limitations](#8-guarantees-and-limitations)

## 2. Continuous Integration

### 2.1. When CI runs

| Event | Condition |
| --- | --- |
| Push | `master`, subject to the path exclusions below |
| Pull request | Targets `master`; opened, synchronized, or reopened; subject to the same exclusions |
| Manual dispatch | Started from GitHub Actions |
| `workflow_call` | Called by another workflow, including CD |

Push and pull request events ignore changes confined to these paths:

```text
README.md
doc/**
.github/pages/**
.github/workflows/pages.yml
```

A change outside these paths remains eligible for CI, including a mixed code and documentation change. Manual dispatch and reusable workflow calls are not subject to these path filters.

`README.md` is excluded from automatic CI triggering but remains the package description and an sdist input. Package Build validates it whenever CI runs. A tag push or release publication does not independently trigger CI; an authorized release calls it through CD.

New pull request runs cancel obsolete runs for the same PR. Other CI runs use a run-specific concurrency group and are not automatically cancelled. If a repository ruleset requires `CI Gate` on every PR, check its compatibility with documentation-only PRs for which CI is not triggered.

### 2.2. Developer checklist

Before submitting changes, prepare a local environment using the project's locked dependencies and run the checks relevant to the change. The following commands reproduce the core quality checks from the repository root:

```shell
uv lock --check
uv sync --locked --all-extras --dev
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync basedpyright
```

CI invokes basedpyright with `--pythonpath .venv/bin/python` on Ubuntu. For local execution, ensure the type checker resolves the project's virtual environment; use the appropriate interpreter path on your platform if needed. The current workflow uses the project configuration and does not pass `--baselinemode=lock`.

To reproduce the core test selection in a separate, locked development environment configuration:

```shell
uv sync --locked --dev
uv run --no-sync pytest -m "not optional"
```

- Keep source, tests, dependency declarations, and the lockfile consistent.
- Run the relevant optional tests when the required runtime is available. Spark requires Java; MPI requires an appropriate MPI installation and thread support.
- Review package inputs, including README, license and metadata, when changing distribution-related files.
- Submit the changes and inspect the remote CI results. Local checks do not replace the full compatibility matrix.

### 2.3. Acceptance criteria

All seven job groups must succeed, including every configured matrix combination. `CI Gate` checks that there are exactly seven upstream results and that every result is `success`. A skipped, cancelled, or failed group does not satisfy this condition.

| Required group | Environment / matrix | Acceptance condition |
| --- | --- | --- |
| Static Quality | Ubuntu, Python 3.12 | Ruff lint, Ruff format check, and basedpyright pass using the project configuration. |
| Core Tests | Ubuntu: Python 3.12, 3.13, 3.14; Windows: Python 3.12 | All non-optional test executions succeed. |
| Cache Tests | Ubuntu, Python 3.12 | Required cache tests execute and pass without skips. |
| Spark Tests | Ubuntu, Python 3.12, Java 17 | Required Spark unit and local integration tests execute and pass without skips. |
| Databricks LTS ML Tests | Ubuntu, Python 3.12, Java 17; runtime equivalents 17.3 and 18 | Exact CPU dependency versions, interpreter and import origins, dependency consistency, and required Spark tests pass without skips. |
| MPI Tests | Ubuntu, Python 3.12, OpenMPI | `MPI_THREAD_MULTIPLE` is available; real MPI integration tests execute and pass without skips. |
| Package Build | Ubuntu, Python 3.12 | Locked base environment, distribution construction, metadata, contents, installation and smoke checks pass. |

The runners are `ubuntu-24.04` and `windows-2025`. The Databricks jobs validate isolated environments with equivalent dependencies, **not actual Databricks clusters**. The pins and checks are defined in [databricks_compat.py](../.github/scripts/databricks_compat.py).

For the designated optional compatibility jobs, [check_junit.py](../.github/scripts/check_junit.py) requires a valid, nonempty JUnit report with consistent test counts and no skips, failures, or errors. This does not prohibit individual skips in every other CI suite. JUnit reports are temporary validation inputs, not published indicators; CI does not generate coverage badges or publish the website.

### 2.4. Package Build

The same validation sequence applies to independent CI and CI called by CD:

1. Check the lockfile and install locked base dependencies.
2. Build exactly one wheel and one sdist; run strict Twine metadata validation.
3. Inspect archive contents using [release.py](../.github/scripts/release.py). Reject missing or unexpected files, unsafe archive paths, development residue, and mismatches with tracked sources, README, license or project metadata.
4. Record the package identity and individual SHA-256 hashes.
5. Install each distribution independently in a clean Python 3.12 environment outside the source checkout, with `PYTHONPATH` cleared. Run `uv pip check` and [smoke_package.py](../.github/scripts/smoke_package.py), which verifies installed-package ownership and exercises a small public-API optimization.
6. Recheck the manifest and distribution hashes after the installation checks.

Independent CI does not persist the built distributions. The reusable input `upload_distributions` defaults to `false`; CD sets it to `true` and receives the `artifact_id`, `manifest_sha256` and `producer_attempt` outputs from Package Build.

An uploaded artifact is not approval by itself: the full CI, including `CI Gate`, must succeed. CD calls the local reusable workflow at `./.github/workflows/ci.yml` on the exact authorized release or dispatch SHA. It does not look for an earlier independent CI run or rebuild the approved distributions in the normal publication flow.

## 3. Continuous Deployment

### 3.1. Choose the correct release operation

| Operation | Trigger | Destination |
| --- | --- | --- |
| Validate distributions without publishing | Manual dispatch on `master`, publication disabled | None |
| Publish a test release | Manual dispatch on `master`, publication explicitly enabled | TestPyPI |
| Publish an official release | Publish a stable GitHub Release | PyPI |

All three operations run full reusable CI. Only the two publication paths enter their respective publishing environments and verify the published files. TestPyPI is optional and is not a prerequisite for PyPI.

### 3.2. Before publishing

The following remote configuration must be in place for the selected destination. The YAML does not establish that these settings have actually been configured.

| Setting | TestPyPI | PyPI |
| --- | --- | --- |
| GitHub environment | `testpypi` | `pypi` |
| Trusted Publisher owner | `renard162` | `renard162` |
| Repository | `tyrannis_optimizer` | `tyrannis_optimizer` |
| Workflow filename | `release.yml` | `release.yml` |
| Trusted Publisher environment | `testpypi` | `pypi` |
| Eligible deployment reference | `master` | Release tags (e.g. `v*`) |

Check the Trusted Publisher registrations, permitted deployment references, required reviewers, action permissions and effective artifact retention in the remote settings. The environments and registrations are separate for the two indices.

### 3.3. Manual validation or TestPyPI publication

In **Actions > Controlled package release > Run workflow**, select `master` in the official repository.

| Input | Default | How to use |
| --- | --- | --- |
| `publish_testpypi` | `false` | Leave disabled for validation only. Enable only to request a real TestPyPI upload. |
| `confirm_version` | Empty | Required for publication; enter the exact static version in `pyproject.toml`, without a `v` prefix. |

No tag or GitHub Release is required. The selected checkout SHA must still be the current remote `master` HEAD.

For **validation only**:

1. Leave publication disabled and start the run.
2. Confirm source authorization and the full CI succeed.
3. Confirm `package-validate` succeeds; it repeats the manual guard and checks the downloaded artifact identity and hashes.
4. Expect the publishing and post-publication jobs to be skipped. No index upload is performed.

For **TestPyPI publication**:

1. Enable publication, enter the exact version, and start the run.
2. Review the CI result and the original artifact identity pinned by `bind-publication`.
3. Approve the `testpypi` environment if required by remote settings.
4. Review `publish-testpypi` and `verify-testpypi`. Before any initial upload, the publisher rechecks the manual authorization, including version confirmation and current `master` HEAD.
5. Require the final result `Published and verified`.

A newer `master` HEAD blocks an initial upload from a stale manual dispatch. See [publication recovery](#6-publication-recovery) if the upload or verification fails.

### 3.4. Official PyPI release

Official publication is authorized only by `release: published`. Pushes, PRs, tag creation alone, draft releases, prereleases, and manual dispatch do not authorize it.

Before creating the release, meet the following conditions enforced by [release_official.py](../.github/scripts/release_official.py):

| Item | Required value or condition |
| --- | --- |
| Repository | `renard162/tyrannis_optimizer`; matching event and live repository identity; non-fork |
| Package version | Static, canonical stable `X.Y.Z` in `pyproject.toml` |
| Tag | Exactly `vX.Y.Z`, matching the package version |
| Release | Published, not draft or prerelease; ID, API URL, tag and status agree with the event and live release |
| Source commit | Event SHA, checkout HEAD and the resolved live tag commit agree |
| Release target | `master` or the exact release SHA; that commit is in current `master` history |

The release commit need not be the current HEAD of `master`. Lightweight and annotated tags are supported, with at most ten annotated-tag dereferences.

**Publication procedure:**

1. Prepare the package version and lockfile on `master`; review the code and release notes.
2. Select the intended commit and create the matching tag. Confirm its SHA before publishing the release.
3. Publish a stable GitHub Release for that tag. This starts CD and authorizes the official publication flow.
4. Check source authorization, full CI and the original artifact binding.
5. If requested, approve the `pypi` environment only after reviewing the approved artifact identity.
6. Check `publish-pypi` and `verify-pypi`. Require `Published and verified` before treating the release as complete.

The publisher reconciles the PyPI index and repeats live release authorization before an initial upload. The official workflow neither queries TestPyPI nor uses its artifacts.

## 4. Publication controls

### 4.1. Artifact identity and retention

Two immutable GitHub Actions artifacts preserve the approved distribution bytes and their selection for publication.

| Record | Artifact / contents | Purpose |
| --- | --- | --- |
| Approved distributions | `tyrannis-<sha>-<run_id>-<producer_attempt>`; contains `dist/` and `evidence/release.json` | The manifest records repository, source SHA, run ID, producing attempt, package name, version and individual file hashes. CI exposes its digest and artifact ID separately. |
| Publication binding | `publication-<index>-<run_id>`; contains `binding/publication.json` | Pins the original artifact ID, manifest digest and producer attempt to the repository, SHA, run, index, version, ref and release ID (or `manual`). |

The distribution pair is:

- `tyrannis-<version>-py3-none-any.whl`
- `tyrannis-<version>.tar.gz`

Official upload/download artifact actions transfer the approved distribution artifact by ID into `release/`. Consumers verify its manifest digest, identity, version, producing attempt and each file hash. They do not select artifacts by package version or download all artifacts in the run.

`bind-publication` persists the binding before any upload. Lookup is restricted to the current authorized run; ambiguous or incomplete listings, mismatched identities, expired artifacts and missing bindings on reruns block recovery.

Both artifact uploads request **90 days** of retention, subject to remote retention policy and deletion. This is distinct from GitHub's **30-day** native rerun window and limit of 50 reruns per run. See [re-running workflows and jobs](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/re-run-workflows-and-jobs).

A full rerun can rebuild the package, but recovery still uses the original binding. A successful producer from an earlier attempt remains valid for a failed-job rerun. Rebuilding from the same SHA does not guarantee binary equivalence.

### 4.2. Initial upload versus verification only

After CI and binding succeed, the publisher enters its environment, retrieves the original artifact, verifies its identity and hashes, and runs [publication.py](../.github/scripts/publication.py).

| Situation during preparation | Behavior |
| --- | --- |
| Initial attempt; no published files observed in a valid index response | Recheck live source authorization, stage the approved pair, and recheck staged hashes before upload. |
| Initial attempt; any approved file already visible with matching metadata | Verification only; no upload. |
| Unexpected or duplicate filenames, inconsistent metadata, wrong hashes or unauthorized download origins | Fail; no upload. |
| Inconclusive index query | Fail safely; no upload authorization. |
| Any rerun (attempt 2 or later) | Verification only, even if the initial upload never started. |

Revalidation after environment approval protects against source changes while waiting. The verification-only preparation path does not repeat the pre-upload live source guard because it cannot upload.

**There is at most one upload action invocation on the first attempt, containing the complete approved pair.** There is no complementary upload mode for partial publication. Index absence is not proof that an earlier upload was rejected.

### 4.3. Publishing action and permissions

Both publishers use the official `pypa/gh-action-pypi-publish` action with OIDC Trusted Publishing. Actions are pinned by commit SHA. Permanent publishing API tokens are not required.

The publisher passes only the staged wheel and sdist to the action; auxiliary evidence is not uploaded to the package index. `skip-existing` is `false` and `verify-metadata` is `false` because CI has already validated metadata.

| Component | Permissions / boundary |
| --- | --- |
| Release workflow default, CI, authorization and verification | `contents: read` |
| Publication binding | `contents: read`, `actions: read` |
| Publishing jobs | `contents: read`, `id-token: write`; use `pypi` or `testpypi` environment |

Only the two publishers receive OIDC publication permission. Actual environment protection is configured remotely.

### 4.4. Concurrency

| Flow | Concurrency group |
| --- | --- |
| Official PyPI release | `cd-pypi-<repository>-<ref>` |
| Manual validation / TestPyPI | `cd-testpypi-<repository>` |

Both use `cancel-in-progress: false`. Runs for the same official reference are serialized, and manual runs share a separate group. These groups are independent of Pages and reused CI.

GitHub does not guarantee dispatch order or preserve every pending run: a newer pending run can replace one already waiting. Concurrency does not exclude external publishers or make the index check and upload atomic. See [GitHub concurrency behavior](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency).

## 5. Post-publication verification

Only `Published and verified` establishes success. The upload step alone uses `continue-on-error: true` so the verifier can reconcile an ambiguous upload response if the other publishing steps succeeded. Authorization, integrity and preparation failures remain fatal; cancellation does not automatically proceed to verification.

### 5.1. Required checks

The verifier recovers and checks the original artifact, then performs the following checks in order:

1. Locate the expected package version and both approved filenames, rejecting unexpected or duplicate distributions.
2. Compare index-reported SHA-256 hashes with the original manifest. Require HTTPS download URLs on the selected index's allowed file host.
3. Download both files completely and compare their SHA-256 hashes against the approved values. Discard incomplete downloads before retrying.
4. Install each downloaded distribution independently, check dependency consistency and run the smoke test outside the checkout.

Metadata lookup starts at the version-specific JSON endpoint and falls back to the project endpoint when the version endpoint is unavailable. A missing version is treated as absent for preparation only if both endpoint checks indicate absence in the same observation; an existing empty file listing also yields no published files. During verification, missing distributions are polled for propagation. Disappearing previously observed files leave publication unconfirmed.

Downloads are restricted to `files.pythonhosted.org` for PyPI or `test-files.pythonhosted.org` for TestPyPI; redirects to another origin are rejected. Base dependencies are exported offline from the checkout's lockfile and installed from PyPI. The downloaded Tyrannis distribution is installed directly with `--no-deps`, rather than selecting another Tyrannis release from an index.

### 5.2. Retry policy

| Parameter | Implemented limit |
| --- | --- |
| Global verification window | 600 seconds, shared by reconciliation, downloads and both installations; monotonic clock and POSIX alarm |
| Verification attempts | Up to 24 per retryable operation, within the shared window |
| Backoff | 2, 4, 8, 16, then 30 seconds; subsequent delays capped at 30 seconds before `Retry-After` is applied |
| Direct network timeout | Up to 30 seconds, reduced to the remaining global time |
| Pre-upload metadata preparation | Separate 180-second window, up to 6 attempts |
| Installer networking | `UV_HTTP_RETRIES=0`, `UV_HTTP_TIMEOUT=30`; subprocesses bounded by remaining time |
| Release job timeouts | Authorization 10 min; binding 5 min; manual artifact validation 20 min; each publisher 20 min; each verifier 15 min |

The verification budget starts after the local original-artifact check. The job timeout also covers checkout, artifact retrieval and setup. The global alarm bounds DNS resolution and slow streams; subprocess cleanup terminates child processes when operations end or the deadline expires.

For direct network requests, `Retry-After` may be seconds or an HTTP date. The delay is the larger of backoff and `Retry-After`, even if it exceeds 30 seconds. If it cannot fit in the remaining budget, verification is unconfirmed. The alternate metadata endpoint is not used to bypass a positive `Retry-After` delay.

| Failure | Treatment |
| --- | --- |
| Direct HTTP 404, 408, 429 or 5xx; transient connection, DNS or read failure; interrupted response or length mismatch | Retry metadata lookup or download; discard partial files. |
| Version or distribution not yet visible | Poll for propagation. |
| Installer fetch failure with explicit transient transport evidence and no deterministic error | Retry the affected installation in a fresh environment; retain earlier successful installations within this invocation. |
| Installer fetch failure reporting transient HTTP without accessible `Retry-After` | Publication unconfirmed; no automatic installer retry. |
| Unexpected identity, filename, hash, schema or download origin | Publication inconsistent; no retry. |
| Other HTTP errors, certificate or configuration errors, dependency conflicts, build, dependency-check or smoke-test failures, undetermined errors | Verification failed; no automatic retry. |
| Deadline or applicable retry limit exhausted | Publication unconfirmed, or partial if a valid final subset remains and time is still available. |

Retries repeat eligible verification operations only. They never upload packages.

## 6. Publication recovery

### 6.1. Interpret the result

A failed upload response is not proof of failed publication. Conversely, a successful upload is not proof of correct installation. The verifier reconciles the actual index state and downloaded bytes.

| State / log label | Exit code | Meaning |
| --- | --- | --- |
| No published files observed | — | A successful initial observation found no files; a preparation condition, not proof that an earlier upload never occurred. |
| `Published and verified` | 0 | Complete metadata, downloads, hashes, installations and smoke checks passed. |
| `Partially published` | 4 | A valid final observation contains only a subset of approved files after retry exhaustion, with time remaining. Missing filenames are reported. |
| `Publication unconfirmed` | 2 | Deadline exhausted, unavailable or inconclusive metadata/downloads, disappearing files or another explicitly unconfirmed condition. A deadline exhausted with a partial snapshot is also unconfirmed. |
| `Publication inconsistent` | 3 | Positive evidence contradicts approved identity, metadata or bytes. |
| `Verification failed` | 1 | Functional, configuration, non-retryable or undetermined failure. |

Artifact download and workflow setup may fail before any script status is emitted. Inspect the failed step as well as the final summary.

### 6.2. Safe recovery procedure

**Recovery is verification only.** The workflow does not provide a complementary upload or a way to finish a persistent partial publication automatically.

Start from the original run in **Actions > Controlled package release**:

1. Record its source SHA, index, artifact ID, manifest digest, producer attempt and upload outcome.
2. Confirm that the original distribution artifact and binding remain available.
3. Determine whether the failure occurred before upload, during upload or during verification.
4. Select the smallest supported rerun scope using the table below. Do not replace the original artifacts.

| Situation | Maintainer action |
| --- | --- |
| Only the verifier failed | Rerun `verify-pypi` or `verify-testpypi`. It retrieves the original artifact and repeats full verification without upload. |
| Verification interrupted or workflow cancelled | After the original run stops, rerun the verifier if available, or the necessary failed/cancelled jobs in the same run. Inspect dependencies. |
| Upload returned an error or its outcome is unknown | Let verification reconcile first. If all files are correct and checks pass, no further upload is needed. Otherwise preserve the run and retry verification after resolving transient availability. |
| Job failed before or within publication | Use **Re-run failed jobs** only with the original identity intact. A missing original binding blocks a rerun that needs it; a rerun of the publisher selects verification only. |
| Partial publication persists | Retry verification to allow for propagation. If the same subset persists, investigate the original upload and index evidence. Do not resend missing filenames based on their absence. |
| Full workflow rerun | Not a general recovery procedure. It reruns CI and may rebuild, but must retain the original binding. No rerun permits another upload, even after a failure before the original upload. |
| Original artifacts temporarily inaccessible | Restore access and rerun the same job. |
| Original files/binding expired or deleted, or native rerun window elapsed | Native recovery is blocked. Local retained copies are not workflow inputs for replacement artifacts. Preserve the available evidence and investigate; do not rebuild as a substitute. |

GitHub reruns preserve the original SHA and ref; rerunning a specific or failed job can also rerun dependent jobs. See [GitHub rerun semantics](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/re-run-workflows-and-jobs).

A validation-only manual run can be rerun without upload, but its live `master` guard may reject a stale SHA. Once established that no upload was possible, a fresh validation-only dispatch can validate current `master`.

A newly published release or a new dispatch creates a different run with no original binding. It cannot recognize an ambiguous earlier upload or an external publisher. Do not use a new run to bypass a recovery block, move published tags, change the source for a published version, or automatically create another version just because verification failed. Do not manually resend distributions, overwrite or delete existing files, use `skip-existing`, or substitute rebuilt bytes.

### 6.3. Troubleshooting

| Symptom | What to check |
| --- | --- |
| Required CI group fails, is skipped or cancelled | Inspect that group; fix source separately if needed. Package Build alone cannot authorize publication. |
| Source authorization fails | Compare live release status, tag, source SHA and `master` history or HEAD as appropriate. Do not move a published tag to resume. |
| OIDC or environment approval fails | Check Trusted Publisher identity, environment protection and permissions. Reconcile in the original run if upload may have begun. |
| Index or download service unavailable | Inspect the bounded retries. Resume verification when service recovers; do not infer absence from an unconfirmed result. |
| Hash, binding, manifest or file identity mismatch | Compare original evidence with index evidence. Do not edit evidence or replace bytes. |
| Installation, dependency or smoke check fails | Diagnose the failed check and the approved package. Retry only after resolving an environmental cause; do not claim verified publication. |
| Installer reports transient HTTP without accessible `Retry-After` | Retry verification later rather than repeatedly invoking it immediately. |

For persistent partial publications, ambiguous uploads and unavailable original artifacts, follow the recovery limits in Section 6.2.

## 7. Documentation: GitHub Pages

### 7.1. When the website is published

The independent `Documentation` workflow runs on pushes to `master` affecting either of these paths:

```text
.github/pages/**
.github/workflows/pages.yml
```

It also supports manual dispatch on `master`. The build guard accepts only the official repository and `refs/heads/master` for push or manual dispatch. Pull requests, tag pushes and GitHub Releases do not trigger it.

The published source is [`.github/pages/`](../.github/pages/), currently a static placeholder site. README and `doc/` are not rendered into the site by this workflow; changes confined to those files or to source code do not trigger Pages deployment.

### 7.2. Maintainer procedure

1. Ensure GitHub Pages is enabled remotely with GitHub Actions as the publication source and the `github-pages` environment has the intended protection and deployment branch rules.
2. Make the site changes under `.github/pages/` and merge them into `master`, or manually dispatch `Documentation` on `master`.
3. Confirm `Prepare documentation` and `Publish documentation` complete successfully.
4. Check that the deployment action actually ran and the expected site content is visible. A successful job can intentionally skip an obsolete deployment.

The workflow does not enable Pages automatically and does not depend on CI or CD.

### 7.3. Preparation and deployment stages

| Job | Timeout | Operations |
| --- | --- | --- |
| `build` — Prepare documentation | 10 min | Checkout, `actions/configure-pages` (`enablement: false`), `actions/upload-pages-artifact` from `.github/pages`. Expose artifact ID and name `github-pages-<run_attempt>`. |
| `deploy` — Publish documentation | 15 min | Wait for `build`, checkout, poll for the expected artifact, check content freshness, and invoke `actions/deploy-pages`. Expose deployment URL through the `github-pages` environment. |

Official actions are pinned by SHA. No artifact retention override is set; the pinned upload action's default applies. Preparation inherits `contents: read`; deployment additionally receives `actions: read`, `pages: write` and `id-token: write`.

**Artifact polling:** the deploy job queries the current run's artifacts by expected name, excludes expired entries, and requires the returned ID to equal the build output. It makes six attempts after waits of 0, 2, 4, 8, 16 and 30 seconds (60 seconds of scheduled waiting, plus API time). API errors and missing or mismatched metadata are retried. Exhaustion fails with `Artifact unavailable after all attempts`. There is no explicit per-request timeout in this shell step; the deploy job has its 15-minute limit.

**Freshness:** after polling and any environment wait, deployment fetches remote `master` and compares the selected commit with HEAD only for `.github/pages` and `.github/workflows/pages.yml`.

| Comparison result | Deployment behavior |
| --- | --- |
| Relevant contents identical | Deploy, even if newer code or unrelated documentation commits exist. Reverted changes with identical content are also allowed. |
| Relevant contents differ | Log `Obsolete documentation: deployment skipped`; skip deployment without failing for the content difference. |
| Git error | Fail the job. |

Runs share the `github-pages` concurrency group with `cancel-in-progress: false`. Active deployments are not automatically cancelled, but pending runs may be replaced and ordering is not guaranteed. Freshness is a snapshot before deployment, not an atomic lock on subsequent changes.

### 7.4. Recover a Pages deployment

| Situation | Maintainer action |
| --- | --- |
| Transient artifact or deployment failure | Rerun the failed deployment while the original artifact is available; otherwise rerun preparation and deployment to produce a new Pages artifact. |
| Obsolete documentation skipped | Inspect the newer run; manually dispatch Documentation on current `master` if publication is still needed. |
| Configuration or permission error | Check Pages source, environment protection and action permissions. |

Polling reduces failures caused by the delay between artifact upload and metadata availability. It does not guarantee GitHub service availability.

## 8. Guarantees and limitations

The infrastructure provides separate checks for source approval, approved distribution integrity, publication authorization and actual published-file availability and functionality.

- Transient verification failures receive bounded retries, not unlimited recovery. Unconfirmed publication does not imply an absent publication.
- Verification-only recovery depends on the original run, binding, retained artifacts and native rerun availability. Persistent partial publications have no automated upload completion path.
- Uploading multiple distributions is not atomic. Concurrency cannot close the interval between index observation, authorization and upload or cover external publishers. The binding protects its own run only.
- Pages freshness cannot prevent a change after the final comparison.
- Remote environment protection, Trusted Publisher registration, effective retention and Pages configuration must be checked separately. Source inspection and local validation alone do not prove successful remote publication or deployment.

Supporting automation is maintained in [`.github/scripts/`](../.github/scripts/). Use the workflow logs and the original artifacts when investigating an operational failure.
