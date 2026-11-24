# Continuous Integration, Documentation, and Continuous Deployment

This document describes the development, documentation, and package release workflows of Tyrannis Optimizer. It defines the triggering events, acceptance criteria, publication procedures, and recovery rules.

The workflows are implemented in [ci.yml](../.github/workflows/ci.yml), [pages.yml](../.github/workflows/pages.yml), and [release.yml](../.github/workflows/release.yml), with supporting scripts in [`.github/scripts/`](../.github/scripts/).

## 1. Workflow overview

The infrastructure has three operational responsibilities:

| Workflow | Responsibility |
| --- | --- |
| `CI` | Code quality, compatibility testing and package validation |
| `Documentation` | GitHub Pages publication |
| `Controlled package release` | Package publication, verification and recovery |

CI runs independently and is also called by the release workflow as a mandatory publication gate. CD consumes the distributions approved by that reused CI execution. Documentation is independent of both: it neither consumes their results nor participates in their approval.

PyPI and TestPyPI have separate publication authorization, artifacts and destination checks. Neither requires a prior publication on the other index. However, installation verification for either destination obtains dependencies and the sdist build backend from PyPI; TestPyPI verification therefore still requires PyPI availability.

```text
CI: push / pull request (path filters), manual dispatch
    -> seven required groups, including Package Build -> CI Gate

Controlled package release: release: published / manual dispatch
    -> authorize -> ci (reuses CI, retains approved distributions)
                      |
                      +-> package-validate [manual, publication disabled]
                      |
                      +-> bind-publication [publication requested]
                            -> publish-pypi OR publish-testpypi
                               (prepare; initial upload only when authorized)
                            -> verify-pypi OR verify-testpypi
                               (index, downloads, installations, smoke tests)

Documentation: filtered master push / manual dispatch on master
    -> build -> deploy (artifact polling, freshness check, Pages deployment)
```

The diagram shows job dependencies and stages, not additional jobs. Each publisher depends on both `ci` and `bind-publication`; each verifier also depends on its publisher. An upload action failure can proceed to verification, but a failed authorization or integrity check cannot.

## 2. Continuous Integration (CI)

### 2.1. Triggering events

The `CI` workflow runs when:

- A commit is pushed to `master`, including commits introduced by merging a pull request.
- A pull request targeting `master` is opened, synchronized, or reopened.
- A maintainer manually dispatches the workflow using GitHub Actions.
- Another workflow calls it through `workflow_call`.

Both push and pull request events exclude exactly these paths through `paths-ignore`:

```text
README.md
doc/**
.github/pages/**
.github/workflows/pages.yml
```

Changes confined to these paths do not start independent CI. A change outside them, including a combined code and documentation change, remains eligible. These filters do not prevent manual execution or CI called directly by CD.

Trigger exclusions do not define package contents. `README.md` remains the package description and an sdist input; Package Build checks it against the checkout whenever CI runs.

New pull request runs cancel obsolete runs for the same pull request. Other CI runs use a run-specific concurrency group and are not automatically cancelled. A tag push or GitHub Release publication is not, by itself, an independent CI trigger; an authorized release calls CI through CD.

### 2.2. Acceptance criteria

The `CI Gate` job aggregates the results of the following mandatory job groups. Every group, including all its matrix combinations, must conclude successfully.

| Required job group | Acceptance condition |
| --- | --- |
| Static Quality | Ruff linting, Ruff formatting checks, and basedpyright type checking pass. Type checking uses the locked baseline and the project configuration. |
| Core Tests | All required executions succeed across the configured operating system and Python version matrix. |
| Cache Tests | The required execution succeeds and produces a valid, nonempty JUnit report with no skipped, failed, or erroneous cases. |
| Spark Tests | The required execution succeeds and produces a valid, nonempty JUnit report with no skipped, failed, or erroneous cases. |
| Databricks LTS ML Tests | Each configured runtime-equivalent dependency environment passes its compatibility checks and required test execution, with a valid, nonempty JUnit report containing no skipped, failed, or erroneous cases. The environments reproduce the selected runtime dependencies; they are not actual Databricks cluster executions. |
| MPI Tests | The environment provides the required MPI thread support, the execution succeeds, and its JUnit report contains executed cases with no skips, failures, or errors. |
| Package Build | Locked base dependencies can be installed; exactly one wheel and one source distribution are built; strict metadata validation succeeds; both distributions can be installed and pass independent smoke checks outside the source checkout. |

The core matrix covers Ubuntu 24.04 with Python 3.12, 3.13, and 3.14, and Windows 2025 with Python 3.12. The Databricks matrix covers runtime-equivalent CPU dependencies for LTS ML 17.3 and 18 on Ubuntu with Python 3.12 and Java 17; exact dependency pins and environment checks are defined in [databricks_compat.py](../.github/scripts/databricks_compat.py). Cache, Spark, and MPI validation also run on Ubuntu with Python 3.12; Spark uses Java 17, and MPI requires OpenMPI with `MPI_THREAD_MULTIPLE`.

JUnit reports are temporary inputs to test validation through [check_junit.py](../.github/scripts/check_junit.py), not published indicators. CI is limited to code quality, tests, compatibility, and package validation.

The gate requires exactly seven upstream job groups, each with a `success` result. A failed, skipped, or cancelled required group does not satisfy this condition.

The no-skip requirement enforced by the JUnit validator applies to the designated optional compatibility executions; it is not a blanket prohibition on individual skips in every CI suite.

### 2.3. Package Build and publication integration

`Package Build` validates the lockfile and installs locked base dependencies, then builds exactly one wheel and one sdist. It runs strict Twine metadata validation and [release.py](../.github/scripts/release.py) inspects archive contents against tracked sources, README, license and project metadata. Missing or unexpected files, unsafe archive paths, development residue and source or metadata mismatches fail validation.

Each distribution is installed in its own clean Python environment outside the checkout, with `PYTHONPATH` cleared. Dependency checks and [smoke_package.py](../.github/scripts/smoke_package.py) confirm that imports belong to the installed distribution and exercise a small public-API optimization. The manifest and individual SHA-256 hashes are verified again after these checks.

Independent CI performs the same build and validation but does not persist the distributions. The reusable input `upload_distributions` defaults to `false`; CD sets it to `true`. After Package Build succeeds, the artifact is uploaded and its `artifact_id`, `manifest_sha256` and `producer_attempt` are returned to the caller. Uploading this artifact alone is not CI approval: all seven required groups and `CI Gate` must succeed before downstream CD jobs proceed.

CD first authorizes its source and calls the local reusable workflow at `./.github/workflows/ci.yml`. The reused CI validates the exact SHA of the calling release or dispatch, rather than selecting a different branch head. No historical independent CI run is required or searched. The normal publication flow builds the distribution pair once in this reused CI; CD does not rebuild it.

## 3. Continuous Deployment: publication policy

### 3.1. Official PyPI publication

Official publication starts exclusively from a valid GitHub Release publication (`release: published`). Creating or pushing a tag alone, pushes to a branch, pull requests, draft releases and prereleases do not authorize PyPI publication. Manual dispatch is not an independent PyPI authorization.

[release_official.py](../.github/scripts/release_official.py) requires:

- Repository `renard162/tyrannis_optimizer`, with matching event/live repository identity and non-fork status.
- A published, non-draft, non-prerelease release whose ID, API URL, tag and status agree with the event and the live release.
- A static stable version in canonical `X.Y.Z` form in `pyproject.toml`, and the exact tag `vX.Y.Z`.
- The event SHA, checkout HEAD and live tag's resolved commit to agree. Lightweight and annotated tags are supported, with at most ten annotated-tag dereferences.
- A release targeting `master` or the exact release SHA, and a commit contained in current `master` history. The commit need not remain its HEAD.

To publish, prepare the version and lockfile on `master`, select the intended commit, create its matching tag, and publish a stable GitHub Release for that tag. The release workflow then authorizes the source, runs full CI, pins the approved artifact, enters the `pypi` environment, reconciles the index and revalidates authorization before any initial upload. Complete the required environment approval, if configured, only after reviewing CI and the artifact identity. Publication is confirmed only after `verify-pypi` reports `Published and verified`.

### 3.2. Manual validation and TestPyPI publication

In **Actions > Controlled package release > Run workflow**, select `master` in the official repository.

| Input | Default | Meaning |
| --- | --- | --- |
| `publish_testpypi` | `false` | Validate and retain distributions without publication; set to `true` to request TestPyPI publication. |
| `confirm_version` | Empty | Required only for publication; must exactly match the static `pyproject.toml` version, without adding a `v` prefix. |

No tag or GitHub Release is required. The guard checks repository, event, branch, checkout SHA and current remote `master` HEAD before CI. For validation only, `package-validate` repeats the guard and verifies the downloaded CI artifact identity and hashes. Success requires authorization, full CI and this validation job; publication jobs are intentionally skipped.

When publication is requested, `bind-publication` pins the artifact and `publish-testpypi` uses the `testpypi` environment. Before an initial upload, it repeats the manual guard, including version confirmation and current `master` HEAD. If `master` has advanced, that upload is blocked. The selected distributions must then pass `verify-testpypi`, regardless of the upload action's outcome.

TestPyPI publication is optional and is never a prerequisite for official PyPI publication. The official path does not query TestPyPI or consume its artifacts.

## 4. Artifact identity and publication authorization

### 4.1. Approved bytes and transfer

The CI artifact is named `tyrannis-<sha>-<run_id>-<producer_attempt>` and contains `dist/` plus `evidence/release.json`. The distributions are `tyrannis-<version>-py3-none-any.whl` and `tyrannis-<version>.tar.gz`. The manifest records repository, source SHA, run ID, producing attempt, package name, version and individual distribution SHA-256 hashes. CI separately exposes the manifest digest and immutable artifact ID; consumers download that ID rather than selecting files by version or downloading every artifact in the run. Transfer uses `actions/upload-artifact` and `actions/download-artifact`, with consumers restoring the files under `release/`.

Before entering a publishing job, `bind-publication` stores `binding/publication.json` in the artifact `publication-<index>-<run_id>`. This binding associates repository, SHA, run, destination index, version, ref and release ID (or `manual`) with the original artifact ID, manifest digest and producer attempt. It is persisted before upload so an interrupted publisher does not erase the recovery identity.

Both records are necessary: the manifest identifies approved bytes; the binding preserves the original selection across reruns. Binding lookup is restricted to the current authorized run. Ambiguous or incomplete artifact listings, wrong identities, expired artifacts or a missing binding on a rerun block recovery.

Consumers verify the manifest digest, repository, SHA, run, version, producer attempt and individual hashes. A successful build from an earlier attempt remains valid for a failed-job rerun. A full rerun may build a new pair, but an existing binding still selects the original pair. Rebuilding from the same commit does not guarantee binary equivalence and cannot replace original files for recovery.

Both distribution and binding uploads request **90 days** of retention. Effective availability also depends on remote retention policy and deletion. This is distinct from GitHub's **30-day** native rerun window; GitHub also limits a run to 50 reruns. See [re-running workflows and jobs](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/re-run-workflows-and-jobs).

### 4.2. Controls immediately before upload

The publishing jobs run only after full reused CI and binding succeed. After any concurrency wait and environment approval, the publisher downloads the original artifact, checks its identity and hashes, and prepares publication through [publication.py](../.github/scripts/publication.py).

On the first run attempt, preparation queries only the destination index. If any expected distribution is already visible with matching metadata, it selects verification without upload. Unexpected files, duplicate names, inconsistent metadata, unauthorized download locations or differing hashes fail immediately. An inconclusive index query does not authorize upload.

Only an initial attempt with a successful observation of no published files proceeds to the live source guard. That guard rechecks the release/tag/ancestry for PyPI or the manual inputs/current `master` HEAD for TestPyPI. The script stages only the approved wheel and sdist and rechecks each staged hash. Revalidation after environment approval prevents an earlier check from authorizing an upload after the release, tag or branch has changed.

On any rerun (`GITHUB_RUN_ATTEMPT` other than `1`), preparation selects verification only, even if upload never actually started. That path does not repeat the pre-upload live source guard because it cannot upload. There is no additional upload authorized by an absent file or an ambiguous result.

### 4.3. Trusted Publishing and external configuration

Both publishers use the official `pypa/gh-action-pypi-publish` action with OIDC Trusted Publishing. Permanent publishing API tokens are not required. Actions are pinned by commit SHA. `skip-existing` is `false`; metadata has already passed CI, so the publisher sets `verify-metadata: false`.

| Required external configuration | TestPyPI | PyPI |
| --- | --- | --- |
| GitHub environment | `testpypi` | `pypi` |
| Trusted Publisher owner | `renard162` | `renard162` |
| Repository | `tyrannis_optimizer` | `tyrannis_optimizer` |
| Workflow filename | `release.yml` | `release.yml` |
| Trusted Publisher environment | `testpypi` | `pypi` |
| Eligible source for environment deployment rules | `master` | Matching release tags (`v*`) |

Configure each index's Trusted Publisher registration and GitHub environment protection rules separately. Required reviewers, deployment restrictions, action permissions and effective retention must be checked remotely; their existence or correctness cannot be proven by the checkout.

The release workflow defaults to `contents: read`. Only the two publishing jobs receive `id-token: write`; CI, binding and verification do not. Binding additionally receives `actions: read` to locate and validate artifacts. The `pypi` and `testpypi` environment declarations select the protected publication boundaries, while actual protection depends on their remote configuration.

### 4.4. Concurrency

Official runs use `cd-pypi-<repository>-<ref>`; manual runs use `cd-testpypi-<repository>`. Both set `cancel-in-progress: false`. Thus runs for the same official release reference are serialized, and all manual release runs for the repository share a separate group. These groups are independent of Pages and the run-specific reused CI group, so reused CI does not cancel its caller.

This configuration does not promise execution in dispatch order or preservation of every pending run: a newer pending run can replace one already waiting. It does not exclude uploads by external publishers or make index checks and uploads atomic. See [GitHub concurrency behavior](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency).

## 5. Post-publication verification

A successful upload action is insufficient to confirm publication. Only `Published and verified` is success. The upload step alone uses `continue-on-error: true`, allowing a failed upload response to be reconciled by the verifier if the publisher's other steps succeed. Authorization, artifact or preparation failures remain fatal; cancelled workflows do not automatically continue to verification.

The verifier recovers and checks the original artifact, then requires:

1. The expected package version and both expected distribution filenames on the selected index, with no unexpected or duplicate files.
2. Index-reported SHA-256 hashes matching the approved manifest and download URLs on the selected index's allowed HTTPS file host.
3. Complete downloads of both distributions and matching hashes of the downloaded bytes. Partial downloads are discarded before retry.
4. Separate clean installations of wheel and sdist, dependency consistency checks and successful smoke tests outside the checkout.

Metadata lookup starts at the version-specific JSON endpoint and falls back to the project endpoint when the version endpoint is unavailable. Before initial publication, both endpoint checks must indicate absence in the same observation to treat missing version metadata as absence; an existing empty file listing also yields no published files. During verification, missing files are polled for propagation and never authorize another upload. If previously observed files disappear, verification is unconfirmed.

Downloads remain on the selected index's allowed file host: `files.pythonhosted.org` for PyPI or `test-files.pythonhosted.org` for TestPyPI. Redirects to another origin are rejected. Base dependencies are exported from the checkout's lockfile offline and installed from PyPI. The downloaded Tyrannis distribution is then installed directly with `--no-deps`, without selecting another Tyrannis package from an index. Each installation checks imported-package ownership and a small optimization through the public API.

### 5.1. Retry policy

| Parameter | Implemented value |
| --- | --- |
| Global verification window | 600 seconds shared by reconciliation, downloads and both installations; monotonic clock and POSIX alarm |
| Verification attempts | At most 24 per retryable operation, always within the shared window |
| Backoff | 2, 4, 8, 16, then 30 seconds, capped at 30 for subsequent retries |
| Direct network timeout | At most 30 seconds, reduced to remaining global time |
| Pre-upload metadata preparation | Separate 180-second window, at most 6 attempts |
| Installer networking | `UV_HTTP_RETRIES=0`, `UV_HTTP_TIMEOUT=30`; subprocesses bounded by remaining global time |
| Workflow job limits | Authorization 10 minutes; binding 5; manual artifact validation 20; each publisher 20; each verifier 15 |

For direct metadata requests and downloads, `Retry-After` accepts seconds or an HTTP date. The delay is the larger of backoff and that value; it is not shortened to the 30-second backoff cap. If it cannot fit in the remaining budget, verification becomes unconfirmed. A server's `Retry-After` is not bypassed through the alternate metadata endpoint. The global alarm also bounds DNS resolution and slow response streams; subprocess cleanup terminates child processes when operations end or time expires.

| Failure category | Treatment |
| --- | --- |
| Direct HTTP 404, 408, 429 or 5xx; temporary DNS/connection/read failure; interrupted response or length mismatch | Retry metadata lookup or download within limits; discard partial download files. |
| Version or one distribution not yet visible | Poll for propagation; an exhausted observation can be partial or unconfirmed, never permission to resend. |
| Installer fetch failure with explicit transient transport evidence and no deterministic error | Retry that distribution's installation in a fresh environment. Previously completed installations are not repeated within this invocation. |
| Installer fetch failure reporting transient HTTP without accessible `Retry-After` | `Publication unconfirmed`; no automatic installer retry. Retry verification later. |
| Unexpected identity, filename, hash, schema or download origin | `Publication inconsistent`; no retry. |
| Other HTTP errors, certificate/configuration errors, dependency conflicts, build failures, dependency-check or smoke-test failures, undetermined errors | `Verification failed`; no automatic retry. |
| Global deadline exhausted or no retry fits | `Publication unconfirmed`; investigate and resume verification later if appropriate. |

Retries repeat eligible verification operations only. They never upload packages. The retry budget begins after local original-artifact verification; job timeouts also bound checkout, artifact retrieval and setup.

## 6. Publication states and recovery

### 6.1. Recognized outcomes

| State or log label | Meaning and result |
| --- | --- |
| No published files observed | A successful initial metadata observation finds no expected files. This is an upload preparation condition, not a terminal verification success or proof that an earlier upload never happened. |
| `Published and verified` | All metadata, downloads, hashes, independent installations and smoke checks passed; exit 0. |
| `Partially published` | A valid final observation contains only a subset of approved files after retry exhaustion, with time still remaining; exit 4. Missing filenames are reported. |
| `Publication unconfirmed` | Deadline exhausted, unavailable or inconclusive metadata/downloads, disappearing files, or another explicitly unconfirmed condition; exit 2. A deadline exhausted with a partial snapshot is still unconfirmed. |
| `Publication inconsistent` | Positive evidence contradicts approved identity, metadata or bytes; exit 3. |
| `Verification failed` | Functional, configuration, non-retryable or undetermined failure; exit 1. |

Artifact download or workflow setup can fail before the script emits one of these labels. Read the failed step as well as the final summary. An unsuccessful upload action and an unconfirmed publication are different facts: either can coexist with files already accepted by the index.

### 6.2. Reconciliation and partial publication

The index state and downloaded bytes, rather than the upload action's response alone, determine confirmation. If all expected files are visible and correct, verification can succeed even after an upload action error. If only some are visible, the verifier waits for propagation. Persistent partial publication is reported without resending files. If none can be found reliably or the index remains unavailable, publication is unconfirmed. Unexpected files or differing hashes fail immediately as inconsistent.

There is **no complementary upload or partial-upload recovery mode**. Existing file metadata is checked against the original manifest; missing filenames are identified, but absence does not prove the index rejected those files. At most one upload action is invoked on the initial attempt, containing the complete approved pair, and no upload is invoked on reruns. No files are overwritten or deleted, and publication of multiple distributions is not atomic.

The supported recovery is to retrieve the original bound artifact and repeat full verification. A genuinely persistent partial publication has no automated completion path in this workflow. Investigate the original upload and index state; do not bypass the gate with manual resends, `skip-existing`, rebuilt files, a new dispatch or republishing the release.

### 6.3. Safe rerun procedures

Start from the original run in **Actions > Controlled package release**. Record its SHA, index, artifact ID, manifest digest, producer attempt and upload outcome. Check whether the original distribution artifact and publication binding remain available.

| Situation | Supported procedure and limits |
| --- | --- |
| Only a verification job failed | Rerun that `verify-pypi` or `verify-testpypi` job. It retrieves original files and performs full verification without upload. |
| Verification was interrupted or workflow cancelled | After the original run has stopped, rerun the verifier if available, or the necessary failed/cancelled jobs in that same run. Inspect dependencies; no rerun authorizes another upload. |
| Upload returned an error or its result is ambiguous | Allow the verifier to reconcile first. If it cannot confirm publication, retain the original run and artifacts and retry verification after resolving transient availability. |
| Failed jobs before or within publication | Use **Re-run failed jobs** only with the original identity intact. Successful earlier jobs can supply outputs from the producing attempt. If binding never completed, the rerun blocks; if the publisher runs again, it selects verification only. |
| Persistent partial publication | Retry verification to account for propagation. If the same subset persists, investigate; automatic completion and complementary uploads are unavailable. |
| Full workflow rerun | Not a general publication recovery procedure. It reruns CI and may rebuild, but an existing binding must keep the original artifact; missing binding blocks recovery. Even a failure before upload does not enable uploading on attempt 2 or later. |
| Original artifacts temporarily inaccessible | Restore access, then rerun the same job. Do not substitute files from another attempt or run. |
| Binding or original files expired/deleted, or native rerun window elapsed | Native recovery cannot proceed. Retained local copies do not provide a workflow input for replacement artifacts. Preserve available evidence and investigate outside publication; do not rebuild as a substitute. |

GitHub reruns preserve the original SHA and ref. Rerunning a specific or failed job can also rerun dependent jobs; inspect the selected scope. See [GitHub rerun semantics](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/re-run-workflows-and-jobs).

A validation-only manual run can safely be rerun without upload, but its live `master` guard may reject a stale SHA. After establishing that no upload was possible, a fresh validation-only dispatch can validate current `master`.

A new dispatch or newly published release is a different run, without the original binding. It cannot recognize an ambiguous upload from an earlier run or an external publisher. Do not use a new run to circumvent a recovery block. Do not move published tags, change the source for a published version or automatically create a new version merely because verification failed.

### 6.4. Operational troubleshooting

| Symptom | Automatic behavior | Maintainer action; operations to avoid |
| --- | --- | --- |
| CI group fails, is skipped or cancelled | Publication is blocked. | Inspect the required group. Fix source separately if needed; Package Build success alone cannot approve publication. |
| Current `master`, release status, tag or SHA guard fails | No upload. | Compare the event and live source identity. Do not move a published tag or bypass authorization to resume. |
| Upload action fails, but verifier succeeds | Reconciled as `Published and verified`. | Record the verified outcome; no further upload is needed. |
| Metadata/download service is temporarily unavailable | Bounded retries, then unconfirmed. | Retry the original verifier when service recovers; do not infer absence or create a replacement publication run. |
| Partial publication persists | Bounded propagation polling, then partial/unconfirmed. | Investigate upload/index evidence. Do not resend the missing filename on the basis of absence. |
| Hash, file identity, binding or manifest mismatch | Immediate block. | Compare original artifact and index evidence; do not edit evidence, use `skip-existing` or replace bytes. |
| Install, dependency or smoke check fails | No retry for functional errors. | Diagnose installation logs and approved package behavior. Retry only after resolving an environmental cause; do not claim verified publication. |
| Installer reports transient HTTP without `Retry-After` | Unconfirmed without installer retry. | Retry verification later; avoid rapid repeated retries. |
| Original artifact or binding unavailable | Recovery blocked. | Restore access if possible; expiration/deletion has no rebuild fallback. |
| OIDC or environment approval fails | Upload cannot be confirmed from that failure alone. | Check Trusted Publisher identity, environment and permissions; reconcile through original verification if upload may have begun. |

## 7. Documentation: GitHub Pages

### 7.1. Triggers and source

The independent `Documentation` workflow runs on pushes to `master` affecting exactly these path patterns:

```text
.github/pages/**
.github/workflows/pages.yml
```

It also supports manual dispatch. Its `build` job accepts only `refs/heads/master` in `renard162/tyrannis_optimizer`, for push or manual dispatch. Pull requests, tag pushes and GitHub Releases do not trigger it. Source-code-only changes do not require a new site publication; changes only to README or `doc/` also do not trigger this workflow.

The published source is [`.github/pages/`](../.github/pages/), currently a static placeholder site. The workflow does not render README or `doc/` into a documentation site.

### 7.2. Preparation and publication

The `build` job (`Prepare documentation`, 10-minute timeout) checks out the selected commit, runs `actions/configure-pages` with `enablement: false`, and packages `.github/pages` using `actions/upload-pages-artifact`. It stores the Pages artifact as `github-pages-<run_attempt>` and exposes both its artifact ID and name. No retention override is set here; the pinned upload action's default applies.

The `deploy` job (`Publish documentation`, 15-minute timeout) depends on `build`. It checks out the source, waits for the identified artifact, checks freshness, then calls `actions/deploy-pages` with the produced artifact name. These are official GitHub actions pinned by SHA. Preparation and publication are separate jobs; neither depends on CI or CD.

GitHub Pages must be enabled remotely with GitHub Actions as the publication source. The workflow does not enable it automatically. `deploy` targets the `github-pages` environment and exposes the deployment URL. It has `contents: read`, `actions: read`, `pages: write` and `id-token: write`; preparation uses the workflow's `contents: read` default. Configure environment protection and permitted deployment branches remotely; the YAML declaration alone does not prove approval rules are enabled.

### 7.3. Artifact availability

Before deployment, polling lists artifacts from the current workflow run, filters by the expected name and excludes expired entries. It proceeds only when the returned ID equals the ID emitted by `build`.

There are six attempts, preceded by waits of **0, 2, 4, 8, 16 and 30 seconds** (60 seconds of scheduled waiting, plus API time). API errors or missing/mismatched metadata lead to another attempt. Exhaustion fails with `Artifact unavailable after all attempts`; no deployment follows. There is no explicit per-request timeout in this polling shell; the deploy job has its 15-minute limit.

This bridges the delay between upload completion and artifact metadata availability. It does not guarantee GitHub service availability. After a transient failure, rerun the failed deployment while its original artifact is available; otherwise rerun preparation and deployment to produce a new Pages artifact.

### 7.4. Protection against obsolete documentation

After artifact polling and any environment wait, `deploy` fetches current remote `master` and compares the selected commit with that HEAD only for `.github/pages` and `.github/workflows/pages.yml`.

A newer commit on `master` does not by itself block deployment. If the relevant content is identical, deployment is allowed even when code or other documentation changed. If those paths differ, the workflow logs `Obsolete documentation: deployment skipped` and skips the deploy action without treating that content difference as a failure. Git errors fail the job. Content equality, rather than commit age, is the criterion; a reverted change with identical content is allowed.

Runs share the `github-pages` concurrency group with `cancel-in-progress: false`, preserving an active deployment while serializing that group. Pending runs can still be replaced and dispatch order is not guaranteed. The freshness check is a snapshot immediately before deployment, not an atomic lock on future `master` changes. After a freshness skip, inspect the newer documentation run or manually dispatch Documentation on current `master` if publication is still needed.

## 8. Guarantees and limitations

The infrastructure separately establishes source approval through full CI, distribution integrity through immutable artifact identity and hashes, publication authorization through event/source/environment controls, and published-file availability through downloads and independent installations.

Transient verification failures have bounded recovery, not unlimited retries. Verification-only recovery depends on the original run, binding, artifacts and native rerun availability. An unconfirmed publication does not necessarily mean no publication occurred. Prolonged external outages, persistent partial publication, expired artifacts or functional package defects can require investigation without any supported upload recovery.

Multiple distribution uploads are not atomic. Concurrency does not close the interval between an index observation, authorization and upload, nor cover publishers outside these groups. The binding protects only its own run. Pages freshness similarly cannot prevent changes after its final comparison.

The checkout establishes these workflow contracts, but does not prove remote environment protections, Trusted Publisher registrations, retention settings, Pages configuration or a successful real publication/deployment. Check those settings and the actual remote run results before declaring an operational release verified.
