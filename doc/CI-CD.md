# Continuous Integration, Documentation, and Continuous Deployment

This document describes the development, documentation, and package release workflows of Tyrannis Optimizer. It defines the triggering events, acceptance criteria, publication procedures, and recovery rules.

The workflows are implemented in [ci.yml](../.github/workflows/ci.yml), [pages.yml](../.github/workflows/pages.yml), and [release.yml](../.github/workflows/release.yml), with supporting scripts in [`.github/scripts/`](../.github/scripts/).

## 1. Workflow overview

The CI/CD infrastructure consists of three independent GitHub Actions workflows and four operational flows:

| Flow | Workflow | Trigger | Purpose |
| --- | --- | --- | --- |
| CI | `CI` | Push to `master`, pull request targeting `master`, or manual dispatch | Validate the source code and package. |
| Documentation | `Documentation` | Push to `master` or manual dispatch; deployment restricted to `master` in the official repository | Publish the documentation site. |
| CD-1 | `Controlled package release` | Manual dispatch on `master` | Validate distributions and optionally publish them to TestPyPI. |
| CD-2 | `Controlled package release` | Publication of a stable GitHub Release | Publish validated distributions to PyPI. |

CI does not automatically publish packages to either index. CD-1 and CD-2 use separate event conditions within the same release workflow. A successful TestPyPI publication does not automatically trigger a PyPI publication.

## 2. Continuous Integration (CI)

### 2.1. Triggering events

The `CI` workflow runs when:

- A commit is pushed to `master`, including commits introduced by merging a pull request.
- A pull request targeting `master` is opened, synchronized, or reopened.
- A maintainer manually dispatches the workflow using GitHub Actions.

New pull request runs cancel obsolete runs for the same pull request. A tag push or GitHub Release publication is not, by itself, a CI trigger.

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

### 2.3. CI approval for package publication

Passing `CI Gate` is necessary but not sufficient to authorize a release. Both CD flows require a completed `ci.yml` workflow run with:

- Event: `push`.
- Branch: `master`.
- Head repository: `renard162/tyrannis_optimizer`.
- Head SHA: the exact commit selected for publication.
- Status: `completed`.
- Conclusion: `success`.

A successful pull request run, a manually dispatched CI run, or a successful run for another commit does not satisfy the release gate. After changing release inputs, push the resulting commit to `master` and wait for its qualifying CI run to succeed before initiating CD.

## 3. CD-1 — Controlled TestPyPI release

### 3.1. Trigger and inputs

CD-1 is triggered exclusively by manually running `Controlled package release` from the `master` branch of the official repository.

In GitHub, open **Actions > Controlled package release > Run workflow**, select `master`, and configure the following inputs:

| Input | Default | Meaning |
| --- | --- | --- |
| `publish_testpypi` | `false` | When false, validate and retain distributions without publishing. When true, request publication to TestPyPI. |
| `confirm_version` | Empty | Required for publication. Must exactly match the static version in `pyproject.toml`, without a `v` prefix. |

No Git tag or GitHub Release is required. Neither a push to `master` nor a successful CI run automatically starts CD-1.

### 3.2. Procedure

1. Prepare the desired package version in `pyproject.toml`, synchronize `uv.lock`, and push the final commit to `master`.
2. Wait for a successful push-triggered CI run for that exact commit.
3. Manually dispatch the release workflow using the inputs described above.
4. The `package-build` job verifies the repository, event, branch, checkout SHA, current `master` HEAD, CI approval, and explicit version confirmation when publication is requested. It checks the lockfile, builds one wheel and one source distribution (sdist), validates metadata and contents, records the SHA-256 hashes and provenance, and checks independent installations of both distributions. The approved files and evidence are retained as a GitHub Actions artifact.
5. If `publish_testpypi` is false, the flow stops after validation. Both publication and post-publication verification are skipped by design.
6. If `publish_testpypi` is true, `publish-testpypi` retrieves the approved artifact, verifies its identity and hashes, and rechecks the current `master` HEAD, CI approval, and authorization immediately before upload. It publishes the exact approved files through the `testpypi` environment using Trusted Publishing (OIDC), without rebuilding.
7. After a successful upload, `verify-testpypi` retrieves the published files exclusively from TestPyPI, checks their metadata, filenames, hashes, and contents against the original evidence, and verifies separate installations of the wheel and sdist.

For a validation-only run, success of `package-build` is sufficient. For a TestPyPI publication, all three CD-1 jobs must succeed.

The dispatched commit must remain the current `master` HEAD until the final pre-upload guard passes. If `master` advances, the publication is blocked and the maintainer must reassess the selected commit.

A validation-only artifact is not proof of a successful TestPyPI publication and cannot be used as the origin of a CD-2 promotion.

## 4. CD-2 — Official PyPI release

### 4.1. Trigger and release requirements

CD-2 starts automatically when a GitHub Release is published (`release: published`). The eligible release must be a stable release in the official repository, not a draft or prerelease.

The release guard requires:

- A stable package version in canonical `X.Y.Z` format, as declared in `pyproject.toml`.
- A tag named exactly `vX.Y.Z`, matching that version.
- A checkout matching the SHA associated with the release event and the live tag. Both annotated and lightweight tags are supported.
- A release targeting `master` or the exact release commit SHA.
- A release commit contained in the current `master` history. Unlike CD-1, the commit does not need to remain the current `master` HEAD.
- A qualifying successful push-triggered CI run for that exact commit.
- No existing publication of the selected version on PyPI, as determined by the pre-publication index checks.

Creating or pushing a tag alone does not trigger CD-2. Saving a draft release does not publish the package. Manual workflow dispatch is reserved for CD-1.

### 4.2. Maintainer procedure

1. Select an unpublished stable version and update `pyproject.toml` and `uv.lock` as required.
2. Commit and push the release preparation to `master`.
3. Confirm the completed, successful push-triggered CI run for the intended release SHA.
4. Optionally publish and verify that exact commit and version on TestPyPI using CD-1. If promotion is intended, retain the original CD-1 artifact and evidence. Do not change the release commit after TestPyPI publication.
5. Create and push a tag matching the version, pointing to the CI-approved commit. For example, from the verified release commit:

   ```sh
   git tag -a vX.Y.Z -m "Release vX.Y.Z"
   git push origin refs/tags/vX.Y.Z
   ```

   Replace `X.Y.Z` with the selected version. Confirm that the tag resolves to the intended SHA before proceeding.
6. Open **Releases > Draft a new release** in GitHub. Select the tag, add the release title and release notes, and leave the prerelease option disabled.
7. Publish the GitHub Release. This automatically starts CD-2.
8. Monitor the three official release jobs and investigate any failure before attempting further publication actions.

### 4.3. Artifact selection

The `release-validate` job first verifies release eligibility and checks whether the version already exists on PyPI. It then selects one of two mutually exclusive strategies:

| TestPyPI status | Strategy |
| --- | --- |
| Version is not found by the required index checks. | Build new distributions once, from the tagged commit. |
| Version exists and a unique, nonexpired CD-1 artifact proves a successful publication from the same commit. | Promote the original distributions without rebuilding. |
| Version exists but the original evidence is missing, expired, ambiguous, or inconsistent. | Block the release. No fallback build is allowed. |
| Index availability or metadata is inconclusive. | Block the release rather than assume absence. |

Promotion requires evidence of success in all three original CD-1 jobs. The original manifest, run ID, run attempt, commit, CI approval, and hashes are preserved and verified against TestPyPI. The version alone is not sufficient to establish provenance.

Publishing to TestPyPI is therefore optional for CD-2. However, once the selected version exists there, the promotion requirements apply.

### 4.4. Official release execution

| Job | Acceptance condition |
| --- | --- |
| `release-validate` | The release guard passes, the artifact strategy is authorized, the lockfile and distribution metadata are valid, provenance and content are verified, and both distributions pass independent installation checks. The exact approved files and evidence are uploaded as a workflow artifact. |
| `publish-pypi` | The approved artifact is retrieved by ID. Evidence, hashes, version absence, live tag, ancestry, and CI approval are checked again after any environment approval. The exact approved distributions are uploaded through the `pypi` environment using OIDC, without rebuilding. |
| `verify-pypi` | The approved evidence is verified, the published wheel and sdist are downloaded exclusively from PyPI, their names, SHA-256 hashes, and contents match the approved distributions, and both pass independent installation and dependency checks. |

CD-2 is fully successful only when all three jobs succeed. A successful publisher job followed by a failed verification job means that the publication occurred, but the overall release has not passed end-to-end verification.

## 5. Publication controls and failure handling

### 5.1. Trusted Publishing and artifact identity

The GitHub environments and the corresponding Trusted Publisher registrations must be configured separately for the two indexes.

| Configuration | TestPyPI | PyPI |
| --- | --- | --- |
| GitHub environment | `testpypi` | `pypi` |
| Recommended deployment restriction | Branch `master` | Tags matching `v*` |
| Trusted Publisher owner | `renard162` | `renard162` |
| Repository | `tyrannis_optimizer` | `tyrannis_optimizer` |
| Workflow filename | `release.yml` | `release.yml` |

Configure any required environment approvals before publication. Package publishing uses OIDC rather than stored PyPI API tokens. Within the release workflow, `id-token: write` is restricted to the publisher jobs.

The build or promotion stage records artifact provenance and SHA-256 hashes. Subsequent jobs reuse the validated bytes instead of rebuilding. The release workflow serializes TestPyPI executions for the repository and official executions for the same release reference; active uploads are not automatically cancelled.

Publication uses `skip-existing: false` for both indexes. PyPI rejects reuse of previously uploaded distribution filenames; the workflow does not implement overwriting, automatic version changes, or rollback.

GitHub Actions retains the release artifacts for 30 days. Preserve the evidence separately if it is needed beyond that retention period, particularly for future promotion or auditing.

### 5.2. Index consistency and post-publication verification

The shared release helper first queries the version-specific JSON endpoint of the selected index. If necessary, it consults the project-level endpoint and searches for the exact version in its release listing.

Potentially transient failures are retried within a bounded policy of up to six rounds, with progressive delays of 2, 4, 8, 16, and 30 seconds after the first attempt and a configured 180-second metadata lookup budget. A version is treated as absent before publication only when both endpoint checks indicate absence in the same round. This is a preventive check, not an absolute guarantee against stale index responses.

After publication, missing metadata triggers further attempts rather than being accepted as absence. Invalid responses or inconsistent project identities are rejected. Once metadata is obtained, unexpected distribution names, file counts, download locations, or hashes cause immediate failure. The verifier does not switch between package indexes.

For installation verification, base dependencies are obtained from PyPI using the locked dependency export. The wheel and sdist from the selected publication index are installed separately without resolving the Tyrannis package again from an index.

### 5.3. Recovery rules

- If validation fails before upload, inspect the logs and correct the underlying cause. A new source commit requires a new qualifying CI run. Rerunning the original release workflow does not substitute a newer commit for its original SHA.
- If upload fails or its outcome is uncertain, inspect the index and original workflow evidence before attempting another publication. A partial upload may already have reserved distribution filenames.
- If upload succeeds but post-publication verification fails, do not automatically rerun the publisher or the entire release workflow. Confirm the published distributions using the original approved evidence and perform a read-only verification.
- Do not move published tags or change the source associated with an already published package version as a routine recovery procedure. Use a new version for subsequent package changes.
- Preserve the original execution identity. Partial reruns using artifacts from another run attempt are rejected by the provenance checks.

For a read-only audit, retrieve the original release artifact and verify its manifest and evidence against the workflow summary. Use the original source checkout and the release helper to download and inspect the already published distributions. Repeat the isolated installation checks without invoking either publisher.

## 6. Documentation — GitHub Pages

The independent `Documentation` workflow in [pages.yml](../.github/workflows/pages.yml) publishes the official documentation site. It runs on pushes to `master` and manual dispatches. The deployment job accepts only `refs/heads/master` in `renard162/tyrannis_optimizer`, for either of those events. Pull requests, tag pushes, and GitHub Releases do not trigger it.

The workflow checks out the selected commit, configures Pages without automatic enablement, and uploads [`.github/pages/`](../.github/pages/) directly as the Pages artifact. That directory currently contains a static placeholder `index.html`; the workflow does not render the README or `doc/` into an expanded documentation site.

GitHub Pages must be configured to use GitHub Actions. Publication uses `actions/deploy-pages` and the `github-pages` environment, with `pages: write` and `id-token: write` permissions limited to the deployment job. Deployments share the `github-pages` concurrency group and do not cancel an active deployment. Immediately before deployment, after environment approval and queueing, the workflow compares the selected commit with the remote `master` HEAD and skips obsolete commits.

Documentation publication is independent of CI and CD: it neither consumes test results nor requires a successful CI run or package release. Documentation failures do not invalidate CI or its approval for package publication, and test failures do not, by themselves, prevent site publication. CI and CD do not publish the site.

