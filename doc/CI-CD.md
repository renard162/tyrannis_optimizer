# Continuous Integration and Continuous Deployment

This document describes the development and package release workflows of Tyrannis Optimizer. It defines the triggering events, acceptance criteria, publication procedures, and recovery rules.

The workflows are implemented in [ci.yml](../.github/workflows/ci.yml) and [release.yml](../.github/workflows/release.yml), with supporting scripts in [`.github/scripts/`](../.github/scripts/).

## 1. Workflow overview

The CI/CD infrastructure consists of two GitHub Actions workflows and three operational flows:

| Flow | Workflow | Trigger | Purpose |
| --- | --- | --- | --- |
| CI | `CI` | Push to `master`, pull request targeting `master`, or manual dispatch | Validate the source code and package. |
| CD-1 | `Controlled package release` | Manual dispatch on `master` | Validate distributions and optionally publish them to TestPyPI. |
| CD-2 | `Controlled package release` | Publication of a stable GitHub Release | Publish validated distributions to PyPI. |

CI does not automatically publish packages to either index. CD-1 and CD-2 use separate event conditions within the same release workflow. A successful TestPyPI publication does not automatically trigger a PyPI publication.

### 1.1. README badges and update policy

All seven badges in [README.md](../README.md) are rendered by Shields.io. However, their data sources and update conditions differ: the coverage and Databricks badges use project-generated JSON endpoints, while the other dynamic badges obtain information directly from external services. The repository link badge is static.

| Badge | Data source | Update condition |
| --- | --- | --- |
| PyPI Version | Official PyPI project JSON API, `info.version` | Changes when a published version becomes the version reported by PyPI and Shields.io refreshes its cached data. |
| Python Versions | Python version classifiers in the official PyPI project metadata | Changes when a published release supplies different Python version classifiers and Shields.io refreshes its cached data. |
| License | License metadata detected by GitHub | Changes when GitHub detects a different repository license and Shields.io refreshes its cached data. |
| GitHub Repository | Static Shields.io badge URL | Does not require an automatic update. Its text, color, and logo are defined in the README. |
| CI Status | GitHub Actions workflow status for `ci.yml`, filtered by `master` | Reflects the workflow status reported by GitHub after the cached badge is refreshed. The badge is not restricted to the `push` event. |
| Code Coverage | GitHub Pages, `badges/coverage.json` | Regenerated from the designated CI coverage evidence and exposed after an authorized Pages deployment. |
| Databricks LTS ML | GitHub Pages, `badges/databricks-lts-ml.json` | Regenerated from the current CI compatibility evidence and exposed after an authorized Pages deployment. |

#### Project-generated badges

The coverage and Databricks badges are produced by [generate_badges.py](../.github/scripts/generate_badges.py) and [pages.py](../.github/scripts/pages.py). On a non-cancelled push-triggered CI run on `master`, the Pages build consolidates evidence from the current workflow attempt, generates the endpoint JSON files, and includes them in the complete Pages site. The deployment guard requires the current `master` HEAD, the latest push-triggered CI run, and the matching run and attempt identifiers. Only an authorized, successful deployment replaces the publicly hosted endpoint data.

The Pages build can run even when the CI Gate fails. If the gate or a required badge-producing job fails, both custom badges report `failed` (red); if approval is unavailable without an explicit failure, they report `not verified` (light grey). Missing or invalid evidence cannot be reused from a previous successful run.

When the evidence is approved, the coverage badge displays the coverage percentage rounded to a whole number. Its color is based on the unrounded percentage: red below 75%, yellow from 75% up to but excluding 80%, and green at or above 80%. These thresholds affect presentation only, not CI acceptance. The Databricks badge displays the verified LTS ML runtime versions in blue. An individual runtime is listed only if its compatibility evidence is approved; see Section 2 for the distinction between runtime-equivalent dependency environments and actual Databricks clusters.

The current published data and its execution provenance can be inspected directly:

- [Coverage endpoint](https://renard162.github.io/tyrannis_optimizer/badges/coverage.json)
- [Databricks LTS ML endpoint](https://renard162.github.io/tyrannis_optimizer/badges/databricks-lts-ml.json)
- [Evidence and provenance](https://renard162.github.io/tyrannis_optimizer/badges/provenance.json)

Pull request CI runs, manual CI dispatches, CD-1, and CD-2 do not deploy these endpoints. If a Pages deployment is skipped, rejected, or unsuccessful, the last successfully deployed data remains available until another deployment replaces it. Shields.io renders the JSON data when the endpoint badge is requested and its cached data is refreshed; a successful Pages deployment does not guarantee an immediate visual update.

#### Standard badges and caching

The PyPI Version and Python Versions badges query the [official PyPI project JSON API](https://pypi.org/pypi/tyrannis/json). They do not read the repository's current `pyproject.toml`, Git tags, GitHub Releases, or TestPyPI metadata. Updating a version or Python compatibility declaration in the source code does not change these badges until the relevant metadata is published to PyPI and retrieved by Shields.io. The Python Versions badge uses published `Programming Language :: Python :: X.Y` classifiers rather than deriving supported versions from `requires-python` or the CI matrix.

The public Shields.io implementation currently defines the following service-specific cache lengths:

| Standard badge | Configured Shields.io cache length |
| --- | --- |
| PyPI Version | 21,600 seconds (6 hours) |
| Python Versions | 43,200 seconds (12 hours) |
| CI Status | 60 seconds (1 minute) |

These values are taken from the public Shields.io service implementations ([PyPI Version](https://github.com/badges/shields/blob/master/services/pypi/pypi-version.service.js), [Python Versions](https://github.com/badges/shields/blob/master/services/pypi/pypi-python-versions.service.js), and [CI Status](https://github.com/badges/shields/blob/master/services/github/github-actions-workflow-status.service.js)). They are external cache settings, not scheduled Tyrannis jobs or guaranteed refresh deadlines measured from package publication. The other dynamic badges are also subject to the service's applicable cache policies.

The README does not specify a custom `cacheSeconds` value, and the project-generated JSON endpoints do not specify one either. Shields.io applies its own defaults and minimum cache lifetimes. Setting `cacheSeconds` below the applicable default does not force faster updates. Additional caching by the [PyPI CDN](https://docs.pypi.org/api/#caching), GitHub Pages delivery infrastructure, GitHub's [Camo image proxy](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/about-anonymized-urls), and the browser may further delay the displayed image.

If a badge appears outdated, first inspect its authoritative data source. For PyPI badges, compare `info.version` and `info.classifiers` in the project JSON response with the expected published metadata. For custom badges, inspect the endpoint JSON and `provenance.json` to confirm the deployed commit and workflow attempt. Then open the Shields.io image URL directly, independently of the README. If the source data is correct but the displayed badge is stale, allow the external caches to refresh; another CI run or package publication is not required solely to refresh the image. A stale badge does not establish that a package publication or CI run has failed. See the Shields.io [badge parameter documentation](https://shields.io/badges/py-pi-version) and [endpoint badge documentation](https://shields.io/badges/endpoint-badge) for the external caching rules.

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
| Core Tests | All required executions succeed across the configured operating system and Python version matrix. The designated coverage execution produces nonempty XML and JSON reports and records its execution evidence. |
| Cache Tests | The required execution succeeds and produces a valid, nonempty JUnit report with no skipped, failed, or erroneous cases. |
| Spark Tests | The required execution succeeds and produces a valid, nonempty JUnit report with no skipped, failed, or erroneous cases. |
| Databricks LTS ML Tests | Each configured runtime-equivalent dependency environment passes its compatibility checks and required test execution, with valid evidence. The environments reproduce the selected runtime dependencies; they are not actual Databricks cluster executions. |
| MPI Tests | The environment provides the required MPI thread support, the execution succeeds, and its JUnit report contains executed cases with no skips, failures, or errors. |
| Package Build | Locked base dependencies can be installed; exactly one wheel and one source distribution are built; strict metadata validation succeeds; both distributions can be installed and pass independent smoke checks outside the source checkout. |

The gate requires exactly seven upstream job groups, each with a `success` result. A failed, skipped, or cancelled required group does not satisfy this condition.

Coverage is a reporting metric, not a minimum acceptance threshold. The no-skip requirement enforced by the JUnit validator applies to the designated optional compatibility executions; it is not a blanket prohibition on individual skips in every CI suite.

### 2.3. GitHub Pages and badges

On pushes to `master`, CI also composes and deploys the GitHub Pages site, including coverage and Databricks compatibility indicators. Evidence is associated with the current workflow attempt, and obsolete deployments are rejected by the deployment guard.

Pages jobs are not dependencies of `CI Gate`. They belong to the overall CI workflow, however, so a failure that makes the workflow conclusion unsuccessful also prevents that run from qualifying for package publication. Pull request and manual CI runs do not deploy the Pages site.

### 2.4. CI approval for package publication

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

