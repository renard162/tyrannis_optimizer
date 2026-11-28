# Changelog Guidelines

## Purpose

The root-level `CHANGELOG.md` provides a cumulative, human-readable history of notable changes to Tyrannis. It records what changed between public releases, focusing on the functionality, compatibility, correctness, and performance of the package from the user's perspective.

The changelog is not a commit log, implementation report, or usage guide. GitHub release notes summarize the highlights of each release and refer readers to the changelog for the complete list of relevant changes.

## Format

- Write all entries in English.
- List versions in reverse chronological order, with the newest version first.
- Use `## Version X.Y.Z` as the heading for each version.
- Use a single Markdown bullet per relevant change, prefixed with one of the flags defined below.
- Follow the exact entry format `- [Flag] : Description.`
- Keep each description short, specific, and self-contained. Use backticks for public API identifiers.
- Group entries with the same flag where practical. Do not add separate headings for flags or list unused flags.
- Preserve previous release entries when adding a new version.

## Change flags

| Flag | Use |
| --- | --- |
| `[Added]` | New features, algorithms, integrations, or public API components. |
| `[Changed]` | Changes to existing functionality, behavior, compatibility, or performance. |
| `[Deprecated]` | Existing public functionality marked for future removal. |
| `[Removed]` | Previously available functionality or public API components that are no longer available. |
| `[Fixed]` | Corrections to mathematical formulations, numerical behavior, or implementation defects. |
| `[Security]` | Corrections to security vulnerabilities. |

Use the flag that best describes the change from the user's perspective. A replacement of an existing public API may warrant separate `[Added]` and `[Removed]` entries.

## Content selection

| Include | Exclude |
| --- | --- |
| New user-facing features and capabilities. | Internal classes, methods, helpers, and implementation details. |
| Additions, changes, deprecations, and removals in the public API. | Refactoring that does not change observable behavior. |
| Mathematical, numerical, and implementation corrections. | Routine linting, formatting, and code organization changes. |
| Relevant changes to supported environments, dependencies, or compatibility. | CI/CD, build automation, and routine infrastructure maintenance. |
| Meaningful improvements in performance, memory consumption, or reliability. | Test additions and reorganizations without an independent user-facing change. |
| Security fixes. | Editorial and formatting changes to documentation. |

Record the observable outcome rather than the mechanism used to achieve it. Do not list several internal changes that collectively implement a single user-facing feature. Conversely, distinct capabilities introduced by the same implementation may deserve separate entries.

Identify affected public API components when useful, especially for changes that require users to adapt existing code. Do not include usage instructions, architectural explanations, or migration procedures; those belong in the relevant documentation.

## Examples

The following examples use entries from version 1.1.0:

```markdown
# Changelog

## Version 1.1.0

- [Added] : SciPy integration through `ScipyMinimizer`.
- [Added] : `OptimizationHistory`.
- [Added] : Pickle serialization and restoration of optimization history.
- [Removed] : Legacy history methods and properties from `Optimizer`.
- [Changed] : Reduced memory consumption of optimization history.
- [Fixed] : History decoding affecting the optimizer's random state.
- [Fixed] : Handling of signed infinities in CSV history exports.

## Version 1.0.1

- [Added] : Initial public release.
```

Version 1.0.1 is treated as the initial public release. Its existing functionality does not need to be enumerated retrospectively.

## Release checklist

Before publishing a new version:

1. Compare the release candidate with the preceding public release, considering the public API, algorithms, documented behavior, and supported environments.
2. Identify notable changes and verify them against the actual implementation. Include only changes delivered by the release.
3. Consolidate related changes, remove redundant entries, and omit internal details.
4. Add the new version section at the top of `CHANGELOG.md`, preserving the existing history.
5. Confirm that the version heading matches the version being released and that the entries use the defined flags and format.
6. Review the GitHub release notes against the changelog to ensure that the highlights are accurate.

## Publication and access

Maintain a single cumulative changelog in the root of the repository. The package metadata links to the changelog on the `master` branch:

```toml
[project.urls]
Changelog = "https://github.com/renard162/tyrannis_optimizer/blob/master/CHANGELOG.md"
```

The link remains unchanged across releases. Users of older versions can locate their version in the same document and consult changes introduced by later releases.
