---
title: Release Checklist
description: Release-gate policy and reproducible release steps
sidebar_position: 1
---

# Release Checklist

This document is the authoritative release gate policy.

## Release-Blocking Conditions

A release is blocked unless all conditions below are true:

1. No merge conflict markers in tracked source/doc files.
2. Naming hygiene checks pass for public docs.
3. Configure/build sanity passes (`cmake -S . -B ...` and `cmake --build ... --target sima_neat`).
4. Docs link checks pass in strict mode (`DOCS_STRICT_LINKS=1`).
5. Working tree is clean after generation steps.
6. Zero unresolved crash/correctness failures before push and on release refs.
7. Crash/correctness/stress/sanitizer gates are green on the release ref.
8. Model-archive security gate is green (`model-archive-security-gate`).
9. Install smoke gate is green (`install-smoke`).
10. Performance regression gate is green (`perf-regression-gate`).
11. Soak stability lane is green for release tags (`soak-weekly`).
12. Fuzz lane is green for release candidates (`fuzz-nightly`).
13. Zero-skip gate is green (`zero-skip-gate`) for strict test lanes.
14. Required governance files are present and valid:
   - `.github/CODEOWNERS`
   - `.github/PULL_REQUEST_TEMPLATE.md`
   - `CONTRIBUTING.md`
   - `docs/contribute/release-checklist.md`
15. Release metadata is complete:
   - `project(SimaNeat VERSION x.y.z)` updated in `CMakeLists.txt`
   - `CHANGELOG.md` has `## [x.y.z]` entry
   - release notes prepared in the release/tag body

No "known crashers" list is allowed in release flow. Any crash regression blocks release until fixed.

## Required Status Checks

The following checks are required on release PRs and release tags:

- `repo-hygiene`
- `configure-build-sanity`
- `docs-link-check`
- `crash-correctness-gate`
- `model-archive-security-gate`
- `install-smoke`
- `perf-regression-gate`
- `zero-skip-gate`
- `soak-weekly` (required for release tags)
- `fuzz-nightly` (required for release candidates)
- `stress-gate`
- `asan-ubsan-gate`
- `release-policy-check`

These checks are implemented in:

- `.github/workflows/release-gate.yml`
- `.github/workflows/test-crash-correctness-nightly.yml`
- `.github/workflows/model-archive-security.yml`
- `.github/workflows/install-smoke.yml`
- `.github/workflows/perf-regression.yml`
- `.github/workflows/zero-skip.yml`
- `.github/workflows/test-soak-weekly.yml`
- `.github/workflows/long-tests-weekly.yml`
- `.github/workflows/vulcan-fuzz-nightly.yml`
- `.github/workflows/test-stress-nightly.yml`
- `.github/workflows/sanitizers.yml`

Trigger ownership to avoid duplicate gate execution:

- Non-release PRs into `main` run `model-archive-security`, `install-smoke`, `perf-regression`, and `zero-skip` from their standalone workflows.
- Release PRs (`release/*` head refs) and release refs (`release/**`, `v*`) run those same lanes from `.github/workflows/release-gate.yml`.

## GitHub Branch and Tag Protection

Configure GitHub repository settings:

1. Protect `main`:
   - Require pull request before merge.
   - Require at least one code-owner approval (two recommended when available).
   - Dismiss stale approvals on new commits.
   - Require all required status checks.
   - Disallow force pushes.
   - Use squash-only or linear history.
2. Protect `v*` tags to restrict who can create release tags.

## Release Flow

1. Cut `release/x.y.z` from green `main`.
2. Freeze non-release PR merges.
3. Run release gate workflow on release branch.
4. Create `vX.Y.Z-rcN` tag(s) for candidate validation.
5. Promote to final `vX.Y.Z` tag.
6. Fast-forward merge release branch back to `main`.
7. Publish release notes and post-release follow-up issues.

## Operational Notes

- No release from dirty branches.
- No release from unreviewed code.
- No release when required checks are red.
- No push allowed when local crash/correctness gate fails.
- No manual bypass path for hygiene failures.

## Perf Regression Contract

- Perf gate entrypoint is `scripts/ci/run_perf_regression_gate.sh`.
- Baselines are profile-scoped under `tests/perf/baselines/v2/modalix_default/`:
  - `profile.json` defines the fixed Modalix environment contract.
  - one scenario file per scenario ID (`<scenario_id>.json`).
- Required scenarios:
  - `runtime_graph_sync_rgb`
  - `runtime_graph_async_rgb`
  - `runtime_graph_fanout`
  - `runtime_graph_join_bundle`
- Every perf run publishes per-scenario result files in `build-perf-gate/perf_results/`.
- Each result must include:
  - `scenario_id`
  - `modalix_profile_id`
  - `status`
  - `failure_class`
  - `reason_code`
  - `metrics`
  - `run_meta`
  - `timestamp`
- Any `REGRESSION`, `HARNESS_ERROR`, or `ENV_BROKEN` classification blocks the lane.
