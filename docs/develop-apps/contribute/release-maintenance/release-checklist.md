---
title: Release Checklist
description: Release-gate policy and reproducible release steps
sidebar_position: 1
slug: /develop-apps/contribute/release-checklist
---

# Release Checklist

This document is the authoritative release gate policy.

## Rolling documentation versions

Update `website/release-data/current.json` once for the release, after updating
`deps/manifest.json`. Run `npm --prefix website run sync:release-readmes` and commit
the generated concrete README command. Run `npm --prefix website run check:releases`
and `./build.sh --doc` to validate tokens, historical exceptions, localization,
and rendered commands. Maintain the compatibility page manually with concrete versions. Keep historical
release notes pinned. Release-data-only changes do not require translation or source-hash updates.
Field ownership and token rules are documented in `website/release-data/README.md`.

## Release-Blocking Conditions

A release is blocked unless all conditions below are true:

1. No merge conflict markers in tracked source/doc files.
2. Naming hygiene checks pass for public docs.
3. Configure/build sanity passes (`cmake -S . -B ...` and `cmake --build ... --target sima_neat`).
4. Docs link checks pass in strict mode (`DOCS_STRICT_LINKS=1`).
5. Working tree is clean after generation steps.
6. Zero unresolved crash/correctness failures before push and on release refs.
7. Crash/correctness/stress/sanitizer gates are green on the release ref.
8. Soak stability lane is green for release tags (`soak-weekly`).
9. Fuzz lane is green for release candidates (`fuzz-nightly`).
10. Zero-skip gate is green (`zero-skip-gate`) for strict test lanes.
11. Required governance files are present and valid:
   - `.github/CODEOWNERS`
   - `.github/PULL_REQUEST_TEMPLATE.md`
   - `CONTRIBUTING.md`
   - `docs/develop-apps/contribute/release-checklist.md`
12. Release metadata is complete:
   - `project(SimaNeat VERSION x.y.z)` updated in `CMakeLists.txt`
   - `package-version` and `platform-version` updated in `deps/manifest.json` when required
   - `modelzoo-version` explicitly selects the validated Model Zoo release when it differs from
     `platform-version`; if omitted, Model Zoo resolution defaults to `platform-version`
   - `abi-version` in `deps/manifest.json` is incremented only when a public C++ type layout or
     exported binary contract changes incompatibly and `develop` still uses the same ABI version
     as `main`. If `develop` already has a higher ABI version for the upcoming release, keep it
     for further unreleased ABI breaks. Rebuild all C++ applications and Python bindings against
     matching headers and libraries after incompatible changes.
   - release notes prepared in the release/tag body

No "known crashers" list is allowed in release flow. Any crash regression blocks release until fixed.

## Required Status Checks

The following checks are required on release PRs and release tags:

- `configure-build-sanity`
- `docs-link-check`
- `crash-correctness-gate`
- `zero-skip-gate`
- `soak-weekly` (required for release tags)
- `fuzz-nightly` (required for release candidates)
- `stress-gate`
- `asan-ubsan-gate`

These checks are implemented in:

- `.github/workflows/vulcan-ci.yml`
- `.github/workflows/test-crash-correctness-nightly.yml`
- `.github/workflows/test-soak-weekly.yml`
- `.github/workflows/long-tests-weekly.yml`
- `.github/workflows/vulcan-fuzz-nightly.yml`
- `.github/workflows/test-stress-nightly.yml`
- `.github/workflows/sanitizers.yml`

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

1. Open **Actions → Release → Run workflow** to create a release candidate. For example, to release 0.7.0, set `release_line` to `0.7` and `tag` to `0.7.0-rc1`, keep `main` as `source_branch`, turn on `prerelease` and turn off `latest`. The workflow creates or reuses `release-0.7`, tags `v0.7.0-rc1` and drafts a prerelease. If `release-0.7` already exists, for example for 0.7.1, the workflow tags its current head, because `branch_exists_behavior` defaults to `reuse`. Merge `main` into `release-0.7` through a pull request before you create the first candidate.
2. Freeze non-release PR merges.
3. Validate the candidate. If it needs fixes, land them on `release-0.7` and run the workflow again with the next candidate tag, for example `0.7.0-rc2`.
4. When a candidate passes, check that `release-0.7` still points at it. The workflow tags the head of `release-0.7` and cannot tag an older commit. After `git fetch --tags origin`, `git rev-parse origin/release-0.7` must print the same commit as `git rev-parse v0.7.0-rcN^{commit}`. Here `rcN` is the candidate that passed. If anything landed after the candidate, create a new candidate first. Then run the workflow with `tag` set to `0.7.0`, `prerelease` off and `latest` on. It tags `v0.7.0` on that commit and drafts the final release. This run also requires every issue with Target Release `core-0.7.0` to be Done, then sets its Release Status to `released`. Candidate runs skip this check.
5. Fast-forward merge release branch back to `main`.
6. Publish release notes and post-release follow-up issues.

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
  - `runtime_session_sync_rgb`
  - `runtime_session_async_rgb`
  - `runtime_graph_fanout`
  - `runtime_graph_join_bundle`
  - `runtime_codec_mjpeg_decode`
  - `runtime_codec_h264_decode`
  - `runtime_codec_h265_decode`
  - `runtime_model_archive_load`
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
