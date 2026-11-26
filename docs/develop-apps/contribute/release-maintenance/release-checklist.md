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
7. Vulcan CI (`.github/workflows/vulcan-ci.yml`) passed on the exact commit being tagged.
8. The zero-skip gate (`scripts/ci/run_zero_skip_gate.sh`) passes in the Vulcan CI test lanes.
9. Required governance files are present and valid:
   - `.github/CODEOWNERS`
   - `.github/PULL_REQUEST_TEMPLATE.md`
   - `CONTRIBUTING.md`
   - `docs/develop-apps/contribute/release-checklist.md`
10. Release metadata is complete:
   - `package-version` in `deps/manifest.json` at the tagged commit equals the release version, for example `0.7.0` for `v0.7.0`
   - `platform-version` updated in `deps/manifest.json` when required
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

Before a release tag is created, Vulcan CI (`.github/workflows/vulcan-ci.yml`) must pass on the
exact commit being tagged. Step 5 of the Release Flow shows how to check it.

## Scheduled Test Lanes

These lanes are not release-blocking. Crash correctness, stress, sanitizers and fuzz run nightly.
Soak and long tests run weekly.

- `.github/workflows/test-crash-correctness-nightly.yml`
- `.github/workflows/test-stress-nightly.yml`
- `.github/workflows/sanitizers.yml`
- `.github/workflows/vulcan-fuzz-nightly.yml`
- `.github/workflows/test-soak-weekly.yml`
- `.github/workflows/long-tests-weekly.yml`

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

Release with the final tag, for example `v0.7.0`. Do not create release-candidate tags. The
Release workflow (**Actions → Release**) creates or reuses `release-X.Y`, tags its head and drafts
the GitHub release. The steps below release 0.7.0 on a new line. A patch release on an existing
line, for example 0.6.3, differs in steps 3 and 6.

1. On `develop`, set `package-version` in `deps/manifest.json` to `0.7.0`. Update `platform-version`, `modelzoo-version` and `abi-version` when the Release-Blocking Conditions require it, and update the release data as described in Rolling documentation versions.
2. Merge `develop` into `main` through a pull request.
3. Find the commit that will be tagged. Run `git fetch origin` first.
   - New line: the workflow creates `release-0.7` from the head of `main`. The commit is `git rev-parse origin/main`.
   - Existing line: merge `main` into `release-0.6` through a pull request. The workflow reuses `release-0.6` and tags its head, because `branch_exists_behavior` defaults to `reuse`. The commit is `git rev-parse origin/release-0.6`.

   Do not merge anything into that branch until the tag exists.
4. Check the version at that commit. `git show <commit>:deps/manifest.json` must show `"package-version": "0.7.0"`. The Release workflow runs the same check in its first job. If the versions differ, it stops before it creates the branch or the tag and before it marks issues released.
5. Check that Vulcan CI passed on that exact commit. `gh run list --repo sima-neat/core --workflow vulcan-ci.yml --commit <commit>` lists its runs. A successful run on `main` or on the release branch counts. If there is none, start Vulcan CI on the branch from step 3 with `gh workflow run vulcan-ci.yml --repo sima-neat/core --ref main`, or `--ref release-0.6` for a patch release. Check that the new run's `headSha` is the commit, then wait with `gh run watch <run-id> --repo sima-neat/core --exit-status`. Do not tag until a run on that commit has passed.
6. Open **Actions → Release → Run workflow**. Set `release_line` to `0.7` and `tag` to `0.7.0`, keep `main` as `source_branch`, leave `prerelease` off and `latest` on. For 0.6.3, set `release_line` to `0.6` and `tag` to `0.6.3`, and turn `latest` off if a newer line is already released. The workflow tags `v0.7.0` and drafts the release. It requires every issue with Target Release `core-0.7.0` to be Done, then sets its Release Status to `released`. Check that **Target SHA** in the run summary is the commit from step 3. If it is not, do not publish the draft release.
7. Publish the release notes in the draft release and open post-release follow-up issues.

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
