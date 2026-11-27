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

Release with the final tag, for example `v0.7.0`. Do not create release-candidate tags. Major and
patch releases use the same workflow, **Actions → Release**. It creates `release-X.Y` from `main`
for a new line or reuses an existing `release-X.Y`, tags the head of that branch and drafts the
GitHub release. Its first job checks that `package-version` at that commit equals `tag`. It requires
every issue with Target Release `core-X.Y.Z` to be Done, then sets their Release Status to
`released`.

### Major Release

This example releases `v0.7.0` on a new line.

1. On `develop`, set `package-version` in `deps/manifest.json` to `0.7.0`. Update `platform-version`, `modelzoo-version` and `abi-version` when the Release-Blocking Conditions require it, and update the release data as described in Rolling documentation versions.
2. Merge `develop` into `main` through a pull request.
3. Run `git fetch origin`. The commit to tag is `git rev-parse origin/main`. Do not merge anything into `main` until the tag exists.
4. Run the checks in Before Tagging on that commit.
5. Open **Actions → Release → Run workflow**. Set `release_line` to `0.7` and `tag` to `0.7.0`, keep `main` as `source_branch`, leave `prerelease` off and `latest` on. The workflow creates `release-0.7` from `main`, tags `v0.7.0` and drafts the release.
6. Continue with After Tagging and Publishing.

### Patch Release

This example releases `v0.6.3` on the existing `release-0.6`.

1. Prepare `release-0.6`:
   - If every change that would reach `release-0.6` by merging `develop` through `main` is intended for `0.6.3`, use that merge path: set `package-version` to `0.6.3` on `develop` as in step 1 of the major release, merge `develop` into `main`, then merge `main` into `release-0.6` through a pull request.
   - Otherwise, branch from `release-0.6`, cherry-pick only the intended fixes with `git cherry-pick -x`, set `package-version` to `0.6.3`, and open a pull request back to `release-0.6`.
2. Run `git fetch origin`. The commit to tag is `git rev-parse origin/release-0.6`. Do not merge anything into `release-0.6` until the tag exists.
3. Run the checks in Before Tagging on that commit.
4. Open **Actions → Release → Run workflow**. Set `release_line` to `0.6` and `tag` to `0.6.3`, keep `main` as `source_branch` and leave `prerelease` off. Turn `latest` off when a newer line is already published. The workflow reuses `release-0.6`, because `branch_exists_behavior` defaults to `reuse`, tags its head `v0.6.3` and drafts the release.
5. Continue with After Tagging and Publishing.

### Common Steps

`<commit>` is the commit to tag from step 3 of the major release or step 2 of the patch release.
The commands use `0.7.0`; for a patch release use its version, for example `0.6.3`.

#### Before Tagging

1. Check the version. `git show <commit>:deps/manifest.json` must show `"package-version": "0.7.0"`. If it does not, the Release workflow stops in its first job, before it creates the branch or the tag.
2. Check that standard Vulcan CI passed on that exact commit. `gh run list --repo sima-neat/core --workflow vulcan-ci.yml --commit <commit>` lists its runs, and an existing successful run on `main` or on the release branch can be reused. If there is none, start one with `gh workflow run vulcan-ci.yml --repo sima-neat/core --ref main`, or `--ref release-0.6` for a patch release. Check that the new run's `headSha` is the commit, then wait with `gh run watch <run-id> --repo sima-neat/core --exit-status`.
3. Check the issue gate. In the Neat Releases project (`https://github.com/orgs/sima-neat/projects/6`), at least one issue must have Target Release `core-0.7.0`, and every issue with that target must be Done. A Release run with `dry_run` on runs this check and the version check without pushing anything or changing issues.

#### After Tagging

1. Check that **Target SHA** in the Release run summary is the commit. If it is not, do not publish the draft release.
2. Start standard Vulcan CI on the tag with `gh workflow run vulcan-ci.yml --repo sima-neat/core --ref v0.7.0`. A tag pushed by the workflow starts no workflow run by itself. This run publishes the versioned packages that `sima-cli neat install core@v0.7.0` installs, and the offline bundles.
3. Wait with `gh run watch <run-id> --repo sima-neat/core --exit-status` until the whole run succeeds, including the job `Publish develop and release offline bundles`. `latest.tag` updates before that job starts, so it does not show that the run is done.

#### Publishing

1. Open the draft GitHub release, check the release notes and click **Publish release**.
2. Open post-release follow-up issues.

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
