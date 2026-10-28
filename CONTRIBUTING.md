# Contributing to SiMa NEAT

This repository follows a release-first workflow: `main` must stay releasable.

## Prerequisites

- Linux (Debian/Ubuntu) or macOS.
- CMake 3.16+.
- C++20 toolchain.
- GStreamer development packages.
- OpenCV 4 development packages.
- Node.js 20+ for docs builds.

Bootstrap dependencies:

```bash
./build.sh --install-deps-only
```

## Local Setup

```bash
# core library only
./build.sh --dev-only

# full build
./build.sh --all --clean --no-doc
```

## Coding Standards

- Public API must use canonical naming from `docs/develop-apps/contribute/start-here/naming.md`.
- C++ namespace is `simaai::neat`.
- Core runtime types are `Model`, `Graph`, and `Run`.
- Keep source headers/implementation paths coherent (no internal headers in `include/`).
- C++ target baseline is C++20.

Run formatting/hygiene checks locally before pushing:

```bash
bash scripts/check_format.sh --changed-only
bash scripts/check_cmake_format.sh --changed-only
bash scripts/check_duplicate_includes.sh --changed-only
```

## Test Tiers

Run the minimum tier required by your change:

- Runtime/library code (`src/`, `include/`):
  - `ctest --test-dir build/tests --output-on-failure`
  - `bash scripts/ci/run_crash_correctness_gate.sh`
- Examples/tutorial wiring:
  - `ctest --test-dir build/tutorials --output-on-failure`
- Docs or naming changes:
  - `bash scripts/ci/check_naming_and_conflicts.sh`
  - `DOCS_STRICT_LINKS=1 npm --prefix website run build`
- Repo hygiene / release policy changes:
  - `bash scripts/ci/check_artifacts.sh`
  - `bash scripts/ci/check_release_policy.sh`
- Model-archive/security changes:
  - `bash scripts/ci/run_model_archive_security_gate.sh`
- Packaging/install changes:
  - `bash scripts/ci/run_install_smoke.sh`
- Performance or long-run stability changes:
  - `bash scripts/ci/run_perf_regression_gate.sh`

Install local hooks once per clone:

```bash
bash scripts/dev/install_hooks.sh
```

Installed hooks:

- `pre-commit`: formatting/CMake/include-hygiene checks on changed files.
- `pre-push`: formatting/CMake/include-hygiene checks on changed files + crash/correctness gate.

## Crash and Correctness Policy

- `Crash/segfault/use-after-free/data race = immediate fix on current branch`.
- `No push if local crash/correctness gate fails`.
- `No merge if CI crash/correctness gate fails`.
- Do not defer crash fixes behind waivers, skip lists, or release exceptions.

## Issues

Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md) for defects
and the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md)
for proposed improvements. Complete the relevant sections and mark others as
not applicable. Include a minimal reproduction for bugs and an application use
case for features; redact secrets and customer data from examples and logs.

## Commit and PR Standards

- Keep commits focused and scoped.
- Use imperative subject lines (for example: `Add release gate policy checks`).
- Use the [pull request template](.github/PULL_REQUEST_TEMPLATE.md), target
  `develop`, and link completed issues with `Fixes #<issue>`.
- Every PR must include:
  - change type,
  - risk assessment,
  - test evidence,
  - docs impact,
  - migration impact.
- If public behavior changes, update docs in the same PR.

## API Deprecation Process

When changing public API in `include/*`:

1. Keep old symbol available for at least one release cycle when feasible.
2. Mark deprecations with explicit migration guidance.
3. Update a dedicated migration page under `docs/develop-apps/` when the change needs one.
4. Update examples/tutorials to canonical replacements.
5. Include downstream migration impact in the PR template.

## Release Hygiene

Releases must follow `docs/develop-apps/contribute/release-maintenance/release-checklist.md` and pass all required checks in `.github/workflows/release-gate.yml`, including model-archive security, install smoke, perf regression, soak, and fuzz gates.

Before merging `develop` into `main`, check that production docs will build. Production docs import each sibling repository's `main`, while push-triggered staging builds import their `develop`. Run the staging-only check on the commit you will promote:

```bash
gh workflow run "Vulcan Docs" --repo sima-neat/core --ref develop -f sibling_branch=main
```

A pass predicts the production docs build when the same Core commit is promoted, no sibling `main` changes before the production deploy (compare the commits in the `[autodoc] ... using` log lines), and `git diff --stat origin/develop...origin/main` prints nothing.

### Adaptive DevKit CI

The standard Vulcan workflow discovers online runners in the resolved DevKit
runner group after the candidate is ready. All workers require the exact firmware
and artifact environment. It uses up to five regular workers for fast/docs,
two E2E partitions, performance and Python/tutorials, plus up to two regular
`modalix-som-16gig` workers for C++ and Python GenAI. Default work does not impose
a RAM size. GenAI reuses writable `/media/nvme/llima/models` when available; fresh
workers cache models under `/media/nvme/kerrigan/neat-core-genai-models`. The model
root is created before taking its preparation lock; explicit `LLIMA_MODELS_PATH`
overrides remain supported. Standard runs consume no RTSP-pod runners.

Weekly external stream coverage adds one RTSP-pod worker when the declared stream
profile is `http` or `all`: HTTP MJPEG sources share that network-access boundary.
The `none` profile records its existing coverage exclusions on a regular worker.
Local RTSP servers and mocked RTSP APIs do not need source-pod access; see the
[test-to-runner audit](tests/devkit-runner-requirements.md).

GitHub has no negative-label selector. The planner uses full candidate label
conjunctions and checks all registered RTSP runners, including offline ones, to
prove that regular selectors cannot land on that pod. An optional repository
variable `CORE_UNREADY_RUNNERS` contains a JSON array of excluded runner names;
the selector must exclude those registrations too. Without a safe selector,
planning fails rather than silently falling back to RTSP hardware. An unavailable
idle pool queues one complete worker for that pool; no online compatible pool
fails explicitly. GitHub owns assignment and queueing: discovery is a snapshot,
not a reservation, and candidate names do not pin jobs to individual devices.

Every worker installs the same commit and verifies firmware, installed package
inventory and runtime. Existing CTest/pytest assertions, timeouts, hardware
isolation, weekly fixtures, capability profiles and explicit exclusions remain
unchanged. E2E partitions use recorded test durations, and their exact union must
match the original canonical selection. The `Test Core on DevKit` aggregate gate
requires every worker to pass before promotion, offline delivery or validation-only
completion. Offline-only and docs-only runs retain their intentional skips.

Plan-specific `core-selected-coverage-*` artifacts contain coverage and outcome
receipts; `core-devkit-shard-plan` records the fleet and assignments. Full reruns
create a new plan ID, excluding obsolete shards. Failed-job reruns retain that ID
and replace only re-executed shards' artifacts. Estimates exclude repeated
installation and downloads. Runtime and public APIs are unchanged.
