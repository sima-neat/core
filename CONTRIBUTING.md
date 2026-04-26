# Contributing to SiMa Neat

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

- Public API must use canonical naming from `docs/contribute/naming.md`.
- C++ namespace is `simaai::neat`.
- Core runtime types are `Model`, `Session`, and `Run`.
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
- MPK/security changes:
  - `bash scripts/ci/run_mpk_security_gate.sh`
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

## Commit and PR Standards

- Keep commits focused and scoped.
- Use imperative subject lines (for example: `Add release gate policy checks`).
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
3. Update `docs/how-to/migration_legacy_names.md` or a dedicated migration page.
4. Update examples/tutorials to canonical replacements.
5. Include downstream migration impact in the PR template.

## Release Hygiene

Releases must follow `docs/release-checklist.md` and pass all required checks in `.github/workflows/release-gate.yml`, including MPK security, install smoke, perf regression, soak, and fuzz gates.
