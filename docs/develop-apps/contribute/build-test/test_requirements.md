---
title: Test Requirements
description: Minimum testing expectations for SiMa.ai Neat contributions
sidebar_position: 2
slug: /develop-apps/contribute/test_requirements
---

# Test Requirements

Every behavioral change should include tests that prove correctness and preserve debuggability.

## Baseline expectations

For most feature work, add or update tests that cover:

- Pipeline build correctness (expected fragments/string shape).
- Parse/validate behavior (caps negotiation and validation failures).
- Runtime behavior (`run`, push/pull, and teardown paths as applicable).
- Diagnostics quality (`PipelineReport` usefulness on failure paths).

## Required test types by change

### New node or node-group

- Unit-level checks for deterministic fragment generation.
- Validation/parse coverage for caps or linking assumptions.
- At least one integration path proving the node works in a realistic chain.

### Runtime/pipeline orchestration changes

- Success-path test for expected output behavior.
- Failure-path test for timeout, plugin absence, or invalid graph conditions.
- Teardown/lifecycle safety assertions when state handling changes.

### Public API signature changes

- Add or update compile-time coverage that exercises changed public headers/usages.
- For non-breaking extensions, verify existing usage still compiles.
- For approved breaking changes, include migration-focused tests/examples that validate the replacement API path.

### Python binding changes (`python/`, `pyneat`)

- Add/update `pytest` coverage under `python/tests`.
- Cover interop contracts (NumPy/PyTorch DLPack paths, copy-vs-zero-copy behavior).
- Add at least one import/smoke test against an installed wheel or editable install.

### Diagnostics or observability changes

- Tests for report fields added/changed.
- Concurrency-safe behavior when data is updated from streaming threads.

## Regression and determinism policy

- Fixes for bugs must include a regression test.
- Prefer tests that assert deterministic naming and stable pipeline generation.
- If output is intentionally non-deterministic, document why and constrain assertions to stable invariants.

## CPU cost in throughput tests

`async_yolov8_tput_test` reports process-wide user, system and total CPU
microseconds per completed frame, plus `average_busy_cores`. The latter is CPU
time divided by wall time, not a percentage of the whole board or accelerator
utilization. It can exceed one when multiple CPU threads work concurrently.

Its `cpu_window_fps` uses the same window as the CPU counters: before the first
measured submission through release of the final output. Warmup, result
verification, reporting and teardown are excluded. The existing `avg_fps`
window and throughput gate are unchanged. `output_extraction_included` and
`output_validation` identify which work the test performed; compare equivalent
settings. Incomplete or numerically failed runs report `cpu_window_complete=0`
without CPU/frame values. A throughput-gate failure still reports measurements
for otherwise successfully completed frames.

New benchmarks can reuse `tests/perf/process_cpu_metrics.h` for two process-wide
snapshots per run. Do not sample CPU counters per frame or add a monitoring
thread merely to calculate CPU/frame.

## Skip policy

- Treat skip paths as exceptions, not normal control flow.
- Strict tests (default) must fail when they cannot execute due missing runtime/tooling/fixtures.
- Only tests explicitly labeled `long` may use skip semantics (`return 77`) and those run in weekly lanes.
- New test additions should not introduce `skip_test(...)` in strict paths.

## Practical commands

Use the project build entrypoint:

```bash
./build.sh --all
```

For docs-impacting changes, also run:

```bash
./build.sh --doc
```

See [Build](/develop-apps/contribute/build) for all supported build/test modes.

## Contributor checklist

Before opening or merging a PR:

- Added/updated tests match the behavioral surface changed.
- Existing related tests still pass.
- Docs are updated for user-visible behavior.
- API/architecture changes are reflected in [Architecture](/develop-apps/contribute/architecture).
- Public API changes comply with the API compatibility policy in [Coding Standard](/develop-apps/contribute/coding_standard).

## Required coverage and capability profiles

Canonical CTest lanes record their exact selection before execution and compare
it with fresh JUnit outcomes. Empty selections, missing or duplicate results,
skips, disabled tests and failed execution fail required coverage. Test selectors,
arguments and deadlines remain owned by the lane—not the accounting tool.

A non-long registration cannot escape through `SKIP_RETURN_CODE`, a skip regex
or long-only environment. A required lane may also select existing long tests
with their canonical return-77 registration, but an actual skip still fails.
Fast retains its long-labeled stress tests; they are not silently deselected.
Source-level skip-reference counts are advisory, not counts of skipped tests.

`cmake/NeatGuardedTests.cmake` registers tests and tutorials through the existing
hardware supervisor when `SIMANEAT_GUARDED_HARDWARE_TESTS=ON`. This does not create
hardware admission. Guarded qualification is opt-in; standard CI uses direct
execution without a supervisor socket. Acquire the configured device ownership
and admission before running guarded native lanes. The result validator checks
registration policy; it does not replace the supervisor or authorize hardware
execution.

Python lanes record selected and explicitly deselected node IDs. Required setup,
call and teardown outcomes must all pass. Configure optional coverage before
selection:

| Profile | Configuration | Requirement |
| --- | --- | --- |
| Python runtime | `NEAT_CI_PYTHON_PROFILE=runtime` (default) | Enables Graph Phase 3 with the prepared ResNet fixture; Torch is explicitly excluded. |
| Torch integration | `NEAT_CI_PYTHON_PROFILE=torch` | Set reviewed exact `NEAT_CI_TORCH_VERSION`; installation/import must succeed. |
| MIPI camera | `SIMA_NEAT_TEST_MIPI_CAMERA=1` | Provide camera hardware, `SIMA_NEAT_TUTORIAL_MODEL_MIPI` and `SIMA_NEAT_TUTORIAL_MIPI_DECODE`. |
| Weekly HTTP streams | `NEAT_CI_STREAM_PROFILE=http` (default) | Provide HTTP MJPEG URL(s) and positive FPS. Both HTTP cases and the synchronous smoke check remain required. |
| Weekly HTTP + RTSP streams | `NEAT_CI_STREAM_PROFILE=all` | Also provide H264, H265 and MJPEG RTSP URL(s), plus positive MJPEG FPS. |

The hardware-free camera API test is always part of the runtime profile. Camera
and Torch exclusions remain coverage gaps, not passes. Stream profiles cover the
`codec_stream` case registrations; they do not qualify other excluded RTSP lanes.
Each stream case has separate smoke and repeated-contract results, so an available
HTTP stream cannot hide an unexecuted RTSP case under an aggregate PASS.

For weekly model preparation, set a local path or HTTPS location and reviewed
SHA-256 for each of `SIMA_CI_MNIST_MODEL`, `SIMA_CI_SSD300_MODEL`,
`SIMA_CI_MNIST_IMAGES` and `SIMA_CI_MNIST_LABELS`. Each checksum uses the matching
`_SHA256` variable. The dataset inputs are compressed MNIST IDX test files.
Preparation checks exact bytes and stages the existing tests' cache/variables;
missing or mismatched inputs are preparation failures. It never modifies an
unrelated existing cache. Model preparation does not depend on RTSP availability.

Keep selection, outcomes and profile-exclusion reports with the lane results.
A selected-profile pass is not a claim that all optional hardware or integration
profiles passed.

### Local weekly preparation

Use `tools/ci/prepare_weekly_fixtures.py` for local runs as well as CI. By default,
preparation requires both `mnist` and `ssd300`. To qualify only MNIST, pass
`--fixture-group mnist` explicitly; the report identifies SSD300 as unqualified.
This selection does not change CTest selection or make a skipped required case
pass. Full weekly qualification still requires both groups.

Use `--environment-json <file>` for a local launcher, `--github-env <file>` for
GitHub Actions, or both. Both outputs contain the same environment mapping.
The local launcher must merge the JSON mapping into the test process environment
before recording selection or admitting a hardware process. The JSON is not a
shell script and is not a test result. Merely creating it does not export variables.

The helper exports `SIMA_RUN_MNIST_PREPROC_CHANNELS1=1` only after the selected
model and MNIST gzip/IDX files pass validation. It checks gzip CRC, IDX geometry,
length, image/label counts and label range, and rejects conflicting compressed or
raw cache entries. Use a new output directory for each preparation attempt;
failed staged bytes remain available for diagnosis.

### Installed packages versus private-prefix integration

Keep these qualification scopes separate:

- **Private-prefix integration** checks the selected headers, libraries and
  plugins without modifying the global package database. It does not qualify
  a Debian installation.
- **Installed-package qualification** checks the actual DevKit root filesystem,
  `dpkg` records, payload bytes, symlinks and file ownership against the selected
  verified archives. A private prefix cannot satisfy this check.

Continue to validate distribution composition with
`scripts/ci/verify_core_dist.sh`, including independently selected
`--expected-manifest`, `--expected-receipt` and `--expected-deb-dir` inputs when
building locally. Use the complete producer distribution, including its five
metadata variants. Remote `sima-cli` installation stages the selected resources
but does not retain those metadata files. The CI workflow downloads this run's
complete `neat-core-dist` artifact separately, then compares the actual
CLI-installed packages against that independent input. Do not regenerate
producer metadata from the installed machine.

Select the required package profile explicitly:

| Profile | Required roles |
| --- | --- |
| `core` | Core runtime/development, Kernels development, Runner, Scheduler and the runtime compatibility package |
| `core-codecs` | `core` plus the Codecs runtime/development pair |
| `core-codecs-llima` | `core-codecs` plus the LLiMa runtime/CLI pair |

Consumed optional backend providers remain required by the selected producer
receipts. A profile is not inferred from missing archives. LLiMa and the retired
memory implementation are not unconditional dependencies of Core-only tests.

On the DevKit, capture `--capture-platform-baseline <file>` with
`tools/ci/verify_core_dist.py` **before** using the existing installer. After
installation, run the distribution verifier with `--package-profile <profile>`,
`--verify-installed`, `--platform-baseline <file>` and
`--write-package-inventory <file>`. The baseline records which tracked native
packages are installed or absent. It preserves that state, unrelated native
package versions, and whether the OTA command is available. When present, its
canonical path, ownership and bytes must remain unchanged. Historical package
names and the OTA command need not exist on every image; Palette remains
required. A provider explicitly present in the
verified distribution may be upgraded and is checked against its selected
archive instead.

Pass the absolute inventory path as `SIMA_TEST_PACKAGE_INVENTORY` and the
pre-install baseline path as `SIMA_TEST_PLATFORM_BASELINE` to the existing
`devkit-package-inventory` CTest lane. Do not recapture the baseline after
installation. The selected inventory contains expected archive data, not an
installation PASS. The native test independently checks actual package
versions/architectures, loaded Core/library paths and ownership, unchanged
unselected native package presence or absence, retired-package absence, and OTA
availability and identity. Continue to compile the installed-header examples
separately; package presence is not proof that a
consumer compiles or that a backend executed correctly.

### Native build-tooling prerequisites

On Ubuntu, install `build-essential`, `file`, `nlohmann-json3-dev` and
`python3-venv` before running the build-tooling pytest selection from
`.github/workflows/vulcan-ci.yml`. The platform-baseline probe compiles its real
shared C++ contract with `-Werror`; a missing compiler/header is a test setup
failure, not a skipped check. Compiler stdout/stderr are included on failure, and
`file` verifies the native executable before it runs.
