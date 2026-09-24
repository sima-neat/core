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

## Decoder streaming qualification

`codec_runtime_stream_test --case NAME` requires that case's URL and source-FPS
fixture configuration; missing selected fixtures fail. Running without `--case`
retains discovery of configured live streams and skips unavailable cases.

Decoded cases check increasing, present PTS within each run. `--determinism`
compares stable output contracts across reconnects, since a live camera does not
restart at the same image. Use `--replay-content --repeat 2 --case NAME` only with
a controlled server that restarts the same decoded sequence and relative PTS on
each connection. This also compares visible NV12 pixel hashes and PTS relative
to the first frame, excluding stride padding and connection start-time offsets.
Do not report a contract-only run as pixel determinism.

`codec_decode_accuracy_test` also sends 30 encoded `Input`/`Sample` frames through
public `SimaDecode` for H.264, H.265 and MJPEG. It compares automatic allocation,
low-latency tuning with automatic counts, explicit input/output counts, and
low-latency tuning with explicit counts. Including each codec's Owned control,
this covers 15 option runs and 450 submitted frames. It
checks exact output count and PTS, visible pixels against the automatic control,
Owned output, and output retention. Roomier configurations retain the first
output through later frames and shutdown. Automatic low-latency configurations
hold the first output for 250 ms, verify its pixels, release it before the next
pull, and retain the final output through shutdown. The automatic two-output
JPEG pool cannot sustain arbitrary downstream retention: GStreamer appsink also
holds its most recently pulled buffer, even with `enable-last-sample=false`.
Keeping an older sample while waiting for further output can occupy both slots;
consumers needing that overlap must provide a larger explicit output count.
Its generated video fixtures have no B frames; it does not qualify low-latency
tuning on streams that require reordering. DMA and descriptor cleanup accounting
must be collected by the board runner after retained samples are released.

## Contributor checklist

Before opening or merging a PR:

- Added/updated tests match the behavioral surface changed.
- Existing related tests still pass.
- Docs are updated for user-visible behavior.
- API/architecture changes are reflected in [Architecture](/develop-apps/contribute/architecture).
- Public API changes comply with the API compatibility policy in [Coding Standard](/develop-apps/contribute/coding_standard).
