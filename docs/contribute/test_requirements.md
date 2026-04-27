---
title: Test Requirements
description: Minimum testing expectations for SiMa Neat contributions
sidebar_position: 3
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

See [Build](../getting-started/build) for all supported build/test modes.

## Contributor checklist

Before opening or merging a PR:

- Added/updated tests match the behavioral surface changed.
- Existing related tests still pass.
- Docs are updated for user-visible behavior.
- API/architecture changes are reflected in [Architecture](./architecture).
- Public API changes comply with the API compatibility policy in [Coding Standard](./coding_standard).
