---
title: Coding Standard
description: Contributor coding standards for SiMa Neat
sidebar_position: 2
---

# Coding Standard

This guide defines contribution rules for code that lands in the SiMa Neat library.

## Language and API constraints

- Use **C++20**.
- Keep public API changes intentional and minimal (`include/*` is treated as stable).
- Prefer backward-compatible extensions over breaking changes.
- Keep internal implementation details out of installed/public headers.

## Formatting and style hygiene

- C/C++ formatting is enforced with `clang-format` (`.clang-format` in repo root).
- CMake style is enforced by `scripts/check_cmake_style.py`.
- Duplicate includes in C/C++ sources are forbidden.
- `.editorconfig` defines baseline whitespace rules (LF, final newline, no trailing spaces).

Run before pushing:

```bash
bash scripts/check_format.sh --changed-only
bash scripts/check_cmake_format.sh --changed-only
bash scripts/check_duplicate_includes.sh --changed-only
```

## API compatibility policy

Public API compatibility is a hard requirement for all installed headers under `include/*`.

- Non-breaking additions are preferred (new overloads, new optional fields, new APIs).
- Breaking signature changes (rename/remove/type change/parameter order change/behavioral contract break) must go through a review process before merge.
- If a breaking change is unavoidable, prefer a deprecation period first (keep old signature + add replacement path) before removal.

### Required process for breaking API signatures

Before merging a breaking API change:

1. Document the change in the PR description under a dedicated `Breaking API Change` section.
2. Include impact analysis: affected headers/symbols, expected downstream breakage, and migration steps.
3. Include versioning/release intent (when the break is allowed to ship).
4. Update docs and examples to the new API in the same change set.
5. Obtain explicit maintainer approval for the breaking API surface.

## Module boundaries

Keep dependency rules strict:

- `builder/` must not depend on GStreamer or `pipeline/`.
- `gst/` must not depend on `pipeline/`.
- `nodes/` must not depend on `pipeline/`.
- `pipeline/` is the orchestrator and may depend on `gst/`, `builder/`, `nodes/`, `contracts/`, `policy/`, and model internals.

## Determinism requirements

- Node output must be deterministic for equal inputs/config.
- Keep element names stable and reproducible.
- Preserve deterministic pipeline string generation where possible.
- When changing naming behavior, ensure diagnostics and validation still map elements to node ownership.

## Error handling and diagnostics

- Prefer structured failures with actionable context.
- Ensure new failure paths include enough detail for `PipelineReport` diagnostics.
- Avoid silent fallbacks that hide plugin/caps/runtime errors.
- Keep diagnostics thread-safe; probe-side updates must use atomics or equivalent lock-free primitives.

## Concurrency and lifecycle

- Never block indefinitely in teardown paths.
- Treat runtime state transitions defensively (`EOS`, `NULL`, timeout-safe teardown paths).
- Keep streaming-thread logic lightweight and side-effect controlled.

## Documentation obligations

When behavior changes:

- Update [Architecture](/contribute/architecture).
- Update user-facing guides if workflow or knobs changed.
- Document new environment variables in reference docs.

## PR quality bar

A contribution is ready when it includes:

- Clear rationale in code and commit/PR message.
- Tests for new behavior and regressions.
- Updated docs for any user-visible change.
- API compatibility assessment for public header changes (and full breaking-change process when applicable).
