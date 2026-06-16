---
name: sima-contribute-to-neat-core
description: Guidance for making safe changes in the SiMa Neat core repository. Use when modifying Neat framework code, public headers, pipeline/runtime behavior, Model/archive/MPK-contract handling, Nodes/NodeGroups, Graph, Tensor/Sample contracts, Python bindings, tutorials, docs, or tests in the core repo.
---

# Contribute To Neat Core

Use this skill as the operating checklist before changing the Neat framework. The repo optimizes for deterministic pipelines, structured diagnostics, stable public APIs, and agent-friendly development loops.

For deeper architecture notes distilled from the architect-level PDF, read `references/architecture-rules.md` when touching `include/`, `src/`, `python/`, `tests/`, `tutorials/`, or model/pipeline behavior.

## First Pass

1. Read the local guardrails first:
   - `AGENTS.md`
   - `docs/develop-apps/contribute/start-here/coding_standard.md`
   - `docs/develop-apps/contribute/build-test/test_requirements.md`
   - `docs/develop-apps/contribute/start-here/architecture.md`
2. Classify the change:
   - Public API: anything under `include/*`.
   - Runtime/pipeline: `src/pipeline`, `src/gst`, `src/nodes`, `src/model`.
   - Python API: `python/`, `pyneat`, nanobind bindings.
   - Docs/tutorials: `docs/`, `tutorials/`, examples, generated tutorial docs.
3. Preserve existing patterns before introducing abstractions. Search with `rg` for nearby implementations, tests, and docs.

## Core Invariants

- Treat `include/*` as stable API. Prefer additive changes. Breaking public signatures require PR documentation, migration notes, tests, and explicit maintainer approval.
- Keep module boundaries intact:
  - `builder/` must stay STL-only and must not depend on GStreamer or `pipeline/`.
  - `gst/` must not depend on `pipeline/`.
  - `nodes/` must not depend on `pipeline/`.
  - `pipeline/` orchestrates and may depend on `gst/`, `builder/`, `nodes/`, `contracts/`, `policy/`, and `model/`.
- Keep behavior deterministic: stable element names, stable pipeline strings, stable report fields, reproducible tests.
- Prefer fail-fast structured errors over hidden fallback. Do not silently convert formats, auto-degrade hardware paths, or mask caps/plugin/runtime failures.
- Keep teardown bounded. Never add indefinite waits in `Run`, streaming, bus, or teardown paths.
- Keep probe/streaming-thread work lightweight and thread-safe. Diagnostics updated from streaming threads need atomics or equivalent protection.

## Model And Pipeline Rules

- Treat the MPK manifest (`mpk.json` or `*_mpk.json`) as the single source of truth for model routing, dtype, shape, quantization, and stage decisions. Do not inspect per-stage JSON configs in core logic; those are plugin-private.
- `Model` is a loaded `.tar.gz` model archive and a simplified user entry point. Internally it resolves to Nodes/NodeGroups assembled into a `Graph`.
- `Graph` is the assembly/validation/materialization boundary. Validate before run when practical, and preserve `GraphReport` usefulness.
- `Run` is the live runtime handle. It is movable, not copyable. Shutdown should drain cleanly and stay bounded.
- `Tensor` carries typed numeric payload plus shape/layout/storage/device. `Sample` wraps runtime/media metadata and may contain tensors, fields, or bundles. Check `Sample::kind` before reading fields.
- There are two graph concepts: builder topology helper (`simaai::neat::BuilderGraph`, STL-only DAG before GStreamer) and runtime graph (`simaai::neat::graph::Graph`, actor-style composition of stages/runs). Keep them separated.

## Diagnostics And Errors

- New failure paths should include actionable context and map into structured diagnostics, not bare strings.
- Preserve `GraphReport.error_code` as the machine-triage field and `repro_note` as the human summary.
- Include node/element context when possible, and keep `describe()`, `describe_backend()`, `last_pipeline()`, and replay/debug paths useful.
- Avoid silent fallback. Unsupported input format, missing plugins, unavailable MLA/dispatcher, invalid model archive or MPK contract, and caps negotiation issues should surface as explicit failures.

## Testing Expectations

- Behavior changes need regression tests. Do not rely on “build passes” as proof of correctness.
- New nodes/node-groups need deterministic fragment tests, validation/parse coverage, and at least one realistic integration path.
- Runtime orchestration changes need success and failure paths, plus teardown/lifecycle coverage when relevant.
- Public API changes need compile coverage for old and new usages when non-breaking.
- Python binding changes need `pytest` coverage under `python/tests` and interop checks for Tensor/Sample behavior.
- Tutorial/docs changes should keep README metadata, CMake targets, smoke tests, and usage comments in sync.

## Validation Commands

Prefer the repo entrypoints when feasible:

```bash
./build.sh --all
./build.sh --doc
```

For smaller scoped work, use targeted checks that match the change surface, for example:

```bash
bash scripts/check_format.sh --changed-only
bash scripts/check_cmake_format.sh --changed-only
bash scripts/check_duplicate_includes.sh --changed-only
pytest -q python/tests
pytest -q tests/tutorials/test_tutorials_run.py
npm --prefix website run build
```

If a check cannot run because hardware, model archives, `pyneat`, or DevKit fixtures are unavailable, state that clearly and run the highest-value local alternatives.

## Done Criteria

- Relevant tests pass or documented blockers are explicit.
- Docs/tutorials/examples are updated for user-visible behavior.
- API compatibility impact is assessed.
- Architecture docs are updated if contracts, module responsibilities, or runtime semantics changed.
- The final response calls out changed files, validation, skipped checks, and residual risk.
