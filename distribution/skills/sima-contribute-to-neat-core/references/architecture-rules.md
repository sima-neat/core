# Neat Core Architecture Rules

This reference summarizes the architecture PDF and local contributor docs for agents changing the SiMa Neat core repo.

## Mental Model

Neat has two related but separate concerns:

- **Neat framework**: the C++/Python library and runtime that loads model packs, builds pipelines, runs on Modalix hardware, and exposes APIs.
- **Neat environment / SDK**: the containerized developer loop with DevKit sync and agent skills.

When changing this repo, optimize the framework. The agent-friendly environment works because the framework is deterministic, inspectable, and strict about errors.

Primary concepts:

- **MPK**: sealed model artifact (`.tar.gz`, `.mpk`, etc.) containing a manifest plus plugin-private configs and binaries.
- **Model**: loaded MPK; parses manifest, runs route planning, extracts artifacts, exposes model stages and simple `run` paths.
- **Tensor**: typed numeric payload with dtype, shape, layout, storage, device, and semantic metadata.
- **Sample**: runtime/media envelope around data; check `Sample::kind` before reading `tensor`, `fields`, or tensor lists.
- **Node**: atomic pipeline building block that emits deterministic backend fragments and owned element names.
- **NodeGroup**: reusable recipe of Nodes for common patterns.
- **Session**: assembly, validation, caps negotiation, and build boundary. `Session::build()` returns a `Run`.
- **Run**: live pipeline handle for push/pull/run/close. Move it, do not copy it.
- **Graph**: use builder graph for DAG composition inside one pipeline; use runtime graph for coordinating stages/runs across pipelines.

## Design Principles

- **Determinism wins**: element names, generated launch strings, reports, serialization, and tests should be reproducible.
- **Debuggability is first-class**: expose `SessionReport`, `describe()`, `describe_backend()`, bus details, repro notes, and replayable launch strings.
- **Validate before run**: prefer cheap structural/caps/contract validation before starting runtime threads or touching hardware.
- **No silent fallback**: fail loudly for wrong formats, missing plugins, unavailable MLA/dispatcher, invalid caps, bad MPKs, or unsupported requests.
- **Single source of truth**: model routing decisions come from the MPK manifest, not per-stage JSON files.
- **Safe concurrency**: streaming-thread code must stay small, bounded, and thread-safe.
- **Never hang the process**: teardown, EOS drain, bus watch, and dispatcher paths must have bounded behavior.
- **Stable public API**: headers under `include/*` are the supported API surface.

## Module Boundaries

Keep dependencies clean:

- `builder/`: graph/composition utilities. No GStreamer, no `pipeline/`.
- `nodes/`: typed pipeline fragments and NodeGroups. No `pipeline/`.
- `gst/`: thin GStreamer helpers. No `pipeline/`.
- `mpk/`: model-pack validation, manifest parse, secure extraction, sequence adaptation.
- `model/`: `Model`, route planning, model semantics, session materialization.
- `pipeline/`: runtime orchestration, parse/build/run/push/pull/teardown, diagnostics.
- `graph/`: runtime graph/stage execution layer. Do not confuse with builder graph.
- `python/`: nanobind bindings; keep Python concepts aligned with C++ concepts.

If a change wants to cross a boundary, stop and look for an existing adapter or contract layer.

## MPK And Route Planning

- Core reads the MPK manifest (`mpk.json` or `*_mpk.json`) as the authoritative model contract.
- Per-stage JSON files inside the MPK are plugin-private. Do not make core decisions by reading them.
- Route planning bridges user-visible FP32 contracts to MLA-native INT8/BF16 and tessellated layouts where needed.
- Generic preprocessing and box decode are explicit model options/stages; do not add hidden host-side conversion.
- A BGR/RGB mismatch should fail or require an explicit conversion stage; it should not be silently corrected.
- If MLA or dispatcher is unavailable, build should fail with structured diagnostics, not fall back to CPU.

## Diagnostics Contract

Good failures are data:

- `SessionReport.error_code`: stable machine-triage code.
- `SessionReport.repro_note`: concise human summary with offending value or next action.
- `SessionReport.bus`: source of truth for GStreamer/plugin runtime errors.
- `repro_gst_launch` / `describe_backend()`: isolate failures outside framework code.
- Build adaptation summaries should explain shape/caps/contract adaptations applied or skipped.

When adding errors, include the context an agent needs to fix the next iteration.

## Runtime And Data Contracts

- `Session::build(input, ...)` uses input shape/dtype/format to seed caps negotiation and build adaptation. Do not bypass that contract for consumer pipelines.
- `Session::build(options)` without an input is for source pipelines where the pipeline generates data internally.
- `close_input()` should send EOS and allow in-flight frames to drain; pull until closed rather than dropping work.
- `Tensor::map(Read)` and `map(Write)` have coherence costs. Preserve declared map intent.
- Preserve zero-copy opportunities and device metadata; do not force CPU copies unless required and documented.
- Preprocessing metadata, including affine transforms, is a runtime contract for downstream stages such as box decode.

## Test Mapping

Change surface to test surface:

- Node fragment/naming change: deterministic backend fragment and element-name tests.
- Caps/contract change: validate/parse failure and success tests.
- RoutePlanner/MPK change: manifest-driven tests, error taxonomy tests, model integration tests.
- Runtime push/pull/teardown change: success, timeout/failure, close/EOS, and lifecycle tests.
- Tensor/Sample change: dtype/shape/layout/storage/device tests plus Python interop if bound.
- Python binding change: `python/tests`, import tests, NumPy/PyTorch/copy-vs-view checks.
- Docs/tutorials change: generated docs build and consistency tests.

Skip paths should be exceptional. Strict tests should fail when fixtures are missing unless they are explicitly long/hardware lanes.
