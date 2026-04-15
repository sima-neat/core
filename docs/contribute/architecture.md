---
title: Architecture
description: Repository architecture and design
sidebar_position: 1
---

# Repository Architecture & Design

This page is for contributors who need to understand how the library is
structured, where responsibilities live, and how to extend the framework
without breaking its module and runtime contracts.

---

## What this library is for

### Primary users
Developers who want to:
- Assemble pipelines from reusable building blocks (without writing raw GStreamer boilerplate)
- Validate pipelines early (CI-friendly) and understand failures quickly
- Run pipelines and consume frames in C++ via `appsink`
- Optionally serve a pipeline over RTSP (via `gst-rtsp-server`)
- Feed ML code via tensor-friendly outputs without writing GStreamer plumbing

### Common workflows
- **Decode / ingest:** file or RTSP -> depay/demux/parse -> decode -> convert/caps -> appsink -> C++ consumer
- **Validate:** build + parse + preroll (PAUSED) to catch negotiation issues early
- **Serve RTSP:** push synthetic frames into an RTSP server pipeline using `appsrc`
- **ML output:** image/video/RTSP -> decode -> convert/scale -> `add_output_tensor(...)` -> `Run::pull_tensor()`
- **Tutorials:** start at [Tutorials](../tutorials) for a runnable, ordered learning path

### Canonical production pipeline (source of truth)
The canonical "production path" for this repo is:
**input -> preprocess -> MLA -> postprocess**. The source of truth lives in:
`tests/e2e_pipelines/obj_detection/sync_yolov8_test.cpp`.

When this test changes, update README + Architecture to keep docs aligned.

### Mental model (business logic &lt;-&gt; pipeline glue)
Your app keeps the business logic; the framework owns the pipeline glue.

```text
Business logic
    |
    v
Nodes/NodeGroups  ->  GStreamer fragments  ->  caps negotiation  ->  runtime (Run)
    |                                                           |
    +-----------------------------------------------------------+
                                Sample / Tensor
```

---

## Repository layout

### High-level structure
- `include/` -- public headers (the supported API surface)
- `src/` -- implementations
- `docs/` -- documentation (this file)
- `examples/` -- small runnable examples
- `tests/` -- unit/integration tests
- `python/` -- `pyneat` package sources, nanobind bindings, and Python tests
- `old_*` -- legacy monolithic implementation snapshots kept for reference/migration

### Public header tree (`include/`)
Public headers live under `include/<module>/...`.
Examples: `include/pipeline/Session.h`, `include/model/Model.h`.

Public convenience entry headers:
- `include/neat.h` (umbrella)
- `include/neat/session.h`
- `include/neat/models.h`
- `include/neat/nodes.h`
- `include/neat/node_groups.h`
- `include/neat/graph.h`

### Internal headers and runtime plugin paths

Public headers under `include/` are installed and treated as stable API.
Internal headers under `src/**/internal` are not installed; examples/tutorials
should use only public API.

Runtime environment notes:

- If using bundled GStreamer plugins in `deps/gst-plugins`, set
  `GST_PLUGIN_PATH` and/or `GST_PLUGIN_PATH_1_0` to include that directory.
- If installed with `cmake --install`, plugins are placed under
  `${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/sima-neat/gst-plugins`.
  Add that path to `GST_PLUGIN_PATH` and/or `GST_PLUGIN_PATH_1_0`.
- Use `scripts/use_neatdecoder.sh` to set plugin paths for the current shell.
- If installing plugins system-wide, rebuild the system GStreamer cache.

---

## Planned vs stable (API surface)

| Area / API | Status | Notes |
| --- | --- | --- |
| Core pipeline API (`Session`, `Run`, `Tensor`, `Sample`) | Stable | Primary supported C++ surface. |
| Builder layer (`Node`, `NodeGroup`, `Graph`, `GraphPrinter`) | Stable | STL-only, pre-GStreamer composition. |
| MPK APIs (`ModelMPK`, `ModelGroups`, `Model`) | Stable | Canonical MPK integration path. |
| `include/policy/*` | Stable | Minimal validated policy contracts and defaults (`Decoder`, `Encoder`, `Memory`, `RTSP`). |
| `include/mpk/MpKLoader.h` / `MpKManifest.h` / `MpKPipelineAdapter.h` | Stable | Implemented MPK inspection/extraction, error taxonomy, and sequence adaptation helpers. |
| `include/nodes/groups/ImageToH264RtspGroup.h` | Planned | Empty placeholder group. |
| `include/nodes/groups/MpKCompatGroup.h` | Planned | Empty placeholder group. |
| Python bindings (`python/`, `pyneat`) | Beta | Nanobind-based bindings and packaging live in-repo; API surface focuses on `Tensor`, `Session/Run`, `Model`, and core node/group helpers. |

---

## Modules and responsibilities

### `builder/` -- graph & composition (no GStreamer)
**Purpose:** Define how pipelines are assembled from logical parts.

Key types:
- `Node` -- interface implemented by each pipeline building block
- `Graph`, `Builder`, `NodeGroup` -- composition utilities and printing

**Rule:** builder must remain mostly STL-only. It should not own GStreamer runtime objects.

---

### `nodes/` -- typed pipeline building blocks
**Purpose:** Provide ready-to-use Node implementations that emit deterministic GStreamer fragments.

Examples:
- `nodes/io/RTSPInput`, `nodes/io/StillImageInput`
- `nodes/common/*` (Caps, Queue, Output, etc.)
- `nodes/sima/*` (SiMa decode/encode/parse/pay nodes)
- `nodes/rtp/*` (depay/payload helpers)
- `nodes/groups/*` (common multi-node recipes)

**Contract:**
Each Node must produce:
- `backend_fragment(index)` -- the GStreamer fragment for this node at a given index
- `element_names(index)` -- deterministic element names owned by this node (for diagnostics and enforcement)

---

### `gst/` -- thin GStreamer utilities
**Purpose:** Small wrappers/helpers around common GStreamer patterns.

Examples:
- initialization (`GstInit`)
- parsing launch strings (`GstParseLaunch`)
- bus draining/stringifying (`GstBusWatch`)
- caps helpers / element introspection (`GstHelpers`, `GstIntrospection`)
- pad taps / probe helpers (`GstPadTap`)

**Rule:** `gst/` must not depend on `pipeline/` (to avoid dependency cycles and "utility layer" bloat).

---

### `pipeline/` -- runtime orchestration and public API
**Purpose:** Own the runtime lifecycle: build -> parse -> run -> consume -> teardown, with diagnostics.

Key types:
- `Session` -- the main entry point for users
- `Run` -- running pipeline handle with push/pull APIs
- `Sample` -- structured output payload returned by pulls
- `SessionReport` -- structured diagnostics for failures, stalls, and reproduction
- `Errors` -- exceptions (`SessionError`) embedding a report

#### Error semantics contract

`SessionReport.error_code` is the canonical machine-triage field. Framework
runtime/build/IO paths map terminal failures into stable code families:

- `misconfig.pipeline_shape`
- `misconfig.caps`
- `misconfig.input_shape`
- `build.parse_launch`
- `runtime.pull`
- `io.parse`
- `io.open`

`SessionReport.repro_note` is the human-facing summary and must include enough
context to reproduce (offending value, node/element context, or hint).
`SessionReport.bus` is the source of truth for plugin/runtime error details.
For build(input) flows, `SessionReport.build_adaptation` records the resolved shape policy/capability, origins for seed/max limits, byte-guard origin, and applied/skipped adaptation actions.
For non-throwing runtime pulls, `PullError.code` uses the same taxonomy.

Support triage order is:
1. bucket by `error_code`
2. read `repro_note`
3. inspect first terminal `bus` errors
4. replay with `repro_gst_launch`

#### Internal pipeline diagnostics
Under `src/pipeline/internal/` (internal-only, test targets via `sima_neat_internal`):
- `Diagnostics.h` -- shared diagnostics types used by runtime:
  - `DiagCtx` (bus log + node reports + boundary/element counters)
  - `BoundaryFlowCounters` (atomic counters updated from streaming threads)
  - `ElementTimingCounters` (atomic per-element compute timing)
  - `ElementFlowCounters` (atomic per-element flow stats)
- `GstDiagnosticsUtil.h` -- helpers for formatting and collecting GStreamer diagnostics

#### SIMA static manifest context contract
For model pipelines, static stage/tensor contract data is built in framework and injected as a
pipeline-level `GstContext`:

- Context type: `sima.model.manifest.v1`
- Context fields:
  - `manifest_version`
  - `manifest_json` (legacy compatibility payload)
  - `manifest_accessor_v1` (ABI-safe accessor table pointer)
  - optional `session_id`, `model_id`
- Manifest ownership/lifetime is tied to pipeline lifetime; plugins borrow pointers and copy what
  they need.
- Repository boundary: this repo must not add build-time dependencies on plugin/dispatcher repos.
  Integration is interface-only (runtime `GstContext`, properties, caps/meta, and C-ABI contracts).

Resolver precedence for migrated fields is deterministic:

1. infer from contract/runtime signal (shape/meta/caps)
2. context/default/property path
3. hard bus error (never abort/SIGSEGV)

`StageTransformRuleRegistry` (internal) is the single mapping table that tells the resolver which
non-MLA stages inherit tensor contracts from MLA inputs vs MLA outputs, and when output quant is
propagated. This keeps pre/post derivation explicit and testable.

For migrated SIMA plugins using the aggregator template, runtime config now follows
context/property-driven resolution:
1. stage static fields come from manifest context
2. runtime knobs come from properties/context defaults
3. unresolved required fields fail explicitly (no stage-JSON fallback in framework)

For `simaaiprocesscvu`, CM-derived wiring is infer-first and context `sink_pad_tensor_index_map`
is used for deterministic multi-input mapping; legacy input-buffer names remain fallback-only.

`logical_stage_id` is resolved from `stage-id`/`stage_id` pipeline properties when provided,
otherwise it falls back to element name.
SIMA model-path fragment builders set `stage-id` on `simaaiprocesscvu`, `simaaiprocessmla`, and
`simaaiboxdecode` elements by default.

---

### `contracts/` -- validation rules
**Purpose:** Encode "what a valid pipeline looks like" beyond "gst_parse_launch succeeded".

Examples:
- validator interfaces and registries
- structured `ValidationReport`

This layer can be used for CI and for catching issues before runtime.

---

### `policy/` -- user-tunable behavior
**Purpose:** Centralize tunables (defaults, memory constraints, encoder/decoder/RTSP policy choices).

The goal is to make "knobs" explicit and discoverable rather than hidden in scattered code.

---

### `mpk/` -- MPK integration
**Purpose:** Load/interpret "model packs" (MPK) and adapt them into pipeline nodes or pipeline fragments.

This module is intentionally optional and should not contaminate the core runtime path unless used.

Key types:
- `ModelMPK` -- loads an MPK tarball, parses its JSON, and exposes model fragments
- `ModelStage` -- `Preprocess`, `MlaOnly`, `Postprocess`, `Full`
- `ModelFragment` -- `{gst, elements}` pair for deterministic fragments

Common usage:

```cpp
// From a cv::Mat (OpenCV enabled)
auto model = sima::mpk::ModelMPK("resnet_50_mpk.tar.gz", rgb_mat,
                                 /*normalize=*/true,
                                 /*mean=*/{0.485f, 0.456f, 0.406f},
                                 /*stddev=*/{0.229f, 0.224f, 0.225f});

// Or with explicit caps/shape (no OpenCV dependency)
// sima::mpk::ModelMPK("resnet_50_mpk.tar.gz", "video/x-raw", "RGB", 224, 224, 3,
//                     /*normalize=*/true, /*mean=*/..., /*stddev=*/...);

sima::Session p;
p.add(sima::nodes::groups::Infer(model));
```

`ModelMPK::to_node_group(ModelStage)` returns a `NodeGroup` for a specific stage.
The `sima::nodes::groups::{Preprocess,MLA,Postprocess,Infer}` helpers wrap that
API and should be preferred when composing pipelines from an already-loaded model.

`ModelMPK::input_appsrc_options(...)` provides caps/config for `Input`
when you need to feed frames or tensors into an MPK pipeline.

---

### `stages/` -- stage-by-stage execution
**Purpose:** Run individual MPK stages without a full pipeline build.

Key APIs:
- `sima::stages::Preproc(cv::Mat, ModelMPK)`
- `sima::stages::MLA(Tensor, ModelMPK)`
- `sima::stages::BoxDecode(Tensor, ModelMPK, BoxDecodeOptions)`

This is used for stage-only tests (`yolov8_stage_route_test.cpp`) and for
hybrid flows where preproc is done once and MLA/BoxDecode are run in a separate
pipeline or thread.

---

### Where work runs (CPU / CVU / MLA)
Processor routing is determined by the MPK graph configuration (the CVU/MLA
stages defined in the model pack) plus optional runtime overrides:

* `ModelMPK` constructors allow setting `preproc_next_cpu`, `num_buffers_cvu`,
  and `num_buffers_mla` to influence throughput and stage placement.
* `SIMA_MLA_NEXT_CPU` can override the next stage for MLA in some configs.
* Pipeline nodes themselves are declarative; actual execution happens in the
  GStreamer plugins and their configs.

Practical impact: more buffers and explicit routing can improve throughput, while
caps mismatches or undersized buffers will fail fast during negotiation.

---

## Runtime model (how execution works)

### Initialization
All runtime entry points call a single safe initialization routine:
- `gst_init_once()` (thread-safe, `std::call_once`)

Additionally, runtime paths may verify required plugins are present:
- `require_element("appsink", ...)`, etc.

### Building pipelines
A `Session` is built by adding `Node` objects:

```cpp
sima::Session s;
s.add(nodes::RTSPInput("rtsp://..."))
 .add(nodes::H264DecodeSima())
 .add(nodes::Caps(/*...NV12...*/))
 .add(nodes::Output());
```

Internally:

1. The session asks each Node for `backend_fragment(i)` and concatenates fragments with `!`
2. Optionally inserts **boundary markers** between nodes:

   * `identity name=sima_b<i> silent=true`
3. Builds a `DiagCtx`:

   * `node_reports` for reproducibility
   * `boundaries` as `BoundaryFlowCounters` (atomics)

### Push/pull runtime model

`Run` owns input/output queues and an input thread:

* `push(...)` enqueues inputs (blocking or dropping based on `RunOptions::overflow_policy`)
* `pull(...)` dequeues `Sample` outputs from the appsink
* `try_push(...)` is non-blocking (returns false if the queue is full)

This supports fully async pipelines (producer/consumer split) as well as
sync flows (`RunMode::Sync` or `Session::run(...)`).

### Parsing & launch

The library primarily uses:

* `gst_parse_launch(pipeline_string, &err)`

This provides flexibility and debuggability (you can replay the exact string with `gst-launch-1.0`).

### Running

Typical flow (`Session::build()` / `Run`):

1. Enforce contracts (e.g., "sink last" for `build()` + pull)
2. Build pipeline string (+ optional boundaries)
3. Parse pipeline
4. Optionally enforce element naming contract
5. Attach optional boundary probes
6. Set pipeline to `PLAYING`
7. Return a `Run` handle for push/pull control

### Life of a frame (plain language)
1. **Build:** Nodes become a deterministic gst-launch string.
2. **Negotiate:** GStreamer negotiates caps between elements (format, size, memory).
3. **Run:** Inputs are pushed (or pulled from sources) into the pipeline.
4. **Sample:** Appsink yields a `Sample` / `Tensor` back to your code.
5. **Error:** Any negotiation or runtime failure becomes a `SessionError` with a `SessionReport`.

Caps negotiation is automatic; failures surface early (validate/preroll) or at runtime with
diagnostics you can reproduce (`describe_backend()` + report).

### Teardown

Teardown is intentionally defensive.
Some plugin stacks can hang on state changes; the runtime prefers to avoid deadlocking the host process/CI.

The common pattern is:

* send EOS
* set `GST_STATE_NULL`
* unref objects
* apply a timeout safeguard (leak instead of hanging if necessary)

---

## SimaAI concurrency

SimaAI plugins support multiple pipelines per process. If you run several
pipelines concurrently, make element names unique via
`SessionOptions` or `Model` name suffixes/prefixes to avoid
GStreamer name collisions.

---

## Constraints & safety

* **Input formats must match caps**: `InputOptions` and model configs must agree on format/width/height.
  Mismatches fail fast during negotiation or when pushing inputs.
* **Capability-gated dynamic input**: runtime renegotiation is allowed only when the built graph advertises dynamic capability. `FullyDynamic` graphs can renegotiate raw-video geometry/format/fps/media caps; `IngressDynamicCvuOnly` allows geometry changes and permits format changes only when build-time downstream contract checks prove stable output behavior.
* **Dynamic within effective bounds**: `max_*` are hard ceilings; if `max_*` is unset, `width/height/depth` act as implicit ceilings.
* **Model vs Session defaults**: both flows now resolve seed/max/byte-guard policy through `src/pipeline/internal/InputPolicy.*`; `Model` still applies its documented metadata-backed defaults (for example 1920x1080 ceilings) while `Session` remains node-option driven unless configured.
* **`caps_override` is authoritative**: when set, renegotiation is blocked and shape changes require rebuild.

| Flow | Seed defaults | Max defaults | Byte-guard default |
| --- | --- | --- | --- |
| `Model` | preproc metadata (if present), otherwise inferred from user format/options | explicit `input_max_*`; otherwise policy defaults (for example `1920x1080`, format-derived depth) | explicit `RunOptions.max_input_bytes`, otherwise bounded estimate or elastic default from `InputPolicy` |
| `Session` | input-node options and/or seed input sample | explicit `max_*`; otherwise implicit from seed `width/height/depth` when provided | explicit `RunOptions.max_input_bytes`, otherwise bounded estimate or elastic default from `InputPolicy` |

* **SimaAI concurrency**: multiple pipelines can run in-process; keep element names unique.

---

## Threading & ownership model

### Threads

* **GStreamer streaming threads**: pad probes, decoding, scheduling
* **User thread**: `appsink` polling + periodic bus draining
* **RTSP server thread**: GLib main loop for `gst-rtsp-server` mode

### Ownership rules (GStreamer objects)

* GStreamer objects are reference counted.
* If you store a `GstObject*` beyond the scope where it was acquired, you must `gst_object_ref()` it.
* Always `gst_object_unref()` exactly once when done.

### Diagnostics thread safety (important)

Pad probes run on streaming threads, so **diagnostics updated from probes must be lock-free**.

The design is:

* `BoundaryFlowCounters` stores **atomics**
* pad probes only do atomic `fetch_add()` / `store()`
* reporting uses `BoundaryFlowCounters::snapshot()` to convert atomics -> `BoundaryFlowStats` (plain ints)

This avoids data races while keeping probes cheap.

---

## Diagnostics & observability

### `DiagCtx` captures:

* the pipeline string (for reproduction)
* node reports (what each node generated)
* bus messages (under a mutex)
* boundary flow counters (atomics)
* element timing + flow counters (atomics)

### Boundary flow probes

When enabled, the runtime attaches pad probes to boundary `identity` elements.
They track:

* buffer counts (in/out)
* last seen PTS (ns)
* last seen wall time (monotonic us)

This is used to generate "likely stall" summaries:

* "we last saw activity entering/leaving boundary X at T"

### Element timing probes

When enabled (`SIMA_GST_ELEMENT_TIMINGS=1`), the runtime attaches sink+src pad probes
to **all pads** (static, dynamic, and request) for each element and records
`src_ts - sink_ts` per buffer. This produces per-element compute timings without
relying on plugin instrumentation.

For elements that replace buffers, the implementation falls back to `GstSimaMeta`
correlation (frame-id/stream-id) and records `missed_in`/`missed_out` counters.

### Element flow probes

When enabled (`SIMA_GST_FLOW_DEBUG=1`), the runtime attaches per-element pad probes
to track buffer/byte counts and caps changes, providing throughput context for
every plugin in the graph.

### Bus logging and errors

The runtime drains bus messages into `DiagCtx`.
On an error message (`GST_MESSAGE_ERROR`), it throws `SessionError` including a `SessionReport` and reproduction hints.

### DOT dumps

If enabled, the runtime can emit DOT graphs via `gst_debug_bin_to_dot_file_with_ts(...)` to a configured directory.

### Debugging playbook (production)
1. **Reproduce the pipeline**: `Session::describe_backend()` or `last_pipeline()`.
2. **Capture a report**: `Run::report()` or `SessionError::report()`.
3. **Enable targeted probes**:
   - `SIMA_GST_BOUNDARY_PROBES=1` for stall localization
   - `SIMA_GST_ELEMENT_TIMINGS=1` for per-element timing
   - `SIMA_GST_FLOW_DEBUG=1` for per-element flow counters
4. **Generate DOT graphs**: set `SIMA_GST_DOT_DIR` and reproduce.
5. **Tighten validation**: `SIMA_GST_ENFORCE_NAMES=1` and validate preroll timeouts.

---

## Output handling

`Run::pull()` yields a `Sample`, which may carry:

* a `Tensor` payload (`SampleKind::Tensor`)
* a bundle of multiple outputs (`SampleKind::Bundle`)

Convenience helpers like `Run::pull_tensor()` and
`Run::pull_tensor_or_throw()` are provided for ML-centric flows.

---

## Pipeline serialization (save/load)

Pipelines can be saved and restored as JSON:

* `Session::save(path)` writes a versioned JSON with node kind/label/fragment/elements
* `Session::load(path)` rehydrates nodes via a `ConfiguredNode` wrapper

The current schema is intentionally minimal and reproducible, and can evolve to richer
node configs later. This also serves as the bridge for future bindings and tooling.

---

## UX helpers

* `Session::describe()` uses `GraphPrinter` to render a human-readable node list
* `Session::describe_backend()` returns the gst-launch string for quick debugging

---

## Element naming & determinism

Deterministic element names are a core design principle because they enable:

* `gst_bin_get_by_name()` for sinks and key elements
* stable probe attachment
* stable diagnostics and reproducibility
* optional naming contract enforcement ("every element belongs to some node")

**Node authors must ensure**:

* fragments include stable `name=` fields when elements must be retrievable
* `element_names()` matches exactly what the fragment creates

---

## Stage naming and wiring

The framework now treats plugin JSON as plugin-owned data and does **not** rewrite or validate
per-stage JSON fields during pipeline build.

Wiring source of truth:

1. Deterministic GStreamer element names from node fragments.
2. `stage-id` on SIMA model-path elements.
3. `sima.model.manifest.v1` context for static stage/tensor contract lookup.

Implications:

* Build no longer mutates `node_name` / `input_buffers[*].name` / `buffers.input[*].name`.
* Build no longer performs JSON-based wiring checks.
* Name transform still applies to element names only.

For model-managed sessions, stage resolution is driven by `stage-id` + manifest context.
For non-model sessions, explicit plugin properties are the runtime control plane.

---

## Validation & contracts

Validation exists to catch issues earlier than runtime:

* `validate()` can parse and preroll (PAUSED) to detect negotiation stalls
* `contracts/` provides structured validators for "pipeline correctness"

The intended behavior:

* runtime flows throw exceptions on fatal errors
* validation flows return structured reports (CI-friendly)

---

## RTSP server mode

`run_rtsp()` uses `gst-rtsp-server`:

* a server runs in a dedicated thread with a GLib main loop
* on `media-configure`, the code locates the `appsrc` by name and configures caps/properties
* frames are pushed periodically (timer-based) with explicit timestamps

Each client may get its own media instance depending on factory configuration.

---

## Environment / configuration knobs

The runtime supports environment-driven debugging knobs:

* `SIMA_GST_DOT_DIR` -- write DOT graphs for failures / debug
* `SIMA_GST_BOUNDARY_PROBES` -- enable boundary flow counters
* `SIMA_GST_STAGE_TIMINGS` -- enable stage timing probes
* `SIMA_GST_ELEMENT_TIMINGS` -- enable element timing probes
* `SIMA_GST_FLOW_DEBUG` -- enable per-element flow counters
* `SIMA_GST_ENFORCE_NAMES` -- enforce naming contract
* `SIMA_GST_RUN_INPUT_TIMEOUT_MS` -- input timeout for run/build input paths
* `SIMA_GST_VALIDATE_TIMEOUT_MS` -- validation timeout for preroll
* `SIMA_GST_VALIDATE_INSERT_BOUNDARIES` -- insert boundaries during validate()
* `SIMA_GST_RUN_INSERT_BOUNDARIES` -- insert boundaries during build/run()
* `SIMA_GST_TEARDOWN_TIMEOUT_MS` -- wait for NULL state (ms)
* `SIMA_GST_TEARDOWN_REAPER_MS` -- reaper retry interval (ms)
* `SIMA_GST_TEARDOWN_ASYNC` -- skip wait, defer to reaper

These knobs are intentionally outside the public API so you can turn them on in CI or in the field without recompiling.
There are additional low-level debug flags in `src/pipeline/internal/*` (input stream
logging, sample dumps, pool debug). Keep those out of user-facing docs unless
you need deep diagnostics.

---

## How to extend the library

### Adding a new Node

1. Create a header in `include/nodes/<category>/<YourNode>.h`
2. Implement in `src/nodes/<category>/<YourNode>.cpp`
3. Ensure:

   * `backend_fragment(i)` is valid and deterministic
   * all important elements are named and returned by `element_names(i)`
4. Add tests (ideally one of):

   * parse/validate tests
   * run/build tests with a simple source/sink pipeline

### Adding runtime diagnostics

* Prefer adding fields to `DiagCtx` and `SessionReport`
* If updates happen from streaming threads, use **atomics** (or another lock-free mechanism)
* Convert to plain snapshot types for reporting

---

## Dependency rules (non-negotiable)

* `builder/` should not depend on GStreamer or `pipeline/`
* `gst/` should not depend on `pipeline/`
* `nodes/` should not depend on `pipeline/` (Nodes are build-time descriptions, not runtime orchestrators)
* `pipeline/` is the orchestrator and can depend on `gst/`, `builder/`, `nodes/`, `contracts/`, `policy/`, `mpk/`

This keeps the architecture modular and prevents circular dependencies.

---

## Tests & examples

* `examples/` show typical end-to-end usage patterns:

  * decode RTSP
  * run MPK
  * run RTSP server
* `tests/` verify critical behaviors:

  * file read paths
  * group expansion equivalence (input groups)
  * tensor output path + save/load round-trip
  * `model_resnet50_multi_test` validates Model accuracy with multiple sessions

When adding features, prefer adding tests that:

* reproduce the pipeline string deterministically
* validate caps negotiation assumptions
* ensure failures produce useful `SessionReport` diagnostics

---

## Docs drift guard

Keep docs and code aligned:

* If you change public headers (`include/*`), update README + Architecture.
* If you change the canonical production pipeline test
  (`tests/e2e_pipelines/obj_detection/sync_yolov8_test.cpp`), update both docs.
* If you add new env knobs, add them to the "Environment / configuration knobs" section.

---

## Design principles

1. **Determinism wins**

   * stable element names, stable pipeline strings, stable reports

2. **Debuggability is first-class**

   * bus logs, DOT dumps, boundary probes, clear reproduction steps

3. **Safe concurrency**

   * streaming-thread probes only touch atomics (snapshots produce plain reports)

4. **Never hang the process**

   * teardown is defensive; avoid blocking forever on broken plugin stacks

5. **Keep the public API stable**

   * internal refactors should not break user code unless intentionally versioned
