---
title: Architecture
description: Repository architecture and design
sidebar_position: 1
slug: /develop-apps/contribute/architecture
---

# Repository Architecture & Design

This page is for contributors who need to understand how the library is
structured, where responsibilities live, and how to extend the framework
without breaking its module and runtime contracts.

---

## Framework vs environment

The word "Neat" is used for two related but separate concerns:

- **Neat Library:** the C++/Python library and runtime in this repository. It loads model
  packs, composes pipelines, validates contracts, runs on Modalix hardware, and exposes the
  public API.
- **Neat SDK / environment:** the containerized development workflow around the framework,
  including DevKit Sync, shared workspaces, and agent tooling.

When changing this repository, optimize for framework properties that support both humans and
agents: explicit APIs, deterministic behavior, structured diagnostics, strict validation, and
stable public contracts.

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
- **Image/video tensor adapter:** image/video/RTSP -> decode -> convert/scale -> `add_output_tensor(...)` -> `Run::pull_tensors()`
- **Tutorials:** start at [Tutorials](/tutorials) for a runnable, ordered learning path

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
Nodes/Graph fragments  ->  GStreamer fragments  ->  caps negotiation  ->  runtime (Run)
    |                                                           |
    +-----------------------------------------------------------+
                                Sample / Tensor
```

---

## Core concepts

The framework is intentionally organized around a small set of concepts. Most user code touches
only `Model`, `Graph`, `Run`, `Tensor`, and `Sample`; lower-level contributors also work with
`Node`, reusable Graph fragments, MPK-contract parsing, and Graph internals.

| Concept | Role |
| --- | --- |
| Model archive | A sealed `.tar.gz` artifact containing the MPK inference contract, plugin-private configs, model binaries, and kernel artifacts. |
| `Model` | The public loader for a `.tar.gz` model archive. It parses the MPK contract, runs route planning, exposes model stages, and provides simple `run(...)` / `Graph` composition entry points. |
| `Tensor` | Typed numeric payload with dtype, shape, layout, storage, device, and semantic metadata. |
| `Sample` | Runtime/media envelope around tensors, tensor lists, or bundles. Check `Sample::kind` before reading fields. |
| `Node` | Atomic pipeline stage that emits a deterministic GStreamer fragment and owned element names. |
| Reusable Graph fragment | A premade `Graph` that expands to multiple Nodes, such as decoded RTSP input or model stages. |
| `Graph` | Assembly and validation boundary. Nodes, Models, and reusable Graph fragments become a negotiated, buildable pipeline. |
| `Run` | Live pipeline handle returned by `Graph::build(...)`; owns push/pull/runtime lifecycle. |
| Graph | Use builder graph for DAG composition inside one pipeline; use runtime graph for coordinating stages/runs across pipelines. |

Read the relationship left to right:

```text
model archive on disk -> Model -> Graph fragments/Nodes -> Graph -> Run
                                           |
                                           v
                                  Tensor/Sample flow
```

`Model` is the beginner-facing entry point, but it is not a separate execution engine. It resolves
to model Graph fragments/Nodes that can be added to a `Graph`. `Graph` is the central assembly
concept; `Run` is the live object after build.

---

## Design principles for contributors

These are the load-bearing architecture principles behind the framework. Use them when deciding
between implementation options.

- **Determinism wins.** Keep element names, generated pipeline strings, serialized pipeline data,
  report fields, and tests reproducible. Diagnostics and agent loops depend on stable identifiers.
- **Debuggability is first-class.** Failures should produce structured data, not only strings:
  `GraphReport.error_code`, `repro_note`, bus messages, and replayable backend pipelines.
- **No silent fallback.** Do not hide model-input bugs or hardware/runtime failures by quietly
  converting formats, changing graph families, falling back to CPU, or swallowing plugin errors.
- **Validate before run.** Prefer structural, caps, shape, and contract validation before runtime
  threads start or hardware resources are acquired.
- **The MPK contract is the model source of truth.** Core routing, dtype, shape, quantization, and
  stage decisions must come from `mpk.json` / `*_mpk.json`. Per-stage JSON files are
  plugin-private.
- **Detess logical rank and runtime geometry are separate.** Core preserves the MPK-authored
  `frame_shape` as the logical output contract and derives explicit MLA geometry when needed. A
  rank-2 shape is accepted as NC or HW only when declared byte spans identify one unique
  interpretation; ambiguous or inconsistent contracts fail during model loading.
- **Public APIs stay stable.** Public headers under `include/*` are installed and supported.
  Prefer additive changes and deprecation paths over breaking signatures.
- **Concurrency must be bounded and observable.** Streaming-thread work should be lightweight;
  probe-side diagnostics need atomics or equivalent thread-safe handling; teardown must not hang.

---

## Model execution path

For model-backed pipelines, the high-level path is:

```text
input Sample/Tensor
  -> optional preprocessing / format normalization
  -> MLA inference stages selected from MPK contract
  -> optional postprocessing / box decode
  -> output Sample/Tensor
```

The user-visible `Model` contract is intentionally friendlier than the MLA hardware contract. The
MLA may require INT8/BF16 and tessellated layouts, while user code generally works with FP32
and normal tensor layouts. The framework bridges that gap with manifest-driven adapter stages.

Preprocessing and postprocessing are explicit framework stages/options. A format mismatch, missing
required preprocessing metadata, unavailable selected hardware transport, invalid model archive or MPK contract, or caps negotiation
failure should surface as an actionable structured error rather than a hidden runtime correction.

### Compiled model execution contract

The DMA-BUF driver migration keeps model semantics separate from per-frame
storage. Core compiles model-load facts into one immutable internal
`ModelExecutionPlan`:

- values contain exact names, required bytes, optional logical type facts, and
  an optional root-relative read expression `{source ValueId, byte offset,
  byte strides}`;
- operations contain exact ordered edges and operation-specific configuration;
- MLA backend ports contain ELF/model order, exact required bytes, alignment
  authority, and access direction; and
- public outputs contain only publication order and the value they expose.

Frozen untyped AFE v2 MPKs use the quarantined `AfeMpkV2Decoder`. It accepts
only the exact registered `(model SDK version, processor, kernel)` vocabulary,
resolves full tensor names, validates operation byte equations, and reconciles
each MPK MLA stage independently with its exact ELF IFM/OFM topology. The
decoder joins executable evidence by both compiler-authored logical stage ID
and the manifest executable token; archive order and filename suffixes are not
semantic evidence. The resulting immutable plan exposes a checked
`{stage index, op ID, logical stage ID, executable}` key and pre-indexed ordered
port spans for every MLA operation. Ambiguity, missing slots, or conflicting
evidence is a model-load error; sidecar JSON, substring matching, environment
state, and runtime buffers are not evidence.

An AFE artifact ending in `.so` is therefore classified by its MPK stage, not
by its suffix. For `processor="MLA"`, Core reads the file as an ELF container
without loading it into the host process, proves its section topology, and
passes the exact artifact to MLArt. For `processor="A65"` with the
compiler-authored `input_names`, `input_types`, and `output_types` contract, it
is a host TVM module and is not an MLA executable.
The strict DMA-BUF route rejects that stage until a typed direct host-module
ABI is available; it never calls `dlopen`, silently skips the stage, or routes
through the dispatcher-based ProcessTVM compatibility element.
The legacy EVO alignment is an explicit 4096-byte migration policy with
`LegacyPolicy` provenance, not a fact inferred from its MPK or ELF. New typed
contracts must declare their own alignment.

Runtime frames do not copy the static plan. The direct path carries standard
DMA-BUF ownership plus checked absolute `{fd, offset, length}` views and a
bounded frame-slot lifetime. Unpack and Slice are consumer read expressions,
not runtime operations: the decoder composes them to one materialized root
carrier only after proving their exact offsets, byte strides, and physical
spans. Pack, tessellation, and precision conversion remain producer/compute
semantics unless a lowering pass proves exact backend production or equivalent
fusion. This prevents storage placement from silently changing MLA port arity
or model behavior.

All MLA stages in one accepted graph project through the same retained
`FrameSlotArenaPlan`. The first materializing strict stage allocates the arena;
later MLA stages use `ReuseInput` and submit exact root-relative IFM/OFM views
over that same DMA-BUF. Stage-less projection helpers remain available only
for an unambiguous one-MLA graph and fail closed for multi-stage plans.

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
Examples: `include/pipeline/Graph.h`, `include/model/Model.h`.

Public convenience entry headers:
- `include/neat.h` (umbrella)
- `include/neat/runtime.h`
- `include/neat/models.h`
- `include/neat/nodes.h`
- `include/neat/node_groups.h`

There is intentionally no `include/neat/graph.h` public umbrella. Runtime/compiler tests that
need the lower-level graph substrate include the narrow `include/graph/...` headers directly.
Applications, examples, and public docs should use the single public `simaai::neat::Graph` from
`<neat.h>`.

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
| Core pipeline API (`Graph`, `Run`, `Tensor`, `Sample`) | Stable | Primary supported C++ surface. |
| Builder internals (`Node`, private node-vector helpers, `GraphPrinter`) | Internal | STL-only, pre-GStreamer composition support. |
| Model API (`Model`, reusable Graph fragments) | Stable | Canonical model-archive integration path. |
| `include/policy/*` | Stable | Minimal validated policy contracts and defaults (`Decoder`, `Encoder`, `Memory`, `RTSP`). |
| `include/nodes/groups/ImageToH264RtspGroup.h` | Planned | Empty placeholder group. |
| Python bindings (`python/`, `pyneat`) | Beta | Nanobind-based bindings and packaging live in-repo; API surface focuses on `Tensor`, `Graph/Run`, `Model`, and core node/group helpers. |

---

## Modules and responsibilities

### `builder/` -- node contracts and private linear composition support (no GStreamer)
**Purpose:** Define how pipelines are assembled from logical parts.

Key types:
- `Node` -- interface implemented by each pipeline building block
- private node-vector helpers and `GraphPrinter` -- composition utilities and diagnostics

**Rule:** builder must remain mostly STL-only. It should not own GStreamer runtime objects.

---

### `nodes/` -- typed pipeline building blocks
**Purpose:** Provide ready-to-use Node implementations that emit deterministic GStreamer fragments.

Examples:
- `nodes/io/HttpSource`, `nodes/io/RTSPInput`, `nodes/io/StillImageInput`
- `nodes/common/*` (Caps, Queue, Output, etc.)
- `nodes/sima/*` (SiMa.ai decode/encode/parse/pay nodes)
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
- `Graph` -- the main entry point for users
- `Run` -- running pipeline handle with push/pull APIs
- `Sample` -- structured output payload returned by pulls
- `GraphReport` -- structured diagnostics for failures, stalls, and reproduction
- `Errors` -- exceptions (`NeatError`) embedding a report

#### Error semantics contract

`GraphReport.error_code` is the canonical machine-triage field. Framework
runtime/build/IO paths map terminal failures into stable code families:

- `misconfig.pipeline_shape`
- `misconfig.caps`
- `misconfig.input_shape`
- `misconfig.input_capacity`
- `misconfig.media_caps`
- `misconfig.tensor_dtype_missing`
- `misconfig.option_out_of_range`
- `build.parse_launch`
- `build.pipeline_syntax`
- `build.plugin_missing`
- `build.property_invalid`
- `runtime.pull`
- `runtime.element_failed`
- `runtime.output_timeout`
- `io.parse`
- `io.open`
- `io.file_not_found`
- `io.permission_denied`
- `io.rtsp_connection_failed`
- `io.camera_not_found`
- `codec.*`, `resource.*`, `infra.*`, and `internal.*`

GStreamer errors pass through one internal parser, classifier, and renderer.
Classification prefers a versioned Neat diagnostic ID, then the native
GStreamer domain/code and element factory, then narrow compatibility mappings
for older plugins. Unknown failures use `runtime.element_failed`; they are not
reported as `misconfig.media_caps` unless negotiation actually failed. When a pipeline
posts several errors, the most specific root cause is rendered and every error
is retained in the bus log.

`GraphReport.repro_note` is the human-facing summary. Production rendering
contains a plain-language cause, relevant observed/expected values, concrete
user actions, and a stable diagnostic ID. Raw plugin strings, source locations,
and GStreamer domain/code are debug-only. The bracketed public code is added
once when the `NeatError` is constructed.
`GraphReport.bus` is the source of truth for plugin/runtime error details.
For build(input) flows, `GraphReport.build_adaptation` records the resolved shape policy/capability, origins for seed/max limits, byte-guard origin, and applied/skipped adaptation actions.
For non-throwing runtime pulls, `PullError.code` uses the same taxonomy.
Input-stream worker failures retain the typed error code and report across the
worker-thread boundary, so `Run::pull()` and the Python exception translator
surface the same `NeatError`.

Support triage order is:
1. bucket by `error_code`
2. read `repro_note`
3. inspect first terminal `bus` errors
4. replay with `repro_gst_launch`

#### Internal pipeline diagnostics
Under `src/pipeline/internal/` (internal-only):
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

For the internal EVO DMA-BUF migration route, Core reads
`SIMA_NEAT_MEMORY_BACKEND` once per process. During migration the only valid
values are exactly `legacy` and `dmabuf-plan`; an unset variable selects
`legacy`, while empty, `auto`, `probe`, case-altered, whitespace-altered, and
unknown values fail closed. `ModelPack` records that immutable choice and is
the sole owner of model admission. Lower transfer and sample-materialization
helpers receive the resolved transport intent explicitly and never reread
mutable environment state. This temporary selector and its legacy branch are
owned by the Phase 7B deletion ledger; the strict-only product has no selector.

Selecting `dmabuf-plan` invokes the same side-effect-free
`try_compile_dmabuf_plan()` operation used by the offline
`neat-dmabuf-plan-audit` tool. Pass `--mpk <mpk.json>` and one repeatable
`--mla-artifact <stage-id> <manifest-executable> <resolved-file>` triple per
MLA stage. The one-stage `--elf` spelling remains an unambiguous compatibility
form. Admission requires an exact MPK manifest and exact identity for every
MLA ELF, a successful strict reverse-AFE decode, and an accepted immutable
frame-arena plan. The audit emits a versioned JSON record
with stable reason codes, contract locations, content digests, and basenames;
it does not allocate accelerator memory, open a device, or expose customer
filesystem paths. Strict setup records the same canonical plan digest and
fails rather than constructing or retrying the legacy executor after a
rejection.

Only after admission does Core set `processmla.dmabuf_plan_contract` in static
manifest ABI version 25. Core also projects each backend port's `required_alignment_bytes`
and the immutable frame-arena placement plan into its physical buffer record;
ProcessMLA consumes that value rather than duplicating the legacy
page-alignment policy. It consumes these Core-owned facts; it must not re-read
the environment, infer missing ports, or fall back to the legacy transport
after selection. Core and every plugin that consumes the static-manifest
header must therefore be built and released together at ABI version 25.

The same Core-owned memory policy controls public Tensor placement. With
`dmabuf-plan`, `transfer_to_device()` allocates standard CMA or DMS DMA-BUF
memory, performs the required cache-synchronized host copy, and records device
placement in Tensor storage metadata without replacing `GstDmaBufMemory` with
the legacy SiMa allocator. Tensor-list ingress adopts one standard DMA-BUF view
per source tensor. It rejects an implicit materialization or segmented-memory
fallback, so the static model proof and per-frame transport cannot silently
select different architectures.

At strict stage boundaries, logical payload size and physical address span are
different facts. The MPK-derived typed operation owns shape/layout and the
required address span; TensorBuffer metadata owns the checked DMA-BUF memory,
offset and available physical span. For example, a C16-strided 12-byte logical
value may touch offsets through byte 176 in a 192-byte MLA OFM. ProcessMLA's
direct binding therefore remains the ordered IFM/OFM mirror
`{tensor_slot, parent_carrier}`. The one boolean identifies a zero-offset
logical anchor for a larger physical port; it does not duplicate layout.
ProcessMLA must not add redundant `padded`, `strided`, or `contiguous` flags.

For a packed one-IFM/one-OFM model, Core proves the upstream logical outputs
are the ordered children of one complete IFM carrier. Downstream Unpack/Slice
values are published as logical reads of the one OFM carrier. ProcessMLA still
submits exactly one physical port in each direction; the consuming CVU kernel
uses the compiled offset/stride expressions directly, without an unpack job or
intermediate copy.

Multiple direct MLA OFMs can be logical views of one physical arena. In that
case every descriptor with the same memory index publishes the same parent
physical segment name, while logical/backend names preserve OFM identity and
exact arena offsets preserve each view. Direct ProcessCVU output publication
likewise uses its exact strict arena layout rather than the legacy packed
output reconstruction heuristic.

For the strict graphs currently admitted by `dmabuf-plan`, ProcessCVU submits
the same Core-projected tensor routes and frame arena through one of two
executors. EV74 placement submits descriptors through `/dev/cvu`. A65
placement maps each unique standard DMA-BUF parent once with the matching
read/write `DMA_BUF_IOCTL_SYNC` boundary, patches a frame-owned typed EV ABI
configuration with host virtual addresses, and executes the in-process A65
kernel. A `ReuseInput` post stage maps its shared MLA arena once as read/write;
it does not copy or reconstruct individual outputs. Both executors keep
ConfigManager and dispatcher execution disabled.

ProcessMLA submits through kernel-driver MLArt on `/dev/mla` without the MLA
dispatcher, MLASHM, segmented allocation, or M4. Legacy libraries may still
be linked into a multi-route plugin for unmigrated graphs; link presence is not
a fallback permission. Once a strict route is selected, any executor, mapping,
or synchronization error is terminal.

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

##### SuperPoint BoxDecode contract

SuperPoint uses the same MPK-to-static-manifest boundary as other model-managed BoxDecode
families, with these additional invariants:

- The MPK record owns the detector-logits and descriptor-grid tensor identities, storage
  representations, dtype/shape facts, numerical-profile provenance, and optional explicit NMS and
  border controls. Core never identifies these roles from tensor values.
- Core binds exactly one tensor to each role, validates the profile fingerprint and supported
  representations, applies explicit `Model::Options::superpoint` overrides, and resolves only
  omitted profile defaults. Changing a profile recomputes its derived defaults while preserving
  controls explicitly authored by the MPK or API.
- The versioned static-manifest ABI carries the resolved contract to `simaaiboxdecode`. Plugins
  borrow manifest pointers only during configuration and must copy any state needed at runtime;
  Core retains manifest ownership for the pipeline lifetime.
- Production output uses the `FEATURE_POINTS_V1` wire format and feature semantic metadata.
  `FEATURE_POINTS_LEGACY_A65_V0` is available only when explicitly selected for compatibility;
  consumers must not infer either format from buffer size.

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

### Model archive integration
**Purpose:** Load `.tar.gz` model archives through `Model` and adapt the parsed MPK
inference contract into routeable graph fragments.

The secure archive loader is internal implementation detail; application code should
construct `Model` and compose `model.graph()` or the stage-specific fragments.

Common usage:

```cpp
simaai::neat::Model model("resnet_50.tar.gz");
simaai::neat::Graph graph;
graph.add(model.graph());
```

---

### Model stage fragments
**Purpose:** Compose the preprocess, inference, postprocess, or full route exposed by
`Model` without exposing the internal archive loader.

Key APIs:
- `Model::preprocess()`
- `Model::inference()`
- `Model::postprocess()`
- `Model::graph()`

This is used for hybrid flows where preproc is done once and MLA/BoxDecode are run
in a separate graph or thread.

---

### Where work runs (CPU / CVU / MLA)
Processor routing is determined by the MPK contract (the CVU/MLA stages defined
in the model archive) plus optional runtime overrides:

* `Model::Options` controls preprocess, postprocess, naming, and buffering choices.
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
A `Graph` is built by adding `Node` objects and reusable Graph fragments. Use a
codec-aware fragment for RTSP so the source is depacketized and parsed before
decode:

```cpp
simaai::neat::nodes::groups::RtspDecodedInputOptions source;
source.url = "rtsp://example/live";
source.codec = simaai::neat::nodes::groups::RtspCodec::H265;
source.source_fps = 30;

simaai::neat::Graph graph;
graph.add(simaai::neat::nodes::groups::RtspDecodedInput(source));
graph.add(simaai::neat::nodes::Output());
```

Internally:

1. The Graph enforces one Node object per logical composition vertex. Repeated `connect()` calls
   can reuse that indexed vertex for fan-out.
2. A composition mutation commits as one unit or rolls back completely.
3. The Graph asks each Node for `backend_fragment(i)` and concatenates fragments with `!`.
4. It optionally inserts **boundary markers** between nodes:

   * `identity name=sima_b<i> silent=true`
5. It analyzes exact `name=` bindings, parses once with GStreamer, and inventories the constructed
   object tree. Duplicate or missing names fail before downstream configuration.
6. It builds a `DiagCtx`:

   * `node_reports` for reproducibility
   * `boundaries` as `BoundaryFlowCounters` (atomics)

### Push/pull runtime model

`Run` owns input/output queues and an input thread:

* `push(...)` enqueues inputs (blocking or dropping based on `RunOptions::overflow_policy`)
* `pull(...)` dequeues `Sample` outputs from the appsink
* `try_push(...)` is non-blocking (returns false if the queue is full)

This supports fully async pipelines (producer/consumer split) as well as
one-shot flows (`Graph::run(...)`).

### Decoder admission lifecycle

Before choosing the single-pipeline or connected-graph runtime, Core scans the
compiled execution plan for typed H.264/H.265 `SimaDecode` nodes. All eligible
decoders are admitted as one group, and the resulting reservation is owned by
the top-level `Run` until its pipeline workers have stopped. This applies
equally to linear `Graph::add(...)` pipelines, ordinary connected segments, and
fused realtime branches.

Admission requires a known decoder width, height, and frame rate. Core never
invents a frame rate. An incomplete contract or unavailable optional admission
endpoint produces a warning and leaves the plan unchanged; with
`SIMA_DECODER_ADMISSION_REQUIRE=1`, either condition fails before decoder
hardware starts. Capacity rejection and malformed lease responses always fail.

### Realtime fan-in lowering

Applications describe realtime edges with ordinary `Graph::connect(...)` and
materialize them with ordinary `Graph::build(...)`. `GraphLinkOptions` carries
the latest-by-stream policy, stream identity, a reserved queue-depth field, and
optional raw-frame admission limits. Latest-by-stream lowering always keeps one
pending sample per stream.

The execution-graph compiler, not the application, decides whether live
multi-source fan-in can be fused into one GStreamer pipeline. Eligible private,
inputless source branches are lowered with their by-stream mux and consumer so
decoded device buffers do not cross an appsink/appsrc boundary. Ineligible
latest-by-stream topology remains segmented. Nested already-fused source
segments remain ineligible until their branches can be preserved recursively.

### Internal boundary timing

One logical `Graph` can lower into several GStreamer pipeline segments. Core
injects an `appsrc` at each internal boundary between them.

**An injected boundary transports the timeline it was handed and never authors a
timestamp. Only a public, application-owned `Input` authors one.**

`appsrc` stamps from its own segment's running time, so a boundary that authors
a timestamp gives each leg of a fan-out a different clock. Video RTP then stops
agreeing with model-output metadata describing the same frame, and no
application can correct it: lowering consumes the app-declared `Input` nodes, so
`InputOptions` set by the application never reach the injected boundary.

When adding a segment-materialization path:

* Build the injected options with `injected_boundary_input_options(...)`, which
  is the single home for this invariant.
* Keep `is_live = true`. Clearing it stalls live segments.
* Leave the public `InputOptions::do_timestamp` default alone, so a pushed
  `cv::Mat` carrying no PTS still receives one at ingress.

Boundaries forward the retained `GstBuffer` zero-copy, so a timestamp that
already exists survives the crossing. Declining to author one can never remove
it.

### Input-contract specialization

Some compound pipeline Nodes have more than one safe backend representation.
The graph compiler specializes those Nodes from a statically established
`OutputSpec`; it does not mutate the public Graph or infer a permanent topology
from the first runtime sample. A `Derived` or `Authoritative` contract may
select an optimized representation. `Hint`, unknown format/memory, or a missing
backend capability selects the conservative representation.

For example, raw `VideoSender` omits its NV12 conversion only for a stable NV12
contract in system or SiMaAI memory and when `neatencoder` advertises its
read-only `input-layout-aware=true` capability. `OutputSpec` does not currently
carry plane strides and offsets, so no memory domain bypasses that capability
gate. An absent or false capability is treated as unsupported so Core remains
safe with older Internals packages.

Raw-video geometry and physical storage layout remain separate contracts.
`OutputSpec` and caps describe visible width and height; Core must not round
those values to codec block, DMA pitch, or surface-height alignment. The
layout-aware plugin derives physical plane offsets and strides from
`GstVideoMeta` or `GstVideoInfo`, repacks when the physical contract is not
compatible, and leaves codec/hardware admission to the encoder service. This
preserves exact decoded geometry while keeping device-specific alignment out of
the public graph API.

### Parsing & launch

The library primarily uses:

* `gst_parse_launch(pipeline_string, &err)`

This provides flexibility and debuggability (you can replay the exact string with `gst-launch-1.0`).

### Running

Typical flow (`Graph::build()` / `Run`):

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
5. **Error:** Any negotiation or runtime failure becomes a `NeatError` with a `GraphReport`.

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

Push/appsrc pipelines normally use deferred teardown. Pipelines containing
driver-backed CVU or MLA stages use the bounded synchronous `NULL` transition
so accepted asynchronous submissions are reaped before `Run::close()` returns.

---

## SimaAI concurrency

SimaAI plugins support multiple pipelines per process. If you run several
pipelines concurrently, make element names unique via
`GraphOptions` or `Model` name suffixes/prefixes to avoid
GStreamer name collisions.

---

## Constraints & safety

* **Input formats must match caps**: `InputOptions` and model configs must agree on format/width/height.
  Mismatches fail fast during negotiation or when pushing inputs.
* **Capability-gated dynamic input**: runtime renegotiation is allowed only when the built graph advertises dynamic capability. `FullyDynamic` graphs can renegotiate raw-video geometry/format/fps/media caps; `IngressDynamicCvuOnly` allows geometry changes and permits format changes only when build-time downstream contract checks prove stable output behavior.
* **Dynamic within effective bounds**: `max_*` are hard ceilings; if `max_*` is unset, `width/height/depth` act as implicit ceilings.
* **Model vs Graph defaults**: both flows now resolve seed/max/byte-guard policy through `src/pipeline/internal/InputPolicy.*`; `Model` still applies its documented metadata-backed defaults (for example 1920x1080 ceilings) while `Graph` remains node-option driven unless configured.
* **`caps_override` is authoritative**: when set, renegotiation is blocked and shape changes require rebuild.

| Flow | Seed defaults | Max defaults | Byte-guard default |
| --- | --- | --- | --- |
| `Model` | preproc metadata (if present), otherwise inferred from user format/options | explicit `input_max_*`; otherwise policy defaults (for example `1920x1080`, format-derived depth) | explicit `RunOptions.max_input_bytes`, otherwise bounded estimate or elastic default from `InputPolicy` |
| `Graph` | input-node options and/or seed input sample | explicit `max_*`; otherwise implicit from seed `width/height/depth` when provided | explicit `RunOptions.max_input_bytes`, otherwise bounded estimate or elastic default from `InputPolicy` |

* **SimaAI concurrency**: multiple pipelines can run in-process; keep element names unique.

---

## Per-frame attribute propagation

Sources attach `Sample::attributes` as a nested structure in `GstSimaMeta`. Elements that
preserve a buffer carry the metadata naturally; Core boundaries that allocate or reuse a
buffer deep-copy the attributes and clear stale values. `neatdecoder` snapshots the same
frame context before decode and restores it by a daemon-provided correlation ID, so reordered
or dropped frames cannot shift attributes onto another output. A negotiated decoder/daemon
protocol owns that correlation contract; the legacy decoder protocol remains FIFO-only.

The supported user-facing paths and limits are documented in
[Per-frame attributes](../../advanced-concepts/data-model-contracts/frame_attributes.md).

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
On an error message (`GST_MESSAGE_ERROR`), it throws `NeatError` including a `GraphReport` and reproduction hints.

### DOT dumps

If enabled, the runtime can emit DOT graphs via `gst_debug_bin_to_dot_file_with_ts(...)` to a configured directory.

### Debugging playbook (production)
1. **Reproduce the pipeline**: `Graph::describe_backend()` or `last_pipeline()`.
2. **Capture a report**: `MeasureReport::to_text()` or `NeatError::report()`.
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

Use `Run::pull_tensors(...)` for ML-centric flows when you want tensor payloads instead of the full `Sample` envelope.

---

## Pipeline serialization (save/load)

Pipelines can be saved and restored as JSON:

* `Graph::save(path)` writes a versioned JSON with node kind/label/fragment/elements
* `Graph::load(path)` rehydrates nodes via a `ConfiguredNode` wrapper

The current schema is intentionally minimal and reproducible, and can evolve to richer
node configs later. This also serves as the bridge for future bindings and tooling.

---

## UX helpers

* `Graph::describe()` uses `GraphPrinter` to render a human-readable node list
* `Graph::describe_backend()` returns the gst-launch string for quick debugging

---

## Element naming & determinism

Deterministic element names are a core design principle because they enable:

* `gst_bin_get_by_name()` for sinks and key elements
* stable probe attachment
* stable diagnostics and reproducibility
* optional naming contract enforcement ("every element belongs to some node")

**Node authors must ensure**:

* fragments include stable `name=` fields when elements must be retrievable
* `element_names()` returns every explicit element name the fragment creates
* declarations and named-pad references stay synchronized

Name integrity is part of `build()` and does not depend on an earlier `validate()` call. Names are
unique across one materialized pipeline segment because framework lookups use recursive short
names. Separately parsed connected segments may reuse the same name. The framework rejects
collisions instead of renaming them because names can participate in pad and routing expressions.

Input-dependent connected segments can materialize on the first input. Their build failure is
therefore reported on the first `push()` or `pull()`, with the original `GraphReport` preserved.

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

For model-managed graph runs, stage resolution is driven by `stage-id` + manifest context.
For non-model graph runs, explicit plugin properties are the runtime control plane.

---

## Validation & contracts

Validation exists to catch issues earlier than runtime:

* `validate()` can parse and preroll (PAUSED) to detect negotiation stalls
* `contracts/` provides structured validators for "pipeline correctness"

Mandatory final launch-name checks also run in the ordinary build path. `ValidateOptions` controls
additional validation work, not whether name integrity is enforced.

For connected Graphs, `validate()` compiles endpoint topology but does not fabricate launch strings
for input-dependent segments. Each segment receives the mandatory check when its real input
contract is available and the segment materializes.

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

* Prefer adding fields to `DiagCtx` and `GraphReport`
* If updates happen from streaming threads, use **atomics** (or another lock-free mechanism)
* Convert to plain snapshot types for reporting

---

## Dependency rules (non-negotiable)

* `builder/` should not depend on GStreamer or `pipeline/`
* `gst/` should not depend on `pipeline/`
* `nodes/` should not depend on `pipeline/` (Nodes are build-time descriptions, not runtime orchestrators)
* `pipeline/` is the orchestrator and can depend on `gst/`, `builder/`, `nodes/`, `contracts/`, `policy/`, model internals

This keeps the architecture modular and prevents circular dependencies.

---

## Tests & examples

* `examples/` show typical end-to-end usage patterns:

  * decode RTSP
  * run model archive
  * run RTSP server
* `tests/` verify critical behaviors:

  * file read paths
  * group expansion equivalence (input groups)
  * tensor output path + save/load round-trip
  * `model_resnet50_multi_test` validates Model accuracy with multiple Graph/Run instances

When adding features, prefer adding tests that:

* reproduce the pipeline string deterministically
* validate caps negotiation assumptions
* ensure failures produce useful `GraphReport` diagnostics

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
