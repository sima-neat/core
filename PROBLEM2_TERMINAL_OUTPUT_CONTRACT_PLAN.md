# Problem 2 Plan: Generic Terminal Output Contract Selection

## Status

Planning only. No implementation in this document.

All future implementation work for this plan must happen inside:

```text
/home/docker/sima-cli/core_graph_changes
```

Do not make production-source changes in sibling copies such as `core/`, repo
root scratch files, or packaged/extracted artifacts.

## Multi-agent review hardening

This plan was reviewed by three independent codebase agents plus local review. The
core direction was validated, but the implementation plan must be hardened in the
following ways before any code is written:

1. **Endpoint-aware from day one.** Do not implement a generic utility whose only
   behavior is "walk the manifest backward and pick the last real stage". That is
   acceptable only as an explicitly scoped single-output-segment fallback. The
   public contract selector must accept an endpoint selector or terminal-stage
   identity when available, and must use rendered route/binding information rather
   than global MPK order.
2. **Public/internal decision must exist before override selection.** The flag
   distinguishing public appsinks from graph-internal transport appsinks must be
   plumbed into input/source build preparation, or all override selection must be
   deferred until after manifest compilation. Setting the flag only after
   `session_build_prepare_build_input_context(...)` is too late because legacy
   override installation currently happens inside that preparation path.
3. **Do not reuse file-local publish code by wishful thinking.**
   `build_publish_contract_from_manifest_stage_local(...)` is inside an anonymous
   namespace in `PreparedRuntimeBuild.cpp`, and it mutates physical sizes upward
   when logical span exceeds the declared physical buffer. Public terminal
   publication needs either a small extracted shared builder with strict/no-grow
   mode, or a direct `StageStaticSpec -> OutputTensorOverride` conversion with
   explicit validation.
4. **Materialized transforms are real producers.** Cast/dequant/detess may be
   actual materialized ProcessCVU stages. Do not demote them to view-only by
   substring matching. Boundary/view classification must prefer manifest/runtime
   semantics first and use string tokens only as a conservative fallback.
5. **Public overrides must be authoritative even for TensorSet metadata.** The
   current multi-tensor branch in `OutputTensorOverride.h` mostly patches names
   and routing. If the incoming TensorSet metadata is stale/downstream-derived,
   that is not enough. The implementation must either bypass stale TensorSet meta
   when a public override exists or fully rewrite dtype/shape/stride/offset/count
   from `OutputTensorOverrideEntry`.
6. **Physical index is not automatically memory index.** Every use of
   `logical.physical_index` as `memory_index` must validate that physical outputs
   are dense and ordered like GstBuffer memory blocks, or build an explicit
   physical-index-to-memory-index map.
7. **Dtype provenance matters.** After MPK parsing, code must know whether dtype
   was explicit, typed-object-derived, alias-derived, or inferred from size. A
   public terminal contract must not trust an element-size-only `FP32` inference.
8. **Unsupported public dtypes need explicit policy.** Typed objects may contain
   dtypes such as `int64`, while current public/runtime enums may not represent
   all such types. Do not silently coerce unsupported typed-object dtypes to
   `UInt8` or `Float32`; either add support deliberately, publish raw bytes with
   diagnostic, or fail validation.
9. **A65/.so genericity is future-facing unless represented today.** Current
   `StagePayloadKind` does not have a first-class A65/shared-object payload. This
   plan must not add a fake special case just to claim support. Treat existing
   represented stages generically now; add a real payload/materialization model
   for host/A65 stages only when the codebase actually represents those stages.
10. **Owned vs zero-copy equality means logical equality.** Backing storage
    capacity may differ, but dtype, shape, stride bytes, byte offset, route
    metadata, and logical readable span must match. Tests must cover nonzero
    byte offsets, padded strides, multiple GstMemory blocks, and stale TensorSet
    metadata.

Official GStreamer references supporting the runtime assumptions:

- appsink owns an internal queue controlled by `max-buffers`, `max-bytes`,
  `max-time`, and leaky/drop behavior:
  https://gstreamer.freedesktop.org/documentation/app/appsink.html
- `GstBuffer` carries metadata and one or more `GstMemory` blocks; buffer/pool
  lifetime is reference-counted:
  https://gstreamer.freedesktop.org/documentation/gstreamer/gstbuffer.html
- `GstMemory` has valid `size`, `offset`, `maxsize`, and map/unmap lifetime
  semantics:
  https://gstreamer.freedesktop.org/documentation/gstreamer/gstmemory.html
- `GstSample` is the application exchange object containing typed memory plus
  timing/caps information:
  https://gstreamer.freedesktop.org/documentation/gstreamer/gstsample.html

## Problem statement

Problem 2 is not an MLA-specific issue. The observed failure happened when the
executed graph ended at MLA, but the metadata exposed to the user was influenced
by downstream MPK nodes that were not part of the executed graph.

Observed case:

```text
Executed graph:
  Input -> Preproc -> MLA -> Output/appsink

MPK also describes downstream route:
  MLA -> slice/view -> A65/TVM/post stage
```

The public zero-copy tensor exposed a direct view into GStreamer pipeline memory.
The metadata reported a logical tensor like:

```text
Float32, 768x1024x1
```

but the payload behaved like integer class IDs / raw MLA boundary data. Owned and
zero-copy paths then diverged because they materialized/mapped different spans of
the same underlying physical buffer.

The root issue is:

> Public output metadata must be selected from the actual terminal producer of
> the executed graph endpoint, not from future/downstream MPK stages that are not
> executed.

This must be solved generically for any terminal producer, not just MLA.

## Core design rule

For every public `Output` / appsink endpoint:

> The public tensor contract comes from the actual terminal producer for that
> endpoint.

The terminal producer may be any real compute/materialization stage:

- MLA / `processmla`
- EV74 / ProcessCVU real kernel
- A65 `.so` stage
- boxdecode
- dequant / detess-dequant if actually executed
- cast if actually materialized
- any future real stage type

The rule must not be:

```text
if MLA is terminal, do X
```

The rule must be:

```text
for each public output endpoint, resolve the executed terminal producer and
publish that producer's boundary contract.
```

## Important slice/view caveat

Slices and similar operators require special conceptual handling.

Although they may appear as operators in the MPK, many of them are really
memory-boundary transformations between stages:

- slice
- batch-flatten
- unpack/view
- pass-through view
- offset view
- naming/view adapters

They should not automatically become public terminal output contracts.

Correct rule:

> A slice/view transform is included only when it describes the memory boundary
> between a producer and an actual downstream kernel in the executed graph.

So:

```text
MLA -> slice -> A65 kernel
```

If `A65 kernel` is executed, the slice is part of the MLA-to-A65 edge contract.
It tells the A65 stage which offset/shape/stride to consume.

But:

```text
MLA -> Output
```

If the MPK contains `MLA -> slice -> A65` but the executed graph stops at MLA,
the slice must not redefine the public output. The public output is the MLA
terminal boundary.

## Stage-role model

We should classify stages/MPK nodes by role, not by hard-coded special cases.
Classification must be based on explicit rendered/runtime semantics first:

- `StagePayloadKind`;
- `processcvu.graph_family_enum`;
- whether the stage has a real materialized runtime contract;
- whether it owns `physical_outputs` and valid producer-local `logical_outputs`;
- stage bindings/route identity.

String tokens such as `slice` or `batchflatten` may be used only as a
conservative fallback, never as the primary authority. Ambiguous stages should
not be guessed into a new public contract.

Proposed roles:

### 1. Real compute/materialization stage

A stage that materially produces a runtime output or owns a meaningful public
boundary.

Examples:

- MLA
- ProcessCVU real EV graph
- A65 `.so`
- boxdecode
- materialized dequant/cast/detess stages

### 2. Boundary/view transform

A no-copy or mostly no-copy memory projection that is meaningful only as an edge
adapter between a producer and consumer.

Examples:

- slice
- batch-flatten
- unpack view
- offset view
- pass-through transport view

### 3. Transport adapter

Pipeline/transport-only element whose purpose is buffering, caps, routing, or
handoff rather than semantic output definition.

Examples:

- queue
- capsfilter
- identity-like adapters
- internal appsink/appsrc transport edges

## Contract types to keep separate

The current bug comes partly from mixing several contract concepts.

We should explicitly separate:

### A. Producer physical contract

What a stage physically emits.

Example:

```text
MLA raw physical output buffer
EV74 physical output buffer
A65 physical output buffer
```

### B. Inter-stage transport/consumer contract

What the next executed real stage expects to consume.

This is where boundary/view transforms belong.

Example:

```text
MLA physical output + slice offset/shape/stride -> A65 input
```

### C. Public endpoint contract

What the user sees at public `Output` / appsink.

This must be based on the terminal producer of that public endpoint.

## Existing code pieces to reuse/generalize

The codebase already has several pieces pointing in the right direction.

### Generic stage contract carrier

File:

```text
core_graph_changes/src/pipeline/internal/sima/SimaPluginStaticManifest.h
```

Relevant type:

```cpp
StageStaticSpec
```

Useful fields:

- `physical_outputs`
- `logical_outputs`
- `output_order`
- `logical_inputs`
- `input_bindings`
- `payload_kind`
- `plugin_kind`
- `kernel_kind`

This is the correct generic data model to build on.

### Generic publish-contract builder

File:

```text
core_graph_changes/src/pipeline/internal/sima/PreparedRuntimeBuild.cpp
```

Relevant function:

```cpp
build_publish_contract_from_manifest_stage_local(...)
```

This already converts a `StageStaticSpec` into a
`TensorBufferPublishContract`, but two constraints are important:

1. it currently has internal linkage inside an anonymous namespace, so new files
   cannot call it directly;
2. it currently grows `physical_outputs[*].size_bytes` upward when a logical
   span exceeds the physical size. That behavior is useful for some prepared
   runtime paths, but it is unsafe as public-terminal validation because it can
   hide an invalid or downstream-derived logical contract.

Implementation must therefore choose one of two surgical options:

- extract a small shared internal publish-contract builder with an explicit
  `StrictNoPhysicalExpansion` mode for public terminal selection; or
- convert `StageStaticSpec` directly into `OutputTensorOverride` inside the
  terminal-output utility and keep all span/physical validation strict there.

Do not duplicate a large second publish-contract subsystem, and do not add
MLA-specific output override logic.

### Existing terminal-stage idea in ProcessCVU semantics

File:

```text
core_graph_changes/src/pipeline/internal/sima/stagesemantics/ProcessCvuStageSemantics.cpp
```

Relevant functions:

```cpp
find_terminal_stage_after_outputs_local(...)
terminal_output_tensor_for_index_local(...)
published_output_name_from_terminal_or_producer_local(...)
```

These are evidence that the code already needs terminal-stage reasoning. The
logic should be lifted/generalized into shared contract utilities.

### Existing rendered terminal query

File:

```text
core_graph_changes/src/pipeline/internal/RenderedMlaContractQuery.h
```

Relevant function:

```cpp
terminal_output_info(...)
```

This walks rendered manifest stages backward and chooses the last stage with
logical outputs. The idea is useful, but it is currently too projection-oriented
and lives in an MLA-named helper. The long-term path should return/derive a full
publish contract for the public endpoint.

## Current problematic areas

### Whole-MPK MLA boundary lifting

Files/functions:

```text
core_graph_changes/src/pipeline/internal/sima/MpkContract.cpp

get_mla_logical_outputs_contract(...)
get_mla_published_outputs_contract(...)
resolve_mla_boundary_tensor_views_local(...)
```

These helpers can walk from MLA into downstream MPK boundary/view stages such as
slice or detess. That is correct only when those downstream stages are part of
the executed edge/consumer relationship.

It is wrong for public output when the executed graph stops before those stages.

### Ad-hoc output overrides

File:

```text
core_graph_changes/src/pipeline/graph/GraphBuildInput.cpp
```

Relevant functions:

```cpp
build_detess_output_override(...)
build_mla_output_override(...)
build_cast_output_override(...)
maybe_apply_detess_output_override(...)
```

These should not grow more special cases. Long term, they should either be
removed or backed by the same generic terminal/public endpoint contract selector.

### Owned vs zero-copy divergence

Files:

```text
core_graph_changes/src/pipeline/gst/InputStreamPull.cpp
core_graph_changes/src/pipeline/tensor/TensorUtil.cpp
core_graph_changes/src/pipeline/internal/OutputTensorOverride.h
```

Current behavior can make owned and zero-copy expose different effective spans:

- owned copy may copy the whole selected `GstMemory`;
- zero-copy maps based on logical shape/stride/span.

After the generic contract fix, both paths must use the same public endpoint
contract and expose identical dtype/shape/stride/span semantics.

### Unsafe dtype inference

File:

```text
core_graph_changes/src/pipeline/internal/sima/MpkContract.cpp
```

Relevant function:

```cpp
infer_dtype_from_shape_and_size(...)
```

Current behavior infers 4-byte geometry as `FP32`. That is unsafe because 4-byte
data may be `INT32`, `UINT32`, or something else. Public output should prefer
explicit terminal-stage metadata. If dtype is genuinely unknown, we should avoid
inventing `Float32`.

## Proposed high-level implementation plan

### Phase 1: Add a generic public endpoint contract query

Add a shared utility conceptually like:

```cpp
select_public_output_contract(rendered_manifest, output_endpoint)
```

Responsibilities:

1. Start from an actual public `Output` / appsink endpoint.
2. Resolve the upstream executed producer path for that endpoint using rendered
   route/binding information where available.
3. Walk backward through transport adapters.
4. Do not treat boundary/view transforms as public terminal producers unless
   they are actual materialized runtime stages.
5. Select the nearest real compute/materialization producer for that endpoint.
6. Build a strict public publication contract from that stage.

This must be endpoint-aware, not only "last stage in manifest", because future
graphs may branch or expose multiple outputs. If the current runtime only gives
us a single rendered segment with one appsink, the implementation may expose that
as a narrow fallback path, but the API should not pretend a whole-manifest
reverse scan is generally endpoint-safe.

### Phase 2: Add edge-contract resolution for real consumer edges

For internal edges:

```text
real producer -> zero or more view transforms -> real consumer
```

the edge contract should include view transforms because the downstream real
consumer requires them.

This is where slice/batch-flatten/unpack/offset-view metadata belongs.

Public output selection and internal edge selection should therefore be separate
queries.

### Phase 3: Reuse or extract the generic publish-contract path carefully

Once the terminal producer `StageStaticSpec` is selected, public metadata should
come from one shared contract-building path. Because the existing
`build_publish_contract_from_manifest_stage_local(...)` is file-local and
non-strict, either extract it into a shared internal helper with strict/no-grow
validation or convert directly to `OutputTensorOverride` while reusing only the
small dtype/layout/span helpers needed.

If the selected producer has only physical output metadata and no trustworthy
logical tensor metadata, expose a conservative raw/physical contract rather than
guessing a logical Float32 tensor.

### Phase 4: Replace/retire ad-hoc output override logic

Do not add more special cases to:

```cpp
build_mla_output_override(...)
build_cast_output_override(...)
build_detess_output_override(...)
```

Instead, route these through the generic terminal/public endpoint contract
selector.

The goal is:

```text
Output override = projection of selected public endpoint contract
```

not:

```text
if last node kind is ModelFragment/Cast/Detess, manually patch metadata
```

### Phase 5: Make owned and zero-copy consume the same contract

Both materialized and zero-copy output tensors must be derived from the same
selected public endpoint contract.

Required invariant:

```text
owned dtype/shape/strides/logical span == zero-copy dtype/shape/strides/logical span
```

Physical storage ownership can differ, but public tensor semantics must not.

### Phase 6: Tighten dtype handling

Rules:

1. Prefer explicit dtype from selected terminal stage contract.
2. Parse typed MPK metadata objects, not only string-array forms.
3. Do not infer `FP32` merely because element size is four bytes.
4. If dtype is unknown, expose unknown/raw/byte semantics or fail diagnostics
   rather than silently lying.

## Detailed implementation plan

This section is the concrete implementation guide. Each step should be small,
reviewable, and validated before moving to the next one.

### Mandatory validation gate after every implementation step

After **every** implementation step below:

1. Make the source changes only under `core_graph_changes/`.
2. Build and run the step-specific tests listed in that step.
3. Run the EVO matrix.
4. If and only if the step-specific tests and EVO matrix pass, commit the
   successful checkpoint in git before starting the next step.

Do not continue to the next step if any route fails, times out, or reports no
`EVO_RESULT`. Do not commit a failed or partially validated step.

Use the existing EVO matrix runner/script. The runner is `aarch64`, so always
check it with `file` and run it on the DevKit via `dk`; do not execute it on the
x86 host.

```bash
cd /home/docker/sima-cli

cmake --build core_graph_changes/build-codex-graph-sdk \
  --target evo_route_matrix_legacy_runner

file core_graph_changes/build-codex-graph-sdk/tests/evo_route_matrix_legacy_runner

mkdir -p core/tmp
cp core_graph_changes/build-codex-graph-sdk/tests/evo_route_matrix_legacy_runner \
  core/tmp/evo_route_matrix_runner
cp core_graph_changes/build-codex-graph-sdk/tests/evo_route_matrix_legacy_runner \
  core/tmp/evo_route_matrix_legacy_runner

dk /home/docker/sima-cli/tmp/run_evo_route_matrix_12x4.sh
```

Expected result:

```text
12 models * 4 routes = 48 summaries
exactly 48 SUMMARY lines
exactly 48 status=PASS lines
no TIMEOUT
no RC_*
no NO_RESULT
```

Save each step's EVO output log under a temporary/run-log path and inspect it
before committing. EVO proves broad routing health, but it does not replace the
step-specific owned/zero-copy contract parity tests above.

If `dk` is not found, check `/usr/local/bin/dk` and source
`/root/.devkit-sync.rc` before concluding it is unavailable.

### Git checkpoint discipline

After each successful implementation step, create a small revertable commit from
inside `core_graph_changes`:

```bash
cd /home/docker/sima-cli/core_graph_changes
git status --short
# Review the diff carefully. Add only files intentionally changed for this step.
git diff -- <changed-files>
git add <changed-files>
git commit -m "<short step-specific message>"
```

Important rules:

- Do not use `git add -A` blindly; this checkout may contain unrelated dirty
  files from other work.
- Commit only the files intentionally changed for the current step.
- Keep one successful implementation step per commit.
- If validation fails, do not commit. Revert only that step's touched files with
  explicit paths, for example:

  ```bash
  git checkout -- <files-touched-by-failed-step>
  ```

- Before starting the next step, `git status --short` should show no uncommitted
  files from the terminal-output-contract work.

For the non-EVO unit/e2e binaries listed in each step, use the same execution
discipline:

```bash
file <test-binary>
```

- If `file` reports `x86-64`, run the test directly on the host.
- If `file` reports `aarch64` / `ARM aarch64`, run it through:

  ```bash
  dk /home/docker/sima-cli/tmp/devkit_env_exec.sh <test-binary>
  ```

Never run an `aarch64` test binary directly on the x86 host.

### Step 1: Add an internal terminal-output contract utility

Add a new internal utility rather than extending the MLA-named rendered query
helper.

Proposed files:

```text
core_graph_changes/src/pipeline/internal/TerminalOutputContractQuery.h
core_graph_changes/src/pipeline/internal/TerminalOutputContractQuery.cpp
```

`CMakeLists.txt` already uses `file(GLOB_RECURSE ... src/*.cpp)`, so a new `.cpp`
under `src/` will be picked up automatically.

Proposed namespace:

```cpp
namespace simaai::neat::pipeline_internal::terminal_output_contract {
```

Proposed public-internal API:

```cpp
enum class StagePublicationRole {
  RealProducer,
  MaterializedTransform,
  BoundaryView,
  TransportOnly,
};

struct PublicOutputEndpointSelector {
  // Optional in the first implementation, but explicit in the API so branches
  // and multi-output segments do not require another redesign.
  std::string terminal_stage_key;
  std::string output_segment_name;
  int output_slot = -1;
  int route_slot = -1;
};

StagePublicationRole classify_stage_for_publication(
    const sima::StageStaticSpec& stage);

const sima::StageStaticSpec* find_terminal_real_producer_for_endpoint(
    const sima::SimaPluginStaticManifest& manifest,
    const PublicOutputEndpointSelector& endpoint);

sima::StageStaticSpec make_publication_stage_for_terminal(
    const sima::StageStaticSpec& terminal_stage);

bool validate_publication_stage_strict(
    const sima::StageStaticSpec& publication_stage,
    std::string* error_message);

std::optional<OutputTensorOverride> build_output_override_from_manifest(
    const sima::SimaPluginStaticManifest& manifest,
    const PublicOutputEndpointSelector& endpoint = {},
    std::string* error_message = nullptr);
```

Implementation details:

1. `find_terminal_real_producer_for_endpoint(...)`
   - If the endpoint selector contains a stage key/route slot/output segment,
     use it to identify the terminal route and walk upstream through
     `StageStaticSpec::input_bindings` where possible.
   - If no endpoint identity is available and the rendered manifest is known to
     be a single-output linear segment, use a clearly documented fallback that
     iterates `manifest.stages` backward.
   - Skip stages classified as `TransportOnly`.
   - Skip stages classified as `BoundaryView` unless a view stage is explicitly
     represented as a materialized runtime stage.
   - Return the nearest `RealProducer` or `MaterializedTransform` for the selected
     endpoint.

2. `classify_stage_for_publication(...)`
   - Use generic stage properties first:
     - `stage.payload_kind`
     - `stage.plugin_kind`
     - `stage.kernel_kind`
     - `stage.processcvu.graph_family_enum`
     - presence of real `physical_outputs` / `logical_outputs`
     - bindings/route identity
   - `TransportOnly` examples:
     - empty payload / no logical or physical outputs;
     - pass-through transport-only wrappers;
     - pipeline adapters that do not own semantic output.
   - `BoundaryView` examples:
     - confirmed slice/view/batch-flatten/offset projection stages that do not
       materialize a new public buffer.
   - `RealProducer` / `MaterializedTransform` examples:
     - `StagePayloadKind::ProcessMla`;
     - `StagePayloadKind::ProcessCvu` when it is an actual materialized processcvu
       graph, including materialized cast/dequant/detess;
     - `StagePayloadKind::BoxDecode`;
     - `StagePayloadKind::Dequant`;
     - future A65/`.so` payloads only once they are represented by real stage
       metadata.
   - Do not classify a stage as view-only solely because its name contains
     `cast`, `dequant`, `detess`, or similar.

3. `build_output_override_from_manifest(...)`
   - Find the terminal real producer for the selected endpoint.
   - Convert it to a publication stage with
     `make_publication_stage_for_terminal(...)`.
   - Validate the publication stage with strict/no-physical-expansion rules.
   - Convert its logical outputs to `OutputTensorOverride`.

Required validation after this step:

```bash
cd /home/docker/sima-cli
cmake --build core_graph_changes/build-codex-graph-sdk \
  --target unit_contract_render_manifest_equivalence_test \
           unit_mla_boundary_output_contract_test
file core_graph_changes/build-codex-graph-sdk/tests/unit_contract_render_manifest_equivalence_test
file core_graph_changes/build-codex-graph-sdk/tests/unit_mla_boundary_output_contract_test
```

Then run the mandatory EVO matrix gate above. After it passes, commit this step's intentional changes under `core_graph_changes` before continuing.

### Step 2: Implement producer-local publication stage construction

The selected terminal producer must be converted into the contract the user
should see at public `Output`.

Proposed helper:

```cpp
sima::StageStaticSpec make_publication_stage_for_terminal(
    const sima::StageStaticSpec& terminal_stage);
```

Rules:

1. Default path:
   - Copy the terminal stage.
   - Use its existing `physical_outputs`, `logical_outputs`, and
     `output_order`.
   - This keeps existing postproc/EV74/dequant/boxdecode paths working.

2. Raw physical fallback path:
   - If terminal logical outputs are missing, invalid, or clearly not
     producer-local, synthesize public logical outputs from physical outputs.
   - This is generic; it is not `if MLA terminal`.
   - It should work for any stage whose terminal output is only described as
     physical buffers.

3. Producer-local check:
   - Every logical output must reference an existing physical output.
   - Build and validate an explicit physical-index-to-vector/memory-index map;
     do not assume `physical_index == vector position` unless the physical outputs
     are dense and sorted.
   - `byte_offset + physical_span(shape, stride, dtype)` must fit inside the
     selected physical output.
   - The dtype must be explicit or typed-object/alias-derived; an element-size
     inference alone is not trustworthy for public terminal publication.
   - If a logical output's segment/name clearly points to a downstream
     view-transform stage while the selected terminal producer is different,
     treat it as non-producer-local and use raw physical fallback.

4. Synthesized raw logical output format:
   - One logical tensor per selected physical output.
   - `logical_index = i`
   - `physical_index = physical.physical_index >= 0 ? physical.physical_index : i`
   - `output_slot = i`
   - `byte_offset = 0`
   - `size_bytes = physical.size_bytes`
   - `shape = {physical.size_bytes}`
   - `stride_bytes = {1}`
   - `dtype = "UINT8"`
   - `layout = "HW"` or empty/unknown if consumers handle unknown layout better.
   - names use `physical.segment_name` if present, otherwise `output_i`.

5. ProcessMLA details under the generic fallback:
   - `StageStaticSpec::processmla.dispatcher_output_sizes` carries the actual
     dispatcher/producer buffer sizes and is the most important producer-local
     fact for terminal raw MLA publication.
   - First validate whether `stage.physical_outputs` is producer-local and agrees
     with dispatcher sizes/names. If it is already downstream-view-influenced or
     cannot be validated, do not trust it for public raw fallback.
   - Prefer dispatcher sizes/names for raw terminal MLA fallback when producer
     locality is uncertain. Use `stage.physical_outputs` only when it clearly
     matches dispatcher raw buffers and provides useful physical indices/segment
     names.
   - If dispatcher names are absent for a single output, synthesize a stable name
     from the physical segment name, MPK output node name, or `output_i`; do not
     require multi-output-only dispatcher names.
   - If `stage.physical_outputs` is empty but dispatcher sizes exist, synthesize
     physical outputs from:

     ```cpp
     stage.processmla.dispatcher_output_names
     stage.processmla.dispatcher_output_sizes
     ```

   - For Mark's MPK, the raw MLA output has:

     ```text
     output_nodes[0].name = "MLA_0"
     output_nodes[0].type = "buffer"
     output_nodes[0].size = 12582912
     ```

     The public terminal publication should therefore be a raw physical buffer
     contract of `12582912` bytes unless a trustworthy producer-local logical
     dtype/shape exists.

6. Do **not** use downstream slice/A65 metadata in this fallback. That metadata
   belongs only to an edge contract when an actual downstream real consumer is
   executed.

Required unit tests for this step:

Add a new test, preferably:

```text
core_graph_changes/tests/unit_testing/unit_terminal_output_contract_query_test.cpp
```

Test cases:

1. Manifest with terminal stage whose logical outputs are normal:
   - output override uses logical shape/dtype/stride unchanged.
2. Manifest with terminal stage whose logical output names/segments point to a
   downstream `slice_*` view but physical output is raw:
   - output override uses raw physical `{size_bytes}`/`UINT8` contract.
3. Manifest ending in a view-transform stage:
   - selector skips it and selects the previous real producer.
4. Manifest ending in a real ProcessCVU/dequant stage:
   - selector does not raw-fallback if the logical contract is producer-local.

Required validation after this step:

```bash
cd /home/docker/sima-cli
cmake --build core_graph_changes/build-codex-graph-sdk \
  --target unit_terminal_output_contract_query_test \
           unit_mla_boundary_output_contract_test \
           unit_detessdequant_output_info_test \
           unit_model_output_spec_contract_test
file core_graph_changes/build-codex-graph-sdk/tests/unit_terminal_output_contract_query_test
file core_graph_changes/build-codex-graph-sdk/tests/unit_mla_boundary_output_contract_test
file core_graph_changes/build-codex-graph-sdk/tests/unit_detessdequant_output_info_test
file core_graph_changes/build-codex-graph-sdk/tests/unit_model_output_spec_contract_test
```

Then run the mandatory EVO matrix gate. After it passes, commit this step's intentional changes under `core_graph_changes` before continuing.

### Step 3: Convert publication contracts to `OutputTensorOverride`

The current output override type is:

```text
core_graph_changes/src/pipeline/internal/OutputTensorOverride.h
```

Relevant fields:

```cpp
struct OutputTensorOverrideEntry {
  std::vector<int64_t> shape;
  std::vector<int64_t> strides_bytes;
  int64_t byte_offset = 0;
  int memory_index = -1;
  int logical_output_index = -1;
  int route_slot = -1;
  TensorDType dtype = TensorDType::UInt8;
  TensorLayout layout = TensorLayout::Unknown;
  std::string format;
  std::string name;
  std::string segment_name;
};
```

Implementation details:

1. Convert each `LogicalTensorStaticSpec` in the publication stage into one
   `OutputTensorOverrideEntry`.
2. Resolve `entry.memory_index` through a validated physical-index-to-memory-index
   map. Only set `memory_index = logical.physical_index` when the physical output
   vector is dense/sorted and that identity mapping has been proven.
3. Set `entry.logical_output_index = logical.logical_index`.
4. Set `entry.route_slot = logical.output_slot`.
5. Set `entry.byte_offset = logical.byte_offset`.
6. Preserve `shape`, `stride_bytes`, `dtype`, `layout`, `logical_name`, and
   `segment_name`.
7. If dtype/layout conversion returns unsupported/unknown for an explicit typed
   object, follow the explicit unsupported-dtype policy: deliberate enum support,
   raw-byte publication with diagnostic, or validation failure. Never silently
   map unsupported explicit dtype to `Float32` or `UInt8`.
8. Add local dtype/layout conversion helpers rather than reusing unrelated caps
   projection helpers:

   ```cpp
   TensorDType tensor_dtype_from_static_token(std::string_view dtype);
   TensorLayout tensor_layout_from_static_token(std::string_view layout);
   ```

9. Unknown dtype rule:
   - If the terminal contract had explicit unknown dtype but physical bytes are
     available, use synthesized raw `UINT8` publication with a diagnostic.
   - Do not default unknown four-byte data to `Float32`.

Required validation after this step:

```bash
cd /home/docker/sima-cli
cmake --build core_graph_changes/build-codex-graph-sdk \
  --target unit_terminal_output_contract_query_test \
           unit_tensor_set_meta_route_contract_test \
           unit_tensorbuffer_runtime_contract_identity_overlay_test
file core_graph_changes/build-codex-graph-sdk/tests/unit_terminal_output_contract_query_test
file core_graph_changes/build-codex-graph-sdk/tests/unit_tensor_set_meta_route_contract_test
file core_graph_changes/build-codex-graph-sdk/tests/unit_tensorbuffer_runtime_contract_identity_overlay_test
```

Then run the mandatory EVO matrix gate. After it passes, commit this step's intentional changes under `core_graph_changes` before continuing.

### Step 4: Integrate only for public appsink endpoints

This step must be careful because public appsinks and graph-internal appsinks
are not the same thing.

Graph-internal appsinks are transport edges. They must preserve consumer-edge
metadata, including slice/view transforms needed by the next executed real
consumer.

Add an internal flag:

```text
core_graph_changes/src/pipeline/internal/InputStream.h
```

```cpp
bool public_output_contract = true;
```

Meaning:

- `true`: this appsink is a user-visible public output; apply terminal public
  endpoint publication.
- `false`: this appsink is graph-internal transport; do not apply terminal
  public publication overrides.

Update runtime segment startup and build preparation:

```text
core_graph_changes/src/pipeline/runtime/RunCore.cpp
core_graph_changes/src/pipeline/runtime/RunCoreGraphStart.cpp
```

Important ordering: do **not** wait until after
`session_build_prepare_build_input_context(...)` to set this decision if legacy
override selection is still in the preparation path. Either remove/defer all
override installation from preparation first, or pass the decision into
preparation.

Concrete rule:

```cpp
const bool public_output_contract = !segment.boundary.graph_internal_output;
```

Pass that value into both push-style and source-like build paths. Graph-internal
output policy is already identified in `RunCoreGraphStart.cpp` through
`seg.boundary.graph_internal_output`; keep that as the single source of truth for
transport appsinks.

For source-like segment paths, pass the same boolean into source-stream build
instead of deriving it from public `RunOptions`.

Update internal build signatures:

```text
core_graph_changes/src/pipeline/graph/internal/GraphBuildInternal.h
core_graph_changes/src/pipeline/graph/GraphBuildInput.cpp
core_graph_changes/src/pipeline/graph/GraphBuildSource.cpp
```

Required signature/plumbing change:

```cpp
BuildInputContext session_build_prepare_build_input_context(...,
                                                            bool public_output_contract);
SourceStreamBuildContext session_build_source_stream_internal(...,
                                                             bool public_output_contract,
                                                             const char* where);
```

If the exact function ordering changes while implementing, the invariant remains:
`InputStreamOptions::public_output_contract` is set before any legacy or generic
output override can be installed.

Required behavior:

1. Direct `Graph::build(...)->Output` remains public by default.
2. Public terminal segment in runtime graph uses terminal publication override.
3. Graph-internal segment output does **not** use terminal publication override.
4. Internal MLA->slice->A65 transport still sees the slice/view metadata when
   the A65 stage is actually executed.
5. Source-like graph-internal routes get the same protection as push-style
   graph-internal routes.

Required validation after this step:

```bash
cd /home/docker/sima-cli
cmake --build core_graph_changes/build-codex-graph-sdk \
  --target unit_terminal_output_contract_query_test \
           graph_migration_phase3_source_segment_run_core_test \
           graph_migration_phase3_public_graph_branching_test \
           graph_migration_phase3_connected_yolo_route_test
file core_graph_changes/build-codex-graph-sdk/tests/unit_terminal_output_contract_query_test
file core_graph_changes/build-codex-graph-sdk/tests/graph_migration_phase3_source_segment_run_core_test
file core_graph_changes/build-codex-graph-sdk/tests/graph_migration_phase3_public_graph_branching_test
file core_graph_changes/build-codex-graph-sdk/tests/graph_migration_phase3_connected_yolo_route_test
```

Then run the mandatory EVO matrix gate. After it passes, commit this step's intentional changes under `core_graph_changes` before continuing.

### Step 5: Replace early ad-hoc overrides with manifest-based terminal override

Current early override path:

```text
core_graph_changes/src/pipeline/graph/GraphBuildInput.cpp

prepare_build_input_context(...)
  -> maybe_apply_detess_output_override(...)
      -> build_cast_output_override(...)
      -> build_mla_output_override(...)
```

This happens before the final build result has a rendered manifest. The generic
terminal selector needs the rendered manifest, so the new hook should be after:

```cpp
maybe_compile_build_result_contracts(&br, ...)
```

Concrete input-mode integration point:

```text
core_graph_changes/src/pipeline/graph/GraphBuildInput.cpp
function: run_input_stream_internal_typed(...)
```

Immediately after `maybe_compile_build_result_contracts(...)` and before
creating `InputStream`, do the manifest-based selection using the same mutable
`InputStreamOptions stream_opt` that will be passed to `InputStream::create`. Do
not create a late shadow copy that loses the public/internal decision.

Conceptually:

```cpp
InputStreamOptions stream_opt = opt;
if (stream_opt.public_output_contract && br.rendered_manifest.has_value()) {
  terminal_output_contract::PublicOutputEndpointSelector endpoint =
      make_endpoint_selector_from_build_result(br);
  std::string terminal_error;
  auto terminal_override =
      terminal_output_contract::build_output_override_from_manifest(*br.rendered_manifest,
                                                                    endpoint,
                                                                    &terminal_error);
  if (terminal_override.has_value()) {
    // Generic manifest-derived public contract is authoritative. It may replace
    // a legacy override installed earlier during the migration window.
    stream_opt.output_override = std::move(*terminal_override);
  }
}
```

Then remove or demote the early call:

```cpp
maybe_apply_detess_output_override(nodes, ctx.stream_opt);
```

Migration approach:

1. First wire generic manifest override after manifest compilation.
2. Legacy override installation must be either removed from preparation or gated
   by `public_output_contract` and treated only as fallback when the generic
   selector returns no override. It must not prevent a successful generic
   manifest override from replacing stale MLA/cast/detess special-case metadata.
3. Once tests/EVO matrix are green, remove the MLA/cast special-case builders or
   make them private fallback tests-only code.

Concrete source-mode integration point:

```text
core_graph_changes/src/pipeline/graph/GraphBuildSource.cpp
function: prepare_source_pipeline_from_nodes(...)
```

After:

```cpp
maybe_compile_source_contracts(&br, build_nodes, sess_opt, where);
```

apply the same manifest-based terminal override if:

```cpp
stream_opt.public_output_contract && br.rendered_manifest.has_value()
```

Required validation after this step:

```bash
cd /home/docker/sima-cli
cmake --build core_graph_changes/build-codex-graph-sdk \
  --target unit_terminal_output_contract_query_test \
           unit_graph_io_contract_test \
           graph_deterministic_routing_regression_test \
           stage_routing_regression_test \
           output_appsink_policy_test
file core_graph_changes/build-codex-graph-sdk/tests/unit_terminal_output_contract_query_test
file core_graph_changes/build-codex-graph-sdk/tests/unit_graph_io_contract_test
file core_graph_changes/build-codex-graph-sdk/tests/graph_deterministic_routing_regression_test
file core_graph_changes/build-codex-graph-sdk/tests/stage_routing_regression_test
file core_graph_changes/build-codex-graph-sdk/tests/output_appsink_policy_test
```

Then run the mandatory EVO matrix gate. After it passes, commit this step's intentional changes under `core_graph_changes` before continuing.

### Step 6: Add MPK typed-object parsing for future terminal A65/.so outputs

The Mark MPK has typed object metadata for A65:

```json
"input_types": [
  {"scalar": "int32", "shape": [1, 768, 1024, 1]}
],
"output_types": [
  {"scalar": "int64", "shape": [1, 768, 1024, 1]}
]
```

Current parser limitation:

```text
core_graph_changes/src/pipeline/internal/sima/MpkContract.cpp
read_string_values_any(...)
```

It only reads strings/string arrays. It ignores object entries like
`{"scalar":"int32","shape":[...]}`.

Implementation details:

1. Add a helper near `read_string_values_any(...)`:

   ```cpp
   struct TypedTensorValues {
     std::vector<std::string> dtypes;
     std::vector<std::vector<std::int64_t>> shapes;
   };

   TypedTensorValues read_typed_tensor_values_any(const json& value);
   ```

2. It must support:
   - string: `"int32"`
   - array of strings: `["int32", "float32"]`
   - object: `{"scalar":"int32","shape":[...]}`
   - array of objects:
     `[{"scalar":"int32","shape":[...]}, ...]`
   - dtype aliases:
     - `scalar`
     - `dtype`
     - `type`
     - `data_type`
   - shape aliases:
     - `shape`
     - `tensor_shape`
     - `dims`

3. In plugin config parsing, after current:

   ```cpp
   input_shapes = read_shape_alias(...)
   output_shapes = read_shape_alias(...)
   input_dtypes = read_string_alias_values(...)
   output_dtypes = read_string_alias_values(...)
   ```

   merge typed-object values:

   ```cpp
   auto typed_inputs = read_typed_tensor_alias_values(*params, {"input_types", ...});
   auto typed_outputs = read_typed_tensor_alias_values(*params, {"output_types", ...});
   if (input_shapes.empty()) input_shapes = typed_inputs.shapes;
   if (output_shapes.empty()) output_shapes = typed_outputs.shapes;
   if (input_dtypes.empty()) input_dtypes = typed_inputs.dtypes;
   if (output_dtypes.empty()) output_dtypes = typed_outputs.dtypes;
   ```

4. Existing string-array behavior must remain unchanged.

5. This is needed so that, when an already-rendered host/A65/`.so` stage is the
   actual terminal producer, we can expose its own typed output contract instead
   of raw bytes. This step must not invent a fake A65 special case if the stage is
   not represented by `StageStaticSpec` yet.
6. Add dtype source/provenance while parsing or finalizing contracts, e.g.

   ```cpp
   enum class DTypeSource {
     Unknown,
     ExplicitMpk,
     TypedObject,
     Alias,
     InferredFromSize,
   };
   ```

   Public terminal publication may trust explicit/typed-object/alias sources, but
   must not trust `InferredFromSize` for `FP32`.
7. Unsupported public dtype policy must be explicit. For example, if typed object
   parsing yields `int64` and public `TensorDType`/tensorbuffer metadata cannot
   represent `Int64`, choose one documented behavior:
   - add proper enum/ABI/public support deliberately;
   - publish raw `UINT8` for that terminal with a diagnostic; or
   - fail strict contract validation.

   Do not silently coerce unsupported explicit dtype to `Float32` or `UInt8`.

Required tests:

Add to an existing MPK parser test or create:

```text
core_graph_changes/tests/unit_testing/unit_mpk_typed_tensor_object_contract_test.cpp
```

Test:

- A plugin with `output_types=[{"scalar":"int64","shape":[1,768,1024,1]}]`
  and `output_nodes[0].size=6291456`.
- Parsed output dtype must come from the typed object, not `FP32`.
- Parsed `mpk_shape == [1,768,1024,1]`.

If the framework does not currently support public `TensorDType::Int64`, the
test should document the conservative fallback policy explicitly instead of
silently converting to Float32 or UInt8.

Required validation after this step:

```bash
cd /home/docker/sima-cli
cmake --build core_graph_changes/build-codex-graph-sdk \
  --target unit_mpk_typed_tensor_object_contract_test \
           unit_mpk_contract_legacy_ingress_test \
           unit_model_output_spec_contract_test \
           unit_model_route_planner_test
file core_graph_changes/build-codex-graph-sdk/tests/unit_mpk_typed_tensor_object_contract_test
file core_graph_changes/build-codex-graph-sdk/tests/unit_mpk_contract_legacy_ingress_test
file core_graph_changes/build-codex-graph-sdk/tests/unit_model_output_spec_contract_test
file core_graph_changes/build-codex-graph-sdk/tests/unit_model_route_planner_test
```

Then run the mandatory EVO matrix gate. After it passes, commit this step's intentional changes under `core_graph_changes` before continuing.

### Step 7: Stop unsafe four-byte `FP32` inference for public contracts

Current risk:

```text
core_graph_changes/src/pipeline/internal/sima/MpkContract.cpp
infer_dtype_from_shape_and_size(...)
```

Current behavior:

```cpp
if (bytes_per_element == 4U) {
  return std::string("FP32");
}
```

Implementation approach:

1. Do not globally rip this out first if it would destabilize existing internal
   contracts. Instead, prevent public terminal publication from relying on this
   inference.
2. Add a provenance/conservative check in terminal publication:
   - if dtype was inferred only from element size, and no explicit dtype source
     exists, raw `UINT8` publication is safer than `Float32`.
3. Once EVO and unit tests are stable, tighten `infer_dtype_from_shape_and_size`
   itself:
   - keep 1-byte as byte/int8 only if existing tests require it;
   - keep 2-byte BF16 only if explicitly justified by MLA metadata;
   - remove or gate 4-byte `FP32` inference behind a source-confidence flag.

Required validation after this step:

```bash
cd /home/docker/sima-cli
cmake --build core_graph_changes/build-codex-graph-sdk \
  --target unit_terminal_output_contract_query_test \
           unit_mla_boundary_output_contract_test \
           unit_yolov8_contract_subset_test \
           unit_model_output_spec_contract_test
file core_graph_changes/build-codex-graph-sdk/tests/unit_terminal_output_contract_query_test
file core_graph_changes/build-codex-graph-sdk/tests/unit_mla_boundary_output_contract_test
file core_graph_changes/build-codex-graph-sdk/tests/unit_yolov8_contract_subset_test
file core_graph_changes/build-codex-graph-sdk/tests/unit_model_output_spec_contract_test
```

Then run the mandatory EVO matrix gate. After it passes, commit this step's intentional changes under `core_graph_changes` before continuing.

### Step 8: Prove owned and zero-copy agree

Current divergence points:

```text
core_graph_changes/src/pipeline/internal/OutputTensorOverride.h
core_graph_changes/src/pipeline/tensor/TensorUtil.cpp
core_graph_changes/src/pipeline/gst/InputStreamPull.cpp
```

Important current behavior:

- `copy_tensor_from_sample_memory(...)` copies the whole selected `GstMemory`.
- `tensor_view_from_sample_memory(...)` zero-copy maps the selected memory.
- `apply_output_tensor_override_entry(...)` then overlays shape/dtype/stride.
- The current multi-tensor TensorSet branch in `apply_output_tensor_override(...)`
  does not fully rewrite dtype/shape/stride/offset/count; it mostly patches
  routing/name fields. That is not authoritative enough when TensorSet metadata
  itself is stale/downstream-derived.

For a raw terminal publication, shape `{physical.size_bytes}` and dtype `UInt8`
make both owned and zero-copy expose the same logical span.

For a true logical terminal publication, shape/stride/dtype must already be
valid and fit within the physical memory.

Owned vs zero-copy equality is defined as logical equality:

```text
dtype, shape, strides_bytes, byte_offset, route metadata, logical readable span
```

Backing storage capacity may differ because owned mode can copy the whole
selected `GstMemory` while zero-copy keeps a view.

Implementation details:

1. Add a helper used by tests:

   ```cpp
   std::uint64_t output_override_entry_physical_span_bytes(
       const OutputTensorOverrideEntry& entry);
   ```

2. Validate each override entry before installing it:
   - dtype byte size nonzero;
   - shape and strides compatible;
   - span fits selected physical output size from the publication stage.

3. Make public overrides authoritative for TensorSet samples. Either bypass
   decoded TensorSet metadata when `output_override` exists, or fully rewrite
   every tensor from `OutputTensorOverrideEntry`, including dtype/shape/stride,
   byte offset, memory mapping, route fields, and tensor count mismatches.
4. Add runtime/contract tests that create fake tensor samples and apply the same
   override with `materialize_output=true` and `false`. Include:
   - one-memory sample;
   - multi-memory sample;
   - nonzero `byte_offset`;
   - padded strides;
   - stale TensorSet metadata with wrong dtype/shape/count.
   Expected:
   - same `dtype`;
   - same `shape`;
   - same `strides_bytes`;
   - same byte offset and route metadata;
   - same `view_read().size_bytes` / logical span.

Required validation after this step:

```bash
cd /home/docker/sima-cli
cmake --build core_graph_changes/build-codex-graph-sdk \
  --target unit_terminal_output_contract_query_test \
           unit_tensor_set_meta_route_contract_test \
           unit_tensorbuffer_runtime_contract_identity_overlay_test \
           graph_deterministic_routing_regression_test
file core_graph_changes/build-codex-graph-sdk/tests/unit_terminal_output_contract_query_test
file core_graph_changes/build-codex-graph-sdk/tests/unit_tensor_set_meta_route_contract_test
file core_graph_changes/build-codex-graph-sdk/tests/unit_tensorbuffer_runtime_contract_identity_overlay_test
file core_graph_changes/build-codex-graph-sdk/tests/graph_deterministic_routing_regression_test
```

Then run the mandatory EVO matrix gate. After it passes, commit this step's intentional changes under `core_graph_changes` before continuing.

### Step 9: Remove stale special-case override builders

Only after all previous steps and EVO gates are green:

1. Remove or deprecate:

   ```cpp
   build_mla_output_override(...)
   build_cast_output_override(...)
   build_detess_output_override(...)
   maybe_apply_detess_output_override(...)
   ```

2. Replace their callers with:

   ```cpp
   terminal_output_contract::build_output_override_from_manifest(...)
   ```

3. Keep any detess/cast behavior only if represented through normal rendered
   `StageStaticSpec` contracts.

4. Remove debug env names that only existed for old detess override debugging if
   no longer used.

Required validation after this step:

```bash
cd /home/docker/sima-cli
cmake --build core_graph_changes/build-codex-graph-sdk \
  --target unit_terminal_output_contract_query_test \
           unit_detessdequant_output_info_test \
           unit_dequant_node_fragment_test \
           unit_boxdecode_uses_upstream_cast_contract_test \
           graph_migration_unified_yolov8_variant_route_matrix_test \
           graph_migration_legacy_yolov8_variant_route_matrix_test
file core_graph_changes/build-codex-graph-sdk/tests/unit_terminal_output_contract_query_test
file core_graph_changes/build-codex-graph-sdk/tests/unit_detessdequant_output_info_test
file core_graph_changes/build-codex-graph-sdk/tests/unit_dequant_node_fragment_test
file core_graph_changes/build-codex-graph-sdk/tests/unit_boxdecode_uses_upstream_cast_contract_test
file core_graph_changes/build-codex-graph-sdk/tests/graph_migration_unified_yolov8_variant_route_matrix_test
file core_graph_changes/build-codex-graph-sdk/tests/graph_migration_legacy_yolov8_variant_route_matrix_test
```

Then run the mandatory EVO matrix gate. After it passes, commit this step's intentional changes under `core_graph_changes` before continuing.

### Step 10: Mark-model runtime proof

After the generic implementation is complete, rerun the Mark-style graph that
originally showed the issue.

Expected for graph:

```text
Input -> Preproc -> MLA -> Output
```

Public output should no longer be downstream slice-shaped unless an executed
downstream real consumer requires the slice.

Expected terminal publication for current Mark MPK if no explicit producer-local
logical dtype/shape exists:

```text
dtype  = UInt8/raw
shape  = {12582912}
span   = 12582912 bytes
source = terminal MLA producer physical output "MLA_0"
```

Owned and zero-copy should report the same public tensor metadata and logical
span.

Required validation after this step:

1. Mark runtime proof, owned.
2. Mark runtime proof, zero-copy.
3. Mandatory EVO matrix gate.
4. Commit the successful Mark-proof/test changes under `core_graph_changes` if any files changed.

Do not add Mark-specific logic to production code. This run is only proof that
the generic terminal-publication rule fixed the motivating case.

## Contract-selection examples

### Example 1: terminal MLA public output

```text
Executed:
  Preproc -> MLA -> Output

MPK also contains:
  MLA -> slice -> A65
```

Public output contract:

```text
MLA terminal producer boundary
```

Do not include the slice because no executed real consumer requires it.

### Example 2: MLA consumed by A65 through slice

```text
Executed:
  Preproc -> MLA -> slice/view -> A65 -> Output
```

Internal MLA-to-A65 edge contract:

```text
MLA physical output plus slice/view offset/shape/stride
```

Public output contract:

```text
A65 terminal output boundary
```

### Example 3: chained mixed graph

```text
Executed:
  MLA -> slice/view -> A65 -> EV74 -> MLA -> .so -> Output
```

Rules:

- slice/view applies only to the edge into A65;
- A65-to-EV74 uses the A65 output / EV74 input edge contract;
- EV74-to-MLA uses the EV74 output / MLA input edge contract;
- public output comes from final `.so`.

No global assumption about "model final output" or "MLA final output" is valid.

## Validation/invariants

Add contract-level validation before relying on runtime behavior.

For every exposed public logical tensor:

```text
byte_offset + physical_span(shape, stride, dtype) <= selected physical buffer size
```

Also validate:

- dtype must come from explicit metadata or be conservatively unknown/raw;
- dense one-lane logical views must not silently point at multi-lane physical
  parents unless stride/offset express that relationship;
- owned and zero-copy output metadata must match;
- public contract must not reference downstream stages outside the executed
  graph endpoint path;
- view transforms must only appear when there is an executed downstream real
  consumer edge.

## Test plan

### Contract-only tests

1. **Mark-style graph cut after MLA**
   - Executed graph: `Preproc -> MLA -> Output`
   - MPK contains downstream slice/A65.
   - Expected: public output contract comes from MLA terminal boundary.
   - Expected: no downstream slice/A65 shape/type leaks into public output.

2. **Same MPK with downstream consumer included**
   - Executed graph: `Preproc -> MLA -> slice -> A65 -> Output`
   - Expected: slice participates in MLA-to-A65 edge contract.
   - Expected: public output contract comes from A65.

3. **EV74 terminal**
   - Executed graph ends at ProcessCVU/EV74.
   - Expected: public output contract comes from ProcessCVU terminal stage.

4. **A65 / host terminal when represented**
   - Executed graph ends at `.so` / host stage represented by `StageStaticSpec`.
   - Expected: public output contract comes from that terminal stage.
   - If the codebase does not yet render A65/host `.so` stages as real
     `StageStaticSpec` producers, this test remains future-facing and should not
     be faked with a one-off MLA rule.

5. **Long chained graph**
   - MLA + view + `.so` + EV74 + MLA + `.so`.
   - Expected: every internal edge uses the appropriate producer/consumer
     boundary; public output comes from final real stage.

### Runtime tests

For representative graphs:

- owned output and zero-copy output report identical dtype/shape/strides;
- owned and zero-copy expose the same logical byte span;
- output decoding/hash is deterministic across owned and zero-copy when the
  public contract describes the same logical payload;
- no false `Float32` appears when the terminal contract does not explicitly say
  Float32.

## What not to do

Do not:

- add an `if MLA terminal` special case;
- keep growing `build_mla_output_override(...)`;
- treat every MPK slice as a public output transform;
- infer `Float32` from four-byte element size;
- let owned and zero-copy materialize different logical tensor spans;
- use future/downstream MPK nodes that are not in the executed graph to define
  public endpoint metadata.

## Desired end state

The runtime should have one generic rule:

```text
For public outputs, expose the selected terminal real producer's contract.
For internal edges, apply view transforms only when needed by the next executed
real consumer.
```

That should make terminal MLA, terminal EV74, terminal A65, and arbitrary future
chains all behave consistently without stage-specific hacks.
