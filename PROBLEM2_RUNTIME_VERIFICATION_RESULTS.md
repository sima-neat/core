# Problem 2 Runtime/Code Verification Results

Date: 2026-05-28

Scope: investigation only. This file records the verification pass requested before
implementing the generic terminal output contract plan. No production-source change
is included in this verification note.

## Executive result

The core Problem 2 mechanism is now verified from the actual Ecorobotix/Mark
artifact plus the `core_graph_changes` code path:

1. Mark's app executes a graph that is terminal at MLA:
   `Input -> Preproc -> MLA -> Output`.
2. Mark's MPK still describes downstream nodes after that MLA boundary:
   `MLA_0 -> slice_MLA_0_slice_transform -> APU_1`.
3. The current MLA boundary resolver follows that downstream slice while building
   MLA output contracts, even though the app's executed endpoint stops at MLA.
4. The slice output has shape `[1, 768, 1024, 1]` and size `3,145,728` bytes.
   Because the current typed-object parser ignores MPK typed objects such as
   `{"scalar":"int32","shape":[...]}`, dtype falls back to bytes-per-element
   inference. Four bytes per element is currently inferred as `FP32`.
5. The actual MLA physical output node is `12,582,912` bytes. So the public
   metadata can say `Float32 768x1024x1` while the backing GstMemory is still the
   raw MLA physical boundary buffer.
6. Owned and zero-copy can then diverge because owned materialization copies the
   selected GstMemory's mapped size, while zero-copy exposes a logical view with
   the stale/derived tensor metadata.

This confirms that the long-term fix must be endpoint/terminal-stage driven and
must not be an MLA-only special case. Slices/views are valid internal edge
contracts only when the downstream kernel they feed is actually executed.

## Verification limitations

Live DevKit runtime checks are blocked in this container:

- `dk` is not installed in this environment.
- `/usr/local/bin/dk` and `/root/.devkit-sync.rc` are absent.
- `/usr/local/bin/devkit.sh` exists, but it is the interactive DevKit setup
  script, not the canonical `dk` runner.
- Existing EVO runner binaries in this workspace are `ARM aarch64`; per the
  execution rule they were not run locally.

So the following still require a connected DevKit/runtime pass:

- the exact `GstBuffer` memory count/order for Mark's terminal MLA sample;
- whether `logical.physical_index` currently equals the correct GstMemory index
  in all routes;
- actual runtime owned-vs-zero-copy dumps for Mark on the current tree;
- current live EVO 12x4 result on this exact checkout.

The root-cause chain above does not depend on those live facts; it is proven by
the artifact contract and current source logic. The remaining runtime pass is
needed to harden the implementation and test coverage.

## Artifact verified

Original shared artifact:

```text
https://drive.google.com/file/d/1ChS1IT6xOn_jQR4M_Pncv-FRKnaS-oRw/view?usp=sharing
```

Downloaded/extracted locally under:

```text
tmp/mark_drive/ecorobotix-model-testing.zip
tmp/mark_drive/extracted/ecorobotix-model-testing/
tmp/mark_drive/models_unpacked/
```

Relevant extracted models:

```text
tmp/mark_drive/models_unpacked/mod_mpk/
tmp/mark_drive/models_unpacked/new-2.0-mod-rand-gs_mpk/
tmp/mark_drive/models_unpacked/new-2.1-mod-rand-gs_mpk/
```

## Mark app endpoint verification

The extracted app builds this explicit graph:

```text
Input(model.input_appsrc_options(false))
  -> Preproc(PreprocOptions(model))
  -> nodes::groups::MLA(model)
  -> Output(EveryFrame(max_buffers=4))
```

Evidence:

- `tmp/mark_drive/extracted/ecorobotix-model-testing/apps/simple_graph_mask.cpp:143-150`
- `tmp/mark_drive/extracted/ecorobotix-model-testing/src/ecorobotix_config.cpp:32`

The model options set:

```text
opt.inference_terminal.mla_only = true
```

`core_graph_changes/src/model/ModelPack.cpp` applies terminal policy in
`ModelPack::infer_block(...)`:

- `resolve_terminal_index_or_throw(...)` resolves `mla_only` by finding the last
  `ExecutionStageKind::Mla` in the infer sequence.
- `infer_seq.resize(terminal_idx + 1)` drops all later infer stages.

Evidence:

- `core_graph_changes/src/model/ModelPack.cpp:3233-3317`
- `core_graph_changes/src/model/ModelPack.cpp:3830-3863`

Conclusion: the Ecorobotix app is not supposed to publish APU or post-process
outputs. Its public endpoint is terminal at the MLA producer.

## MPK contract facts

All three extracted MPKs share the same critical output contracts.

### `mod_mpk`

Pipeline sequence:

```text
MLA_0 pipeline: preproc -> processmla
```

Manifest facts:

```text
MLA_0 output node:
  name=MLA_0
  size=12,582,912 bytes

slice_MLA_0_slice_transform:
  input node=MLA_0, size=12,582,912 bytes
  output node=slice_MLA_0_slice_transform, size=3,145,728 bytes
  begin=[0,0,0,0]
  end=[1,768,1024,1]
  input_shape=[1,768,1024,4]
  output_shape=[1,768,1024,1]

APU_1:
  input node=slice_MLA_0_slice_transform, size=3,145,728 bytes
  output node=APU_1/output_, size=6,291,456 bytes
  input_types=[{scalar=int32, shape=[1,768,1024,1]}]
  output_types=[{scalar=int64, shape=[1,768,1024,1]}]
```

### `new-2.0-mod-rand-gs_mpk`

Pipeline sequence:

```text
MLA_0 pipeline: preproc -> processmla
APU_1 pipeline: processtvm, input from processmla
```

Critical contract facts are the same as `mod_mpk`:

```text
MLA_0 size=12,582,912
slice output size=3,145,728, output_shape=[1,768,1024,1]
APU input_types=int32 [1,768,1024,1]
APU output_types=int64 [1,768,1024,1]
```

### `new-2.1-mod-rand-gs_mpk`

Pipeline sequence:

```text
MLA_0 pipeline: quanttess -> processmla -> detessellate
APU_1 pipeline: processtvm, input from processmla
```

Critical contract facts are again the same:

```text
MLA_0 size=12,582,912
slice output size=3,145,728, output_shape=[1,768,1024,1]
APU input_types=int32 [1,768,1024,1]
APU output_types=int64 [1,768,1024,1]
```

### Size arithmetic

```text
12,582,912 = 1 * 768 * 1024 * 4 lanes * 4 bytes
 3,145,728 = 1 * 768 * 1024 * 1 lane  * 4 bytes
 6,291,456 = 1 * 768 * 1024 * 1 lane  * 8 bytes
```

So the MPK itself makes clear there are two different concepts:

- raw terminal MLA physical boundary: `12,582,912` bytes;
- downstream slice/APU logical input: `3,145,728` bytes, `int32`,
  `[1,768,1024,1]`.

For Mark's MLA-terminal app, the public output must be the first concept, not the
second.

## Dtype provenance verification

Current parser behavior explains the reported false `Float32`.

`MpkContract.cpp::read_string_values_any(...)` only reads strings or arrays of
strings. Arrays of typed objects are ignored.

Evidence:

- `core_graph_changes/src/pipeline/internal/sima/MpkContract.cpp:439-454`
- `core_graph_changes/src/pipeline/internal/sima/MpkContract.cpp:473-484`

Mark's useful dtype information is encoded as typed objects:

```json
{"scalar":"int32","shape":[1,768,1024,1]}
{"scalar":"int64","shape":[1,768,1024,1]}
```

Because these are objects, the existing string reader does not extract `int32` or
`int64` from them.

Then `MpkContract.cpp::infer_dtype_from_shape_and_size(...)` infers dtype from
shape and bytes-per-element:

- 1 byte -> `INT8`
- 2 bytes -> `BF16`
- 4 bytes -> `FP32`

Evidence:

- `core_graph_changes/src/pipeline/internal/sima/MpkContract.cpp:662-704`

For the slice:

```text
elements = 1 * 768 * 1024 * 1 = 786,432
size     = 3,145,728 bytes
bytes/element = 4
current inferred dtype = FP32
```

Conclusion: the observed `Float32 768x1024x1` is not a random runtime corruption.
It is the deterministic result of typed-object metadata being ignored and the
slice's four-byte elements being guessed as FP32.

## Why the downstream slice leaks into the terminal MLA public output

`resolve_mla_boundary_tensor_views_local(...)` in `MpkContract.cpp` starts from
MLA output, follows the earliest outgoing edge, and treats slice/batch-flatten
stages as boundary views.

Evidence:

- `core_graph_changes/src/pipeline/internal/sima/MpkContract.cpp:6662-6748`
- `core_graph_changes/src/pipeline/internal/sima/MpkContract.cpp:6806-6880`
- `core_graph_changes/src/pipeline/internal/sima/MpkContract.cpp:7124-7196`

For Mark's MPK the path is:

```text
boundary root = MLA_0
earliest edge = MLA_0 -> slice_MLA_0_slice_transform
slice is classified as a boundary view
boundary tensor becomes slice_MLA_0_slice_transform output
next edge is slice -> APU_1, which is a real/non-view consumer
resolver stops, keeping the slice output as the MLA boundary tensor
```

That behavior is correct for an internal `MLA -> slice -> APU` handoff when the
APU stage is actually executed. It is wrong for a public appsink endpoint that
terminates before the APU. In the terminal-public case, the slice is future graph
metadata and must not redefine the output.

This is the exact place where the planned fix must become endpoint-aware: the
contract selector must know whether it is selecting a public terminal endpoint or
an internal producer-to-consumer boundary.

## MlaStaticContract physical/logical split verification

`MlaStaticContractExtractor.cpp::build_output_contracts(...)` builds dispatcher
physical outputs directly from MPK physical outputs, and separately builds
published/logical outputs from the selected logical contract.

Evidence:

- `core_graph_changes/src/pipeline/internal/sima/MlaStaticContractExtractor.cpp:789-817`
- `core_graph_changes/src/pipeline/internal/sima/MlaStaticContractExtractor.cpp:831-895`
- `core_graph_changes/src/pipeline/internal/sima/MlaStaticContractExtractor.cpp:920-990`
- `core_graph_changes/src/pipeline/internal/sima/stagesemantics/ProcessMlaStageSemantics.cpp:159-234`

For the Mark terminal case under current logic:

```text
dispatcher physical output comes from MLA_0:
  size=12,582,912

selected logical output can come from downstream slice:
  dtype=FP32 after fallback inference
  shape=[768,1024,1] after batch dimension normalization
  logical size=3,145,728
```

This produces a real split between backing capacity and exposed logical tensor
metadata. That split is not inherently bad for views/padded/packed outputs, but
it is bad when the logical contract was selected from a future non-executed node.

## Owned vs zero-copy path verification

Current output override/materialization behavior explains why owned and
zero-copy can have different apparent spans.

`TensorUtil.cpp::copy_tensor_from_sample_memory(...)`:

- picks a `GstMemory` from the sample buffer;
- maps it with `gst_memory_map`;
- allocates CPU owned storage of `map.size`;
- copies the entire mapped `map.size`;
- preserves the reference tensor metadata before any later override overlay.

Evidence:

- `core_graph_changes/src/pipeline/tensor/TensorUtil.cpp:1261-1378`

`TensorUtil.cpp::tensor_view_from_sample_memory(...)`:

- keeps a `GstSample`/GstMemory-backed storage holder;
- selects the GstMemory by memory index;
- returns a read-only direct view into that memory;
- preserves/overlays the logical tensor metadata.

Evidence:

- `core_graph_changes/src/pipeline/tensor/TensorUtil.cpp:1380-1425`

`OutputTensorOverride.h::apply_output_tensor_override_entry(...)` overlays shape,
dtype, byte offset, and strides from the override entry after selecting/copying
sample memory.

Evidence:

- `core_graph_changes/src/pipeline/internal/OutputTensorOverride.h:97-138`

Important multi-tensor risk: the current `TensorSet` multi-output branch mostly
patches names/routing and only fills shape/layout/stride when absent. It does not
fully rewrite dtype/shape/stride for already-populated multi-tensor metadata.

Evidence:

- `core_graph_changes/src/pipeline/internal/OutputTensorOverride.h:168-232`

Conclusion: even after selecting the correct public terminal contract, the fix
must make public overrides authoritative for stale TensorSet metadata, otherwise
multi-output/padded-view cases can keep old metadata.

## Physical index vs GstMemory index verification

The source currently routes override entries with `entry.memory_index` and often
sets route physical index to the same value.

Evidence:

- `core_graph_changes/src/pipeline/internal/OutputTensorOverride.h:216-221`
- `core_graph_changes/src/pipeline/tensor/TensorUtil.cpp:1292-1298`
- `core_graph_changes/src/pipeline/tensor/TensorUtil.cpp:1404-1407`

This is not a proof that `physical_index == GstMemory index` for every route. It
is a current assumption. The implementation plan therefore remains correct to
require an explicit physical-index-to-memory-index map or a validation step before
reusing physical indexes as memory indexes.

## EVO matrix verification

Existing matrix coverage was inspected:

- Script: `tmp/run_evo_route_matrix_12x4.sh`
- Runner: `core_graph_changes/tests/graph_migration/legacy/evo_route_matrix_legacy_runner.cpp`

The matrix covers 12 EVO models across 4 routes:

```text
EV74-A65
A65-A65
A65-EV74
EV74-EV74
```

Model families include EV74 tessellation, MLA tessellation, MLA multi-buffer,
BF16, INT8, and v2 variants.

However, the current legacy runner only checks that `Model::run()` succeeds and
prints output counts. It does not print/validate dtype, shape, strides, byte
span, byte offset, memory index, physical index, or backing GstMemory size.

Evidence:

- `core_graph_changes/tests/graph_migration/legacy/evo_route_matrix_legacy_runner.cpp:86-100`

Historical result available in this workspace:

```text
tmp/graph_phase1_hardware_matrix_20260516T152724Z/logs/evo_route_matrix_12x4_ci_like_fixed_env_20260516T165523Z.log
```

Parsed historical summary:

```text
cases=48
PASS=48
TIMEOUT=0
RC failures=0
NO_RESULT=0
outputs distribution: 24 cases with 25 outputs, 24 cases with 28 outputs
EVO_MATRIX_SUMMARY failures=0 total=48
```

Live run on this exact checkout is blocked by missing `dk` and by ARM64 runner
binaries. The historical matrix is useful regression context but cannot prove the
metadata/span properties involved in Problem 2.

Plan impact: add an augmented EVO metadata/span probe before implementation and
keep running the 12x4 matrix after each successful change. The augmented probe
must log, for every output tensor:

```text
kind, dtype, shape, strides_bytes, byte_offset, logical size_bytes,
storage capacity/mapped size, route logical_index, physical_index, memory_index,
segment_name, tensor name, and whether owned/zero-copy agree logically
```

## Host/unit verification attempt

A host-side build was attempted for `unit_mla_boundary_output_contract_test` to
exercise the contract code without DevKit hardware. The build currently fails in
unrelated existing source/header mismatch around `pad_value`:

```text
PreparedRuntimeBuild.cpp:3176:11: error:
  'struct simaai::neat::GraphProcessCvuStageRequest' has no member named 'pad_value'

SimaPluginStaticManifest.cpp:1492:35: error:
  'SimaPluginProcessCvuStagePayload' has no member named 'pad_value'
```

Captured log:

```text
tmp/mark_verify_build_unit_mla_boundary_output_contract_test.log
```

This blocks local unit runtime verification in the current dirty checkout. It
does not change the static root-cause evidence above.

## What this proves for the implementation plan

The plan direction is still the right one, but the verification sharpened the
requirements:

1. **Public endpoint contract selection must be endpoint-aware.** It must use the
   actual terminal producer of each public output/appsink endpoint, not global MPK
   downstream order and not generic MLA boundary walking.
2. **Internal edge contracts must still include view transforms.** For an executed
   `MLA -> slice -> APU` edge, the slice describes the consumer handoff and must
   remain available to the internal route.
3. **Public terminal contracts must not include future view transforms.** For an
   executed `MLA -> Output` endpoint, a downstream MPK slice must not redefine the
   public tensor.
4. **Dtype parsing needs provenance.** Typed-object dtypes must be parsed as
   explicit metadata, and element-size fallback (`4 bytes -> FP32`) must be marked
   as inferred and treated as weaker evidence.
5. **Public overrides must be authoritative.** Owned and zero-copy must agree on
   logical dtype/shape/stride/offset/span even if the backing storage capacity is
   larger.
6. **Memory routing needs validation.** Do not assume physical index equals
   GstMemory index unless the rendered route proves it or a dense map is built.
7. **EVO matrix needs a metadata/span probe.** The existing pass/fail runner is
   not sufficient to catch this class of bug, especially for slicing, striding,
   padded views, and multi-buffer outputs.

## Remaining runtime checks to perform once `dk` is available

1. Run the Mark MLA-terminal graph in both owned and zero-copy output modes and
   dump per-output metadata plus GstMemory sizes.
2. Confirm whether Mark's terminal GstBuffer has one memory block of
   `12,582,912` bytes or multiple memories/segments.
3. Confirm the physical-index-to-memory-index relationship for Mark and for EVO
   multi-buffer models.
4. Run the augmented EVO metadata/span probe on all 12x4 cases.
5. Re-run the normal EVO 12x4 matrix after each successful source change.
6. Add contract-only tests for:
   - terminal MLA with downstream slice/APU metadata;
   - terminal EV74 after a real materialized transform;
   - internal `producer -> slice/view -> consumer` edge;
   - multi-output TensorSet with stale metadata;
   - nonzero byte offset;
   - padded/strided view;
   - multiple physical buffers/GstMemory blocks.
