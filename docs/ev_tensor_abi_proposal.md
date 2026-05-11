# EV Tensor ABI Proposal

Date: 2026-04-02
Status: proposal only
Scope: host/firmware ABI design for EV kernels

## Goal

Replace graph-specific EV config structs with:
- one shared ABI header
- one shared tensor descriptor
- one shared physical-layout descriptor
- one optional quantization descriptor
- small per-op parameter structs

This is intended for all EV kernels, not only tess/detess.

## Why Change

The current EV graph ABIs have drifted in two bad ways:
- different graphs evolved different struct shapes
- some graphs are half-migrated and carry both legacy scalar fields and array-based fields

Examples from the current tree:
- [tessellate_config.h](/home/docker/sima-cli/internals/sima-ai-cvu-sw/graphs/tessellate/config/tessellate_config.h) mixes array fields with duplicated legacy scalar fields.
- [detessellate_config.h](/home/docker/sima-cli/internals/sima-ai-cvu-sw/graphs/detessellate/config/detessellate_config.h) is already array-based but different from tess.
- [quantize_config.h](/home/docker/sima-cli/internals/sima-ai-cvu-sw/graphs/quantize/config/quantize_config.h) and [dequantize_config.h](/home/docker/sima-cli/internals/sima-ai-cvu-sw/graphs/dequantize/config/dequantize_config.h) repeat the same pattern with different per-op fields.
- [detessdequant_config.h](/home/docker/sima-cli/internals/sima-ai-cvu-sw/graphs/detessdequant/config/detessdequant_config.h) is a larger multi-head variant of the same idea.

The result is more ABI drift, harder roundtrip validation, and easier host/device mismatch.

## Design Principles

1. Logical tensor view and physical storage layout are different concepts.
2. `sizes + strides` are authoritative for the logical tensor, like PyTorch.
3. Tiled EV storage needs its own first-class descriptor.
4. Bus addresses should be the preferred canonical address domain for EV dispatch.
5. Per-op configs should stay small and readable.
6. Versioning must be explicit.

## Proposed Header

Draft header:
- [EvTensorAbi.h](/home/docker/sima-cli/core/include/pipeline/EvTensorAbi.h)

The proposal introduces:
- `sima_ev_abi_header`
- `sima_ev_storage_desc`
- `sima_ev_strided_desc`
- `sima_ev_tiled_desc`
- `sima_ev_quant_desc`
- `sima_ev_tensor_desc`

Then each kernel embeds those descriptors in a small op-specific config, for example:
- `sima_ev_tess_config_v1`
- `sima_ev_detess_config_v1`
- `sima_ev_quantize_config_v1`
- `sima_ev_dequantize_config_v1`
- `sima_ev_detessdequant_config_v1`
- `sima_ev_preproc_config_v1`

## Why This Shape

### Better Than Parallel Arrays

Parallel arrays like:
- `input_width_array[32]`
- `input_height_array[32]`
- `tile_width_array[32]`
- `input_dtype_array[32]`

are compact, but they are brittle:
- fields drift independently
- host/device generation mismatches are easy
- validation is awkward
- multi-IO graphs duplicate logic

`array-of-structs` is better than `struct-of-parallel-arrays` for ABI stability here.

### Better Than One Giant Universal Config Blob

One giant `transform_config` for every EV graph is too blunt:
- `preproc` has crop/resize/color params that do not belong to `quantize`
- `quantize` has rounding/saturation params that do not belong to `tess`
- `detessdequant` naturally wants many input/output descriptors

So the right layering is:
- shared descriptor family
- per-op wrappers

## Mapping To Current EV Kernels

### Tessellate

Logical input:
- dense NDHWC or NCHW-style tensor described by `logical.sizes` and `logical.strides_bytes`

Physical output:
- tiled descriptor with explicit tile sizes, tile traversal order, channel blocking, and tile alignment

Recommended config:
- `sima_ev_tess_config_v1`

### Detessellate

Logical output:
- dense tensor described exactly the same way as tess input

Physical input:
- tiled descriptor that must exactly match tess output layout

Recommended config:
- `sima_ev_detess_config_v1`

This is the key contract win: tess and detess can roundtrip through the same descriptor family.

### Quantize

Logical input:
- dense tensor

Logical output:
- dense tensor

Quantization:
- carried in `sima_ev_quant_desc`

Per-op params:
- rounding mode
- saturation mode

Recommended config:
- `sima_ev_quantize_config_v1`

### Dequantize

Logical input:
- dense tensor with quant metadata

Logical output:
- dense tensor with floating-point dtype

Recommended config:
- `sima_ev_dequantize_config_v1`

### QuantTess

This should not invent a new descriptor shape.
It should be one op that:
- consumes a dense logical tensor with quant metadata
- produces a tiled output tensor

That can use the same `sima_ev_tensor_desc` building blocks plus a small combined-op wrapper.

### DetessDequant

This is the main multi-head case.
It should use:
- one shared ABI header
- a list of input tensor descriptors
- a list of output tensor descriptors
- a per-head params block where needed

Recommended config:
- `sima_ev_detessdequant_config_v1`

## Address Domain

Why `bus` is the preferred canonical domain:
- EV hardware dereferences device-visible addresses, not host CPU physical addresses
- the current platform already pins several working graphs to `bus/bus`
- the current shared helper reflects that in [configManagerCommon.h](/home/docker/sima-cli/internals/sima-ai-soc-pipeline/config_manager/include/configManagerCommon.h)
- the current policy note records graph-specific correctness differences in `internals/sima-ai-soc-pipeline/config_manager/doc/evxx_graph_address_policy.md`

The proposal still keeps `addr_space` explicit during migration.

## Migration Plan

1. Freeze the descriptor vocabulary
- land the shared header
- do not change live kernels yet

2. Convert host builders first
- make config-manager emit the new descriptors in parallel with old configs
- add roundtrip/validation helpers from the shared descriptors

3. Convert single-IO kernels
- tess
- detess
- quantize
- dequantize

4. Convert combined and multi-head kernels
- quanttess
- detessdequant
- preproc

5. Delete legacy per-graph ABI structs
- once host and firmware both use the shared descriptor family

## Non-Goals

- This proposal does not try to encode every preproc algorithm detail into the shared tensor descriptor.
- This proposal does not force every kernel to use one identical top-level config blob.
- This proposal does not remove graph-specific validation; it gives those validators a common descriptor vocabulary.

## Recommendation

Adopt the shared descriptor ABI in:
- [EvTensorAbi.h](/home/docker/sima-cli/core/include/pipeline/EvTensorAbi.h)

Then migrate EV kernels in this order:
1. tessellate
2. detessellate
3. quantize
4. dequantize
5. quanttess
6. detessdequant
7. preproc

That gives the cleanest path to one durable EV ABI without repeating the current half-legacy, half-array drift.
