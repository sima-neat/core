---
title: The dtype contract
description: How tensors flow through preprocess, MLA inference, and postprocess — the FP32-in / INT8-or-BF16-MLA / FP32-out pipeline.
sidebar_position: 1
---

# The dtype contract

The Neat framework's pipelines have a deliberately small dtype vocabulary at their public boundaries:

- **Inputs to preprocess**: FP32 (or images that the framework converts to FP32 before feeding the model).
- **Inside the MLA**: either **INT8** (with quant/dequant at the boundary) or **BF16** (no quant/dequant — straight through).
- **Outputs from postprocess**: FP32, ready for the application.

Everything else — quantization, tessellation, layout conversion — is an *internal* transformation the framework inserts when the model's MPK contract demands it. This page explains the four corners of that contract and how the planner picks the right preprocess graph family.

## The four cases

A model's MPK contract tells the framework two things about its first MLA stage: the **MLA input dtype** (BF16 or INT8) and whether **MLA-side tessellation** is part of the compiled kernel. That gives four combinations:

| MLA dtype | MLA tess | Preprocess graph family the planner picks | What the framework inserts before the MLA |
|-----------|----------|-------------------------------------------|--------------------------------------------|
| BF16      | yes      | `Preproc`                                  | Resize, color convert, normalize. The MLA stage tessellates internally. |
| BF16      | no       | `Tess`                                     | Resize, color convert, normalize, **tessellate**. |
| INT8      | yes      | `Quant`                                    | Resize, color convert, normalize, **quantize**. The MLA stage tessellates internally. |
| INT8      | no       | `QuantTess`                                | Resize, color convert, normalize, **quantize**, **tessellate**. |

The planner picks one of these four `PreprocessGraphFamily` values when building the preprocess Node. See [`PreprocessGraphFamily`](/reference/cppapi/files/include-model-preprocessplan-h) in the C++ reference and [`ResolvedPreprocessPlan`](/reference/cppapi/structs/simaai-neat-resolvedpreprocessplan) for the field.

## What "tessellation" means here

Tessellation is the tile-shuffle that arranges a tensor into the geometry the MLA's input scratchpad expects. It's a pure layout transformation — same bytes, different order. The planner inserts a tess node only when the MLA's compiled kernel does **not** include tessellation in its first op (the "MLA tess" column above).

The matching **detessellation** happens after the MLA stage if the MLA's compiled output kernel doesn't include detess. The same four-case table applies on the output side, with `Detess`/`Dequant`/`DetessDequant`/passthrough as the dual operations.

## Boundary upgrades — Generic Preproc and BoxDecode

Two upgrades the planner can apply on top of the four-case decision:

- **Generic Preproc**: when the application supplies arbitrary user-defined transforms (`PreprocessOptions::transforms`), the planner upgrades the chosen graph family to a "generic" variant that fuses those transforms with the standard preprocess. The contract at the MLA boundary doesn't change; the upgrade only affects what runs *before* the boundary.
- **BoxDecode**: a postprocess upgrade that fuses NMS / decode steps for detection models. Visible to the application as a `BoxDecodeType` and a `DetectionMeta` on the output sample. See [`BoxDecodeType.h`](/reference/cppapi/files/include-pipeline-boxdecodetype-h) and [Tutorial 006: read detection boxes](/tutorials/006-read-detection-boxes).

Both upgrades preserve the FP32-in / FP32-out vocabulary at the application boundary; they only change which kernels run inside the framework.

## Application-visible consequences

What the dtype contract means in practice for application code:

- **Sample-level dtype is FP32 at every public boundary.** The application writes FP32 inputs into samples and reads FP32 outputs from them. INT8/BF16 only ever exists inside the framework.
- **The application never sees tessellated tensors at the public API.** Tessellation/detess is an internal layout — `Tensor` objects you push or pull are always in their natural layout (HWC, CHW, etc.).
- **Conversion costs are visible only via tracing.** If you want to know whether the planner inserted a quantize or tess kernel, enable a [`ConversionTraceCollector`](/reference/cppapi/structs/simaai-neat-conversiontracecollector). Each insertion shows up as a `ConversionTrace` entry with its `ConversionKind`.

## Related types

- [`PreprocessOptions`](/reference/cppapi/structs/simaai-neat-preprocessoptions) — application intent.
- [`ResolvedPreprocessPlan`](/reference/cppapi/structs/simaai-neat-resolvedpreprocessplan) — what the planner compiled.
- [`PreprocessGraphFamily`](/reference/cppapi/files/include-model-preprocessplan-h) — which of the four corners was chosen.
- [`ConversionKind`](/reference/cppapi/files/include-pipeline-tensorconversion-h) — what kind of conversion the framework inserted.

## Further reading

- "Tessellation, quant, and cast" — §17 of the design deep dive ([Architecture](/contribute/architecture#17-tessellation-quant-and-cast)).
- "Input planner" — §82 of the design deep dive.
