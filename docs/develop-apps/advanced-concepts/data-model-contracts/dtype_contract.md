---
title: The dtype contract
description: How tensor dtype, quantization, tessellation, and public payload contracts fit together
sidebar_position: 2
slug: /develop-apps/advanced-concepts/dtype_contract
---

# The dtype contract

A Neat model route has two contracts:

- the **public contract** your app sees through `Tensor`, `Sample`, `InputOptions`, model specs, and graph endpoints;
- the **model route contract** Neat resolves from the compiled model archive and the selected preprocess/postprocess path.

Do not assume every public boundary is FP32. Some boundaries carry images, encoded media, packed detection payloads, INT8 tensors, BF16 tensors, or application-defined tensor semantics. Inspect the specs first; the specs are the contract.

Inside the route, Neat inserts quantization, tessellation, cast, detessellation, dequantization, and postprocess stages when the compiled model contract requires them.

## The four MLA input cases

A model archive tells Neat two important things about the first MLA stage:

- the MLA input dtype, usually **BF16** or **INT8**;
- whether MLA-side tessellation is already part of the compiled kernel.

That gives four preprocess graph families:

| MLA dtype | MLA tess | Preprocess graph family | What Neat inserts before the MLA |
| --- | --- | --- | --- |
| BF16 | yes | `Preproc` | Resize, color convert, normalize. The MLA stage tessellates internally. |
| BF16 | no | `Tess` | Resize, color convert, normalize, tessellate. |
| INT8 | yes | `Quant` | Resize, color convert, normalize, quantize. The MLA stage tessellates internally. |
| INT8 | no | `QuantTess` | Resize, color convert, normalize, quantize, tessellate. |

Inspect [`ResolvedPreprocessPlan`](/reference/cppapi/structs/simaai-neat-resolvedpreprocessplan) to see what the planner chose.

## What tessellation means

Tessellation arranges tensor bytes into the tile geometry the MLA input scratchpad expects. It is a layout transformation: same logical tensor, different memory order.

The matching detessellation happens after MLA output when the route needs to return a natural tensor layout to the next stage or to the app.

## Boundary upgrades

Neat can add higher-level route stages on top of the four-case dtype decision:

- **Generic Preproc**: uses `PreprocessOptions` to apply resize, color, layout, normalize, quantize, tessellate, or explicit transform intent before inference.
- **BoxDecode**: decodes detection heads for models that need a detection postprocess stage. The app selects the family with `BoxDecodeType`, such as `YoloV8`, and filtering fields such as `score_threshold`, `nms_iou_threshold`, and `top_k`.

These upgrades change which kernels run and what output contract the app receives. For example, a raw model output tensor and a decoded detection tensor are not the same public contract.

## What this means for app code

- Inspect `model.input_specs()` and `model.output_specs()` before allocating inputs or decoding outputs.
- Use `ModelOptions.preprocess` to state what kind of input you provide: image input, tensor input, resize, color, layout, normalization, quantization, or tessellation intent.
- Use `model.resolved_preprocess_plan()` / `model.preprocess_plan()` to see what Neat planned from your options plus the model archive.
- Do not assume output dtype, shape, or layout. Read the output spec and, when needed, the returned tensor metadata.
- Decode boxes, pose, or segmentation only when the output contract is the matching packed format.
- Treat INT8/BF16/tessellation details as route behavior unless a public spec or tensor explicitly exposes them.

No vibes. Read the contract, then move the bytes.

## Decode output deliberately

Use the decode helper that matches the output contract.

| Output contract | C++ | Python |
| --- | --- | --- |
| Raw tensor | Use the returned `Tensor` / `TensorList` directly | Use the returned tensor directly, or `to_numpy(...)` / `to_torch(...)` |
| Packed boxes | `simaai::neat::decode_bbox(...)` | `pyneat.decode_bbox(...)` |
| Packed pose | `simaai::neat::decode_pose(...)` | `pyneat.decode_pose(...)` |
| Packed segmentation | `simaai::neat::decode_segmentation(...)` | `pyneat.decode_segmentation(...)` |

Decoded boxes use a float32 `[N, 6]` tensor with columns `x1`, `y1`, `x2`, `y2`, `score`, and `class_id`. Pose and segmentation decoders return the boxes plus task-specific tensors for keypoints or masks.

## Preserve coordinate metadata

Detection coordinates often need preprocessing metadata to map from model space back to source-frame space. Preserve metadata through the graph when you use letterbox, resize, ROI lists, render, or detection decode.

Relevant metadata can include target size, scaled size, padding, color conversion, axis permutation, normalization, quantization, tessellation, ROI windows, and per-ROI affine transforms.

If decoded boxes land in the wrong place, check metadata propagation before you blame NMS. See [Data formats](/develop-apps/advanced-concepts/data_formats#preprocess-metadata-and-roi-breadcrumbs) and [Preproc ROI Lists](/reference/preproc_roi).

## Related types

- [`PreprocessOptions`](/reference/cppapi/structs/simaai-neat-preprocessoptions) — application preprocess intent.
- [`ResolvedPreprocessPlan`](/reference/cppapi/structs/simaai-neat-resolvedpreprocessplan) — what the planner compiled.
- [`PreprocessGraphFamily`](/reference/cppapi/files/include-model-preprocessplan-h) — which preprocess family was chosen.
- [`Tensor`](/reference/{lsa}/structs/simaai-neat-tensor) — public tensor payload and metadata.
- [`Sample`](/reference/{lsa}/structs/simaai-neat-sample) — payload plus runtime metadata.

## Further reading

- [Tensor and Sample](/develop-apps/development-workflow/core_types)
- [Data formats](/develop-apps/advanced-concepts/data_formats)
