---
title: Data Formats
description: Format tags, payload families, layout, and tensor semantics
sidebar_position: 1
slug: /develop-apps/advanced-concepts/data_formats
---

# Data formats and tensor semantics

This page explains the public format vocabulary used by `InputOptions::format`, `OutputTensorOptions::format`, tensor image metadata, and sample payload tags.

For task-level usage, start with [Tensor and Sample](/develop-apps/development-workflow/core_types). Come here when a graph boundary needs an explicit format contract.

## Format tags

`FormatTag` / `FormatSpec` names the payload format. In Python, use `pyneat.Format` or `pyneat.FormatTag` values for format fields. Do not assign raw strings to Python format fields.

Python exposes the common user-facing format tags. Some lower-level C++ tags, such as `BBOX`, `MLA`, `ARGMAX`, and `DETESSDEQUANT`, usually appear through tensor semantic metadata, payload tags, or diagnostics rather than as assignable `pyneat.Format` values.

Common tags:

| Tag | Typical payload | Meaning |
| --- | --- | --- |
| `RGB` | image | Packed RGB, 8 bits per channel. |
| `BGR` | image | Packed BGR, 8 bits per channel. OpenCV uses this by default. |
| `GRAY8` | image | 8-bit grayscale. |
| `NV12` | image/video | Y plane plus interleaved UV plane. Width and height must be even. |
| `I420` | image/video | Y, U, and V planes. Width and height must be even. |
| `H264` | encoded | H.264 access unit / NAL stream. |
| `ENCODED` | encoded | Generic encoded payload. The caps string identifies codecs without a dedicated format tag, including H.265. |
| `FP32` | tensor | Float32 tensor payload. |
| `INT8` | tensor | Signed INT8 tensor payload. |
| `UINT8` | tensor | Unsigned UINT8 tensor payload. |
| `BF16` | tensor | BF16 tensor payload. |
| `BBOX` | detection | Packed bounding-box payload. |
| `ByteStream` | tensor semantics | Opaque byte stream interpreted by downstream contract. |

## Payload families

`PayloadType` selects the broad family crossing a graph boundary.

| Payload family | Internal/media meaning | Common metadata |
| --- | --- | --- |
| `Image` | decoded pixels | pixel format, width, height, layout, image semantic metadata |
| `Tensor` | model or app tensor | dtype, shape, layout, tensor semantic metadata |
| `Encoded` | encoded media such as H.264, H.265, or JPEG | caps string, codec format, timestamps |
| `Auto` | infer when possible | use only when tensor/sample metadata is enough |

Text, audio, byte-stream, and opaque-byte payloads use tensor semantics or specialized specs. They are not separate `PayloadType` enum values in the public API reviewed for this release.

## Raw image mapping

| Format | PayloadType | Tensor layout / shape | Notes |
| --- | --- | --- | --- |
| `RGB` | `Image` | `HWC`, `[H, W, 3]` | Dense packed pixels. |
| `BGR` | `Image` | `HWC`, `[H, W, 3]` | Use for `cv2.imread` or OpenCV BGR frames. |
| `GRAY8` | `Image` | `HW`, `[H, W]` | Single-channel grayscale. |
| `NV12` | `Image` | `HW`, `[H, W]` plus plane metadata | Composite Y + UV planes. |
| `I420` | `Image` | `HW`, `[H, W]` plus plane metadata | Composite Y + U + V planes. |

For packed formats, depth is the channel count. For tensor payloads, depth is derived from the selected layout and shape.

## Read format, layout, and axis semantics together

Do not read one field in isolation:

| Field | What it tells you |
| --- | --- |
| `PixelFormat` / image format metadata | How to interpret pixel channels, such as RGB, BGR, GRAY8, NV12, or I420. |
| `TensorLayout` | How tensor dimensions are ordered, such as HWC, CHW, or HW. |
| `TensorAxisSemantic` | What an axis means when the tensor carries richer semantic metadata. |
| `TensorDType` | How each element is stored, such as UInt8, INT8, FP32, or BF16. |
| `ByteFormat` / byte-stream metadata | How opaque bytes should be interpreted by the next stage. |

Bytes are not meaning. Use the metadata fields together before you reinterpret a buffer.

## InputOptions format example

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
simaai::neat::InputOptions input;
input.payload_type = simaai::neat::PayloadType::Image;
input.format = simaai::neat::FormatTag::BGR;
input.width = 640;
input.height = 480;
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
input_options = pyneat.InputOptions()
input_options.payload_type = pyneat.PayloadType.Image
input_options.format = pyneat.Format.BGR
input_options.width = 640
input_options.height = 480
```

</CodeTab>
</CodeTabs>

Set only the fields the boundary needs. If the tensor or sample already carries enough metadata, avoid duplicate guesses.

H.264 has the dedicated `H264` tag. H.265 does not have a `H265` format tag;
use `ENCODED` and provide `video/x-h265` caps on the input boundary and encoded
sample.

## Advanced image/video output adapter

For normal model output, use `nodes.output(...)` and pull tensors with `pull_tensors(...)`. Use `OutputTensorOptions` only when image or video output must be converted, resized, or rate-adjusted into a CPU-friendly `UInt8` tensor before the app pulls it.

<CodeTabs>
<CodeTab label="C++" lang="cpp">

```cpp
simaai::neat::OutputTensorOptions output;
output.format = simaai::neat::FormatTag::BGR;
output.target_width = 640;
output.target_height = 480;

graph.add_output_tensor(output);
```

</CodeTab>
<CodeTab label="Python" lang="python">

```python
output = pyneat.OutputTensorOptions()
output.format = pyneat.Format.BGR
output.target_width = 640
output.target_height = 480

graph.add_output_tensor(output)
```

</CodeTab>
</CodeTabs>

`add_output_tensor(...)` accepts `TensorDType::UInt8`, which is the default output dtype. Keep the normal `nodes.output(...)` path for model tensors and for outputs where you want the full `Sample` envelope. Add explicit graph or app-side conversion when you need another dtype.

## Sample payload tags

`Sample::payload_tag` is the preferred label for downstream consumers. It supersedes the deprecated `Sample::format` field.

Use `payload_tag`, `payload_type`, `media_type`, and `caps_string` together when debugging encoded media or graph boundary negotiation.

## Preprocess metadata and ROI breadcrumbs

Detection decode, render, and ROI workflows need preprocessing metadata to map model-space coordinates back to source-frame coordinates.

That metadata can include:

- target width and height;
- scaled content width and height;
- resize or letterbox mode;
- padding value and geometry;
- input and output color formats;
- axis permutation;
- normalization, quantization, and tessellation flags;
- ROI windows, source image size, ROI batch size, and per-ROI affine transforms.

If boxes or masks land in the wrong place, check whether preprocessing metadata reached the decode or render stage before changing thresholds. For ROI-list preprocessing details, see [Preproc ROI Lists](/reference/preproc_roi).

## See also

- [Tensor and Sample](/develop-apps/development-workflow/core_types)
- [The dtype contract](/develop-apps/advanced-concepts/dtype_contract)
- [Node](/develop-apps/development-workflow/node)
