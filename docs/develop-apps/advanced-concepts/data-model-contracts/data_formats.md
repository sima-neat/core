---
title: Data Formats
description: Format tokens, layout, and tensor semantics
sidebar_position: 1
slug: /develop-apps/advanced-concepts/data_formats
---

# Data formats & tensor semantics

This page summarizes the **format tokens** used by `InputOptions::format`
(and `OutputTensorOptions::format`) and how they map to `Tensor` layout,
shape, and plane semantics.

## Quick mapping table (raw video)

| Token | Media type | ImageSpec | Layout / shape | Planes |
| --- | --- | --- | --- | --- |
| `RGB` | `video/x-raw` | `RGB` | `HWC`, shape `[H,W,3]` | dense (no planes) |
| `BGR` | `video/x-raw` | `BGR` | `HWC`, shape `[H,W,3]` | dense (no planes) |
| `GRAY8` | `video/x-raw` | `GRAY8` | `HW`, shape `[H,W]` | dense (no planes) |
| `NV12` | `video/x-raw` | `NV12` | `HW`, shape `[H,W]` | composite planes: `Y` then `UV` |
| `I420` | `video/x-raw` | `I420` | `HW`, shape `[H,W]` | composite planes: `Y`, `U`, `V` |

Notes:
- `GRAY` is normalized to `GRAY8`.
- NV12/I420 require **even** width/height.
- For packed formats (RGB/BGR/GRAY8), **depth = channels** and is validated against
  shape when present.

## Tensor media type (`application/vnd.simaai.tensor`)

If `InputOptions::media_type` is set to `application/vnd.simaai.tensor`, the
`format` token is interpreted as a **dtype** (e.g., `FP32`, `BF16`, `INT8`) or a
known tessellated format (e.g., `MLA`). In this case:

- Layout must be explicit: `HWC`, `CHW`, or `HW` (no `Planar`).
- Shape rules:
  - `HWC` => `[H,W,C]`
  - `CHW` => `[C,H,W]`
  - `HW`  => `[H,W]` (depth inferred as 1)
- `format` is validated against the dtype in `Tensor::dtype`.

## OutputTensorOptions limitations (important)

`Graph::add_output_tensor()` currently:
- Supports only `UInt8` output tensors.
- Forces **SystemMemory** via a capsfilter.
- Does **not** transform layout (e.g., `layout=CHW` is metadata only).

If you need other dtypes or layout transforms, insert explicit nodes or do a
post‑processing step in your code.

## Mapping examples

RGB (dense):

```cpp
simaai::neat::Tensor t = /* RGB tensor */;
auto map = t.map_read();
const uint8_t* bytes = static_cast<const uint8_t*>(map.data);
```

NV12 (composite planes):

```cpp
simaai::neat::Tensor t = /* NV12 tensor */;
auto nv12 = t.map_nv12_read();
if (nv12) {
  const uint8_t* y  = nv12->view.y;
  const uint8_t* uv = nv12->view.uv;
}
```

I420 (composite planes):

```cpp
simaai::neat::Tensor t = /* I420 tensor */;
auto i420 = t.map_i420_read();
if (i420) {
  const uint8_t* y = i420->view.y;
  const uint8_t* u = i420->view.u;
  const uint8_t* v = i420->view.v;
}
```

## Depth vs channels

- For packed video: **depth == channels** (RGB/BGR = 3, GRAY8 = 1).
- For tensor media: depth is derived from the chosen layout and shape.

## Sample payload tags

`Sample::payload_tag` is the preferred label for downstream consumers. It
supersedes the deprecated `Sample::format` field.

## See also

- [Tensor and Sample](/develop-apps/development-workflow/core_types)
- [Tutorials](/tutorials)
- [Architecture](/develop-apps/contribute/architecture)
