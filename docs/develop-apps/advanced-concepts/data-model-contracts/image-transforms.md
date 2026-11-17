# Resize, color conversion, and normalization

Use the typed image frontends when you need one preprocessing operation rather
than a model's fused `Preproc`. They execute on EV74; they do not insert a CPU
fallback, repack, or payload copy.

```cpp
#include <neat.h>

simaai::neat::ResizeOptions size;
size.width = 640;
size.height = 480;
size.max_input_width = 1920;
size.max_input_height = 1080;
auto resize = simaai::neat::nodes::Resize(size);

simaai::neat::ColorConvertOptions color;
color.output_format = simaai::neat::ImageFormat::NV12;
color.surface = simaai::neat::ImageSurfaceLayout::EncoderNv12;
auto convert = simaai::neat::nodes::ColorConvert(color);
```

Python exposes the same option types and `pyneat.nodes.resize`,
`pyneat.nodes.color_convert`, and `pyneat.nodes.normalization` factories.

## Input and output contracts

- **Resize** changes geometry and preserves the raw image format. Output dimensions
  are fixed. Input dimensions may vary within the prepared capacity; an omitted
  maximum uses the initial input dimension. Use `ResizeFit::Contain` for letterboxing
  or `Stretch` for independent axis scaling. The input format stays fixed.
- **ColorConvert** preserves dimensions and produces RGB, BGR, GRAY8, NV12, I420, or
  NV16. Input geometry and format are fixed when the node is first materialized.
  Construct a fresh node and graph to change them; rebuilding the same graph
  does not reset the node.
- **Normalization** accepts UInt8 RGB, BGR, or GRAY8 and produces a dense BF16
  tensor, not raw video. It preserves dimensions and channel order. As with
  ColorConvert, use a fresh node and graph to change materialized input geometry
  or format. Coefficients
  use semantic RGB order, including for BGR storage. The operation is
  `(x - float(mean * 255)) * float(1 / (stddev * 255))`, followed by the existing
  Preproc BF16 truncation. Standard deviations must be positive and finite.

GRAY8 output currently requires GRAY8 input; RGB/YUV-to-luma conversion is not
provided. Convert YUV to RGB or BGR before normalization. NV12 and I420 require even
width and height; NV16 requires even width and has full-height chroma. YUV
letterbox placement must meet the same chroma grid. Explicit source colorimetry
and chroma siting must agree with the authored policy. Interlaced and DV/alternate-line chroma are rejected. These
frontends process one frame per invocation, not ROI lists or frame batches.
Upstream scalar ROI results with complete affine provenance are not a new crop
request. A scalar ROI request without that provenance is rejected. Chaining
composes output-to-original coordinates.

EV74 bounds each source and output dimension to 4096 pixels. `BoxAverage` admits
at most eight touched component cells per axis, including chroma phase and
siting; a nonaligned eight-to-one reduction can therefore be rejected. Unsupported
geometry fails preparation rather than changing the interpolation or using A65.

## Frame rate

These operations do not change the frame rate. Prepared raw-image output caps
preserve a known upstream rate, including fractional rates. Author it on the
Input or an upstream caps boundary when a downstream consumer requires it;
configuring an encoder's rate does not establish the input rate. An unknown rate
remains absent rather than becoming an invented default.

Image publication uses a strict cold boundary projection: a node without an
`OutputSpecProvider` makes the inherited rate unknown, even if it happens to
pass frames through. An explicit downstream caps boundary can establish the
rate again. This is stricter than the general best-effort OutputSpec projection.
Changing the prepared input rate requires a new run; bounded dynamic Resize
input geometry does not authorize frame-rate changes.

## Color and storage

`ImageColorOptions` selects the BT.601 or BT.709 matrix family. Direct YUV-to-YUV
operations support full or limited range and centered or cosited chroma in each
axis; they preserve the coding range and matrix, rather than converting them.
RGB/GRAY-to-YUV supports only full range with centered chroma. Other profiles
are rejected during preparation. These options do not perform transfer-function
conversion. Packed RGB/GRAY outputs
retain legacy full-range, centered YUV interpretation; requesting other
range/siting settings fails instead of silently approximating them.

A multi-plane result is **one image with one storage owner**. Plane offsets and
positive pitches come from `GstVideoMeta`; consumers must not assume tightly
packed planes. Public NV16 samples carry Y and UV planes, with UV shape
`[height, width]` interleaved bytes.

`ImageSurfaceLayout::Native` uses tightly packed RGB/GRAY rows and aligns YUV
pitches to 16 bytes. It preserves YUV row padding and inter-plane holes.
`EncoderNv12` uses the encoder's layout authority: a
64-byte-aligned pitch and luma storage height rounded to eight rows. EV74 writes
all declared row tails and storage-only rows on every invocation. No per-frame
CPU zero-fill is inserted. This surface option is valid only for NV12; it does
not add NV16 support to a codec that rejects NV16.

For YUV letterboxing, `ResizeOptions::yuv_padding` is raw `[Y, U, V]`. If omitted,
the output range determines black: `[0,128,128]` for full range or `[16,128,128]`
for limited range. `pad_value` remains the packed RGB/GRAY fill value and must
remain zero for YUV output.

The existing fused `Preproc` API and its model-derived behavior remain unchanged.
The new factories retain the canonical `Preproc` node kind for graph tooling.
