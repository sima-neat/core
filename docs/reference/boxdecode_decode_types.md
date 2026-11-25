---
title: BoxDecode Decode Types
description: Choose the correct BoxDecodeType for object-detection postprocessing
sidebar_position: 6
---

# BoxDecode Decode Types

`nodes::SimaBoxDecode` converts raw detection-head tensors into detection results. It runs after model inference, applies the decode math for the selected model family and filters detections. NMS and output encoding depend on the selected family. Detection models can parse that payload as boxes; pose and segmentation models can also parse the keypoints or masks that follow the boxes.

For normal model-pack usage, prefer the `Model`-aware constructor. The model archive supplies the tensor order, layout, quantization, class count, resize metadata, and score-domain hints needed by the decoder. Your application usually only chooses the decode family and filtering thresholds.

## Runtime backend

BoxDecode and SuperPoint use `neatkernelrunner` with checked Kernels executables.
Existing detection and feature output formats are unchanged. YOLO26 oriented-box decoding uses a separate versioned format. For direct tensor
ingress, use shared-CMA tensors (`TensorMemory::EV74`), even when decode executes
on A65: this memory choice supplies DMA-BUF transport, not the execution target.

RCNN stage 1 uses the kernel-owned per-level top-K contract: global NMS,
then up to `top_k` survivors per model-area level. Its output is a proposal wire,
not the ordinary BBOX format: four groups, each containing an INT32 count and
`top_k` six-FP16 records (`score, 0, x1, y1, x2, y2`). Coordinate conversion
preserves fractional geometry until final serialization. Use it as stage-two
proposal input, not with `decode_bbox_tensor()`. No legacy plugin is loaded.

Padded, interleaved INT8, BF16 and FP32 SuperPoint heads use the native dense
reader with explicit pixel strides. Padding is excluded from detector reductions
and descriptor sampling without a Cast stage or materialization. INT8 ports
retain their own authored quantization. Native MLA-tiled heads retain their
declared physical layout; dense scalars are never relabeled as MLA byte planes.

## Quick start

```cpp
using namespace simaai::neat;

Model model("/path/to/yolov8_model.tar.gz");

auto boxdecode = nodes::SimaBoxDecode(
    model,
    BoxDecodeType::YoloV8,
    /* detection_threshold */ 0.25,
    /* nms_iou_threshold */ 0.45,
    /* top_k */ 100);
```

For standalone stage usage:

```cpp
simaai::neat::stages::BoxDecodeOptions opt(simaai::neat::BoxDecodeType::YoloV8);
opt.detection_threshold = 0.25;
opt.nms_iou_threshold = 0.45;
opt.top_k = 100;
```

## Arguments

| Argument | Meaning |
| --- | --- |
| `decode_type` | Model family/head format, such as `BoxDecodeType::YoloV8` or `BoxDecodeType::YoloX`. Required. |
| `detection_threshold` | Minimum score required to keep a detection. Use a model-appropriate value such as `0.25`. |
| `nms_iou_threshold` | IoU threshold used by non-maximum suppression. `0` keeps the model's packaged value; when none is packaged, YOLO26 skips NMS and other decoders suppress every overlapping box. Set a model-appropriate value such as `0.45`. |
| `top_k` | Maximum output count. Raw bounding-box decoding needs an explicit positive cap. On a model-bound node, `0` inherits the resolved Model cap. SuperPoint uses its profile default for `0`. |
| `original_width`, `original_height` | Source image size for coordinate mapping when using the raw-geometry constructor. |
| `model_width`, `model_height` | Model input size override. With the `Model` constructor this changes spatial decode knobs, not the packaged tensor contract. |
| `resize_mode_override` | Use only when no upstream `Preproc` stage writes resize metadata and you need to specify stretch/letterbox/crop behavior explicitly. |
| `decode_type_option` | Advanced sub-layout selector. Leave as `Auto` for model-pack usage unless you know the exported head layout. |

## Inputs and outputs

**Input:** raw detection tensors from the model. The expected tensor shapes depend on the model family. With an MPK/model archive, Neat reads those details from the packaged contract.

**Output:** one BoxDecode tensor containing decoded detections. Detection models use the standard `BBOX` payload. Pose and segmentation models keep the same leading boxes and append their task-specific payload:

| Model task | C++ helper | Python helper | Decoded tensors |
| --- | --- | --- | --- |
| Detection | `decode_bbox(...)` | `pyneat.decode_bbox(...)` | `[N, 6]` float32 boxes: `x1, y1, x2, y2, score, class_id` |
| Oriented detection | `decode_obb(...)` | `pyneat.decode_obb(...)` | Owned geometry `[N, 5]` float32 (`cx, cy, width, height, angle_radians`), scores `[N]` float32, class IDs `[N]` int32 |
| Pose | `decode_pose(...)` | `pyneat.decode_pose(...)` | boxes `[N, 6]` and keypoints `[N, 17, 3]` float32: `x, y, visibility` |
| Segmentation | `decode_segmentation(...)` | `pyneat.decode_segmentation(...)` | boxes `[N, 6]` float32 and masks `[N, 160, 160]` uint8 |
| Segmentation + pose | `decode_segmentation_pose(...)` | `pyneat.decode_segmentation_pose(...)` | boxes `[N, 6]` float32, masks `[N, 160, 160]` uint8, keypoints `[N, 17, 3]` float32 |
| Segment Anything | `decode_segmentation(...)` | `pyneat.decode_segmentation(...)` | boxes `[N, 6]` float32 and masks `[N, H, W]` uint8 at the original image size; N is 1 for a prompt decoder |
| SuperPoint | `decode_superpoint(...)` | `pyneat.decode_superpoint(...)` | keypoints `[N,2]`, scores `[N]`, descriptors `[N,D]` |

Axis-aligned detection-display graphs can feed the result to `SimaRender`. Use `decode_bbox(...)` only for the existing box-leading formats; neither it nor `SimaRender` accepts oriented boxes.

## Letterbox placement {#letterbox-placement}

A non-centered letterbox, such as `LetterboxPlacement::TopLeft` (see [Preproc resize](/reference/nodes/preproc#resize-and-aspect-ratio)), is supported by the detection profiles, which map boxes through each frame's model-to-original affine: the YOLO detection families, decoded YOLO, EfficientDet, DETR, DetrFocal, CenterNet and SSD. Pose, segmentation and 91-class multiclass SSD do the same when the installed Kernels release accepts the transform for them (see [Crop coordinate mapping](#crop-coordinate-mapping)); otherwise, and always for RCNN proposals, the profile rejects it instead of returning shifted results. Centered letterboxes keep their mapping. The Segment Anything types have their own requirements; see [Segment Anything rules](#segment-anything-rules).

## Classic DETR {#classic-detr}

Select `Detr` for one four-channel box-logit head and one class-logit head with a final background channel on the same query grid. Leave `decode_type_option` as `Auto`: the recipe selects Softmax without tensor-name hints. The background participates in the denominator but is excluded from the foreground class count and output selection. Box coordinates use sigmoid; this is not a decoder for already-normalized focal-loss boxes.

Use Stretch and a positive `top_k`. The decoder applies no NMS and maps results to the original frame using preprocessing metadata. Explicit probability-valued layouts or conflicting score activations are rejected.

```python
options = pyneat.ModelOptions()
options.decode_type = pyneat.BoxDecodeType.Detr
options.preprocess.resize.mode = pyneat.ResizeMode.Stretch
options.top_k = 100
options.score_threshold = 0.7
```

## Focal-loss query detectors

Use `BoxDecodeType::DetrFocal` (Python: `pyneat.BoxDecodeType.DetrFocal`, token
`detr-focal`) for two ordered tensors: normalized `cx,cy,width,height` coordinates
with depth 4, then independent class logits with depth N. Their query axes and dtypes must
match; N is 1–1024. FP32, BF16 and authored INT8 storage are supported.

The profile applies sigmoid only to class logits and selects the global top-K
query/class pairs. One query can produce several classes. There is no background
class, box sigmoid, per-query argmax, NMS or small-area filter. Set `top_k` to a
positive cap, `detection_threshold=0` for unfiltered top-K, and
`nms_iou_threshold=0`. Use Stretch and the actual image/model dimensions.

The existing `BBOX` output clips boxes to the image and rounds XYWH coordinates
to integers; `decode_bbox` exposes these coordinates as floats, without restoring
subpixel precision. Use `decode_bbox(..., strict=True)` in Python to preserve
zero-area rows created by rounding instead of the default degenerate-box filter.
Measure this representation's accuracy separately from an
unclipped floating-point reference. Classic `BoxDecodeType::Detr` is unchanged:
it expects box logits and background-aware softmax scores, not these tensors.

The Python raw-geometry overload mirrors the C++ factory when you supply the
head tensors yourself. For dense FP32 heads, retain the model's complete output
adapters and declare their actual storage explicitly. Do not truncate to raw
MLA output or relabel padded bytes as dense tensors.

```python
node = pyneat.nodes.sima_box_decode(
    pyneat.BoxDecodeType.DetrFocal,
    original_width=1280, original_height=720,
    model_width=640, model_height=640,
    detection_threshold=0.0, nms_iou_threshold=0.0, top_k=300,
    decode_type_option=pyneat.BoxDecodeTypeOption.InterleavedByHeadLogit,
    source_storage=pyneat.BoxDecodeSourceStorage.DenseHwc,
    detess=False, dequant=False,
    resize_mode=pyneat.ResizeMode.Stretch,
)
```

Finite reversed extents use the same endpoint clipping and become empty BBOX
records, without replacing the selected pair or changing its score. Nonfinite
coordinates or logits still fail before publication.

## Segment Anything masks {#segment-anything-masks}

Each Segment Anything model has its own decode type. On a prompt decoder, BoxDecode turns one
decoder run into one detection: the selected mask at the original image size, its tight box and
its predicted IoU. On an EfficientSAM3 concept image model, it turns one run into up to `top_k`
detections, each with its own mask.

| Model | `decode_type` (token) | Preprocess of the Model that sees the image | Decoder options |
| --- | --- | --- | --- |
| MobileSAM | `MobileSam` (`mobilesam`) | Encoder: `resize.mode=Letterbox`, `resize.placement=TopLeft`, 1024×1024, `pad_value=116`; `preset=ImageNet` | `multimask_output` |
| EdgeTAM | `EdgeTamSeg` (`edgetam-seg`) for the mask; `EdgeTamTrack` (`edgetam-track`) for tracking | Image encoder: `resize.mode=Stretch`, 1024×1024, no padding; `preset=ImageNet` | `multimask_output` |
| EfficientSAM-T / -S | `EfficientSam` (`efficientsam`) | Encoder: `resize.mode=Stretch`, 1024×1024, no padding; no preset, `normalize.enable=On` (values 0–1; the encoder normalizes itself) | Leave `multimask_output` true |
| EfficientSAM3 interactive | `EfficientSam3` (`efficientsam3`) | Image encoder: `resize.mode=Stretch`, 1008×1008, no padding; `normalize.enable=On` with `normalize.mean` and `normalize.stddev` (0.5, 0.5, 0.5) | `multimask_output` |
| EfficientSAM3 concept | `EfficientSam3` (`efficientsam3`) | The concept image model itself, on the tensor route: the application stretches the image to 1008×1008, scales it to [−1, 1] and attaches `semantic.preprocess` with `resize_mode="stretch"` | `top_k`, `score_threshold` |

Python uses the same names: `pyneat.BoxDecodeType.MobileSam`, `EdgeTamSeg`, `EdgeTamTrack`,
`EfficientSam` and `EfficientSam3`. MobileSAM pads with zeros after normalizing, which is the ImageNet mean pixel
(124, 116, 104); `pad_value=116` approximates it. Masks never cover the padding. An
EfficientSAM3 Model is a concept image model when its outputs are detections `[Q, 6]` and mask
logits with Q channels, and a prompt decoder otherwise. Each model's README in
[sima-neat/models](https://github.com/sima-neat/models) has its full workflow, such as the
[EdgeTAM README](https://github.com/sima-neat/models/tree/develop/models/edgetam).

`Model::Options::multimask_output` (Python: `ModelOptions.multimask_output`) mirrors upstream
`predict(multimask_output=...)`. Leave it `true` for one click, which is ambiguous; set it to
`false` for a box or several points.

| `decode_type` | `multimask_output=true` (default) | `multimask_output=false` |
| --- | --- | --- |
| `MobileSam` | Best predicted IoU of masks 1–3 | Mask 0 |
| `EdgeTamSeg`, `EdgeTamTrack` | Best predicted IoU of masks 1–3 | Mask 0 when it is stable, otherwise the best of masks 1–3 |
| `EfficientSam` | Best predicted IoU of masks 0–2 | Rejected: EfficientSAM exports no single-mask token |
| `EfficientSam3` prompt decoder | Best predicted IoU of masks 1–3 of four, or 0–2 of three | Four masks: as EdgeTAM. Three masks: rejected |
| `EfficientSam3` concept | Up to `top_k` queries above `score_threshold` | Rejected: concept detection has no multimask choice |

- Mask 0 is stable, as in SAM2, when at least 98% of its logits above −0.05 are also above 0.05.
- A decoder exported with one mask channel uses that mask with either value.
- Other decode types reject `multimask_output=false`. The choice belongs to the `Model`: two
  decoder Models can use the same MPK.

### What each output means {#segment-anything-outputs}

The BoxDecode tensor uses the [versioned masks payload](#versioned-masks-payload).
`decode_segmentation` returns boxes `[N, 6]` and masks `[N, H, W]`:

- **Box.** `x1, y1, x2, y2` in original-image pixels. A prompt decoder's box is the tight box of
  the mask pixels: `x2` and `y2` are one past the last mask column and row, so `x2 - x1` is the
  mask's width. A concept detection's box comes from its detection head, scaled to the original
  image and not clipped.
- **Score.** The chosen mask's predicted IoU, or the concept query's score.
- **Class.** The chosen mask channel (the index in the table above), or 0 for a concept query.
- **Mask.** `uint8`, 255 where the logit resampled to the original image size is above 0,
  otherwise 0.
- **Count.** Always 1 for a prompt decoder: an empty mask has an all-zero box and keeps its score
  and class. EdgeTAM's object score of 0 or below empties the mask, as upstream. A concept model
  returns the kept queries in descending score order (ties by query index).

`EdgeTamSeg` and every other Segment Anything type return the BoxDecode tensor alone. With
`EdgeTamTrack`, the decoder's own outputs follow it; see [EdgeTAM tracking](#edgetam-tracking).

`SimaRender` does not draw these masks: it draws only the headerless segmentation payload, whose
masks are smaller than the frame. Draw Segment Anything masks from the `decode_segmentation`
result.

### Rules {#segment-anything-rules}

- **Decoder outputs.** A prompt decoder exports mask logits with the mask count of the named model
  (1, 3 or 4), one predicted IoU per mask and, for EdgeTAM, an object score; INT8, BF16 and FP32
  outputs are accepted. A concept image model exports detections `[Q, 6]` and per-query mask
  logits.
- **Options.** Prompt decoders: leave `top_k` at 0 or 1 and `nms_iou_threshold` at 0;
  `score_threshold` must be below 1 and is otherwise ignored; `num_classes` is the mask count or 0.
  Concept image models: set `top_k` to 1..Q.
- **Image geometry.** The canvas, the original image size and the resize come from the decoder
  inputs' `tensor.semantic.preprocess` and must match the table; otherwise building fails and names
  the setting to change. Pass the encoder's outputs to the decoder with `.cvu()`, and build a prompt
  decoder from its inputs with `build(inputs)` or `execution_plan(inputs)`. On the concept image
  model's tensor route, attach the facts to its pixel tensor.
- **Whole image, resized once.** A crop or ROI before the resize is rejected, and so is a scaled
  size the decoder does not reproduce. For MobileSAM, letterbox with `resize.placement=TopLeft` on
  the Model that preprocesses the image, which rounds as Segment Anything does, not with a
  standalone `Preproc` node. Facts you attach yourself must carry the affine: `affine_m00` =
  original width / scaled width and `affine_m11` = original height / scaled height.
- **One image size per runner.** The masks have the original image size, so a built runner serves
  one size. `Model::run` builds another runner for a new size; a streaming graph rejects a frame of
  another size.

### Encoder and decoder {#segment-anything-example}

MobileSAM; for another model, use its encoder settings and decode type from the table.

```python
encoder_options = pyneat.ModelOptions()
encoder_options.preprocess.resize.mode = pyneat.ResizeMode.Letterbox
encoder_options.preprocess.resize.placement = pyneat.LetterboxPlacement.TopLeft
encoder_options.preprocess.resize.pad_value = 116
encoder_options.preprocess.preset = pyneat.NormalizePreset.ImageNet
encoder = pyneat.Model("/path/to/mobile_sam_image_encoder_mpk.tar.gz", encoder_options)

options = pyneat.ModelOptions()
options.decode_type = pyneat.BoxDecodeType.MobileSam
options.multimask_output = False  # a box prompt; keep True for one click
decoder = pyneat.Model("/path/to/mobile_sam_prompt_decoder_mpk.tar.gz", options)

embeddings = encoder.run([image])[0]
# .cvu() keeps the encoder's preprocess metadata on the embeddings.
outputs = decoder.run([embeddings.cvu(), point_encoding, point_types])
result = pyneat.decode_segmentation(outputs)[0]
box = result.boxes.to_numpy()[0]  # x1, y1, x2, y2, IoU, mask index
mask = result.masks.to_numpy()[0]  # uint8 [H, W], 0 or 255
```

### EdgeTAM tracking {#edgetam-tracking}

EdgeTAM's memory encoder and memory bank need the chosen mask's low-resolution logits and its
object pointer, which BoxDecode does not decode. `EdgeTamTrack` returns the BoxDecode tensor, then
the decoder's outputs from the same run, exactly as the same Model returns them without BoxDecode:
`mask_logits` `[1, 256, 256, 4]`, `iou_scores` `[1, 4]`, `object_score_logits` `[1, 1]` and
`object_pointers` `[1, 4, 256]`, all float32. Each costs one conversion per frame. Decode only the
first tensor; the raw tensors carry no detection tag. `EdgeTamTrack` is rejected when the model's
route without BoxDecode cannot return the tensors.

- The detection's class is the chosen plane: take `mask_logits[0, :, :, class]` for the memory
  encoder, upsampled to 1024×1024 as the model's README describes.
- Set `multimask_output=False` for a prompt of two or more points; leave it `True` for one click
  and for tracked frames. The class reports the plane either rule chose.
- For the memory bank, take `object_pointers[0, class]` when `multimask_output=True`. When it is
  `False`, take `object_pointers[0, 0]`: SAM 2 keeps the single-mask pointer even when its
  stability fallback chooses a mask from planes 1–3.
- An object score of 0 or below empties the mask and its box, but the class still names the
  chosen plane. Apply the README's object-score rule to the logits you pass to the memory encoder.

```python
options = pyneat.ModelOptions()
options.decode_type = pyneat.BoxDecodeType.EdgeTamTrack
decoder = pyneat.Model("/path/to/edgetam_mask_decoder_mpk.tar.gz", options)

outputs = decoder.run(decoder_inputs)
result = pyneat.decode_segmentation(outputs[:1])[0]
plane = int(result.boxes.to_numpy()[0, 5])
logits = outputs[1].to_numpy()[0, :, :, plane]  # [256, 256] low-resolution logits
pointer = outputs[4].to_numpy()[0, plane if options.multimask_output else 0]  # [256]
```

### EfficientSAM3 concept {#efficientsam3-concept}

```python
options = pyneat.ModelOptions()
options.decode_type = pyneat.BoxDecodeType.EfficientSam3
options.top_k = 20
options.score_threshold = 0.5
options.preprocess.kind = pyneat.InputKind.Tensor  # two inputs: pixels and packed text
model = pyneat.Model("/path/to/efficientsam3_image_model_mpk.tar.gz", options)

# Stretch to 1008x1008 and scale to [-1, 1], as SAM 3 does, and record that geometry.
stretched = cv2.resize(rgb, (1008, 1008), interpolation=cv2.INTER_LINEAR)
pixels = pyneat.Tensor.from_numpy(stretched.astype(np.float32) / 127.5 - 1, copy=True,
                                  layout=pyneat.TensorLayout.HWC)
facts = pyneat.PreprocessRuntimeMeta()
facts.resize_mode = "stretch"
facts.original_width, facts.original_height = rgb.shape[1], rgb.shape[0]
facts.resized_width = facts.resized_height = 1008
facts.affine_m00 = facts.original_width / 1008  # model-to-original scale
facts.affine_m11 = facts.original_height / 1008
semantic = pixels.semantic
semantic.preprocess = facts
pixels.semantic = semantic

outputs = model.run([pixels, packed_text])
result = pyneat.decode_segmentation(outputs)[0]
boxes = result.boxes.to_numpy()  # [N, 6]: x1, y1, x2, y2, score, 0
masks = result.masks.to_numpy()  # uint8 [N, H, W]
```

## YOLOv8 and YOLO11 oriented boxes {#yolov8-oriented-boxes}

Select `BoxDecodeType::YoloV8Obb` (Python: `pyneat.BoxDecodeType.YoloV8Obb`; tokens
`yolov8-obb` or `yolo11-obb`). Supply nine grouped INT8, BF16 or FP32 heads: three
64-channel DFL box heads, three class-logit heads, and three angle-logit heads, in
stride-8/16/32 order. The complete class count is inferred from the class heads.

BoxDecode applies DFL, class sigmoid, `(sigmoid(angle)-0.25)*pi`, and class-aware
rotated Fast-NMS using `nms_iou_threshold`. Inputs must be raw logits. Geometry and
`BBOX_OBB_V1` output follow the YOLO26 contract below; use the same `decode_obb` helper.

## YOLO26 oriented boxes {#yolo26-oriented-boxes}

Select `BoxDecodeType::YoloV26Obb` (Python: `pyneat.BoxDecodeType.YoloV26Obb`,
token `yolo26-obb`) explicitly. This profile consumes nine heads in FP32, BF16, or INT8, grouped as
three four-channel distance heads, three class-logit heads, and three one-channel
raw-angle heads, at strides 8, 16, and 32. All heads must agree on the complete
class count (1–256). `Auto`, `GroupedByRole`, and `GroupedByRoleLogit` select this
grouped layout; probability-valued class heads and interleaved layouts are rejected.

INT8 heads require positive finite per-tensor quantization metadata and an INT8 zero point. BF16 and FP32 heads do not use integer quantization metadata.

Use a positive `top_k` and `nms_iou_threshold = 0`. The profile applies sigmoid to
class logits, selects anchors by their highest activated score, then selects
anchor/class pairs. Equal scores preserve stage-one rank and class order. The
score cutoff is strict (`score > detection_threshold`). The raw angle is already
in radians; the decoder does not apply sigmoid or normalize its range.

Preprocess metadata must supply an explicit, finite model-to-original transform
with positive equal X/Y scales and no shear. An explicit identity transform is
required even for model-sized tiles. The profile rejects missing or anisotropic
transforms rather than approximating a rotated rectangle. It preserves signed
finite widths/heights and does not clip to image bounds. For tiled DOTA evaluation,
use an identity transform for each tile, then perform tile-offset recovery and
cross-tile polygon suppression in the application.

In source-driven graphs, Core derives cold admission geometry from a checked upstream Preproc contract. This is only a preparation seed: every frame must still carry a finite, positive uniform affine and a matching model canvas. A missing frame transform is an error, not an identity fallback.

The output is tagged `BBOX_OBB_V1`: a 32-byte versioned header followed by `top_k`
32-byte records. Unused records are zero. `decode_obb_tensor(tensor)` validates a
single carrier; `decode_obb(tensors)` returns one typed result for each carrier.
Both copy into owned CPU tensors; class IDs remain INT32. For example:

```python
result = pyneat.decode_obb(outputs)[0]
geometry = result.geometry.to_numpy()  # cx, cy, width, height, angle_radians
scores = result.scores.to_numpy()
class_ids = result.class_ids.to_numpy()
```

This format is intentionally not a leading-BBOX extension. Use the typed helpers,
not `decode_bbox`, segmentation/pose parsers, or the axis-aligned renderer.

## SuperPoint

SuperPoint remains part of the BoxDecode product surface, but emits feature points rather than
pretending that they are boxes. The minimal A65-default configuration is:

```cpp
BoxDecodeOptions options{BoxDecodeType::SuperPoint};
options.superpoint.descriptor_output_dtype = TensorDType::Float32;

auto decoder = nodes::SimaBoxDecode(model, options);
```

Python uses the same defaults:

```python
options = pyneat.BoxDecodeOptions(pyneat.BoxDecodeType.SuperPoint)
options.superpoint.descriptor_output_dtype = pyneat.TensorDType.Float32

decoder = pyneat.nodes.sima_box_decode(model, options=options)
```

`A65V1` is the default profile. Select another profile explicitly when the
model requires different numerical behavior; Neat does not infer behavior from
tensor shapes or values:

| Profile | When to select it | Production status |
|---|---|---|
| `LightGlueV1` | LightGlue-compatible detector, NMS, coordinate, and descriptor behavior | Supported |
| `MagicLeapDemoV1` | The pinned Magic Leap demo behavior | Supported |
| `A65V1` | Compatibility with the former A65 SuperPoint decoder | Supported; default |
| `PaperBicubicV1` | Reserved numeric ID for a future fully specified bicubic policy | Rejected until production-defined |

Numerical behavior and output encoding are independent. For example, select A65 numerical
behavior with the default V1 output:

```cpp
BoxDecodeOptions options{BoxDecodeType::SuperPoint};
options.superpoint.profile = SuperPointProfile::A65V1;
options.superpoint.output_format = SuperPointOutputFormat::FeaturePointsV1;
```

The legacy byte layout is opt-in and has additional constraints:

```cpp
options.superpoint.profile = SuperPointProfile::A65V1;
options.superpoint.output_format = SuperPointOutputFormat::LegacyA65InterleavedV0;
options.superpoint.descriptor_output_dtype = TensorDType::Int8;
```

`SuperPointProfile::Auto` first uses authoritative MPK `superpoint.profile` metadata. If neither
the API, `Model::Options.superpoint.profile`, nor MPK supplies a profile, it resolves to `A65V1`.
Neat never guesses a profile from tensor shapes, values, filenames, or downstream nodes.

When their public sentinel values are left unchanged, `detection_threshold=0.0`, `top_k=0`,
`nms_radius=-1`, and `border_margin=-1` resolve from the selected profile. `A65V1` resolves to a
threshold of `0.1`, Top-K `600`, NMS radius `4`, and border margin `0`. LightGlueV1 and
MagicLeapDemoV1 use thresholds `0.0005` and `0.015`, respectively; both use Top-K `600`, NMS
radius `4`, and border margin `4`.

`nms_iou_threshold` does not apply to SuperPoint; use the pixel-radius
`superpoint.nms_radius`. The default output is the versioned `FEATURE_POINTS_V1` structure-of-arrays
payload. `LegacyA65InterleavedV0` is an explicit migration format and requires 256-dimensional INT8
descriptors. Use `decode_superpoint` rather than `decode_bbox` or `BoxDecodeResults`.

Versioned MPK `superpoint` schema v1 records are fail-closed. They must name the profile, distinct
detector and descriptor tensor IDs, a `sha256:` fingerprint with 64 hexadecimal digits, and the
supported input representations `raw-logits-65` and `coarse-pre-l2`. Schema 0 remains accepted only
as a migration/manual record; omitted schema-0 representation fields are canonicalized to those two
raw-input representations and recorded as defaults in diagnostics. Unknown schema versions or
representation tokens fail contract compilation.
If an API profile override conflicts with a fingerprint stamped for a different MPK profile,
re-stamp the MPK for the selected profile; Neat does not discard or reinterpret that provenance.

## BBOX wire payload

Detection decode emits one tensor tagged `BBOX` per input frame. That tensor is
a rank-1 `UInt8` byte buffer:

| Field | Value |
| --- | --- |
| `semantic.detection.format` | `"BBOX"` |
| `dtype` | `UInt8` |
| `shape` | `[N_bytes]`, where `N_bytes` is the packed buffer capacity from the model archive |

The tensor shape is a byte count, not a detection count. The payload uses
little-endian layout:

```text
offset  size  content
------  ----  -------
  0      4    uint32  N = valid detections in this frame
  4     24    RawBox[0]
 28     24    RawBox[1]
  .      .      ...
  .      .    RawBox[N-1]
                   trailing bytes are padding and must be ignored
```

Each `RawBox` record is 24 bytes:

| Offset | Size | Type | Field | Meaning |
| --- | --- | --- | --- | --- |
| 0 | 4 | `int32` | `x` | Top-left x in source pixels. |
| 4 | 4 | `int32` | `y` | Top-left y in source pixels. |
| 8 | 4 | `int32` | `w` | Width in source pixels. |
| 12 | 4 | `int32` | `h` | Height in source pixels. |
| 16 | 4 | `float32` | `score` | Post-NMS confidence in `[0.0, 1.0]`. |
| 20 | 4 | `int32` | `class_id` | Model-defined class id. |

The matching Python `struct` format for one record is `"<iiiifi"`.

Coordinates are in original-image pixels when upstream preprocessing metadata is
present. For a frame preprocessed from an ROI, the original image is the full source
frame, not the ROI (see [Crop coordinate mapping](#crop-coordinate-mapping)). They
are not normalized to `[0, 1]` and are not expressed in the model's internal
letterboxed input space.

### Combined segmentation + pose payload

`yolox-seg-pose` emits a single buffer carrying three regions. Every region is
strided by the same slot count, `top_k`:

| Region | Offset | Stride | Contents |
| --- | --- | --- | --- |
| header | `0` | 4 | `int32` detection count |
| boxes | `4` | 24 | `BoundingBoxOut` records, as above |
| masks | `4 + 24*top_k` | `mask_w * mask_h` | `uint8`, one plane per slot |
| poses | `4 + (24 + mask_w*mask_h)*top_k` | 204 | 17 x `{uint32 x, uint32 y, float32 visibility}` |

Use `decode_segmentation_pose(...)` to get boxes, masks and keypoints. Row `i` in each tensor describes the same detection. Use `decode_bbox(...)` or `BoxDecodeResults(...)` if you only need boxes.

The combined decoder also accepts the legacy `BBOX` tag. Simultaneous mask-and-pose
rendering is not supported. Do not pass the combined buffer to `poseyolo`: that
renderer expects keypoints directly after the boxes, whereas the combined payload
stores masks there.

There are 17 keypoint slots per detection. The backend zeros unused slots and keypoints for classes excluded by `pose_classes`. The helper copies those values unchanged.

Core derives `num_classes` from the class-head depth minus one, because channel 0 holds objectness. For example, 30 channels means 29 classes. An explicit `num_classes` must match.

### Keypoint classes

Set the class IDs that have keypoints:

```python
options = pyneat.BoxDecodeOptions(pyneat.BoxDecodeType.YoloXSegPose)
options.yolox_seg_pose.pose_classes = [0, 5]
```

`ModelOptions.yolox_seg_pose` accepts the same setting.

- Classes outside the list get zero keypoint coordinates and visibility.
- An empty `BoxDecodeOptions` list inherits the model's settings. With no list configured, every class gets keypoints.
- IDs must be unique and in `[0, num_classes)`. This option is only supported for `YoloXSegPose`.

### Versioned masks payload {#versioned-masks-payload}

Masks that are not 160×160, such as Segment Anything output, use a versioned header instead of
the leading count. Its layout is
[`SegmentationMasksHeaderV1`](/reference/cppapi/structs/simaai-neat-segmentationmasksheaderv1)
in `pipeline/DetectionTypes.h`: the header, then `RawBox` records, then one `uint8` mask per
record, 0 or 255. Read it with `decode_segmentation`, which returns the boxes and masks and rejects
a malformed header or extent. The tensor keeps the `BBOX` tag, and `decode_segmentation` and
`decode_bbox` recognize the payload by its magic, so existing segmentation payloads are unchanged.
`SimaRender` does not accept it.

## When `model.run` returns raw heads

Some model routes return raw feature-map heads from `model.run(...)` instead of
a decoded `BBOX` tensor. That is not a failed run. It means the model executed,
but the route did not include BoxDecode at the point where you read output.

Use this rule:

- `detections=...` or a `BBOX` tensor: parse the packed BBOX payload or use the
  decode helpers.
- `raw_output_heads=...`: add a BoxDecode stage, inspect the model route, or
  consume the raw tensors with model-specific postprocessing.

Do not parse raw heads as boxes. The raw tensor layout depends on the exported
model family and model archive contract.

## Override contract

The MPK supplies tensor contracts and decode-family facts. Set filtering controls
in `ModelOptions`. For a `SimaBoxDecode(Model, ...)` node, positive filtering
arguments override the resolved Model controls; zero arguments inherit them.
`ModelOptions.score_threshold = 0.0` is a real zero score gate, not a request to
recover a packaged threshold.

| Runtime argument | Value passed | Behavior |
| --- | --- | --- |
| `decode_type` | empty / `Unspecified` | Preserve model archive or route-planner inference where supported. |
| `decode_type` | concrete type | Override the decode family for this run. |
| `original_width` / `original_height` | `0` | Preserve packaged geometry or upstream preprocess metadata. |
| `original_width` / `original_height` | positive integer | Override source dimensions for coordinate mapping. |
| Model-bound node `detection_threshold` | `0.0` | Inherit the resolved Model score gate. |
| Model-bound node `detection_threshold` | `> 0.0` | Override the score gate. |
| Model-bound node `nms_iou_threshold` | `0.0` | Inherit the resolved Model NMS IoU. |
| Model-bound node `nms_iou_threshold` | `> 0.0` | Override NMS IoU. |
| Model-bound node `top_k` | `0` | Inherit the resolved Model output cap. |
| Model-bound node `top_k` | `> 0` | Override the maximum kept detections. |

Bounding-box decoding requires a positive resolved `top_k`; zero does not request
unbounded output storage. For raw heads without a bound `Model`, set `top_k` before
building the graph. A model-bound node can inherit a positive Model cap by leaving
its own `top_k` at zero. These controls do not change the separate result-reader
helpers: their `top_k=0` means no additional cap on an already bounded output.
A zero score gate permits zero-valued scores, but does
not bypass geometry validation, background exclusion, NMS or the output cap.
At NMS IoU zero, YOLO26 skips NMS. Other NMS-enabled YOLO and SSD profiles still
apply their quantized-IoU suppression. Positive YOLO26 thresholds enable NMS.

`num_classes` compares the value configured by the caller with the class count
derived from the MPK tensor contract:

| Model family | Configured `num_classes` | MPK-derived `num_classes` | Behavior |
| --- | --- | --- | --- |
| Any supported model | `0` | positive, inferable value | Use the MPK-derived class count. |
| Any supported model | positive integer | same value | Use the configured class count. |
| Model with an ambiguous class split | positive integer | unavailable | Use the configured class count. This is required when a split single-class head cannot be inferred reliably. |
| YOLOv5 or YOLO26 | positive integer | different value | Fail before pipeline construction and report both values. These raw-head layouts derive their class count from tensor depth. |
| SSD or another pre-YOLO26 non-pose YOLO family | positive integer | different value | Apply the existing family-specific explicit-override behavior. Pose decoders and SuperPoint retain their family-specific rules. |

`detection_threshold` is the name used by the BoxDecode node/stage
constructors. `ModelOptions.score_threshold` is the model-route option that
feeds the same control.

## Decode type mapping

| API enum | Backend token | Typical model family |
| --- | --- | --- |
| `BoxDecodeType::Yolo` | `yolo` | Decoded `cx,cy,w,h` in model pixels |
| `BoxDecodeType::YoloV5` | `yolov5` | YOLOv5 detection |
| `BoxDecodeType::YoloV5Seg` | `yolov5-seg` | YOLOv5 segmentation |
| `BoxDecodeType::YoloV7` | `yolov7` | YOLOv7 detection |
| `BoxDecodeType::YoloV7Seg` | `yolov7-seg` | YOLOv7 segmentation |
| `BoxDecodeType::YoloV8` | `yolov8` | YOLOv8 detection |
| `BoxDecodeType::YoloV8Seg` | `yolov8-seg` | YOLOv8 segmentation |
| `BoxDecodeType::YoloV8Pose` | `yolov8-pose` | YOLOv8 pose |
| `BoxDecodeType::YoloV9` | `yolov9` | YOLOv9 detection |
| `BoxDecodeType::YoloV9Seg` | `yolov9-seg` | YOLOv9 segmentation |
| `BoxDecodeType::YoloV10` | `yolov10` | YOLOv10 detection |
| `BoxDecodeType::YoloV10Seg` | `yolov10-seg` | YOLOv10 segmentation |
| `BoxDecodeType::YoloV26` | `yolo26` | YOLO26 detection |
| `BoxDecodeType::YoloV26Pose` | `yolo26-pose` | YOLO26 pose |
| `BoxDecodeType::YoloV26Seg` | `yolo26-seg` | YOLO26 segmentation |
| `BoxDecodeType::YoloV26Obb` | `yolo26-obb` | YOLO26 oriented detection (`BBOX_OBB_V1`) |
| `BoxDecodeType::YoloV6` | `yolov6` | YOLOv6 detection |
| `BoxDecodeType::YoloX` | `yolox` | YOLOX detection |
| `BoxDecodeType::YoloXSegPose` | `yolox-seg-pose` | YOLOX packed export carrying box, mask and keypoint heads together |
| `BoxDecodeType::Ssd` | `ssd` | Exact prepared SSD300, SSD-Mobile-300, SSD-Mobile-320, or SSDlite-Mobile-320 contract, selected from ordered head geometry |
| `BoxDecodeType::SuperPoint` | `superpoint` | SuperPoint detector and descriptor postprocessing |
| `BoxDecodeType::Detr` | `detr` | DETR-style transformer detection |
| `BoxDecodeType::MobileSam` | `mobilesam` | MobileSAM prompt decoder (top-left letterbox) |
| `BoxDecodeType::EdgeTamSeg` | `edgetam-seg` | EdgeTAM mask decoder, with its object score (stretch) |
| `BoxDecodeType::EdgeTamTrack` | `edgetam-track` | `EdgeTamSeg`, then the decoder's raw outputs for tracking |
| `BoxDecodeType::EfficientSam` | `efficientsam` | EfficientSAM prompt decoder, three multimask tokens (stretch) |
| `BoxDecodeType::EfficientSam3` | `efficientsam3` | EfficientSAM3 prompt decoder, or concept image model with one mask per kept query (stretch) |
| `BoxDecodeType::EffDet` | `effdet` | EfficientDet detection |
| `BoxDecodeType::RcnnStage1` | `rcnn-stage1` | Five ordered RPN heads; global NMS then top-K per area level |
| `BoxDecodeType::Centernet` | `centernet` | CenterNet detection |

`BoxDecodeType::Unspecified` is an unset sentinel and fails before runtime. SSD recipe identity is
an internal Core contract (`ssd300-v1`, `ssd-mobile-300-v1`, `ssd-mobile-320-v1`, or
`ssdlite-mobile-320-v1`), not another public decode type or
backend token. Core resolves it before lowering, while the installed object decoder continues to
receive its supported `ssd` family token and selects the corresponding fixed implementation from
the already validated head geometry.

## Choosing the right type

- Choose by the exported head contract: `YoloV6` expects raw l/t/r/b distances and class logits. For YOLOv6 packs with in-graph box decoding, use `Yolo`; for grouped `[bbox_0..2, class_prob_0..2]` outputs, set `decode_type_option=GroupedByRoleProbability` and `num_classes` to the trained class count.
- If boxes are shifted or scaled incorrectly, check the image resize policy. Use `resize_mode_override` only when your graph does not have an upstream `Preproc` stage writing resize metadata.
- If you are authoring a custom model pack, ensure the archive describes the detection heads accurately: tensor order, logical shape, physical storage, dtype/quantization, score domain, class count, and any sliced outputs. Application code should not need to compensate for these details.

## Shape and layout guidance

Different detection models expose different head layouts. Some use one tensor per feature-map level; others split boxes, objectness, classes, keypoints, or masks into separate tensors. Some model outputs are dense HWC tensors; others are packed or sliced by the compiler/runtime.

For model-pack flows this is handled by the packaged contract. For manually wired tensors, the key rule is: match the exported head format exactly. Do not choose a decode type based only on rank or channel count.

Advanced tensor-contract rules:

- YOLO-family decode types other than `YoloV5` detection (`Yolo`, `YoloV7`,
  `YoloV8`, `YoloV9`, `YoloV10`, and segmentation/pose variants) expect either
  decoupled heads or packed heads that match the model family.
- Packed YOLO heads must keep class count and head depth consistent across
  feature levels.
- `YoloV5` detection accepts exactly three undecoded packed heads ordered
  P3/P4/P5. Their grids must have stride-8/16/32 geometry and each logical
  depth must be `3 * (num_classes + 5)`. BoxDecode applies sigmoid, the grid
  and stride transform, and the standard YOLOv5 anchors
  (`{10,13},{16,30},{33,23}`; `{30,61},{62,45},{59,119}`;
  `{116,90},{156,198},{373,326}`). Custom AutoAnchor tables and decoded
  six-tensor box/class exports must use another contract.
- `YoloV26` uses grouped raw l/t/r/b bbox heads plus class-score heads.
- `Ssd` is **not** a generic SSD decoder. It resolves exactly **four prepared profiles**
  from the complete ordered loc/conf H/W/C signature at compile time. Any other
  head set or order is rejected with an error that prints the observed and supported
  signatures:
  - **SSD300** (`dboxes300_coco`): 300×300 input, feature maps
    `{38,19,10,5,3,1}`, priors-per-cell `{4,6,6,6,4,4}`, confidence channel order
    `class*A + anchor`, class scores via **softmax** over the class dimension
    (background at index 0 included).
  - **SSD-Mobile-300-v1** (`ssd_anchor_generator`): 300×300 input, feature
    maps `{19,10,5,3,2,1}`, priors-per-cell `{3,6,6,6,6,6}`, confidence channel
    order `anchor*C + class`, class scores via per-class **sigmoid** (background
    ignored).
  - **SSD-Mobile-320-v1** (`ssd_anchor_generator`): 320×320 input, feature
    maps `{20,10,5,3,2,1}`, priors-per-cell `{3,6,6,6,6,6}`, confidence channel
    order `anchor*C + class`, class scores via per-class **sigmoid** (background
    ignored).
  - **SSDlite-Mobile-320-v1** (TorchVision `DefaultBoxGenerator`): 320×320
    input, feature maps `{20,10,5,3,2,1}`, six priors per cell at every level,
    localization order `anchor*4 + {dx,dy,dw,dh}`, confidence order
    `anchor*C + class`, and class scores via **softmax** over all 91 classes
    including background.

  All recipes use grouped per-level localization heads (depth =
  `4 * priors-per-cell`) paired with class-confidence heads (depth =
  `num_classes * priors-per-cell`), FasterRcnnBoxCoder variance scaling
  (`scale_xy 0.1`, `scale_wh 0.2`), and a **stretch** (anisotropic) preprocessing
  resize. The score activation is fixed by the recipe (matching the on-device
  decoder), and the grouped-by-role layout is selected automatically — leave
  `decode_type_option` as `Auto`. A non-grouped layout token is rejected.

  The **model frame is part of the profile**, not just the head geometry. SSD300-v1
  and SSD-Mobile-300-v1 require 300×300; both 320-v1 profiles require 320×320. A
  resolved preprocess resize target or model-dimension override of any other
  size is rejected at build time, because the prior tables and the stretch
  back-projection are only valid at that frame.

  Raw/standalone `SimaBoxDecode` construction never invents a resize mode. Keep
  the upstream `Preproc` metadata requirement, or use the explicit raw overload
  to assert externally performed `ResizeMode::Stretch`; Letterbox and Crop are
  rejected.

  **`num_classes` contract.** The encoded class count is always derived from the
  confidence-head depth (`conf_depth / priors-per-cell`, background at index 0
  included). SSD300-v1 permits a contiguous prefix selection such as the prepared
  81-to-8 route; the other three profiles require the exact encoded count. An
  invalid selection is rejected at build time. Leave it unset to use the profile
  default.

  **91-class MobileNet exports.** With exactly 91 encoded classes, SSD-Mobile-300-v1
  (MobileNet V1 and V2) and SSDlite-Mobile-320-v1 (TorchVision MobileNet V3) decode
  as the source models do: every foreground anchor/class pair is kept and NMS runs
  per class in floating point (SSDlite keeps up to 300 candidates per class before
  NMS). The output is the same BBOX payload. Other SSD profiles and class counts
  keep their existing behavior.
- `Detr` infers class channels from the maximum head depth and requires a valid
  class dimension.
- `EffDet` and `Centernet` use their model-family contracts; do
  not route them through a YOLO decode type.
- `*-seg` decode types produce box-leading output plus task-specific mask data.

If a custom model pack does not match either complete ordered signature, prepare a
new explicitly supported profile rather than weakening the matcher.

## Python note

When configuring model options from Python, use the typed enum rather than a string when available:

```python
opt = pyneat.ModelOptions()
opt.decode_type = pyneat.BoxDecodeType.YoloV8
```

Parse outputs with the helper that matches the model task:

```python
outputs = model.run([image])

boxes = pyneat.decode_bbox(outputs)[0].to_numpy()

pose = pyneat.decode_pose(outputs)[0]
pose_boxes = pose.boxes.to_numpy()
keypoints = pose.keypoints.to_numpy()

seg = pyneat.decode_segmentation(outputs)[0]
seg_boxes = seg.boxes.to_numpy()
masks = seg.masks.to_numpy()
```

## Upgrading

The earlier BoxDecode options introduced ABI 6. The current source-arithmetic options require ABI 7 (`libsima_neat.so.7`). Rebuild C++ applications, plugins and Python bindings with matching headers and libraries. Do not load ABI 5 or 6 binaries through compatibility symlinks.


### Combined YOLOX input admission

`YoloXSegPose` accepts BF16 and INT8 heads, or native FP32 heads in contiguous
or positive, non-overlapping strided storage (including planar CHW views).
FP32 addresses and strides must be four-byte aligned; channel-blocked FP32 and
UINT8 are rejected. No implicit lossy FP32 conversion is added. The prototype
has its own dtype and per-tensor INT8 quantization; the roles of each detection
head must still agree on dtype. Core preserves these per-port facts for Kernels
admission rather than applying the first head's dtype or quantization to every port.
The three detection levels have strides 8, 16 and 32 in both axes. The model
width and height must be multiples of 32; they need not be equal. The prototype
remains 160×160×32 and does not resize silently with model dimensions.

Use preprocess metadata for off-center padding or crop coordinates. The affine
maps **model to original** coordinates; shear, rotation and ROI lists with more than one slot are
unsupported by this combined profile. Crop needs authoritative affine metadata at
preparation. This is a decoder contract for externally preprocessed tensors,
not new support for crop production in `Model` preprocessing. A model with
fewer than 17 keypoints still publishes the fixed
17-slot public wire format, with unused slots set to zero.


## Crop coordinate mapping {#crop-coordinate-mapping}

Model-managed Crop supplies the actual inverse image transform to supported detection profiles, including classic DETR, DetrFocal and ordinary YOLO detection. Selection and NMS stay in model coordinates; accepted boxes are then mapped and clipped to the original image. Missing or conflicting transform metadata fails explicitly. This does not enable crop for RCNN proposals. Existing oriented-box profiles still require equal axis scales; integer resize dimensions may make the scales unequal.

A Preproc that crops a region of interest (ROI) and letterboxes or stretches it into the model canvas publishes, for each frame, the affine that maps model coordinates back to the full source frame. The detection profiles decode every frame with the affine its preprocess published, so boxes come out in source-frame pixels for a single ROI, a centered or non-centered letterbox, a stretch and a whole frame alike. The source frame is the uncropped image, never the ROI. Whole-frame detections are reported in source-frame pixels. Ingress that publishes only the image dimensions, with no affine, keeps the whole-frame mapping.

Pass one ROI per BoxDecode call. For an ROI-list buffer, use capacity 1.

| Profile | Single ROI, crop or stretch |
| --- | --- |
| YOLO26, YOLOv5, YOLOv6, YOLOv7, YOLOv8 to YOLOv11, YOLOX, decoded YOLO, EfficientDet, CenterNet, DETR, DetrFocal, SSD (not the 91-class multiclass SSD recipes) | Boxes in source-frame pixels |
| YOLOX segmentation and pose, oriented boxes | Boxes in source-frame pixels (oriented boxes need equal axis scales) |
| Pose (YOLOv8, YOLO26), segmentation (YOLOv5, v7, v8, v9, v10, YOLO26), 91-class multiclass SSD | Boxes and keypoints in source-frame pixels, when the installed Kernels release accepts the transform for the profile; otherwise fails, naming the profile |
| RCNN proposals | Fails, naming the profile |
| Segment Anything | Fails: it needs the whole image resized once |

Segmentation masks stay in model space: each 160×160 mask covers the whole model canvas, not the box and not the source frame, and is cropped by the box in model coordinates. A mask therefore does not move or scale with the ROI, while its box does; map a mask pixel to the source frame with the same affine the Preproc published for the frame.

A profile fails instead of decoding in whole-frame coordinates when its kernel accepts no transform: RCNN proposals always, and pose, segmentation and multiclass SSD with a Kernels release that predates the transform. It still decodes whole frames. A Crop resize requested for such a profile fails when the graph is built.
