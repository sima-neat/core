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
| SuperPoint | `decode_superpoint(...)` | `pyneat.decode_superpoint(...)` | keypoints `[N,2]`, scores `[N]`, descriptors `[N,D]` |

Axis-aligned detection-display graphs can feed the result to `SimaRender`. Use `decode_bbox(...)` only for the existing box-leading formats; neither it nor `SimaRender` accepts oriented boxes.

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
present. They are not normalized to `[0, 1]` and are not expressed in the
model's internal letterboxed input space.

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
| `BoxDecodeType::Yolo` | `yolo` | Generic YOLO-style heads |
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

- If you are using a SiMa-provided or SiMa-compiled model pack, choose the `BoxDecodeType` that matches the model family and leave `decode_type_option` as `Auto`.
- If your detections are missing or all scores are unexpectedly low, first verify that the decode family matches the exported model head. YOLOX, YOLOv6, and YOLO26 use raw/logit-style heads and should not be treated like probability-only YOLO heads.
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

The new options change C++ object layouts. Core uses ABI 6, `libsima_neat.so.6`. Rebuild C++ applications, plugins and Python bindings with matching headers and libraries. Do not link ABI 5 binaries to ABI 6 through a compatibility symlink.


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
maps **model to original** coordinates; shear, rotation and ROI lists are not
supported by this combined profile. Crop needs authoritative affine metadata at
preparation. This is a decoder contract for externally preprocessed tensors,
not new support for crop production in `Model` preprocessing. A model with
fewer than 17 keypoints still publishes the fixed
17-slot public wire format, with unused slots set to zero.
