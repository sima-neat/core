---
title: BoxDecode Decode Types
description: Choose the correct BoxDecodeType for object-detection postprocessing
sidebar_position: 6
---

# BoxDecode Decode Types

`nodes::SimaBoxDecode` converts raw detection-head tensors into detection results. It runs after model inference, applies the decode math for the selected model family, filters low-confidence boxes, runs NMS, and emits a tensor payload that starts with decoded boxes. Detection models can parse that payload as boxes; pose and segmentation models can also parse the keypoints or masks that follow the boxes.

For normal model-pack usage, prefer the `Model`-aware constructor. The model archive supplies the tensor order, layout, quantization, class count, resize metadata, and score-domain hints needed by the decoder. Your application usually only chooses the decode family and filtering thresholds.

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
| `nms_iou_threshold` | IoU threshold used by non-maximum suppression. |
| `top_k` | Maximum number of detections to keep. `0` uses the backend/model default. |
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
| Pose | `decode_pose(...)` | `pyneat.decode_pose(...)` | boxes `[N, 6]` and keypoints `[N, 17, 3]` float32: `x, y, visibility` |
| Segmentation | `decode_segmentation(...)` | `pyneat.decode_segmentation(...)` | boxes `[N, 6]` float32 and masks `[N, 160, 160]` uint8 |

Detection-display graphs can feed the result to `SimaRender`. Application code that only needs boxes can continue to use `decode_bbox(...)` on BoxDecode outputs.

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

The model archive can provide defaults for decode type, thresholds, `top_k`, and
source geometry. Runtime arguments override those defaults only when you pass a
non-empty or positive value.

| Runtime argument | Value passed | Behavior |
| --- | --- | --- |
| `decode_type` | empty / `Unspecified` | Preserve model archive or route-planner inference where supported. |
| `decode_type` | concrete type | Override the decode family for this run. |
| `original_width` / `original_height` | `0` | Preserve packaged geometry or upstream preprocess metadata. |
| `original_width` / `original_height` | positive integer | Override source dimensions for coordinate mapping. |
| `detection_threshold` / `score_threshold` | `0.0` | Preserve packaged threshold. |
| `detection_threshold` / `score_threshold` | `> 0.0` | Override the score gate. |
| `nms_iou_threshold` | `0.0` | Preserve packaged NMS IoU. |
| `nms_iou_threshold` | `> 0.0` | Override NMS IoU. |
| `top_k` | `0` | Preserve packaged top-K. |
| `top_k` | `> 0` | Override the maximum kept detections. |

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
| `BoxDecodeType::YoloV6` | `yolov6` | YOLOv6 detection |
| `BoxDecodeType::YoloX` | `yolox` | YOLOX detection |
| `BoxDecodeType::Detr` | `detr` | DETR-style transformer detection |
| `BoxDecodeType::EffDet` | `effdet` | EfficientDet detection |
| `BoxDecodeType::RcnnStage1` | `rcnn-stage1` | R-CNN proposal stage |
| `BoxDecodeType::Centernet` | `centernet` | CenterNet detection |

`BoxDecodeType::Unspecified` is an unset sentinel and fails before runtime. Always choose a concrete type.

## Choosing the right type

- If you are using a SiMa-provided or SiMa-compiled model pack, choose the `BoxDecodeType` that matches the model family and leave `decode_type_option` as `Auto`.
- If your detections are missing or all scores are unexpectedly low, first verify that the decode family matches the exported model head. YOLOX, YOLOv6, and YOLO26 use raw/logit-style heads and should not be treated like probability-only YOLO heads.
- If boxes are shifted or scaled incorrectly, check the image resize policy. Use `resize_mode_override` only when your graph does not have an upstream `Preproc` stage writing resize metadata.
- If you are authoring a custom model pack, ensure the archive describes the detection heads accurately: tensor order, logical shape, physical storage, dtype/quantization, score domain, class count, and any sliced outputs. Application code should not need to compensate for these details.

## Shape and layout guidance

Different detection models expose different head layouts. Some use one tensor per feature-map level; others split boxes, objectness, classes, keypoints, or masks into separate tensors. Some model outputs are dense HWC tensors; others are packed or sliced by the compiler/runtime.

For model-pack flows this is handled by the packaged contract. For manually wired tensors, the key rule is: match the exported head format exactly. Do not choose a decode type based only on rank or channel count.

Advanced tensor-contract rules:

- YOLO-family decode types (`Yolo`, `YoloV5`, `YoloV7`, `YoloV8`, `YoloV9`,
  `YoloV10`, and segmentation/pose variants) expect either decoupled heads or
  packed heads that match the model family.
- Packed YOLO heads must keep class count and head depth consistent across
  feature levels.
- `YoloV26` uses grouped raw l/t/r/b bbox heads plus class-score heads.
- `Detr` infers class channels from the maximum head depth and requires a valid
  class dimension.
- `EffDet`, `RcnnStage1`, and `Centernet` use their model-family contracts; do
  not route them through a YOLO decode type.
- `*-seg` decode types produce box-leading output plus task-specific mask data.

If a custom model pack cannot infer class count or head order, fix the model
archive contract or set the explicit decode options documented above. Guessing
from tensor shape is brittle and hard to debug.

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
