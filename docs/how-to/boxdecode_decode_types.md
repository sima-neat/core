---
title: BoxDecode Decode Types
description: Choose the correct BoxDecodeType for object-detection postprocessing
sidebar_position: 6
---

# BoxDecode Decode Types

`nodes::SimaBoxDecode` converts raw detection-head tensors into bounding boxes. It runs after model inference, applies the decode math for the selected model family, filters low-confidence boxes, runs NMS, and emits a `BBOX` tensor that can be parsed into typed detections.

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

**Output:** one `BBOX` tensor containing decoded detections. Parse it with `decode_bbox_tensor()` or use `stages::BoxDecodeResults()` when calling the standalone stage API. Detection-display graphs can feed the result to `SimaRender`.

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

## Python note

When configuring model options from Python, use the typed enum rather than a string when available:

```python
opt = neat.ModelOptions()
opt.decode_type = neat.BoxDecodeType.YoloV8
```
