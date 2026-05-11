---
title: BoxDecode Decode Types
description: Choose the correct BoxDecodeType and match tensor contracts expected by genericboxdecode_v2
sidebar_position: 6
---

# BoxDecode Decode Types

`BoxDecodeType` is now a typed API (`simaai::neat::BoxDecodeType` / `neat.BoxDecodeType`) and should always be set explicitly for decode stages.

The runtime contract below comes from `internals/gst_plugins/genericboxdecode_v2/gstneatboxdecode.cpp` (`infer_num_classes`, `infer_yolo_decoupled_classes`, `infer_yolo_packed_classes`, `compute_required_output_size`).

## Core Tensor Contract Rules

- YOLO-family decode types (`yolo`, `yolov5*`, `yolov7*`, `yolov8*`, `yolov9*`, `yolov10*`):
  - Decoupled heads: class-head depths must be repeatable and `> 4`.
  - Packed heads: each head depth must satisfy `depth = 3 * (num_classes + 5)` and be consistent across heads.
- `detr`: class channels are inferred from the maximum depth across heads, and must be `> 4`.
- Other non-YOLO decode types (`effdet`, `rcnn-stage1`, `centernet`): fallback class inference uses max depth and requires `> 4`.
- Segmentation decode tokens (`*-seg`) enable segmentation-like output sizing in v2 (adds mask payload per detection).

## Type Mapping

| API enum | Backend token | Expected contract |
| --- | --- | --- |
| `BoxDecodeType::Yolo` | `yolo` | YOLO decoupled or packed depth contract |
| `BoxDecodeType::YoloV5` | `yolov5` | YOLO decoupled or packed depth contract |
| `BoxDecodeType::YoloV5Seg` | `yolov5-seg` | YOLO depth contract + segmentation path |
| `BoxDecodeType::YoloV7` | `yolov7` | YOLO decoupled or packed depth contract |
| `BoxDecodeType::YoloV7Seg` | `yolov7-seg` | YOLO depth contract + segmentation path |
| `BoxDecodeType::YoloV8` | `yolov8` | YOLO decoupled or packed depth contract |
| `BoxDecodeType::YoloV8Seg` | `yolov8-seg` | YOLO depth contract + segmentation path |
| `BoxDecodeType::YoloV8Pose` | `yolov8-pose` | YOLO decoupled or packed depth contract |
| `BoxDecodeType::YoloV9` | `yolov9` | YOLO decoupled or packed depth contract |
| `BoxDecodeType::YoloV9Seg` | `yolov9-seg` | YOLO depth contract + segmentation path |
| `BoxDecodeType::YoloV10` | `yolov10` | YOLO decoupled or packed depth contract |
| `BoxDecodeType::YoloV10Seg` | `yolov10-seg` | YOLO depth contract + segmentation path |
| `BoxDecodeType::Detr` | `detr` | `num_classes = max(depth)` (must be `> 4`) |
| `BoxDecodeType::EffDet` | `effdet` | fallback max-depth inference (`> 4`) |
| `BoxDecodeType::RcnnStage1` | `rcnn-stage1` | fallback max-depth inference (`> 4`) |
| `BoxDecodeType::Centernet` | `centernet` | fallback max-depth inference (`> 4`) |

## Fail-Fast Behavior

- `stages::BoxDecodeOptions` now requires explicit construction with a decode type.
- `stages::BoxDecode(...)` and `nodes::SimaBoxDecode(...)` fail fast on `BoxDecodeType::Unspecified`.

## Examples

```cpp
simaai::neat::stages::BoxDecodeOptions opt(simaai::neat::BoxDecodeType::YoloV8);
opt.detection_threshold = 0.25;
opt.nms_iou_threshold = 0.5;
opt.top_k = 100;
```

```python
opt = neat.ModelOptions()
opt.decode_type = neat.BoxDecodeType.YoloV8
```
