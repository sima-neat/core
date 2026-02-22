# 006 Postproc Boxdecode

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Intermediate |
| Estimated Read Time | 15-20 minutes |
| Labels | postprocessing, boxdecode, detection |

## Concept
Postprocessing converts raw model outputs into actionable detection results (boxes, scores, classes).

In this chapter, `BoxDecode` is the key focus. It is a highly optimized detection postprocessing path for vision workloads, designed to efficiently transform inference tensors into final bounding-box results with thresholding and NMS.

Common box-decode controls in this tutorial:
- `decode_type` (for example `yolov8`): selects model-family decode behavior.
- `score_threshold`: drops low-confidence detections early.
- `nms_iou_threshold`: controls overlap suppression aggressiveness.
- `top_k`: limits final detection count for deterministic downstream cost.
- `original_width`, `original_height`: maps decoded boxes to the source image coordinate space.

Use-case guidance:
- Too many noisy boxes: increase `score_threshold` and/or reduce `top_k`.
- Duplicate overlapping boxes: lower `nms_iou_threshold` to make suppression stricter.
- Missed true positives: decrease `score_threshold` cautiously.
- Boxes appear scaled/offset incorrectly: verify `original_width` and `original_height` match real source frames.
- Porting between detector variants: ensure `decode_type` matches the model family expected by the MPK.

Reference:
- [Model](/getting-started/programming-model/model)
- [Model Options](/reference/{lsa}/structs/simaai-neat-model-options)

## Learning Process
1. Configure model/postproc options for a detector-style pipeline.
2. Run deterministic preproc + inference + boxdecode flow.
3. Inspect decoded output signals (box count, output kind/fields).
4. Validate run completion via `CHECK`, `SIGNATURE`, and `[OK]` markers.

## What To Observe
- `CHECK ...` lines should indicate contract and runtime validation outcomes.
- `SIGNATURE ...` output should summarize machine-parseable chapter behavior.
- `[OK] ...` indicates the chapter flow completed successfully.

## Run
```bash
./tutorial_v2_006_postproc_boxdecode
python3 tutorials/006_postproc_boxdecode/postproc_boxdecode.py
```

## Source Files
- C++: `tutorials/006_postproc_boxdecode/postproc_boxdecode.cpp`
- Python: `tutorials/006_postproc_boxdecode/postproc_boxdecode.py`
