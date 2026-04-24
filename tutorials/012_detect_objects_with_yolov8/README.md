# 012 Detect Objects with YOLOv8

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Intermediate |
| Estimated Read Time | 15-20 minutes |
| Labels | yolo, detection, mpk |

## Concept

Load a YOLOv8 MPK, feed it one image as a `Tensor`, and read back detection boxes. End-to-end object detection in a handful of lines — the sanity check that says "the YOLO path works on this board."

This chapter bridges "I have a YOLO MPK" to "I can run detection and inspect outputs." It uses the same `model.run()` call as chapter 001, now with detection-shaped output instead of classification logits.

**APIs introduced**
- `pyneat.ModelOptions()` with YOLO-specific fields (`.decode_type="yolov8"`, `.score_threshold`, `.nms_iou_threshold`, `.top_k`, `.original_width/height`).
- `pyneat.Tensor.from_numpy(rgb, image_format=pyneat.PixelFormat.RGB)` — explicit tensor input.
- `model.run(tensor, timeout_ms)` — same call as chapter 001, YOLO-shaped output.

**When to use this**
- First detector bring-up on a new board/runtime.
- Verifying required plugins and model assets are present.
- Establishing a known-good baseline before threshold/NMS tuning.

**Prerequisites**
Chapter 001 (Model). Chapter 004 for `ModelOptions`. Chapter 006 for reading detection outputs.

**References**
- [Model](/getting-started/programming-model/model)
- [Model Options](/reference/{lsa}/structs/simaai-neat-model-options)

## Learning Process
1. Resolve YOLO MPK and deterministic image/tensor input.
2. Build and run detection path (preprocess + inference + decode).
3. Inspect output structure (kind/fields) to validate detector wiring.

## Run

Fetch the YOLOv8-s MPK once: `sima-cli modelzoo -v 2.0.0 get yolo_v8s`.

**Python:**
```bash
python3 share/sima-neat/tutorials/012_detect_objects_with_yolov8/detect_objects_with_yolov8.py \
  --mpk /path/to/yolo_v8s.tar.gz --image /path/to/frame.jpg --size 640
```

**C++:**
```bash
./lib/sima-neat/tutorials/tutorial_v2_012_detect_objects_with_yolov8 \
  --mpk /path/to/yolo_v8s.tar.gz --image /path/to/frame.jpg
```

To compile this chapter's C++ source in your own project with a custom `CMakeLists.txt` (no extras folder required), see [How to Run Tutorials](/tutorials/v2#compile-a-copy-yourself) on the landing page.

## Source Files
- C++: `tutorials/012_detect_objects_with_yolov8/detect_objects_with_yolov8.cpp`
- Python: `tutorials/012_detect_objects_with_yolov8/detect_objects_with_yolov8.py`
