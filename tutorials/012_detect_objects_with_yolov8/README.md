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
4. Confirm completion with `CHECK`, `SIGNATURE`, and `[OK]`.

## What To Observe
- `CHECK ...` lines should indicate contract and runtime validation outcomes.
- `SIGNATURE ...` output should summarize machine-parseable chapter behavior.
- `[OK] ...` indicates the chapter flow completed successfully.

## Run

This tutorial loads a YOLOv8-s MPK. You must supply one via `--mpk`. If the flag is omitted and no MPK is cached under `tmp/`, the tutorial prints a `SKIP` line and exits.

### Download the YOLOv8-s MPK

Fetch `yolo_v8s.tar.gz` from the SiMa modelzoo once:

```bash
sima-cli modelzoo -v 2.0.0 get yolo_v8s
```

Note the absolute path to the downloaded `yolo_v8s.tar.gz`; you will pass it to every `--mpk` invocation below. For more on asset resolution (env vars, fallback paths), see [Assets and MPK Setup](/how-to/assets_mpk).

### eLxr SDK (C++)

From inside the paired NEAT eLxr SDK container shell, `dk` forwards execution to the DevKit while keeping paths consistent across the shared NFS workspace:

```bash
NEAT_EXTRAS_ROOT=<sima-neat-*-Linux-extras>
cd $NEAT_EXTRAS_ROOT/lib/sima-neat/tutorials
dk ./tutorial_v2_012_detect_objects_with_yolov8 --mpk /absolute/path/to/yolo_v8s.tar.gz
```

### eLxr SDK (Python)

```bash
NEAT_EXTRAS_ROOT=<sima-neat-*-Linux-extras>
dk python3 $NEAT_EXTRAS_ROOT/share/sima-neat/tutorials/012_detect_objects_with_yolov8/detect_objects_with_yolov8.py \
  --mpk /absolute/path/to/yolo_v8s.tar.gz
```

`dk` runs the script on the paired DevKit using the DevKit-side `pyneat` venv. No activation is required on the SDK side.

### DevKit (C++)

From a shell on the DevKit itself:

```bash
NEAT_EXTRAS_ROOT=<sima-neat-*-Linux-extras>
cd $NEAT_EXTRAS_ROOT/lib/sima-neat/tutorials
./tutorial_v2_012_detect_objects_with_yolov8 --mpk /absolute/path/to/yolo_v8s.tar.gz
```

### DevKit (Python)

```bash
source ~/pyneat/bin/activate
NEAT_EXTRAS_ROOT=<sima-neat-*-Linux-extras>
python3 $NEAT_EXTRAS_ROOT/share/sima-neat/tutorials/012_detect_objects_with_yolov8/detect_objects_with_yolov8.py \
  --mpk /absolute/path/to/yolo_v8s.tar.gz
```

## Source Files
- C++: `tutorials/012_detect_objects_with_yolov8/detect_objects_with_yolov8.cpp`
- Python: `tutorials/012_detect_objects_with_yolov8/detect_objects_with_yolov8.py`
