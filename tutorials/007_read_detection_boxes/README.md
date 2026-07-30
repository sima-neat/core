# 007 Read Detection Boxes from Model Output

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Intermediate |
| Estimated Read Time | 15-20 minutes |
| Model | yolo_v8s |
| Labels | postprocessing, boxdecode, detection |

## Concept

Decode raw model output into usable bounding boxes using `SimaBoxDecode` — thresholding, NMS, and coordinate mapping built into one postprocessing stage — then read the result as either parsed boxes or the raw packed byte buffer.

## Walkthrough

A detector doesn't return boxes directly. Its raw output is a stack of feature maps that still needs thresholding, non-maximum suppression, and coordinate mapping before it means anything. `SimaBoxDecode` is the postprocessing stage that does all three in one optimized step, turning inference tensors into final detections in source-image pixels.

This chapter configures that decode — picking the model family with `decode_type`, gating confidence with the score threshold, suppressing overlaps with the NMS IoU threshold, and capping output with `top_k` — then runs the model and reads how many detections came back. By the end you will have a configured detector pipeline and a detection count read from its output, plus (in the In Practice reference below) the full wire format so you can parse boxes yourself in any runtime.

### Configure the decode {#step-configure-decode}

These options set both the input contract and the postprocessing behavior. `decode_type` (`YoloV8` here) selects the model-family decode path. The confidence threshold drops weak candidates before NMS; the NMS IoU threshold controls how aggressively overlapping boxes are merged; `top_k` caps the final count for deterministic downstream cost; and `boxdecode_original_width`/`boxdecode_original_height` map decoded coordinates back into source-image pixels. Tuning guidance for each of these is in In Practice below.

**C++:** `decode_type` takes the `BoxDecodeType::YoloV8` enum. The threshold/NMS/`top_k` values are passed later through `stages::BoxDecodeOptions`, not on `Model::Options`.

**Python:** `decode_type` takes the `pyneat.BoxDecodeType.YoloV8` enum, and `score_threshold`, `nms_iou_threshold`, `top_k`, and `boxdecode_original_width`/`boxdecode_original_height` are set directly on `ModelOptions`. (`score_threshold` and the C++ `detection_threshold` name the same control — see the naming note in In Practice.)

### Build the model {#step-load-model}

Constructing the `Model` from the archive plus options binds the decode configuration to the model so the inference and postprocessing stages derived from it use the settings above.

### Run preprocess, inference, and decode {#step-run-decode}

This is where a frame flows through preprocess, the MLA inference, and the box decoder to produce the detection output.

**C++:** The path is made explicit stage-by-stage: `stages::Preproc` produces the input tensor, `stages::Infer` runs the model, and a `stages::BoxDecodeOptions` (with `detection_threshold = 0.55`, `nms_iou_threshold = 0.5`, `top_k = 100`) configures the decode that runs next.

**Python:** `model.run([tensor])` runs the whole configured path in one call and returns a `TensorList`. When BoxDecode is wired into the model route, the first tensor is the packed `BBOX` output.

### Read the boxes {#step-read-boxes}

Finally, turn the decode output into something you can use.

**C++:** `stages::BoxDecodeResults(...)` returns a `BoxDecodeResultList`; the front result's `boxes` vector is already parsed into `{x1, y1, x2, y2, score, class_id}` clamped to source pixels, so `decoded.boxes.size()` is the detection count.

**Python:** The result is a single `BBOX` `uint8` tensor in `outputs[0]`. The first four little-endian bytes are the detection count (`struct.unpack_from("<I", buf, 0)`); the full record layout is documented in In Practice. If a runtime doesn't wire BoxDecode into `model.run`, the returned `TensorList` contains the raw feature-map heads instead.

## In Practice

`SimaBoxDecode` emits a single output tensor tagged `BBOX`. The tensor carries a
packed byte buffer that the runtime parser interprets into floating-point
detections. Understanding that two-level contract (wire buffer vs. parsed
`Box` records) is the key to reading the output from either Python or C++.

### BBOX tensor

The decode stage produces one `BBOX` tensor per input frame with:

| Field | Value |
| --- | --- |
| `semantic.detection.format` | `"BBOX"` |
| `dtype` | `UInt8` |
| `shape` | rank-1: `[N_bytes]`, where `N_bytes` is the model archive-packed buffer capacity (for example `[20160]` on the stock YOLOv8 pack) |

The tensor shape is a **byte count**, not a detection count. The packed bytes
hold both a small header and a contiguous array of fixed-size box records.
`N_bytes` is determined by the model archive's `buffers.input[0].size` field (inside
the boxdecode stage's config JSON) and bounds the maximum number of
detections the decoder can emit in a single frame (see "Override contract"
below for how runtime dims interact with packaged values).

### Packed wire format

The `uint8` buffer is laid out little-endian:

```
offset  size  content
------  ----  -------
  0      4    uint32  N = number of valid detections in this frame
  4     24    RawBox[0]
 28     24    RawBox[1]
  .      .      ...
  .      .    RawBox[N-1]
                   (trailing bytes up to buffer capacity are padding, ignored)
```

Each `RawBox` record is 24 bytes:

| Offset in record | Size | Type  | Field | Meaning |
| --- | --- | --- | --- | --- |
|  0 | 4 | int32 | `x`     | top-left x, in source pixels |
|  4 | 4 | int32 | `y`     | top-left y, in source pixels |
|  8 | 4 | int32 | `w`     | width, in source pixels  |
| 12 | 4 | int32 | `h`     | height, in source pixels |
| 16 | 4 | float32 | `score` | post-NMS detection confidence in `[0.0, 1.0]` (the value `detection_threshold` gates on) |
| 20 | 4 | int32 | `class_id` | predicted class id (model-defined; 0-indexed; class-name map lives in the model archive metadata) |

The canonical Python `struct` format matching one record is `"<iiiifi"`
(little-endian, 4 signed ints, one float, one signed int).

The runtime's parsing helpers (`parse_bbox_bytes` /
`decode_bbox_tensor` in `include/pipeline/DetectionTypes.h`,
`tests/unit_testing/unit_detection_types_bbox_test.cpp` pins the wire contract)
expand each `RawBox` into a `Box` struct for downstream code:

```cpp
struct Box {
  float x1, y1, x2, y2;  // x2 = x + w, y2 = y + h; clamped to [0, img_w|h]
  float score;
  int   class_id;
};
```

### Coordinate space

Coordinates decoded from `BBOX` are in **original-image pixels**, the same
coordinate system you passed as `original_width` / `original_height` (or that
the model archive was packaged with). They are **not** normalized to `[0, 1]`, and they
are **not** expressed in the model's internal letterboxed input space. The
parser clamps `(x1, y1, x2, y2)` to `[0, original_width]` / `[0, original_height]`
so caller code can draw them directly on the source frame.

### Worked example

With the tutorial's runtime configuration (`original_width = 640`,
`original_height = 640`, `top_k = 100`) and the stock YOLOv8 pack
(`buffers.input[0].size = 20160` in the boxdecode config), a single decoded
frame yields:

- `out.kind == SampleKind.Tensor`
- `out.payload_tag == "BBOX"`
- `out.tensor.dtype == UInt8`, `out.tensor.shape == [20160]`
- Bytes `[0:4]` give `N` in little-endian; `0 <= N <= 100` because
  `top_k = 100`. An `N` of `0` means "no detections above threshold this
  frame" — iterate zero times and emit nothing.
- Bytes `[4 : 4 + 24 * N]` hold the valid detections; everything after that
  offset is zero/padding and must be ignored.

Reading a box in Python is a `struct.unpack_from`:

```python
import struct
payload = out.tensor.copy_payload_bytes()
count = struct.unpack_from("<I", payload, 0)[0]
for i in range(count):
    x, y, w, h, score, cls = struct.unpack_from("<iiiifi", payload, 4 + 24 * i)
    # (x, y, w, h) in source pixels; x2 = x + w, y2 = y + h
```

In C++ the `stages::BoxDecode` helper returns a `BoxDecodeResult` that has
already done this unpack for you: `result.boxes[i]` is a `Box` with
`(x1, y1, x2, y2)` already populated from `(x, y, x+w, y+h)` and clamped to
the image.

### Override contract: runtime dims vs. packaged model archive defaults

`SimaBoxDecode` is constructed from a trained model archive that ships with packaged
defaults for `decode_type`, `detection_threshold`, `nms_iou_threshold`,
`top_k`, `original_width`, and `original_height`. The public constructor

```cpp
SimaBoxDecode(const Model& model,
              const std::string& decode_type = "",
              int original_width = 0, int original_height = 0,
              double detection_threshold = 0.0,
              double nms_iou_threshold = 0.0,
              int top_k = 0);
```

and its Python twin `pyneat.nodes.sima_box_decode(model, ...)` use a simple
"positive overrides, zero/empty preserves" rule per field.

> **Naming note.** `detection_threshold` is the name used by
> `SimaBoxDecode`'s constructor. `ModelOptions.score_threshold` (used in the
> Python tutorial) is plumbed into that same argument. The two names refer
> to the same underlying control.

| Runtime argument | Value passed | Behavior |
| --- | --- | --- |
| `decode_type` | `""` (empty) | preserve model archive / model-path inference |
| `decode_type` | non-empty string | override the model archive value for this run |
| `original_width` / `original_height` | `0` | preserve model archive packaged dimension |
| `original_width` / `original_height` | positive int | rewrite `original_width` / `original_height` in the effective config |
| `detection_threshold` | `0.0` | preserve model archive packaged threshold |
| `detection_threshold` | `> 0.0` | override (also triggers the YOLOv8 cliff-warning below) |
| `nms_iou_threshold` | `0.0` | preserve model archive packaged NMS IoU |
| `nms_iou_threshold` | `> 0.0` | override |
| `top_k` | `0` | preserve model archive packaged top-K |
| `top_k` | `> 0` | override |

The rule is strictly per-field:

- **Python path** — the tutorial overrides every field because
  `ModelOptions` sets positive values.
- **C++ path** — `read_detection_boxes.cpp` passes `0.55f, 0.5f, 100` (so
  `detection_threshold`, `nms_iou_threshold`, and `top_k` are overridden)
  plus `bgr.cols, bgr.rows` positively (so `original_width` /
  `original_height` are overridden too).

Practical consequences:

- If your model archive was packed for a different resolution than your source frames,
  pass `original_width` and `original_height` explicitly so coordinates land
  in source pixels.
- Leaving `detection_threshold` and `nms_iou_threshold` at `0.0` is the
  safest way to get the model archive's validated defaults; only override when you are
  deliberately retuning.
- Be deliberate with a low `detection_threshold`. The lower it is, the more
  candidate boxes survive thresholding, and NMS cost grows with the square of
  the surviving-box count — so a very low threshold can sharply increase
  postprocess compute and latency. Lower it only as far as you need to catch
  weak detections; pair it with `top_k` to cap the worst case.

### Decode types and tensor contracts

`BoxDecodeType` is a typed API (`simaai::neat::BoxDecodeType` / `neat.BoxDecodeType`) and should always be set explicitly for decode stages. The runtime contract below comes from `internals/gst_plugins/genericboxdecode_v2/gstneatboxdecode.cpp` (`infer_num_classes`, `infer_yolo_decoupled_classes`, `infer_yolo_packed_classes`, `compute_required_output_size`).

Core tensor contract rules:
- YOLO-family decode types (`yolo`, `yolov5*`, `yolov7*`, `yolov8*`, `yolov9*`, `yolov10*`):
  - Decoupled heads: class-head depths must be repeatable and `> 4`.
  - Packed heads: each head depth must satisfy `depth = 3 * (num_classes + 5)` and be consistent across heads.
- `yolo26`: decoupled grouped heads with 4-channel raw l/t/r/b bbox tensors and repeatable class-head depths `> 4`.
- `detr`: class channels are inferred from the maximum depth across heads, and must be `> 4`.
- Other non-YOLO decode types (`effdet`, `rcnn-stage1`, `centernet`): fallback class inference uses max depth and requires `> 4`.
- Segmentation decode tokens (`*-seg`) enable segmentation-like output sizing in v2 (adds mask payload per detection).

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
| `BoxDecodeType::YoloV26` | `yolo26` | YOLO26 grouped raw l/t/r/b bbox heads + class-score heads |
| `BoxDecodeType::Detr` | `detr` | `num_classes = max(depth)` (must be `> 4`) |
| `BoxDecodeType::EffDet` | `effdet` | fallback max-depth inference (`> 4`) |
| `BoxDecodeType::RcnnStage1` | `rcnn-stage1` | fallback max-depth inference (`> 4`) |
| `BoxDecodeType::Centernet` | `centernet` | fallback max-depth inference (`> 4`) |

Fail-fast behavior:
- `stages::BoxDecodeOptions` requires explicit construction with a decode type.
- `stages::BoxDecode(...)` and `nodes::SimaBoxDecode(...)` fail fast on `BoxDecodeType::Unspecified`.

Setting the decode type explicitly:

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

## Run

Run the **Python** and **C++ (prebuilt)** commands from the **Neat install root** (the directory that contains `share/` and `lib/`); run the **build from source** commands from the **repo root**.

**Python:**
```bash
python3 share/sima-neat/tutorials/007_read_detection_boxes/read_detection_boxes.py \
  --model /tmp/yolo_v8s.tar.gz --width 640 --height 640
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_007_read_detection_boxes \
  --model /tmp/yolo_v8s.tar.gz --image /path/to/frame.jpg
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_007_read_detection_boxes
./build/tutorials-standalone/tutorial_007_read_detection_boxes \
  --model /tmp/yolo_v8s.tar.gz --image /path/to/frame.jpg
```

Expected output (the box count depends on the frame; a synthetic frame yields zero):

```text
boxes=0
[OK] 007_read_detection_boxes
```

(The Python build prints `detections=...`, or `raw_output_heads=...` if the runtime does not wire BoxDecode into `model.run`.) To integrate this chapter's C++ source into your own project with a custom `CMakeLists.txt` (no extras folder required), see [How to Run Tutorials](/tutorials#compile-a-copy-yourself) on the landing page.

## Source Files
- C++: `tutorials/007_read_detection_boxes/read_detection_boxes.cpp`
- Python: `tutorials/007_read_detection_boxes/read_detection_boxes.py`
