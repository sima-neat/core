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

## Output Structure

`SimaBoxDecode` emits a single output tensor tagged `BBOX`. The tensor carries a
packed byte buffer that the runtime parser interprets into floating-point
detections. Understanding that two-level contract (wire buffer vs. parsed
`Box` records) is the key to reading the output from either Python or C++.

### Sample wrapper

Regardless of language, the decode stage produces one `Sample` per input frame
with:

| Field | Value |
| --- | --- |
| `kind` | `SampleKind.Tensor` (single-tensor sample, not a bundle) |
| `payload_tag` / `format` | `"BBOX"` |
| `media_type` | `"application/vnd.simaai.tensor"` |
| `tensor.dtype` | `UInt8` |
| `tensor.shape` | rank-1: `[N_bytes]`, where `N_bytes` is the MPK-packed buffer capacity (for example `[20160]` on the stock YOLOv8 pack) |
| `fields` | empty (all payload lives in `tensor`) |

The tensor shape is a **byte count**, not a detection count. The packed bytes
hold both a small header and a contiguous array of fixed-size box records.
`len(N_bytes)` is determined by the MPK's `buffers.output.size` field and
bounds the maximum number of detections the decoder can emit in a single
frame (see "Override contract" below for how runtime dims interact with
packaged values).

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
| 16 | 4 | float32 | `score` | post-NMS detection confidence (the value `detection_threshold` gates on) |
| 20 | 4 | int32 | `class_id` | predicted class id (model-defined) |

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
the MPK was packaged with). They are **not** normalized to `[0, 1]`, and they
are **not** expressed in the model's internal letterboxed input space. The
parser clamps `(x1, y1, x2, y2)` to `[0, original_width]` / `[0, original_height]`
so caller code can draw them directly on the source frame.

### Worked example

With the tutorial's runtime configuration (`original_width = 640`,
`original_height = 640`, `top_k = 100`) and the stock YOLOv8 pack
(`buffers.output.size = 20160`), a single decoded frame yields:

- `out.kind == SampleKind.Tensor`
- `out.payload_tag == "BBOX"`
- `out.tensor.dtype == UInt8`, `out.tensor.shape == [20160]`
- Bytes `[0:4]` give `N` in little-endian; `N <= 100` because `top_k = 100`.
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

### Override contract: runtime dims vs. packaged MPK defaults

`SimaBoxDecode` is constructed from a trained MPK that ships with packaged
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
"positive overrides, zero/empty preserves" rule per field:

| Runtime argument | Value passed | Behavior |
| --- | --- | --- |
| `decode_type` | `""` (empty) | preserve MPK / model-path inference |
| `decode_type` | non-empty string | override the MPK value for this session |
| `original_width` / `original_height` | `0` | preserve MPK packaged dimension |
| `original_width` / `original_height` | positive int | rewrite `original_width` / `original_height` in the effective config |
| `detection_threshold` | `0.0` | preserve MPK packaged threshold |
| `detection_threshold` | `> 0.0` | override (also triggers the YOLOv8 cliff-warning below) |
| `nms_iou_threshold` | `0.0` | preserve MPK packaged NMS IoU |
| `nms_iou_threshold` | `> 0.0` | override |
| `top_k` | `0` | preserve MPK packaged top-K |
| `top_k` | `> 0` | override |

The rule is strictly per-field: the tutorial's Python path overrides all of
them because `ModelOptions` sets positive values, while
`postproc_boxdecode.cpp`'s constructor form passes `0.52f, 0.5f, 100` —
those override thresholds and top-K but, because it also sets
`bgr.cols, bgr.rows` positively, `original_*` is overridden too.

Practical consequences:

- If your MPK was packed for a different resolution than your source frames,
  pass `original_width` and `original_height` explicitly so coordinates land
  in source pixels.
- Leaving `detection_threshold` and `nms_iou_threshold` at `0.0` is the
  safest way to get the MPK's validated defaults; only override when you are
  deliberately retuning.
- For YOLOv8-family packs, a resolved `detection_threshold <= 0.5` triggers
  a `[WARN] SimaBoxDecode: resolved detection-threshold=...` on `stderr` at
  construction time. Prefer `>= 0.51` (the tutorial uses `0.52`) to avoid
  severe pre-NMS latency cliffs from borderline 0-logit candidates.

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
