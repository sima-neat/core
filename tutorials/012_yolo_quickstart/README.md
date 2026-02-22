# 012 YOLO Quickstart

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Intermediate |
| Estimated Read Time | 15-20 minutes |
| Labels | yolo, detection, mpk |

## Concept
This tutorial is the fastest path to run a YOLO-style detector in NEAT and confirm the detection pipeline is wired correctly.

For new users, this chapter provides a practical bridge from "I have a YOLO MPK" to "I can run detection and inspect outputs." It uses explicit preprocess + MLA + boxdecode composition in C++, and the equivalent model-option path in Python.

What this chapter demonstrates:
- Loading a YOLO MPK and preparing image/tensor input.
- Running detection-oriented pipeline stages.
- Validating output kind and field structure.

Use-case guidance:
- First detector bring-up on a new board/runtime.
- Verifying required plugins and model assets are present.
- Establishing a known-good baseline before threshold/NMS tuning.

Reference:
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
```bash
./tutorial_v2_012_yolo_quickstart
python3 tutorials/012_yolo_quickstart/yolo_quickstart.py
```

## Source Files
- C++: `tutorials/012_yolo_quickstart/yolo_quickstart.cpp`
- Python: `tutorials/012_yolo_quickstart/yolo_quickstart.py`
