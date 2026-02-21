# 012 YOLO Quickstart

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Intermediate |
| Estimated Read Time | 15-20 minutes |
| Labels | yolo, detection, mpk |

## Concept
Run a YOLO-style detection path quickly and validate expected output shape.

## Learning Process
1. Parse flags and establish deterministic defaults.
2. Exercise the chapter's primary runtime path.
3. Emit checks and machine-parseable signature.

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
