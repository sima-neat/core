# 002 Async Push Pull

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Beginner |
| Estimated Read Time | 10-15 minutes |
| Labels | async, push-pull, throughput, runtime |

## Concept
Understand async execution by separating producer push from consumer pull.

## Learning Process
1. Parse CLI and prepare ResNet50 model + local cv::Mat dataloader.
2. Run async inference with producer-thread push and main-thread pull.
3. Emit top1 lines, async stats, and stable signature.

## What To Observe
- `CHECK ...` lines should indicate contract and runtime validation outcomes.
- `SIGNATURE ...` output should summarize machine-parseable chapter behavior.
- `[OK] ...` indicates the chapter flow completed successfully.

## Run
```bash
./tutorial_v2_002_async_push_pull
python3 tutorials/002_async_push_pull/async_push_pull.py
```

## Source Files
- C++: `tutorials/002_async_push_pull/async_push_pull.cpp`
- Python: `tutorials/002_async_push_pull/async_push_pull.py`
