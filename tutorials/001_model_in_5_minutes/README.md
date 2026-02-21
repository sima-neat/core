# 001 Model in 5 Minutes

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Beginner |
| Estimated Read Time | <10 minutes |
| Labels | model, mpk, inference, foundations |

## Concept
Work with a compiled model end-to-end using the smallest practical sync loop.

## Learning Process
1. Parse CLI and prepare ResNet50 model + local cv::Mat dataloader.
2. Run synchronous inference over cv::Mat inputs.
3. Emit top1 lines and a stable tutorial signature.
4. Run synchronous inference from PyTorch dataloader batches.

## What To Observe
- `CHECK ...` lines should indicate contract and runtime validation outcomes.
- `SIGNATURE ...` output should summarize machine-parseable chapter behavior.
- `[OK] ...` indicates the chapter flow completed successfully.

## Run
```bash
./tutorial_v2_001_model_in_5_minutes
python3 tutorials/001_model_in_5_minutes/model_in_5_minutes.py
```

## Source Files
- C++: `tutorials/001_model_in_5_minutes/model_in_5_minutes.cpp`
- Python: `tutorials/001_model_in_5_minutes/model_in_5_minutes.py`
