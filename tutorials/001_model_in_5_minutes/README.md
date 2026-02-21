# 001 Model in 5 Minutes

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Beginner |
| Estimated Read Time | <5 minutes |
| Labels | model, mpk, inference, foundations |

## Concept
This tutorial teaches the quickest practical path to run inference with a compiled model in NEAT.

A compiled model is a deployable model package (`.tar.gz`, often called an MPK) that NEAT can load and execute on the target device. It contains the model artifacts and runtime metadata needed for inference. You provide input data, run inference, and consume model outputs.

After this chapter, you should understand the minimum end-to-end loop:
- Load a compiled model package.
- Prepare input data that matches model expectations.
- Run synchronous inference.
- Read and validate output behavior.

**References**
- [Model](/getting-started/programming-model/model)

## Learning Process
1. Set up runtime inputs: parse CLI args, locate the compiled ResNet50 MPK, and prepare sample input data.
2. Build the minimal model execution path for one model and one input stream.
3. Run synchronous inference to keep behavior deterministic and easy to debug.
4. Inspect top-1 predictions and validation output (`CHECK`, `SIGNATURE`, `[OK]`).

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
