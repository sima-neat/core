# 002 Async Push Pull

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Beginner |
| Estimated Read Time | 10-15 minutes |
| Labels | async, push-pull, throughput, runtime |

## Concept
This tutorial explains how to use asynchronous APIs to build high-performance production quality applications.

In a synchronous loop, one thread blocks while waiting for each result. That is simple, but it underutilizes compute when input production and output consumption can overlap. Async execution improves throughput by decoupling these stages:
- `push(...)` feeds inputs as they become ready.
- `pull(...)` consumes outputs independently.

This chapter focuses on the core async pattern used in real applications: producer-side async `push` and consumer-side async `pull`, with explicit queue/backpressure behavior.

For the programming concepts behind this flow, see:
- [Session](/getting-started/programming-model/session)
- [Pipeline](/getting-started/programming-model/pipeline)

## Learning Process
1. Prepare runtime inputs: parse CLI args, load ResNet50 MPK, and construct local input samples.
2. Build the async run path and split responsibilities between producer `push(...)` and consumer `pull(...)`.
3. Observe queue-driven behavior and verify throughput-oriented execution.
4. Validate results with top-1 output, async stats, and stable tutorial signature.

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
