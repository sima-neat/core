# 017 Performance Tuning

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Advanced |
| Estimated Read Time | 15-20 minutes |
| Labels | performance, tuning, async, queues |

## Concept
This tutorial explains the first performance knobs most teams tune in async pipelines: queue depth, overflow policy, and metrics.

Why this comes late in the tutorial sequence: performance tuning only helps when your correctness baseline is already stable. This chapter assumes you can already run async push/pull and now want controlled throughput/latency tradeoffs.

What this chapter demonstrates:
- Running async mode with configurable queue depth.
- Comparing overflow strategies (`Block`, `KeepLatest`, `DropIncoming`).
- Reading runtime metrics (`inputs_enqueued`, drops, latency, push cost, renegotiations).

Use-case guidance:
- Throughput bottlenecks: increase queue depth and inspect drop/latency behavior.
- Low-latency preference: favor latest-frame behavior in bursty streams.
- Backpressure-sensitive ingestion: prefer block policy for strict loss control.

Reference:
- [Pipeline](/getting-started/programming-model/pipeline)
- [Session](/getting-started/programming-model/session)

## Learning Process
1. Build an async run path with explicit queue and overflow settings.
2. Push a deterministic workload and drain outputs to completion.
3. Inspect metrics and input stats for latency/drop behavior.
4. Validate tuning run completion with `CHECK`, `SIGNATURE`, and `[OK]`.

## What To Observe
- `CHECK ...` lines should indicate contract and runtime validation outcomes.
- `SIGNATURE ...` output should summarize machine-parseable chapter behavior.
- `[OK] ...` indicates the chapter flow completed successfully.

## Run
```bash
./tutorial_v2_017_performance_tuning
python3 tutorials/017_performance_tuning/performance_tuning.py
```

## Source Files
- C++: `tutorials/017_performance_tuning/performance_tuning.cpp`
- Python: `tutorials/017_performance_tuning/performance_tuning.py`
