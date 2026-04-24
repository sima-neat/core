# 015 Tune Throughput and Queue Depth

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Advanced |
| Estimated Read Time | 15-20 minutes |
| Labels | performance, tuning, async, queues |

## Concept

Tune three knobs that control async pipeline behavior under load — queue depth, overflow policy, metrics — and read back what happened. Performance tuning only helps once the correctness baseline is stable; this chapter assumes it is.

The chapter exercises `RunOptions` at the level a production pipeline needs to control:
- `queue_depth`: how many in-flight samples the runtime accepts.
- `overflow_policy`: `Block`, `KeepLatest`, or `DropIncoming` when the queue is full.
- `enable_metrics`: turn on per-run metric collection.

**APIs introduced**
- `pyneat.RunOptions()` with `.queue_depth`, `.overflow_policy`, `.output_memory`, `.enable_metrics`.
- `pyneat.OverflowPolicy.{Block,KeepLatest,DropIncoming}` — the policy values.
- `run.try_push(sample)` — non-blocking push that returns whether the sample was accepted.
- `run.stats()` — latency/enqueue/pull counters.
- `run.input_stats()` — push-side counters (accepted, dropped, queue fullness).

**When to use this**
- Throughput bottlenecks: increase queue depth and inspect drop/latency behavior.
- Low-latency preference: favor `KeepLatest` in bursty streams.
- Backpressure-sensitive ingestion: prefer `Block` for strict loss control.

**Prerequisites**
Chapter 002 (async basics). Chapter 011 (diagnostics).

**References**
- [Pipeline](/getting-started/programming-model/pipeline)
- [Session](/getting-started/programming-model/session)

## Learning Process
1. Build an async run path with explicit queue and overflow settings.
2. Push a deterministic workload and drain outputs to completion.
3. Inspect metrics and input stats for latency/drop behavior.

## Run

**Python:**
```bash
python3 share/sima-neat/tutorials/015_tune_throughput_and_queues/tune_throughput_and_queues.py \
  --iters 32 --queue 4 --drop block
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_015_tune_throughput_and_queues \
  --iters 32 --queue 4 --drop block
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_015_tune_throughput_and_queues
./build/tutorials-standalone/tutorial_015_tune_throughput_and_queues \
  --iters 32 --queue 4 --drop block
```

To integrate this chapter's C++ source into your own project with a custom `CMakeLists.txt` (no extras folder required), see [How to Run Tutorials](/tutorials#compile-a-copy-yourself) on the landing page.

## Source Files
- C++: `tutorials/015_tune_throughput_and_queues/tune_throughput_and_queues.cpp`
- Python: `tutorials/015_tune_throughput_and_queues/tune_throughput_and_queues.py`
