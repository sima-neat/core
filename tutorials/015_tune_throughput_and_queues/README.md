# 015 Tune Throughput and Queue Depth

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Advanced |
| Estimated Read Time | 15-20 minutes |
| Model | None |
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
- [Graph](/reference/programming-model/graph)
- [Graph](/reference/programming-model/graph)

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

## In Practice

Practical guidance for queue sizing, drop policies, caps changes, and output-lifetime safety.

### Queue sizing (`queue_depth`)

Heuristics:
- Start with `queue_depth = 4–16` for low‑latency pipelines.
- Increase queues if your producer is bursty or if downstream elements have variable latency (decode/MLA/postproc).
- Keep queues small if you need **freshest** frames (e.g., live camera preview).

### Overflow policy (`RunOptions::overflow_policy`)

- `Block`: safest for correctness; producer waits when queue is full.
- `DropIncoming`: keep queued work, drop incoming samples when saturated.
- `KeepLatest`: prefer freshest frames, drop the oldest queued samples.

For live feeds, `KeepLatest` usually yields the lowest end-to-end latency.

### Presets and renegotiation

Use `RunOptions::preset` to control latency/safety tradeoffs:
- `Realtime`: lowest latency, aggressive freshness behavior.
- `Balanced`: starts zero-copy when possible, runs startup probe checks, and falls back to copy mode if reliability trips.
- `Reliable`: conservative behavior and stable output ownership.

Input shape renegotiation is automatic for dynamic inputs.

### Output lifetimes (`output_memory`)

- `output_memory = Owned`: returned `Tensor` owns its data.
- `output_memory = ZeroCopy`: tensor may reference runtime buffers reused after pull.
- `output_memory = Auto`: runtime chooses zero-copy first and falls back to owned where reliability requires it.

If you need to keep tensor data beyond the current step, call `clone()` or `cpu().contiguous()`.

### Caps change decision flow

Use this guide for **push pipelines** (`Input`):

1. **Do you need fixed caps?**
   - Yes → set `InputOptions::caps_override`. Renegotiation is disabled.
   - No → leave `caps_override` empty.
2. **Do you expect size changes?**
   - Yes → no extra flag is required; dynamic dimensions are accepted by default.
   - No → set fixed dimensions in `InputOptions` and optionally use `caps_override`.

### Buffer pool safety

- `RunAdvancedOptions::max_input_bytes` sets a hard upper bound on input buffer allocation.
- If a larger buffer is required, runtime fails fast with an explicit error.

Use these to protect long‑running processes from unbounded allocations when inputs change size.

## Source Files
- C++: `tutorials/015_tune_throughput_and_queues/tune_throughput_and_queues.cpp`
- Python: `tutorials/015_tune_throughput_and_queues/tune_throughput_and_queues.py`
