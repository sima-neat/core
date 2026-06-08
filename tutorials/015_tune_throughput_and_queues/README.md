# 015 Tune Throughput and Queue Depth

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Advanced |
| Estimated Read Time | 15-20 minutes |
| Model | None |
| Labels | performance, tuning, async, queues |

## Concept

Tune the three `RunOptions` knobs that control async pipeline behavior under load — queue depth, overflow policy, and metrics — then read back what actually happened.

## Walkthrough

Performance tuning only helps once your correctness baseline is stable; this chapter assumes it is, and turns to the knobs that decide how an async pipeline behaves when work arrives faster than it can be processed. You will set the queue depth, choose what happens when that queue fills, push a deterministic burst of frames non-blockingly, drain the results, and read the metrics that tell you whether you dropped anything and how long each frame took.

By the end you will have a working harness for measuring an async run under backpressure: enqueue counts, drop counts, outputs pulled, average latency, and push cost. The same loop is the basis for tuning a real pipeline against the heuristics in [In Practice](#in-practice).

### Configure the run options {#step-configure-run-options}

`RunOptions` is where async behavior under load is decided. We set `queue_depth` (how many in-flight samples the runtime accepts), `overflow_policy` (what happens when that queue is full — `Block`, `KeepLatest`, or `DropIncoming`), `output_memory = Owned` (returned tensors own their data so they survive past the pull), and `enable_metrics = true` so the runtime collects the counters we read later. We then `build()` the graph in `Async` mode, which gives us a run with independent producer and consumer sides.

**C++:** The overflow policy is parsed from `--drop` into `simaai::neat::OverflowPolicy::{Block,KeepLatest,DropIncoming}`; `graph.build(input, RunMode::Async, opt)` returns the run handle.

**Python:** The policy is resolved with `getattr(pyneat.OverflowPolicy, ...)`; `graph.build([tensor], pyneat.RunMode.Async, opt)` returns the run handle.

### Push the workload and drain {#step-push-workload}

This is where the queue policy is exercised. We call `try_push(...)` in a tight loop — a non-blocking push that simply returns whether the sample was accepted, so a full queue under `DropIncoming`/`KeepLatest` shows up as rejected pushes rather than a stall. After the burst we call `close_input()` to signal no more inputs, then drain the consumer side with a `pull(...)` loop until it returns empty. Pairing `try_push` with `close_input` plus a drain loop is the canonical non-blocking async pattern.

### Read the metrics {#step-read-metrics}

With the run drained, we snapshot two counter views. `stats()` gives runtime-side numbers — inputs enqueued, inputs dropped, average latency. `input_stats()` gives push-side numbers — average push cost in microseconds and how many input renegotiations occurred. Printed together, these tell you whether your queue depth and overflow policy did what you intended: did frames drop, did latency climb, was the push path cheap.

## Run

This chapter needs no model archive. Run the **Python** and **C++ (prebuilt)** commands from the **Neat install root** (the directory that contains `share/` and `lib/`); run the **build from source** commands from the **repo root**.

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

Expected output (exact counts and timings depend on the host and policy):

```text
inputs_enqueued=32
inputs_dropped=0
outputs_pulled=32
avg_latency_ms=0.42
avg_push_us=18.0
renegotiations=0
[OK] 015_tune_throughput_and_queues
```

(The Python build prints the same keys without the trailing `[OK]` line.)

To integrate this chapter's C++ source into your own project with a custom `CMakeLists.txt` (no extras folder required), see [How to Run Tutorials](/tutorials#compile-a-copy-yourself) on the landing page.

## In Practice

Practical guidance for queue sizing, drop policies, presets, and output-lifetime safety.

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

Input shape renegotiation is automatic for dynamic inputs (the `renegotiations` counter above reports how often it happened).

### Output lifetimes (`output_memory`)

- `output_memory = Owned`: returned `Tensor` owns its data.
- `output_memory = ZeroCopy`: tensor may reference runtime buffers reused after pull.
- `output_memory = Auto`: runtime chooses zero-copy first and falls back to owned where reliability requires it.

If you need to keep tensor data beyond the current step, call `clone()` or `cpu().contiguous()`.

### Buffer pool safety

- `RunAdvancedOptions::max_input_bytes` sets a hard upper bound on input buffer allocation.
- If a larger buffer is required, the runtime fails fast with an explicit error.

Use these to protect long‑running processes from unbounded allocations when inputs change size.

## Source Files
- C++: `tutorials/015_tune_throughput_and_queues/tune_throughput_and_queues.cpp`
- Python: `tutorials/015_tune_throughput_and_queues/tune_throughput_and_queues.py`
