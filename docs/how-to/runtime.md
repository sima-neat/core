# Runtime {#runtime}

## Build vs run

- `Session::build(...)` constructs the pipeline and returns a
  `Run` handle for push/pull control.
- `Session::run(...)` is the synchronous convenience path: it builds
  (if needed), pushes one input, and pulls one output.

## Sync vs async

- **Sync mode** (`RunMode::Sync`) is optimized for correctness and
  simplicity. You typically use `push_and_pull(...)` or `Session::run(...)`.
- **Async mode** (`RunMode::Async`) separates producer and consumer
  threads. You call `push(...)` and `pull(...)` independently.

## Push/pull API

`Run` exposes:
- `push(...)` / `try_push(...)` for inputs (`cv::Mat`, `Tensor`, or `Sample`).
- `pull(...)`, `pull_tensor(...)`, `pull_tensor_or_throw(...)` for outputs.

If you need output metadata (timestamps, stream ids), use `pull()` to get a
`Sample`. If you only need the tensor payload, use `pull_tensor()`.

## RunOptions (simple API)

Common knobs:
- `preset`: latency/safety profile (`Realtime`, `Balanced`, `Reliable`).
- `queue_depth`: runtime queue depth.
- `overflow_policy`: queue overflow behavior (`Block`, `KeepLatest`, `DropIncoming`).
- `output_memory`: output ownership policy (`Auto`, `ZeroCopy`, `Owned`).
- `on_input_drop`: callback hook for dropped input events.

## RunAdvancedOptions (expert API)

Advanced knobs are opt-in under `RunOptions::advanced`:
- `advanced.max_input_bytes`: cap input buffer growth.
- `advanced.copy_input`: force defensive input copies.

Use `Run::stats()` and `Run::diag_snapshot()` to inspect
throughput, latency, and flow counters.

See [Runtime Tuning](/how-to/runtime_tuning) for practical tuning guidance.
