# Runtime {#runtime}

## Build vs run

- `Graph::build(...)` constructs the pipeline and returns a
  `Run` handle for push/pull control.
- `Graph::run(...)` is the synchronous convenience path: it builds
  (if needed), pushes one input, and pulls one output.

## Sync vs async

- **Sync mode** (`RunMode::Sync`) is optimized for correctness and
  simplicity. You typically use `push_and_pull(...)` or `Graph::run(...)`.
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

Use `Run::metrics()` to inspect latency, derived throughput, input counters,
and optional board PMIC power telemetry in one call. `Model::Runner` and `GraphRun`
expose the same `metrics()` / `metrics_report()` surface through runtime-specific
adapters. For lower-level compatibility diagnostics, use `Run::stats()` and
`Run::diag_snapshot()`.

To include board power, enable it in code (no environment variable required):

```cpp
simaai::neat::RunOptions run_opt;
run_opt.enable_board_power(); // default 100 ms sampling, auto-detects built-in profile
auto run = graph.build(inputs, simaai::neat::RunMode::Async, run_opt);
auto metrics = run.metrics();
```

Python uses the same shape:

```python
run_opt = neat.RunOptions()
run_opt.enable_board_power()  # default 100 ms sampling, auto-detects built-in profile
run = graph.build(tensor, neat.RunMode.Async, run_opt)
metrics = run.metrics()
```

`Model::build(run_opt)` and `Model::build(route_opt, run_opt)` forward the same
runtime options to the underlying `Run`. For graphs, prefer
`graph::GraphRunOptions::enable_board_power()` to get one graph-level board
power monitor instead of per-pipeline duplicate rail sampling.

If you need to force a specific built-in profile, board-specific helpers remain
available:

- `enable_modalix_som_power()`
- `enable_modalix_dvt_power()`

## Verbosity presets

Framework build/run messaging is controlled with `VerboseOptions` on `GraphOptions`,
`Model::Options`, `Model::RouteOptions`, and `graph::GraphRunOptions`.

Current development default: `VerboseOptions::debug_all()`.
Call `production()` or `quiet()` explicitly when you want less output.

| Preset | Intended use |
| --- | --- |
| `VerboseOptions::quiet()` | Suppress framework progress and detail output. |
| `VerboseOptions::production()` | Show clean phase progress only. |
| `VerboseOptions::debug_plugins()` | Keep production UX, but also surface plugin and GStreamer topics. |
| `VerboseOptions::debug_all()` | Force the full verbose/detail sweep across all topics. |

See [Runtime Tuning](/how-to/runtime_tuning) for practical tuning guidance.
