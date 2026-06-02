# 003 Build an Inference Pipeline

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Beginner |
| Estimated Read Time | 5 minutes |
| Model | None |
| Labels | graph, build, run, pipeline |

## Concept

Compose a `Graph` by hand — input node, output node, no model — and run one frame through it. See the pipeline primitives in isolation before a model is added to the picture.

A `Graph` is where you define pipeline structure by adding nodes and reusable Graph fragments in order. It is not a one-off inference call; it is a reusable runtime graph definition that can be built once and executed many times.

`build(...)` turns that definition into a runnable `Run` handle — the transition from "graph description" to "executable runtime":
- Resolves the added nodes/fragments into a concrete pipeline.
- Validates input/output contracts for the selected input type.
- Configures runtime behavior (sync vs async mode, output memory policy).
- Returns a `Run` object for push/pull calls.

**APIs introduced**
- `pyneat.Graph()` — the composition entry point.
- `pyneat.InputOptions()`, `pyneat.nodes.input(opts)`, `pyneat.nodes.output()` — the most basic node pair.
- `graph.build(tensor, pyneat.RunMode.Sync)` — materialize the pipeline.
- `run.run(tensor, timeout_ms)` — sync one-shot inference on the built pipeline.

**When to use this**
Learning the `Graph` / `Run` lifecycle without a model in the loop, or building custom pipelines from primitives.

**Prerequisites**
Chapter 001.

**References**
- [Graph](/reference/programming-model/graph)
- [Graph](/reference/programming-model/graph)

## Learning Process
1. Create a minimal `Graph` with explicit input and output nodes.
2. Build the Graph with a concrete sample input to materialize a runnable pipeline.
3. Execute one deterministic sync run to verify output contract behavior.

## Run

**Python:**
```bash
python3 share/sima-neat/tutorials/003_build_inference_pipeline/build_inference_pipeline.py \
  --width 320 --height 240
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_003_build_inference_pipeline \
  --width 320 --height 240
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_003_build_inference_pipeline
./build/tutorials-standalone/tutorial_003_build_inference_pipeline \
  --width 320 --height 240
```

To integrate this chapter's C++ source into your own project with a custom `CMakeLists.txt` (no extras folder required), see [How to Run Tutorials](/tutorials#compile-a-copy-yourself) on the landing page.

## In Practice

How `build`/`run`, execution modes, the push/pull surface, and `RunOptions` fit together once you move past a single sync call.

### Build vs run

- `Graph::build(...)` constructs the pipeline and returns a `Run` handle for push/pull control.
- `Graph::run(...)` is the synchronous convenience path: it builds (if needed), pushes one input, and pulls one output.

### Sync vs async

- **Sync mode** (`RunMode::Sync`) is optimized for correctness and simplicity. You typically use `push_and_pull(...)` or `Graph::run(...)`.
- **Async mode** (`RunMode::Async`) separates producer and consumer threads. You call `push(...)` and `pull(...)` independently — see [Run Inference Asynchronously](/tutorials/002-run-inference-async).

### Push/pull API

`Run` exposes:
- `push(...)` / `try_push(...)` for inputs (`cv::Mat`, `Tensor`, or `Sample`).
- `pull(...)`, `pull_tensor(...)`, `pull_tensor_or_throw(...)` for outputs.

If you need output metadata (timestamps, stream ids), use `pull()` to get a `Sample`. If you only need the tensor payload, use `pull_tensor()`.

### RunOptions (simple API)

Common knobs:
- `preset`: latency/safety profile (`Realtime`, `Balanced`, `Reliable`).
- `queue_depth`: runtime queue depth.
- `overflow_policy`: queue overflow behavior (`Block`, `KeepLatest`, `DropIncoming`).
- `output_memory`: output ownership policy (`Auto`, `ZeroCopy`, `Owned`).
- `on_input_drop`: callback hook for dropped input events.

For queue-depth, overflow, and metrics tuning under load, see [Tune Throughput and Queue Depth](/tutorials/015-tune-throughput-and-queues).

### RunAdvancedOptions (expert API)

Advanced knobs are opt-in under `RunOptions::advanced`:
- `advanced.max_input_bytes`: cap input buffer growth.
- `advanced.copy_input`: force defensive input copies.

Use `Run::metrics()` to inspect latency, derived throughput, input counters, and optional board PMIC power telemetry in one call. `Model::Runner` forwards the same `metrics()` / `metrics_report()` surface through its public runner adapter. For lower-level compatibility diagnostics, use `Run::stats()` and `Run::diag_snapshot()`.

To include board power, enable it in code (no environment variable required):

```cpp
simaai::neat::RunOptions run_opt;
run_opt.enable_board_power(); // default 100 ms sampling, auto-detects built-in profile
auto run = graph.build(inputs, simaai::neat::RunMode::Async, run_opt);
auto metrics = run.metrics();
```

```python
run_opt = neat.RunOptions()
run_opt.enable_board_power()  # default 100 ms sampling, auto-detects built-in profile
run = graph.build(tensor, neat.RunMode.Async, run_opt)
metrics = run.metrics()
```

`Model::build(run_opt)`, `Model::build(route_opt, run_opt)`, and `Graph::build(run_opt)` forward the same runtime options to the underlying `Run`, so one graph-level board power monitor is used instead of per-pipeline duplicate rail sampling. If you need to force a specific built-in profile, board-specific helpers remain available: `enable_modalix_som_power()`, `enable_modalix_dvt_power()`.

## Source Files
- C++: `tutorials/003_build_inference_pipeline/build_inference_pipeline.cpp`
- Python: `tutorials/003_build_inference_pipeline/build_inference_pipeline.py`
