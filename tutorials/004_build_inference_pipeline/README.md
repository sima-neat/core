# 004 Build an Inference Pipeline

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Beginner |
| Estimated Read Time | 5 minutes |
| Model | None |
| Labels | graph, build, run, pipeline |

## Concept

Compose a `Graph` by hand — input node, output node, no model — and run one frame through it. See the pipeline primitives in isolation before a model is added to the picture.

## Walkthrough

Chapter 001 ran a model in three lines. That convenience hides a two-part lifecycle that every non-trivial Neat program uses directly: you first **describe** a pipeline as a `Graph`, then **build** that description into a runnable `Run`. This chapter makes that lifecycle visible by composing the smallest possible pipeline — one input node wired to one output node, no model in between — and pushing a single frame through it.

The payoff is conceptual: a `Graph` is a *reusable definition* you build once and execute many times, not a one-off call. By the end you will have created a graph, turned it into a runnable pipeline, and read back the rank of the output tensor — proving the frame made it through.

### Describe the input {#step-configure-input}

Before wiring up nodes, declare what a frame looks like. `InputOptions` is that contract: pixel `format`, `width`/`height`, channel `depth`, and whether the runtime timestamps each buffer. The input node built from these options validates incoming frames against the shape the pipeline expects.

**C++:** C++ additionally sets `is_live = false` to mark this as a non-live (file/tensor) source.

### Compose the graph {#step-compose-graph}

Now build the structure. A fresh `Graph` is an empty composition surface, and `add()` appends nodes in order. We add exactly two — an input node (configured above) and a bare output node. That is the entire topology: frames enter at the input and leave at the output, with nothing in between. This is the seam where, in later chapters, a model or preprocessing stage slots in.

**C++:** Nodes come from `simaai::neat::nodes::Input(...)` and `nodes::Output()`.

**Python:** Nodes come from `pyneat.nodes.input(...)` and `pyneat.nodes.output()`.

### Build the pipeline {#step-build-pipeline}

`build()` is the transition from *description* to *executable*. It resolves the added nodes into a concrete pipeline, validates the input/output contracts against a real sample, selects the execution mode (`Sync` here, for deterministic one-at-a-time behavior), and returns a `Run` handle. We pass a representative frame so `build()` can lock in the negotiated tensor shapes.

**C++:** The sample frame is a `cv::Mat`, and `run_opt.output_memory = Owned` asks the runtime to return owned output buffers.

**Python:** We first materialize the frame as a `Tensor` from a NumPy array with `Tensor.from_numpy(...)`, then build with it.

### Run a frame and read the result {#step-run-frame}

With a `Run` in hand, `run()` pushes one frame and pulls one result synchronously. Because there's no model, the output mirrors the input contract — so reading the tensor's *rank* is enough to confirm a frame completed the round trip. In real pipelines this same `run()`/push/pull surface is how you drive inference.

**C++:** `run()` returns a `TensorList`; read `sample.front().shape.size()`.

**Python:** `run()` with tensor inputs returns a `TensorList`; read `len(outputs[0].shape)`.

## Run

Run it and you should see the rank of the output tensor printed to stdout. Run the **Python** and **C++ (prebuilt)** commands from the **Neat install root** (the directory that contains `share/` and `lib/`); run the **build from source** commands from the **repo root**. This chapter needs no model archive.

**Python:**
```bash
python3 share/sima-neat/tutorials/004_build_inference_pipeline/build_inference_pipeline.py \
  --width 320 --height 240
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_004_build_inference_pipeline \
  --width 320 --height 240
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_004_build_inference_pipeline
./build/tutorials-standalone/tutorial_004_build_inference_pipeline \
  --width 320 --height 240
```

Expected output:

```text
tensor_rank=3
[OK] 004_build_inference_pipeline
```

(The Python build prints `output_rank=...`.) To integrate this chapter's C++ source into your own project with a custom `CMakeLists.txt` (no extras folder required), see [How to Run Tutorials](/tutorials#compile-a-copy-yourself) on the landing page.

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

For queue-depth, overflow, and metrics tuning under load, see [Tune Throughput and Queue Depth](/tutorials/016-tune-throughput-and-queues).

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
- C++: `tutorials/004_build_inference_pipeline/build_inference_pipeline.cpp`
- Python: `tutorials/004_build_inference_pipeline/build_inference_pipeline.py`
