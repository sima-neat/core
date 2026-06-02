---
title: Async vs sync timing model
description: How `Graph::run()` and `Run::push()/pull()` relate — when work happens, when results return.
sidebar_position: 4
---

# Async vs sync timing model

`Graph` and `Run` expose two ways to drive a pipeline:

- **Synchronous** (`Graph::run()` with no inputs): the framework opens the pipeline, runs it to completion, and returns. Control returns once the source has emitted EOS.
- **Asynchronous** (`Run::push()` / `Run::pull()`): the application owns the loop. Work happens whenever there are samples in the queue; control returns to the application between pushes / pulls.

Both modes use the same Nodes, the same plan, and the same hardware. The difference is who drives the clock.

## Synchronous mode

```cpp
sima::Graph graph;
graph.add(sima::nodes::groups::FileMp4H264In("input.mp4"));
graph.add(model.graph());
graph.add(sima::nodes::groups::Mp4FileOut("output.mp4"));
sess.run();   // blocks until EOS or error
```

This is the simplest mode. The pipeline is a self-contained job: it has its own source (a file) and sink (another file). There are no live inputs or outputs.

Use sync mode for:

- File-to-file conversion or batch inference.
- One-off validation runs.
- Reproducibility tests where you want an entire run as one transaction.

## Asynchronous mode

```cpp
sima::Graph graph;
graph.add(sima::nodes::Push("rgb"));
graph.add(model.graph());
graph.add(sima::nodes::Pull("detections"));
auto run = graph.build();
run.start();

while (have_more_inputs()) {
  run.push(make_sample(...));
  if (auto out = run.pull(0); out) {
    consume(*out);
  }
}
run.stop();
```

The `Run` is a long-lived runtime. Push samples in via the `Push` Node (input is `InputRole::Push`); pull results out via the `Pull` Node. The framework schedules work as samples arrive.

Use async mode for:

- Live video / RTSP / camera input.
- Stream processing where the application controls cadence.
- Mixing inference with non-framework code (sensor fusion, business logic).

## Push timing

`Run::push()` returns once the sample has been **enqueued** at the input boundary. It does not wait for the sample to traverse the pipeline. If the input queue is full, push blocks until space is available, governed by `RunOptions::push_timeout_ms` and the configured `OverflowPolicy`.

`OverflowPolicy::Wait` (the default) blocks; `OverflowPolicy::Drop` returns false and drops the sample; `OverflowPolicy::Replace` evicts the oldest pending sample. Pick the right policy for your latency budget.

## Pull timing

`Run::pull()` returns the next sample available at the output boundary. The signatures vary:

- `pull(timeout_ms = -1)` — block indefinitely (default), or up to `timeout_ms`.
- `pull(0)` — non-blocking; returns `nullopt` if no sample is ready.
- `pull_or_throw()` — like `pull()` but raises on timeout, for code paths that treat "no sample" as a failure.

The framework does not promise FIFO across multiple inputs streams unless you explicitly ask for it via `RunPreset` or per-stream queues — see [Tutorial 009: feed a multi-input model](/tutorials/009-feed-multi-input-model).

## Telemetry — what was the actual latency?

For both modes, the runtime can record per-sample latency via `LatencyProfiler` and per-stage timing in `RunStageStats`. After the run, `RunDiagSnapshot` exposes:

- Per-element timing (where time was spent).
- Per-pad flow stats (how many buffers crossed each pad).
- Boundary flow stats (push/pull rates).

Useful when async mode shows unexpected back-pressure — the snapshot tells you which stage is the bottleneck.

## Related types

- [`Graph::run()`](/reference/cppapi/classes/simaai-neat-graph) — synchronous entry point.
- [`Graph::build()`](/reference/cppapi/classes/simaai-neat-graph) — async entry point (returns a `Run`).
- [`Run::push()` / `pull()`](/reference/cppapi/classes/simaai-neat-run) — async drive methods.
- [`OverflowPolicy`](/reference/cppapi/files/include-pipeline-graphoptions-h) — back-pressure behavior.
- [`RunStats` / `RunDiagSnapshot`](/reference/cppapi/files/include-pipeline-run-h) — post-hoc telemetry.

## Further reading

- "Runs and parallelism" — §0.13, §12, §48, §79 of the design deep dive.
- "Async dispatch loop" — internals (§57).
