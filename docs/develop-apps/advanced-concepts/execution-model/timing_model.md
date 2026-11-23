---
title: Async vs sync timing model
description: How `Graph.run(...)` and `Graph.build(...)` / `Run` relate — when work happens, when results return.
sidebar_position: 1
slug: /develop-apps/advanced-concepts/timing_model
---

# Async vs sync timing model

`Graph` and `Run` expose two ways to drive work:

- **One-shot execution**: `Graph.run(input, ...)` pushes input and waits for the output in one call. `Graph.run()` with no input runs a source-owned graph until EOS.
- **Reusable execution**: `Graph.build(...)` returns a live `Run`. Your application pushes input, pulls output, closes input, drains, measures, and stops the run.

Both modes use the same `Graph` plan and the same hardware. The difference is who owns the loop.

## One-shot mode

Use one-shot mode when you have one input and want the shortest correct path.

```cpp
simaai::neat::Graph graph("classifier");
graph.add(simaai::neat::nodes::Input("image"));
graph.add(model);
graph.add(simaai::neat::nodes::Output("classes"));

simaai::neat::TensorList out = graph.run(
    simaai::neat::TensorList{image_tensor});
```

For a source-owned graph, call `Graph.run()` with no input:

```cpp
simaai::neat::Graph graph("file_job");
graph.add(source_fragment);
graph.add(model);
graph.add(sink_fragment);

graph.run();  // Blocks until EOS or error.
```

Use one-shot mode for:

- smoke tests;
- short validation runs;
- source-owned jobs that should run to completion;
- code that should not manage a long-lived runtime handle.

## Reusable Run mode

Use `Graph.build(...)` when your application owns the loop.

```cpp
auto run = graph.build();

while (have_more_inputs()) {
  run.push("image", simaai::neat::TensorList{next_tensor()});

  if (auto out = run.pull("classes", std::chrono::milliseconds(0))) {
    consume(*out);
  }
}

run.close_input();
while (auto out = run.pull("classes", std::chrono::milliseconds(1000))) {
  consume(*out);
}
run.close();
```

Use reusable mode for:

- live video, RTSP, or camera input;
- stream processing where the app controls cadence;
- multi-input or multi-output graphs;
- throughput testing;
- measurement, export, drain, and stop control.

## Push timing

`push()` waits for input space under every overflow policy. If a graph input times out, it returns `false`. Drain accepted work before retrying an individual sample.

Both `push()` and `try_push()` can accept the first items in a bundle or Python list of samples before returning `false`. Submit one sample per call when you need to retry.

`try_push()` returns immediately when the input queue is full.

- `Block` returns `false`. The caller can retry.
- `DropIncoming` returns `false` and counts the incoming sample as dropped.
- `KeepLatest` replaces the oldest pending sample, counts one drop, and returns `true`.

## Inline queues

`RunOptions::overflow_policy` controls the queue at a `Run` input. To control a queue between graph nodes, add a configured `Queue`:

```cpp
simaai::neat::QueueOptions queue;
queue.max_buffers = 1;
queue.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
graph.add(simaai::neat::nodes::Queue(queue));
```

```python
queue = pyneat.QueueOptions()
queue.max_buffers = 1
queue.overflow_policy = pyneat.OverflowPolicy.KeepLatest
graph.add(pyneat.nodes.queue(queue))
```

Leave `max_buffers` unset to use the default queue limits. Setting it to a positive value makes the buffer count the only capacity limit. `Block` waits for space, `KeepLatest` drops the oldest waiting buffer, and `DropIncoming` drops the new buffer.

## Pull timing

`Run::pull(timeout)` returns the next available output of an output boundary as a `Result<TensorList>`. The structured status overload below fills a `Sample` instead.

Your application can pause submissions and resume the same `Run`; output silence alone does not fail it. A finite `pull(...)` timeout ends that wait without stopping the Run. `RunOptions::input_timeout_ms` controls output waits for input-mode `Graph::run()` and supported seeded build preflight. `Run::run()` and `Model::run()` use their per-call timeout instead; `-1` means unlimited, as a negative timeout does in `pull(timeout)`. Applications that need stall detection should set deadlines for pending work.

Test the `Result` when timeout and EOS can share the same “no output” path; its `status()` tells them apart when needed:

```cpp
if (auto out = run.pull("classes", std::chrono::milliseconds(1000))) {
  consume(*out);
}
```

Use the structured status overload when timeout, closed, and error must be handled differently and you need the `Sample` itself:

```cpp
simaai::neat::Sample sample;
simaai::neat::PullError error;

switch (run.pull("classes", /*timeout_ms=*/1000, sample, &error)) {
case simaai::neat::PullStatus::Ok:
  consume(sample);
  break;
case simaai::neat::PullStatus::Timeout:
  break;
case simaai::neat::PullStatus::Closed:
  break;
case simaai::neat::PullStatus::Error:
  throw std::runtime_error(error.message);
}
```

No hidden fallback: a multi-output graph needs a named `pull("output", ...)` unless there is one unambiguous output.

## Telemetry: measure the loop you own

Use `Run::start_measurement(...)` around the workload your app owns. The returned `MeasureReport` is the public timing surface and includes:

- end-to-end push-to-output latency and throughput;
- runtime counters for pushed, pulled, and dropped samples;
- plugin/kernel and edge timing when requested in `MeasureOptions`;
- power from the deprecated `RunOptions::power_monitor`, when enabled.

Record board power, temperatures, CPU usage, and memory over the same window with a
[`TelemetryMonitor`](/develop-apps/development-workflow/model#record-board-telemetry).

Keep setup, file downloads, per-frame logging, and report export outside the measured hot loop unless the question is end-to-end application cost.

## Further reading

- [Run a Graph](/develop-apps/development-workflow/pipeline) — lifecycle, options, backpressure, measurement, and throughput.
- [`Graph::run()`](/reference/cppapi/classes/simaai-neat-graph) — one-shot and source-owned entry point.
- [`Graph::build()`](/reference/cppapi/classes/simaai-neat-graph) — reusable execution entry point.
- [`Run`](/reference/cppapi/classes/simaai-neat-run) — push, pull, close, drain, stop, and measurement.
