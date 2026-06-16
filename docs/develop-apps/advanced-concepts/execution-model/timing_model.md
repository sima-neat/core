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

  if (auto out = run.pull("classes", /*timeout_ms=*/0)) {
    consume(*out);
  }
}

run.close_input();
while (auto out = run.pull("classes", /*timeout_ms=*/1000)) {
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

`Run::push(...)` returns after the input is accepted at the graph boundary. It does not wait for that input to traverse every node.

When the input queue is full:

- `OverflowPolicy::Block` applies backpressure to the producer;
- `OverflowPolicy::DropIncoming` rejects the new input;
- `OverflowPolicy::KeepLatest` drops older queued input so the live path stays fresh;
- `try_push(...)` returns `false` instead of blocking.

Choose the policy that matches the source. File and batch jobs usually want `Block`. Live streams usually want a freshness policy.

## Pull timing

`Run::pull(...)` returns the next available `Sample` from an output boundary.

Use the convenience overload when timeout and EOS can share the same “no sample” path:

```cpp
if (auto sample = run.pull("classes", /*timeout_ms=*/1000)) {
  consume(*sample);
}
```

Use the structured status overload when timeout, closed, and error must be handled differently:

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
- optional power telemetry when the run has power monitoring enabled.

Keep setup, file downloads, per-frame logging, and report export outside the measured hot loop unless the question is end-to-end application cost.

## Further reading

- [Run a Graph](/develop-apps/development-workflow/pipeline) — lifecycle, options, backpressure, measurement, and throughput.
- [`Graph::run()`](/reference/cppapi/classes/simaai-neat-graph) — one-shot and source-owned entry point.
- [`Graph::build()`](/reference/cppapi/classes/simaai-neat-graph) — reusable execution entry point.
- [`Run`](/reference/cppapi/classes/simaai-neat-run) — push, pull, close, drain, stop, and measurement.
