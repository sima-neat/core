---
title: Threading model
description: Which threads exist in a running graph, what each does, and where application code may run on them.
sidebar_position: 2
slug: /develop-apps/advanced-concepts/threading
---

# Threading model

A live `Run` does work on framework and runtime threads. Your app does not need to manage those threads, but it does need to keep its own push, pull, callback, and shutdown code disciplined.

## Thread census

For a typical `Run`:

| Thread | Role | Owned by |
| --- | --- | --- |
| GStreamer streaming threads | Move buffers through source, transform, and sink paths. | GStreamer runtime |
| MLA dispatcher threads | Submit MLA work and reap completions. | Neat runtime |
| EV74 / CVU dispatcher threads | Submit CVU-side kernels and poll for completion. | Neat runtime |
| Pull-side waiters and bus watchers | Move output samples into public queues and report runtime errors. | Neat runtime |
| Application threads | Call `Graph.build(...)`, `Graph.run(...)`, `Run.push(...)`, `Run.pull(...)`, `Run.stop()`, or callbacks. | Your application |

Graph fragments, source nodes, and sink nodes can add runtime work, but the rule stays the same: the public `Run` is the handle your app controls.

## Where application code runs

User code appears in these places:

- **Direct API calls**: `build(...)`, `run(...)`, `push(...)`, `try_push(...)`, `pull(...)`, `close_input()`, `stop()`, and `close()` run on the application thread that invokes them.
- **Drop callbacks**: `RunOptions.on_input_drop` runs on the pushing path. Keep it short; count and return.
- **Tensor callbacks**: `Graph::set_tensor_callback(...)` is C++ callback-style consumption. Keep callbacks short because long work backpressures the runtime path.
- **Custom node description**: public node construction and graph composition happen at build time, on the application thread.

Do not hide slow work in callbacks. If a callback needs to do heavy work, hand the sample to your own queue and return.

## Locking rules

Treat `Graph`, `Run`, `Tensor`, and `Sample` as single-owner objects unless the API says otherwise.

Supported pattern:

- one producer thread calls `push(...)` or `try_push(...)`;
- one consumer thread calls `pull(...)`;
- another coordinator thread may call `stop()` during shutdown.

Risky pattern:

- several producer threads push to the same `Run` without a lock;
- several consumer threads pull from the same output without a clear ownership rule;
- a callback calls back into the same `Run` and waits for more work.

If you need multiple producers, serialize access before calling `Run.push(...)`. If you need multiple consumers, fan out after one thread pulls from Neat.

## Shutdown and cancellation

Use the shutdown primitive that matches intent:

| Intent | Use |
| --- | --- |
| Finish queued work after the last input | `run.close_input()`, then pull until no more output arrives or `PullStatus::Closed` is returned. |
| Stop now and unblock waiting work | `run.stop()` |
| Release runtime resources | `run.close()` or let the `Run` object leave scope |

`stop()` is the cancellation path. After cancellation, in-flight pulls unblock and pushes should stop. Do not keep pushing into a run that is closing.

## Throughput thread shape

For app-pushed live or high-throughput graphs, start with two application threads:

1. A producer thread reads input, stamps `stream_id` / `frame_id`, and calls `try_push(...)` or `push(...)` according to the selected `OverflowPolicy`.
2. A consumer thread pulls continuously and releases or copies outputs before they pin runtime-backed buffers.

Add more application threads around your own queues, not around the same `Run` object. Make the hot loop boring. Boring is fast.

## Further reading

- [Run a Graph](/develop-apps/development-workflow/pipeline) — runtime lifecycle, throughput, measurement, and backpressure.
- [`Run`](/reference/cppapi/classes/simaai-neat-run) — the public runtime object.
- [Async vs sync timing model](/develop-apps/advanced-concepts/timing_model) — when work happens and when calls return.
