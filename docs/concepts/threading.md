---
title: Threading model
description: Which threads exist in a running pipeline, what each does, and where application code may run on them.
sidebar_position: 5
---

# Threading model

A live `Run` has more threads than is obvious from the public API. This page enumerates the thread census and explains where application code may run, and where it must not.

## Thread census

For a typical async-mode `Run`:

| Thread | Role | Owned by |
|--------|------|----------|
| GStreamer streaming threads | One per source pad path. Each pushes buffers down its branch of the pipeline. | GStreamer / `gst-launch` runtime |
| MLA dispatcher thread | Submits MLA work and reaps completions. | The framework's MLA backend |
| EV74 / CVU dispatcher threads | One per CVU-side stage. Submit kernels, poll for completion. | The framework's CVU backend |
| Pull-side waiter threads | One per output Node — blocks on the output queue, hands samples to `Run::pull()`. | The framework |
| Application threads | Whatever calls `push()`, `pull()`, `start()`, `stop()`, or any user callback. | The application |

The runtime graph (`graph::Graph`) adds:

| Thread | Role |
|--------|------|
| Stage executor threads | One per actor stage; serially handles `on_input` / `on_tick` for that stage. |
| Mailbox dispatcher thread | Routes outgoing `StageOutMsg`s to consumer mailboxes. |

## Where application code runs

User code shows up in a few places:

- **Direct calls into the framework** — `push()`, `pull()`, `build()`, `run()`, `start()`, `stop()`. These run on whatever application thread invokes them. They are safe to call concurrently from different threads as long as they target different `Run` instances; multiple threads calling `pull()` on the same `Run` is supported and fans out.
- **Sample callbacks** — `GraphRun::PullSession::on_sample()` and similar callbacks run on the runtime's pull-side waiter thread. Keep them short; long-running callbacks back-pressure the pipeline.
- **Custom `Node` implementations** — `Node::backend_fragment()` / `element_names()` etc. are called only at build time, on the application thread that called `Session::build()`. They never run during `push()` / `pull()`.
- **Custom `StageExecutor` implementations** — `on_input()` / `on_tick()` run on the executor thread the runtime owns. The runtime guarantees serial invocation per-stage, so stages may keep non-atomic per-instance state without locks.

## Locking rules

The framework's public-API objects are not thread-safe by default — `Session`, `Run`, `Tensor`, `TensorBuffer`, etc. assume one thread at a time. The exceptions:

- `Run::push()` / `pull()` are mutually exclusive with each other but **internally synchronized** — concurrent push from one thread and pull from another is supported.
- `GraphRunStats` is internally locked.
- `Run::stop()` is idempotent and may be called from any thread.

When in doubt, treat objects as single-threaded and serialize external access.

## Cancellation

`Run::stop()` is the cancellation primitive. Calling it once stops the runtime and unblocks any in-flight `pull()` (which returns `nullopt` for non-throwing variants, or raises for `pull_or_throw`). Pending `push()`es time out per the configured `push_timeout_ms`.

## Further reading

- "Runs, parallelism, telemetry" — §13 and §48 of the design deep dive.
- [`Run`](/reference/cppapi/classes/simaai-neat-run), [`GraphRun`](/reference/cppapi/classes/simaai-neat-graph-graphrun) — the runtime objects.
- [`StageExecutor`](/reference/cppapi/files/include-graph-stageexecutor-h) — runtime-graph stage threading contract.
