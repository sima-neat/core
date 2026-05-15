---
title: The two graph systems
description: Why the framework has two `Graph` types — the builder DAG and the runtime actor graph — and when to use each.
sidebar_position: 3
---

# The two graph systems

The Neat framework has two types named `Graph`. They live in different namespaces, do different jobs, and rarely appear in the same line of code. This page explains what each is for.

## `simaai::neat::Graph` — the builder graph

Defined in `include/builder/Graph.h`. Pure STL, no GStreamer, no threads. It is a directed acyclic graph of `Node` objects, used by application code that wants to express **non-linear** pipeline shapes (fan-in, fan-out, tee, multi-input).

Use it when:

- You have multiple inputs joining at one node (e.g., two RTSP streams feeding one detector).
- You want to validate topology (cycle detection, connectivity) before runtime.
- You need to linearize a non-linear shape into a `NodeGroup` that `Session` can consume.

```cpp
sima::Graph g;
auto src1 = g.add(sima::nodes::FileInput("a.mp4"));
auto src2 = g.add(sima::nodes::FileInput("b.mp4"));
auto join = g.add(sima::nodes::Join());
g.add_edge(src1, join);
g.add_edge(src2, join);
sess.add(g.to_node_group_topo());
```

When `Session` consumes the resulting `NodeGroup`, it builds a single linear GStreamer pipeline — the builder graph is *just a topology authoring layer*. It doesn't run anything itself.

## `simaai::neat::graph::Graph` — the runtime graph

Defined in `graph/Graph.h`. This is a different beast: an **actor-style** runtime that schedules `StageExecutor` nodes via mailboxes, with named ports on every node so a stage can have multiple distinct inputs and outputs. It runs alongside (or instead of) GStreamer for use cases that don't fit a linear pipeline.

Use it when:

- You're building a multi-stream system where different streams take different routes through the same set of stages.
- You want to compose AI inference with non-GStreamer logic (custom samplers, fan-out to multiple sinks with different rates).
- You need explicit named ports (not just `in` / `out`) for a multi-input stage.

```cpp
sima::neat::graph::Graph rg;
auto detect = rg.add(std::make_shared<MyDetector>());
auto track  = rg.add(std::make_shared<MyTracker>());
rg.connect(detect, track);
sima::neat::graph::GraphSession gs(std::move(rg));
auto run = gs.build();
run.input(detect).push(sample);
auto out = run.output(track).pull();
```

This is the world `GraphSession` / `GraphRun` / `StageExecutor` live in.

## Comparison

| Aspect | `simaai::neat::Graph` (builder) | `simaai::neat::graph::Graph` (runtime) |
|--------|----------------------------------|----------------------------------------|
| Purpose | Author non-linear pipeline shapes | Run actor-style stage networks |
| Threads | None — it's a data structure | Owns scheduler threads, mailboxes, queues |
| Backend | Linearized into a GStreamer pipeline via `Session` | Custom executor; doesn't go through GStreamer |
| Element naming | `Node::backend_fragment()` produces GStreamer launch text | Stages are scheduled directly by the runtime |
| Ports | Edges are simple `from → to` | Edges carry named `from_port` / `to_port` |
| Common case | Compose Nodes for `Session::add()` | Multi-stream / multi-rate fan-out applications |

## How to tell them apart in code

- `simaai::neat::Graph` lives in the **outer** `simaai::neat` namespace. Its `add_edge()` takes two `NodeId`s; nodes hold `std::shared_ptr<Node>`.
- `simaai::neat::graph::Graph` lives in the **inner** `simaai::neat::graph` namespace. Its `connect()` takes named port strings; nodes hold `std::shared_ptr<simaai::neat::graph::Node>` (a different `Node` type).

## Further reading

- "Graphs" — §0.14, §10, §73 of the design deep dive ([Architecture](/contribute/architecture)).
- `include/builder/Graph.h`
- `graph/Graph.h`
- [`graph/GraphSession.h`](/reference/cppapi/files/include-graph-graphsession-h)
- [`graph/StageExecutor.h`](/reference/cppapi/files/include-graph-stageexecutor-h)
