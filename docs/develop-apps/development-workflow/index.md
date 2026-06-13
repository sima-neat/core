---
title: Development Workflow
description: A high-level tour of the SiMa.ai Neat development loop, from install to deployment
sidebar_position: 4
---

# Development Workflow

This page is a high-level map of how you actually use SiMa.ai Neat day to day. Each step links into a deeper page when you are ready.

## The Loop

A typical Neat development cycle looks like this:

1. **Install** — get the `sima-neat` package (and optionally the `pyneat` Python bindings) on your host or device.
2. **Try Hello Neat** — confirm the library is wired up by compiling a minimal example.
3. **Pick a compiled model** — Neat consumes a model package (`.tar.gz`, often called an MPK). You can grab one from the Model Zoo or compile your own with the SiMa.ai toolchain.
4. **Author a `Model` / `Graph` / `Run`** — load the model, compose the graph, and execute it synchronously or asynchronously.
5. **Run and inspect** — feed inputs, pull outputs, and use `GraphReport` / `MeasureReport` to verify behavior.
6. **Iterate with tutorials** — graduate from a single inference to pipelines, multi-input models, multi-stream graphs, and production-grade error handling.
7. **Deploy** — link your application against the installed Neat library on the target device.

## Core Concepts at a Glance

The Development Workflow pages break each of these down in depth. At a glance:

- [Model](/develop-apps/development-workflow/model) — load a compiled model package and expose it as a runnable unit.
- [GenAIModel](/develop-apps/development-workflow/genai-model) — the generative-model counterpart to `Model`.
- [Tensor and Sample](/develop-apps/development-workflow/core_types) — the payload and metadata envelope passed between stages.
- [Run / Inference](/develop-apps/development-workflow/overview) — execute synchronously (`run`) or asynchronously (`push` / `pull`).
- [Graph](/develop-apps/development-workflow/graph) — hybrid DAG runtime for combining model stages and custom logic.
- [Pipeline](/develop-apps/development-workflow/pipeline) — the runtime view of a built graph.
- [Node](/develop-apps/development-workflow/node) — the atomic building block of a graph.

If you only learn one page first, start with the [Run / Inference overview](/develop-apps/development-workflow/overview) — it ties `Model`, `Graph`, and `Run` together end to end.

## Where to go next

Step-by-step entry points for new users:

- [Neat Development Environment](/getting-started/dev-environment/) — install the SDK, pair a DevKit, and run on hardware with `dk`.
- [Build](/develop-apps/contribute/build) — build Neat from source with `build.sh` (contributor workflow).
- [Hello Neat!](/develop-apps/hello-neat/minimal) — minimal CMake application that links against the installed library.
- [Tutorials](/tutorials) — guided chapters that scale from "first model" to "production pipeline".

Reference material for when you need depth:

- [Run / Inference](/develop-apps/development-workflow/overview) — concept-by-concept breakdown of `Model`, `Run`, `Node`, `Pipeline`, `Graph`, and I/O.
- [C++ Reference](/reference/cppapi) — full API surface for the installed headers.
- [Python Reference](/reference/pythonapi) — `pyneat` bindings reference.

## What you write vs. what Neat provides

Neat owns the runtime: model loading, validation, pipeline construction, scheduling, teardown, and diagnostics. You own the application code that wires inputs to outputs and reacts to results. The boundary is the public API in `include/`, which is treated as **stable** — you can upgrade Neat without rewriting your application.

If you only remember three lines of code from Neat, remember these:

```cpp
simaai::neat::Model      model(mpk_path);
simaai::neat::TensorList outputs = model.run(input_tensors, /*timeout_ms=*/2000);
simaai::neat::Mapping    view = outputs[0].map_read();  // inspect the output bytes
```

Everything else in this documentation — pipelines, graphs, async queues, multi-stream — is a controlled expansion of that core three-line story.
