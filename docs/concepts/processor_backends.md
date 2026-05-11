---
title: Processor backends
description: A65, EV74 (CVU), MLA, MLASHM, APU, TVM, M4 — what each processor does and how the framework targets them.
sidebar_position: 6
---

# Processor backends

The Modalix SoC has several processors, each suited to different parts of an inference pipeline. The Neat framework's planner picks one (or a chain) for every stage based on what the stage does. This page describes each backend and where it shows up in a typical pipeline.

## A65 — application core

Standard ARM A65 cores running Linux. The framework's main process, all application code, and most non-accelerated GStreamer elements run here.

Used for:

- The `Session` / `Run` event loop.
- File / RTSP / network I/O.
- Diagnostic taps and pull-side application code.
- Light-weight glue between accelerated stages.

## EV74 / CVU — vision compute unit

A vector-friendly DSP-style processor. The framework refers to it as **EV74** in places and **CVU** (Compute Vision Unit) in others. Used for kernels that are SIMD-shaped but not large enough to warrant the MLA — preprocess (resize, color convert, normalize), tess / detess / quant / dequant boundary kernels, fused preprocess (Generic Preproc), and BoxDecode postprocess.

EV74 work is dispatched via per-stage CVU submission threads. Kernel binaries are part of the MPK (`lib/`).

## MLA — machine learning accelerator

The Modalix MLA. The model's compiled weights and graph live here. Dispatched via an MLA submission thread that takes tessellated input from the EV74 (or directly from a quant stage) and produces tessellated output to be detessellated downstream.

MLA work has two flavors:

- **MLA inference** — the main model graph.
- **MLA prep / fused ops** — pre/post kernels compiled into the MLA when the contract allows it (the "MLA tess" column of the [dtype contract](dtype_contract)).

## MLASHM — MLA shared memory

A specialized memory region the MLA can read with the lowest latency. Buffers destined for MLA inputs are allocated from MLASHM segments when possible. The planner ensures the EV74-side preprocess writes directly into MLASHM so the MLA can consume without a transfer.

## APU — audio processing unit

Used by the audio path and by some preprocessing stages that benefit from SIMD-on-scalar work. The framework's audio Nodes (resample, codec) target the APU.

## TVM — TVM-compiled fallback

For ops that the MLA's compiler can't generate, the framework can fall back to TVM-compiled CPU kernels. Visible in the route plan as a TVM-target stage. Slower than MLA execution but guarantees coverage when an MPK contains an op the MLA backend doesn't support.

## M4 — coordinator core

A small Cortex-M4 used for low-level coordination — RPMsg between A65 and the accelerators, watchdog, hardware sequencing. Application code never runs on the M4 directly; the framework communicates with it through the OS layer.

## How the planner picks

When a Session is built, the route planner walks each stage and asks:

1. **Which processor can run this kernel?** — MLA inference goes to MLA; preprocess goes to EV74; I/O goes to A65.
2. **What's the cheapest way to get there?** — minimize transfers (the planner inserts `ConversionKind::Transfer` only when unavoidable).
3. **Can adjacent stages share segments?** — the [memory model](memory_model) dictates what's possible; the planner uses it.

The output is a `RouteGraph` where every stage carries a target processor and a segment policy. You can inspect this via `Session::describe()`.

## Further reading

- "Processor backends" — §21 and §22 of the design deep dive.
- "CVU kernels and graphs catalog" — see [CVU kernels](cvu_kernels).
- "Memory model" — see [memory model](memory_model).
- [`Session::describe()`](/reference/cppapi/classes/simaai-neat-session) — dump the route plan.
