---
title: Processor backends
description: A65, EV74 (CVU), MLA, MLASHM, APU, TVM, M4 — what each processor does and how the framework targets them.
sidebar_position: 2
slug: /develop-apps/advanced-concepts/processor_backends
---

# Processor backends

The Modalix SoC has several processors, each suited to different parts of an inference pipeline. The Neat framework's planner picks one (or a chain) for every stage based on what the stage does. This page describes each backend and where it shows up in a typical pipeline.

## A65 — application core

Standard ARM A65 cores running Linux. The framework's main process, all application code, and most non-accelerated GStreamer elements run here.

Used for:

- The `Graph` / `Run` event loop.
- File / RTSP / network I/O.
- Diagnostic taps and pull-side application code.
- Light-weight glue between accelerated stages.

## EV74 / CVU — vision compute unit

A vector-friendly DSP-style processor. The framework refers to it as **EV74** in places and **CVU** (Compute Vision Unit) in others. Used for kernels that are SIMD-shaped but not large enough to warrant the MLA — preprocess (resize, color convert, normalize), tess / detess / quant / dequant boundary kernels, fused preprocess (Generic Preproc), and BoxDecode postprocess.

EV74 work is dispatched via per-stage CVU submission threads. Kernel binaries are part of the model archive (`lib/`).

## MLA — machine learning accelerator

The Modalix MLA runs compiler-produced QMLA model stages. In a Neat graph,
`ProcessMLA` is the single MLA execution path:

```text
Neat model route
  -> ProcessMLA
  -> direct MLA backend
  -> /dev/mla
  -> terminal job completion
```

ProcessMLA loads the compiled model, imports dma-buf-backed inputs and outputs,
binds their compiler-defined byte ranges, and submits the resulting immutable
job to the kernel. The kernel owns hardware dispatch, priority arbitration,
fault handling, and completion. The graph path does not select Dispatcher or
MLA-RT, and it does not switch between synchronous and asynchronous MLA
engines.

MLA work still has two model-level flavors:

- **MLA inference** — the main model graph.
- **MLA prep or fused operations** — pre/post operations compiled into the MLA
  when the MPK contract allows it. See the "MLA tess" column of the
  [dtype contract](/develop-apps/advanced-concepts/dtype_contract).

### MLA scheduling intent

Independent graphs can state one of three intents:

| Intent | Use |
| --- | --- |
| `Foreground` | Latency-sensitive work that should run before lower-priority contexts at the next compiled-job boundary. |
| `Normal` | Default application inference. |
| `Background` | Throughput-oriented work, such as a long-running language-model session. |

The resolved intent configures the ProcessMLA kernel context once, before it
accepts work. MLA commands are non-preemptive: a higher-priority context may
run between compiled jobs, not in the middle of a job already executing.

`Foreground` requires `CAP_SYS_NICE` for the application service or container.
Missing privilege is a setup error; Neat does not silently fall back to
`Normal`. Numeric kernel bands, per-job priority, fairness policy, and queue
depth remain private implementation details.

### DMS0 — direct MLA buffers

A graph whose first effective consumer is the MLA uses coherent,
dma-buf-exportable DMS0 storage. CPU code may populate the coherent mapping
before the buffer is transferred to device ownership. ProcessMLA imports that
same allocation; it does not hide a private input copy.

When EV74 preprocessing runs first, the planner preserves the memory contract
required by that stage and the MLA handoff. Application code normally does not
choose a segment directly. See the
[memory model](/develop-apps/advanced-concepts/memory_model) for explicit
tensor placement and ownership rules.

## APU — audio processing unit

Used by the audio path and by some preprocessing stages that benefit from SIMD-on-scalar work. The framework's audio Nodes (resample, codec) target the APU.

## TVM — TVM-compiled fallback

For ops that the MLA's compiler can't generate, the framework can fall back to TVM-compiled CPU kernels. Visible in the route plan as a TVM-target stage. Slower than MLA execution but guarantees coverage when an MPK contract describes an op the MLA backend doesn't support.

## M4 — coordinator core

A small Cortex-M4 used for low-level coordination — RPMsg between A65 and the accelerators, watchdog, hardware sequencing. Application code never runs on the M4 directly; the framework communicates with it through the OS layer.

## How the planner picks

When a Graph is built, the route planner walks each stage and asks:

1. **Which processor can run this kernel?** — MLA inference goes to MLA; preprocess goes to EV74; I/O goes to A65.
2. **What's the cheapest way to get there?** — minimize transfers (the planner inserts `ConversionKind::Transfer` only when unavoidable).
3. **Can adjacent stages share segments?** — the [memory model](/develop-apps/advanced-concepts/memory_model) dictates what's possible; the planner uses it.

The output is a `RouteGraph` where every stage carries a target processor and a segment policy. You can inspect this via `Graph::describe()`.

ProcessMLA intentionally exposes no application control for kernel queue depth,
numeric priority, per-frame priority, or a second MLA transport. Those details
would duplicate the kernel scheduler and make otherwise equivalent graphs
behave differently.

## Further reading

- "Processor backends" — §21 and §22 of the design deep dive.
- "CVU kernels and graphs catalog" — see [CVU kernels](/develop-apps/advanced-concepts/cvu_kernels).
- "Memory model" — see [memory model](/develop-apps/advanced-concepts/memory_model).
- [`Graph::describe()`](/reference/cppapi/classes/simaai-neat-graph) — dump the route plan.
