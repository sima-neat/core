# Neat framework

The **Neat framework** is the C++/Python library and runtime that runs models on the SiMa Modalix chip. It loads compiled `.tar.gz` model archives, assembles them into deterministic GStreamer pipelines, and executes inference at frame rate on heterogeneous compute (A65 host CPU + EV74 vector cores + MLA accelerator + hardware codec).

This site is the **reference documentation** for the framework's public API. For design rationale and worked examples, see the [Architecture deep dive](/contribute/architecture), which mirrors the architect-level documentation.

## Neat has two parts

People at SiMa use the word "Neat" to mean two related but distinct things. Both exist; both are called Neat; they're complementary:

- **Neat framework** — the library you `#include`, the runtime that loads models, the GStreamer-based pipelines that execute inference. **The subject of this site.**
- **Neat environment** — a Docker container with shared filesystem mounts to the DevKit, integrated agent skills, and a single-point-of-contact workflow that lets developers (and AI agents) compile on a fast x86 host and test on real Modalix hardware. The environment is what enables the *agentic workflow* — but it works because the framework underneath is built for agents.

The framework runs on the chip, in production, on every Modalix everywhere. The environment is a developer convenience that exists alongside. **Production ships the framework; developers also use the environment.**

## What the framework solves

Modern AI inference on the edge has hard constraints: frame-rate throughput on small power budgets, heterogeneous compute (CPU + vector + accelerator), zero-copy memory across processors, deterministic latency, and the ability to integrate cleanly with media pipelines (cameras, codecs, RTSP). Solving all of these by hand for every application is a year of work. The Neat framework absorbs that work into a single typed C++/Python API.

A typical Neat-framework application looks like this:

```cpp
sima::Model model("/models/yolov8.tar.gz");
sima::Graph graph;
graph.add(sima::nodes::groups::RtspDecodedInput({.url = "rtsp://camera/stream"}));
graph.add(model.graph());
graph.add(sima::Output{});

auto run = graph.build(sima::RunMode::Async);
while (running) {
  auto sample = run.pull(/*timeout_ms=*/100);
  if (sample) handle_detection(*sample);
}
```

A dozen lines from "RTSP camera → real-time YOLO detection running on Modalix accelerator." Behind that simplicity is a lot of carefully-designed machinery — that's what this reference documents.

## The eight main concepts

The framework's public surface is a small set of primary concepts. They form a clean progression from "the file on disk" to "the data that flows between processors at runtime":

| # | Concept | Purpose | Reference |
|---|---|---|---|
| 1 | **Model archive** | Sealed `.tar.gz` file from the compiler — kernels, weights, configs, and the MPK contract. | [`Model`](/reference/cppapi/classes/simaai-neat-model), [MPK contract](/concepts/mpk_contract) |
| 2 | **Model** | Loaded form of a model archive; the simplified entry point. | [`Model`](/reference/cppapi/classes/simaai-neat-model) |
| 3 | **Tensor** | Typed data unit that flows between stages. | [`Tensor`](/reference/cppapi/structs/simaai-neat-tensor), [Memory model](/concepts/memory_model), [dtype contract](/concepts/dtype_contract) |
| 4 | **Nodes** | Smallest building blocks; each wraps one (or a few) GStreamer elements. | [Node APIs](/reference/cppapi/groups/nodes), [GStreamer underneath](/concepts/gstreamer_layer) |
| 5 | **Reusable Graph fragments** | Pre-made `Graph` fragments capturing common patterns. | [Reusable Graph fragments](/reference/cppapi/groups/nodes-groups) |
| 6 | **Graph** | Assembly stage that turns Nodes, Models, and reusable Graph fragments into a runnable pipeline. | [`Graph`](/reference/cppapi/classes/simaai-neat-graph), [`GraphOptions`](/reference/cppapi/structs/simaai-neat-graphoptions) |
| 7 | **Run** | Live, running pipeline produced by `Graph::build()`. | [`Run`](/reference/cppapi/classes/simaai-neat-run), [Async vs sync timing](/concepts/timing_model), [Threading model](/concepts/threading) |

## Where to read what

| What you want | Where to look |
|---|---|
| Conceptual overview, architecture, design rationale | [Architecture deep dive](/contribute/architecture) — the architect-level documentation, surfaced as navigable site pages |
| C++ public API — every class, method, field | [C++ API Reference](/reference/cppapi/) — generated from Doxygen comments |
| Python public API — pyneat module surface | [Python API Reference](/reference/pythonapi/) — generated from nanobind bindings |
| Conceptual deep-dives (dtype contract, memory model, error codes, etc.) | [dtype contract](/concepts/dtype_contract), [memory model](/concepts/memory_model), [graphs](/concepts/graphs), [timing model](/concepts/timing_model), [threading](/concepts/threading), [processor backends](/concepts/processor_backends), [GStreamer underneath](/concepts/gstreamer_layer), [CVU kernels](/concepts/cvu_kernels), [MPK contract](/concepts/mpk_contract), [error codes](/concepts/error_codes), [build options](/concepts/build_options), [agentic workflow](/concepts/agentic_workflow) |
| Glossary, environment variables, scripts, error format | [Glossary](/reference/glossary), [env vars](/reference/env_vars), [scripts inventory](/reference/scripts), [plugin error format](/reference/error_format) |
| Onboarding, build, minimal example | [Installation](/getting-started/installation), [Build](/getting-started/build), [Minimal example](/getting-started/minimal_example) |
| How to do specific things (debugging, runtime tuning, plugin failures) | [How-to: runtime tuning](/how-to/runtime_tuning), [How-to: diagnostics](/how-to/diagnostics), [How-to: plugin failures](/how-to/plugin_failures) |
| Coding standards, MPK contract, contribution policy | [Coding standard](/contribute/coding_standard), [MPK contract](/contribute/mpk_contract), [Architecture](/contribute/architecture) |

## Why the framework is built for agents

A consequence of how the framework was designed: it's an exceptionally good substrate for AI code generation. Every framework error produces a structured `GraphReport` (machine-readable error code + reproducer command). Every public symbol has stable naming. Validation runs before any pipeline starts. Pipelines are serializable JSON. The public API is ABI-stable. Errors fail fast with actionable messages instead of silent fallbacks.

These properties weren't picked for AI agents specifically — they came from "make the framework deterministic, debuggable, and never hang the process" (see [`docs/contribute/architecture.md`](/contribute/architecture)). But they happen to be exactly what an AI agent needs to write code that converges quickly. That's why the **agentic workflow** in the Neat environment delivers what it does — the framework is the substrate, the environment is the workshop.

The full story of why the framework is good for agents — fifteen specific design properties, each tied to architect-doc rationale — is in the [Architecture deep dive](/contribute/architecture).

## Conventions in this reference

- All public types live under the `simaai::neat` namespace.
- Nodes live under `simaai::neat::nodes::{common,io,rtp,sima,groups}`.
- Headers are organized by responsibility: `model/`, `pipeline/`, `builder/`, `nodes/`,
  `contracts/`, `policy/`, `gst/`, plus convenience umbrellas under `neat/`. The
  source-tree `include/graph/` runtime substrate is intentionally excluded from
  the public reference; application code should use `simaai::neat::Graph` / `Run`.
- Every public header has a file-level `@file` / `@ingroup` / `@brief` block. Group definitions live in `docs/doxygen/groups.dox`.
- Errors are returned as `NeatError` exceptions carrying a structured `GraphReport`. See the [Error code catalog](/concepts/error_codes) for the full taxonomy.

The rest of this site is the API itself. Start by browsing [classes](/reference/cppapi/classes), or jump to the headline types: [`Model`](/reference/cppapi/classes/simaai-neat-model), [`Graph`](/reference/cppapi/classes/simaai-neat-graph), [`Run`](/reference/cppapi/classes/simaai-neat-run), [`Tensor`](/reference/cppapi/structs/simaai-neat-tensor), [Node APIs](/reference/cppapi/groups/nodes).
