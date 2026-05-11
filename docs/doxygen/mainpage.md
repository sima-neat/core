# Neat framework

The **Neat framework** is the C++/Python library and runtime that runs models on the SiMa Modalix chip. It loads compiled model packs (MPKs), assembles them into deterministic GStreamer pipelines, and executes inference at frame rate on heterogeneous compute (A65 host CPU + EV74 vector cores + MLA accelerator + hardware codec).

This site is the **reference documentation** for the framework's public API. For design rationale and worked examples, see the [Design / Deep Dive](../design) section, which mirrors the architect-level documentation.

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
sima::Session sess;
sess.add(sima::nodes::groups::RtspDecodedInput({.url = "rtsp://camera/stream"}));
sess.add(model.session());
sess.add(sima::Output{});

auto run = sess.build(sima::RunMode::Async);
while (running) {
  auto sample = run.pull(/*timeout_ms=*/100);
  if (sample) handle_detection(*sample);
}
```

A dozen lines from "RTSP camera → real-time YOLO detection running on Modalix accelerator." Behind that simplicity is a lot of carefully-designed machinery — that's what this reference documents.

## The eight main concepts

The framework's surface is **eight primary concepts**. They form a clean progression from "the file on disk" to "the data that flows between processors at runtime":

| # | Concept | Purpose | Reference |
|---|---|---|---|
| 1 | **MPK** | Sealed model file on disk (`.tar.gz` from the compiler — kernels + weights + manifest). | [`MpKLoader`](../reference/cppapi/classes/simaai-neat-mpk-mpkloader), [`MpKManifest`](../reference/cppapi/structs/simaai-neat-mpk-mpkmanifest), [MPK contract](../concepts/mpk_contract) |
| 2 | **Model** | Loaded form of an MPK; the simplified entry point. | [`Model`](../reference/cppapi/classes/simaai-neat-model) |
| 3 | **Tensor** | Typed data unit that flows between stages. | [`Tensor`](../reference/cppapi/classes/simaai-neat-tensor), [Memory model](../concepts/memory_model), [dtype contract](../concepts/dtype_contract) |
| 4 | **Nodes** | Smallest building blocks; each wraps one (or a few) GStreamer elements. | [`Node`](../reference/cppapi/classes/simaai-neat-node), [GStreamer underneath](../concepts/gstreamer_layer) |
| 5 | **NodeGroups** | Pre-made bundles of Nodes capturing common patterns. | [`NodeGroup`](../reference/cppapi/classes/simaai-neat-nodegroup) |
| 6 | **Session** | Assembly stage that turns Nodes into a runnable pipeline. | [`Session`](../reference/cppapi/classes/simaai-neat-session), [`SessionOptions`](../reference/cppapi/structs/simaai-neat-sessionoptions) |
| 7 | **Run** | Live, running pipeline produced by `Session::build()`. | [`Run`](../reference/cppapi/classes/simaai-neat-run), [Async vs sync timing](../concepts/timing_model), [Threading model](../concepts/threading) |
| 8 | **Graph** | Composition of pipelines for non-linear shapes. | [Builder Graph](../reference/cppapi/classes/simaai-neat-graph), [Runtime Graph](../reference/cppapi/classes/simaai-neat-graph-graph), [Two graph systems](../concepts/graphs) |

## Where to read what

| What you want | Where to look |
|---|---|
| Conceptual overview, architecture, design rationale | [Design / Deep Dive](../design) — the architect-level documentation, surfaced as navigable site pages |
| C++ public API — every class, method, field | [C++ API Reference](../reference/cppapi) — generated from Doxygen comments |
| Python public API — pyneat module surface | [Python API Reference](../reference/pythonapi) — generated from nanobind bindings |
| Conceptual deep-dives (dtype contract, memory model, error codes, etc.) | [Concepts](../concepts) — [dtype contract](../concepts/dtype_contract), [memory model](../concepts/memory_model), [graphs](../concepts/graphs), [timing model](../concepts/timing_model), [threading](../concepts/threading), [processor backends](../concepts/processor_backends), [GStreamer underneath](../concepts/gstreamer_layer), [CVU kernels](../concepts/cvu_kernels), [MPK contract](../concepts/mpk_contract), [error codes](../concepts/error_codes), [build options](../concepts/build_options), [agentic workflow](../concepts/agentic_workflow) |
| Glossary, environment variables, scripts, error format | [Glossary](../reference/glossary), [env vars](../reference/env_vars), [scripts inventory](../reference/scripts), [plugin error format](../reference/error_format) |
| Onboarding, build, minimal example, common pitfalls | [Install](../getting-started/install), [Build](../getting-started/build), [Minimal example](../getting-started/minimal_example), [Pitfalls](../getting-started/pitfalls) |
| How to do specific things (debugging, runtime tuning, plugin failures) | [How-to Guides](../how-to) |
| Coding standards, MPK contract, contribution policy, migration | [Coding standard](../contribute/coding_standard), [MPK contract](../contribute/mpk_contract), [Architecture](../contribute/architecture), [Migration](../contribute/migration) |

## Why the framework is built for agents

A consequence of how the framework was designed: it's an exceptionally good substrate for AI code generation. Every framework error produces a structured `SessionReport` (machine-readable error code + reproducer command). Every public symbol has stable naming. Validation runs before any pipeline starts. Pipelines are serializable JSON. The public API is ABI-stable. Errors fail fast with actionable messages instead of silent fallbacks.

These properties weren't picked for AI agents specifically — they came from "make the framework deterministic, debuggable, and never hang the process" (see [`docs/contribute/architecture.md`](../contribute/architecture)). But they happen to be exactly what an AI agent needs to write code that converges quickly. That's why the **agentic workflow** in the Neat environment delivers what it does — the framework is the substrate, the environment is the workshop.

The full story of why the framework is good for agents — fifteen specific design properties, each tied to architect-doc rationale — is in the [Design / Deep Dive §0.1 Introducing Neat](../design) chapter.

## Conventions in this reference

- All public types live under the `simaai::neat` namespace.
- Nodes live under `simaai::neat::nodes::{common,io,rtp,sima,groups}`.
- Headers are organized by responsibility: `model/`, `pipeline/`, `mpk/`, `builder/`, `graph/`, `nodes/`, `contracts/`, `policy/`, `gst/`, plus convenience umbrellas under `neat/`.
- Every public header has a file-level `@file` / `@ingroup` / `@brief` block. Group definitions are in [`docs/doxygen/groups.dox`](groups.dox).
- Errors are returned as `SessionError` exceptions carrying a structured `SessionReport`. See the [Error code catalog](../concepts/error_codes) for the full taxonomy.

The rest of this site is the API itself. Start by browsing [classes](../reference/cppapi/classes), or jump to the headline types: [`Model`](../reference/cppapi/classes/simaai-neat-model), [`Session`](../reference/cppapi/classes/simaai-neat-session), [`Run`](../reference/cppapi/classes/simaai-neat-run), [`Tensor`](../reference/cppapi/classes/simaai-neat-tensor), [`Node`](../reference/cppapi/classes/simaai-neat-node).
