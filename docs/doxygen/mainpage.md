# Neat framework

The **Neat framework** is the C++/Python library and runtime that runs models on the SiMa Modalix chip. It loads compiled `.tar.gz` model archives, assembles them into deterministic GStreamer pipelines, and executes inference at frame rate on heterogeneous compute (A65 host CPU + EV74 vector cores + MLA accelerator + hardware codec).

This site is the **reference documentation** for the framework's public API. For design rationale and worked examples, see the [Architecture deep dive](/develop-apps/contribute/architecture), which mirrors the architect-level documentation.

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
| 1 | **Model archive** | Sealed `.tar.gz` file from the compiler — kernels, weights, configs, and the MPK contract. | [`Model`](/reference/cppapi/classes/simaai-neat-model), [MPK contract](/develop-apps/advanced-concepts/mpk_contract) |
| 2 | **Model** | Loaded form of a model archive; the simplified entry point. | [`Model`](/reference/cppapi/classes/simaai-neat-model) |
| 3 | **Tensor** | Typed data unit that flows between stages. | [`Tensor`](/reference/cppapi/structs/simaai-neat-tensor), [Memory model](/develop-apps/advanced-concepts/memory_model), [dtype contract](/develop-apps/advanced-concepts/dtype_contract) |
| 4 | **Nodes** | Smallest building blocks; each wraps one (or a few) GStreamer elements. | [Node APIs](/reference/cppapi/groups/nodes), [GStreamer underneath](/develop-apps/advanced-concepts/gstreamer_layer) |
| 5 | **Reusable Graph fragments** | Pre-made `Graph` fragments capturing common patterns. | [Reusable Graph fragments](/reference/cppapi/groups/nodes-groups) |
| 6 | **Graph** | Assembly stage that turns Nodes, Models, and reusable Graph fragments into a runnable pipeline. | [`Graph`](/reference/cppapi/classes/simaai-neat-graph), [`GraphOptions`](/reference/cppapi/structs/simaai-neat-graphoptions) |
| 7 | **Run** | Live, running pipeline produced by `Graph::build()`. | [`Run`](/reference/cppapi/classes/simaai-neat-run), [Async vs sync timing](/develop-apps/advanced-concepts/timing_model), [Threading model](/develop-apps/advanced-concepts/threading) |

## Building and linking against Neat (CMake)

Application code consumes Neat as the installed `sima-neat` package — it provides `libsima_neat.{a,so}`, the public headers, and a CMake package config, `SimaNeatConfig.cmake`. A complete `CMakeLists.txt` for an app is just:

```cmake
cmake_minimum_required(VERSION 3.16)
project(my_app LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SimaNeat REQUIRED CONFIG)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE SimaNeat::sima_neat)
```

Two lines are all you add to link and target the Neat library:

- **`find_package(SimaNeat REQUIRED CONFIG)`** — finds the installed Neat package by reading its package config file `SimaNeatConfig.cmake` (installed under `lib/cmake/SimaNeat/`). `CONFIG` selects config-file mode (use the package's own exported config rather than a `Find<Pkg>.cmake` module); `REQUIRED` aborts configuration with a clear error if Neat isn't found. On success it defines the imported target `SimaNeat::sima_neat`.
- **`target_link_libraries(my_app PRIVATE SimaNeat::sima_neat)`** — links your target against Neat. `SimaNeat::sima_neat` is an *imported target* that carries Neat's usage requirements, so linking it automatically adds Neat's public include directories, its C++20 requirement, and its transitive dependencies (GStreamer, etc.) to your target. You set no `-I`, `-L`, or `-l` flags by hand.

In your sources, pull in the umbrella header:

```cpp
#include "neat.h"   // simaai::neat::Model, Graph, Run, Tensor, nodes, …
```

### Cross-compiling from the Neat SDK

On a native DevKit install, `SimaNeatConfig.cmake` is on the default system prefix and `find_package` resolves with no extra setup. In an SDK cross-build, point CMake at the exported sysroot before `find_package` so it can locate the aarch64 package:

```cmake
if(DEFINED ENV{SYSROOT} AND NOT "$ENV{SYSROOT}" STREQUAL "")
  list(APPEND CMAKE_PREFIX_PATH
    "$ENV{SYSROOT}/usr"
    "$ENV{SYSROOT}/usr/lib/aarch64-linux-gnu")
endif()
```

Then configure, build, and run:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/my_app
```

If `find_package(SimaNeat ...)` fails, see [Troubleshooting](/reference/troubleshooting) (the `find_package(SimaNeat CONFIG)` entry) and the worked [Hello Neat](/develop-apps/hello-neat/minimal) and [Run an App](/develop-apps/hello-neat/run_an_app) examples.

## Where to read what

| What you want | Where to look |
|---|---|
| Conceptual overview, architecture, design rationale | [Architecture deep dive](/develop-apps/contribute/architecture) — the architect-level documentation, surfaced as navigable site pages |
| C++ public API — every class, method, field | [C++ API Reference](/reference/cppapi/) — generated from Doxygen comments |
| Python public API — pyneat module surface | [Python API Reference](/reference/pythonapi/) — generated from nanobind bindings |
| Conceptual deep-dives (dtype contract, memory model, error codes, etc.) | [dtype contract](/develop-apps/advanced-concepts/dtype_contract), [memory model](/develop-apps/advanced-concepts/memory_model), [graphs](/develop-apps/advanced-concepts/graphs), [timing model](/develop-apps/advanced-concepts/timing_model), [threading](/develop-apps/advanced-concepts/threading), [processor backends](/develop-apps/advanced-concepts/processor_backends), [GStreamer underneath](/develop-apps/advanced-concepts/gstreamer_layer), [CVU kernels](/develop-apps/advanced-concepts/cvu_kernels), [MPK contract](/develop-apps/advanced-concepts/mpk_contract), [error codes](/develop-apps/advanced-concepts/error_codes), [build options](/develop-apps/advanced-concepts/build_options), [agentic workflow](/develop-apps/advanced-concepts/agentic_workflow) |
| Glossary, environment variables, scripts, error format | [Glossary](/reference/glossary), [env vars](/reference/environment-variables), [scripts inventory](/reference/scripts), [plugin error format](/reference/error_format) |
| Onboarding, build, first inference | [Installation](/getting-started/installation), [Build](/develop-apps/contribute/build), [Hello Neat](/develop-apps/hello-neat/minimal) |
| How to do specific things (debugging, runtime tuning, plugin failures) | [Tutorial 015: tune throughput & queues](/tutorials/015-tune-throughput-and-queues), [Tutorial 011: diagnose a pipeline](/tutorials/011-diagnose-a-pipeline) |
| Coding standards, MPK contract, contribution policy | [Coding standard](/develop-apps/contribute/coding_standard), [MPK contract](/develop-apps/contribute/mpk_contract), [Architecture](/develop-apps/contribute/architecture) |

## Why the framework is built for agents

A consequence of how the framework was designed: it's an exceptionally good substrate for AI code generation. Every framework error produces a structured `GraphReport` (machine-readable error code + reproducer command). Every public symbol has stable naming. Validation runs before any pipeline starts. Pipelines are serializable JSON. The public API is ABI-stable. Errors fail fast with actionable messages instead of silent fallbacks.

These properties weren't picked for AI agents specifically — they came from "make the framework deterministic, debuggable, and never hang the process" (see [`docs/develop-apps/contribute/architecture.md`](/develop-apps/contribute/architecture)). But they happen to be exactly what an AI agent needs to write code that converges quickly. That's why the **agentic workflow** in the Neat environment delivers what it does — the framework is the substrate, the environment is the workshop.

The full story of why the framework is good for agents — fifteen specific design properties, each tied to architect-doc rationale — is in the [Architecture deep dive](/develop-apps/contribute/architecture).

## Conventions in this reference

- All public types live under the `simaai::neat` namespace.
- Nodes live under `simaai::neat::nodes::{common,io,rtp,sima,groups}`.
- Headers are organized by responsibility: `model/`, `pipeline/`, `builder/`, `nodes/`,
  `contracts/`, `policy/`, `gst/`, plus convenience umbrellas under `neat/`. The
  source-tree `include/graph/` runtime substrate is intentionally excluded from
  the public reference; application code should use `simaai::neat::Graph` / `Run`.
- Every public header has a file-level `@file` / `@ingroup` / `@brief` block. Group definitions live in `docs/doxygen/groups.dox`.
- Errors are returned as `NeatError` exceptions carrying a structured `GraphReport`. See the [Error code catalog](/develop-apps/advanced-concepts/error_codes) for the full taxonomy.

The rest of this site is the API itself. Start by browsing [classes](/reference/cppapi/classes), or jump to the headline types: [`Model`](/reference/cppapi/classes/simaai-neat-model), [`Graph`](/reference/cppapi/classes/simaai-neat-graph), [`Run`](/reference/cppapi/classes/simaai-neat-run), [`Tensor`](/reference/cppapi/structs/simaai-neat-tensor), [Node APIs](/reference/cppapi/groups/nodes).
