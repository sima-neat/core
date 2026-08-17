---
title: Develop Apps
description: Build and run AI applications on Modalix with SiMa.ai Neat
sidebar_position: 1
---

# Develop Apps with SiMa.ai Neat

<LanguageContent lang="cpp">

<div className="overview-workflow-image overview-workflow-image-light">

![SiMa.ai Neat composed-application workflow for C++](./images/neat-overview-workflow-cpp.svg)

</div>

<div className="overview-workflow-image overview-workflow-image-dark">

![SiMa.ai Neat composed-application workflow for C++](./images/neat-overview-workflow-cpp-dark.svg)

</div>

</LanguageContent>

<LanguageContent lang="py">

<div className="overview-workflow-image overview-workflow-image-light">

![SiMa.ai Neat composed-application workflow for Python](./images/neat-overview-workflow-python.svg)

</div>

<div className="overview-workflow-image overview-workflow-image-dark">

![SiMa.ai Neat composed-application workflow for Python](./images/neat-overview-workflow-python-dark.svg)

</div>

</LanguageContent>

## What SiMa.ai Neat Is

SiMa.ai Neat is an application development framework for building and running
AI applications on the SiMa.ai platform. It provides Python and C++ APIs for
loading and running compiled model archives (`.tar.gz`), composing end-to-end
applications that use Modalix processing resources, and managing runtime
execution.

Within the broader SiMa.ai software stack, SiMa.ai Neat sits at the application
layer. It builds on the SiMa.ai runtime stack and uses GStreamer underneath, so
developers can focus on application logic instead of manually connecting
lower-level runtime components.

For the shortest path to inference, load a compiled model archive as a `Model`
and run it directly. When an application needs multiple inputs, processing
stages, models, or outputs, compose those components as a `Graph` and build it
into a `Run`. The same public APIs support traditional and agentic development,
so teams can review, extend, and maintain applications using either workflow.

### Choose Your Deployment Model

- **Run on a Modalix DevKit** — the application and Neat graph run directly on
  the device. Use `simaai::neat` or `pyneat` with `Model`, `Graph`, `Node`, and
  `Run`. Start with [Run / Inference](/develop-apps/development-workflow/overview/).
- **Use a Modalix PCIe Card for co-processing** — the application runs on a
  host machine and sends tensors or images to the card for model execution. Use
  `simaai::neat::pcie` or `pyneatpcie`. Start with
  [PCIe Co-processing](/develop-apps/development-workflow/pcie-model/).

### C++ or PyNeat

For applications running directly on a Modalix DevKit, SiMa.ai Neat provides
the same core workflow through two language interfaces, so you can choose the
one that fits your application:

- **PyNeat** — the Python bindings (`pyneat`). Best for quick iteration, notebooks, data-science workflows, and running Python applications directly on the DevKit.
- **C++** — the native `simaai::neat` API. Best for larger applications, tight integration with existing C++ codebases, and cross-compiled host-to-DevKit workflows.

Both use the same compiled model artifacts and Modalix runtime; the concepts and
pages below apply to either. PCIe co-processing provides separate C++ and Python
interfaces through `simaai::neat::pcie` and `pyneatpcie`.

## Develop the application. <span className="neat-heading-highlight">SiMa.ai Neat maps it for you.</span>

Modalix combines application cores, vision processing, machine learning
acceleration, video engines, shared memory, and high-speed I/O in one SoC.
Through its Python and C++ APIs, SiMa.ai Neat provides one programming model for
building applications across the application-relevant processing resources in
the system.

Build an end-to-end flow from a camera or network stream through processing and
inference to the final result. SiMa.ai Neat constructs the runtime pipeline,
selects accelerated implementations where applicable, and coordinates
execution and data movement across Modalix. You focus on the application while
SiMa.ai Neat handles the underlying hardware and runtime complexity.

<div className="overview-workflow-image modalix-application-map-desktop">

![A SiMa.ai Neat application mapped onto an MLSoC Modalix floorplan](../images/neat-modalix-floorplan-animated.svg)

</div>

<div className="overview-workflow-image modalix-application-map-mobile">

![A mobile view of a SiMa.ai Neat application mapped onto an MLSoC Modalix floorplan](../images/neat-modalix-floorplan-mobile-animated.svg)

</div>

<p className="overview-figure-caption"><strong>Illustrative mapping:</strong> the selected route depends on the application, model, and available hardware acceleration. See <a href="/develop-apps/advanced-concepts/processor_backends/">Processor backends</a> for the technical mapping.</p>

## Describe your application. <span className="agentic-heading-highlight">An agent with Neat skills develops it.</span>

SiMa.ai Neat supports agentic application development out of the box through
skills included with the Neat Development Environment (referred to as Neat SDK). These skills give coding
agents the context to use the public Python and C++ APIs, follow established
application patterns, and work with the Modalix development and validation
workflow.

The recommended agentic path can create an application, run it on a paired
Modalix DevKit, inspect results and diagnostics, and refine the implementation.
Traditional development remains a parallel path for direct control through the
same APIs. Both produce standard, inspectable SiMa.ai Neat applications, so you
can review or modify agent-developed code and move between the two workflows as
the application evolves. See [Set up the Neat SDK](/getting-started/dev-environment/) to enable agentic
development.

<div className="overview-workflow-image agentic-visual-desktop">

![A coding agent creates, runs, diagnoses, and refines a SiMa.ai Neat application](../images/agentic-development-loop-animated.svg)

</div>

<div className="overview-workflow-image agentic-visual-mobile">

![Mobile view of the SiMa.ai Neat agentic development loop](../images/agentic-development-loop-mobile-animated.svg)

</div>

## Requirements

Before building applications, complete the Getting Started setup:

- **Install for the deployment model** — for a Modalix DevKit, install the Neat
  Library in the Neat SDK or directly on the device. For PCIe co-processing,
  install `core/pciehost` on the host machine and a compatible Neat Library on the
  Modalix PCIe Card.
- **Model artifact** — use a precompiled model from the Model Zoo or compile your own model into a Modalix-ready archive.
- **Runtime target** — run native applications on the DevKit, or build and run a
  co-processing application directly on the host machine. Pair and sync the DevKit
  when cross-compiling native C++ applications in the Neat SDK.

The Hello Neat! pages help you run your first inference, the Development Workflow pages explain the main concepts in more detail, and the tutorials show how to apply them to real application patterns.

For complete applications you can study, adapt, and run, browse through the [application examples](https://developer.sima.ai/examples).

<div class="overview-link-columns">
  <section class="overview-link-panel overview-link-panel-start">
    <h2>Start Here</h2>
    <p>Start from a working environment and build up the core SiMa.ai Neat application workflow.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/hello-neat/minimal/"><strong>Hello Neat!</strong><span>Run a minimal Neat application and verify the development loop.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/development-workflow/overview/"><strong>Development Workflow</strong><span>Learn the `Model`, `Graph`, and `Run` workflow in more detail.</span></a></li>
      <li><a class="overview-link-card" href="/tutorials/"><strong>Tutorials</strong><span>Follow guided examples that walk through real SiMa.ai Neat application patterns.</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-explore">
    <h2>Build More</h2>
    <p>Use these sections when you are ready to build richer applications or inspect the API surface.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/"><strong>Advanced Concepts</strong><span>Understand graphs, formats, memory, threading, and runtime behavior.</span></a></li>
      <li><a class="overview-link-card" href="/reference/"><strong>Reference</strong><span>Browse C++, Python, Model Compiler, troubleshooting, and supporting material.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/architecture/"><strong>Contribute</strong><span>Understand architecture, source builds, testing expectations, and repo conventions.</span></a></li>
    </ul>
  </section>
</div>
