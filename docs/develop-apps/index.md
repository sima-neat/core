---
title: Develop Apps
description: Build and run AI applications on the SiMa.ai platform with Neat
sidebar_position: 1
---

# Develop Apps with Neat

<LanguageContent lang="cpp">

<div className="overview-workflow-image overview-workflow-image-light">

![Neat three-step workflow diagram](./images/neat-overview-workflow-cpp.svg)

</div>

<div className="overview-workflow-image overview-workflow-image-dark">

![Neat three-step workflow diagram](./images/neat-overview-workflow-cpp-dark.svg)

</div>

</LanguageContent>

<LanguageContent lang="py">

<div className="overview-workflow-image overview-workflow-image-light">

![Neat three-step workflow diagram](./images/neat-overview-workflow-python.svg)

</div>

<div className="overview-workflow-image overview-workflow-image-dark">

![Neat three-step workflow diagram](./images/neat-overview-workflow-python-dark.svg)

</div>

</LanguageContent>

<div class="overview-section-label">What Neat Is</div>

SiMa.ai Neat is an application-development framework for building and running AI applications on the SiMa.ai platform.
It provides developers a set of Python and C++ APIs to execute and test compiled model artifacts (`tar.gz models`), compose AI applications that leverage the SoC's hardware blocks, and manage runtime execution.

In the broader SiMa.ai software ecosystem, Neat sits at the application layer, building on the SiMa.ai runtime stack and using GStreamer-based execution underneath so developers can stay focused on application logic instead of manually stitching together lower-level runtime pieces.

<div class="overview-section-label">C++ or PyNeat?</div>

Neat ships the same workflow through two front ends, so you can pick the one that fits your application:

- **PyNeat** — the Python bindings (`pyneat`). Best for quick iteration, notebooks, data-science workflows, and validating directly on the DevKit, where the Python runtime executes.
- **C++** — the native `simaai::neat` API. Best for larger applications, tight integration with existing C++ codebases, and cross-compiled host-to-DevKit workflows.

Both target the same compiled model artifacts and the same DevKit runtime; the concepts and pages below apply to either.

<div class="overview-section-label">Requirements</div>

Before building applications, complete the Getting Started setup:

- **Install and sync** — install the Neat Library in your Palette SDK or Modalix environment, then pair and sync your DevKit if you are developing from a host workflow.
- **Model artifact** — use a precompiled model from the Model Zoo or compile your own model into a Modalix-ready archive.
- **Runtime target** — decide whether you want to iterate directly on Modalix with PyNeat or cross-compile a C++ application from Palette SDK.

See [Installation](/getting-started/installation/), [Pair a DevKit](/getting-started/pair-a-devkit/), and [Compile a Model](/compile-a-model/) if any of those steps are not ready yet.

<div class="overview-section-label">How It Works</div>

Neat gives you a direct mental model for that path. A compiled model archive (`.tar.gz`) becomes a `Model` component, application logic is assembled as a `Graph`, and that graph is built and executed as a `Run` object on the SoC. The same workflow is designed to work well with agentic development too, so teams can explore, build, and iterate faster.

The Hello Neat! pages help you run your first inference, the Development Workflow pages explain the main concepts in more detail, and the tutorials show how to apply them to real application patterns.

<div class="overview-link-columns">
  <section class="overview-link-panel overview-link-panel-start">
    <h2>Start Here</h2>
    <p>Start from a working environment and build up the core Neat application workflow.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/hello-neat/minimal/"><strong>Hello Neat!</strong><span>Run a minimal Neat application and verify the development loop.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/development-workflow/overview/"><strong>Development Workflow</strong><span>Learn the `Model`, `Graph`, and `Run` workflow in more detail.</span></a></li>
      <li><a class="overview-link-card" href="/tutorials/"><strong>Tutorials</strong><span>Follow guided examples that walk through real Neat application patterns.</span></a></li>
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
