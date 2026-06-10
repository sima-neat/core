---
title: Palette Neat
sidebar_position: 1
---

# Palette Neat

Palette Neat is the SiMa.ai software development toolkit for building AI
applications on Modalix. It includes the development environment, runtime
library, model tooling, and DevKit validation workflow. Together, these pieces
help you prepare models, build applications, and verify them on hardware.

Use this overview to understand the main parts of Palette Neat and choose the
right setup, model preparation, or application development path.

<div class="overview-section-label">Software stack</div>

Palette Neat includes four software components and the Modalix DevKit:

- **Neat Development Environment** — the containerized environment for cross-compilation, DevKit pairing, agent-assisted workflows, and local validation.
- **Neat Library** — the C++ and Python library and runtime for loading models, executing pipelines, and coordinating work across Modalix compute resources.
- **Model Compiler** — the optional component for model quantization, compilation, and validation.
- **LLiMa** — the GenAI toolkit, included with the development and model tooling, for compiling, testing, and benchmarking GenAI models.
- **Modalix DevKit** — the target hardware where compiled model artifacts and Neat Library applications run and are validated.

Together, these components support the full development path: prepare a model,
build an application, and validate the result on Modalix hardware.

<div class="overview-section-label">Developer journey</div>

If you are new to Palette Neat, work through these sections in order: confirm
compatibility, install and pair your development environment, prepare a model,
and then build the application that uses it.

<div class="overview-link-columns">
  <section class="overview-link-panel overview-link-panel-start">
    <h2>Start Here</h2>
    <p>Prepare your host machine, Neat Development Environment, and DevKit for local development and hardware validation.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/getting-started/compatibility/"><strong>Check Compatibility</strong><span>Confirm supported host, Neat Development Environment, DevKit software, Neat Library, and Model Compiler versions.</span></a></li>
      <li><a class="overview-link-card" href="/getting-started/installation/"><strong>Install Neat</strong><span>Choose the setup path for the Neat Development Environment or DevKit.</span></a></li>
      <li><a class="overview-link-card" href="/getting-started/pair-a-devkit/"><strong>Pair a DevKit</strong><span>Connect your host workflow to Modalix hardware.</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-model">
    <h2>Model Preparation</h2>
    <p>Turn trained models into deployable artifacts that run on Modalix hardware.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/compile-a-model/"><strong>Compile a Model</strong><span>Compile pretrained ONNX vision models or GenAI models for Modalix.</span></a></li>
      <li><a class="overview-link-card" href="/tools/model-zoo/"><strong>Use a Precompiled Model</strong><span>Start quickly with a ready-to-run model artifact.</span></a></li>
      <li><a class="overview-link-card" href="/tools/model-sdk/"><strong>GenAI with LLiMa</strong><span>Compile, test, and benchmark GenAI models on Modalix.</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-app">
    <h2>Build an App</h2>
    <p>Use the Neat Library to run models and compose production application pipelines.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/hello-neat/minimal/"><strong>Hello Neat!</strong><span>Run your first Neat application and verify the development workflow.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/"><strong>Develop Apps</strong><span>Build AI applications with the Neat Library using C++ or PyNeat.</span></a></li>
      <li><a class="overview-link-card" href="/tutorials/"><strong>Tutorials</strong><span>Follow guided examples for real Neat application patterns.</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-reference">
    <h2>Tools &amp; Reference</h2>
    <p>Use supporting tools and reference material when you need more detail.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/tools/"><strong>Tools</strong><span>sima-cli, Model Compiler, Model Zoo, Insight, and the Neat Development Environment.</span></a></li>
      <li><a class="overview-link-card" href="/reference/"><strong>Reference</strong><span>Browse APIs, troubleshooting, environment variables, data formats, and release notes.</span></a></li>
      <li><a class="overview-link-card" href="/reference/troubleshooting/"><strong>Troubleshooting</strong><span>Find fixes for setup, runtime, and pipeline issues.</span></a></li>
    </ul>
  </section>
</div>
