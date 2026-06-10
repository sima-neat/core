---
title: Software Overview
sidebar_position: 1
---

# Software Overview

Welcome to the SiMa.ai software documentation. This guide introduces Palette SDK,
the Neat Framework, and the developer journey from model preparation to a running
AI application on Modalix.

<div class="overview-section-label">Software stack</div>

SiMa.ai software for application development has three main parts:

- **Palette SDK** — the SDK for cross-compilation, DevKit pairing, and building performant applications with the Neat Framework. Optionally, it provides the ModelSDK extension for quantizing and compiling AI models, including LLiMa for compiling, testing, and benchmarking GenAI models.
- **Neat Framework** — the application development and runtime framework inside Palette SDK and on Modalix. Developers use it to build and run AI applications.
- **Modalix DevKit** — the target hardware where compiled model artifacts and Neat Framework applications run.

Together, Palette SDK and the Neat Framework provide the **Palette Neat** workflow: prepare models, build applications, and validate them on Modalix.

<div class="overview-section-label">Developer journey</div>

If you are new, work through the sections in order. Start with environment and
hardware readiness, prepare the model you want to run, then build the application
that uses it.

<div class="overview-link-columns">
  <section class="overview-link-panel overview-link-panel-start">
    <h2>Start Here</h2>
    <p>Get your host, Palette SDK, and DevKit ready for local development and hardware validation.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/getting-started/compatibility/"><strong>Check Compatibility</strong><span>Confirm supported host, Palette SDK, DevKit, and package versions.</span></a></li>
      <li><a class="overview-link-card" href="/getting-started/installation/"><strong>Install Neat Framework</strong><span>Choose the right setup path for DevKit or Palette SDK development.</span></a></li>
      <li><a class="overview-link-card" href="/getting-started/pair-a-devkit/"><strong>Pair a DevKit</strong><span>Connect your host workflow to Modalix hardware.</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-model">
    <h2>Model Preparation</h2>
    <p>Turn trained models into artifacts that run on SiMa.ai hardware.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/compile-a-model/"><strong>Compile a Model</strong><span>Choose the right path for precompiled models, ONNX vision models, or GenAI models.</span></a></li>
      <li><a class="overview-link-card" href="/tools/model-zoo/"><strong>Use a Precompiled Model</strong><span>Start quickly with a ready-to-run model artifact.</span></a></li>
      <li><a class="overview-link-card" href="/tools/model-sdk/"><strong>GenAI with LLiMa</strong><span>Compile, test and benchmark GenAI models on Modalix.</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-app">
    <h2>Build an App</h2>
    <p>Use the Neat Framework to run models and compose production application flows.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/hello-neat/minimal/"><strong>Hello Neat!</strong><span>Run your first Neat application and verify the development loop.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/"><strong>Develop Apps</strong><span>Build AI applications with the Neat Framework using C++ or PyNeat.</span></a></li>
      <li><a class="overview-link-card" href="/tutorials/"><strong>Tutorials</strong><span>Follow guided examples for real Neat application patterns.</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-reference">
    <h2>Tools &amp; Reference</h2>
    <p>Use supporting tools and API material when you need more detail.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/tools/"><strong>Tools</strong><span>sima-cli, ModelSDK, Model Zoo, Insight, and Palette SDK.</span></a></li>
      <li><a class="overview-link-card" href="/reference/"><strong>Reference</strong><span>Browse APIs, troubleshooting, environment variables, data formats, and release notes.</span></a></li>
      <li><a class="overview-link-card" href="/reference/troubleshooting/"><strong>Troubleshooting</strong><span>Find fixes for setup, runtime, and pipeline issues.</span></a></li>
    </ul>
  </section>
</div>
