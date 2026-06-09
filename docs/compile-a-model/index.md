---
title: Overview
---

# Compile a Model

The compiled model artifact you use when building an application can come from
one of three paths:

- **Model Zoo** — precompiled models
- **Normal compilation workflow** — vision and ONNX models
- **GenAI compilation flow with LLiMa** — GenAI models

## Start with the Model Zoo

Before compiling your own model, check the <a href="/tools/model-zoo/">Model Zoo</a>.
It provides ready-to-run artifacts for vision, other ONNX-style models, and
GenAI models, which is the fastest path when a suitable model already exists.

## Choose the right workflow

For **GenAI** models, use <a href="/genai-llima/">LLiMa</a> to compile, test,
and benchmark the model for Modalix.

For **vision and ONNX models**, use the normal compilation workflow in the
ModelSDK extension to check compatibility, quantize, compile, and validate the
model artifact before using it from the Neat Framework.

<div class="overview-link-columns compile-workflow-columns">
  <section class="overview-link-panel overview-link-panel-model">
    <h2>Compilation workflow</h2>
    <p>Follow this path for vision and other ONNX-style models that need ModelSDK preparation.</p>
    <ul class="overview-link-list compile-workflow-list">
      <li><a class="overview-link-card" href="/compile-a-model/compile-your-first-model/"><strong>Compile your first model</strong><span>Create a first compiled artifact with the ModelSDK extension.</span></a></li>
      <li><a class="overview-link-card" href="/compile-a-model/model-compatibility/"><strong>Model compatibility</strong><span>Check whether the model can be prepared for Modalix.</span></a></li>
      <li><a class="overview-link-card" href="/compile-a-model/graph-surgery/"><strong>Graph Surgery</strong><span>Adjust model graphs when compatibility work is required.</span></a></li>
      <li><a class="overview-link-card" href="/compile-a-model/quantization/"><strong>Quantization</strong><span>Prepare numeric precision for efficient execution.</span></a></li>
      <li><a class="overview-link-card" href="/compile-a-model/model-compilation/"><strong>Compilation</strong><span>Compile the prepared model into a Modalix-ready artifact.</span></a></li>
      <li><a class="overview-link-card" href="/compile-a-model/validate-accuracy-performance/"><strong>Validate accuracy and performance</strong><span>Compare outputs and measure runtime behavior.</span></a></li>
    </ul>
  </section>
</div>
