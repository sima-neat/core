---
title: Advanced Concepts
description: Design details for building richer Neat applications
sidebar_position: 1
slug: /develop-apps/advanced-concepts/
---

# Advanced Concepts

Use these pages when your application needs more than the basic `Model`,
`Graph`, and `Run` workflow. They explain the contracts and runtime behavior
behind richer Neat applications.

<div class="overview-link-columns">
  <section class="overview-link-panel overview-link-panel-app">
    <h2>Application Design</h2>
    <p>Design how your application is composed and how it exposes outputs.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/graphs/"><strong>Graphs</strong><span>Compose models, nodes, named inputs and outputs, branches, and runs.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/mipi-camera-input/"><strong>Use a MIPI Camera</strong><span>Build a source-owned camera-to-model graph with libcamera input and CVU preprocessing.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/video_sender/"><strong>Send Video</strong><span>Stream video output from a Neat application over H.264 RTP/UDP.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/metadata_sender/"><strong>Send JSON Metadata</strong><span>Publish structured application metadata over UDP JSON.</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-reference">
    <h2>Execution Model</h2>
    <p>Understand how work is scheduled, threaded, and mapped onto the pipeline layer.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/timing_model/"><strong>Timing Model</strong><span>Understand sync and async execution, push/pull, and when work happens.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/threading/"><strong>Threading Model</strong><span>Know which threads exist and where application code may run.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/gstreamer_layer/"><strong>GStreamer Layer</strong><span>Learn what Neat abstracts and when raw GStreamer details matter.</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-model">
    <h2>Data &amp; Model Contracts</h2>
    <p>Understand the tensor, memory, and model contracts that pipelines rely on.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/data_formats/"><strong>Data Formats</strong><span>Map format tokens to tensor layout, shape, and plane semantics.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/dtype_contract/"><strong>Dtype Contract</strong><span>Follow how tensor precision changes across preprocess, MLA, and postprocess.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/memory_model/"><strong>Memory Model</strong><span>Understand zero-copy buffers, physical addresses, and cache behavior.</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-start">
    <h2>Model Runtime</h2>
    <p>Go deeper into compiled model artifacts and the hardware backends that execute them.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/mpk_contract/"><strong>MPK Contract</strong><span>See how compiled model archives define inference stages and contracts.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/processor_backends/"><strong>Processor Backends</strong><span>Understand A65, CVU, MLA, MLASHM, APU, TVM, and M4 roles.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/advanced-concepts/cvu_kernels/"><strong>CVU Kernels</strong><span>Review preprocess and postprocess graph building blocks.</span></a></li>
    </ul>
  </section>
</div>
