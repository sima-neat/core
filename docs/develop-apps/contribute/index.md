---
title: Contribute
description: Contributor guide for Neat framework architecture, builds, tests, and release work
sidebar_position: 1
slug: /develop-apps/contribute/
---

# Contribute

This section is for contributors changing the Neat Framework itself. It explains
how the repository is structured, how to build and test changes, and which
contracts must stay stable for application developers.

<div class="overview-section-label">Contributor Orientation</div>

The **Neat Framework** is the C++/Python library and runtime in this repository.
It loads model archives, composes and validates pipelines, runs on Modalix
hardware, and exposes the public application API. Palette SDK and DevKit Sync
are the surrounding development workflow.

When changing this repository, optimize for framework properties that help both
humans and agent-assisted development: explicit APIs, deterministic behavior,
structured diagnostics, strict validation, and stable public contracts.

<div class="overview-section-label">Contributor Principles</div>

- **Determinism wins** — keep element names, generated pipelines, reports, and tests reproducible.
- **Debuggability is first-class** — failures should produce structured data, not only strings.
- **No silent fallback** — do not hide model-input bugs or hardware/runtime failures.
- **Validate before run** — catch structural, caps, shape, and contract errors before runtime.
- **Public APIs stay stable** — installed headers under `include/*` require additive, compatible changes.
- **Concurrency must be bounded and observable** — teardown must not hang and diagnostics must be thread-safe.

<div class="overview-link-columns">
  <section class="overview-link-panel overview-link-panel-start">
    <h2>Start Here</h2>
    <p>Understand the repository, naming rules, and expectations before changing code.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/contribute/architecture/"><strong>Architecture</strong><span>Learn repository structure, module ownership, runtime flow, and extension points.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/naming/"><strong>Naming Contract</strong><span>Use canonical product, API, package, and type names consistently.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/coding_standard/"><strong>Coding Standard</strong><span>Follow C++ style, public API, compatibility, and documentation expectations.</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-app">
    <h2>Build &amp; Test</h2>
    <p>Build the framework, validate changes, and work on Python bindings.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/contribute/build/"><strong>Build</strong><span>Build Neat from source and choose CMake profiles or build toggles.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/test_requirements/"><strong>Test Requirements</strong><span>Know which tests and docs updates are expected for each kind of change.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/python_bindings/"><strong>Python Bindings</strong><span>Build, test, and package PyNeat bindings as a contributor.</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-model">
    <h2>Contracts &amp; Internals</h2>
    <p>Use these when changing model archives, plugin contracts, or internal packaging.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/contribute/mpk_contract/"><strong>MPK Contract</strong><span>Understand model archive ingestion, validation, and security rules.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/sima_plugin_json_truth_map/"><strong>Plugin JSON Truth Map</strong><span>Review the frozen SIMA plugin JSON contract map.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/appcomplex_workspace_packaging/"><strong>AppComplex Packaging</strong><span>Build and install the gated appcomplex workspace service package.</span></a></li>
    </ul>
  </section>

  <section class="overview-link-panel overview-link-panel-reference">
    <h2>Release &amp; Maintenance</h2>
    <p>Use these for release gates, cleanup plans, and long-lived design guidance.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/develop-apps/contribute/release-checklist/"><strong>Release Checklist</strong><span>Follow release-blocking conditions, required checks, and reproducible steps.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/error-taxonomy-rollout/"><strong>Error Taxonomy Rollout</strong><span>Track structured error-code migration and verification status.</span></a></li>
      <li><a class="overview-link-card" href="/develop-apps/contribute/agentic-workflow/"><strong>Agentic Workflow</strong><span>See why the API is structured for human and AI-assisted development.</span></a></li>
    </ul>
  </section>
</div>
