---
title: Neat Development Environment
description: Set up the Neat Development Environment for fast, agent-ready Neat application development
sidebar_position: 1
---

The Neat Development Environment, also known as the SDK, is the recommended
host-side workspace for building Neat applications and validating them on a
Modalix DevKit. It brings the build tools, model tools, hardware connection, and
agent-ready source context into one containerized workflow.

The SDK connects three places:

- **Host machine:** where you install and start the SDK container.
- **SDK container:** where you build applications, compile models, use agent
  tooling, and inspect shared files.
- **Modalix DevKit:** where compiled model artifacts and Neat applications run
  on hardware.

DevKit Sync connects those places with a shared `/workspace`, so build outputs,
logs, model artifacts, and application files are visible from the host, SDK
container, and DevKit without a manual copy step. That shared workspace is the
center of the SDK workflow.

<div class="overview-section-label">Start here</div>

Follow these sections in order when setting up the Neat Development Environment.

<div class="overview-link-columns">
  <section class="overview-link-panel overview-link-panel-start">
    <h2>Setup Flow</h2>
    <p>Install the SDK, connect it to a DevKit when available, then run applications on hardware from the same workspace.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/getting-started/dev-environment/install-the-environment/"><strong>Install the Environment</strong><span>Install the SDK image that matches your DevKit software version.</span></a></li>
      <li><a class="overview-link-card" href="/getting-started/dev-environment/pair-with-a-devkit/"><strong>Pair with a DevKit</strong><span>Set up DevKit Sync, shared workspace mapping, and pairing updates.</span></a></li>
      <li><a class="overview-link-card" href="/getting-started/dev-environment/run-on-the-devkit/"><strong>Run on the DevKit</strong><span>Use <code>dk</code> from the SDK shell to execute ARM64 binaries and Python scripts on hardware.</span></a></li>
    </ul>
  </section>
</div>

## What's Included

- **Cross-compilation environment:** build C++ Neat applications for Modalix
  from a Linux container on your host.
- **DevKit Sync:** pair the SDK with a Modalix DevKit and expose the same
  workspace in both places.
- **Model tooling:** install the matching Model Compiler in the SDK when you
  need to quantize and compile ONNX or GenAI models.
- **Insight:** inspect workspace files, media sources, stream delivery, and
  runtime behavior from a browser.
- **Agent-ready context:** use bundled Codex and Claude skills with current
  Neat source references and examples.

## Supported Platforms

| Platform | Arch | SDK | Model Compiler |
| --- | --- | --- | --- |
| Ubuntu 22.04 and 24.04 through Docker Engine | x86_64 | Yes | Yes |
| Windows 11 through WSL and Docker Engine | x86_64 | Yes | Yes |
| Ubuntu 22.04 and 24.04 through Docker Engine | aarch64 | Yes | Yes with 2.1.2 or above |
| macOS 15.5 or above through Colima | aarch64 | Yes | Yes with 2.1.2 or above |

## Tools in the SDK

The SDK is the recommended place to install and update the
[Neat Library](/getting-started/neat-library/) when you are building
applications for a paired DevKit.

Neat Insight is served over HTTPS on port `9900` inside the SDK. Open
`https://localhost:9900` from the same host, or `https://<host-ip>:9900` from
another machine on the network. For more information, see
[Insight](/tools/insight/).

During SDK setup, `sima-cli` can prompt you to install the matching Model
Compiler. For compiler setup and usage, see
[Compile a Model](/compile-a-model/).

## VS Code Workflow

Connect VS Code to `ghcr.io-sima-neat-elxr-latest` with
[Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers),
then run `sima-cli login` once inside the container so the SDK can retrieve
assets such as prebuilt models.

Install the Codex or Claude Code extension in VS Code and ask what skills are
available. When `Neat` appears, the environment is ready for assisted
application development.

## Next Step

Start with [Install the Environment](/getting-started/dev-environment/install-the-environment/).
