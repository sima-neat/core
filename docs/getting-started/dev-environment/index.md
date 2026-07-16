---
title: Neat SDK
description: Set up the Neat SDK for fast, agent-ready Neat application development
sidebar_position: 1
---

The Neat Development Environment (referred to as Neat SDK) is the recommended
host-side workspace for building large scale Neat applications and validating them on a
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

Start with SDK installation. Use the other SDK topics when you need to change
settings, add Model Compiler later, understand DevKit Sync behavior, or prepare
offline packages for restricted networks.

:::tip SDK-only happy path
If you installed the SDK and have not paired a DevKit, you only need two steps:
[Install the Environment](/getting-started/dev-environment/install-the-environment/),
then [Compile a Model](/compile-a-model/) — model compilation runs entirely in the
SDK. Configure SDK, DevKit Sync, Install Model Compiler (it is offered during
setup), and the Neat Library and PyNeat pages are optional side-trips; visit them
only when you need them or once you pair a DevKit.
:::

<div class="overview-link-columns">
  <section class="overview-link-panel overview-link-panel-start">
    <h2>SDK Topics</h2>
    <p>Install the SDK, then use optional configuration topics when you need to change settings, install Model Compiler, or work with a paired DevKit.</p>
    <ul class="overview-link-list">
      <li><a class="overview-link-card" href="/getting-started/dev-environment/install-the-environment/"><strong>Install the Environment</strong><span>Install and set up the SDK package that matches your DevKit software version.</span></a></li>
      <li><a class="overview-link-card" href="/getting-started/dev-environment/devkit-sync/"><strong>DevKit Sync</strong><span>Understand workspace sharing, pairing updates, rsync fallback, and <code>dk</code> command execution.</span></a></li>
      <li><a class="overview-link-card" href="/getting-started/dev-environment/offline-installation/"><strong>Offline Installation</strong><span>Download SDK and Model Compiler packages for restricted network environments.</span></a></li>
    </ul>
  </section>
</div>

To change SDK settings after installation, such as workspace location or DevKit
pairing, see
[Configure SDK](/getting-started/dev-environment/configure-sdk/).

Model Compiler is offered during SDK setup. To install it later, pin a specific
version, or use a standalone host, see
[Install Model Compiler](/getting-started/dev-environment/install-model-compiler/).

For hosts that cannot download packages directly, see
[Offline Installation](/getting-started/dev-environment/offline-installation/).

## What's Included

- **Cross-compilation environment:** build C++ Neat applications for Modalix
  from a Linux container on your host.
- **DevKit Sync:** pair the SDK with a Modalix DevKit and expose the same
  workspace in both places.
- **Model tooling:** install the matching Model Compiler in the SDK. It is
  required to compile or quantize ONNX or GenAI models yourself, and optional
  only if you exclusively use precompiled model packages.
- **Insight:** inspect workspace files, media sources, stream delivery, and
  runtime behavior from a browser.
- **Agent-ready context:** use bundled Codex and Claude skills with current
  Neat source references and examples.

## Host requirements

Confirm your host machine meets these minimums before you install the SDK.
Administrator (`sudo`) privileges are required to install the SDK on every
supported host — not only for optional DevKit networking — because installing
`sima-cli`, Docker Engine, the SDK image, and NFS packages all need elevated
permissions.

| Host OS | CPU | RAM | Free disk | Admin / sudo |
| --- | --- | --- | --- | --- |
| Ubuntu 22.04 / 24.04 (`x86_64` or `arm64`) | 4 cores min | 16 GB min | 100 GB | `sudo` for SDK install (`sima-cli`, Docker, SDK image), NFS install/config, and shared-network/firewall setup |
| Windows 11 via WSL (`x86_64`) | 4 cores min | 16 GB min | 100 GB | Administrator for SDK install in WSL (Docker, `sima-cli`), WSL networking, and NFS firewall rules |
| macOS 15.5+ Apple Silicon (`arm64`) | 4 cores min | 16 GB min | 100 GB | Administrator for SDK install (Homebrew, Colima, `sima-cli`), Full Disk Access (`nfsd`), and Internet Sharing |

:::note GenAI model compilation needs more
Compiling GenAI models with LLiMa is far heavier than the base SDK: 128 GB RAM
is recommended and 512 GB of disk is preferred, and a higher core count helps.
See [GenAI setup](/genai-llima/setup/) for the full requirements.
:::

## Supported Platforms

| Platform | Arch | SDK | Model Compiler |
| --- | --- | --- | --- |
| Ubuntu 22.04 and 24.04 through Docker Engine | `x86_64` | Yes | Yes |
| Windows 11 through WSL and Docker Engine | `x86_64` | Yes | Yes |
| Ubuntu 22.04 and 24.04 through Docker Engine | `arm64` | Yes | Model Compiler 2.1.2 and later |
| macOS 15.5 or above through Colima | `arm64` | Yes | Model Compiler 2.1.2 and later; install it inside the Neat SDK |

:::note Architecture names
`arm64` and `aarch64` are the same 64-bit Arm architecture — macOS reports it as
`arm64`, Linux reports it as `aarch64`. Likewise, `x86_64` and `amd64` are the same
architecture. Run `uname -m` on your host (or inside the SDK) to see which one you
have. The Model Compiler install commands use `arm64` and `amd64` — see
[Install Model Compiler](/getting-started/dev-environment/install-model-compiler/).
:::

:::note Installing a specific version
Standard installation pulls the current supported defaults. The `release-2.1`
channel used on the [Install the Environment](/getting-started/dev-environment/install-the-environment/)
page always tracks the latest 2.1 patch release. To pin an exact SDK, Neat Library,
or Model Compiler version, see the [Compatibility Guide](/getting-started/compatibility/).
:::

## Tools in the SDK

The SDK is the recommended place to install and update the
[Neat Library](/getting-started/neat-library/) when you are building
applications for a paired DevKit.

To access the SDK from a terminal, VS Code, or a browser with Neat Insight, see
[Install the Environment](/getting-started/dev-environment/install-the-environment/#access-the-sdk).

During SDK setup, `sima-cli` prompts you to install the matching Model Compiler
automatically. Install it if you compile or quantize models yourself; you can
skip it if you only use precompiled model packages. For compiler setup and
usage, see [Compile a Model](/compile-a-model/).

## Next Step

Start with [Install the Environment](/getting-started/dev-environment/install-the-environment/).
