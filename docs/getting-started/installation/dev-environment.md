---
title: Neat Development Environment
description: Set up the Neat Development Environment for fast, agent-ready Neat application development
sidebar_position: 3
---

The Neat Development Environment (also known as the SDK) is the recommended host-side setup for compiling models, building Neat applications, and running them on a paired DevKit - all using one containerized workflow.

The SDK includes the following Neat application development features:

- **Build on the host architecture that fits your machine.**<br />
  The SDK provides x86 and arm64 Linux container images; on Apple Silicon Macs, use the arm64 image to avoid Rosetta overhead.
- **Keep build and test workflows in one shell.**<br />
  DevKit Sync lets developers and coding agents cross-compile and run DevKit binaries from the SDK without switching context.
- **Compile models and build applications in the same environment.**<br />
  Install the Model Compiler inside the SDK for a consistent workflow from model compilation to application development.
- **Debug vision applications from the browser.**<br />
  Insight helps configure video streams, view metadata overlays, inspect workspace files, and profile Neat applications.
- **Start agent-assisted development with the right context already available.**<br />
  The SDK ships with Codex and Claude skills, plus current source references for Neat Library and examples.

This setup supports end-to-end build-and-run loops against a paired DevKit from a single workspace.

The SDK source code is available on GitHub at [sima-neat/sdk](https://github.com/sima-neat/sdk).

## Supported platforms

| Platform | Arch | SDK | Model Compiler |
| --- | --- | --- | --- |
| Ubuntu 22.04 and 24.04 (through Docker Engine) | x86_64 | ✅ | ✅ |
| Windows 11 (through [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) and Docker Engine) | x86_64 | ✅ | ✅ |
| Ubuntu 22.04 and 24.04 (through Docker Engine) | aarch64 | ✅ | ✅ with 2.1.2 or above |
| macOS 15.5 or above (through [Colima](https://github.com/abiosoft/colima)) | aarch64 | ✅ | ✅ with 2.1.2 or above |

## Prerequisites

- Install `sima-cli` on your host machine using the [sima-cli installation guide](/tools/sima-cli/).
- Complete any host-specific setup for your platform:
  - [Ubuntu Host Notes](/reference/elxr-sdk-host-setup/ubuntu)
  - [Windows Host Notes](/reference/elxr-sdk-host-setup/windows)
  - [macOS Host Notes](/reference/elxr-sdk-host-setup/macos)

To use DevKit Sync, you also need:

- A Modalix DevKit running a supported DevKit software version. The DevKit software, SDK, and Neat Library versions must be compatible with each other; see [Compatibility Guide](/getting-started/compatibility/).
- Your host machine and DevKit on the same network with NFS traffic allowed.
- Your DevKit IP address.

## Install the SDK

Install the SDK version that matches your DevKit software and Neat Library version:

For Neat Development Environment 2.1.2:

<ShellCommand prompt="user-host-machine">
sima-cli install ghcr:sima-neat/sdk-v2.1.2
</ShellCommand>

For Neat Development Environment 2.0.0:

<ShellCommand prompt="user-host-machine">
sima-cli install ghcr:sima-neat/sdk-v2.0.0
</ShellCommand>

You can also install `ghcr:sima-neat/sdk` to use the latest SDK image from the main branch of the open source repository.

## Set Up the SDK

After installation, run one of the setup flows below.

### Pair with a DevKit

Use this flow when the DevKit is on the same network as the SDK host:

<ShellCommand prompt="user-host-machine">
sima-cli sdk setup --devkit {devkit-ip}
</ShellCommand>

During setup:

- Select container `ghcr.io-sima-neat-sdk-latest` if multiple SDK containers are present.
- Enter host admin password when prompted (NFS server setup on host).
- Enter DevKit user `sima` password (`edgeai`) when prompted (NFS client setup on DevKit).

When setup succeeds, you should see output similar to:

```text
============================================================
  DevKit Connected
============================================================
  DevKit target : sima@192.168.91.221:22
  Mounted path  : /workspace
  Host export   : 192.168.74.48:/Users/joey/workspace

  You can now run DevKit binaries from this SDK shell:
    dk /workspace/<path-to-arm64-binary> [args...]
============================================================

__SIMA_DEVKIT_BOOTSTRAP_STATUS=sourced_no_dk
✅ DevKit bootstrap completed in container 'ghcr.io-sima-neat-elxr-latest' (interactive).

✅ All selected containers started successfully!
```

### Set up without DevKit pairing

Use this flow when the DevKit is not reachable from the SDK host:

<ShellCommand prompt="user-host-machine">
sima-cli sdk setup
</ShellCommand>

You can still build binaries in the SDK container, but you must transfer them to the DevKit manually for testing. Make sure the DevKit is running a compatible Neat Library version.

### SDK CLI

Open the SDK shell with:

<ShellCommand prompt="user-host-machine">
sima-cli sdk neat
</ShellCommand>

## Insight

Neat Insight is a web app for configuring and inspecting vision ML application tests. It provides a workspace viewer, media library, RTSP source control, low-latency WebRTC video viewing with metadata overlays, and system/runtime statistics for debugging stream and application behavior.

For more information, see the [Insight documentation](/tools/insight/).

In the SDK, Insight is served over HTTPS on port `9900`. Open one of the following URLs:

- `https://localhost:9900` when browsing from the same machine where the SDK is running.
- `https://<host-ip>:9900` when browsing from another machine on the network.

The `neat` command also reports the actual Insight Web UI URL when the SDK exposes host-port mappings:

<ShellCommand prompt="sdk-container">
neat
</ShellCommand>

For more detail on the status output and JSON format, see [Neat CLI](/getting-started/installation/neat-library/#neat-cli).

## Model Compiler

During SDK setup, `sima-cli` can prompt you to install the matching Model Compiler.

The Model Compiler quantizes and compiles ONNX models so they can run on SiMa.ai's MLA (Machine Learning Accelerator). To install it, opt in when prompted during SDK setup.

For more information, see the [Model Compiler documentation](/tools/model-compiler/).

Example installation prompt:

```text
Setting up build environment for modalix...
Aliases loaded in this session.
✅ sima-cli installed for user 'sima' in 'ghcr.io-sima-neat-elxr-latest'.
╭───────────────────────────────────────────────── Model Compiler ─────────────────────────────────────────────────────╮
│ This SDK supports the installable Model Compiler.                                                                    │
│                                                                                                                       │
│ The Model Compiler lets you quantize and compile models so they can run on SiMa.ai hardware accelerated.               │
│ It will be installed on your host under $HOME/sima-sdk-extensions and mounted into this container at /sdk-extensions. │
│ Depending on network conditions, installation may take up to 15 minutes.                                              │
│                                                                                                                       │
│                                                                                                                       │
│ If you decide to install it later, run the matching install command from your host machine.                          │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
Install the Model Compiler now? (Y/n): y
ℹ️  Logging in to sima-cli before installing the Model Compiler...
```

Install the Model Compiler later with the command that matches your host architecture:

For `amd64` hosts:

<ShellCommand prompt="user-host-machine">
sima-cli install -v 2.1.2 tools/model-compiler/amd64
</ShellCommand>

For `arm64` hosts:

<ShellCommand prompt="user-host-machine">
sima-cli install -v 2.1.2 tools/model-compiler/arm64
</ShellCommand>

After installation, activate the compiler environment from inside the SDK shell:

<ShellCommand prompt="sdk-container">
activate-model-compiler
</ShellCommand>

To return to the default SDK shell, run:

<ShellCommand prompt="sdk-container">
deactivate-model-compiler
</ShellCommand>

Model Compiler support is available on Ubuntu and Windows through WSL. On macOS, use Model Compiler 2.1.2 or above.

## File Sharing with DevKit Sync

The DevKit Sync setup connects three environments:

1. Host
2. SDK container
3. DevKit

`sima-cli sdk setup --devkit {devkit-ip}` configures NFS so the same workspace is available in all three environments:

- The host workspace folder is exported through host NFS.
- The folder is mounted into the SDK container as `/workspace`.
- The same content appears on the DevKit as `/workspace` through NFS.
- The mounted folder name defaults to `/workspace` and can be changed during SDK setup.

![Host-Container-DevKit workspace mapping](../../images/elxr-sdk-workspaces.svg)

The diagram above shows how NFS-backed volume mapping exposes the same workspace across the host, SDK container, and DevKit.

This setup provides a seamless workflow for build artifacts:

- Artifacts produced in the SDK are immediately visible on the DevKit without a separate deploy step.
- Agents can access logs, outputs, traces, and other interim files generated while the app runs on the DevKit.
- Developers and agents can inspect the same files from one workspace context.

With Insight, you can view the workspace from a web browser. Some SiMa.ai-specific model archives, such as `*.tar.gz` model artifacts, are automatically optimized for easier inspection.

## Run on the DevKit with `dk`

The SDK includes the `dk` helper, also known as `devkit-run`, to run ARM64 executables on the DevKit from inside the SDK shell.

When you invoke `dk`, the SDK runs the command on the paired DevKit and translates paths so file arguments from the container resolve correctly on the DevKit.

`dk` usage pattern:

<ShellCommand prompt="sdk-container">
dk <file> [args...]
</ShellCommand>

Examples:

Run an ARM64 executable built in the SDK workspace:

<ShellCommand prompt="sdk-container">
dk build/sima_neat_hello
</ShellCommand>

Run a Python entry point on the paired DevKit:

<ShellCommand prompt="sdk-container">
dk hello_neat.py
</ShellCommand>

For Python scripts, `dk` runs the script on the paired DevKit and uses the DevKit PyNeat runtime environment. The SDK remains useful as a unified workspace and orchestration environment, but Python-only workflows do not require the C++ cross-compilation toolchain.

:::note Where `dk` Comes From
`dk` is a shell function defined in `~/devkit-sync.rc` inside the SDK container. The shell loads it through `~/.bashrc`, so it is available in interactive sessions.
:::

## VS Code Workflow

1. Connect VS Code to `ghcr.io-sima-neat-elxr-latest` with [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers). Install the Dev Containers extension if needed.
2. In the container terminal, run the following command once so the SDK can retrieve assets such as prebuilt models:

<ShellCommand prompt="sdk-container">
sima-cli login
</ShellCommand>

3. Install the [Codex Extension](https://marketplace.visualstudio.com/items?itemName=openai.chatgpt) or [Claude Code extension](https://marketplace.visualstudio.com/items?itemName=anthropic.claude-code) in VS Code and sign in.
4. Ask Codex or Claude what skills are available. When `Neat` appears, the environment is ready.

You can then use a prompt such as:

```text
Build a C++ app doing image classification with RESNET50 using the SiMa.ai Neat Library, run it on the DevKit until you get the result.
```

## Upgrade the SDK

To upgrade the SDK to the latest version, rerun the following command from the host:

<ShellCommand prompt="user-host-machine">
sima-cli install ghcr:sima-neat/sdk
</ShellCommand>

To update the Neat Library inside an existing SDK container, run the Neat CLI from the container shell:

<ShellCommand prompt="sdk-container">
neat update
</ShellCommand>

This updates the installed Neat Library components in the current SDK. It does not replace a full container image upgrade when you need SDK-level changes.

If you delete or recreate the SDK container later, run `neat update` again inside the new container. The image may not include the latest Neat Library or `neat-insight` artifacts at the time you recreate it.

For more detail on status checks, JSON output, and component updates, see [Neat CLI](/getting-started/installation/neat-library/#neat-cli).

## Next step

To install or update the library/runtime itself (the same flow for DevKit and the SDK), continue to [Neat Library](/getting-started/installation/neat-library/).
