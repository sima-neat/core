---
title: Install the Environment
description: Install the Neat Development Environment SDK container
sidebar_position: 2
---

Install the SDK version that matches your DevKit software and Neat Library
version.

## Prerequisites

- Install `sima-cli` on your host machine using the
  [sima-cli installation guide](/tools/sima-cli/).
- Complete any host-specific setup for your platform:
  - [Ubuntu Host Notes](/reference/elxr-sdk-host-setup/ubuntu)
  - [Windows Host Notes](/reference/elxr-sdk-host-setup/windows)
  - [macOS Host Notes](/reference/elxr-sdk-host-setup/macos)

To use DevKit Sync later, you also need:

- A Modalix DevKit running a compatible DevKit software version.
- Your host machine and DevKit on the same network with NFS traffic allowed.
- Your DevKit IP address.

## Install

For Neat Development Environment 2.1.2:

<ShellCommand prompt="user-host-machine">
sima-cli install ghcr:sima-neat/sdk:v2.1.2
</ShellCommand>

For Neat Development Environment 2.0.0:

<ShellCommand prompt="user-host-machine">
sima-cli install ghcr:sima-neat/sdk:v2.0.0
</ShellCommand>

To install the latest patch release for a supported release branch, use the
branch's `latest` tag. For example, to install the latest 2.0 release:

<ShellCommand prompt="user-host-machine">
sima-cli install ghcr:sima-neat/sdk:v2.0-latest
</ShellCommand>

You can also install `ghcr:sima-neat/sdk` to use the latest SDK image from the
main branch of the open source repository.

## Set Up the SDK Workspace

After the image is installed, run SDK setup before opening the SDK shell. Setup
creates the workspace mapping used by the container and can also pair the SDK
with your DevKit.

If your DevKit is reachable from the host, pair it during setup:

<ShellCommand prompt="user-host-machine">
sima-cli sdk setup --devkit {devkit-ip}
</ShellCommand>

Replace `{devkit-ip}` with your DevKit's IP address.

If your DevKit is not reachable yet, set up the SDK workspace without pairing:

<ShellCommand prompt="user-host-machine">
sima-cli sdk setup
</ShellCommand>

You can add or update DevKit pairing later from
[Pair with a DevKit](/getting-started/dev-environment/pair-with-a-devkit/).

## Open the SDK Shell

After SDK setup succeeds, open the SDK shell with:

<ShellCommand prompt="user-host-machine">
sima-cli sdk neat
</ShellCommand>

## Install the Model Compiler

During SDK setup, `sima-cli` can prompt you to install the matching Model
Compiler.

The Model Compiler quantizes and compiles ONNX models so they can run on
SiMa.ai's MLA. To install it, opt in when prompted during SDK setup.

Install the Model Compiler later with the command that matches your host
architecture.

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

Model Compiler support is available on Ubuntu and Windows through WSL. On macOS,
use Model Compiler 2.1.2 or above.

## Upgrade

To upgrade the SDK to the latest version, rerun the following command from the
host:

<ShellCommand prompt="user-host-machine">
sima-cli install ghcr:sima-neat/sdk
</ShellCommand>

To update the Neat Library inside an existing SDK container, run the Neat CLI
from the container shell:

<ShellCommand prompt="sdk-container">
neat update
</ShellCommand>

This updates the installed Neat Library components in the current SDK. It does
not replace a full container image upgrade when you need SDK-level changes.

If you delete or recreate the SDK container later, run `neat update` again
inside the new container.

## Next Step

Continue to [Pair with a DevKit](/getting-started/dev-environment/pair-with-a-devkit/).
