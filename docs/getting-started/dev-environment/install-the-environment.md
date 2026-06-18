---
title: Install the Environment
description: Install the Neat Development Environment container
sidebar_position: 2
---

Install the Neat Development Environment version that matches your DevKit
software and Neat Library version.

## Prerequisites

- Confirm your host meets the
  [host requirements](/getting-started/dev-environment/#host-requirements),
  including administrator/`sudo` access for installation.
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

Install the current released Neat Development Environment:

<ShellCommand prompt="user-host-machine">
sima-cli install ghcr:sima-neat/sdk:v2.1.2
</ShellCommand>

The first install can take several minutes because it downloads the Neat
Development Environment container image. The command may not show a progress bar
while the download is in progress.

Use an older Neat Development Environment only when your DevKit software is from
that release family. For Neat Development Environment 2.0.0:

<ShellCommand prompt="user-host-machine">
sima-cli install ghcr:sima-neat/sdk:v2.0.0
</ShellCommand>

## Set Up the Neat Development Environment Workspace

After the image is installed, run `sima-cli sdk setup` before opening the Neat
Development Environment shell. Choose the command for your situation and follow
the prompts in your terminal.

If your DevKit is reachable from the host, pair it during setup before opening
the Neat Development Environment shell:

<ShellCommand prompt="user-host-machine">
sima-cli sdk setup --devkit {devkit-ip}
</ShellCommand>

If your DevKit is not reachable yet, set up the Neat Development Environment
workspace without pairing:

<ShellCommand prompt="user-host-machine">
sima-cli sdk setup
</ShellCommand>

During setup, `sima-cli` may ask you to:

- select the installed `sdk:v2.1.2` image if more than one SDK image is present;
- choose a host workspace directory. Accept the default unless you need a
  different workspace;
- choose an SDK extensions directory and whether to install Model Compiler.
  Install it if you plan to quantize or compile models, and complete
  `sima-cli login` if prompted.

With `--devkit`, setup enables DevKit Sync. It exports your host workspace over
NFS and mounts it as `/workspace` on the DevKit by default. Have these ready:

- the DevKit IP address;
- your host administrator password, if NFS needs to be installed or configured;
- the DevKit user credentials when prompted. The default user is `sima` and the
  default password is `edgeai`.

You can add or update DevKit pairing later from
[Pair with a DevKit](/getting-started/dev-environment/pair-with-a-devkit/).

## Open the Neat Development Environment Shell

After setup succeeds, open the Neat Development Environment shell with:

<ShellCommand prompt="user-host-machine">
sima-cli sdk neat
</ShellCommand>

DevKit pairing is configured during setup. This command opens the configured
Neat Development Environment container.

## Install the Model Compiler

During `sima-cli sdk setup`, `sima-cli` can prompt you to install the matching
Model Compiler as an extension inside the Neat Development Environment.

The Model Compiler quantizes and compiles ONNX models so they can run on
SiMa.ai's MLA. Install it during setup if you plan to quantize or compile
models.

If you skip it during setup, install it later from inside the Neat Development
Environment. Run the command that matches your Neat Development Environment
container architecture.

For `amd64` Neat Development Environment containers:

<ShellCommand prompt="sdk-container">
sima-cli install -v 2.1.2 tools/model-compiler/amd64
</ShellCommand>

For `arm64` Neat Development Environment containers:

<ShellCommand prompt="sdk-container">
sima-cli install -v 2.1.2 tools/model-compiler/arm64
</ShellCommand>

After installation, activate the compiler environment from inside the Neat
Development Environment shell:

<ShellCommand prompt="sdk-container">
activate-model-compiler
</ShellCommand>

To return to the default Neat Development Environment shell, run:

<ShellCommand prompt="sdk-container">
deactivate-model-compiler
</ShellCommand>

## Upgrade

To reinstall the current released Neat Development Environment image, rerun the
install command above from the host.

To update the Neat Library inside an existing Neat Development Environment
container, run the Neat CLI from the container shell:

<ShellCommand prompt="sdk-container">
neat update
</ShellCommand>

This updates the installed Neat Library components in the current Neat
Development Environment. It does not replace a full container image upgrade when
you need container-level changes.

If you delete or recreate the Neat Development Environment container later, run
`neat update` again inside the new container.

## Next Step

Continue to [Pair with a DevKit](/getting-started/dev-environment/pair-with-a-devkit/).
