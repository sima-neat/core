---
title: Install the Environment
description: Install the Neat SDK container
sidebar_position: 2
---

Install the latest Neat Development Environment (referred to as Neat SDK). It
runs as a container on your host machine.

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

- Update your DevKit firmware to the latest with
  [`sima-cli update`](/tools/sima-cli/update/). To stay on an older release, see
  [Compatibility](/getting-started/compatibility/).
- Your host machine and DevKit on the same network with NFS traffic allowed.
- Your DevKit IP address.

## Install

Install the current released Neat SDK:

<ShellCommand prompt="user-host-machine">
sima-cli install ghcr:sima-neat/sdk:v2.1-latest
</ShellCommand>

The first install can take several minutes because it downloads the Neat SDK
container image. The command may not show a progress bar while the download is
in progress.

The `v2.1-latest` tag tracks the latest Neat SDK patch release in the 2.1
series. The current release is Neat SDK 2.1.2.1, which is compatible with
DevKit software 2.1.2.

To install an older release for a DevKit on the 2.0.0 firmware, see
[Compatibility](/getting-started/compatibility/).

## Set Up the Neat SDK Workspace

After the image is installed, run `sima-cli sdk setup` before opening the Neat
SDK shell. Choose the command for your situation and follow the prompts in your
terminal.

If your DevKit is reachable from the host, pair it during setup before opening
the Neat SDK shell:

<ShellCommand prompt="user-host-machine">
sima-cli sdk setup --devkit {devkit-ip}
</ShellCommand>

If your DevKit is not reachable yet, set up the Neat SDK workspace without
pairing:

<ShellCommand prompt="user-host-machine">
sima-cli sdk setup
</ShellCommand>

During setup, `sima-cli` may ask you to:

- select the installed `sdk:v2.1-latest` image if more than one SDK image is present;
- choose a host workspace directory. Accept the default unless you need a
  different workspace;
- choose an SDK extensions directory and whether to install Model Compiler.
  The Model Compiler is required to compile or quantize models yourself and is
  optional only if you exclusively use precompiled model packages. Install it
  here if you plan to compile or quantize models, and complete `sima-cli login`
  if prompted.

With `--devkit`, setup enables DevKit Sync. It exports your host workspace over
NFS and mounts it as `/workspace` on the DevKit by default. Have these ready:

- the DevKit IP address;
- your host administrator password, if NFS needs to be installed or configured;
- the DevKit user credentials when prompted. The default user is `sima` and the
  default password is `edgeai`.

You can add or update DevKit pairing later from
[Pair with a DevKit](/getting-started/dev-environment/pair-with-a-devkit/).

## Open the Neat SDK Shell

After setup succeeds, open the Neat SDK shell with:

<ShellCommand prompt="user-host-machine">
sima-cli sdk neat
</ShellCommand>

DevKit pairing is configured during setup. This command opens the configured
Neat SDK container.

## Install the Model Compiler

The Model Compiler quantizes and compiles ONNX models so they can run on
SiMa.ai's MLA. It is **required** when you compile or quantize models yourself,
including GenAI models, and is **optional** only if you exclusively use
precompiled model packages.

The automatic path is during `sima-cli sdk setup`: `sima-cli` prompts you to
install the matching Model Compiler as an extension inside the Neat SDK. You can
also install it later, either inside the Neat SDK container (below) or standalone
on a supported Ubuntu host (see
[Compatibility](/getting-started/compatibility/#model-compiler)).

If you skip it during setup, install it later from inside the Neat SDK. Run the
command that matches your Neat SDK container architecture.

For `amd64` Neat SDK containers:

<ShellCommand prompt="username@neat-sdk-latest">
sima-cli install -v 2.1.2 tools/model-compiler/amd64
</ShellCommand>

For `arm64` Neat SDK containers:

<ShellCommand prompt="username@neat-sdk-latest">
sima-cli install -v 2.1.2 tools/model-compiler/arm64
</ShellCommand>

After installation, activate the compiler environment from inside the Neat SDK
shell:

<ShellCommand prompt="username@neat-sdk-latest">
activate-model-compiler
</ShellCommand>

To return to the default Neat SDK shell, run:

<ShellCommand prompt="username@neat-sdk-latest">
deactivate-model-compiler
</ShellCommand>

## Upgrade

To reinstall the current released Neat SDK image, rerun the install command
above from the host.

To update the Neat Library inside an existing Neat SDK container, run the Neat
CLI from the container shell:

<ShellCommand prompt="username@neat-sdk-latest">
neat update
</ShellCommand>

This updates the installed Neat Library components in the current Neat SDK. It
does not replace a full container image upgrade when you need container-level
changes.

If you delete or recreate the Neat SDK container later, run `neat update` again
inside the new container.

## Next Step

Continue to [Pair with a DevKit](/getting-started/dev-environment/pair-with-a-devkit/).
