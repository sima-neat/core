---
title: Install the Environment
description: Install and set up the Neat SDK container
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

- A DevKit running compatible software to the Neat SDK. Read the [Compatibility Guide](/getting-started/compatibility/) for more information.
- Your host machine and DevKit on the same network with NFS traffic allowed.
- Your DevKit IP address.

## Install

Install the current Neat SDK 2.1 release channel:

<ShellCommand prompt="user-host-machine">
sima-cli neat install sdk@release-2.1
</ShellCommand>

The first install can take several minutes because it downloads the Neat SDK
container image. After the image is downloaded, the installer starts SDK setup
and asks whether you want to pair with a Modalix DevKit and whether to install
the matching Model Compiler inside the SDK.

If you choose to pair with a DevKit, enter the DevKit IP address when prompted.
The setup flow configures the SDK workspace, starts the SDK container, and
configures DevKit Sync. If you skip pairing, the SDK workspace is still created
and you can pair later.

The `release-2.1` package tracks the latest Neat SDK patch release in the 2.1
series. The current release is Neat SDK 2.1.2.2, which is compatible with
DevKit software 2.1.2.

During setup, `sima-cli` also offers to install the matching Model Compiler
(2.1.2) inside the SDK — accept the prompt if you compile or quantize models
yourself; there is no separate version to choose. Skip it if you only run
precompiled model packages. To install it later, pin a specific patch, or use a
standalone host, see
[Install Model Compiler](/getting-started/dev-environment/install-model-compiler/)
and the [Compatibility Guide](/getting-started/compatibility/).

:::note Older SDK releases use the legacy two-step install flow
For SDK 2.0.0, 2.1.2.0, or 2.1.2.1, install with the legacy image pull and setup
commands. See [Two Step SDK Installation](/reference/two-step-sdk-installation/).
:::

To change SDK settings after installation, see
[Configure SDK](/getting-started/dev-environment/configure-sdk/). For restricted
network environments, see
[Offline Installation](/getting-started/dev-environment/offline-installation/).

## Access the SDK

After setup succeeds, you can access the SDK from a terminal, Chrome browser,
or VS Code.

### Use the SDK Shell

Open the Neat SDK shell with:

<ShellCommand prompt="user-host-machine">
sima-cli sdk neat
</ShellCommand>

### Use Chrome Browser

Neat Insight is served from inside the SDK and can be opened from a browser.
From inside the SDK shell, run:

<ShellCommand prompt="username@neat-sdk-latest">
neat
</ShellCommand>

The command output includes the Insight URL. Open that URL in Chrome to inspect
workspace files, media sources, stream delivery, and runtime behavior. On a
local host, the URL is typically `https://localhost:9900`. From another machine
on the network, use the host IP address shown by the SDK environment. For more
information, see [Insight](/tools/insight/).

### Use VS Code

Starting with SDK 2.1.2.3, you can access VS Code from a browser through the
SDK Code UI. At the end of SDK installation, `sima-cli` prints the Code UI URL
as `codeUI`, for example:

<ShellCommand prompt="user-host-machine">
codeUI      | https://192.168.76.4:10000/?tkn=gA5CS...&folder=/workspace
</ShellCommand>

Open that URL in your browser to work inside the SDK workspace. SDK 2.1.2.3 and
later preinstall the Codex and Claude Code extensions in the browser Code UI.

Native VS Code is another option if you prefer it instead of the browser. Connect
VS Code to the SDK container with
[Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).

Run `sima-cli login` once inside the SDK container so the SDK can retrieve assets
such as prebuilt models.

## Upgrade

To reinstall or upgrade to the current SDK package, rerun the install command
above from the host:

<ShellCommand prompt="user-host-machine">
sima-cli neat install sdk@release-2.1
</ShellCommand>

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

## Uninstall

To remove an installed SDK container, run:

<ShellCommand prompt="user-host-machine">
sima-cli sdk remove
</ShellCommand>

## Next Step

Continue to [DevKit Sync](/getting-started/dev-environment/devkit-sync/) to
learn how SDK workspace sharing and `dk` command execution work.
