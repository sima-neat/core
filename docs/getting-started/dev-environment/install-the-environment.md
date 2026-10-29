---
title: Install the SDK
description: Install and set up the Neat SDK container
sidebar_position: 2
---

Install the latest Neat SDK. It runs as a container on your host machine.

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

Install the current Neat SDK %%release.sdk_series%% release channel:

<ShellCommand prompt="host">
sima-cli neat install sdk@%%release.sdk_channel%%
</ShellCommand>

The first install can take several minutes because it downloads the Neat SDK
container image. After the image is downloaded, the installer starts SDK setup
and asks whether you want to pair with a Modalix DevKit and whether to install
the matching Model Compiler inside the SDK.

If you choose to pair with a DevKit, enter the DevKit IP address when prompted.
The setup flow configures the SDK workspace, starts the SDK container, and
configures DevKit Sync. If you skip pairing, the SDK workspace is still created
and you can pair later.

The `%%release.sdk_channel%%` package tracks the latest Neat SDK patch release in the %%release.sdk_series%%
series. The current release is Neat SDK %%release.sdk%%, which is compatible with
DevKit software %%release.platform%%.

During setup, `sima-cli` also offers to install the matching Model Compiler
inside the SDK — accept the prompt if you compile or quantize models
yourself; there is no separate version to choose. Skip it if you only run
precompiled model packages. To install it later, pin a specific patch, or use a
standalone host, see
[Install Model Compiler](/getting-started/dev-environment/install-model-compiler/)
and the [Compatibility Guide](/getting-started/compatibility/).

### Install a specific SDK version {#install-a-specific-sdk-version}

For SDK 2.1.3.0 and later, use a published `v<version>` tag to install an exact
version, including an older patch release. For example, run this on the host:

<ShellCommand prompt="host">
sima-cli neat install sdk@v2.1.3.0
</ShellCommand>

The `v` prefix is required. Do not use `release-<full-version>` as an exact
version tag. A `release-<series>` channel tracks the latest published release in
that series. Both selectors run the same installation and SDK setup steps when
they select the same SDK package.

Find published tags in the [SDK releases](https://github.com/sima-neat/sdk/releases)
and choose a version compatible with your DevKit software using the
[Compatibility Guide](/getting-started/compatibility/).

:::note Legacy two-step installation
SDK 2.0.0, 2.1.2.0, and 2.1.2.1 require separate image pull and setup commands.
For those releases only, see
[Two Step SDK Installation](/reference/two-step-sdk-installation/).
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

<ShellCommand prompt="host">
sima-cli sdk neat
</ShellCommand>

### Use Chrome Browser

Neat Insight is served from inside the SDK and can be opened from a browser.
From inside the SDK shell, run:

<ShellCommand prompt="sdk">
neat
</ShellCommand>

The `neat` output includes a **Web Access** section with local and remote URLs for Insight and browser-based VS Code. Open the local `127.0.0.1` URL when the browser runs on the SDK host. From another computer, use the remote URL based on `NFS_SERVER_HOST_IP`. The VS Code URL opens the configured workspace; use the credentials shown by `neat` for the installed server. See [SDK web access](/reference/environment-variables/) and [Insight](/tools/insight/).

### Use VS Code

Run `neat` inside the SDK shell and open the **VS Code (local)** URL in a browser on the SDK host. Use **VS Code (remote)** from another computer. For images with code-server, `neat` also displays **VS Code password**:

```text
VS Code (local)    https://127.0.0.1:10000/?folder=/workspace
VS Code password   <password shown by neat>
```

Enter the displayed password at the code-server sign-in prompt. The password is not included in the URL.

Legacy OpenVSCode images use a token URL instead; open the URL printed by `neat`. The SDK preinstalls the Codex and Claude Code extensions in the browser Code UI.

Native VS Code is another option if you prefer it instead of the browser. Connect
VS Code to the SDK container with
[Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).

Run `sima-cli login` once inside the SDK container so the SDK can retrieve assets
such as prebuilt models.

## Upgrade

To reinstall or upgrade to the current SDK package, rerun the install command
above from the host:

<ShellCommand prompt="host">
sima-cli neat install sdk@%%release.sdk_channel%%
</ShellCommand>

To update the Neat Library inside an existing Neat SDK container, run the Neat
CLI from the container shell:

<ShellCommand prompt="sdk">
neat update
</ShellCommand>

This updates the installed Neat Library components in the current Neat SDK. It
does not replace a full container image upgrade when you need container-level
changes.

If you delete or recreate the Neat SDK container later, run `neat update` again
inside the new container.

## Uninstall

To remove an installed SDK container, run:

<ShellCommand prompt="host">
sima-cli sdk remove
</ShellCommand>

## Next Step

- **Compiling a model in the SDK?** Continue to
  [Compile a Model](/compile-a-model/) — this runs entirely in the SDK, with no
  DevKit required.
- **Pairing a DevKit?** Set up
  [DevKit Sync](/getting-started/dev-environment/devkit-sync/) to share your SDK
  workspace and run `dk` commands on the hardware.
