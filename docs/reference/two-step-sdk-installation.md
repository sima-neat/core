---
title: Two Step SDK Installation
description: Legacy two-step installation flow for SDK 2.0.0, 2.1.2, and 2.1.2.1
sidebar_position: 20
---

Use this reference only for SDK releases that require a separate image pull and
setup command. This applies to SDK 2.0.0, SDK 2.1.2.0, and SDK 2.1.2.1. Newer
SDK releases use the simplified package install flow documented in
[Install the Environment](/getting-started/dev-environment/install-the-environment/).

The legacy SDK installation has two steps:

1. Pull the SDK container image.
2. Run SDK setup to create the workspace and, optionally, pair with a DevKit.

## Pull the SDK Image

Run the image install command that matches your SDK release from the host.

For SDK 2.0.0:

<ShellCommand prompt="user-host-machine">
sima-cli install ghcr:sima-neat/sdk:v2.0.0
</ShellCommand>

For SDK 2.1.2 or 2.1.2.1, use the matching 2.1 image tag supplied with that
release.

The first install can take several minutes because it downloads the SDK
container image.

## Run SDK Setup

If your DevKit is reachable from the host, pair it during setup:

<ShellCommand prompt="user-host-machine">
sima-cli sdk setup --devkit {devkit-ip}
</ShellCommand>

If your DevKit is not reachable yet, set up the SDK workspace without pairing:

<ShellCommand prompt="user-host-machine">
sima-cli sdk setup
</ShellCommand>

During setup, `sima-cli` may ask you to choose the installed SDK image, choose a
host workspace directory, and configure the SDK extensions directory. With
`--devkit`, setup also asks for DevKit connection details and configures DevKit
Sync.

## Open the SDK Shell

After setup succeeds, open the SDK shell:

<ShellCommand prompt="user-host-machine">
sima-cli sdk neat
</ShellCommand>

## Compatibility

SDK 2.0.0 is intended for DevKit software 2.0.0. SDK 2.1.2 and 2.1.2.1 are
intended for DevKit software 2.1.2. For current compatibility guidance, see
[Compatibility](/getting-started/compatibility/).
