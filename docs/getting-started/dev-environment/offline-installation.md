---
title: Offline Installation
description: Download SDK and Model Compiler packages for restricted network environments
sidebar_position: 6
---

:::tip When to use offline installation
Use offline installation when the target machine cannot download packages
directly from SiMa.ai services. This is common on corporate networks that block
large downloads, restrict registry access, require manual artifact approval, or
separate internet-connected machines from development hosts.
:::

In this workflow, use an internet-connected machine to retrieve the installation
package, then move the downloaded directory to the target host with a USB drive,
internal file share, or an internally hosted package location.

## Download the SDK Offline Package

Run the command for the target host architecture on a machine that can access
SiMa.ai package services.

For `amd64` hosts:

<ShellCommand prompt="online-machine">
sima-cli neat install sdk@v2.1.2.3 -t offline-amd64
</ShellCommand>

For `arm64` hosts:

<ShellCommand prompt="online-machine">
sima-cli neat install sdk@v2.1.2.3 -t offline-arm64
</ShellCommand>

Copy the downloaded directory to the target host. From that directory, run:

<ShellCommand prompt="offline-host">
bash ./install_offline_sdk.sh
</ShellCommand>

:::note
SDK offline packages are supported for SDK 2.1.2.3 or later.
:::

## Download the Model Compiler Offline Package

Download the Model Compiler package that matches the target environment and SDK
compatibility requirements. For compatibility details, see
[Compatibility](/getting-started/compatibility/#model-compiler).

For Model Compiler 2.1.2 on `amd64` hosts:

<ShellCommand prompt="online-machine">
sima-cli install -v 2.1.2 tools/model-compiler/amd64 -t offline -d ./model-compiler-offline-amd64
</ShellCommand>

For Model Compiler 2.1.2 on `arm64` hosts:

<ShellCommand prompt="online-machine">
sima-cli install -v 2.1.2 tools/model-compiler/arm64 -t offline -d ./model-compiler-offline-arm64
</ShellCommand>

:::note
Model Compiler offline packages are supported for Model Compiler 2.1.2 or later.
:::

To install Model Compiler inside the Neat SDK, copy the downloaded directory into
the host workspace folder that is mapped to the SDK container's `/workspace`
folder. Then open the SDK shell and run the installer from the matching
`/workspace` path:

<ShellCommand prompt="username@neat-sdk-latest">
cd /workspace/model-compiler-offline-amd64
bash ./install_modelsdk_wheels.sh
</ShellCommand>

Use the `arm64` directory name instead if you downloaded the ARM64 package.

For standalone host installation, copy the downloaded directory directly to the
target host and run the same installer from that directory.

After installation, reload your shell environment or restart the SDK shell. Then
activate Model Compiler with:

<ShellCommand prompt="offline-host">
activate-model-compiler
</ShellCommand>

To leave the Model Compiler environment, run:

<ShellCommand prompt="offline-host">
deactivate-model-compiler
</ShellCommand>

## Hosting Packages Internally

If your organization mirrors approved artifacts internally, keep the downloaded
package directory intact when publishing it to the internal location. The
metadata file, install script, checksums, and package resources are expected to
remain together.

Users can then download the internal package directory and run the same install
script on the target host.
