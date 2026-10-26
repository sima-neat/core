---
title: Offline Installation
description: Install SDK and Model Compiler packages without internet access on the target host
sidebar_position: 6
---

:::tip When to use offline installation
Use offline installation when the target host cannot download packages from SiMa.ai services.
This can occur when your organization:

- Blocks large downloads.
- Restricts access to package registries.
- Requires approval before you can use downloaded packages.
- Keeps development hosts separate from computers with internet access.
:::

Download the package on a computer with internet access.
Then transfer the package to the target host for installation.

For runtime packages, see [PCIe host offline installation](/getting-started/neat-library/pcie-host/#offline-installation) or [Neat Library offline installation](/getting-started/neat-library/install-or-update/#offline-installation).

## Select a package

| Package | Use | Availability |
| --- | --- | --- |
| Neat SDK | Set up the SDK development environment. | SDK 2.1.3.0 or later |
| Model Compiler | Install Model Compiler in the SDK or on a standalone host. | Model Compiler 2.1.3 or later |

## Before you start

- Install `sima-cli` on the computer with internet access. See [Install the Environment](/getting-started/dev-environment/install-the-environment/).
- Select a package version that is compatible with your system. See [Compatibility](/getting-started/compatibility/).
- Identify the operating system and processor architecture of the target host.
- Prepare a USB drive or an internal file share for the transfer.

The download computer and the target host can have different processor architectures.
Select the package for the **target host**.
Keep all package files together, including scripts, metadata, checksums, and dependencies.

## Install the Neat SDK {#download-the-sdk-offline-package}

### 1. Download the package

Run one command on the computer with internet access.
These examples use SDK 2.1.3.0.

For an `amd64` target host:

<ShellCommand prompt="host">
sima-cli neat install sdk@v2.1.3.0 -t offline-amd64
</ShellCommand>

For an `arm64` target host:

<ShellCommand prompt="host">
sima-cli neat install sdk@v2.1.3.0 -t offline-arm64
</ShellCommand>

### 2. Install the package

To install Model Compiler with the SDK, put the Model Compiler offline package file in the same directory as the SDK offline package.
The SDK installer finds the file automatically.
When prompted, select installation from the local package.

1. Copy the complete downloaded directory to the target host.
2. Open a terminal in that directory.
3. Run the installer:

<ShellCommand prompt="host">
bash ./install_offline_sdk.sh
</ShellCommand>

## Install Model Compiler {#download-the-model-compiler-offline-package}

### 1. Download the package

Check the [Model Compiler compatibility requirements](/getting-started/compatibility/#model-compiler).
Run one command on the computer with internet access.
These examples use Model Compiler 2.1.3.

For an `amd64` target host:

<ShellCommand prompt="host">
sima-cli neat install model-compiler/amd64@v2.1.3 -t offline
</ShellCommand>

For an `arm64` target host:

<ShellCommand prompt="host">
sima-cli neat install model-compiler/arm64@v2.1.3 -t offline
</ShellCommand>

### 2. Install the package

Choose one installation location.

**In the Neat SDK:**

1. Copy the complete downloaded directory to the host workspace mapped to `/workspace` in the SDK container.
2. Open the SDK shell.
3. Go to the package directory. For an `amd64` package, use:

<ShellCommand prompt="sdk">
cd /workspace/model-compiler-offline-amd64
</ShellCommand>

For an `arm64` package, use `/workspace/model-compiler-offline-arm64`.

4. Run the installer:

<ShellCommand prompt="sdk">
bash ./install_modelsdk_wheels.sh
</ShellCommand>

**On a standalone host:**

1. Copy the complete downloaded directory to the target host.
2. Open a terminal in that directory.
3. Run the installer:

<ShellCommand prompt="host">
bash ./install_modelsdk_wheels.sh
</ShellCommand>

### 3. Activate Model Compiler

1. Restart the SDK shell or the host shell that you used for installation.
2. Activate Model Compiler:

<ShellCommand prompt="sdk|host">
activate-model-compiler
</ShellCommand>

To leave the Model Compiler environment, run:

<ShellCommand prompt="sdk|host">
deactivate-model-compiler
</ShellCommand>

## Store packages internally {#hosting-packages-internally}

You can store approved packages on an internal file server.
Keep each downloaded directory complete and separate from other package versions.
Transfer the complete directory to the target host before you run its installer.
