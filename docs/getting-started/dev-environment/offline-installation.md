---
title: Offline Installation
description: Install SDK, Model Compiler, and PCIe packages without internet access on the target host
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

## Select a package

| Package | Use | Availability |
| --- | --- | --- |
| Neat SDK | Set up the SDK development environment. | SDK 2.1.3.0 or later |
| Model Compiler | Install Model Compiler in the SDK or on a standalone host. | Model Compiler 2.1.3 or later |
| Neat PCIe | Install Neat on a prepared PCIe host and all attached compatible cards. | 0.6.0 or later |
| Neat Library | Install the core package without internet access. | 0.6.0 or later |

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

## Install Neat for PCIe

The PCIe package installs Neat on the host and all attached compatible cards.
The target system does not need Docker or internet access for this installation.
The host must have a working PCIe network connection to each card.

### 1. Prepare the host and cards

Complete these tasks before you run the Neat installer:

- Install the compatible Platform host driver.
- Install the matching Platform 3.0 firmware on each card.
- Configure discovery on each card's PCIe interface.
- Assign a unique PCIe IPv4 address to each card.
- Configure SSH access and trusted SSH host keys for each card.
- Give the SSH account permission to use `sudo` on each card.

The Neat installer does not install the Platform driver or change the Platform firmware.
Use the system setup instructions supplied with your Platform release.

### 2. Download the package

Select version 0.6.0 or later for the target host.
Packages support Ubuntu 22.04 and 24.04 hosts with `amd64` or `arm64` processors.

This example downloads version 0.6.0 on the computer with internet access.
For an Ubuntu 24.04 `amd64` target host, run:

<ShellCommand prompt="host">
sima-cli neat install 'core/pciehost/ubuntu24/amd64@v0.6.0' -t offline -d ./pcie-offline
</ShellCommand>

For Ubuntu 22.04, replace `ubuntu24` with `ubuntu22`.
For an `arm64` host, replace `amd64` with `arm64`.
The command downloads and extracts the package. It does not install Neat.

### 3. Check the system

1. Copy the complete `pcie-offline` directory to the target host. Use a path without spaces.
2. Open the extracted `neat-offline` directory that contains `install.py`.
3. Read the supplied `INSTALL.md` for required system tools and SSH settings.
4. Run the checks:

<ShellCommand prompt="host">
sudo python3 -B install.py --preflight-only
</ShellCommand>

The installer discovers all attached cards automatically. Do not supply card IP addresses.
The checks create temporary files but do not install packages.
Correct any reported errors before installation.

### 4. Install and check the results

Run this command from the same directory:

<ShellCommand prompt="host">
sudo python3 -B install.py
</ShellCommand>

The installer checks all cards before it installs packages.
It installs and verifies the host first, then each card in sequence.
Check that the final summary reports successful installation and verification for the host and every card.
Then run your application acceptance test.

If installation fails, read the diagnostic log at the printed path.
If a card connection fails during installation, check that the card's installation process has stopped before you try again.
The installer does not automatically restore the previous installation after a failure.

## Install Neat Library

Neat Library offline packages are available in version 0.6.0 or later.
Use this package for a Modalix DevKit with Platform 3.0 or later.

### 1. Download the package

Run this command on the computer with internet access:

<ShellCommand prompt="host">
sima-cli neat install core@v0.6.0 -t offline --download-only
</ShellCommand>

The default package includes the runtime, tutorials, and tests.
To download only the runtime, use:

<ShellCommand prompt="host">
sima-cli neat install core/offline@v0.6.0 -t minimal --download-only
</ShellCommand>

### 2. Install the package

1. Copy the downloaded files to the Modalix DevKit or the Neat SDK workspace.
2. Open a terminal in that directory. For SDK installation, use the SDK shell.
3. Extract the installer archive:

<ShellCommand prompt="sdk|devkit">
tar -xzf neat-core-offline-installer.tar.gz
</ShellCommand>

4. Go to the extracted directory:

<ShellCommand prompt="sdk|devkit">
cd neat-core-offline
</ShellCommand>

5. Start the installer as your regular user. It requests `sudo` permissions when necessary.

<ShellCommand prompt="sdk|devkit">
./install.sh
</ShellCommand>

If you need tutorials or tests, extract the separate extras archive into a directory of your choice.

### Package options and installer behavior

- The installer checks system compatibility and available disk space before installation.
- The installer uses bundled dependencies. It stops if a required dependency is missing.
- In the SDK, the installer copies Neat Library into `/opt/toolchain/aarch64/modalix`.
- If `DEVKIT_SYNC_DEVKIT_IP` is set, the installer also installs on the paired DevKit through SSH. Both environments must pass checks.

If installation fails, read the log at the printed path: `/tmp/neat-core-offline-*.log`.
For detailed output, use `./install.sh --verbose`.

If SSH disconnects during installation, the remote installer can continue to run.
Check that it has stopped before you retry or remove the temporary directory shown in the report.

## Store packages internally {#hosting-packages-internally}

You can store approved packages on an internal file server.
Keep each downloaded directory complete and separate from other package versions.
Transfer the complete directory to the target host before you run its installer.
