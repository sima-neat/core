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

### 2. Install the package

1. Copy the complete downloaded directory to the target host.
2. Open a terminal in that directory.
3. Use the installation instructions supplied with the package.

### Package options and installer behavior

Download on a connected computer, then copy the archives to the provisioned Modalix DevKit. The default profile includes the runtime bundle and tutorials/tests extras. Use `-t minimal` to omit extras. The target installer checks OS, architecture, and Platform compatibility.

```bash
sima-cli neat install core/offline@v0.6.0 --download-only
sima-cli neat install core/offline@v0.6.0 -t minimal --download-only
```

Extract `neat-core-offline-installer.tar.gz`, then run `cd neat-core-offline && ./install.sh`. Extract the separate extras archive in a directory of your choice when you need tutorials or tests.

The installer accepts only the Neat eLxr SDK or a provisioned Modalix DevKit. It checks `/etc/sdk-release` or `/etc/buildinfo` against the package manifest bundled in the archive; downloaded publication metadata is not required. Other environments and incompatible Platform versions are rejected before installation. In the SDK, Neat Library is copied into `/opt/toolchain/aarch64/modalix`. If `DEVKIT_SYNC_DEVKIT_IP` is configured, the same offline bundle is transferred over SSH to the paired DevKit. Both environments must pass preflight before installation begins. On a DevKit, installation runs locally.

Inside the standard SDK container, the offline installer does not require `unshare`, privileged mode, or additional container capabilities. It extracts packages into the sysroot and enables only local APT `file`, `copy`, and `store` methods. HTTP, HTTPS, and other network transports are disabled; missing dependencies fail without online fallback. Native DevKit installation retains its isolated network namespace.

The native installer uses the offline APT transaction plan to reserve three times the expanded contents of Debian packages selected for installation or reinstallation and all Python wheels, plus 128 MiB for metadata and the Python environment. Dependencies left untouched by APT do not consume this installation budget. It does not count already-staged archives again, including bundles on NVMe or tmpfs. A disk-space failure reports available, required, and additional root space in MiB.

Paired DevKit transfers use a private directory under `/tmp`, which is RAM-backed on the standard DevKit. The staging directory is removed after the operation finishes, including reported installation failures and incomplete transfers. If SSH disconnects or times out during installation, the directory is retained and its path is printed because the remote installer may still be running.

Run `./install.sh` as your regular SDK or DevKit user. The launcher requests `sudo` privileges automatically and preserves only the explicit pairing, DevKit authentication, sync preference, and SSH-agent settings. Paired deployment continues to use the invoking SDK user’s SSH identity.

Standalone Neat Library and PCIe installation use the same live terminal renderer. Each standalone step keeps its numbered phase and elapsed time, with up to 10 recent command-output lines refreshed every second and clipped to the terminal size. Redirected output uses plain 15-second activity updates. Detailed reports remain in a private `/tmp/neat-core-offline-*.log` file, including on failure. Use `NO_COLOR=1 ./install.sh` to disable colors or `./install.sh --verbose` to stream the full log. Repeated compatibility checks protect the package transaction; they are separate from installation.

## Store packages internally {#hosting-packages-internally}

You can store approved packages on an internal file server.
Keep each downloaded directory complete and separate from other package versions.
Transfer the complete directory to the target host before you run its installer.
