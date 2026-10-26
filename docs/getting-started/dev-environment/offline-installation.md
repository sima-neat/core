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

<ShellCommand prompt="host">
sima-cli neat install sdk@v2.1.3.0 -t offline-amd64
</ShellCommand>

For `arm64` hosts:

<ShellCommand prompt="host">
sima-cli neat install sdk@v2.1.3.0 -t offline-arm64
</ShellCommand>

Copy the downloaded directory to the target host. From that directory, run:

<ShellCommand prompt="host">
bash ./install_offline_sdk.sh
</ShellCommand>

:::note
SDK offline packages are supported for SDK 2.1.3.0 or later.
:::

## Download the Model Compiler Offline Package

Download the Model Compiler package that matches the target environment and SDK
compatibility requirements. For compatibility details, see
[Compatibility](/getting-started/compatibility/#model-compiler).

For Model Compiler 2.1.3 on `amd64` hosts:

<ShellCommand prompt="host">
sima-cli neat install model-compiler/amd64@v2.1.3 -t offline
</ShellCommand>

For Model Compiler 2.1.3 on `arm64` hosts:

<ShellCommand prompt="host">
sima-cli neat install model-compiler/arm64@v2.1.3 -t offline
</ShellCommand>

:::note
Model Compiler offline packages are supported for Model Compiler 2.1.3 or later.
:::

To install Model Compiler inside the Neat SDK, copy the downloaded directory into
the host workspace folder that is mapped to the SDK container's `/workspace`
folder. Then open the SDK shell and run the installer from the matching
`/workspace` path:

<ShellCommand prompt="sdk">
cd /workspace/model-compiler-offline-amd64
bash ./install_modelsdk_wheels.sh
</ShellCommand>

Use the `arm64` directory name instead if you downloaded the ARM64 package.

For standalone host installation, copy the downloaded directory directly to the
target host and run the same installer from that directory.

After installation, reload your shell environment or restart the SDK shell. Then
activate Model Compiler with:

<ShellCommand prompt="sdk|host">
activate-model-compiler
</ShellCommand>

To leave the Model Compiler environment, run:

<ShellCommand prompt="sdk|host">
deactivate-model-compiler
</ShellCommand>

## Hosting Packages Internally

If your organization mirrors approved artifacts internally, keep the downloaded
package directory intact when publishing it to the internal location. The
metadata file, install script, checksums, and package resources are expected to
remain together.

Users can then download the internal package directory and run the same install
script on the target host.

## Offline Core packages

Download on a connected computer, then copy the archives to the provisioned Modalix DevKit. The default profile includes the runtime bundle and tutorials/tests extras. Use `-t minimal` to omit extras. The target installer checks OS, architecture, and Platform compatibility.

```bash
sima-cli neat install core/offline@v0.6.0 --download-only
sima-cli neat install core/offline@v0.6.0 -t minimal --download-only
```

Extract `neat-core-offline-installer.tar.gz`, then run `cd neat-core-offline && ./install.sh`. Extract the separate extras archive in a directory of your choice when you need tutorials or tests.

The installer accepts only the Neat eLxr SDK or a provisioned Modalix DevKit. It checks `/etc/sdk-release` or `/etc/buildinfo` against the package manifest bundled in the archive; downloaded publication metadata is not required. Other environments and incompatible Platform versions are rejected before installation. In the SDK, Core is copied into `/opt/toolchain/aarch64/modalix`. If `DEVKIT_SYNC_DEVKIT_IP` is configured, the same offline bundle is transferred over SSH to the paired DevKit. Both environments must pass preflight before installation begins. On a DevKit, installation runs locally.

Inside the standard SDK container, the offline installer does not require `unshare`, privileged mode, or additional container capabilities. It extracts packages into the sysroot and enables only local APT `file`, `copy`, and `store` methods. HTTP, HTTPS, and other network transports are disabled; missing dependencies fail without online fallback. Native DevKit installation retains its isolated network namespace.

The native installer uses the offline APT transaction plan to reserve three times the expanded contents of Debian packages selected for installation or reinstallation and all Python wheels, plus 128 MiB for metadata and the Python environment. Dependencies left untouched by APT do not consume this installation budget. It does not count already-staged archives again, including bundles on NVMe or tmpfs. A disk-space failure reports available, required, and additional root space in MiB.

Paired DevKit transfers use a private directory under `/tmp`, which is RAM-backed on the standard DevKit. The staging directory is removed after the operation finishes, including reported installation failures and incomplete transfers. If SSH disconnects or times out during installation, the directory is retained and its path is printed because the remote installer may still be running.

Run `./install.sh` as your regular SDK or DevKit user. The launcher requests `sudo` privileges automatically and preserves only the explicit pairing, DevKit authentication, sync preference, and SSH-agent settings. Paired deployment continues to use the invoking SDK user’s SSH identity.
