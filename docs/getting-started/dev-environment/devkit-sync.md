---
title: DevKit Sync
description: Share an SDK workspace with a Modalix DevKit and run commands on hardware
sidebar_position: 5
---

:::tip Reference only
DevKit Sync setup is optional during SDK installation, and the installer already
prompts you for it. Use this page as a reference for how DevKit Sync works, or
when you want to change the SDK pairing to a different DevKit later.
:::

DevKit Sync connects the Neat Development Environment (referred to as Neat SDK)
with a Modalix DevKit on the same network. It exposes one shared workspace
across the host, Neat SDK container, and DevKit, and it provides the `dk` helper
for running SDK-built commands on hardware.

![Host-Container-DevKit workspace mapping](../../images/elxr-sdk-workspaces.svg)

The same workspace is mounted into the Neat SDK container
and the DevKit as `/workspace`, so build artifacts, logs, traces, and model
files are visible from each environment.

## Configure DevKit Sync

If you skipped DevKit pairing during installation or need to change it later,
run this setup command from the host:

<ShellCommand prompt="user-host-machine">
sima-cli sdk setup --devkit {devkit-ip}
</ShellCommand>

During setup:

- Select the installed `sdk:v2.1-latest` image if more than one SDK image is present.
- Accept the default `/workspace` DevKit mount path unless you need a different
  path.
- Enter the host admin password when prompted for NFS server setup on the host.
- Enter the DevKit user credentials when prompted. The default user is `sima`
  and the default password is `edgeai`.
- Starting with SDK 2.1.2.2, setup can fall back to rsync-over-SSH when
  NFS cannot be configured between the host and DevKit.

When setup succeeds, you should see output similar to:

```text
============================================================
  DevKit Connected
============================================================
  DevKit target : sima@192.168.91.221:22
  Mounted path  : /workspace
  Host export   : 192.168.74.48:/Users/joey/workspace

  You can now run DevKit binaries from this SDK shell:
    dk /workspace/<path-to-arm64-binary> [args...]
============================================================
```

From inside the SDK shell, use `dk status` to confirm the paired DevKit and the
active workspace sync method:

<ShellCommand prompt="username@neat-sdk-latest">
dk status
</ShellCommand>

## Update the Pairing from the SDK

Use this flow when the Neat SDK is already installed and you
need to pair a different DevKit or update the pairing after your DevKit IP
address changed.

From inside the Neat SDK container, run:

<ShellCommand prompt="username@neat-sdk-latest">
source devkit.sh {devkit-ip}
</ShellCommand>

Replace `{devkit-ip}` with the IP address of the DevKit you want to use.

Example:

<ShellCommand prompt="username@neat-sdk-latest">
source devkit.sh 192.168.91.221
</ShellCommand>

## Configure SDK Without DevKit Sync

If the DevKit is not reachable from the Neat SDK host, you can still configure
the SDK workspace without pairing:

<ShellCommand prompt="user-host-machine">
sima-cli sdk setup
</ShellCommand>

You can still build binaries in the Neat SDK container, but you must transfer
them to the DevKit manually for testing. Make sure the DevKit is running a
compatible Neat Library version.

## File Sharing with DevKit Sync

DevKit Sync connects three environments:

1. Host
2. Neat SDK container
3. DevKit

`sima-cli sdk setup --devkit {devkit-ip}` configures NFS so the same workspace is
available in all three environments:

- The host workspace folder is exported through host NFS.
- The folder is mounted into the Neat SDK container as
  `/workspace`.
- The same content appears on the DevKit as `/workspace` through NFS.
- The mounted folder name defaults to `/workspace` and can be changed during
  setup.

This setup provides a direct workflow for build artifacts:

- Artifacts produced in the Neat SDK are immediately visible
  on the DevKit without a separate deploy step.
- Agents can access logs, outputs, traces, and other interim files generated
  while the app runs on the DevKit.
- Developers and agents can inspect the same files from one workspace context.

With Insight, you can view the workspace from a web browser. Some SiMa.ai
specific model archives, such as `*.tar.gz` model artifacts, are automatically
optimized for easier inspection.

## Rsync Fallback

Starting with SDK 2.1.2.2, DevKit Sync can use rsync-over-SSH as a fallback
when NFS setup does not succeed. This is useful on networks or hosts where SSH
to the DevKit works, but the host NFS export cannot be mounted by the DevKit.

When rsync fallback is active:

- The host and SDK container still use the local `/workspace` directory.
- The DevKit uses a synchronized remote workspace, normally
  `/workspace-rsync`.
- `dk status` reports `Sync method : rsync` and shows the local and remote
  workspace paths.
- `dk <file> [args...]` maps paths from the SDK workspace to the DevKit rsync
  workspace before running the command remotely.
- Before `dk` runs a file, it automatically syncs the top-level workspace folder
  that contains that file. For example, `dk apps/demo.py` syncs the
  `/workspace/apps` folder before executing the DevKit-side copy.

Check the current pairing and sync method:

<ShellCommand prompt="username@neat-sdk-latest">
dk status
</ShellCommand>

Manually sync the current workspace scope:

<ShellCommand prompt="username@neat-sdk-latest">
dk sync
</ShellCommand>

Sync a specific file or folder scope:

<ShellCommand prompt="username@neat-sdk-latest">
dk sync /workspace/apps
</ShellCommand>

Sync the full workspace:

<ShellCommand prompt="username@neat-sdk-latest">
dk sync --all
</ShellCommand>

When rsync fallback is active, keep the files needed by one `dk` command under
the same top-level workspace folder. If a command is launched from
`/workspace/apps`, arguments that point into `/workspace/models` are outside the
auto-synced scope and should be synced separately with `dk sync
/workspace/models`, or the project should be organized so required files live
under the same top-level folder.

## Run on the DevKit with dk

The SDK includes the `dk` helper, also known as `devkit-run`, to run ARM64
executables on the paired DevKit from inside the SDK shell.

When you invoke `dk`, the SDK runs the command on the paired DevKit and
translates paths so file arguments from the container resolve correctly on the
DevKit.

<ShellCommand prompt="username@neat-sdk-latest">
dk <file> [args...]
</ShellCommand>

After compiling a C++ application in the SDK workspace, run the generated ARM64
executable on the DevKit:

<ShellCommand prompt="username@neat-sdk-latest">
dk build/sima_neat_hello
</ShellCommand>

After creating or copying a Python script into the SDK workspace, run it on the
paired DevKit:

<ShellCommand prompt="username@neat-sdk-latest">
dk hello_neat.py
</ShellCommand>

For Python scripts, `dk` runs the script on the paired DevKit and uses the
DevKit PyNeat runtime environment. The SDK remains useful as a unified
workspace and orchestration environment, but Python-only workflows do not
require the C++ cross-compilation toolchain.

:::note Where `dk` Comes From
`dk` is a shell function defined in `~/devkit-sync.rc` inside the SDK container.
The shell loads it through `~/.bashrc`, so it is available in interactive
sessions.
:::

## Next Step

To install or update the library/runtime itself, continue to
[Neat Library](/getting-started/neat-library/).
