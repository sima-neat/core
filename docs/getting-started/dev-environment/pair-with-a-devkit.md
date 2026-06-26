---
title: Pair with a DevKit
description: Pair the Neat SDK with a Modalix DevKit
sidebar_position: 3
---

Pair the Neat Development Environment (referred to as Neat SDK) with a DevKit
when the DevKit is on the same network as the Neat SDK host. Pairing enables
DevKit Sync, which exposes one shared workspace across the host, Neat SDK
container, and DevKit.

![Host-Container-DevKit workspace mapping](../../images/elxr-sdk-workspaces.svg)

The same workspace is mounted into the Neat SDK container
and the DevKit as `/workspace`, so build artifacts, logs, traces, and model
files are visible from each environment.

## Pair the Environment

Run this setup command from the host before opening the Neat SDK shell:

<ShellCommand prompt="user-host-machine">
sima-cli sdk setup --devkit {devkit-ip}
</ShellCommand>

After setup succeeds, open the paired Neat SDK container
with `sima-cli sdk neat`.

During setup:

- Select the installed `sdk:v2.1-latest` image if more than one SDK image is present.
- Accept the default `/workspace` DevKit mount path unless you need a different
  path.
- Enter the host admin password when prompted for NFS server setup on the host.
- Enter the DevKit user credentials when prompted. The default user is `sima`
  and the default password is `edgeai`.

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

## Update an Existing Pairing

Use this flow when the Neat SDK is already installed and you
need to pair a different DevKit or update the pairing after your DevKit IP
address changed.

From inside the Neat SDK container, run:

<ShellCommand prompt="username@neat-sdk-latest">
source devkit.sh <devkit-ip>
</ShellCommand>

Replace `<devkit-ip>` with the IP address of the DevKit you want to use.

Example:

<ShellCommand prompt="username@neat-sdk-latest">
source devkit.sh 192.168.91.221
</ShellCommand>

## Set Up Without DevKit Pairing

Use this flow when the DevKit is not reachable from the Neat SDK host:

<ShellCommand prompt="user-host-machine">
sima-cli sdk setup
</ShellCommand>

You can still build binaries in the Neat SDK container, but
you must transfer them to the DevKit manually for testing. Make sure the DevKit
is running a compatible Neat Library version.

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

## Next Step

Continue to [Run on the DevKit](/getting-started/dev-environment/run-on-the-devkit/).
