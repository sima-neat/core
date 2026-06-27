---
title: Configure SDK
description: Change Neat SDK workspace, DevKit pairing, or setup options after installation
sidebar_position: 4
---

:::tip Start here only when changing SDK settings
The SDK install command already downloads the SDK image and configures it. Use this
page if you need to change SDK settings, such as workspace location,
DevKit pairing, SDK extensions, or other setup options.
:::

Run `sima-cli sdk setup` directly when you need to configure an existing Neat
SDK container.

## Configure with DevKit Pairing

Use this command when your DevKit is reachable from the host and you want to add
or update DevKit pairing:

<ShellCommand prompt="user-host-machine">
sima-cli sdk setup --devkit {devkit-ip}
</ShellCommand>

Have these ready:

- the DevKit IP address;
- your host administrator password, if NFS needs to be installed or configured;
- the DevKit user credentials when prompted. The default user is `sima` and the
  default password is `edgeai`.

With `--devkit`, setup enables DevKit Sync. It exports your host workspace over
NFS and mounts it as `/workspace` on the DevKit by default.

## Configure without DevKit Pairing

Use this command when you want to update SDK settings but the DevKit is not
reachable yet:

<ShellCommand prompt="user-host-machine">
sima-cli sdk setup
</ShellCommand>

During setup, `sima-cli` may ask you to:

- choose a host workspace directory. Accept the default unless you need a
  different workspace;
- choose an SDK extensions directory;
- choose whether to install Model Compiler.

The Model Compiler is required to compile or quantize models yourself and is
optional only if you exclusively use precompiled model packages. Install it here
if you plan to compile or quantize models, and complete `sima-cli login` if
prompted.

## Next Step

Continue to [DevKit Sync](/getting-started/dev-environment/devkit-sync/) for
workspace sharing, pairing updates, rsync fallback, and `dk` command details.
