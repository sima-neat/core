---
title: macOS Host Notes
description: Prepare a macOS host for Neat SDK and DevKit-Sync
sidebar_position: 3
---

Use this guide when your host machine is macOS and you want to run Neat SDK with DevKit-Sync.

## Prerequisites

- macOS host.
- `sima-cli` installed on host.
- Modalix DevKit reachable on the same network.

:::info Network topology
On macOS, you can either connect the DevKit directly to the host through USB/Ethernet or place the
host and DevKit separately on an existing network. If they are on an existing network, no special
sharing setup is required as long as the host and DevKit can reach each other for SSH and NFS traffic.
:::

## Install and run Colima

Install and start Colima so Docker workloads can run on macOS.

```bash
brew install colima docker
colima start
docker ps
```

If Colima is already installed, make sure it is running before you use `sima-cli sdk setup`.

## NFS permissions on macOS

DevKit-Sync uses a host NFS export during SDK setup. On macOS, ensure `nfsd` has Full Disk Access, or host workspace export/mount can fail.

Steps:

1. Open System Settings.
2. Go to Privacy & Security > Full Disk Access.
3. Click `+`, then press `Cmd + Shift + G` and enter `/sbin/`.
4. Select `nfsd` and ensure it is allowed.
5. Re-run SDK setup after permission is granted.

## Internet Sharing for a direct Mac-to-DevKit connection

Use this flow when the DevKit cannot connect directly to your normal network and must access the internet through a direct USB/Ethernet link to your Mac.

1. On the DevKit, set the connected network interface to `DHCP` (this is usually the default).
2. On macOS, open `System Settings > General > Sharing > Internet Sharing`.
3. Set **Share your connection from** to `Wi-Fi`.
4. Enable sharing to the USB/Ethernet dongle interface connected to the DevKit.
5. On the Mac side, make sure the USB/Ethernet dongle interface is also configured for `DHCP`.

After Internet Sharing is enabled, the active USB/Ethernet interface on the Mac should receive an address (for example `en0` or `en1`, depending on your adapter and host setup).

### DNS workaround on DevKit

In this direct-link scenario, DNS on the DevKit may remain misconfigured even after DHCP succeeds. If name resolution fails, update `/etc/resolv.conf` on the DevKit:

```bash
sudo nano /etc/resolv.conf
```

Set:

```text
nameserver 8.8.8.8
nameserver 127.0.0.1
```

## Next step

Return to [Neat Development Environment](/getting-started/dev-environment/) and continue with install/setup.
