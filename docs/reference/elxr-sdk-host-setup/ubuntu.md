---
title: Ubuntu Host Notes
description: Prepare an Ubuntu host for Neat SDK and DevKit-Sync
sidebar_position: 1
---

Use this guide when your host machine is Ubuntu and you want to run Neat SDK with DevKit-Sync.

## Prerequisites

- Ubuntu 22.04 or 24.04 host.
- Docker Engine installed and working.
- `sima-cli` installed on the host.
- Modalix DevKit reachable on the same network.

:::info Network topology
On Ubuntu, you can either connect the DevKit directly to the host through USB/Ethernet or place the
host and DevKit separately on an existing network. If they are on an existing network, no special
sharing setup is required as long as the host and DevKit can reach each other for SSH and NFS traffic.
:::

## Direct Ubuntu-to-DevKit connection

Use this flow when the DevKit is connected directly to the Ubuntu machine through USB/Ethernet
and needs to share the Ubuntu machine's network connection.

Disable IPv6 on the shared DevKit-facing network interface. DevKit-Sync relies on predictable IPv4
addressing for SSH and NFS, and leaving IPv6 enabled on the shared link can make discovery and route
selection unreliable.

### NetworkManager GUI

1. Connect the Ubuntu machine to the internet through Wi-Fi or another upstream interface.
2. Connect the DevKit to the Ubuntu machine through the USB/Ethernet adapter.
3. On the DevKit, leave the connected network interface set to `DHCP`.
4. On Ubuntu, open `Settings > Network`.
5. Open the settings for the wired interface connected to the DevKit.
6. In the `IPv4` tab, set `IPv4 Method` to `Shared to other computers`.
7. In the `IPv6` tab, set `IPv6 Method` to `Disabled`.
8. Apply the change, then disconnect and reconnect the wired interface.

After the link comes up, find the DevKit IPv4 address from Ubuntu:

```bash
ip neigh
```

Confirm SSH access before starting SDK setup:

```bash
ssh sima@<devkit-ip>
```

Then continue with DevKit pairing:

```bash
sima-cli sdk setup --devkit <devkit-ip>
```

### NetworkManager CLI

If you prefer command-line setup, identify the DevKit-facing interface:

```bash
nmcli device status
```

Create a shared IPv4 connection with IPv6 disabled:

```bash
sudo nmcli connection add type ethernet ifname <devkit-interface> con-name devkit-shared ipv4.method shared ipv6.method disabled
sudo nmcli connection up devkit-shared
```

If a connection profile already exists for that interface, modify it instead:

```bash
sudo nmcli connection modify "<connection-name>" ipv4.method shared ipv6.method disabled
sudo nmcli connection down "<connection-name>"
sudo nmcli connection up "<connection-name>"
```

## Firewall notes

If Ubuntu firewall rules are enabled, allow SSH and NFS traffic on the DevKit-facing interface or
subnet before running DevKit-Sync setup. At minimum, the DevKit must be able to reach SSH and the
host NFS export created by `sima-cli sdk setup --devkit`.

## Next step

Return to [Neat SDK](/getting-started/installation/neat-elxr-sdk/) and continue with install/setup.
