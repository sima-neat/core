---
title: macOS Host Notes
description: Prepare a macOS host for Neat SDK and DevKit-Sync
sidebar_position: 3
---

Use this guide when your host machine is macOS and you want to run Neat
Development Environment (referred to as Neat SDK) with DevKit-Sync.

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

## Troubleshoot Insight video with Colima UDP forwarding

If Insight opens in the browser but live video does not appear, UDP packets may
not be reaching the SDK container. On macOS with Colima, this can happen when
Colima uses SSH port forwarding. Docker can still show the expected UDP port
mappings, but Colima's host-to-VM forwarding path may not deliver UDP traffic
into the container.

The SDK normally publishes these UDP ranges:

- `9000-9079/udp` for video
- `9100-9179/udp` for metadata
- `40000-40199/udp` for WebRTC

If the SDK container is running and those UDP mappings are present, check the
Colima port forwarder. SSH forwarding is TCP-only, while gRPC forwarding supports
TCP and UDP.

Reconfigure Colima to use gRPC forwarding:

```bash
colima stop
colima start --edit
```

In the editor, change:

```yaml
portForwarder: ssh
```

to:

```yaml
portForwarder: grpc
```

Then restart the SDK:

```bash
sima-cli sdk stop
sima-cli sdk start
sima-cli sdk neat
```

Verify Docker still publishes the UDP ports:

```bash
docker ps --format 'table {{.Names}}\t{{.Ports}}'
```

Confirm the SDK container lists `9000-9079/udp`, `9100-9179/udp`, and the WebRTC
UDP range.

While sending video, check whether Insight sees incoming packets from inside the
SDK:

```bash
curl -k 'https://127.0.0.1:9900/api/ingest/stats?all=1&verbose=1'
```

Look for `packets_received` increasing on the expected channel. Make sure the
sender targets the Mac host IP and the correct UDP port, not the SDK container
IP. Channel 0 uses UDP `9000`, channel 1 uses UDP `9001`, and so on.

If UDP still does not reach Insight after switching Colima to gRPC forwarding,
test with Docker Desktop or a Linux host. In that case, the SDK and Docker port
mappings are likely correct, and the remaining suspicious layer is Colima's
macOS UDP forwarding path.

## Next step

Return to [Neat SDK](/getting-started/dev-environment/) and continue with install/setup.
