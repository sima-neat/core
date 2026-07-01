---
title: Windows Host Notes
description: Prepare a Windows 11 host for the Neat SDK and DevKit-Sync
sidebar_position: 2
---

Use this guide when your host machine is Windows 11 and you want to run the
Neat Development Environment (referred to as Neat SDK) with DevKit-Sync.

## Prerequisites

- Windows 11 host.
- [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) installed and working.
- Docker Engine installed inside WSL.
- `sima-cli` installed inside WSL.

## Start from WSL

Run Neat SDK commands from inside your WSL Linux
distribution, not from PowerShell or Command Prompt. This includes
`sima-cli neat install sdk@release-2.1`.

## WSL network mode

Configure `%UserProfile%\\.wslconfig`:

```ini
[wsl2]
networkingMode=mirrored
```

Then restart WSL:

```powershell
wsl --shutdown
```

This allows WSL to share host network configuration, which helps DevKit-Sync and NFS communication.

## Recommended topology: direct Windows-to-DevKit connection

For Windows hosts, a direct USB/Ethernet connection between the Windows machine and DevKit is the
recommended topology. It is usually simpler to set up than placing the DevKit on a broader shared
network, and Windows firewall changes can be scoped to the local DevKit-facing interface instead of
the whole network. Unlike Ubuntu and macOS, prefer this direct-link setup for Windows unless your
environment already has validated firewall and WSL networking rules for a shared network.

Use Internet Connection Sharing (ICS) when the DevKit needs to share the Windows machine's network
connection over the direct link.

1. Connect the Windows machine to the internet through Wi-Fi or another upstream interface.
2. Connect the DevKit to the Windows machine through the USB/Ethernet adapter.
3. On the DevKit, leave the connected network interface set to `DHCP`.
4. On Windows, open `Control Panel > Network and Internet > Network Connections`.
   You can also press `Win + R`, run `ncpa.cpl`, and press Enter.
5. Right-click the internet-facing adapter, then select `Properties`.
6. Open the `Sharing` tab.
7. Enable `Allow other network users to connect through this computer's Internet connection`.
8. In `Home networking connection`, select the USB/Ethernet adapter connected to the DevKit.
9. Apply the change, then reconnect the DevKit-facing adapter if the DevKit does not receive an
   IPv4 address.

After ICS is enabled, Windows usually assigns the shared adapter an address on `192.168.137.0/24`.
Find the DevKit IPv4 address from WSL or from the DevKit console, then confirm SSH access from WSL:

```bash
ssh sima@<devkit-ip>
```

Then continue with DevKit pairing from WSL:

```bash
sima-cli sdk setup --devkit <devkit-ip>
```

:::note Insight access on Windows direct links
With Windows direct network sharing, Windows firewall and WSL port-forwarding behavior can prevent
the Neat Insight web interface from being reached by other machines on the network. In this setup,
open Insight directly on the Windows Neat SDK host, for
example at `https://localhost:9900`.
:::

## NFS firewall rules (PowerShell)

Allow NFS-related traffic in Windows firewall. Run PowerShell as Administrator and add rules for required NFS ports/protocols using `New-NetFirewallRule`.

Example pattern:

```powershell
New-NetFirewallRule -DisplayName "Allow NFS TCP 2049" -Direction Inbound -Protocol TCP -LocalPort 2049 -Action Allow
New-NetFirewallRule -DisplayName "Allow NFS UDP 2049" -Direction Inbound -Protocol UDP -LocalPort 2049 -Action Allow
```

Add any additional ports required by your NFS server/client setup.

## Next step

Return to [Neat SDK](/getting-started/dev-environment/) and continue with install/setup.
