---
title: Windows Host Notes
description: Prepare a Windows 11 host for Neat SDK and DevKit-Sync
sidebar_position: 1
---

Use this guide when your host machine is Windows 11 and you want to run Neat SDK with DevKit-Sync.

## Prerequisites

- Windows 11 host.
- [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) installed and working.
- Docker Engine installed inside WSL.
- `sima-cli` installed inside WSL.

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

## NFS firewall rules (PowerShell)

Allow NFS-related traffic in Windows firewall. Run PowerShell as Administrator and add rules for required NFS ports/protocols using `New-NetFirewallRule`.

Example pattern:

```powershell
New-NetFirewallRule -DisplayName "Allow NFS TCP 2049" -Direction Inbound -Protocol TCP -LocalPort 2049 -Action Allow
New-NetFirewallRule -DisplayName "Allow NFS UDP 2049" -Direction Inbound -Protocol UDP -LocalPort 2049 -Action Allow
```

Add any additional ports required by your NFS server/client setup.

## Next step

Return to [Neat SDK](../../getting-started/installation/neat-elxr-sdk.mdx) and continue with install/setup.
