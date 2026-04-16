---
title: macOS Host Notes
description: Prepare a macOS host for NEAT eLxr SDK and DevKit-Sync
sidebar_position: 2
---

Use this guide when your host machine is macOS and you want to run NEAT eLxr SDK with DevKit-Sync.

## Prerequisites

- macOS host.
- `sima-cli` installed on host.
- Modalix DevKit reachable on the same network.

## Install and run Colima

Install and start Colima so Docker workloads can run on macOS.

```bash
brew install colima docker
colima start
docker ps
```

If Colima is already installed, make sure it is running before you use `sima-cli sdk setup`.

## NFS permission on macOS

DevKit-Sync uses host NFS export during SDK setup. On macOS, ensure `nfsd` has Full Disk Access, or host workspace export/mount can fail.

Steps:

1. Open System Settings.
2. Go to Privacy & Security > Full Disk Access.
3. Click `+`, then press `Cmd + Shift + G` and enter `/sbin/`.
4. Select `nfsd` and ensure it is allowed.
5. Re-run SDK setup after permission is granted.

## Next step

Return to [NEAT eLxr SDK](../../getting-started/installation/neat-elxr-sdk.mdx) and continue with install/setup.
