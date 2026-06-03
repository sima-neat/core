---
title: AppComplex Workspace Packaging
description: Build and install a gated appcomplex workspace service package
sidebar_position: 7
---

# AppComplex Workspace Packaging

This guide packages `tmp/core/sima-ai-appcomplex` into a gated system package that does not replace the live `simaai-appcomplex.service` unless you explicitly request it.

## What gets installed

- Binary and libs under `/opt/simaai/appcomplex-workspace/`
- Systemd unit: `simaai-appcomplex-workspace.service`
- Config file: `/etc/default/simaai-appcomplex-workspace`

The workspace unit defaults to isolated endpoints:

- Control socket: `/tmp/mlactrl_workspace`
- SHM object: `/mlashmdata_workspace`
- MLA init gate: `APP_COMPLEX_RUN_INIT=0` (skip init for parallel run)

## Build package

```bash
./scripts/release/build_appcomplex_workspace_deb.sh
```

The script prints the generated `.deb` path in `build/packages/`.

## Install (gated default)

```bash
./scripts/release/install_appcomplex_workspace_deb.sh --deb <path-to-deb>
```

Default install behavior:

- Does not stop/disable `simaai-appcomplex.service`
- Does not auto-enable/start `simaai-appcomplex-workspace.service`

## Activate workspace service manually

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now simaai-appcomplex-workspace.service
```

If you need MLA re-init before workspace startup (for cutover mode), set:

```bash
sudo sed -i 's/^APP_COMPLEX_RUN_INIT=.*/APP_COMPLEX_RUN_INIT=1/' /etc/default/simaai-appcomplex-workspace
```

## Optional cutover (explicit only)

To request stopping old service and activating workspace service:

```bash
./scripts/release/install_appcomplex_workspace_deb.sh --deb <path-to-deb> --activate --switch-system
```

Or update `/etc/default/simaai-appcomplex-workspace`:

- `APP_COMPLEX_ACTIVATE_ON_INSTALL=1`
- `APP_COMPLEX_SWITCH_SYSTEM_SERVICE=1`

and then run:

```bash
sudo dpkg-reconfigure simaai-appcomplex-workspace
```
