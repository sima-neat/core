---
title: B4593 DMA-BUF Package Contract
description: Build and install the matching Core and Internals camera-stack branches on the B4593 kernel
sidebar_position: 5
slug: /develop-apps/contribute/b4586-dmabuf-package-contract
---

# B4593 DMA-BUF Package Contract

The `feature/b4586-dmabuf-zero-copy` integration is intentionally restricted to
the B4593/pre4593 Modalix package cohort. It combines the DMA-BUF kernel ABI with
patched libcamera and SiMaAI memory packages. Do not install only one of these
overrides or use this bundle on an older running kernel.

## Matching-branch build

Create the same branch name in Core and Internals:

```text
feature/b4586-dmabuf-zero-copy
```

Core keeps the Internals dependency on `policy: snap`. The additional
`required-branch` contract makes resolution fail closed:

- Core must be built from `feature/b4586-dmabuf-zero-copy`.
- The resolved Internals artifact must report that same branch.
- An unavailable branch artifact fails the build; it never falls back to
  `develop:latest`.
- The resolved Internals commit is written to the resolved dependency manifest
  and build metadata as provenance. It is not pinned in `deps/manifest.json`.

Publish a green Internals Vulcan artifact before starting the Core build.

## Reviewed package set

The package contract in `deps/manifest.json` is the source of truth:

| Component | Required identity |
| --- | --- |
| Running kernel | `linux-image-6.18.3-modalix=6.18.3-4593`, release `6.18.3-modalix`, build marker `#4593` |
| Libcamera override set | `libcamera`, `libcamera-dev`, and `libcamera-tools` at `2.1.3+neat1` |
| Libcamera Palette compatibility | Each package provides its `2.1.3~pre4593` identity |
| Libcamera DMA-BUF capability | Runtime provides `simaai-libcamera-dmabuf-abi (= 1)` |
| Memory override set | `simaai-memory-lib` and `simaai-memory-lib-dev` at `2.1.1-0neat4` |
| Memory Palette compatibility | Both packages provide their `2.1.1~pre4593` identity |
| Memory DMA-BUF capability | Runtime provides `simaai-memory-dmabuf-export-abi (= 1)` |

The pre-release repository is a source for the native B4593 cohort, not a
moving dependency selector for this branch. Build and install artifacts must
retain exact filenames, versions, and checksums in their generated metadata.

## Installer behavior

On a Modalix board, `install_neat_framework.sh` completes the following
preflight before creating the PyNeat environment, stopping services, or changing
packages:

1. Validate the exact reviewed contract copied into `manifest.json`.
2. Confirm the B4593 kernel package is installed and is the running kernel.
3. Require exactly one complete libcamera runtime/dev/tools override set.
4. Require exactly one complete memory runtime/dev override set.
5. Validate architecture, exact package versions, Palette compatibility
   `Provides`, DMA-BUF capability markers, internal exact dependencies, ELF
   build IDs, the memory export symbol, libcamerasrc zero-copy property strings,
   and available artifact checksums.

APT then simulates and installs all five override packages together with
`--reinstall --no-remove`. The installer rejects removals and changes to any
preinstalled package that is not supplied by the bundle. After installation it
checks package ownership and payload hashes/build IDs, runs `apt-get check`,
preserves `simaai-palette-modalix` and `simaai-ota`, and verifies that
`gst-inspect-1.0 libcamerasrc` exposes both strict zero-copy properties.

The existing `NEAT_INSTALLER_SKIP_PLATFORM_CHECK=ON` option remains an explicit
development escape hatch. Do not use it for B4593 qualification evidence.

## Qualification checks

Run the host contract tests before publishing:

```bash
python3 -m unittest \
  tests.tools.test_internals_snap_branch_contract \
  tests.tools.test_b4586_package_contract \
  tests.tools.test_validate_neat_package_bundle \
  tests.tools.test_install_neat_framework_native_restore
bash scripts/ci/check_artifacts.sh
```

After installing on the DevKit, confirm the final identities:

```bash
uname -r
cat /proc/version
dpkg-query -W -f='${Package}\t${Version}\t${Provides}\n' \
  linux-image-6.18.3-modalix \
  libcamera libcamera-dev libcamera-tools \
  simaai-memory-lib simaai-memory-lib-dev
gst-inspect-1.0 libcamerasrc | \
  grep -E 'simaai-zero-copy(-required)?'
sudo apt-get check
sudo dpkg --audit
```

Run the package inventory test and strict CameraInput smoke test through the
DevKit wrapper. Check each target binary with `file` first; execute an ARM64
binary with `dk` or the repository DevKit wrapper, never directly on the host.
