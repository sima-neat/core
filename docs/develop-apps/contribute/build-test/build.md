---
title: Build
description: Build SiMa.ai Neat from source with build.sh
sidebar_position: 1
slug: /develop-apps/contribute/build
---

# Build Neat

This guide covers source builds of Neat.
For prebuilt package installation, see [Neat Library](/getting-started/installation/neat-library/).

`build.sh` is the supported build entry point. It handles dependency checks, optional deps sync, CMake configure/build, optional docs generation, install sanity checks, and packaging.

## Build Environments

`build.sh` automatically detects the active environment:

- Modalix DevKit native environment
- Palette SDK environment (cross-compilation)

You can run the same `build.sh` commands in either environment.

### Cross Compilation Prerequisites

Cross-compilation is typically faster than building directly on the DevKit, but you must transfer build artifacts to the DevKit afterward. You will need Palette SDK for cross compilation.

Install `sima-cli` first on the host machine, then install the SDK.

```bash
curl -fsSL https://artifacts.neat.sima.ai/sima-cli/linux-mac.sh | bash
sima-cli install sdk
```

When prompted by `sima-cli`, select the SDK option.

Then start the SDK:

```bash
sima-cli sdk elxr
```

Then install `sima-cli` inside the SDK, then install the SDK patch.

```bash
curl -fsSL https://artifacts.neat.sima.ai/sima-cli/linux-mac.sh | bash
source ~/.bash_profile
sima-cli install tools/sdk-patch
```

- SDK installation is supported on Windows and Ubuntu.
- If you are building natively on a Modalix DevKit, the SDK install/patch steps are not required.

## Build Options

Supported `build.sh` options:

- `--dev-only`: Build only the core library and headers (default).
- `--all`: Build library + tests + tutorials + Python wheel; enables docs and deps.
- `--python`: Build Python bindings (`pyneat`) in addition to selected targets.
- `--install-neat-internals`, `--install-deps`: Download and install deps artifacts before build.
- `--doc`: Build docs only.
- `--install`: After build/package, install generated artifacts into the current environment. In paired Palette SDK mode, this also deploys and installs matching artifacts on the paired DevKit.
- `--no-dist`: Skip distribution packaging.
- `--clean`: Remove `build/` before configuring.
- `--no-doc`: Skip docs build (even with `--all`).
- `--no-node`: Skip Node.js install (docs build may fail if Node is missing).
- `--install-deps-only`: Install system dependencies and dependency headers, then exit.

## Typical Builds

Core library only (default):

```bash
./build.sh
```

Full build (library, tests, tutorials, docs, wheel, packaging):

```bash
./build.sh --all
```

Core library + Python bindings:

```bash
./build.sh --dev-only --python
```

Docs only:

This command also works on macOS.

```bash
./build.sh --doc
```

Clean full build:

```bash
./build.sh --all --clean
```

Install dependencies without building core:

```bash
./build.sh --install-deps-only
```

## Outputs

- Build tree: `build/`
- Docusaurus site output (when docs build runs): `website/build/`
- Install sanity-check prefix: `/tmp/sima-neat-install-test`
- Core package (`*.deb`) is generated on Linux full builds unless `--no-dist` is used.
- Extras package (`*extras.tar.gz`) is generated on Linux full builds unless `--no-dist` is used.
- Python wheel (`dist/*.whl`) is generated when Python build is enabled.

## Build Profiles & CMake Options

The framework's top-level `CMakeLists.txt` exposes a handful of options that
control what gets built and how. The options below are the load-bearing ones.

### Build profiles

The framework supports three named profiles:

| Profile | Use case | What's compiled |
|---------|----------|-----------------|
| **Production** | Customer-facing builds | All public Nodes, model-archive loading, Modalix backends, optimized |
| **Developer** | Framework engineers | Production set + debug Nodes + extended diagnostics + tests |
| **Sandbox** | Multi-tenant deployments | Production set + tightened model-archive security defaults |

Pick via `-DSIMA_NEAT_PROFILE=Production|Developer|Sandbox` at configure time,
or accept the default in `CMakeLists.txt`.

### Common CMake options

| Option | Default | Effect |
|--------|---------|--------|
| `SIMA_NEAT_BUILD_TESTS` | `ON` (Developer) | Build the gtest suite. Disable for faster CI on production builds. |
| `SIMA_NEAT_BUILD_TUTORIALS` | `OFF` | Build the tutorial binaries. |
| `SIMA_NEAT_BUILD_PYTHON` | `ON` | Build the `pyneat` nanobind module. |
| `SIMA_NEAT_BUILD_INTERNALS` | `OFF` (public) | Build the internal reach-through tier (`core/src/pipeline/internal/sima/`). |
| `SIMA_NEAT_ENABLE_TVM_FALLBACK` | `ON` | Compile TVM-backed fallback kernels for ops the MLA can't handle. |
| `SIMA_NEAT_ENABLE_RTSP` | `ON` | Build the RTSP source/sink Nodes. |
| `SIMA_NEAT_DEBUG_PLUGINS` | `OFF` | Forward GStreamer plugin debug to stdout. |
| `SIMA_NEAT_USE_SYSTEM_GSTREAMER` | `ON` (host) / `OFF` (cross) | Link against the system's GStreamer instead of bundling. |
| `SIMA_NEAT_WARN_AS_ERROR` | `OFF` | Promote compile warnings to errors. Recommended for CI. |

### Toolchain knobs

For cross-compilation toward Modalix:

```bash
cmake -B build \
  -DCMAKE_TOOLCHAIN_FILE=cmake/modalix.cmake \
  -DSIMA_NEAT_PROFILE=Production
```

For host-side development:

```bash
cmake -B build -DSIMA_NEAT_PROFILE=Developer
```

To enumerate what your tree exposes:

```bash
cmake -L -B build       # list all cache variables
cmake -LA -B build      # include advanced
```

The top-level `CMakeLists.txt` is the source of truth for option names.
