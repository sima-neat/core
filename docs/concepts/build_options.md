---
title: Build profiles & CMake options
description: How the framework's CMake build is configured, what each option toggles, and which profile to pick.
sidebar_position: 10
---

# Build profiles & CMake options

The framework's top-level `CMakeLists.txt` exposes a handful of options that control what gets built and how. This page documents the load-bearing ones.

## Build profiles

The framework supports three named profiles:

| Profile | Use case | What's compiled |
|---------|----------|-----------------|
| **Production** | Customer-facing builds | All public Nodes, MPK loader, Modalix backends, optimized |
| **Developer** | Framework engineers | Production set + debug Nodes + extended diagnostics + tests |
| **Sandbox** | Multi-tenant deployments | Production set + tightened MPK security defaults |

Pick via `-DSIMA_NEAT_PROFILE=Production|Developer|Sandbox` at configure time (or accept the default in `CMakeLists.txt`).

## Common options

| Option | Default | Effect |
|--------|---------|--------|
| `SIMA_NEAT_BUILD_TESTS` | `ON` (Developer) | Build the gtest suite. Disable for faster CI on production builds. |
| `SIMA_NEAT_BUILD_TUTORIALS` | `OFF` | Build the tutorial binaries. (Tutorials are being deprecated.) |
| `SIMA_NEAT_BUILD_PYTHON` | `ON` | Build the `pyneat` nanobind module. |
| `SIMA_NEAT_BUILD_INTERNALS` | `OFF` (public) | Build the internal reach-through tier (`core/src/pipeline/internal/sima/`). |
| `SIMA_NEAT_ENABLE_TVM_FALLBACK` | `ON` | Compile TVM-backed fallback kernels for ops the MLA can't handle. |
| `SIMA_NEAT_ENABLE_RTSP` | `ON` | Build the RTSP source/sink Nodes. |
| `SIMA_NEAT_DEBUG_PLUGINS` | `OFF` | Forward GStreamer plugin debug to stdout. |
| `SIMA_NEAT_USE_SYSTEM_GSTREAMER` | `ON` (host) / `OFF` (cross) | Link against the system's GStreamer instead of bundling. |
| `SIMA_NEAT_WARN_AS_ERROR` | `OFF` | Promote compile warnings to errors. Recommended for CI. |

## Toolchain knobs

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

## Option discovery

To enumerate what your tree exposes (versions and defaults can drift across releases):

```bash
cmake -L -B build       # list all cache variables
cmake -LA -B build      # include advanced
```

## Further reading

- "Build profiles" — §32, §60 of the design deep dive.
- [Build / install](/getting-started/build) — the user-facing build guide.
- The top-level `CMakeLists.txt` is the source of truth for option names.
