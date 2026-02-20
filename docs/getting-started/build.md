---
title: Build
description: Build SiMa NEAT from source with build.sh
sidebar_position: 2
---

# Build NEAT

> **Current requirement:** Source builds are supported on a **Modalix DevKit** environment at this time. Cross compilation is not supported yet.

This guide is for developers who need a source build. If you are installing prebuilt packages, use the install guide instead.

`build.sh` is the supported entry point. It handles dependency checks, runtime plugin assembly, CMake build, optional docs site generation, install sanity-check, and (Linux only) `.deb` packaging.

## Prerequisites

- A Modalix DevKit build environment.
- On Linux: `apt-get` and root/passwordless sudo are needed if dependencies are missing.
- On macOS: The documentation can be built and tested on Mac for convenience, Homebrew is required to install dependencies.
- For docs builds: Node.js 20.x (auto-installed by `build.sh` unless `--no-node` is used).

## Build options

`build.sh` options:

- `--dev-only` (default): Build the core library and headers only.
- `--all`: Build library + samples + tests + docs.
- `--example`: Build example executables only (and core library dependencies).
- `--python`: Build Python bindings (`pyneat`) in addition to selected targets.
- `--doc`: Build docs only.
- `--clean`: Remove `build/` before configuring.
- `--no-doc`: Skip docs build (useful with `--all`).
- `--no-node`: Skip Node installation (docs site build may fail if Node is missing).
- `--no-dist`: Skip `.deb` packaging.
- `--install-deps-only`: Install missing system deps and exit.

## Typical builds

Core library only (default):

```bash
./build.sh
```

Full build (library, samples, tests, docs):

```bash
./build.sh --all
```

Examples only:

```bash
./build.sh --example
```

Core library + Python bindings:

```bash
./build.sh --dev-only --python
```

Docs only (API + tutorial docs + Docusaurus site):

```bash
./build.sh --doc
```

Clean full build:

```bash
./build.sh --all --clean
```

Install dependencies only:

```bash
./build.sh --install-deps-only
```

## Outputs

- Build tree: `build/`
- Docusaurus output (when docs build runs): `website/build/`
- Install sanity-check prefix: `/tmp/sima-neat-install-test`
- `.deb` packages are Linux-only.
- `.deb` packages are generated via CPack.
- `.deb` packages are produced only when using `--all` and not `--no-dist`.
