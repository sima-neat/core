---
title: Build
description: Build SiMa NEAT from source with build.sh
sidebar_position: 2
---

# Build NEAT

This guide covers source builds of NEAT.
For prebuilt package installation, see [NEAT Framework](./installation/neat-framework).

`build.sh` is the supported build entry point. It handles dependency checks, optional neat-internals sync, CMake configure/build, optional docs generation, install sanity checks, and packaging.

## Build Environments

`build.sh` automatically detects the active environment:

- Modalix DevKit native environment
- eLxr SDK environment (cross-compilation)

You can run the same `build.sh` commands in either environment.

### Cross Compilation Prerequisites

Cross-compilation is typically faster than building directly on the DevKit, but you must transfer build artifacts to the DevKit afterward. You will need the SiMa eLxr SDK for cross compilation.

Install `sima-cli` first on the host machine, then install eLxr SDK.

```bash
curl https://docs.sima.ai/_static/tools/sima-cli-installer.sh | bash
sima-cli install sdk
```

When prompted by `sima-cli`, select the **eLxr SDK** option.

Then start the eLxr SDK:

```bash
sima-cli sdk elxr
```

Then install sima-cli inside the eLxr SDK, after that install the eLxr SDK patch.

```bash
curl https://docs.sima.ai/_static/tools/sima-cli-installer.sh | bash
source ~/.bash_profile
sima-cli install tools/sdk-patch
```

- eLxr SDK installation is supported on Windows and Ubuntu.
- If you are building natively on a Modalix DevKit, the eLxr SDK install/patch steps are not required.

## Build Options

Supported `build.sh` options:

- `--dev-only`: Build only the core library and headers (default).
- `--all`: Build library + tests + tutorials + Python wheel; enables docs and neat-internals.
- `--python`: Build Python bindings (`pyneat`) in addition to selected targets.
- `--install-neat-internals`: Download and install neat-internals artifacts before build.
- `--doc`: Build docs only.
- `--no-dist`: Skip distribution packaging.
- `--clean`: Remove `build/` before configuring.
- `--no-doc`: Skip docs build (even with `--all`).
- `--no-node`: Skip Node.js install (docs build may fail if Node is missing).
- `--install-deps-only`: Install system dependencies only, then exit.

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

Install dependencies only:

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
