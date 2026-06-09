---
title: Python Bindings (pyneat)
description: Build, test, and package pyneat bindings as a contributor
sidebar_position: 6
---

# Python Bindings (`pyneat`)

This page is for `pyneat` contributors and maintainers.

`pyneat` is the Python binding layer for SiMa.ai Neat, built with `nanobind` and packaged with `scikit-build-core`.

See the [Python API Reference](/reference/pythonapi/modules/pyneat) for generated API docs.

## Prerequisites

`pyneat` links against the same native dependencies as the C++ library, including:

- GStreamer development/runtime packages
- OpenCV development/runtime packages
- C++ toolchain (`cmake`, compiler, `pkg-config`)

See [Build](/develop-apps/contribute/build) for host setup guidance.

## Install from source

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

Editable install for development:

```bash
python -m pip install -e .[dev]
```

## Run tests

```bash
pytest -q python/tests
```

## Packaging

`pyproject.toml` in the repository root defines wheel/sdist build configuration for `pyneat`.
