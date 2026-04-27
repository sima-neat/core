# SiMa Neat Framework

[![CI Build](https://github.com/sima-neat/core/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/sima-neat/core/actions/workflows/build.yml)
[![Sanitization Nightly](https://github.com/sima-neat/core/actions/workflows/sanitizers.yml/badge.svg?branch=main)](https://github.com/sima-neat/core/actions/workflows/sanitizers.yml)
[![Soak Weekly](https://github.com/sima-neat/core/actions/workflows/test-soak-weekly.yml/badge.svg?branch=main)](https://github.com/sima-neat/core/actions/workflows/test-soak-weekly.yml)
[![Fuzz Nightly](https://github.com/sima-neat/core/actions/workflows/test-fuzz-nightly.yml/badge.svg?branch=main)](https://github.com/sima-neat/core/actions/workflows/test-fuzz-nightly.yml)
[![Crash Correctness Nightly](https://github.com/sima-neat/core/actions/workflows/test-crash-correctness-nightly.yml/badge.svg?branch=main)](https://github.com/sima-neat/core/actions/workflows/test-crash-correctness-nightly.yml)
![SDK](https://img.shields.io/badge/SDK-2.0-green)
![Language](https://img.shields.io/badge/C%2B%2B-20-informational)

SiMa Neat is a C++20 library for building, validating, running, and debugging GStreamer pipelines with a typed, composable API.

It helps teams ship production media/ML pipelines with reproducible pipeline generation, strong diagnostics, and clean C++ integration.

<img src="docs/images/concepts.jpg" alt="NEAT concepts diagram" width="80%" />

## Programming Model

Neat is built around three core concepts:

- `Model`: loads a compiled model pack (`.tar.gz`) and exposes reusable pipeline stages (`preprocess`, `inference`, `postprocess`, or full `session()`).
- `Session` + `Run`: composes typed `Node`/`NodeGroup` blocks into a deterministic pipeline, then executes it in `Sync` or `Async` mode via push/pull APIs.
- `Tensor` + `Sample`: `Tensor` is the typed data container (dtype/layout/shape/device/storage), while `Sample` wraps tensors with stream metadata (timestamps, caps, routing info, bundles).

Typical flow:

1. Build a `Model` from compiled .tar.gz model file.
2. Add model stages (and optional custom nodes) to a `Session`.
3. `build(...)` to get a `Run`, then `run()` or `push()/pull()` tensors/samples.

```cpp
#include "model/Model.h"
#include "pipeline/Session.h"
#include "nodes/io/Input.h"
#include "nodes/common/Output.h"

simaai::neat::Model model("resnet_50_mpk.tar.gz");

simaai::neat::Session session;
session.add(simaai::neat::nodes::Input());
session.add(model.session());
session.add(simaai::neat::nodes::Output());

auto run = session.build(input_tensor, simaai::neat::RunMode::Sync);
auto out = run.push_and_pull(input_tensor);
```

## Build NEAT

For source builds, use `build.sh`:

```bash
./build.sh
```

Common build modes:

```bash
./build.sh --all           # library + samples + tests + docs
./build.sh --doc           # docs only
./build.sh --all --clean   # clean full build
./build.sh -h              # print help
```

Build output is generated under `build/`.

## Python Bindings (`pyneat`)

This repository includes a production-oriented Python binding layer powered by `nanobind`.

Key points:

- Python package source lives under `python/`.
- Extension module is `pyneat._pyneat_core`.
- High-level Python wrappers provide ergonomic APIs on top of the C++ layer.
- NumPy/PyTorch interop is supported through DLPack (`Tensor.from_numpy`, `Tensor.to_numpy`, `Tensor.from_torch`, `Tensor.to_torch`).

Build/install from source:

Running `./build.sh --all` builds both C++ libs and pyneat. If you want to build pyneat specifically do the following. 

```bash
./build.sh --python
```

Run Python tests:

```bash
python -m pip install -e .[dev]
pytest -q
```

## Install

If you are installing from release artifacts, follow instructions [here](https://docs.sima-neat.com/getting-started/install).

## Documentation

Full documentation, guides, and API references are published [here](https://docs.sima-neat.com).
Contributor and agent quality guidance is documented in [AGENTS.md](AGENTS.md).
