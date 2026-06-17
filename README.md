# SiMa NEAT Framework

[![Vulcan CI](https://github.com/sima-neat/core/actions/workflows/vulcan-ci.yml/badge.svg)](https://github.com/sima-neat/core/actions/workflows/vulcan-ci.yml)
![SDK](https://img.shields.io/badge/SDK-2.0-green)
![Language](https://img.shields.io/badge/C%2B%2B-20-informational)

SiMa NEAT is a C++20 library for building, validating, running, and debugging GStreamer pipelines with a typed, composable API.

It helps teams ship production media/ML pipelines with reproducible pipeline generation, strong diagnostics, and clean C++ integration.

<img src="docs/images/concepts.jpg" alt="NEAT concepts diagram" width="80%" />

## Programming Model

NEAT is built around three core concepts:

- `Model`: loads a compiled model archive (`.tar.gz`) and exposes reusable `Graph`
  fragments (`preprocess`, `inference`, `postprocess`, or the full model route).
- `Graph` + `Run`: composes typed `Node`, `Model`, and reusable `Graph` fragments into
  a deterministic pipeline, then executes it in `Sync` or `Async` mode via push/pull APIs.
- `Tensor` + `Sample`: `Tensor` is the typed data container (dtype/layout/shape/device/storage), while `Sample` wraps tensors with stream metadata (timestamps, caps, routing info, bundles).

Typical flow:

1. Build a `Model` from a compiled `.tar.gz` model archive.
2. Add model routes, reusable fragments, and optional custom nodes to a `Graph`.
3. Use `run(...)` for one-shot execution, or `build(...)` to get a reusable `Run` for `push()/pull()`.

```cpp
#include <neat.h>

simaai::neat::Model model("resnet_50.tar.gz");

simaai::neat::Graph graph;
graph.add(simaai::neat::nodes::Input());
graph.add(model);
graph.add(simaai::neat::nodes::Output());

auto out = graph.run(simaai::neat::TensorList{input_tensor});
```

For unified runtime reporting, measure an explicit workload window:

```cpp
simaai::neat::RunOptions run_opt;
run_opt.enable_board_power();

auto run = graph.build(simaai::neat::TensorList{input_tensor}, run_opt);
auto scope = run.start_measurement();
run.push(simaai::neat::TensorList{input_tensor});
(void)run.pull_tensors(5000);
auto report = scope.stop();
std::cout << report.to_text();
```

`start_measurement()` / `MeasureReport` is the single public surface for latency,
throughput, counters, plugin/edge timing, and optional board power.

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
```

Build output is generated under `build/`.

## Python Bindings (`pyneat`)

This repository now includes a production-oriented Python binding layer powered by `nanobind`.

Key points:

- Python package source lives under `python/`.
- Extension module is `pyneat._pyneat_core`.
- High-level Python wrappers provide ergonomic APIs on top of the C++ layer.
- NumPy/PyTorch interop is supported through DLPack (`Tensor.from_numpy`, `Tensor.to_numpy`, `Tensor.from_torch`, `Tensor.to_torch`).

Build/install from source:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

Run Python tests:

```bash
python -m pip install -e .[dev]
pytest -q
```

## Install

If you are installing from release artifacts for C++ development, install the runtime and development `.deb` packages and extract prebuilt examples, tutorials from `.tar.gz`:

```bash
sudo apt install ./sima-neat-*-Linux-core.deb ./sima-neat-*-Linux-dev.deb
mkdir -p "${HOME}/sima-neat-extras"
tar -xzf ./sima-neat-*-Linux-extras.tar.gz -C "${HOME}/sima-neat-extras"
```

## Documentation

Full documentation, guides, and API references are published [here](https://neat.modalix.info).
Contributor and agent quality guidance is documented in `AGENTS.md` at the repository root.
