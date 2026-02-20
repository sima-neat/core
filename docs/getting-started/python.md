---
title: Python (pyneat)
description: Build and use pyneat bindings from this repository
sidebar_position: 4
---

# Python (`pyneat`)

`pyneat` is the Python binding layer for SiMa NEAT.

It exposes the core runtime APIs:

- `Tensor`
- `Session` and `Run`
- `Model`
- Node and node-group factory helpers
- MPK inspection/extraction helpers

The package uses `nanobind` and is built with `scikit-build-core`.

See the [Python API Reference](/reference/pythonapi/modules/pyneat) for module-based docs.

## Prerequisites

`pyneat` links against the same native dependencies as the C++ library, including:

- GStreamer development/runtime packages
- OpenCV development/runtime packages
- C++ toolchain (`cmake`, compiler, pkg-config)

See [Build](./build) for host setup guidance.

## Install From Source

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

## Quickstart

```python
import numpy as np
import pyneat as pn

arr = np.random.rand(1, 224, 224, 3).astype(np.float32)
tensor = pn.Tensor.from_numpy(arr, copy=False)

session = pn.Session()
session.add(pn.nodes.input())
session.add(pn.nodes.output())

# build() / run() require a valid NEAT pipeline/runtime environment
print(session.describe_backend())
```

You can also pass NumPy/PyTorch tensors directly to `run(...)` and `build(...)`.
By default this follows zero-copy DLPack paths unless you opt into `copy=True`:

```python
out1 = model.run(arr)                    # no timeout override
out2 = model.run(arr, timeout_ms=2000)  # explicit timeout

torch_tensor = torch.randint(0, 255, (224, 224, 3), dtype=torch.uint8)
out3 = model.run(torch_tensor)
out4 = model.run(torch_tensor, timeout_ms=2000)

# Build overloads accept Tensor/Sample/NumPy/Torch natively in C++ bindings
# (wrappers now only provide compatibility fallback on older cores)
run = session.build(arr, layout=pn.TensorLayout.HWC, image_format=pn.PixelFormat.RGB)
runner = model.build(torch_tensor, layout=pn.TensorLayout.CHW)
```

## NumPy and PyTorch Interop

`Tensor` supports DLPack-based interop:

- `Tensor.from_numpy(...)`
- `Tensor.to_numpy(...)`
- `Tensor.from_torch(...)`
- `Tensor.to_torch(...)`
- `Tensor.from_dlpack(...)`
- `Tensor.__dlpack__()`

This keeps interop paths explicit and enables zero-copy where the backend data layout permits it.

## Tests

```bash
pytest -q python/tests
```

## Packaging

`pyproject.toml` in the repository root defines wheel/sdist build configuration for `pyneat`.
