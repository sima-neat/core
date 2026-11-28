"""Kernel Node bindings: built-in and runtime-loaded kernels-library operations.

The graph cases run one Kernel Node on the A65 and need a DevKit. Provider
libraries come from beside the Kernels registry this process loaded (see
conftest.py), so they always match its build.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pytest
import pyneat

ARGMAX_PROVIDER = "libsima_kernel_argmax_loaded.so"
RESIZE_LINEAR_PROVIDER = "libsima_kernel_resize_linear.so"

# The kernels repo's tests/provider_resize_linear.cpp reference: a seeded random
# 1600 x 900 BGR frame resized to 960 x 540. Its std::uniform_int_distribution
# <int>(0, 255) over std::mt19937(20261128) is, in libstdc++, the top byte of
# each raw draw; NumPy's legacy MT19937 seeds the same way. Written out so the
# bytes do not depend on a library's distribution algorithm. The digests are
# those of that test's image and of the directly linked operation's output.
RESIZE_INPUT_SHAPE = (900, 1600, 3)
RESIZE_OUTPUT_SHAPE = [540, 960, 3]
RESIZE_INPUT_SHA256 = "24aaaccc8be3f6c7bfdbcb13849c8317c435d78f1b95c8ce8a587fdf1cb81c5b"
RESIZE_OUTPUT_SHA256 = "fc4a2df15af45efcf7affa148d974853c16f9da47bf02933fc403bcedd2a9769"

ARGMAX_PARAMETERS = {"axis": -1, "keepdims": False}


def _argmax_input() -> np.ndarray:
  # Distinct scores (37 is coprime to 128), so the first-maximum tie rule never decides.
  return ((np.arange(128) * 37) % 128).astype(np.float32).reshape(1, 4, 4, 8)


def _resize_input() -> np.ndarray:
  draws = np.random.RandomState(20261128).randint(
      0, 2**32, size=int(np.prod(RESIZE_INPUT_SHAPE)), dtype=np.uint32)
  image = (draws >> 24).astype(np.uint8).reshape(RESIZE_INPUT_SHAPE)
  assert hashlib.sha256(image.tobytes()).hexdigest() == RESIZE_INPUT_SHA256
  return image


def _options(operation: str, **fields) -> pyneat.KernelOptions:
  options = pyneat.KernelOptions()
  options.operation = operation
  for name, value in fields.items():
    setattr(options, name, value)
  return options


def _graph(options: pyneat.KernelOptions) -> pyneat.Graph:
  graph = pyneat.Graph()
  graph.add(pyneat.nodes.input("in"))
  graph.add(pyneat.nodes.kernel(options))
  graph.add(pyneat.nodes.output("out"))
  return graph


def _tensor(values: np.ndarray) -> pyneat.Tensor:
  return pyneat.Tensor.from_numpy(values, copy=True, memory=pyneat.TensorMemory.EV74)


def _run(options: pyneat.KernelOptions, values: np.ndarray) -> list[np.ndarray]:
  tensor = _tensor(values)
  run = _graph(options).build([tensor])
  try:
    assert run.push([tensor])
    outputs = run.pull_tensors(30000)
  finally:
    run.close()
  return [output.to_numpy(copy=True) for output in outputs]


def _expect_rejected(options: pyneat.KernelOptions, values: np.ndarray, needle: str) -> None:
  with pytest.raises(pyneat.NeatError, match=re.escape(needle)):
    _graph(options).build([_tensor(values)])


def test_kernel_options_round_trip():
  options = pyneat.KernelOptions()
  assert options.operation == ""
  assert options.target == pyneat.SimaKernelTarget.A65
  assert options.parameters == {}
  assert options.outputs is None
  assert options.library is None
  parameters = {"flag": True, "count": -3, "scale": 0.5, "mode": "nearest",
                "sizes": [1, 2], "weights": [0.25, 1.0], "empty": []}
  options.parameters = parameters
  assert options.parameters == parameters
  assert type(options.parameters["flag"]) is bool
  assert type(options.parameters["count"]) is int
  options.outputs = [([1, 4, 4], "int32"), ([2], "bfloat16")]
  assert options.outputs == [([1, 4, 4], "int32"), ([2], "bfloat16")]
  options.outputs = None
  assert options.outputs is None
  options.library = "/opt/providers/libexample.so"
  assert options.library == "/opt/providers/libexample.so"
  options.target = pyneat.SimaKernelTarget.EV74
  assert options.target == pyneat.SimaKernelTarget.EV74
  with pytest.raises(TypeError, match="list\\[int\\] or list\\[float\\]"):
    options.parameters = {"mixed": [1, 2.5]}
  with pytest.raises(TypeError):
    options.parameters = {"none": None}
  with pytest.raises(ValueError, match="uint8, int8, int32, float32 or bfloat16"):
    options.outputs = [([4], "float64")]
  with pytest.raises(ValueError, match="operation must name"):
    pyneat.nodes.kernel(pyneat.KernelOptions())
  assert pyneat.nodes.kernel(_options("argmax")).kind() == "Kernel"


def test_kernel_node_builtin_argmax():
  values = _argmax_input()
  options = _options("argmax", parameters=ARGMAX_PARAMETERS, outputs=[([1, 4, 4], "int32")])
  (result,) = _run(options, values)
  assert result.dtype == np.int32
  np.testing.assert_array_equal(result, np.argmax(values, axis=-1).astype(np.int32))


def test_kernel_node_loaded_argmax(kernels_provider):
  values = _argmax_input()
  options = _options("argmax_loaded", parameters=ARGMAX_PARAMETERS,
                     library=kernels_provider(ARGMAX_PROVIDER))
  (result,) = _run(options, values)
  assert result.dtype == np.int32
  np.testing.assert_array_equal(result, np.argmax(values, axis=-1).astype(np.int32))


def test_kernel_node_loaded_resize_linear(kernels_provider):
  options = _options("resize_linear", outputs=[(RESIZE_OUTPUT_SHAPE, "uint8")],
                     library=kernels_provider(RESIZE_LINEAR_PROVIDER))
  (result,) = _run(options, _resize_input())
  assert result.dtype == np.uint8 and list(result.shape) == RESIZE_OUTPUT_SHAPE
  assert hashlib.sha256(np.ascontiguousarray(result).tobytes()).hexdigest() == RESIZE_OUTPUT_SHA256


def test_kernel_node_missing_outputs():
  _expect_rejected(_options("argmax", parameters=ARGMAX_PARAMETERS), _argmax_input(),
                   "argmax: has no output inference; pass outputs")


def test_kernel_node_inference_mismatch(kernels_provider):
  options = _options("argmax_loaded", parameters=ARGMAX_PARAMETERS,
                     outputs=[([1, 4, 4, 1], "int32")], library=kernels_provider(ARGMAX_PROVIDER))
  _expect_rejected(options, _argmax_input(),
                   "argmax_loaded: stated outputs differ from the provider's inference")


def test_kernel_node_bad_library(kernels_provider):
  # A file that does not exist; nothing is created.
  path = kernels_provider("libsima_kernel_absent.so")
  with pytest.raises(RuntimeError, match=re.escape(path)):
    pyneat.load_kernel_library(path)
  options = _options("argmax", parameters=ARGMAX_PARAMETERS, outputs=[([1, 4, 4], "int32")],
                     library=path)
  _expect_rejected(options, _argmax_input(), path)
