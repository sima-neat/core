import math
import os
from pathlib import Path

import pytest

import pyneatpcie as pcie


def test_load_metadata_from_yolov8_model():
  model = os.environ.get("SIMAPCIE_YOLOV8_MODEL")
  if not model:
    pytest.skip("SIMAPCIE_YOLOV8_MODEL is not set")

  model_path = Path(model)
  if not model_path.is_file():
    pytest.skip(f"SIMAPCIE_YOLOV8_MODEL does not exist: {model_path}")

  model = pcie.Model(str(model_path))
  info = model.info()

  assert [tensor.name for tensor in info.inputs] == ["images"]
  assert info.inputs[0].dtype == "FP32"
  assert info.inputs[0].shape == [640, 640, 3]
  assert [tensor.name for tensor in info.outputs] == [
      "bbox_0",
      "bbox_1",
      "bbox_2",
      "class_prob_0",
      "class_prob_1",
      "class_prob_2",
  ]
  assert all(tensor.size_bytes > 0 for tensor in info.outputs)
  assert all(tensor.quant is None for tensor in info.inputs + info.outputs)


@pytest.mark.parametrize("model_env", ["SIMAPCIE_MLA_ONLY_MODEL", "SIMAPCIE_MLA_ONLY_BF16_MODEL"])
def test_load_metadata_from_mla_only_model(model_env):
  model = os.environ.get(model_env)
  if model_env == "SIMAPCIE_MLA_ONLY_MODEL":
    model = model or os.environ.get("SIMAPCIE_YOLOV8_MODEL")
  if not model:
    pytest.skip(f"{model_env} is not set")

  model_path = Path(model)
  if not model_path.is_file():
    pytest.skip(f"MLA-only model does not exist: {model_path}")

  options = pcie.ModelOptions()
  options.mla_only = True
  info = pcie.Model(str(model_path), options).info()

  dtype = info.inputs[0].dtype
  itemsize = {"INT8": 1, "BF16": 2}[dtype]
  assert len(info.inputs) == 1
  assert info.inputs[0].size_bytes == math.prod(info.inputs[0].shape) * itemsize
  assert info.outputs
  assert all(tensor.dtype == dtype for tensor in info.outputs)
  assert all(tensor.size_bytes == math.prod(tensor.shape) * itemsize for tensor in info.outputs)
  for tensor in info.inputs + info.outputs:
    if dtype == "BF16":
      assert tensor.quant is None
      continue
    assert tensor.quant is not None
    assert tensor.quant.scale > 0.0

  default_info = pcie.Model(str(model_path)).info()
  assert default_info.inputs[0].dtype == "FP32"
  assert default_info.inputs[0].shape == info.inputs[0].shape
  assert [tensor.name for tensor in default_info.outputs] == [
      tensor.name for tensor in info.outputs
  ]
  assert [tensor.shape for tensor in default_info.outputs] == [
      tensor.shape for tensor in info.outputs
  ]

  options.preprocess.kind = pcie.InputKind.Image
  with pytest.raises(ValueError, match="mla_only"):
    pcie.Model(str(model_path), options)
