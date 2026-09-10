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


def test_load_metadata_from_mla_only_model():
  model = os.environ.get("SIMAPCIE_YOLOV8_MODEL")
  if not model:
    pytest.skip("SIMAPCIE_YOLOV8_MODEL is not set")

  model_path = Path(model)
  if not model_path.is_file():
    pytest.skip(f"SIMAPCIE_YOLOV8_MODEL does not exist: {model_path}")

  options = pcie.ModelOptions()
  options.mla_only = True
  try:
    info = pcie.Model(str(model_path), options).info()
  except (RuntimeError, ValueError) as error:
    wrong_build = ("does not support stage", "hybrid host/card quantization", "must be INT8")
    if not any(reason in str(error) for reason in wrong_build):
      raise
    pytest.skip(f"{model_path.name} is not an MLA-only capable build: {error}")

  assert len(info.inputs) == 1
  assert info.inputs[0].dtype == "INT8"
  assert info.inputs[0].size_bytes == math.prod(info.inputs[0].shape)
  assert info.outputs
  assert all(tensor.dtype == "INT8" for tensor in info.outputs)
  assert all(tensor.size_bytes == math.prod(tensor.shape) for tensor in info.outputs)
  for tensor in info.inputs + info.outputs:
    assert tensor.quant is not None
    assert len(tensor.quant.scales) == 1
    assert len(tensor.quant.zero_points) == 1
    assert tensor.quant.scales[0] > 0.0

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
