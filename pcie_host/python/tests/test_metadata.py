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
  model = os.environ.get("SIMAPCIE_MLA_INT8_MODEL")
  if not model:
    pytest.skip("SIMAPCIE_MLA_INT8_MODEL is not set")

  model_path = Path(model)
  if not model_path.is_file():
    pytest.skip(f"SIMAPCIE_MLA_INT8_MODEL does not exist: {model_path}")

  options = pcie.ModelOptions()
  options.mla_only = True
  info = pcie.Model(str(model_path), options).info()

  assert len(info.inputs) == 1
  assert info.inputs[0].dtype == "INT8"
  assert all(tensor.dtype == "INT8" for tensor in info.outputs)
  for tensor in info.inputs + info.outputs:
    assert tensor.quant is not None
    assert len(tensor.quant.scales) == 1
    assert len(tensor.quant.zero_points) == 1
    assert tensor.quant.scales[0] > 0.0
