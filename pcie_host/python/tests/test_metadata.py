import os
from pathlib import Path

import pytest

import pypciehost as pcie


def test_load_metadata_from_yolov8_model():
  model = os.environ.get("SIMAPCIE_YOLOV8_MODEL")
  if not model:
    pytest.skip("SIMAPCIE_YOLOV8_MODEL is not set")

  model_path = Path(model)
  if not model_path.is_file():
    pytest.skip(f"SIMAPCIE_YOLOV8_MODEL does not exist: {model_path}")

  host = pcie.SimaPCIeHost()
  info = host.load_metadata(str(model_path))

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
