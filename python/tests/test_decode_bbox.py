"""Tests for pyneat.decode_bbox (TensorList -> TensorList).

decode_bbox takes a list of BBOX-format tensors (e.g. a model's run output) and
returns a list of decoded-boxes tensors, positional 1:1. Each output is a float32
tensor of shape [num_detections, 6] with columns (x1, y1, x2, y2, score, class_id).

Validated against the byte-level _parse_bbox_payload oracle in
test_real_model_fixtures.py. Gated on SIMA_YOLO_TAR; image from SIMA_TEST_IMAGE
or the standard fixture lookup.
"""
import math
import os
from pathlib import Path

import numpy as np
import pytest

import model_fixture_helpers as model_fixtures
import pyneat
from test_real_model_fixtures import (
    _decode_rgb_image,
    _fixture_image_path,
    _parse_bbox_payload,
    _rgb_tensor,
)

BOX_COLUMNS = 6  # x1, y1, x2, y2, score, class_id


def _yolov8_image_model(model_tar):
  """Image-input YOLOv8 model with the full ModelOptions the doc example uses."""
  opt = pyneat.ModelOptions()
  opt.preprocess.kind   = pyneat.InputKind.Image
  opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
  opt.decode_type       = pyneat.BoxDecodeType.YoloV8
  opt.score_threshold   = 0.25
  opt.nms_iou_threshold = 0.45
  opt.top_k             = 100
  return pyneat.Model(str(model_tar), opt)


@pytest.fixture(scope="module")
def yolov8_run():
  """Run YOLOv8 once and yield (outputs: TensorList, image_w, image_h)."""
  model_tar = model_fixtures.strict_model_tar_path("SIMA_YOLO_TAR")
  model = _yolov8_image_model(model_tar)

  override = os.environ.get("SIMA_TEST_IMAGE")
  if override:
    image_path = Path(override)
    if not image_path.is_file():
      pytest.skip(f"SIMA_TEST_IMAGE={override} is not a file")
  else:
    image_path = _fixture_image_path(Path("test.jpg"))
  image = _decode_rgb_image(image_path)
  image_h, image_w = image.shape[:2]

  outputs = model.run([_rgb_tensor(image)], timeout_ms=20000)
  return outputs, image_w, image_h


def test_decode_bbox_preserves_tensor_count(yolov8_run):
  outputs, _w, _h = yolov8_run
  decoded = pyneat.decode_bbox(outputs)
  assert isinstance(decoded, list)
  assert len(decoded) == len(outputs), "decode_bbox must return one tensor per input"


def test_decoded_tensor_shape_and_dtype(yolov8_run):
  outputs, _w, _h = yolov8_run
  decoded = pyneat.decode_bbox(outputs)
  arr = decoded[0].to_numpy()
  assert arr.dtype == np.float32
  assert arr.ndim == 2
  assert arr.shape[1] == BOX_COLUMNS
  assert arr.shape[0] >= 1, "expected at least one detection from yolo_v8s"


def test_decoded_values_match_struct_unpack_oracle(yolov8_run):
  outputs, img_w, img_h = yolov8_run
  payload = outputs[0].copy_payload_bytes()
  oracle = _parse_bbox_payload(payload, img_w, img_h, min_score=0.0)

  arr = pyneat.decode_bbox(outputs, clamp_to=(img_w, img_h))[0].to_numpy()

  assert arr.shape[0] == len(oracle)
  for row, ref in zip(arr, oracle):
    assert math.isclose(float(row[0]), ref["x1"], abs_tol=1e-6)
    assert math.isclose(float(row[1]), ref["y1"], abs_tol=1e-6)
    assert math.isclose(float(row[2]), ref["x2"], abs_tol=1e-6)
    assert math.isclose(float(row[3]), ref["y2"], abs_tol=1e-6)
    assert math.isclose(float(row[4]), ref["score"], rel_tol=1e-6, abs_tol=1e-6)
    assert int(row[5]) == ref["class_id"]


def test_decode_bbox_raises_on_non_bbox_tensor():
  rgb = np.zeros((64, 64, 3), dtype=np.uint8)
  rgb_tensor = pyneat.Tensor.from_numpy(rgb, copy=True, image_format=pyneat.PixelFormat.RGB)
  with pytest.raises(TypeError) as info:
    pyneat.decode_bbox([rgb_tensor])
  assert "BBOX" in str(info.value)


def test_decode_bbox_top_k_caps_per_tensor(yolov8_run):
  outputs, _w, _h = yolov8_run
  full = pyneat.decode_bbox(outputs)[0].to_numpy()
  if full.shape[0] < 2:
    pytest.skip("need at least 2 detections to exercise top_k cap")

  capped = pyneat.decode_bbox(outputs, top_k=1)[0].to_numpy()
  assert capped.shape[0] == 1
  assert int(capped[0][5]) == int(full[0][5])
  assert math.isclose(float(capped[0][4]), float(full[0][4]), rel_tol=1e-6, abs_tol=1e-6)


def test_decode_bbox_clamp_to_clips_coordinates(yolov8_run):
  outputs, img_w, img_h = yolov8_run
  arr = pyneat.decode_bbox(outputs, clamp_to=(img_w, img_h))[0].to_numpy()
  assert (arr[:, 0] >= 0).all() and (arr[:, 0] <= img_w).all()
  assert (arr[:, 2] >= 0).all() and (arr[:, 2] <= img_w).all()
  assert (arr[:, 1] >= 0).all() and (arr[:, 1] <= img_h).all()
  assert (arr[:, 3] >= 0).all() and (arr[:, 3] <= img_h).all()


def test_decode_bbox_recognizes_untagged_wire_tensor(yolov8_run):
  """The current EV74 box-decode producer does not stamp semantic.detection on
  its output; decode_bbox still recognizes the canonical rank-1 UInt8 BBOX wire
  tensor and decodes it. (When the producer is updated to set
  semantic.detection.format = 'BBOX', the tag becomes the primary signal and
  this wire-shape fallback becomes belt-and-suspenders.)"""
  outputs, _w, _h = yolov8_run
  bbox = outputs[0]
  assert len(bbox.shape) == 1 and bbox.dtype == pyneat.TensorDType.UInt8
  decoded = pyneat.decode_bbox(outputs)
  assert len(decoded) == len(outputs)
