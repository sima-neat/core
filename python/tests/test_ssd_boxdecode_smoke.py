"""Python smoke test for the SSD box-decode surface.

Without a model pack: the BoxDecodeType.Ssd binding, the ResizeMode enum used by
the stretch-resize contract, and ModelOptions mutability.

With an SSD pack (SIMA_SSD_TAR / SIMA_SSD_MOBILENET_TAR): a stretch resize builds
and decodes while a letterbox resize is rejected. Skipped when no pack env is set.
"""
import os
from pathlib import Path

import numpy as np
import pytest

import pyneat


def _ssd_tar():
  for env in ("SIMA_SSD_TAR", "SIMA_SSD_MOBILENET_TAR"):
    value = os.environ.get(env, "")
    if value and Path(value).is_file():
      return Path(value)
  return None


def test_ssd_boxdecode_type_binding():
  # The SSD decode type is exposed and round-trips by name.
  assert hasattr(pyneat.BoxDecodeType, "Ssd")
  assert pyneat.BoxDecodeType.Ssd == pyneat.BoxDecodeType.Ssd
  # It is distinct from the YOLO/DETR families it sits alongside.
  assert pyneat.BoxDecodeType.Ssd != pyneat.BoxDecodeType.YoloV8


def test_resize_mode_enum_for_ssd_contract():
  # The enum the stretch contract validates against must expose all three modes.
  for name in ("Stretch", "Letterbox", "Crop"):
    assert hasattr(pyneat.ResizeMode, name), f"ResizeMode.{name} missing"


def test_ssd_model_options_are_mutable():
  opt = pyneat.ModelOptions()
  opt.decode_type = pyneat.BoxDecodeType.Ssd
  opt.num_classes = 91
  opt.score_threshold = 0.3
  opt.nms_iou_threshold = 0.6
  opt.top_k = 100
  opt.preprocess.resize.mode = pyneat.ResizeMode.Stretch
  assert opt.decode_type == pyneat.BoxDecodeType.Ssd
  assert opt.num_classes == 91
  assert opt.preprocess.resize.mode == pyneat.ResizeMode.Stretch


def _ssd_options(resize_mode):
  opt = pyneat.ModelOptions()
  opt.preprocess.kind = pyneat.InputKind.Image
  opt.preprocess.resize.mode = resize_mode
  opt.preprocess.resize.width = 300
  opt.preprocess.resize.height = 300
  opt.decode_type = pyneat.BoxDecodeType.Ssd
  opt.score_threshold = 0.3
  opt.nms_iou_threshold = 0.6
  opt.top_k = 100
  return opt


def test_ssd_stretch_builds_and_letterbox_rejected():
  tar = _ssd_tar()
  if tar is None:
    pytest.skip("no SSD pack; set SIMA_SSD_TAR or SIMA_SSD_MOBILENET_TAR")

  # Stretch is supported: the model builds and the box-decode node accepts it.
  try:
    model = pyneat.Model(str(tar), _ssd_options(pyneat.ResizeMode.Stretch))
    assert model is not None
    assert pyneat.nodes.sima_box_decode(model, decode_type=pyneat.BoxDecodeType.Ssd) is not None
  except Exception as exc:  # pragma: no cover - environment dependent
    if "dispatcher" in str(exc).lower():
      pytest.skip(f"dispatcher unavailable: {exc}")
    raise

  # The contract is enforced when the node is built, not by the Model constructor.
  # Path 1: the model's own resolved letterbox preprocess plan.
  letterbox_model = pyneat.Model(str(tar), _ssd_options(pyneat.ResizeMode.Letterbox))
  with pytest.raises(Exception) as excinfo:
    pyneat.nodes.sima_box_decode(letterbox_model, decode_type=pyneat.BoxDecodeType.Ssd)
  assert "stretch" in str(excinfo.value).lower()

  # Path 2: an explicit per-node override, which wins over the plan.
  with pytest.raises(Exception) as excinfo:
    pyneat.nodes.sima_box_decode(
      model,
      decode_type=pyneat.BoxDecodeType.Ssd,
      resize_mode=pyneat.ResizeMode.Letterbox,
    )
  assert "stretch" in str(excinfo.value).lower()
