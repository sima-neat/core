"""Strict, device-independent Python API coverage for SSD box decode.

Fixture- and dispatcher-dependent behavior lives in the explicitly long
``python/tests_long/test_ssd_boxdecode_device.py`` lane.
"""

import pyneat


def test_ssd_boxdecode_type_binding():
  # Recipe selection is intentionally internal; the stable public/runtime family is SSD.
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
