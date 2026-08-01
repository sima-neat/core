"""Long, device-dependent Python coverage for strict SSD recipes.

This module is intentionally installed outside ``python/tests`` and run only by
the Modalix long-test lane. Missing model/runtime fixtures may therefore skip
without weakening the default strict Python suite.
"""

import os
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2", exc_type=Exception)
np = pytest.importorskip("numpy", exc_type=Exception)
pyneat = pytest.importorskip("pyneat", exc_type=Exception)

pytestmark = pytest.mark.long


def _ssd_mobilenet_tar():
  value = os.environ.get("SIMA_SSD_MOBILENET_TAR", "")
  if value and Path(value).is_file():
    return Path(value)

  roots = [Path.cwd(), *Path.cwd().parents, Path("/tmp/sima-ssd-mobilenet")]
  for root in roots:
    for candidate in (
      root / "tmp" / "ssd_mobilenet_v2_heads_mpk.tar.gz",
      root / "ssd_mobilenet_v2_heads_mpk.tar.gz",
    ):
      if candidate.is_file():
        return candidate
  return None


def _ssd_tar():
  for env in ("SIMA_SSD_TAR", "SIMA_SSD_MOBILENET_TAR"):
    value = os.environ.get(env, "")
    if value and Path(value).is_file():
      return Path(value)
  return _ssd_mobilenet_tar()


def _zidane_image():
  override = os.environ.get("SIMA_TEST_IMAGE", "")
  candidates = [Path(override)] if override else []
  for root in (Path.cwd(), *Path.cwd().parents):
    candidates.extend(
      (
        root / "tmp/coco_sample.jpg",
      )
    )
  candidates.extend(
    (
      Path("/tmp/sima-coco-sample/coco_sample.jpg"),
      Path("/tmp/coco_sample.jpg"),
    )
  )
  for candidate in candidates:
    if candidate.is_file():
      image = cv2.imread(str(candidate), cv2.IMREAD_COLOR)
      if image is None:
        raise AssertionError(f"failed to decode SSD test image: {candidate}")
      height, width = image.shape[:2]
      assert (width, height) == (1280, 720), (
        "SSD MobileNet golden boxes require the 1280x720 Zidane fixture; "
        f"got {width}x{height} from {candidate}"
      )
      return image
  pytest.skip(
    "missing 1280x720 Zidane SSD fixture; set SIMA_TEST_IMAGE or stage "
    "tmp/coco_sample.jpg"
  )


def _ssd_options(resize_mode, num_classes=0):
  opt = pyneat.ModelOptions()
  opt.preprocess.kind = pyneat.InputKind.Image
  opt.preprocess.enable = pyneat.AutoFlag.On
  opt.preprocess.resize.enable = pyneat.AutoFlag.On
  opt.preprocess.resize.mode = resize_mode
  opt.preprocess.resize.width = 300
  opt.preprocess.resize.height = 300
  opt.preprocess.color_convert.enable = pyneat.AutoFlag.On
  opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.BGR
  opt.preprocess.color_convert.output_format = pyneat.PreprocessColorFormat.RGB
  opt.preprocess.normalize.enable = pyneat.AutoFlag.On
  opt.preprocess.normalize.mean = [0.5, 0.5, 0.5]
  opt.preprocess.normalize.stddev = [0.5, 0.5, 0.5]
  opt.preprocess.normalize.has_explicit_stats = True
  opt.decode_type = pyneat.BoxDecodeType.Ssd
  if num_classes > 0:
    opt.num_classes = num_classes
  opt.score_threshold = 0.3
  opt.nms_iou_threshold = 0.6
  opt.top_k = 24
  return opt


def _iou(box, expected):
  left = max(float(box[0]), expected[0])
  top = max(float(box[1]), expected[1])
  right = min(float(box[2]), expected[2])
  bottom = min(float(box[3]), expected[3])
  intersection = max(0.0, right - left) * max(0.0, bottom - top)
  box_area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
  expected_area = (expected[2] - expected[0]) * (expected[3] - expected[1])
  union = box_area + expected_area - intersection
  return intersection / union if union > 0.0 else 0.0


def test_ssd_stretch_builds_and_letterbox_rejected():
  tar = _ssd_tar()
  if tar is None:
    pytest.skip("no SSD pack; set SIMA_SSD_TAR or SIMA_SSD_MOBILENET_TAR")

  try:
    model = pyneat.Model(str(tar), _ssd_options(pyneat.ResizeMode.Stretch))
    assert model is not None
    assert pyneat.nodes.sima_box_decode(
      model, decode_type=pyneat.BoxDecodeType.Ssd
    ) is not None
  except Exception as exc:  # pragma: no cover - target availability
    if "dispatcher" in str(exc).lower():
      pytest.skip(f"dispatcher unavailable: {exc}")
    raise

  letterbox_model = pyneat.Model(
    str(tar), _ssd_options(pyneat.ResizeMode.Letterbox)
  )
  with pytest.raises(Exception) as excinfo:
    pyneat.nodes.sima_box_decode(
      letterbox_model, decode_type=pyneat.BoxDecodeType.Ssd
    )
  assert "stretch" in str(excinfo.value).lower()

  with pytest.raises(Exception) as excinfo:
    pyneat.nodes.sima_box_decode(
      model,
      decode_type=pyneat.BoxDecodeType.Ssd,
      resize_mode=pyneat.ResizeMode.Letterbox,
    )
  assert "stretch" in str(excinfo.value).lower()


def test_ssd_mobilenet_full_python_pipeline_matches_people_golden():
  tar = _ssd_mobilenet_tar()
  if tar is None:
    pytest.skip(
      "prepared MobileNet SSD pack not staged; set SIMA_SSD_MOBILENET_TAR"
    )

  image = _zidane_image()
  height, width = image.shape[:2]
  model = pyneat.Model(
    str(tar), _ssd_options(pyneat.ResizeMode.Stretch, num_classes=91)
  )
  tensor = pyneat.Tensor.from_numpy(
    image,
    copy=True,
    image_format=pyneat.PixelFormat.BGR,
  )

  outputs = model.run([tensor], timeout_ms=30000)
  assert outputs, "SSD Python route returned no BBOX output"
  boxes = pyneat.decode_bbox(outputs, clamp_to=(width, height))[0].to_numpy()

  assert boxes.dtype == np.float32
  assert boxes.ndim == 2 and boxes.shape[1] == 6
  assert boxes.shape[0] > 0
  assert np.isfinite(boxes).all()
  assert (boxes[:, 0] >= 0).all() and (boxes[:, 2] <= width).all()
  assert (boxes[:, 1] >= 0).all() and (boxes[:, 3] <= height).all()
  assert (boxes[:, 4] >= 0.0).all() and (boxes[:, 4] <= 1.0).all()
  assert (boxes[:, 5] >= 1).all() and (boxes[:, 5] < 91).all()

  people = boxes[boxes[:, 5] == 1]
  expected_people = (
    (737.0, 54.0, 1147.0, 699.0),
    (96.0, 194.0, 1105.0, 707.0),
  )
  assert people.shape[0] >= len(expected_people)
  for expected in expected_people:
    assert max(_iou(box, expected) for box in people) >= 0.70
