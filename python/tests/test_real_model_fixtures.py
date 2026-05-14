import struct
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

import model_fixture_helpers as model_fixtures
import pyneat


ROOT = model_fixtures.ROOT
SANDBOX_API_TESTS = ROOT / "sandbox" / "api-tests"
MODELZOO_PLATFORM_VERSION = model_fixtures.MODELZOO_PLATFORM_VERSION
BOXDECODE_THRESHOLDS = (0.50, 0.60, 0.70)
TOP_SCORE_HIGHLIGHT_COUNT = 3


def _env_path(name: str) -> Path:
  return model_fixtures.strict_model_tar_path(name)


def test_model_fixture_path_helper_skips_when_unresolved(monkeypatch):
  model_fixtures._MODEL_PATH_CACHE.clear()
  monkeypatch.delenv("SIMA_RESNET50_TAR", raising=False)
  monkeypatch.setattr(
      model_fixtures,
      "resolve_model_tar",
      lambda _name: None,
      raising=False,
  )

  with pytest.raises(pytest.skip.Exception, match="SIMA_RESNET50_TAR"):
    model_fixtures.fixture_model_path("SIMA_RESNET50_TAR")


def test_image_fixture_path_helper_falls_back_to_cwd_when_root_missing(monkeypatch, tmp_path):
  image_path = tmp_path / "test.jpg"
  image_path.write_bytes(b"fixture")

  monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path / "missing")
  monkeypatch.chdir(tmp_path)

  assert _fixture_image_path(Path("test.jpg")) == image_path


def _candidate_fixture_roots() -> list[Path]:
  roots: list[Path] = []

  def append_unique(path: Path) -> None:
    if path in roots:
      return
    roots.append(path)

  for base in (ROOT, Path.cwd()):
    append_unique(base)
    for parent in base.parents:
      append_unique(parent)

  return roots


def _resolve_fixture_image(rel_path: Path) -> Path | None:
  rel_path = Path(rel_path)
  if rel_path.is_absolute():
    return rel_path if rel_path.is_file() else None

  for root in _candidate_fixture_roots():
    candidate = root / rel_path
    if candidate.is_file():
      return candidate

  return None


def _fixture_image_path(rel_path: Path) -> Path:
  path = _resolve_fixture_image(rel_path)
  if path is not None:
    return path
  pytest.skip(f"missing real image fixture: {rel_path}")


def _decode_rgb_image(path: Path) -> np.ndarray:
  image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
  if image_bgr is None:
    raise AssertionError(f"failed to decode image fixture: {path}")
  return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _tensor_input_model(
    model_tar: Path, decode_type: pyneat.BoxDecodeType = pyneat.BoxDecodeType.Unspecified
) -> pyneat.Model:
  opt = pyneat.ModelOptions()
  opt.preprocess.kind = pyneat.InputKind.Tensor
  opt.preprocess.enable = pyneat.AutoFlag.Off
  opt.decode_type = decode_type
  return pyneat.Model(str(model_tar), opt)


def _image_input_model(
    model_tar: Path, decode_type: pyneat.BoxDecodeType = pyneat.BoxDecodeType.Unspecified
) -> pyneat.Model:
  opt = pyneat.ModelOptions()
  opt.preprocess.kind = pyneat.InputKind.Image
  opt.preprocess.enable = pyneat.AutoFlag.On
  opt.preprocess.resize.enable = pyneat.AutoFlag.On
  opt.preprocess.normalize.enable = pyneat.AutoFlag.On
  opt.decode_type = decode_type
  return pyneat.Model(str(model_tar), opt)


def _rgb_tensor(image: np.ndarray) -> pyneat.Tensor:
  return pyneat.Tensor.from_numpy(image, copy=True, image_format=pyneat.PixelFormat.RGB)


def _run_model_on_image(model: pyneat.Model, image: np.ndarray, *parts) -> pyneat.Sample:
  session = pyneat.Session()
  session.add(pyneat.nodes.input(model.input_appsrc_options(False)))
  for part in parts:
    session.add(part)
  session.add(pyneat.nodes.output())
  input_tensor = _rgb_tensor(image)
  runner = session.build(input_tensor)
  try:
    return runner.run(input_tensor, timeout_ms=30000)
  finally:
    runner.close()


def _custom_preproc_node(model: pyneat.Model, image: np.ndarray):
  pre = pyneat.PreprocOptions(model)
  channels = image.shape[2] if image.ndim >= 3 else 1
  pre.input_shape = [image.shape[0], image.shape[1], channels]
  return pyneat.nodes.preproc(pre)


def _cpu_quanttess_input(model: pyneat.Model, image: np.ndarray) -> np.ndarray:
  pre = pyneat.PreprocOptions(model)
  dst_h = int(pre.output_shape[0])
  dst_w = int(pre.output_shape[1])
  aspect_ratio = bool(pre.aspect_ratio)
  padding_type = str(pre.padding_type or "CENTER").upper()
  src_h, src_w = image.shape[:2]

  if aspect_ratio:
    scale = min(dst_w / src_w, dst_h / src_h)
    scaled_w = max(1, int(round(src_w * scale)))
    scaled_h = max(1, int(round(src_h * scale)))
  else:
    scaled_w = dst_w
    scaled_h = dst_h

  pad_x = 0
  pad_y = 0
  if padding_type == "CENTER":
    pad_x = (dst_w - scaled_w) // 2
    pad_y = (dst_h - scaled_h) // 2

  resized = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
  image_u8 = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
  image_u8[pad_y : pad_y + scaled_h, pad_x : pad_x + scaled_w] = resized
  return image_u8.astype(np.float32) / 255.0


def _extract_bbox_payload(sample: pyneat.Sample) -> bytes:
  stack = [sample]
  while stack:
    current = stack.pop()
    stack.extend(reversed(list(current.fields)))
    if current.kind != pyneat.SampleKind.Tensor or current.tensor is None:
      continue
    fmt = (current.payload_tag or current.format or "").upper()
    if fmt and fmt != "BBOX":
      continue
    payload = current.tensor.copy_payload_bytes()
    if payload:
      return payload
  raise AssertionError("failed to locate BBOX payload in sample")


def _parse_bbox_payload(payload: bytes, img_w: int, img_h: int, min_score: float) -> list[dict]:
  if len(payload) < 4:
    return []
  count = min(struct.unpack_from("<I", payload, 0)[0], (len(payload) - 4) // 24)
  boxes = []
  offset = 4
  for _ in range(count):
    x, y, w, h, score, cls_id = struct.unpack_from("<iiiifi", payload, offset)
    offset += 24
    x1 = max(0.0, min(float(img_w), float(x)))
    y1 = max(0.0, min(float(img_h), float(y)))
    x2 = max(0.0, min(float(img_w), float(x + w)))
    y2 = max(0.0, min(float(img_h), float(y + h)))
    if x2 <= x1 or y2 <= y1 or float(score) < min_score:
      continue
    boxes.append(
        {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "score": float(score),
            "class_id": int(cls_id),
        }
    )
  return boxes


def _bbox_payload_extent(payload: bytes) -> tuple[int, int]:
  if len(payload) < 4:
    return (0, 0)
  count = min(struct.unpack_from("<I", payload, 0)[0], (len(payload) - 4) // 24)
  max_x2 = 0
  max_y2 = 0
  offset = 4
  for _ in range(count):
    x, y, w, h, _, _ = struct.unpack_from("<iiiifi", payload, offset)
    offset += 24
    max_x2 = max(max_x2, x + w)
    max_y2 = max(max_y2, y + h)
  return (max_x2, max_y2)


def _threshold_suffix(threshold: float) -> str:
  return f"t{int(round(threshold * 100)):03d}"


def _write_overlay_artifact(
    path: Path, image: np.ndarray, boxes: list[dict], highlight_top_k: int = TOP_SCORE_HIGHLIGHT_COUNT
) -> Path:
  SANDBOX_API_TESTS.mkdir(parents=True, exist_ok=True)

  overlay = image.copy()
  highlighted_ids = {
      id(box)
      for box in sorted(boxes, key=lambda box: box["score"], reverse=True)[: max(0, highlight_top_k)]
  }
  for box in boxes:
    x1 = max(0, min(int(round(box["x1"])), overlay.shape[1] - 1))
    y1 = max(0, min(int(round(box["y1"])), overlay.shape[0] - 1))
    x2 = max(0, min(int(round(box["x2"])), overlay.shape[1]))
    y2 = max(0, min(int(round(box["y2"])), overlay.shape[0]))
    if x2 <= x1 or y2 <= y1:
      continue
    color = [0, 255, 0] if id(box) in highlighted_ids else [255, 0, 0]
    overlay[y1:y2, x1 : min(x1 + 2, x2)] = color
    overlay[y1:y2, max(x2 - 2, x1):x2] = color
    overlay[y1 : min(y1 + 2, y2), x1:x2] = color
    overlay[max(y2 - 2, y1):y2, x1:x2] = color

  overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
  if not cv2.imwrite(str(path), overlay_bgr):
    raise AssertionError(f"failed to write overlay artifact: {path}")
  return path


def _assert_image_dims(path: Path, width: int, height: int) -> None:
  image = cv2.imread(str(path), cv2.IMREAD_COLOR)
  if image is None:
    raise AssertionError(f"failed to read image artifact: {path}")
  assert image.shape[1] == width
  assert image.shape[0] == height


def test_resnet_real_fixture_run_preserves_stable_classification_contract():
  model = _image_input_model(_env_path("SIMA_RESNET50_TAR"))
  image = _decode_rgb_image(_fixture_image_path(Path("test.jpg")))

  input_tensor = _rgb_tensor(image)
  runner = model.build(input_tensor)
  summaries = []
  try:
    for _ in range(3):
      output = runner.run(_rgb_tensor(image), timeout_ms=20000)
      assert output.kind == pyneat.SampleKind.Tensor
      assert output.tensor is not None
      summaries.append(
          {
              "shape": list(output.tensor.shape),
              "dtype": output.tensor.dtype,
              "payload_tag": output.payload_tag,
              "format": output.format,
              "media_type": output.media_type,
              "argmax": int(output.tensor.to_numpy(copy=True).reshape(-1).argmax()),
          }
      )
      del output
  finally:
    runner.close()

  assert all(summary["shape"] == [1, 1, 1, 1000] for summary in summaries)
  assert all(summary["dtype"] == pyneat.TensorDType.Float32 for summary in summaries)
  assert all(summary["payload_tag"] == "DETESSDEQUANT" for summary in summaries)
  assert all(summary["format"] == "DETESSDEQUANT" for summary in summaries)
  assert all(summary["media_type"] == "application/vnd.simaai.tensor" for summary in summaries)
  assert [summary["argmax"] for summary in summaries] == [839, 839, 839]


def test_real_fixture_paths_resolve():
  assert _env_path("SIMA_RESNET50_TAR").name.endswith(".tar.gz")
  assert _env_path("SIMA_YOLO_TAR").name.endswith(".tar.gz")
  assert _fixture_image_path(Path("test.jpg")).name == "test.jpg"
  assert _fixture_image_path(Path("tests/images/people.jpg")).name == "people.jpg"
  assert SANDBOX_API_TESTS.parent == ROOT / "sandbox"


def test_model_backed_option_structs_preserve_real_fixture_semantics():
  resnet_preproc_opt = pyneat.ModelOptions()
  resnet_preproc_opt.preprocess.normalize.enable = pyneat.AutoFlag.On
  resnet_preproc_model = pyneat.Model(str(_env_path("SIMA_RESNET50_TAR")), resnet_preproc_opt)
  quant_model = _tensor_input_model(_env_path("SIMA_YOLO_TAR"))
  yolo_model = _image_input_model(_env_path("SIMA_YOLO_TAR"))

  pre = pyneat.PreprocOptions(resnet_preproc_model)
  quant = pyneat.QuantTessOptions(quant_model)
  detess = pyneat.DetessDequantOptions(yolo_model)

  assert pre.node_name == "preproc"
  assert pre.has_input_shape()
  assert pre.has_output_shape()
  assert pre.num_buffers == pre.num_buffers_model == 4
  assert pre.num_buffers_locked is True

  assert quant.config_json is None
  assert quant.num_buffers == quant.num_buffers_model == 4
  assert quant.num_buffers_locked is True
  assert pyneat.nodes.quant_tess(quant) is not None

  assert detess.upstream_name
  assert detess.element_name
  assert detess.num_buffers == detess.num_buffers_model == 4
  assert detess.num_buffers_locked is True


def test_yolo_mla_group_matches_explicit_preprocess_plus_inference_structure():
  image = _decode_rgb_image(_fixture_image_path(Path("tests/images/people.jpg")))
  model = _image_input_model(_env_path("SIMA_YOLO_TAR"))

  mla_output = _run_model_on_image(
      model,
      image,
      pyneat.groups.mla(model),
  )
  explicit_output = _run_model_on_image(
      model,
      image,
      pyneat.nodes.preproc(pyneat.PreprocOptions(model)),
      pyneat.groups.mla(model),
  )

  assert mla_output.kind == pyneat.SampleKind.Tensor
  assert explicit_output.kind == pyneat.SampleKind.Tensor
  assert mla_output.payload_tag == explicit_output.payload_tag == "MLA"
  assert mla_output.tensor is not None
  assert explicit_output.tensor is not None
  assert list(mla_output.tensor.shape) == [80, 80, 64]
  assert list(explicit_output.tensor.shape) == [80, 80, 64]
  assert mla_output.tensor.dtype == pyneat.TensorDType.Int8
  assert explicit_output.tensor.dtype == pyneat.TensorDType.Int8


def test_yolo_detess_and_boxdecode_real_fixture_paths_are_runtime_usable(tmp_path):
  image = _decode_rgb_image(_fixture_image_path(Path("tests/images/people.jpg")))
  detess_model = _image_input_model(_env_path("SIMA_YOLO_TAR"))
  boxdecode_model = _image_input_model(_env_path("SIMA_YOLO_TAR"), pyneat.BoxDecodeType.YoloV8)

  detess_output = _run_model_on_image(
      detess_model,
      image,
      pyneat.groups.mla(detess_model),
      pyneat.nodes.detess_dequant(pyneat.DetessDequantOptions(detess_model)),
  )
  session = pyneat.Session()
  session.add(pyneat.nodes.input(boxdecode_model.input_appsrc_options(False)))
  session.add(_custom_preproc_node(boxdecode_model, image))
  session.add(pyneat.groups.mla(boxdecode_model))
  session.add(pyneat.nodes.sima_box_decode(boxdecode_model, decode_type=pyneat.BoxDecodeType.YoloV8))
  session.add(pyneat.nodes.output())

  backend = session.describe_backend()
  assert "detection-threshold=" not in backend.lower()

  input_tensor = _rgb_tensor(image)
  runner = session.build(input_tensor)
  try:
    box_output = runner.run(input_tensor, timeout_ms=30000)
  finally:
    runner.close()

  assert detess_output.kind == pyneat.SampleKind.Bundle
  assert detess_output.payload_tag == "DETESSDEQUANT"
  assert len(detess_output.fields) == 6
  assert [list(field.tensor.shape) for field in detess_output.fields] == [
      [1, 80, 80, 64],
      [1, 40, 40, 64],
      [1, 20, 20, 64],
      [1, 80, 80, 80],
      [1, 40, 40, 80],
      [1, 20, 20, 80],
  ]
  assert all(field.tensor.dtype == pyneat.TensorDType.Float32 for field in detess_output.fields)

  assert box_output.kind == pyneat.SampleKind.Tensor
  assert box_output.payload_tag == "BBOX"
  assert box_output.tensor is not None
  assert box_output.tensor.dtype == pyneat.TensorDType.UInt8
  assert list(box_output.tensor.shape) == [20160]


def test_tensor_input_model_uses_quanttess_frontend_contract():
  model_a = _tensor_input_model(_env_path("SIMA_YOLO_TAR"))
  model_b = _tensor_input_model(_env_path("SIMA_YOLO_TAR"))

  for model in (model_a, model_b):
    input_spec = model.input_spec()
    appsrc = model.input_appsrc_options(True)
    backend = model.backend_fragment(pyneat.ModelStage.Preprocess).lower()

    assert input_spec.dtypes == [pyneat.TensorDType.Float32]
    assert list(input_spec.shape) == [640, 640, 3]
    assert appsrc.media_type == "application/vnd.simaai.tensor"
    assert appsrc.format == "FP32"
    assert "quanttess" in backend
    assert "preproc" not in backend


def test_cpu_quanttess_input_matches_letterboxed_fp32_tensor_contract():
  image = _decode_rgb_image(_fixture_image_path(Path("tests/images/people.jpg")))
  preproc_model = _image_input_model(_env_path("SIMA_YOLO_TAR"))

  quant_input = _cpu_quanttess_input(preproc_model, image)

  assert list(quant_input.shape) == [640, 640, 3]
  assert quant_input.dtype == np.float32
  assert float(quant_input.min()) == 0.0
  assert float(quant_input.max()) <= 1.0
  assert np.allclose(quant_input[0], 0.0)
  assert np.allclose(quant_input[-1], 0.0)
  assert np.any(quant_input[106] > 0.0)
  assert np.any(quant_input[532] > 0.0)


def _run_quanttess_boxdecode_on_real_input(
    model: pyneat.Model,
    image: np.ndarray,
    quant_input: np.ndarray,
    preprocess_part,
    detection_threshold: float,
) -> tuple[str, pyneat.Sample]:
  session = pyneat.Session()
  session.add(pyneat.nodes.input(model.input_appsrc_options(True)))
  session.add(preprocess_part)
  session.add(pyneat.groups.mla(model))
  session.add(
      pyneat.nodes.sima_box_decode(
          model,
          decode_type=pyneat.BoxDecodeType.YoloV8,
          original_width=image.shape[1],
          original_height=image.shape[0],
          model_width=quant_input.shape[1],
          model_height=quant_input.shape[0],
          detection_threshold=detection_threshold,
          nms_iou_threshold=0.55,
          top_k=120,
      )
  )
  session.add(pyneat.nodes.output())

  backend = session.describe_backend().lower()
  runner = session.build(quant_input)
  try:
    box_output = runner.run(quant_input, timeout_ms=30000)
  finally:
    runner.close()
  return backend, box_output


def test_tensor_model_quanttess_mla_boxdecode_writes_model_overlay(tmp_path):
  image = _decode_rgb_image(_fixture_image_path(Path("tests/images/people.jpg")))
  model = _tensor_input_model(_env_path("SIMA_YOLO_TAR"), pyneat.BoxDecodeType.YoloV8)
  preproc_model = _image_input_model(_env_path("SIMA_YOLO_TAR"), pyneat.BoxDecodeType.YoloV8)
  quant_input = _cpu_quanttess_input(preproc_model, image)

  counts = []
  for threshold in BOXDECODE_THRESHOLDS:
    backend, box_output = _run_quanttess_boxdecode_on_real_input(
        model,
        image,
        quant_input,
        pyneat.nodes.quant_tess(pyneat.QuantTessOptions(model)),
        threshold,
    )

    assert "quanttess" in backend
    assert "preproc" not in backend
    assert "neatprocessmla" in backend
    assert "neatboxdecode" in backend

    payload = _extract_bbox_payload(box_output)
    max_x2, max_y2 = _bbox_payload_extent(payload)
    boxes = _parse_bbox_payload(payload, image.shape[1], image.shape[0], min_score=threshold)
    counts.append(len(boxes))
    out_path = SANDBOX_API_TESTS / f"people_model_overlay_{_threshold_suffix(threshold)}.png"

    assert list(quant_input.shape) == [640, 640, 3]
    assert box_output.kind == pyneat.SampleKind.Tensor
    assert box_output.payload_tag == "BBOX"
    assert boxes
    assert _write_overlay_artifact(out_path, image, boxes).is_file()
    assert max_x2 <= int(image.shape[1] * 1.1)
    assert max_y2 <= int(image.shape[0] * 1.1)
    _assert_image_dims(out_path, image.shape[1], image.shape[0])

  assert counts == sorted(counts, reverse=True)


def test_custom_session_quanttess_mla_boxdecode_writes_explicit_overlay(tmp_path):
  image = _decode_rgb_image(_fixture_image_path(Path("tests/images/people.jpg")))
  model = _tensor_input_model(_env_path("SIMA_YOLO_TAR"), pyneat.BoxDecodeType.YoloV8)
  preproc_model = _image_input_model(_env_path("SIMA_YOLO_TAR"), pyneat.BoxDecodeType.YoloV8)
  quant_input = _cpu_quanttess_input(preproc_model, image)

  counts = []
  for threshold in BOXDECODE_THRESHOLDS:
    backend, box_output = _run_quanttess_boxdecode_on_real_input(
        model,
        image,
        quant_input,
        pyneat.nodes.quant_tess(pyneat.QuantTessOptions(model)),
        threshold,
    )

    assert "quanttess" in backend
    assert "preproc" not in backend
    assert "neatprocessmla" in backend
    assert "neatboxdecode" in backend

    payload = _extract_bbox_payload(box_output)
    max_x2, max_y2 = _bbox_payload_extent(payload)
    boxes = _parse_bbox_payload(payload, image.shape[1], image.shape[0], min_score=threshold)
    counts.append(len(boxes))
    out_path = SANDBOX_API_TESTS / (
        f"people_quanttess_mla_boxdecode_overlay_{_threshold_suffix(threshold)}.png"
    )

    assert list(quant_input.shape) == [640, 640, 3]
    assert box_output.kind == pyneat.SampleKind.Tensor
    assert box_output.payload_tag == "BBOX"
    assert boxes
    assert _write_overlay_artifact(out_path, image, boxes).is_file()
    assert max_x2 <= int(image.shape[1] * 1.1)
    assert max_y2 <= int(image.shape[0] * 1.1)
    _assert_image_dims(out_path, image.shape[1], image.shape[0])

  assert counts == sorted(counts, reverse=True)


def test_boxdecode_runtime_output_produces_overlay_artifact(tmp_path):
  image = _decode_rgb_image(_fixture_image_path(Path("tests/images/people.jpg")))
  model = _image_input_model(_env_path("SIMA_YOLO_TAR"), pyneat.BoxDecodeType.YoloV8)

  counts = []
  for threshold in BOXDECODE_THRESHOLDS:
    box_output = _run_model_on_image(
        model,
        image,
        _custom_preproc_node(model, image),
        pyneat.groups.mla(model),
        pyneat.nodes.sima_box_decode(
            model,
            decode_type=pyneat.BoxDecodeType.YoloV8,
            original_width=image.shape[1],
            original_height=image.shape[0],
            model_width=640,
            model_height=640,
            detection_threshold=threshold,
            nms_iou_threshold=0.55,
            top_k=120,
        ),
    )
    payload = _extract_bbox_payload(box_output)
    max_x2, max_y2 = _bbox_payload_extent(payload)
    boxes = _parse_bbox_payload(payload, image.shape[1], image.shape[0], min_score=threshold)
    counts.append(len(boxes))
    out_path = SANDBOX_API_TESTS / (
        f"people_preproc_mla_boxdecode_overlay_{_threshold_suffix(threshold)}.png"
    )

    assert _write_overlay_artifact(out_path, image, boxes).is_file()
    assert max_x2 <= int(image.shape[1] * 1.1)
    assert max_y2 <= int(image.shape[0] * 1.1)
    _assert_image_dims(out_path, image.shape[1], image.shape[0])
    assert boxes

  assert counts == sorted(counts, reverse=True)
