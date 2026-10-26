import gc
import json
import struct
import sys
import tarfile
import time
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
    installed_candidate = root / "lib" / "sima-neat" / rel_path
    if installed_candidate.is_file():
      return installed_candidate

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


def _sample_from_tensor(tensor: pyneat.Tensor) -> pyneat.Sample:
  sample = pyneat.Sample()
  sample.kind = pyneat.SampleKind.Tensor
  sample.tensor = tensor
  return sample


def _single_sample(samples) -> pyneat.Sample:
  if isinstance(samples, pyneat.Sample):
    if samples.kind == pyneat.SampleKind.Bundle:
      assert len(samples.fields) == 1
      return samples.fields[0]
    return samples
  assert isinstance(samples, list)
  assert len(samples) == 1
  return samples[0]


def _sample_tensor(sample: pyneat.Sample) -> pyneat.Tensor:
  if sample.kind == pyneat.SampleKind.Tensor:
    assert sample.tensor is not None
    return sample.tensor
  if sample.kind == pyneat.SampleKind.TensorSet:
    assert len(sample.tensors) == 1
    return sample.tensors[0]
  raise AssertionError(f"expected tensor sample, got {sample.kind}")


def _sample_tensors(sample: pyneat.Sample) -> list[pyneat.Tensor]:
  if sample.kind == pyneat.SampleKind.Tensor:
    assert sample.tensor is not None
    return [sample.tensor]
  if sample.kind == pyneat.SampleKind.TensorSet:
    return list(sample.tensors)
  raise AssertionError(f"expected tensor sample, got {sample.kind}")


def _release_accelerator_outputs() -> None:
  """Drop Python-side refs before starting the next accelerator-backed graph.

  The DevKit has a small number of RPMsg/EV channels. Keeping output tensor
  objects from one graph alive while building/running the next graph can keep
  device-side resources pinned long enough for the next graph to report
  transient "channel busy" or mailbox-fd errors. Convert outputs to plain Python
  metadata/payload bytes first, then call this helper before the next run.
  """
  gc.collect()
  time.sleep(0.2)


def _run_one_sample(runner, tensor: pyneat.Tensor, timeout_ms: int) -> pyneat.Sample:
  return _single_sample(runner.run([_sample_from_tensor(tensor)], timeout_ms=timeout_ms))


def _run_model_on_image(model: pyneat.Model, image: np.ndarray, *parts) -> pyneat.Sample:
  graph = pyneat.Graph()
  graph.add(pyneat.nodes.input(model.input_appsrc_options(False)))
  for part in parts:
    graph.add(part)
  graph.add(pyneat.nodes.output())
  input_tensor = _rgb_tensor(image)
  runner = graph.build([input_tensor])
  try:
    return _run_one_sample(runner, input_tensor, timeout_ms=30000)
  finally:
    runner.close()


def _custom_preproc_node(model: pyneat.Model, image: np.ndarray):
  pre = pyneat.PreprocOptions(model)
  channels = image.shape[2] if image.ndim >= 3 else 1
  pre.input_shape = [image.shape[0], image.shape[1], channels]
  return pyneat.nodes.preproc(pre)


def _mla_input_byte_stream_contract(model_tar: Path) -> tuple[str, int]:
  with tarfile.open(model_tar, "r:gz") as tar:
    member = next(m for m in tar.getmembers() if Path(m.name).name.endswith("_mpk.json"))
    extracted = tar.extractfile(member)
    assert extracted is not None
    data = json.load(extracted)

  mla = next(plugin for plugin in data["plugins"] if plugin.get("processor") == "MLA")
  node = mla["input_nodes"][0]
  return str(node["name"]), int(node["size"])


def _mla_byte_stream_input_options(model_tar: Path) -> pyneat.InputOptions:
  name, num_bytes = _mla_input_byte_stream_contract(model_tar)
  opt = pyneat.InputOptions()
  opt.payload_type = pyneat.PayloadType.Tensor
  opt.format = pyneat.Format.ByteStream
  opt.width = num_bytes
  opt.height = 1
  opt.depth = 1
  opt.max_width = num_bytes
  opt.max_height = 1
  opt.max_depth = 1
  opt.buffer_name = name
  return opt


def _run_mla_byte_stream_pipeline(
    model: pyneat.Model, model_tar: Path, *post_mla_parts
) -> list[pyneat.Tensor]:
  _name, num_bytes = _mla_input_byte_stream_contract(model_tar)
  payload = np.zeros((num_bytes,), dtype=np.int8)
  tensor = pyneat.Tensor.from_numpy(payload, copy=True, byte_format=pyneat.ByteFormat.Raw)

  graph = pyneat.Graph()
  graph.add(pyneat.nodes.input(_mla_byte_stream_input_options(model_tar)))
  graph.add(pyneat.groups.mla(model))
  for part in post_mla_parts:
    graph.add(part)
  graph.add(pyneat.nodes.output())

  runner = graph.build([tensor])
  try:
    outputs = runner.run([tensor], timeout_ms=30000)
  finally:
    runner.close()

  assert isinstance(outputs, list)
  assert all(isinstance(tensor, pyneat.Tensor) for tensor in outputs)
  return outputs


def _run_mla_on_byte_stream(model: pyneat.Model, model_tar: Path) -> list[pyneat.Tensor]:
  return _run_mla_byte_stream_pipeline(model, model_tar)


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
    fmt = (current.payload_tag or current.format or "").upper()
    if fmt and fmt != "BBOX":
      continue
    if current.kind == pyneat.SampleKind.Tensor and current.tensor is not None:
      payload = current.tensor.copy_payload_bytes()
      if payload:
        return payload
    if current.kind == pyneat.SampleKind.TensorSet:
      for tensor in current.tensors:
        payload = tensor.copy_payload_bytes()
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


def _validate_bbox_payload(
    payload: bytes,
    img_w: int,
    img_h: int,
    *,
    top_k: int,
    detection_threshold: float,
    num_classes: int = 80,
) -> list[dict]:
  """Validate the raw plain-BBOX ABI before any display clipping/filtering."""
  context = f"BBOX payload_bytes={len(payload)} capacity={top_k} geometry={img_w}x{img_h}"
  assert len(payload) == 4 + top_k * 24, f"fixed-capacity extent changed: {context}"
  count = struct.unpack_from("<I", payload)[0]
  assert count <= top_k, f"count={count} exceeds capacity: {context}"
  assert not any(payload[4 + count * 24 :]), f"unused BBOX tail is not zero: {context}"
  cutoff = float(np.float32(detection_threshold))
  records = []
  previous_score = float("inf")
  for index in range(count):
    x, y, w, h, score, cls_id = struct.unpack_from("<iiiifi", payload, 4 + index * 24)
    record_context = f"record={index} xywh={(x, y, w, h)} score={score} cutoff={cutoff}: {context}"
    assert np.isfinite(score) and 0.0 <= score <= 1.0, f"invalid score: {record_context}"
    assert score <= previous_score, f"scores are not confidence-descending: {record_context}"
    assert score >= cutoff, f"emitted score is below threshold: {record_context}"
    assert 0 <= cls_id < num_classes, f"class={cls_id} outside model contract: {record_context}"
    assert w >= 0 and h >= 0, f"negative extent: {record_context}"
    assert 0 <= x <= img_w and 0 <= y <= img_h, f"origin outside image: {record_context}"
    # Independent rounding of x and w can put their sum one pixel over the edge.
    assert x + w <= img_w + 1 and y + h <= img_h + 1, f"end outside image: {record_context}"
    records.append(
        {"x1": x, "y1": y, "x2": x + w, "y2": y + h, "score": score, "class_id": cls_id}
    )
    previous_score = score
  return records


def _validated_bbox_output(
    sample: pyneat.Sample, img_w: int, img_h: int, threshold: float, top_k: int = 120
) -> tuple[bytes, list[dict]]:
  tensor = _sample_tensor(sample)
  context = f"BBOX dtype={tensor.dtype} shape={list(tensor.shape)} geometry={img_w}x{img_h}"
  assert tensor.dtype == pyneat.TensorDType.UInt8, context
  assert list(tensor.shape) == [4 + top_k * 24], context
  # A plain-BBOX route may return an untagged SingleTensor. Any advertised
  # Sample or Tensor format must still agree; precedence must not hide conflict.
  for owner, advertised in (
      ("Sample.payload_tag", sample.payload_tag),
      ("Sample.format", sample.format),
      ("Tensor.detection", pyneat.detections.read_detection_format(tensor)),
  ):
    assert not advertised or advertised.upper() == "BBOX", (
        f"conflicting {owner}={advertised!r}: {context}"
    )
  payload = _extract_bbox_payload(sample)
  records = _validate_bbox_payload(
      payload, img_w, img_h, top_k=top_k, detection_threshold=threshold
  )
  # Passing zero image dimensions disables the public parser's display clipping.
  decoded = pyneat.detections.decode_bbox_tensor(
      tensor, img_w=0, img_h=0, expected_topk=top_k, strict=True
  )
  assert decoded.raw == payload
  assert len(decoded.boxes) == len(records)
  for box, record in zip(decoded.boxes, records):
    assert [box.x1, box.y1, box.x2, box.y2, box.score, box.class_id] == [
        record[key] for key in ("x1", "y1", "x2", "y2", "score", "class_id")
    ]
  return payload, records


@pytest.mark.parametrize(
    "payload_tag,sample_format,tensor_format,valid",
    (
        ("", "", "", True),
        ("BBOX", "", "", True),
        ("", "BBOX", "", True),
        ("", "", "BBOX", True),
        ("BBOX_POSE", "", "", False),
        ("", "BBOX_POSE", "", False),
        ("", "", "BBOX_POSE", False),
        ("BBOX", "BBOX_POSE", "", False),
        ("BBOX", "", "BBOX_POSE", False),
    ),
)
def test_raw_bbox_output_accepts_untagged_tensor_and_rejects_conflicting_formats(
    payload_tag, sample_format, tensor_format, valid
):
  tensor = pyneat.Tensor.from_numpy(np.zeros(2884, dtype=np.uint8), copy=True)
  if tensor_format:
    detection = pyneat.DetectionSpec()
    detection.format = tensor_format
    semantic = tensor.semantic
    semantic.detection = detection
    tensor.semantic = semantic
  sample = _sample_from_tensor(tensor)
  sample.payload_tag = payload_tag
  sample.format = sample_format
  if valid:
    assert _validated_bbox_output(sample, 640, 360, 0.5) == (bytes(2884), [])
  else:
    with pytest.raises(AssertionError, match="conflicting"):
      _validated_bbox_output(sample, 640, 360, 0.5)


@pytest.mark.parametrize(
    "mutation",
    (
        "short_header", "short_record", "extra_byte", "count_overflow", "nan_score",
        "score_over_one", "below_threshold", "descending", "invalid_class", "negative_extent",
        "negative_origin", "end_outside", "tail_nonzero", "nextafter_below_threshold",
    ),
)
def test_raw_bbox_validator_rejects_malformed_payload(mutation):
  records = [(10, 20, 30, 40, 0.75, 0), (15, 25, 30, 40, 0.60, 1)]
  payload = struct.pack("<I", 2) + b"".join(struct.pack("<iiiifi", *row) for row in records)
  if mutation == "short_header":
    payload = payload[:3]
  elif mutation == "short_record":
    payload = payload[:-1]
  elif mutation == "extra_byte":
    payload += b"\x00"
  elif mutation == "count_overflow":
    payload = struct.pack("<I", 3) + payload[4:]
  elif mutation == "tail_nonzero":
    payload = struct.pack("<I", 1) + payload[4:]
  else:
    first = list(records[0])
    replacements = {
        "nan_score": (4, float("nan")), "score_over_one": (4, 1.1),
        "below_threshold": (4, 0.4), "descending": (4, 0.55),
        "invalid_class": (5, 80), "negative_extent": (2, -1),
        "negative_origin": (0, -1), "end_outside": (2, 1000),
        "nextafter_below_threshold": (4, float(np.nextafter(np.float32(0.5), np.float32(0.0)))),
    }
    field, value = replacements[mutation]
    first[field] = value
    payload = struct.pack("<I", 2) + struct.pack("<iiiifi", *first) + payload[28:]
  with pytest.raises(AssertionError):
    _validate_bbox_payload(payload, 640, 360, top_k=2, detection_threshold=0.5)


def test_raw_bbox_validator_preserves_count_and_unclipped_geometry():
  payload = struct.pack("<Iiiiifi", 1, 10, 20, 30, 40, 0.75, 2) + bytes(24)
  records = _validate_bbox_payload(payload, 640, 360, top_k=2, detection_threshold=0.5)
  assert records == [{"x1": 10, "y1": 20, "x2": 40, "y2": 60, "score": 0.75, "class_id": 2}]
  assert _validate_bbox_payload(bytes(52), 640, 360, top_k=2, detection_threshold=0.5) == []


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
  path.parent.mkdir(parents=True, exist_ok=True)

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


def _assert_overlay_changed(path: Path, image: np.ndarray) -> None:
  overlay = cv2.imread(str(path), cv2.IMREAD_COLOR)
  assert overlay is not None
  assert np.any(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB) != image), "overlay drew no boxes"


def test_resnet_real_fixture_run_preserves_stable_classification_contract():
  model = _image_input_model(_env_path("SIMA_RESNET50_TAR"))
  image = _decode_rgb_image(_fixture_image_path(Path("test.jpg")))

  input_tensor = _rgb_tensor(image)
  runner = model.build([input_tensor])
  summaries = []
  try:
    for _ in range(3):
      output = _run_one_sample(runner, _rgb_tensor(image), timeout_ms=20000)
      tensor = _sample_tensor(output)
      summaries.append(
          {
              "shape": list(tensor.shape),
              "dtype": tensor.dtype,
              "layout": tensor.layout,
              "payload_tag": output.payload_tag,
              "format": output.format,
              "media_type": output.media_type,
              "argmax": int(tensor.to_numpy(copy=True).reshape(-1).argmax()),
          }
      )
      del output
  finally:
    runner.close()

  # The fixture explicitly flattens the Detess frame before terminal dequantization.
  assert all(summary["shape"] == [1, 1000] for summary in summaries)
  assert all(summary["dtype"] == pyneat.TensorDType.Float32 for summary in summaries)
  assert all(summary["layout"] == pyneat.TensorLayout.Unknown for summary in summaries)
  assert all(summary["payload_tag"] == "EVXX_FLOAT32" for summary in summaries)
  assert all(summary["format"] == "EVXX_FLOAT32" for summary in summaries)
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
  model_tar = _env_path("SIMA_YOLO_TAR")
  model = _image_input_model(model_tar)

  mla_tensors = _run_mla_on_byte_stream(model, model_tar)
  mla_shapes = [list(tensor.shape) for tensor in mla_tensors]
  mla_dtypes = [tensor.dtype for tensor in mla_tensors]
  del mla_tensors
  _release_accelerator_outputs()

  explicit_output = _run_model_on_image(
      model,
      image,
      pyneat.nodes.preproc(pyneat.PreprocOptions(model)),
      pyneat.groups.mla(model),
  )
  explicit_tensors = _sample_tensors(explicit_output)
  explicit_shapes = [list(tensor.shape) for tensor in explicit_tensors]
  explicit_dtypes = [tensor.dtype for tensor in explicit_tensors]
  del explicit_output, explicit_tensors
  _release_accelerator_outputs()

  assert len(mla_shapes) == len(explicit_shapes) == 6
  assert mla_shapes == explicit_shapes
  assert all(dtype == pyneat.TensorDType.Int8 for dtype in mla_dtypes)
  assert all(dtype == pyneat.TensorDType.Int8 for dtype in explicit_dtypes)


def test_yolo_mla_group_rejects_image_input_with_byte_stream_contract_message():
  image = _decode_rgb_image(_fixture_image_path(Path("tests/images/people.jpg")))
  model_tar = _env_path("SIMA_YOLO_TAR")
  model = _image_input_model(model_tar)
  _name, expected_bytes = _mla_input_byte_stream_contract(model_tar)
  got_bytes = int(image.nbytes)

  graph = pyneat.Graph()
  graph.add(pyneat.nodes.input(model.input_appsrc_options(False)))
  graph.add(pyneat.groups.mla(model))
  graph.add(pyneat.nodes.output())

  with pytest.raises(
      pyneat.NeatError,
      match=(
          "inference-only expects application/vnd.simaai.tensor / "
          "ByteFormat.Raw byte-stream input"
      ),
  ) as exc:
    graph.build([_rgb_tensor(image)])

  assert f"expected {expected_bytes} bytes, got {got_bytes} bytes" in str(exc.value)


def test_yolo_detess_and_boxdecode_real_fixture_paths_are_runtime_usable(tmp_path):
  image = _decode_rgb_image(_fixture_image_path(Path("tests/images/people.jpg")))
  model_tar = _env_path("SIMA_YOLO_TAR")
  detess_model = _image_input_model(model_tar)
  boxdecode_model = _image_input_model(model_tar, pyneat.BoxDecodeType.YoloV8)
  boxdecode_top_k = 120

  detess_tensors = _run_mla_byte_stream_pipeline(
      detess_model,
      model_tar,
      pyneat.nodes.detess_dequant(pyneat.DetessDequantOptions(detess_model)),
  )
  detess_shapes = [list(tensor.shape) for tensor in detess_tensors]
  detess_dtypes = [tensor.dtype for tensor in detess_tensors]
  del detess_tensors
  _release_accelerator_outputs()

  graph = pyneat.Graph()
  graph.add(pyneat.nodes.input(boxdecode_model.input_appsrc_options(False)))
  graph.add(_custom_preproc_node(boxdecode_model, image))
  graph.add(pyneat.groups.mla(boxdecode_model))
  graph.add(
      pyneat.nodes.sima_box_decode(
          boxdecode_model,
          decode_type=pyneat.BoxDecodeType.YoloV8,
          top_k=boxdecode_top_k,
      )
  )
  graph.add(pyneat.nodes.output())

  backend = graph.describe_backend()
  assert "detection-threshold=" not in backend.lower()

  input_tensor = _rgb_tensor(image)
  runner = graph.build([input_tensor])
  try:
    box_output = _run_one_sample(runner, input_tensor, timeout_ms=30000)
  finally:
    runner.close()

  assert len(detess_shapes) == 6
  assert detess_shapes == [
      [1, 80, 80, 64],
      [1, 40, 40, 64],
      [1, 20, 20, 64],
      [1, 80, 80, 80],
      [1, 40, 40, 80],
      [1, 20, 20, 80],
  ]
  assert all(dtype == pyneat.TensorDType.Float32 for dtype in detess_dtypes)

  box_tensor = _sample_tensor(box_output)
  assert box_output.payload_tag in ("", "BBOX")
  assert box_tensor.dtype == pyneat.TensorDType.UInt8
  assert list(box_tensor.shape) == [4 + boxdecode_top_k * 24]


def test_tensor_input_model_uses_quanttess_frontend_contract():
  model_a = _tensor_input_model(_env_path("SIMA_YOLO_TAR"))
  model_b = _tensor_input_model(_env_path("SIMA_YOLO_TAR"))

  for model in (model_a, model_b):
    input_spec = model.input_specs()[0]
    appsrc = model.input_appsrc_options(True)
    info = model.info()

    assert input_spec.dtypes == [pyneat.TensorDType.Float32]
    assert list(input_spec.shape) == [640, 640, 3]
    assert appsrc.payload_type == pyneat.PayloadType.Tensor
    assert appsrc.format == pyneat.Format.FP32
    # Image preprocessing is disabled; the authored tensor frontend remains active.
    assert list(info.pre_kernels) == ["quantize", "tessellate"]
    assert info.needs.pre_quantization and info.needs.pre_tessellation
    assert info.capabilities.has_pre_quantization and info.capabilities.has_pre_tessellation
    assert "neatkernelrunner" in model.backend_fragment(pyneat.ModelStage.Preprocess).lower()


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
    original_size: tuple[int, int] | None = None,
) -> tuple[str, pyneat.Sample]:
  original_width, original_height = original_size or (image.shape[1], image.shape[0])
  graph = pyneat.Graph()
  graph.add(pyneat.nodes.input(model.input_appsrc_options(True)))
  graph.add(preprocess_part)
  graph.add(pyneat.groups.mla(model))
  graph.add(
      pyneat.nodes.sima_box_decode(
          model,
          decode_type=pyneat.BoxDecodeType.YoloV8,
          original_width=original_width,
          original_height=original_height,
          model_width=quant_input.shape[1],
          model_height=quant_input.shape[0],
          detection_threshold=detection_threshold,
          nms_iou_threshold=0.55,
          top_k=120,
          # _cpu_quanttess_input letterboxes when the model expects aspect-ratio
          # preservation (the default for YOLOv8). The pipeline below skips the
          # model's Preproc stage, so per-buffer preproc_resize_mode meta is
          # never written by upstream — pass the override explicitly so the
          # boxdecode contract drops the field from required-meta. Without it,
          # the plugin chain rejects buffers with
          # "missing required preprocess meta field 'preproc_resize_mode'".
          resize_mode=pyneat.ResizeMode.Letterbox,
      )
  )
  graph.add(pyneat.nodes.output())

  backend = graph.describe_backend().lower()
  assert "simaboxdecode" in graph.describe().lower()
  input_tensor = pyneat.Tensor.from_numpy(quant_input, copy=True)
  runner = graph.build([input_tensor])
  try:
    box_output = _run_one_sample(runner, input_tensor, timeout_ms=30000)
    retained_payload = _extract_bbox_payload(box_output)
  finally:
    runner.close()
  assert _extract_bbox_payload(box_output) == retained_payload, "runner close invalidated BBOX output"
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
    payload, raw_records = _validated_bbox_output(box_output, image.shape[1], image.shape[0], threshold)
    max_x2, max_y2 = _bbox_payload_extent(payload)
    boxes = _parse_bbox_payload(payload, image.shape[1], image.shape[0], min_score=threshold)
    counts.append(len(raw_records))
    out_path = tmp_path / f"people_model_overlay_{_threshold_suffix(threshold)}.png"

    assert list(quant_input.shape) == [640, 640, 3]
    _sample_tensor(box_output)
    assert box_output.payload_tag in ("", "BBOX")
    assert boxes
    assert _write_overlay_artifact(out_path, image, boxes).is_file()
    _assert_overlay_changed(out_path, image)
    assert max_x2 <= int(image.shape[1] * 1.1)
    assert max_y2 <= int(image.shape[0] * 1.1)
    _assert_image_dims(out_path, image.shape[1], image.shape[0])

  assert counts == sorted(counts, reverse=True)


def test_custom_graph_quanttess_mla_boxdecode_writes_explicit_overlay(tmp_path):
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
    payload, raw_records = _validated_bbox_output(box_output, image.shape[1], image.shape[0], threshold)
    max_x2, max_y2 = _bbox_payload_extent(payload)
    boxes = _parse_bbox_payload(payload, image.shape[1], image.shape[0], min_score=threshold)
    counts.append(len(raw_records))
    out_path = tmp_path / (
        f"people_quanttess_mla_boxdecode_overlay_{_threshold_suffix(threshold)}.png"
    )

    assert list(quant_input.shape) == [640, 640, 3]
    _sample_tensor(box_output)
    assert box_output.payload_tag in ("", "BBOX")
    assert boxes
    assert _write_overlay_artifact(out_path, image, boxes).is_file()
    _assert_overlay_changed(out_path, image)
    assert max_x2 <= int(image.shape[1] * 1.1)
    assert max_y2 <= int(image.shape[0] * 1.1)
    _assert_image_dims(out_path, image.shape[1], image.shape[0])

  assert counts == sorted(counts, reverse=True)


def _assert_letterbox_geometry(
    baseline: list[dict], actual: list[dict], model_size: tuple[int, int], image_size: tuple[int, int]
) -> None:
  """Independent image-coordinate oracle, starting from model-space rectangles."""
  model_w, model_h = model_size
  image_w, image_h = image_size
  scale = max(image_w / model_w, image_h / model_h)
  margin_x = (model_w * scale - image_w) / 2
  margin_y = (model_h * scale - image_h) / 2
  assert len(actual) == len(baseline), "image geometry changed candidate count"
  for source, result in zip(baseline, actual):
    assert result["class_id"] == source["class_id"]
    assert result["score"] == source["score"]
    expected = [
        min(limit, max(0, source[key] * scale - margin))
        for key, margin, limit in (
            ("x1", margin_x, image_w), ("y1", margin_y, image_h),
            ("x2", margin_x, image_w), ("y2", margin_y, image_h),
        )
    ]
    # Baseline origins/extents were rounded separately; allow only their two
    # pixel rounding error, not an image-size percentage or display clipping.
    assert all(
        abs(result[key] - coordinate) <= 2
        for key, coordinate in zip(("x1", "y1", "x2", "y2"), expected)
    ), "explicit image dimensions did not reach numerical BoxDecode"


def test_letterbox_geometry_oracle_rejects_ignored_image_override():
  source = [{"x1": 100, "y1": 200, "x2": 150, "y2": 280, "score": 0.75, "class_id": 0}]
  expected = [{**source[0], "y1": 60, "y2": 140}]
  _assert_letterbox_geometry(source, expected, (640, 640), (640, 360))
  with pytest.raises(AssertionError, match="dimensions did not reach"):
    _assert_letterbox_geometry(source, source, (640, 640), (640, 360))


def test_boxdecode_explicit_original_geometry_reaches_numerical_output():
  image = _decode_rgb_image(_fixture_image_path(Path("tests/images/people.jpg")))
  model = _tensor_input_model(_env_path("SIMA_YOLO_TAR"), pyneat.BoxDecodeType.YoloV8)
  preproc_model = _image_input_model(_env_path("SIMA_YOLO_TAR"), pyneat.BoxDecodeType.YoloV8)
  quant_input = _cpu_quanttess_input(preproc_model, image)
  assert list(quant_input.shape) == [640, 640, 3]
  threshold = 0.25

  def decode(size):
    _, output = _run_quanttess_boxdecode_on_real_input(
        model, image, quant_input, pyneat.nodes.quant_tess(pyneat.QuantTessOptions(model)),
        threshold, original_size=size,
    )
    payload, records = _validated_bbox_output(output, *size, threshold)
    return output, payload, records

  baseline, retained_payload, baseline_records = decode((640, 640))
  assert baseline_records
  changed, _, changed_records = decode((640, 360))
  # The exact same tensor bytes and MLA heads are decoded in two image spaces.
  # No margins in the baseline; the override independently subtracts140 in Y.
  _assert_letterbox_geometry(baseline_records, changed_records, (640, 640), (640, 360))
  assert _extract_bbox_payload(baseline) == retained_payload, "new run mutated retained output"
  del changed
  _, restored_payload, restored_records = decode((640, 640))
  assert restored_payload == retained_payload
  assert restored_records == baseline_records
  assert _extract_bbox_payload(baseline) == retained_payload


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


def _benchmark_options(inputs=(), **geometry) -> pyneat.BenchmarkOptions:
  options = pyneat.BenchmarkOptions()
  options.num_samples = 2
  options.min_duration_ms = 0  # These tests check input plumbing, not steady-state timing.
  options.inputs = list(inputs)
  for name, value in geometry.items():
    setattr(options, name, value)
  return options


def _assert_measured(report: pyneat.BenchmarkReport, input_source: str) -> None:
  assert report.input == input_source
  assert report.latency_ms > 0.0
  assert report.fps > 0.0


def _benchmark_input_error(model: pyneat.Model, inputs, **geometry) -> str:
  started = time.monotonic()
  with pytest.raises(pyneat.NeatError) as excinfo:
    model.benchmark(_benchmark_options(inputs, **geometry))
  elapsed = time.monotonic() - started
  assert excinfo.value.error_code == "misconfig.input_shape"
  # The inputs are checked before the runner is built, so no warm-up sample ran.
  assert elapsed < 2.0, f"input rejection took {elapsed:.2f}s"
  return str(excinfo.value)


def _people_bgr() -> np.ndarray:
  image = cv2.imread(str(_fixture_image_path(Path("tests/images/people.jpg"))), cv2.IMREAD_COLOR)
  assert image is not None
  return image


@pytest.mark.parametrize("variant", ["bgr", "rgb", "batched", "small_odd_size"])
def test_benchmark_measures_provided_image_on_image_route(variant):
  model = _image_input_model(_env_path("SIMA_RESNET50_TAR"))
  bgr = cv2.imread(str(_fixture_image_path(Path("test.jpg"))), cv2.IMREAD_COLOR)
  assert bgr is not None
  if variant == "rgb":
    tensor = _rgb_tensor(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
  else:
    if variant == "batched":
      bgr = bgr[np.newaxis]
    elif variant == "small_odd_size":
      bgr = cv2.resize(bgr, (333, 217))
    tensor = pyneat.Tensor.from_numpy(bgr, copy=True, image_format=pyneat.PixelFormat.BGR)

  _assert_measured(model.benchmark(_benchmark_options([tensor])), "provided")


def test_benchmark_without_inputs_stays_synthetic():
  model = _image_input_model(_env_path("SIMA_RESNET50_TAR"))
  _assert_measured(model.benchmark(_benchmark_options()), "synthetic")


def test_benchmark_rejects_image_inputs_that_do_not_fit_the_image_route():
  model = _image_input_model(_env_path("SIMA_RESNET50_TAR"))
  bgr = _people_bgr()
  image = pyneat.Tensor.from_numpy(bgr, copy=True, image_format=pyneat.PixelFormat.BGR)

  untagged = pyneat.Tensor.from_numpy(bgr, copy=True)
  assert "explicit image format" in _benchmark_input_error(model, [untagged])
  assert "exactly one image tensor, got 2" in _benchmark_input_error(model, [image, image])


def test_benchmark_boxdecode_image_route_uses_provided_image_geometry(tmp_path):
  model = _image_input_model(_env_path("SIMA_YOLO_TAR"), pyneat.BoxDecodeType.YoloV8)
  bgr = _people_bgr()
  height, width = bgr.shape[:2]
  image = pyneat.Tensor.from_numpy(bgr, copy=True, image_format=pyneat.PixelFormat.BGR)

  report = model.benchmark(_benchmark_options([image]))
  _assert_measured(report, "provided")
  saved = json.loads(Path(report.save(str(tmp_path / "report.json"))).read_text())
  assert saved["input"] == "provided"

  matching = _benchmark_options([image], original_width=width, original_height=height)
  _assert_measured(model.benchmark(matching), "provided")
  with pytest.raises(RuntimeError, match=rf"must match the provided image \({width}x{height}\)"):
    model.benchmark(_benchmark_options([image], original_width=640, original_height=640))


def test_benchmark_tensor_route_takes_real_tensors_and_rejects_mismatches():
  preproc_model = _image_input_model(_env_path("SIMA_YOLO_TAR"))
  model = _tensor_input_model(_env_path("SIMA_YOLO_TAR"), pyneat.BoxDecodeType.YoloV8)
  letterboxed = _cpu_quanttess_input(preproc_model, cv2.cvtColor(_people_bgr(), cv2.COLOR_BGR2RGB))
  tensor = pyneat.Tensor.from_numpy(letterboxed, copy=True)

  _assert_measured(model.benchmark(_benchmark_options([tensor])), "provided")
  # A tensor route keeps the model shape; the source size only describes BoxDecode geometry.
  geometry = _benchmark_options([tensor], original_width=1280, original_height=720)
  _assert_measured(model.benchmark(geometry), "provided")

  small = pyneat.Tensor.from_numpy(np.zeros((320, 320, 3), dtype=np.float32), copy=True)
  shape_error = _benchmark_input_error(model, [small])
  assert "ingress[0]" in shape_error and "shape=640x640x3" in shape_error
  assert "shape=320x320x3" in shape_error
  as_uint8 = pyneat.Tensor.from_numpy((letterboxed * 255).astype(np.uint8), copy=True)
  assert "format=FP32" in _benchmark_input_error(model, [as_uint8])
  assert "expected exactly 1 ingress tensor inputs" in _benchmark_input_error(model, [tensor, tensor])
