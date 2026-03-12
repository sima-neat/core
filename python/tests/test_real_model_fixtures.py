import os
import subprocess
import struct
from pathlib import Path

import numpy as np

import pyneat


ROOT = Path(__file__).resolve().parents[2]
SANDBOX_API_TESTS = ROOT / "sandbox" / "api-tests"


def _env_path(name: str) -> Path:
  value = os.environ.get(name, "")
  if not value:
    raise AssertionError(f"missing required env var: {name}")
  path = Path(value)
  if not path.is_file():
    raise AssertionError(f"missing required fixture file for {name}: {path}")
  return path


def _decode_rgb_image(path: Path) -> np.ndarray:
  probe = subprocess.run(
      [
          "ffprobe",
          "-v",
          "error",
          "-select_streams",
          "v:0",
          "-show_entries",
          "stream=width,height",
          "-of",
          "csv=p=0:s=x",
          str(path),
      ],
      check=True,
      capture_output=True,
      text=True,
  )
  width, height = (int(part) for part in probe.stdout.strip().split("x"))
  raw = subprocess.run(
      [
          "ffmpeg",
          "-v",
          "error",
          "-i",
          str(path),
          "-f",
          "rawvideo",
          "-pix_fmt",
          "rgb24",
          "-",
      ],
      check=True,
      capture_output=True,
  ).stdout
  return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()


def _tensor_input_model(model_tar: Path) -> pyneat.Model:
  opt = pyneat.ModelOptions()
  opt.media_type = "application/vnd.simaai.tensor"
  opt.format = ""
  return pyneat.Model(str(model_tar), opt)


def _run_model_on_image(model: pyneat.Model, image: np.ndarray, *parts) -> pyneat.Sample:
  session = pyneat.Session()
  session.add(pyneat.nodes.input(model.input_appsrc_options(False)))
  for part in parts:
    session.add(part)
  session.add(pyneat.nodes.output())
  runner = session.build(image)
  try:
    return runner.run(image, timeout_ms=30000)
  finally:
    runner.close()


def _custom_preproc_node(model: pyneat.Model, image: np.ndarray):
  pre = pyneat.PreprocOptions(model)
  cfg = dict(pre.config_json)
  cfg["input_width"] = image.shape[1]
  cfg["input_height"] = image.shape[0]
  pre.config_json = cfg
  return pyneat.nodes.preproc(pre)


def _cpu_quanttess_input(model: pyneat.Model, image: np.ndarray) -> np.ndarray:
  pre = pyneat.PreprocOptions(model)
  cfg = dict(pre.config_json)
  dst_w = int(cfg["output_width"])
  dst_h = int(cfg["output_height"])
  aspect_ratio = bool(cfg.get("aspect_ratio", False))
  padding_type = str(cfg.get("padding_type", "CENTER")).upper()
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

  raw = subprocess.run(
      [
          "ffmpeg",
          "-y",
          "-v",
          "error",
          "-f",
          "rawvideo",
          "-pix_fmt",
          "rgb24",
          "-s",
          f"{src_w}x{src_h}",
          "-i",
          "-",
          "-vf",
          f"scale={scaled_w}:{scaled_h}:flags=bilinear,pad={dst_w}:{dst_h}:{pad_x}:{pad_y}:color=black",
          "-frames:v",
          "1",
          "-f",
          "rawvideo",
          "-pix_fmt",
          "rgb24",
          "-",
      ],
      check=True,
      input=image.tobytes(),
      capture_output=True,
  ).stdout
  image_u8 = np.frombuffer(raw, dtype=np.uint8).reshape(dst_h, dst_w, 3).copy()
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


def _write_overlay_artifact(path: Path, image: np.ndarray, boxes: list[dict]) -> Path:
  SANDBOX_API_TESTS.mkdir(parents=True, exist_ok=True)

  overlay = image.copy()
  for box in boxes:
    x1 = max(0, min(int(round(box["x1"])), overlay.shape[1] - 1))
    y1 = max(0, min(int(round(box["y1"])), overlay.shape[0] - 1))
    x2 = max(0, min(int(round(box["x2"])), overlay.shape[1]))
    y2 = max(0, min(int(round(box["y2"])), overlay.shape[0]))
    if x2 <= x1 or y2 <= y1:
      continue
    overlay[y1:y2, x1 : min(x1 + 2, x2)] = [255, 0, 0]
    overlay[y1:y2, max(x2 - 2, x1):x2] = [255, 0, 0]
    overlay[y1 : min(y1 + 2, y2), x1:x2] = [255, 0, 0]
    overlay[max(y2 - 2, y1):y2, x1:x2] = [255, 0, 0]

  subprocess.run(
      [
          "ffmpeg",
          "-y",
          "-v",
          "error",
          "-f",
          "rawvideo",
          "-pix_fmt",
          "rgb24",
          "-s",
          f"{overlay.shape[1]}x{overlay.shape[0]}",
          "-i",
          "-",
          "-frames:v",
          "1",
          str(path),
      ],
      check=True,
      input=overlay.tobytes(),
      capture_output=True,
  )
  return path


def _assert_image_dims(path: Path, width: int, height: int) -> None:
  dims = subprocess.run(
      [
          "ffprobe",
          "-v",
          "error",
          "-select_streams",
          "v:0",
          "-show_entries",
          "stream=width,height",
          "-of",
          "csv=p=0:s=x",
          str(path),
      ],
      check=True,
      capture_output=True,
      text=True,
  )
  assert dims.stdout.strip() == f"{width}x{height}"


def test_resnet_real_fixture_run_preserves_stable_classification_contract():
  model = pyneat.Model(str(_env_path("SIMA_RESNET50_TAR")))
  image = _decode_rgb_image(ROOT / "test.jpg")

  outputs = [model.run(image, timeout_ms=20000) for _ in range(3)]

  assert all(out.kind == pyneat.SampleKind.Tensor for out in outputs)
  assert all(out.tensor is not None for out in outputs)

  tensors = [out.tensor for out in outputs]
  assert all(list(tensor.shape) == [1, 1, 1, 1000] for tensor in tensors)
  assert all(tensor.dtype == pyneat.TensorDType.Float32 for tensor in tensors)
  assert all(out.payload_tag == "DETESSDEQUANT" for out in outputs)
  assert all(out.format == "DETESSDEQUANT" for out in outputs)
  assert all(out.media_type == "application/vnd.simaai.tensor" for out in outputs)

  argmaxes = [int(tensor.to_numpy(copy=True).reshape(-1).argmax()) for tensor in tensors]
  assert argmaxes == [839, 839, 839]


def test_real_fixture_paths_resolve():
  assert _env_path("SIMA_RESNET50_TAR").name.endswith(".tar.gz")
  assert _env_path("SIMA_YOLO_TAR").name.endswith(".tar.gz")
  assert (ROOT / "test.jpg").is_file()
  assert (ROOT / "tests" / "images" / "people.jpg").is_file()
  assert SANDBOX_API_TESTS.parent == ROOT / "sandbox"


def test_model_backed_option_structs_preserve_real_fixture_semantics():
  resnet_model = pyneat.Model(str(_env_path("SIMA_RESNET50_TAR")))
  yolo_model = pyneat.Model(str(_env_path("SIMA_YOLO_TAR")))

  pre = pyneat.PreprocOptions(resnet_model)
  quant = pyneat.QuantTessOptions(resnet_model)
  detess = pyneat.DetessDequantOptions(yolo_model)

  assert pre.config_json is not None
  assert pre.config_json["node_name"] == "preproc"
  assert pre.num_buffers == pre.num_buffers_model == 4
  assert pre.num_buffers_locked is True

  assert quant.config_path.endswith("0_quanttess.json")
  assert quant.config_json is None
  assert quant.num_buffers == quant.num_buffers_model == 4
  assert quant.num_buffers_locked is True
  assert pyneat.nodes.quant_tess(quant) is not None

  assert detess.config_path.endswith("0_postproc.json")
  assert detess.upstream_name == "simaaiprocessmla_1"
  assert detess.element_name == "simaaiprocessdetess_dequant_1"
  assert detess.num_buffers == detess.num_buffers_model == 4
  assert detess.num_buffers_locked is True


def test_yolo_mla_group_matches_explicit_preprocess_plus_inference_structure():
  image = _decode_rgb_image(ROOT / "tests" / "images" / "people.jpg")
  model = pyneat.Model(str(_env_path("SIMA_YOLO_TAR")))

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
  image = _decode_rgb_image(ROOT / "tests" / "images" / "people.jpg")
  model = pyneat.Model(str(_env_path("SIMA_YOLO_TAR")))

  detess_output = _run_model_on_image(
      model,
      image,
      pyneat.groups.mla(model),
      pyneat.nodes.detess_dequant(pyneat.DetessDequantOptions(model)),
  )
  box_output = _run_model_on_image(
      model,
      image,
      _custom_preproc_node(model, image),
      pyneat.groups.mla(model),
      pyneat.nodes.sima_box_decode(
          model,
          "yolov8",
          image.shape[1],
          image.shape[0],
          0.25,
          0.55,
          120,
      ),
  )

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
    assert list(input_spec.shape) == [-1, -1, 3]
    assert appsrc.media_type == "application/vnd.simaai.tensor"
    assert appsrc.format == "FP32"
    assert "quanttess" in backend
    assert "preproc" not in backend


def test_cpu_quanttess_input_matches_letterboxed_fp32_tensor_contract():
  image = _decode_rgb_image(ROOT / "tests" / "images" / "people.jpg")
  model = _tensor_input_model(_env_path("SIMA_YOLO_TAR"))

  quant_input = _cpu_quanttess_input(model, image)

  assert list(quant_input.shape) == [640, 640, 3]
  assert quant_input.dtype == np.float32
  assert float(quant_input.min()) == 0.0
  assert float(quant_input.max()) <= 1.0
  assert np.allclose(quant_input[0], 0.0)
  assert np.allclose(quant_input[-1], 0.0)
  assert np.any(quant_input[106] > 0.0)
  assert np.any(quant_input[532] > 0.0)


def _run_quanttess_boxdecode_on_real_input(
    model: pyneat.Model, image: np.ndarray, quant_input: np.ndarray, preprocess_part
) -> tuple[str, pyneat.Sample]:
  session = pyneat.Session()
  session.add(pyneat.nodes.input(model.input_appsrc_options(True)))
  session.add(preprocess_part)
  session.add(pyneat.groups.mla(model))
  session.add(
      pyneat.nodes.sima_box_decode(
          model,
          "yolov8",
          image.shape[1],
          image.shape[0],
          0.25,
          0.55,
          120,
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
  image = _decode_rgb_image(ROOT / "tests" / "images" / "people.jpg")
  model = _tensor_input_model(_env_path("SIMA_YOLO_TAR"))
  quant_input = _cpu_quanttess_input(model, image)

  backend, box_output = _run_quanttess_boxdecode_on_real_input(
      model, image, quant_input, pyneat.nodes.quant_tess(pyneat.QuantTessOptions(model))
  )

  assert "quanttess" in backend
  assert "preproc" not in backend
  assert "neatprocessmla" in backend
  assert "neatboxdecode" in backend

  payload = _extract_bbox_payload(box_output)
  max_x2, max_y2 = _bbox_payload_extent(payload)
  boxes = _parse_bbox_payload(payload, image.shape[1], image.shape[0], min_score=0.25)
  out_path = _write_overlay_artifact(SANDBOX_API_TESTS / "people_model_overlay.png", image, boxes)

  assert list(quant_input.shape) == [640, 640, 3]
  assert box_output.kind == pyneat.SampleKind.Tensor
  assert box_output.payload_tag == "BBOX"
  assert boxes
  assert out_path.is_file()
  assert max_x2 <= int(image.shape[1] * 1.1)
  assert max_y2 <= int(image.shape[0] * 1.1)
  _assert_image_dims(out_path, image.shape[1], image.shape[0])


def test_custom_session_quanttess_mla_boxdecode_writes_explicit_overlay(tmp_path):
  image = _decode_rgb_image(ROOT / "tests" / "images" / "people.jpg")
  model = _tensor_input_model(_env_path("SIMA_YOLO_TAR"))
  quant_input = _cpu_quanttess_input(model, image)

  backend, box_output = _run_quanttess_boxdecode_on_real_input(
      model, image, quant_input, pyneat.nodes.quant_tess(pyneat.QuantTessOptions(model))
  )

  assert "quanttess" in backend
  assert "preproc" not in backend
  assert "neatprocessmla" in backend
  assert "neatboxdecode" in backend

  payload = _extract_bbox_payload(box_output)
  max_x2, max_y2 = _bbox_payload_extent(payload)
  boxes = _parse_bbox_payload(payload, image.shape[1], image.shape[0], min_score=0.25)
  out_path = _write_overlay_artifact(
      SANDBOX_API_TESTS / "people_quanttess_mla_boxdecode_overlay.png", image, boxes
  )

  assert list(quant_input.shape) == [640, 640, 3]
  assert box_output.kind == pyneat.SampleKind.Tensor
  assert box_output.payload_tag == "BBOX"
  assert boxes
  assert out_path.is_file()
  assert max_x2 <= int(image.shape[1] * 1.1)
  assert max_y2 <= int(image.shape[0] * 1.1)
  _assert_image_dims(out_path, image.shape[1], image.shape[0])


def test_boxdecode_runtime_output_produces_overlay_artifact(tmp_path):
  image = _decode_rgb_image(ROOT / "tests" / "images" / "people.jpg")
  model = pyneat.Model(str(_env_path("SIMA_YOLO_TAR")))

  box_output = _run_model_on_image(
      model,
      image,
      _custom_preproc_node(model, image),
      pyneat.groups.mla(model),
      pyneat.nodes.sima_box_decode(
          model,
          "yolov8",
          image.shape[1],
          image.shape[0],
          0.25,
          0.55,
          120,
      ),
  )
  payload = _extract_bbox_payload(box_output)
  max_x2, max_y2 = _bbox_payload_extent(payload)
  boxes = _parse_bbox_payload(payload, image.shape[1], image.shape[0], min_score=0.25)
  out_path = _write_overlay_artifact(
      SANDBOX_API_TESTS / "people_preproc_mla_boxdecode_overlay.png", image, boxes
  )

  assert out_path.is_file()
  assert max_x2 <= int(image.shape[1] * 1.1)
  assert max_y2 <= int(image.shape[0] * 1.1)
  _assert_image_dims(out_path, image.shape[1], image.shape[0])
  assert boxes
