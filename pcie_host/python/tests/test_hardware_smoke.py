from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import struct
from pathlib import Path

import numpy as np
import pytest

import pypciehost as pcie


def _env(name: str, default: str = "") -> str:
  value = os.environ.get(name, "")
  return value if value else default


def _env_int(name: str, default: int) -> int:
  value = _env(name)
  return int(value) if value else default


def _env_bool(name: str, default: bool) -> bool:
  value = _env(name)
  if not value:
    return default
  return value.lower() in {"1", "true", "yes", "on"}


def _require_file_env(name: str) -> Path:
  value = _env(name)
  if not value:
    pytest.skip(f"{name} is not set")
  path = Path(value)
  if not path.is_file():
    pytest.skip(f"{name} does not exist: {path}")
  return path


def _connection() -> pcie.ConnectionOptions:
  card_host = _env("SIMAPCIE_CARD_HOST")
  if not card_host:
    pytest.skip("SIMAPCIE_CARD_HOST is not set")

  conn = pcie.ConnectionOptions(
      card_host=card_host,
      card_id=_env_int("SIMAPCIE_CARD_ID", 0),
      user=_env("SIMAPCIE_USER", "sima"),
      queue=_env_int("SIMAPCIE_QUEUE", 0),
  )
  conn.card_env = _env("SIMAPCIE_CARD_ENV")
  conn.card_gst_debug = _env("SIMAPCIE_CARD_GST_DEBUG")
  conn.card_gst_debug_file = _env("SIMAPCIE_CARD_GST_DEBUG_FILE")
  return conn


def _connection_for_queue(queue: int) -> pcie.ConnectionOptions:
  conn = _connection()
  conn.queue = queue
  return conn


def _readiness_timeout_ms() -> int:
  return _env_int("SIMAPCIE_READINESS_TIMEOUT_MS", 180000)


def _pull_timeout_ms() -> int:
  return _env_int("SIMAPCIE_PULL_TIMEOUT_MS", 30000)


def _test_iterations() -> int:
  return _env_int("SIMAPCIE_TEST_ITERATIONS", 1000)


def _sync_iterations() -> int:
  return _env_int("SIMAPCIE_SYNC_ITERATIONS", 50)


def _stress_iterations() -> int:
  return _env_int("SIMAPCIE_STRESS_ITERATIONS", _test_iterations())


def _dtype_from_fact(dtype: str) -> pcie.TensorDType:
  normalized = dtype.upper()
  if normalized == "UINT8":
    return pcie.TensorDType.UInt8
  if normalized in {"INT8", "EVXX_INT8", "EV74_INT8"}:
    return pcie.TensorDType.Int8
  if normalized == "UINT16":
    return pcie.TensorDType.UInt16
  if normalized == "INT16":
    return pcie.TensorDType.Int16
  if normalized == "INT32":
    return pcie.TensorDType.Int32
  if normalized in {"BF16", "BFLOAT16", "EVXX_BFLOAT16", "EV74_BFLOAT16"}:
    return pcie.TensorDType.BFloat16
  if normalized in {"FP32", "FLOAT32"}:
    return pcie.TensorDType.Float32
  if normalized in {"FP64", "FLOAT64"}:
    return pcie.TensorDType.Float64
  raise AssertionError(f"unsupported tensor dtype from model facts: {dtype}")


def _dtype_bytes(dtype: pcie.TensorDType) -> int:
  if dtype in {pcie.TensorDType.UInt8, pcie.TensorDType.Int8}:
    return 1
  if dtype in {pcie.TensorDType.UInt16, pcie.TensorDType.Int16, pcie.TensorDType.BFloat16}:
    return 2
  if dtype in {pcie.TensorDType.Int32, pcie.TensorDType.Float32}:
    return 4
  if dtype == pcie.TensorDType.Float64:
    return 8
  raise AssertionError(f"unsupported tensor dtype: {dtype}")


def _dense_size_bytes(shape: list[int], dtype: pcie.TensorDType) -> int:
  size = _dtype_bytes(dtype)
  for dim in shape:
    assert dim >= 0
    size *= dim
  return size


def _contiguous_strides(shape: list[int], dtype: pcie.TensorDType) -> list[int]:
  stride = _dtype_bytes(dtype)
  strides = [0] * len(shape)
  for index in range(len(shape) - 1, -1, -1):
    strides[index] = stride
    stride *= shape[index]
  return strides


def _pattern_bytes(size: int, dtype: pcie.TensorDType, tensor_index: int) -> bytes:
  if dtype == pcie.TensorDType.Float32:
    values = (np.arange(size // 4, dtype=np.float32) + tensor_index) % 255
    return (values / np.float32(255.0)).astype(np.float32).tobytes()
  if dtype == pcie.TensorDType.Float64:
    values = (np.arange(size // 8, dtype=np.float64) + tensor_index) % 255
    return (values / np.float64(255.0)).astype(np.float64).tobytes()
  pattern = np.arange(size, dtype=np.uint8)
  pattern = (pattern + np.uint8((tensor_index * 17) & 0xFF)).astype(np.uint8)
  return pattern.tobytes()


def _make_inputs(info: pcie.ModelInfo) -> list[pcie.Tensor]:
  tensors = []
  for index, spec in enumerate(info.inputs):
    dtype = _dtype_from_fact(spec.dtype)
    shape = spec.shape if spec.shape else [spec.size_bytes]
    payload_size = spec.size_bytes or _dense_size_bytes(shape, dtype)
    assert payload_size > 0, f"input tensor has zero payload size: {spec.name}"
    tensor = pcie.Tensor.from_bytes(
        _pattern_bytes(payload_size, dtype, index),
        dtype,
        shape,
        _contiguous_strides(shape, dtype),
        route_name=spec.name or f"input_{index}",
    )
    tensors.append(tensor)
  return tensors


def _load_bgr_image(path: Path) -> np.ndarray:
  try:
    from PIL import Image
  except ImportError:
    pytest.skip("Pillow is required for image PCIe Python hardware tests")

  with Image.open(path) as image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
  return np.ascontiguousarray(rgb[:, :, ::-1])


def _image_options(image: np.ndarray) -> pcie.ModelOptions:
  options = pcie.ModelOptions()
  options.preprocess.kind = pcie.InputKind.Image
  options.preprocess.color_convert.input_format = pcie.ColorFormat.BGR
  options.preprocess.input_max_width = int(image.shape[1])
  options.preprocess.input_max_height = int(image.shape[0])
  options.preprocess.input_max_depth = int(image.shape[2])
  options.preprocess.resize.enable = pcie.AutoFlag.On
  options.preprocess.resize.mode = pcie.ResizeMode.Letterbox
  return options


def _assert_outputs_match_metadata(outputs: list[pcie.Tensor], expected: list[pcie.TensorInfo]):
  assert len(outputs) == len(expected)
  for output, spec in zip(outputs, expected):
    assert output.route.name == spec.name
    assert output.shape == spec.shape
    assert output.size_bytes == spec.size_bytes


def _stress_queues() -> list[int]:
  queues = _env("SIMAPCIE_STRESS_QUEUES", "0 1 2 3").split()
  assert queues, "SIMAPCIE_STRESS_QUEUES must contain at least one queue"
  return [int(queue) for queue in queues]


def _run_sync_push_pull(push_once, pull_once, iterations: int):
  assert iterations > 0
  for _ in range(iterations):
    push_once()
    pull_once()


def _run_async_push_pull(push_once, pull_once, iterations: int):
  assert iterations > 0

  def produce():
    for _ in range(iterations):
      push_once()

  def consume():
    for _ in range(iterations):
      pull_once()

  with ThreadPoolExecutor(max_workers=2) as executor:
    producer = executor.submit(produce)
    consumer = executor.submit(consume)
    producer.result()
    consumer.result()


def test_metadata_yolov8():
  model = _require_file_env("SIMAPCIE_YOLOV8_MODEL")
  host = pcie.SimaPCIeHost()

  info = host.load_metadata(str(model))

  assert [tensor.name for tensor in info.inputs] == ["images"]
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


def test_tensor_run_yolov8():
  model = _require_file_env("SIMAPCIE_YOLOV8_MODEL")
  host = pcie.SimaPCIeHost(_connection())
  sync_iterations = _sync_iterations()
  async_iterations = _test_iterations()

  try:
    info = host.init_pipeline(str(model), pcie.ModelOptions(), _readiness_timeout_ms())
    inputs = _make_inputs(info)

    def push_once():
      assert host.push(inputs)

    def pull_once():
      outputs = host.pull(_pull_timeout_ms())

      assert outputs is not None
      _assert_outputs_match_metadata(outputs, info.outputs)

    _run_sync_push_pull(push_once, pull_once, sync_iterations)
    _run_async_push_pull(push_once, pull_once, async_iterations)
  finally:
    host.stop()


def test_tensor_parallel_queues_yolov8():
  model = _require_file_env("SIMAPCIE_YOLOV8_MODEL")
  queues = _stress_queues()
  sync_iterations = _sync_iterations()
  iterations = _stress_iterations()
  assert iterations > 0

  def run_queue(queue: int):
    host = pcie.SimaPCIeHost(_connection_for_queue(queue))
    try:
      info = host.init_pipeline(str(model), pcie.ModelOptions(), _readiness_timeout_ms())
      inputs = _make_inputs(info)

      def push_once():
        assert host.push(inputs)

      def pull_once():
        outputs = host.pull(_pull_timeout_ms())
        assert outputs is not None
        _assert_outputs_match_metadata(outputs, info.outputs)

      _run_sync_push_pull(push_once, pull_once, sync_iterations)
      _run_async_push_pull(push_once, pull_once, iterations)
      return queue
    finally:
      host.stop()

  with ThreadPoolExecutor(max_workers=len(queues)) as executor:
    futures = [executor.submit(run_queue, queue) for queue in queues]
    completed = sorted(future.result() for future in as_completed(futures))

  assert completed == sorted(queues)


def test_image_run_yolov8():
  model = _require_file_env("SIMAPCIE_YOLOV8_MODEL")
  image_path = _require_file_env("SIMAPCIE_TEST_IMAGE")
  image = _load_bgr_image(image_path)
  host = pcie.SimaPCIeHost(_connection())
  sync_iterations = _sync_iterations()
  async_iterations = _test_iterations()

  try:
    options = _image_options(image)
    info = host.init_pipeline(str(model), options, _readiness_timeout_ms())
    image_tensor = pcie.Tensor.from_numpy(
        image,
        image_format=pcie.PixelFormat.BGR,
        route_name="input_image",
    )

    def push_once():
      assert host.push(image_tensor)

    def pull_once():
      outputs = host.pull(_pull_timeout_ms())

      assert outputs is not None
      _assert_outputs_match_metadata(outputs, info.outputs)

    _run_sync_push_pull(push_once, pull_once, sync_iterations)
    _run_async_push_pull(push_once, pull_once, async_iterations)
  finally:
    host.stop()


def _parse_bbox_payload(outputs: list[pcie.Tensor], image_width: int, image_height: int, top_k: int):
  assert len(outputs) == 1
  payload = outputs[0].to_bytes()
  assert len(payload) >= 4

  count = struct.unpack_from("<I", payload, 0)[0]
  record_size = 24
  capacity = (len(payload) - 4) // record_size
  assert count <= capacity
  assert count <= top_k

  records = []
  for index in range(count):
    base = 4 + index * record_size
    x, y, w, h, score, class_id = struct.unpack_from("<iiiifi", payload, base)
    assert 0.0 <= score <= 1.0
    assert class_id >= 0
    assert w > 0 and h > 0
    assert x >= -2 and y >= -2
    assert x + w <= image_width + 2
    assert y + h <= image_height + 2
    records.append((x, y, w, h, score, class_id))
  return records


def test_image_boxdecode_run_yolov8():
  model = _require_file_env("SIMAPCIE_YOLOV8_MODEL")
  image_path = Path(_env("SIMAPCIE_BOXDECODE_IMAGE") or _env("SIMAPCIE_TEST_IMAGE"))
  if not image_path.is_file():
    pytest.skip("SIMAPCIE_BOXDECODE_IMAGE/SIMAPCIE_TEST_IMAGE is not set to an existing file")
  image = _load_bgr_image(image_path)
  host = pcie.SimaPCIeHost(_connection())

  score_threshold = float(_env("SIMAPCIE_BOXDECODE_SCORE_THRESHOLD", "0.25"))
  top_k = _env_int("SIMAPCIE_BOXDECODE_TOP_K", 100)
  require_person = _env_bool("SIMAPCIE_BOXDECODE_REQUIRE_PERSON", True)
  sync_iterations = _sync_iterations()
  async_iterations = _test_iterations()

  try:
    options = _image_options(image)
    options.decode_type = pcie.BoxDecodeType.YoloV8
    options.score_threshold = score_threshold
    options.nms_iou_threshold = float(_env("SIMAPCIE_BOXDECODE_NMS_IOU_THRESHOLD", "0.45"))
    options.top_k = top_k

    host.init_pipeline(str(model), options, _readiness_timeout_ms())

    image_tensor = pcie.Tensor.from_numpy(
        image,
        image_format=pcie.PixelFormat.BGR,
        route_name="input_image",
    )

    def push_once():
      assert host.push(image_tensor)

    def pull_once():
      outputs = host.pull(_pull_timeout_ms())

      assert outputs is not None
      records = _parse_bbox_payload(outputs, image.shape[1], image.shape[0], top_k)
      if _env("SIMAPCIE_BOXDECODE_ALLOW_EMPTY", "") not in {"1", "true", "TRUE", "yes"}:
        assert any(record[4] >= score_threshold for record in records)
      if require_person:
        assert any(record[5] == 0 for record in records)

    _run_sync_push_pull(push_once, pull_once, sync_iterations)
    _run_async_push_pull(push_once, pull_once, async_iterations)
  finally:
    host.stop()
