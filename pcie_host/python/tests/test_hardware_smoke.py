from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import struct
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import pyneatpcie as pcie


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
      max_inflight=_env_int("SIMAPCIE_MAX_INFLIGHT", 10),
  )
  conn.card_env = _env("SIMAPCIE_CARD_ENV")
  conn.card_gst_debug = _env("SIMAPCIE_CARD_GST_DEBUG")
  conn.card_gst_debug_file = _env("SIMAPCIE_CARD_GST_DEBUG_FILE")
  return conn


def _connection_for_queue(queue: int) -> pcie.ConnectionOptions:
  conn = _connection()
  conn.queue = queue
  if conn.card_gst_debug_file:
    conn.card_gst_debug_file = conn.card_gst_debug_file.replace("{queue}", str(queue))
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


def _run_async_push_pull(push_once, pull_once, iterations: int, cancel):
  assert iterations > 0
  cancelled = threading.Event()
  failure_lock = threading.Lock()
  first_failure = []

  def produce():
    for _ in range(iterations):
      if cancelled.is_set():
        return
      push_once()

  def consume():
    for _ in range(iterations):
      if cancelled.is_set():
        return
      pull_once()

  def run(worker):
    try:
      worker()
    except BaseException as error:
      with failure_lock:
        first = not first_failure
        if first:
          first_failure.append((error, error.__traceback__))
          cancelled.set()
      if first:
        try:
          cancel()
        except BaseException:
          # Preserve the worker failure that caused cancellation.
          pass

  with ThreadPoolExecutor(max_workers=2) as executor:
    producer = executor.submit(run, produce)
    consumer = executor.submit(run, consume)
    producer.result()
    consumer.result()

  if first_failure:
    error, traceback = first_failure[0]
    raise error.with_traceback(traceback)


def test_async_test_runner_cancels_blocked_peer():
  producer_waiting = threading.Event()
  stop_requested = threading.Event()
  cancel_calls = 0

  def push_once():
    producer_waiting.set()
    assert stop_requested.wait(5), "blocked producer was not cancelled"

  def pull_once():
    assert producer_waiting.wait(5), "producer did not enter its blocking operation"
    raise RuntimeError("original consumer failure")

  def cancel():
    nonlocal cancel_calls
    cancel_calls += 1
    stop_requested.set()

  with pytest.raises(RuntimeError, match="^original consumer failure$"):
    _run_async_push_pull(push_once, pull_once, 1, cancel)

  assert cancel_calls == 1


def test_metadata_yolov8():
  model = _require_file_env("SIMAPCIE_YOLOV8_MODEL")
  runtime = pcie.Model(str(model))

  info = runtime.info()

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


def test_runtime_request_correlation():
  model = _require_file_env("SIMAPCIE_YOLOV8_MODEL")
  info = pcie.Model(str(model)).info()
  inputs = _make_inputs(info)
  request_id = -123456789

  with pcie.Runtime(_connection()) as runtime:
    model_id = runtime.load(
        str(model),
        readiness_timeout_ms=_readiness_timeout_ms(),
    )
    assert runtime.try_enqueue(
        model_id,
        request_id,
        inputs,
    ) == pcie.EnqueueResult.Accepted

    completion = runtime.retrieve(_pull_timeout_ms())
    assert completion is not None
    assert completion.model_id == model_id
    assert completion.request_id == request_id
    _assert_outputs_match_metadata(completion.outputs, info.outputs)
    runtime.unload(model_id, _pull_timeout_ms())


def test_runtime_two_models_and_independent_unload():
  model = _require_file_env("SIMAPCIE_YOLOV8_MODEL")
  info = pcie.Model(str(model)).info()
  inputs = _make_inputs(info)

  configs = []
  for _ in range(2):
    config = pcie.ModelConfig()
    config.path = str(model)
    configs.append(config)

  with pcie.Runtime(_connection()) as runtime:
    model_ids = runtime.load_models(
        configs,
        readiness_timeout_ms=_readiness_timeout_ms(),
    )
    assert len(model_ids) == 2
    assert model_ids[0] != model_ids[1]

    expected = {
        (model_ids[0], 101),
        (model_ids[1], 202),
    }
    for model_id, request_id in expected:
      assert runtime.try_enqueue(
          model_id,
          request_id,
          inputs,
      ) == pcie.EnqueueResult.Accepted

    received = set()
    for _ in expected:
      completion = runtime.retrieve(_pull_timeout_ms())
      assert completion is not None
      received.add((completion.model_id, completion.request_id))
      _assert_outputs_match_metadata(completion.outputs, info.outputs)
    assert received == expected

    runtime.unload(model_ids[0], _pull_timeout_ms())

    assert runtime.try_enqueue(
        model_ids[1],
        303,
        inputs,
    ) == pcie.EnqueueResult.Accepted
    completion = runtime.retrieve(_pull_timeout_ms())
    assert completion is not None
    assert completion.model_id == model_ids[1]
    assert completion.request_id == 303
    _assert_outputs_match_metadata(completion.outputs, info.outputs)
    runtime.unload(model_ids[1], _pull_timeout_ms())


def test_tensor_run_yolov8():
  model = _require_file_env("SIMAPCIE_YOLOV8_MODEL")
  runtime = None
  sync_iterations = _sync_iterations()
  async_iterations = _test_iterations()

  try:
    runtime = pcie.Model(str(model), pcie.ModelOptions(), _connection())
    runtime.build(_readiness_timeout_ms())
    info = runtime.info()
    inputs = _make_inputs(info)

    def push_once():
      assert runtime.push(inputs)

    def pull_once():
      outputs = runtime.pull(_pull_timeout_ms())

      assert outputs is not None
      _assert_outputs_match_metadata(outputs, info.outputs)

    _run_sync_push_pull(push_once, pull_once, sync_iterations)
    _run_async_push_pull(push_once, pull_once, async_iterations, runtime.close)
  finally:
    if runtime is not None:
      runtime.close()


def test_tensor_latency_yolov8():
  model = _require_file_env("SIMAPCIE_YOLOV8_MODEL")
  runtime = None
  warmup = 100
  iterations = 3000
  max_mean_latency_ms = 14.0

  try:
    runtime = pcie.Model(str(model), pcie.ModelOptions(), _connection())
    runtime.build(_readiness_timeout_ms())
    info = runtime.info()
    inputs = _make_inputs(info)

    for _ in range(warmup):
      assert runtime.push(inputs)
      outputs = runtime.pull(_pull_timeout_ms())
      assert outputs is not None
      _assert_outputs_match_metadata(outputs, info.outputs)

    latencies_ms = []
    for _ in range(iterations):
      started = time.perf_counter()
      assert runtime.push(inputs)
      outputs = runtime.pull(_pull_timeout_ms())
      elapsed_ms = (time.perf_counter() - started) * 1000.0
      assert outputs is not None
      _assert_outputs_match_metadata(outputs, info.outputs)
      latencies_ms.append(elapsed_ms)

    mean_latency_ms = sum(latencies_ms) / len(latencies_ms)
    print(f"Python PCIe mean latency: {mean_latency_ms:.3f} ms")
    assert mean_latency_ms < max_mean_latency_ms
  finally:
    if runtime is not None:
      runtime.close()


def test_tensor_throughput_yolov8():
  model = _require_file_env("SIMAPCIE_YOLOV8_MODEL")
  runtime = None
  warmup = 100
  iterations = 3000
  min_throughput_fps = 380.0

  try:
    runtime = pcie.Model(str(model), pcie.ModelOptions(), _connection())
    runtime.build(_readiness_timeout_ms())
    info = runtime.info()
    inputs = _make_inputs(info)

    for _ in range(warmup):
      assert runtime.push(inputs)
      outputs = runtime.pull(_pull_timeout_ms())
      assert outputs is not None
      _assert_outputs_match_metadata(outputs, info.outputs)

    def push_once():
      assert runtime.push(inputs)

    def pull_once():
      outputs = runtime.pull(_pull_timeout_ms())
      assert outputs is not None
      _assert_outputs_match_metadata(outputs, info.outputs)

    started = time.perf_counter()
    _run_async_push_pull(push_once, pull_once, iterations, runtime.close)
    elapsed_seconds = time.perf_counter() - started
    throughput_fps = iterations / elapsed_seconds

    print(f"Python PCIe throughput: {throughput_fps:.3f} FPS")
    assert throughput_fps > min_throughput_fps
  finally:
    if runtime is not None:
      runtime.close()


def test_tensor_parallel_queues_yolov8():
  model = _require_file_env("SIMAPCIE_YOLOV8_MODEL")
  queues = _stress_queues()
  sync_iterations = _sync_iterations()
  iterations = _stress_iterations()
  assert iterations > 0

  def run_queue(queue: int):
    runtime = None
    try:
      runtime = pcie.Model(str(model), pcie.ModelOptions(), _connection_for_queue(queue))
      runtime.build(_readiness_timeout_ms())
      info = runtime.info()
      inputs = _make_inputs(info)

      for iteration in range(1, sync_iterations + 1):
        assert runtime.push(inputs), (
            f"queue {queue} synchronous push {iteration}/{sync_iterations} returned false"
        )
        outputs = runtime.pull(_pull_timeout_ms())
        assert outputs is not None, (
            f"queue {queue} synchronous pull {iteration}/{sync_iterations} "
            f"timed out after {_pull_timeout_ms()} ms"
        )
        _assert_outputs_match_metadata(outputs, info.outputs)

      push_iteration = 0
      pull_iteration = 0

      def push_once():
        nonlocal push_iteration
        push_iteration += 1
        assert runtime.push(inputs), (
            f"queue {queue} asynchronous push {push_iteration}/{iterations} returned false"
        )

      def pull_once():
        nonlocal pull_iteration
        pull_iteration += 1
        outputs = runtime.pull(_pull_timeout_ms())
        assert outputs is not None, (
            f"queue {queue} asynchronous pull {pull_iteration}/{iterations} "
            f"timed out after {_pull_timeout_ms()} ms"
        )
        _assert_outputs_match_metadata(outputs, info.outputs)

      _run_async_push_pull(push_once, pull_once, iterations, runtime.close)
      return queue
    finally:
      if runtime is not None:
        runtime.close()

  with ThreadPoolExecutor(max_workers=len(queues)) as executor:
    futures = [executor.submit(run_queue, queue) for queue in queues]
    completed = sorted(future.result() for future in as_completed(futures))

  assert completed == sorted(queues)


def test_image_run_yolov8():
  model = _require_file_env("SIMAPCIE_YOLOV8_MODEL")
  image_path = _require_file_env("SIMAPCIE_TEST_IMAGE")
  image = _load_bgr_image(image_path)
  sync_iterations = _sync_iterations()
  async_iterations = _test_iterations()

  runtime = None
  try:
    options = _image_options(image)
    runtime = pcie.Model(str(model), options, _connection())
    runtime.build(_readiness_timeout_ms())
    info = runtime.info()
    image_tensor = pcie.Tensor.from_numpy(
        image,
        image_format=pcie.PixelFormat.BGR,
        route_name="input_image",
    )

    def push_once():
      assert runtime.push(image_tensor)

    def pull_once():
      outputs = runtime.pull(_pull_timeout_ms())

      assert outputs is not None
      _assert_outputs_match_metadata(outputs, info.outputs)

    _run_sync_push_pull(push_once, pull_once, sync_iterations)
    _run_async_push_pull(push_once, pull_once, async_iterations, runtime.close)
  finally:
    if runtime is not None:
      runtime.close()


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

  score_threshold = float(_env("SIMAPCIE_BOXDECODE_SCORE_THRESHOLD", "0.25"))
  top_k = _env_int("SIMAPCIE_BOXDECODE_TOP_K", 100)
  require_person = _env_bool("SIMAPCIE_BOXDECODE_REQUIRE_PERSON", True)
  sync_iterations = _sync_iterations()
  async_iterations = _test_iterations()

  runtime = None
  try:
    options = _image_options(image)
    options.decode_type = pcie.BoxDecodeType.YoloV8
    options.score_threshold = score_threshold
    options.nms_iou_threshold = float(_env("SIMAPCIE_BOXDECODE_NMS_IOU_THRESHOLD", "0.45"))
    options.top_k = top_k

    runtime = pcie.Model(str(model), options, _connection())
    runtime.build(_readiness_timeout_ms())

    image_tensor = pcie.Tensor.from_numpy(
        image,
        image_format=pcie.PixelFormat.BGR,
        route_name="input_image",
    )

    def push_once():
      assert runtime.push(image_tensor)

    def pull_once():
      outputs = runtime.pull(_pull_timeout_ms())

      assert outputs is not None
      records = _parse_bbox_payload(outputs, image.shape[1], image.shape[0], top_k)
      if _env("SIMAPCIE_BOXDECODE_ALLOW_EMPTY", "") not in {"1", "true", "TRUE", "yes"}:
        assert any(record[4] >= score_threshold for record in records)
      if require_person:
        assert any(record[5] == 0 for record in records)

    _run_sync_push_pull(push_once, pull_once, sync_iterations)
    _run_async_push_pull(push_once, pull_once, async_iterations, runtime.close)
  finally:
    if runtime is not None:
      runtime.close()
