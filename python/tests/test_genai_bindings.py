import os
from pathlib import Path

import numpy as np
import pyneat
import pytest


_MODEL_ENV = "SIMA_TEST_LLIMA_TEXT_MODEL"
_LEGACY_MODEL_ENV = "SIMA_NEAT_GENAI_TEST_MODEL"
_VLM_MODEL_ENV = "SIMA_TEST_LLIMA_VLM_MODEL"
_VLM_REPO_ID = "simaai/LFM2-VL-450M-a16w4"
_VLM_PROMPT = "Describe this image in a short phrase."
_EXPECTED_VLM_TEXT = "Skier jumping high in the air."
_PROMPT = "What is the capital of Germany?"
_EXPECTED_TEXT = "The capital of Germany is Berlin."


def _trim_text(text):
  return text.strip()


def _text_model_dir():
  value = os.environ.get(_MODEL_ENV, "").strip()
  if not value:
    value = os.environ.get(_LEGACY_MODEL_ENV, "").strip()
  if not value:
    pytest.skip(f"{_MODEL_ENV} is not set")

  model_dir = Path(value)
  if not (model_dir / "devkit" / "vlm_config.json").is_file():
    pytest.skip(f"{model_dir} is not a LLiMa VLM model directory")
  return str(model_dir)


def _vlm_model_dir():
  value = os.environ.get(_VLM_MODEL_ENV, "").strip()
  if not value:
    pytest.skip(f"{_VLM_MODEL_ENV} is not set; expected {_VLM_REPO_ID}")

  model_dir = Path(value)
  if not (model_dir / "devkit" / "vlm_config.json").is_file():
    pytest.skip(f"{model_dir} is not a LLiMa VLM model directory")
  return str(model_dir)


def _candidate_roots():
  roots = []

  def append_unique(path):
    path = Path(path)
    if path not in roots:
      roots.append(path)

  for base in (Path.cwd(), Path(__file__).resolve()):
    append_unique(base)
    for parent in base.parents:
      append_unique(parent)
  return roots


def _people_image_path():
  rel = Path("tests/images/people.jpg")
  for root in _candidate_roots():
    candidate = root / rel
    if candidate.is_file():
      return candidate
  pytest.skip(f"missing image fixture: {rel}")


def _people_rgb_image():
  cv2 = pytest.importorskip("cv2", exc_type=Exception)
  image_bgr = cv2.imread(str(_people_image_path()), cv2.IMREAD_COLOR)
  if image_bgr is None:
    raise AssertionError("failed to decode tests/images/people.jpg")
  image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
  return np.asarray(image_rgb)


def _skip_if_dispatcher_unavailable(exc):
  text = str(exc).lower()
  if "dispatcher unavailable" in text or "connecting to server failed" in text:
    pytest.skip("dispatcher unavailable")


def _make_request():
  request = pyneat.GenerationRequest()
  request.system_prompt = "You are concise."
  request.prompt = _PROMPT
  request.max_new_tokens = 24
  return request


def _assert_finish_reason(value):
  assert value in ("stop", "interrupted")


def _bundle_field_text(bundle, name):
  assert bundle.kind == pyneat.SampleKind.Bundle
  for field in bundle.fields:
    if field.port_name == name or field.stream_label == name:
      return field.to_text()
  raise AssertionError(f"missing bundle field: {name}")


def _pull_language_outputs(run, node_id, stop_on_error=False):
  tokens = []
  done = None
  error = None
  token_samples = 0

  for _ in range(64):
    sample = run.pull(node_id, 60000)
    assert sample is not None, "GraphRun.pull timed out"
    port = sample.port_name or sample.stream_label
    if port == "tokens":
      tokens.append(sample.to_text())
      token_samples += 1
    elif port == "done":
      done = sample
      break
    elif port == "error":
      error = sample.to_text()
      if stop_on_error:
        break
    else:
      raise AssertionError(f"unexpected graph output port: {port!r}")

  return "".join(tokens), done, error, token_samples


def _pull_encoded(run, node_id):
  for _ in range(16):
    sample = run.pull(node_id, 60000)
    assert sample is not None, "GraphRun.pull encoded timed out"
    port = sample.port_name or sample.stream_label
    if port == "encoded":
      return sample
    if port == "error":
      raise AssertionError(f"image input emitted error: {sample.to_text()}")
    raise AssertionError(f"unexpected graph output before encoded: {port!r}")
  raise AssertionError("image input did not emit encoded")


def _make_image_sample(image, memory=pyneat.TensorMemory.CPU):
  tensor = pyneat.Tensor.from_numpy(
      image,
      copy=True,
      image_format=pyneat.PixelFormat.RGB,
      memory=memory,
  )
  sample = pyneat.make_tensor_sample("image", tensor)
  sample.stream_label = "image"
  return sample


def test_genai_value_types_and_text_sample_helpers():
  req = pyneat.GenerationRequest()
  req.prompt = "hello"
  req.system_prompt = "be concise"
  req.max_new_tokens = 8
  req.temperature = 0.25
  req.top_p = 0.9
  req.messages = [pyneat.ChatMessage()]
  req.messages[0].role = "user"
  req.messages[0].content = "hello"
  req.images = [np.zeros((2, 2, 3), dtype=np.uint8)]
  req.use_cached_images = False

  assert req.prompt == "hello"
  assert req.system_prompt == "be concise"
  assert req.messages[0].content == "hello"
  assert len(req.images) == 1
  assert list(req.images[0].shape) == [2, 2, 3]
  assert req.images[0].image_format() == pyneat.PixelFormat.RGB
  assert req.use_cached_images is False

  image_list = pyneat.ImageList(req.images)
  assert not image_list.empty()
  assert image_list.size() == 1

  req.messages[0].images = req.images
  req.messages[0].use_cached_images = False
  assert len(req.messages[0].images) == 1
  assert req.messages[0].use_cached_images is False

  tensor = pyneat.Tensor.from_text("token")
  assert tensor.to_text() == "token"

  sample = pyneat.make_text_sample("prompt", "What is NEAT?")
  assert sample.kind == pyneat.SampleKind.TensorSet
  assert sample.port_name == "prompt"
  assert sample.stream_label == "prompt"
  assert sample.to_text() == "What is NEAT?"

  result = pyneat.GenerationResult()
  result.text = "done"
  result.finish_reason = "stop"
  result.metrics.generated_tokens = 1
  assert result.metrics.generated_tokens == 1

  token = pyneat.TokenSample()
  token.text = "d"
  token.is_final = False
  assert token.text == "d"


def test_genai_top_level_and_namespace_aliases_exist():
  assert pyneat.genai.VisionLanguageModel is pyneat.VisionLanguageModel
  assert pyneat.genai.GenAIModel is pyneat.GenAIModel
  assert pyneat.genai.ImageList is pyneat.ImageList
  assert pyneat.genai.GenerationRequest is pyneat.GenerationRequest
  assert hasattr(pyneat.genai.nodes, "vision_language")
  assert hasattr(pyneat.graph.nodes, "genai_vision_language")
  assert not hasattr(pyneat.genai.nodes, "language")
  assert not hasattr(pyneat.graph.nodes, "genai_language")
  assert not hasattr(pyneat.genai.nodes, "LanguageOptions")
  assert not hasattr(pyneat.graph.nodes, "GenAIVisionLanguageOptions")

  options = pyneat.genai.nodes.VisionLanguageOptions()
  options.system_prompt = "Answer exactly."
  options.max_new_tokens = 24
  options.streaming = False
  options.encode_images_on_input = False
  assert options.system_prompt == "Answer exactly."
  assert options.max_new_tokens == 24
  assert options.streaming is False
  assert options.encode_images_on_input is False


def test_genai_direct_text_generation_and_streaming():
  try:
    model = pyneat.VisionLanguageModel(_text_model_dir())
    assert not model.accepts_image()

    request = _make_request()
    result = model.run(request)
    assert _trim_text(result.text) == _EXPECTED_TEXT
    _assert_finish_reason(result.finish_reason)
    assert result.metrics.generated_tokens > 0

    streamed_text = []
    final_sample = None
    for sample in model.stream(request):
      if sample.is_final:
        final_sample = sample
        break
      streamed_text.append(sample.text)

    assert streamed_text
    assert final_sample is not None
    assert _trim_text("".join(streamed_text)) == _EXPECTED_TEXT
    _assert_finish_reason(final_sample.finish_reason)
    assert final_sample.metrics.generated_tokens > 0
  except Exception as exc:
    _skip_if_dispatcher_unavailable(exc)
    raise


def test_genai_direct_vlm_generation_and_cached_image():
  try:
    model = pyneat.VisionLanguageModel(_vlm_model_dir())
    assert model.accepts_image()

    image = _people_rgb_image()

    request = pyneat.GenerationRequest()
    request.prompt = _VLM_PROMPT
    request.images = [image]
    request.max_new_tokens = 48
    result = model.run(request)
    assert _trim_text(result.text) == _EXPECTED_VLM_TEXT
    _assert_finish_reason(result.finish_reason)
    assert result.metrics.generated_tokens > 0
    print(f"GENAI_PY_VLM text={result.text}")

    ev74_tensor = pyneat.Tensor.from_numpy(
        image,
        copy=True,
        image_format=pyneat.PixelFormat.RGB,
        memory=pyneat.TensorMemory.EV74,
    )
    ev74_request = pyneat.GenerationRequest()
    ev74_request.prompt = _VLM_PROMPT
    ev74_request.images = [ev74_tensor]
    ev74_request.max_new_tokens = 48
    ev74_result = model.run(ev74_request)
    assert _trim_text(ev74_result.text) == _EXPECTED_VLM_TEXT
    _assert_finish_reason(ev74_result.finish_reason)
    assert ev74_result.metrics.generated_tokens > 0
    print(f"GENAI_PY_VLM_EV74_TENSOR text={ev74_result.text}")

    streamed_text = []
    final_sample = None
    for sample in model.stream(request):
      if sample.is_final:
        final_sample = sample
        break
      streamed_text.append(sample.text)

    assert streamed_text
    assert final_sample is not None
    assert _trim_text("".join(streamed_text)) == _EXPECTED_VLM_TEXT
    _assert_finish_reason(final_sample.finish_reason)
    assert final_sample.metrics.generated_tokens > 0
    print(f"GENAI_PY_VLM_STREAM text={''.join(streamed_text)}")

    assert model.encode([image])
    assert model.cached_image_count() == 1

    cached = pyneat.GenerationRequest()
    cached.prompt = _VLM_PROMPT
    cached.use_cached_images = True
    cached.max_new_tokens = 48
    cached_result = model.run(cached)
    assert _trim_text(cached_result.text) == _EXPECTED_VLM_TEXT
    _assert_finish_reason(cached_result.finish_reason)
    assert cached_result.metrics.generated_tokens > 0
    print(f"GENAI_PY_VLM_CACHED text={cached_result.text}")

  except Exception as exc:
    _skip_if_dispatcher_unavailable(exc)
    raise


def test_genai_language_graph_node_generation_and_errors():
  try:
    model = pyneat.VisionLanguageModel(_text_model_dir())

    graph = pyneat.graph.Graph()
    prompt_port = graph.intern_port("prompt")

    streaming_options = pyneat.genai.nodes.VisionLanguageOptions()
    streaming_options.system_prompt = "You are concise."
    streaming_options.max_new_tokens = 24
    streaming_options.streaming = True
    streaming_node = graph.add(
        pyneat.genai.nodes.vision_language(
            model, streaming_options, "vision_language_streaming"
        )
    )

    sync_options = pyneat.genai.nodes.VisionLanguageOptions()
    sync_options.system_prompt = "You are concise."
    sync_options.max_new_tokens = 24
    sync_options.streaming = False
    sync_node = graph.add(
        pyneat.graph.nodes.genai_vision_language(model, sync_options, "vision_language_sync")
    )

    run = pyneat.graph.GraphSession(graph).build()
    try:
      assert run.push_port(streaming_node, prompt_port, pyneat.make_text_sample("prompt", _PROMPT))
      text, done, error, token_samples = _pull_language_outputs(run, streaming_node)
      assert error is None
      assert done is not None
      assert token_samples > 0
      assert _trim_text(text) == _EXPECTED_TEXT
      _assert_finish_reason(_bundle_field_text(done, "finish_reason"))
      assert int(_bundle_field_text(done, "generated_tokens")) > 0

      assert run.push_port(sync_node, prompt_port, pyneat.make_text_sample("prompt", _PROMPT))
      text, done, error, token_samples = _pull_language_outputs(run, sync_node)
      assert error is None
      assert done is not None
      assert token_samples == 1
      assert _trim_text(text) == _EXPECTED_TEXT

      invalid = pyneat.Sample()
      invalid.kind = pyneat.SampleKind.Tensor
      invalid.tensor = pyneat.Tensor()
      invalid.port_name = "prompt"
      invalid.stream_label = "prompt"
      assert run.push_port(streaming_node, prompt_port, invalid)
      _, _, error, _ = _pull_language_outputs(run, streaming_node, stop_on_error=True)
      assert error
    finally:
      run.stop()
  except Exception as exc:
    _skip_if_dispatcher_unavailable(exc)
    raise


def test_genai_vision_language_graph_node_generation_and_errors():
  try:
    model = pyneat.VisionLanguageModel(_vlm_model_dir())
    assert model.accepts_image()
    image = _people_rgb_image()

    graph = pyneat.graph.Graph()
    prompt_port = graph.intern_port("prompt")
    image_port = graph.intern_port("image")
    use_cached_image_port = graph.intern_port("use_cached_image")

    streaming_options = pyneat.genai.nodes.VisionLanguageOptions()
    streaming_options.max_new_tokens = 48
    streaming_options.streaming = True
    streaming_options.encode_images_on_input = True
    streaming_node = graph.add(
        pyneat.genai.nodes.vision_language(
            model, streaming_options, "vision_language_streaming"
        )
    )

    sync_options = pyneat.genai.nodes.VisionLanguageOptions()
    sync_options.max_new_tokens = 48
    sync_options.streaming = False
    sync_options.encode_images_on_input = False
    sync_node = graph.add(
        pyneat.graph.nodes.genai_vision_language(model, sync_options, "vision_language_sync")
    )

    missing_image_node = graph.add(
        pyneat.genai.nodes.vision_language(
            model, streaming_options, "vision_language_missing_image"
        )
    )

    run = pyneat.graph.GraphSession(graph).build()
    try:
      assert run.push_port(streaming_node, image_port, _make_image_sample(image))
      encoded = _pull_encoded(run, streaming_node)
      assert _bundle_field_text(encoded, "mode") == "cached"
      assert int(_bundle_field_text(encoded, "cached_image_count")) == 1

      assert run.push_port(
          streaming_node,
          use_cached_image_port,
          pyneat.make_text_sample("use_cached_image", "true"),
      )
      assert run.push_port(
          streaming_node,
          prompt_port,
          pyneat.make_text_sample("prompt", _VLM_PROMPT),
      )
      text, done, error, token_samples = _pull_language_outputs(run, streaming_node)
      assert error is None
      assert done is not None
      assert token_samples > 0
      assert _trim_text(text) == _EXPECTED_VLM_TEXT
      _assert_finish_reason(_bundle_field_text(done, "finish_reason"))
      assert int(_bundle_field_text(done, "generated_tokens")) > 0

      assert run.push_port(sync_node, image_port, _make_image_sample(image))
      encoded = _pull_encoded(run, sync_node)
      assert _bundle_field_text(encoded, "mode") == "direct"

      assert run.push_port(
          sync_node,
          use_cached_image_port,
          pyneat.make_text_sample("use_cached_image", "false"),
      )
      assert run.push_port(
          sync_node,
          prompt_port,
          pyneat.make_text_sample("prompt", _VLM_PROMPT),
      )
      text, done, error, token_samples = _pull_language_outputs(run, sync_node)
      assert error is None
      assert done is not None
      assert token_samples == 1
      assert _trim_text(text) == _EXPECTED_VLM_TEXT

      invalid = pyneat.make_text_sample("image", "not-an-image")
      assert run.push_port(streaming_node, image_port, invalid)
      _, _, error, _ = _pull_language_outputs(run, streaming_node, stop_on_error=True)
      assert error

      assert run.push_port(
          missing_image_node,
          prompt_port,
          pyneat.make_text_sample("prompt", _VLM_PROMPT),
      )
      _, _, error, _ = _pull_language_outputs(run, missing_image_node, stop_on_error=True)
      assert error
    finally:
      run.stop()
  except Exception as exc:
    _skip_if_dispatcher_unavailable(exc)
    raise
