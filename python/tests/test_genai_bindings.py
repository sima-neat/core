import base64
import os
import socket
import time
from pathlib import Path

import numpy as np
import pyneat
import pytest


_MODEL_ENV = "SIMA_TEST_LLIMA_TEXT_MODEL"
_LEGACY_MODEL_ENV = "SIMA_NEAT_GENAI_TEST_MODEL"
_VLM_MODEL_ENV = "SIMA_TEST_LLIMA_VLM_MODEL"
_ASR_MODEL_ENV = "SIMA_TEST_LLIMA_ASR_MODEL"
_LLIMA_MODELS_PATH_ENV = "LLIMA_MODELS_PATH"
_DEFAULT_LLIMA_MODELS_PATH = Path("/media/nvme/llima/models")
_DEFAULT_TEXT_MODEL = "Qwen2.5-0.5B-Instruct-GPTQ-a16w4"
_DEFAULT_VLM_MODEL = "LFM2.5-VL-450M-a16w4"
_DEFAULT_ASR_MODEL = "whisper-small-a16w8"
_VLM_PROMPT = "Describe this image in a short phrase."
_EXPECTED_VLM_TEXT = "Skier in the air."
_PROMPT = "What is the capital of Germany?"
_EXPECTED_TEXT = "The capital of Germany is Berlin."
_EXPECTED_ASR_TEXT = "tell me a joke please"
_GENAI_UNAVAILABLE = "NEAT GenAI/LLiMa support is not available in this build"


def _trim_text(text):
  return text.strip()


def _normalize_transcript(text):
  out = []
  pending_space = False
  for ch in text:
    if ch.isalnum():
      if pending_space and out:
        out.append(" ")
      out.append(ch.lower())
      pending_space = False
    elif ch.isspace() or not ch.isalnum():
      pending_space = True
  return "".join(out)


def _llima_models_path():
  value = os.environ.get(_LLIMA_MODELS_PATH_ENV, "").strip()
  return Path(value) if value else _DEFAULT_LLIMA_MODELS_PATH


def _model_name(env_name, default_name, legacy_env_name=None):
  value = os.environ.get(env_name, "").strip()
  if not value and legacy_env_name is not None:
    value = os.environ.get(legacy_env_name, "").strip()
  if not value:
    value = default_name
  if Path(value).is_absolute() or "/" in value or ".." in value:
    pytest.skip(
        f"{env_name} must be a model directory name under {_LLIMA_MODELS_PATH_ENV}, "
        f"not a path or Hugging Face repo id: {value}"
    )
  return value


def _model_dir(env_name, default_name, config_rel, label, legacy_env_name=None):
  model_dir = _llima_models_path() / _model_name(env_name, default_name, legacy_env_name)
  if not (model_dir / config_rel).is_file():
    pytest.skip(f"{model_dir} is not a {label} model directory")
  return str(model_dir)


def _text_model_dir():
  return _model_dir(
      _MODEL_ENV,
      _DEFAULT_TEXT_MODEL,
      Path("devkit/vlm_config.json"),
      "LLiMa text",
      _LEGACY_MODEL_ENV,
  )


def _vlm_model_dir():
  return _model_dir(
      _VLM_MODEL_ENV,
      _DEFAULT_VLM_MODEL,
      Path("devkit/vlm_config.json"),
      "LLiMa VLM",
  )


def _asr_model_dir():
  return _model_dir(
      _ASR_MODEL_ENV,
      _DEFAULT_ASR_MODEL,
      Path("devkit/whisper_config.json"),
      "LLiMa ASR",
  )


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


def _audio_fixture_path():
  rel = Path("tests/assets/genai/audio.wav")
  for root in _candidate_roots():
    candidate = root / rel
    if candidate.is_file():
      return candidate
  pytest.skip(f"missing audio fixture: {rel}")


def _free_local_port():
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    return sock.getsockname()[1]


def _server_url(port, path):
  return f"http://127.0.0.1:{port}{path}"


def _requests():
  return pytest.importorskip("requests")


def _wait_for_server(port):
  http = _requests()
  deadline = time.monotonic() + 30
  while time.monotonic() < deadline:
    try:
      response = http.get(_server_url(port, "/v1/models"), timeout=5)
      if response.status_code == 200:
        return
    except http.RequestException:
      time.sleep(0.1)
  raise AssertionError(f"GenAIServer did not become ready on port {port}")


def _image_data_url(path):
  encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
  return f"data:image/jpeg;base64,{encoded}"


def _json_response(response):
  http = _requests()
  try:
    response.raise_for_status()
  except http.HTTPError as exc:
    raise AssertionError(
        f"{response.request.method} {response.request.path_url} returned "
        f"HTTP {response.status_code}: {response.text}"
    ) from exc
  return response.json()


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


def _pull_public_language_outputs(run, stop_on_error=False):
  tokens = []
  done = None
  error = None
  token_samples = 0

  for _ in range(256):
    sample = run.pull("tokens", 250)
    if sample is not None:
      tokens.append(sample.to_text())
      token_samples += 1
      continue
    sample = run.pull("done", 10)
    if sample is not None:
      done = sample
      break
    sample = run.pull("error", 10)
    if sample is not None:
      error = sample.to_text()
      if stop_on_error:
        break

  return "".join(tokens), done, error, token_samples


def _pull_public_encoded(run):
  for _ in range(16):
    sample = run.pull("encoded", 60000)
    if sample is not None:
      return sample
    sample = run.pull("error", 10)
    if sample is not None:
      raise AssertionError(f"image input emitted error: {sample.to_text()}")
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
  req.audio_file = "audio.wav"
  req.language = "en"
  req.messages = [pyneat.ChatMessage()]
  req.messages[0].role = "user"
  req.messages[0].content = "hello"
  req.messages[0].tool_calls = [
      {
          "id": "call_0",
          "type": "function",
          "function": {"name": "lookup", "arguments": "{}"},
      }
  ]
  req.messages[0].tool_call_id = "call_0"
  req.messages[0].name = "lookup"
  req.images = [np.zeros((2, 2, 3), dtype=np.uint8)]
  req.use_cached_images = False
  req.tools = [
      {
          "type": "function",
          "function": {
              "name": "lookup",
              "parameters": {"type": "object"},
          },
      }
  ]
  req.tool_choice = "auto"

  assert req.prompt == "hello"
  assert req.system_prompt == "be concise"
  assert not hasattr(req, "temperature")
  assert not hasattr(req, "top_p")
  assert req.messages[0].content == "hello"
  assert req.messages[0].tool_calls[0]["function"]["name"] == "lookup"
  assert req.messages[0].tool_call_id == "call_0"
  assert req.messages[0].name == "lookup"
  assert str(req.audio_file).endswith("audio.wav")
  assert req.language == "en"
  assert len(req.images) == 1
  assert list(req.images[0].shape) == [2, 2, 3]
  assert req.images[0].image_format() == pyneat.PixelFormat.RGB
  assert req.use_cached_images is False
  assert req.tools[0]["function"]["name"] == "lookup"
  assert req.tool_choice == "auto"

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
  result.tool_calls = req.messages[0].tool_calls
  assert result.tool_calls[0]["id"] == "call_0"

  token = pyneat.TokenSample()
  token.tool_calls = req.messages[0].tool_calls
  assert token.tool_calls[0]["function"]["arguments"] == "{}"

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
  assert pyneat.genai.ASRModel is pyneat.ASRModel
  assert pyneat.genai.GenAIModel is pyneat.GenAIModel
  assert pyneat.genai.GenAIServer is pyneat.GenAIServer
  assert pyneat.genai.GenAIServerOptions is pyneat.GenAIServerOptions
  assert pyneat.genai.ImageList is pyneat.ImageList
  assert pyneat.genai.GenerationRequest is pyneat.GenerationRequest
  assert hasattr(pyneat.genai.graphs, "vision_language")
  assert hasattr(pyneat.genai.graphs, "speech_transcriber")
  assert not hasattr(pyneat.genai, "nodes")
  with pytest.raises(AttributeError, match="removed"):
    getattr(pyneat, "_graph")

  options = pyneat.genai.VisionLanguageOptions()
  options.system_prompt = "Answer exactly."
  options.max_new_tokens = 24
  options.streaming = False
  options.encode_images_on_input = False
  assert options.system_prompt == "Answer exactly."
  assert options.max_new_tokens == 24
  assert not hasattr(options, "temperature")
  assert not hasattr(options, "top_p")
  assert options.streaming is False
  assert options.encode_images_on_input is False

  speech_options = pyneat.genai.SpeechTranscriberOptions()
  speech_options.language = "en"
  speech_options.streaming = False
  assert speech_options.language == "en"
  assert speech_options.streaming is False

  server_options = pyneat.GenAIServerOptions()
  server_options.host = "127.0.0.1"
  server_options.port = 9999
  assert server_options.host == "127.0.0.1"
  assert server_options.port == 9999


def test_genai_server_constructor_is_llima_backed():
  options = pyneat.GenAIServerOptions()
  options.host = "127.0.0.1"
  options.port = 0

  try:
    server = pyneat.GenAIServer(options)
  except RuntimeError as exc:
    if _GENAI_UNAVAILABLE in str(exc):
      pytest.fail("pyneat GenAIServer is backed by unavailable LLiMa stubs")
    raise
  else:
    server.stop()


def test_genai_direct_text_generation_and_streaming():
  try:
    model = pyneat.VisionLanguageModel(_text_model_dir())
    assert not model.accepts_image()

    request = _make_request()
    result = model.run(request)
    print(f"GENAI_PY_LLM text={result.text}")
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
    print(f"GENAI_PY_LLM_STREAM text={''.join(streamed_text)}")
    assert _trim_text("".join(streamed_text)) == _EXPECTED_TEXT
    _assert_finish_reason(final_sample.finish_reason)
    assert final_sample.metrics.generated_tokens > 0

    generic = pyneat.GenAIModel(_text_model_dir())
    assert generic.task() == pyneat.GenAITask.VisionLanguage
    assert generic.accepts_text()
    assert not generic.accepts_image()
    assert not generic.accepts_audio()
    generic_result = generic.run(request)
    print(f"GENAI_PY_MODEL_LLM text={generic_result.text}")
    assert _trim_text(generic_result.text) == _EXPECTED_TEXT

    generic_streamed_text = []
    generic_final = None
    for sample in generic.stream(request):
      if sample.is_final:
        generic_final = sample
        break
      generic_streamed_text.append(sample.text)
    assert generic_final is not None
    print(f"GENAI_PY_MODEL_LLM_STREAM text={''.join(generic_streamed_text)}")
    assert _trim_text("".join(generic_streamed_text)) == _EXPECTED_TEXT
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
    print(f"GENAI_PY_VLM text={result.text}")
    assert _trim_text(result.text) == _EXPECTED_VLM_TEXT
    _assert_finish_reason(result.finish_reason)
    assert result.metrics.generated_tokens > 0

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
    print(f"GENAI_PY_VLM_EV74_TENSOR text={ev74_result.text}")
    assert _trim_text(ev74_result.text) == _EXPECTED_VLM_TEXT
    _assert_finish_reason(ev74_result.finish_reason)
    assert ev74_result.metrics.generated_tokens > 0

    streamed_text = []
    final_sample = None
    for sample in model.stream(request):
      if sample.is_final:
        final_sample = sample
        break
      streamed_text.append(sample.text)

    assert streamed_text
    assert final_sample is not None
    print(f"GENAI_PY_VLM_STREAM text={''.join(streamed_text)}")
    assert _trim_text("".join(streamed_text)) == _EXPECTED_VLM_TEXT
    _assert_finish_reason(final_sample.finish_reason)
    assert final_sample.metrics.generated_tokens > 0

    try:
      assert model.encode([image])
      assert model.cached_image_count() == 1

      cached = pyneat.GenerationRequest()
      cached.prompt = _VLM_PROMPT
      cached.use_cached_images = True
      cached.max_new_tokens = 48
      cached_result = model.run(cached)
      print(f"GENAI_PY_VLM_CACHED text={cached_result.text}")
      assert _trim_text(cached_result.text) == _EXPECTED_VLM_TEXT
      _assert_finish_reason(cached_result.finish_reason)
      assert cached_result.metrics.generated_tokens > 0
    except RuntimeError as exc:
      if "cached reuse is not supported" not in str(exc):
        raise
      print(f"GENAI_PY_VLM_CACHED skipped: {exc}")

    del model

    generic = pyneat.GenAIModel(_vlm_model_dir())
    assert generic.task() == pyneat.GenAITask.VisionLanguage
    assert generic.accepts_text()
    assert generic.accepts_image()
    assert not generic.accepts_audio()
    generic_request = pyneat.GenerationRequest()
    generic_request.prompt = _VLM_PROMPT
    generic_request.images = [image]
    generic_request.max_new_tokens = 48
    generic_result = generic.run(generic_request)
    print(f"GENAI_PY_MODEL_VLM text={generic_result.text}")
    assert _trim_text(generic_result.text) == _EXPECTED_VLM_TEXT

  except Exception as exc:
    _skip_if_dispatcher_unavailable(exc)
    raise


def test_genai_language_graph_node_generation_and_errors():
  try:
    model = pyneat.VisionLanguageModel(_text_model_dir())

    streaming_options = pyneat.genai.VisionLanguageOptions()
    streaming_options.system_prompt = "You are concise."
    streaming_options.max_new_tokens = 24
    streaming_options.streaming = True
    streaming_graph = pyneat.genai.graphs.vision_language(
        model, streaming_options, "vision_language_streaming"
    )
    streaming_run = streaming_graph.build()
    try:
      assert streaming_run.push(
          "prompt", [pyneat.make_text_sample("prompt", _PROMPT)]
      )
      text, done, error, token_samples = _pull_public_language_outputs(streaming_run)
      assert error is None
      assert done is not None
      assert token_samples > 0
      print(f"GENAI_PY_GRAPH_LLM_STREAM text={text}")
      assert _trim_text(text) == _EXPECTED_TEXT
      _assert_finish_reason(_bundle_field_text(done, "finish_reason"))
      assert int(_bundle_field_text(done, "generated_tokens")) > 0

      invalid = pyneat.Sample()
      invalid.kind = pyneat.SampleKind.Tensor
      invalid.tensor = pyneat.Tensor()
      invalid.port_name = "prompt"
      invalid.stream_label = "prompt"
      assert streaming_run.push("prompt", [invalid])
      _, _, error, _ = _pull_public_language_outputs(streaming_run, stop_on_error=True)
      assert error
    finally:
      streaming_run.stop()

    sync_options = pyneat.genai.VisionLanguageOptions()
    sync_options.system_prompt = "You are concise."
    sync_options.max_new_tokens = 24
    sync_options.streaming = False
    sync_graph = pyneat.genai.graphs.vision_language(model, sync_options, "vision_language_sync")
    sync_run = sync_graph.build()
    try:
      assert sync_run.push("prompt", [pyneat.make_text_sample("prompt", _PROMPT)])
      text, done, error, token_samples = _pull_public_language_outputs(sync_run)
      assert error is None
      assert done is not None
      assert token_samples == 1
      print(f"GENAI_PY_GRAPH_LLM_SYNC text={text}")
      assert _trim_text(text) == _EXPECTED_TEXT
    finally:
      sync_run.stop()
  except Exception as exc:
    _skip_if_dispatcher_unavailable(exc)
    raise


def test_genai_vision_language_graph_node_generation_and_errors():
  try:
    model = pyneat.VisionLanguageModel(_vlm_model_dir())
    assert model.accepts_image()
    image = _people_rgb_image()

    streaming_options = pyneat.genai.VisionLanguageOptions()
    streaming_options.max_new_tokens = 48
    streaming_options.streaming = True
    streaming_options.encode_images_on_input = False
    streaming_graph = pyneat.genai.graphs.vision_language(
        model, streaming_options, "vision_language_streaming"
    )
    streaming_run = streaming_graph.build()
    try:
      assert streaming_run.push("image", [_make_image_sample(image)])
      encoded = _pull_public_encoded(streaming_run)
      assert _bundle_field_text(encoded, "mode") == "direct"
      assert streaming_run.push(
          "prompt", [pyneat.make_text_sample("prompt", _VLM_PROMPT)]
      )
      text, done, error, token_samples = _pull_public_language_outputs(streaming_run)
      assert error is None
      assert done is not None
      assert token_samples > 0
      print(f"GENAI_PY_GRAPH_VLM_DIRECT_STREAM text={text}")
      assert _trim_text(text) == _EXPECTED_VLM_TEXT
      _assert_finish_reason(_bundle_field_text(done, "finish_reason"))
      assert int(_bundle_field_text(done, "generated_tokens")) > 0

      invalid = pyneat.make_text_sample("image", "not-an-image")
      assert streaming_run.push("image", [invalid])
      _, _, error, _ = _pull_public_language_outputs(streaming_run, stop_on_error=True)
      assert error
    finally:
      streaming_run.stop()

    sync_options = pyneat.genai.VisionLanguageOptions()
    sync_options.max_new_tokens = 48
    sync_options.streaming = False
    sync_options.encode_images_on_input = False
    sync_graph = pyneat.genai.graphs.vision_language(model, sync_options, "vision_language_sync")
    sync_run = sync_graph.build()
    try:
      assert sync_run.push("image", [_make_image_sample(image)])
      encoded = _pull_public_encoded(sync_run)
      assert _bundle_field_text(encoded, "mode") == "direct"

      assert sync_run.push(
          "use_cached_image",
          [pyneat.make_text_sample("use_cached_image", "false")],
      )
      assert sync_run.push("prompt", [pyneat.make_text_sample("prompt", _VLM_PROMPT)])
      text, done, error, token_samples = _pull_public_language_outputs(sync_run)
      assert error is None
      assert done is not None
      assert token_samples == 1
      print(f"GENAI_PY_GRAPH_VLM_SYNC text={text}")
      assert _trim_text(text) == _EXPECTED_VLM_TEXT
    finally:
      sync_run.stop()

    cached_encode_options = pyneat.genai.VisionLanguageOptions()
    cached_encode_options.max_new_tokens = 48
    cached_encode_options.streaming = True
    cached_encode_options.encode_images_on_input = True
    cached_encode_graph = pyneat.genai.graphs.vision_language(
        model, cached_encode_options, "vision_language_cached_encode"
    )
    cached_run = cached_encode_graph.build()
    try:
      assert cached_run.push("image", [_make_image_sample(image)])
      encoded = _pull_public_encoded(cached_run)
      assert _bundle_field_text(encoded, "mode") == "cached"
      assert cached_run.push("prompt", [pyneat.make_text_sample("prompt", _VLM_PROMPT)])
      _, _, error, _ = _pull_public_language_outputs(cached_run, stop_on_error=True)
      assert error
      assert "cached reuse is not supported" in error
    finally:
      cached_run.stop()

    missing_graph = pyneat.genai.graphs.vision_language(
        model, streaming_options, "vision_language_missing_image"
    )
    missing_run = missing_graph.build()
    try:
      assert missing_run.push("prompt", [pyneat.make_text_sample("prompt", _VLM_PROMPT)])
      _, _, error, _ = _pull_public_language_outputs(missing_run, stop_on_error=True)
      assert error
    finally:
      missing_run.stop()
  except Exception as exc:
    _skip_if_dispatcher_unavailable(exc)
    raise


def test_genai_direct_asr_generation_and_streaming():
  try:
    model = pyneat.ASRModel(_asr_model_dir())
    assert model.accepts_audio()

    request = pyneat.GenerationRequest()
    request.audio_file = str(_audio_fixture_path())
    request.language = "en"
    result = model.run(request)
    assert _trim_text(result.text)
    assert _normalize_transcript(result.text) == _EXPECTED_ASR_TEXT
    assert result.finish_reason == "stop"
    print(f"GENAI_PY_ASR text={result.text}")

    stream_text = ""
    saw_final = False
    for sample in model.stream(request):
      if sample.is_final:
        assert sample.finish_reason == "stop"
        saw_final = True
        break
      stream_text += sample.text
    assert saw_final
    assert _trim_text(stream_text)
    assert _normalize_transcript(stream_text) == _EXPECTED_ASR_TEXT
    print(f"GENAI_PY_ASR_STREAM text={stream_text}")

    generic = pyneat.GenAIModel(_asr_model_dir())
    assert generic.task() == pyneat.GenAITask.ASR
    assert not generic.accepts_text()
    assert not generic.accepts_image()
    assert generic.accepts_audio()
    generic_result = generic.run(request)
    assert _normalize_transcript(generic_result.text) == _EXPECTED_ASR_TEXT

    generic_stream_text = ""
    generic_saw_final = False
    for sample in generic.stream(request):
      if sample.is_final:
        assert sample.finish_reason == "stop"
        generic_saw_final = True
        break
      generic_stream_text += sample.text
    assert generic_saw_final
    assert _normalize_transcript(generic_stream_text) == _EXPECTED_ASR_TEXT
    print(f"GENAI_PY_MODEL_ASR_STREAM text={generic_stream_text}")
  except Exception as exc:
    _skip_if_dispatcher_unavailable(exc)
    raise


def test_genai_speech_transcriber_graph_node_generation_and_errors():
  try:
    model = pyneat.ASRModel(_asr_model_dir())

    options = pyneat.genai.SpeechTranscriberOptions()
    options.language = "en"
    assert options.streaming
    graph = pyneat.genai.graphs.speech_transcriber(model, options, "speech_transcriber")

    run = graph.build()
    try:
      assert run.push(
          "audio_path",
          [pyneat.make_text_sample("audio_path", str(_audio_fixture_path()))],
      )
      text, done, error, token_samples = _pull_public_language_outputs(run)
      assert error is None
      assert done is not None
      assert token_samples >= 1
      assert _trim_text(text)
      assert _normalize_transcript(text) == _EXPECTED_ASR_TEXT
      assert _bundle_field_text(done, "finish_reason") == "stop"
      assert _bundle_field_text(done, "language") == "en"
      print(f"GENAI_PY_GRAPH_ASR text={text}")

      invalid = pyneat.make_text_sample("audio", "not-audio")
      assert run.push("audio", [invalid])
      _, _, error, _ = _pull_public_language_outputs(run, stop_on_error=True)
      assert error
    finally:
      run.stop()
  except Exception as exc:
    _skip_if_dispatcher_unavailable(exc)
    raise


def test_genai_server_http_text_image_and_audio_requests():
  http = _requests()
  port = _free_local_port()
  options = pyneat.GenAIServerOptions()
  options.host = "127.0.0.1"
  options.port = port

  server = None
  try:
    server = pyneat.GenAIServer(options)
    server.add_model(_text_model_dir(), "llm")
    server.add_model(_vlm_model_dir(), "vlm")
    server.add_model(_asr_model_dir(), "asr")
    server.start()
    _wait_for_server(port)

    models = _json_response(http.get(_server_url(port, "/v1/models"), timeout=30))
    served_names = {item["id"] for item in models["data"]}
    assert {"llm", "vlm", "asr"}.issubset(served_names)

    text_body = _json_response(
        http.post(
            _server_url(port, "/v1/chat/completions"),
            json={
                "model": "llm",
                "messages": [{"role": "user", "content": _PROMPT}],
                "max_tokens": 24,
                "stream": False,
            },
            timeout=180,
        )
    )
    text = text_body["choices"][0]["message"]["content"]
    print(f"GENAI_PY_SERVER_TEXT text={text}")
    assert _trim_text(text) == _EXPECTED_TEXT

    vlm_body = _json_response(
        http.post(
            _server_url(port, "/v1/chat/completions"),
            json={
                "model": "vlm",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _VLM_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": _image_data_url(_people_image_path())},
                            },
                        ],
                    }
                ],
                "max_tokens": 48,
                "stream": False,
            },
            timeout=180,
        )
    )
    vlm_text = vlm_body["choices"][0]["message"]["content"]
    print(f"GENAI_PY_SERVER_VLM text={vlm_text}")
    assert _trim_text(vlm_text) == _EXPECTED_VLM_TEXT

    with _audio_fixture_path().open("rb") as audio:
      asr_body = _json_response(
          http.post(
              _server_url(port, "/v1/audio/transcriptions"),
              data={"model": "asr", "language": "en"},
              files={"file": (_audio_fixture_path().name, audio, "audio/wav")},
              timeout=180,
          )
      )
    asr_text = asr_body["text"]
    print(f"GENAI_PY_SERVER_ASR text={asr_text}")
    assert _normalize_transcript(asr_text) == _EXPECTED_ASR_TEXT
  except Exception as exc:
    _skip_if_dispatcher_unavailable(exc)
    raise
  finally:
    if server is not None:
      server.stop()
