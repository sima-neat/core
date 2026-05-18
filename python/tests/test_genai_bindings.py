import os
from pathlib import Path

import numpy as np
import pyneat
import pytest


_MODEL_ENV = "SIMA_TEST_LLIMA_TEXT_MODEL"
_LEGACY_MODEL_ENV = "SIMA_NEAT_GENAI_TEST_MODEL"
_VLM_MODEL_ENV = "SIMA_TEST_LLIMA_VLM_MODEL"
_ASR_MODEL_ENV = "SIMA_TEST_LLIMA_ASR_MODEL"
_VLM_REPO_ID = "simaai/LFM2-VL-450M-a16w4"
_VLM_PROMPT = "Describe this image in a short phrase."
_EXPECTED_VLM_TEXT = "Skier jumping high in the air."
_PROMPT = "What is the capital of Germany?"
_EXPECTED_TEXT = "The capital of Germany is Berlin."
_EXPECTED_ASR_TEXT = "tell me a joke please"


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


def _asr_model_dir():
  value = os.environ.get(_ASR_MODEL_ENV, "").strip()
  if not value:
    pytest.skip(f"{_ASR_MODEL_ENV} is not set")

  model_dir = Path(value)
  if not (model_dir / "devkit" / "whisper_config.json").is_file():
    pytest.skip(f"{model_dir} is not a LLiMa Whisper model directory")
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


def _audio_fixture_path():
  rel = Path("tests/assets/genai/audio.wav")
  for root in _candidate_roots():
    candidate = root / rel
    if candidate.is_file():
      return candidate
  pytest.skip(f"missing audio fixture: {rel}")


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
  req.audio_file = "audio.wav"
  req.language = "en"
  req.messages = [pyneat.ChatMessage()]
  req.messages[0].role = "user"
  req.messages[0].content = "hello"
  req.images = [np.zeros((2, 2, 3), dtype=np.uint8)]
  req.use_cached_images = False

  assert req.prompt == "hello"
  assert req.system_prompt == "be concise"
  assert not hasattr(req, "temperature")
  assert not hasattr(req, "top_p")
  assert req.messages[0].content == "hello"
  assert str(req.audio_file).endswith("audio.wav")
  assert req.language == "en"
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
  assert pyneat.genai.ASRModel is pyneat.ASRModel
  assert pyneat.genai.GenAIModel is pyneat.GenAIModel
  assert pyneat.genai.ImageList is pyneat.ImageList
  assert pyneat.genai.GenerationRequest is pyneat.GenerationRequest
  assert hasattr(pyneat.genai.nodes, "vision_language")
  assert hasattr(pyneat.genai.nodes, "speech_transcriber")
  assert hasattr(pyneat.graph.nodes, "genai_vision_language")
  assert hasattr(pyneat.graph.nodes, "genai_speech_transcriber")
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
  assert not hasattr(options, "temperature")
  assert not hasattr(options, "top_p")
  assert options.streaming is False
  assert options.encode_images_on_input is False

  speech_options = pyneat.genai.nodes.SpeechTranscriberOptions()
  speech_options.language = "en"
  speech_options.streaming = False
  assert speech_options.language == "en"
  assert speech_options.streaming is False


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

    generic = pyneat.GenAIModel(_text_model_dir())
    assert generic.task() == pyneat.GenAITask.VisionLanguage
    assert generic.accepts_text()
    assert not generic.accepts_image()
    assert not generic.accepts_audio()
    generic_result = generic.run(request)
    assert _trim_text(generic_result.text) == _EXPECTED_TEXT

    generic_streamed_text = []
    generic_final = None
    for sample in generic.stream(request):
      if sample.is_final:
        generic_final = sample
        break
      generic_streamed_text.append(sample.text)
    assert generic_final is not None
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
    assert _trim_text(generic_result.text) == _EXPECTED_VLM_TEXT
    print(f"GENAI_PY_MODEL_VLM text={generic_result.text}")

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
    graph = pyneat.graph.Graph()
    audio_path_port = graph.intern_port("audio_path")
    audio_port = graph.intern_port("audio")

    options = pyneat.genai.nodes.SpeechTranscriberOptions()
    options.language = "en"
    assert options.streaming
    node = graph.add(
        pyneat.genai.nodes.speech_transcriber(model, options, "speech_transcriber")
    )

    run = pyneat.graph.GraphSession(graph).build()
    try:
      assert run.push_port(
          node,
          audio_path_port,
          pyneat.make_text_sample("audio_path", str(_audio_fixture_path())),
      )
      text, done, error, token_samples = _pull_language_outputs(run, node)
      assert error is None
      assert done is not None
      assert token_samples >= 1
      assert _trim_text(text)
      assert _normalize_transcript(text) == _EXPECTED_ASR_TEXT
      assert _bundle_field_text(done, "finish_reason") == "stop"
      assert _bundle_field_text(done, "language") == "en"
      print(f"GENAI_PY_GRAPH_ASR text={text}")

      invalid = pyneat.make_text_sample("audio", "not-audio")
      assert run.push_port(node, audio_port, invalid)
      _, _, error, _ = _pull_language_outputs(run, node, stop_on_error=True)
      assert error
    finally:
      run.stop()
  except Exception as exc:
    _skip_if_dispatcher_unavailable(exc)
    raise
