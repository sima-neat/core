import os
from pathlib import Path

import pyneat
import pytest


_MODEL_ENV = "SIMA_TEST_LLIMA_TEXT_MODEL"
_LEGACY_MODEL_ENV = "SIMA_NEAT_GENAI_TEST_MODEL"
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

  assert req.prompt == "hello"
  assert req.system_prompt == "be concise"
  assert req.messages[0].content == "hello"

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
  assert pyneat.genai.GenerationRequest is pyneat.GenerationRequest
  assert hasattr(pyneat.genai.nodes, "language")
  assert hasattr(pyneat.graph.nodes, "genai_language")

  options = pyneat.genai.nodes.LanguageOptions()
  options.system_prompt = "Answer exactly."
  options.max_new_tokens = 24
  options.streaming = False
  assert options.system_prompt == "Answer exactly."
  assert options.max_new_tokens == 24
  assert options.streaming is False


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


def test_genai_language_graph_node_generation_and_errors():
  try:
    model = pyneat.VisionLanguageModel(_text_model_dir())

    graph = pyneat.graph.Graph()
    prompt_port = graph.intern_port("prompt")
    formatted_prompt_port = graph.intern_port("formatted_prompt")

    streaming_options = pyneat.genai.nodes.LanguageOptions()
    streaming_options.system_prompt = "You are concise."
    streaming_options.max_new_tokens = 24
    streaming_options.streaming = True
    streaming_node = graph.add(
        pyneat.genai.nodes.language(model, streaming_options, "language_streaming")
    )

    sync_options = pyneat.genai.nodes.LanguageOptions()
    sync_options.system_prompt = "You are concise."
    sync_options.max_new_tokens = 24
    sync_options.streaming = False
    sync_node = graph.add(pyneat.graph.nodes.genai_language(model, sync_options, "language_sync"))

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

      assert run.push_port(
          streaming_node,
          formatted_prompt_port,
          pyneat.make_text_sample("formatted_prompt", "The capital of Germany is"),
      )
      text, done, error, _ = _pull_language_outputs(run, streaming_node)
      assert error is None
      assert done is not None
      assert _trim_text(text)

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
