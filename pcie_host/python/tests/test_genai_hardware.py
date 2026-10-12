"""Real-model PCIe tests mirroring Core's direct GenAI tests, without image caching.

Run prepare_pcie_genai_models.sh first. No card access occurs unless
SIMAPCIE_CARD_HOST is set; once enabled, missing models/assets are failures.
"""
import os
import re
from pathlib import Path

import numpy as np
import pytest
from pyneatpcie import genai


@pytest.fixture
def connection():
  if not os.environ.get("SIMAPCIE_CARD_HOST"):
    pytest.skip("SIMAPCIE_CARD_HOST is not set")
  result = genai.ConnectionOptions()
  result.card_host = os.environ["SIMAPCIE_CARD_HOST"]
  result.card_id = int(os.environ.get("SIMAPCIE_CARD_ID", "0"))
  result.user = os.environ.get("SIMAPCIE_USER", "sima")
  result.ssh_key = os.environ.get("SIMAPCIE_SSH_KEY", "")
  result.card_receive_directory = os.environ.get(
      "SIMAPCIE_GENAI_RECEIVE_ROOT", str(result.card_receive_directory))
  result.startup_timeout_ms = int(os.environ.get("SIMAPCIE_READINESS_TIMEOUT_MS", "180000"))
  result.request_timeout_ms = int(os.environ.get("SIMAPCIE_GENAI_REQUEST_TIMEOUT_MS", "120000"))
  return result


def _model(kind):
  defaults = {
      "llm": ("TEXT", "Qwen2.5-0.5B-Instruct-Autoround-a16w4"),
      "vlm": ("VLM", "LFM2.5-VL-450M-Autoround-a16w4"),
      "asr": ("ASR", "whisper-small-a16w8"),
  }
  variable, name = defaults[kind]
  root = Path(os.environ.get("SIMAPCIE_GENAI_MODELS_PATH", Path.home() / "workspace/models_genai"))
  path = root / os.environ.get(f"SIMA_TEST_LLIMA_{variable}_MODEL", name)
  config = "whisper_config.json" if kind == "asr" else "vlm_config.json"
  assert (path / "devkit" / config).is_file(), f"Missing {path}; run prepare_pcie_genai_models.sh"
  return str(path)


def _asset(name):
  root = os.environ.get("SIMAPCIE_GENAI_TEST_ASSETS")
  if root:
    path = Path(root) / name
  else:
    tests = Path(__file__).resolve().parents[3] / "tests"
    path = tests / ("images" if name == "people.jpg" else "assets/genai") / name
  assert path.is_file(), f"Missing GenAI fixture: {path}"
  return path


def _normalized(text):
  return re.sub(r"[^\w]+", " ", text.lower(), flags=re.UNICODE).strip()


def _consume(stream):
  text = []
  final = None
  for sample in stream:
    assert final is None, "Output after terminal sample"
    text.append(sample.text)
    if sample.is_final:
      final = sample
  assert final is not None, "Missing terminal sample"
  assert stream.next() is None, "Stream did not remain exhausted"
  return "".join(text), final


def _generated(result, text=None):
  text = result.text if text is None else text
  print(f"text={text} finish={result.finish_reason}")
  assert _normalized(text)
  assert result.metrics.generated_tokens > 0
  assert result.finish_reason in {"stop", "interrupted"}


def _asr_metadata(result, language):
  assert result.finish_reason == "stop"
  assert result.language == language
  assert result.no_speech_prob is not None and np.isfinite(result.no_speech_prob)
  assert 0 <= result.no_speech_prob <= 1
  assert result.avg_logprob is not None and np.isfinite(result.avg_logprob)


def test_genai_llm_run_stream_history_and_cancel(connection):
  expected = "The capital of Germany is Berlin."
  with genai.GenAIModel(_model("llm"), connection) as model:
    assert model.task() == genai.GenAITask.VisionLanguage
    assert model.accepts_text() and not model.accepts_image() and not model.accepts_audio()
    request = genai.GenerationRequest()
    request.system_prompt = "You are concise."
    request.prompt = "What is the capital of Germany?"
    request.max_new_tokens = 24
    thinking = genai.GenerationRequest()
    thinking.prompt = request.prompt
    thinking.enable_thinking = True
    thinking.max_new_tokens = 24
    with pytest.raises((RuntimeError, ValueError)):
      model.run(thinking)
    with pytest.raises((RuntimeError, ValueError)):
      _consume(model.stream(thinking))
    result = model.run(request)
    _generated(result)
    assert result.text.strip() == expected
    text, final = _consume(model.stream(request))
    _generated(final, text)
    assert text.strip() == expected

    history = genai.GenerationRequest()
    messages = []
    for role, content in [("system", "You are concise."),
                          ("user", "What is the capital of Germany?"),
                          ("assistant", expected),
                          ("user", "Repeat your previous answer exactly.")]:
      message = genai.ChatMessage()
      message.role, message.content = role, content
      messages.append(message)
    history.messages = messages
    history.max_new_tokens = 24
    history_result = model.run(history)
    _generated(history_result)
    assert "berlin" in history_result.text.lower()

    tools = genai.GenerationRequest()
    message = genai.ChatMessage()
    message.role = "user"
    message.content = "Use the available tool to set coolant flow to 80 percent for machine CNC-01."
    tools.messages = [message]
    tools.max_new_tokens = 128
    tools.tools = [{"type": "function", "function": {
        "name": "set_coolant_flow", "description": "Set the coolant flow percentage for a CNC machine.",
        "parameters": {"type": "object", "properties": {
            "machine_id": {"type": "string"}, "flow_percentage": {"type": "integer"}},
            "required": ["machine_id", "flow_percentage"]}}}]
    result = model.run(tools)
    print(f"tool_calls={result.tool_calls} text={result.text}")
    bad_audio = genai.GenerationRequest()
    bad_audio.audio_file = "audio.wav"
    with pytest.raises((RuntimeError, ValueError)):
      model.run(bad_audio)

    active_request = genai.GenerationRequest()
    active_request.prompt = "Count from one to one hundred."
    active_request.max_new_tokens = 256
    for _ in range(3):
      stream = model.stream(active_request)
      first = stream.next()
      assert first is not None and not first.is_final
      stream.cancel()
      _, final = _consume(stream)
      assert final.finish_reason == "interrupted"
    assert model.run(request).text.strip() == expected
    model.close()
    model.close()


def test_genai_vlm_run_and_stream(connection):
  import cv2

  with genai.GenAIModel(_model("vlm"), connection) as model:
    assert model.task() == genai.GenAITask.VisionLanguage
    assert model.accepts_text() and model.accepts_image() and not model.accepts_audio()
    image = cv2.imread(str(_asset("people.jpg")), cv2.IMREAD_COLOR)
    assert image is not None
    request = genai.GenerationRequest()
    request.prompt = "Describe this image in a short phrase."
    request.images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB)]
    request.max_new_tokens = 48
    _generated(model.run(request))
    text, final = _consume(model.stream(request))
    _generated(final, text)
    _generated(model.run(request))
    model.close()
    model.close()


def test_genai_asr_file_pcm_translation_and_stream(connection):
  def check(result, expected, language="en", text=None):
    text = result.text if text is None else text
    print(f"ASR text={text}")
    assert _normalized(text) == expected
    _asr_metadata(result, language)

  with genai.GenAIModel(_model("asr"), connection) as model:
    assert model.task() == genai.GenAITask.ASR
    assert model.accepts_audio() and not model.accepts_text() and not model.accepts_image()
    request = genai.GenerationRequest()
    request.audio_file = str(_asset("audio.wav"))
    check(model.run(request), "tell me a joke please")
    german = genai.GenerationRequest()
    german.audio_file = str(_asset("audio_de.wav"))
    check(model.run(german), "erzähl mir bitte einen witz", "de")
    german.asr_task = genai.ASRTask.Translate
    check(model.run(german), "please tell me a joke", "de")
    pcm = genai.GenerationRequest()
    samples = np.fromfile(_asset("audio_16k_mono_f32le.raw"), dtype="<f4")
    assert samples.size > 0
    pcm.audio = np.ascontiguousarray(samples, dtype=np.float32)
    pcm.sample_rate = 16000
    pcm.language = "english"
    check(model.run(pcm), "tell me a joke please")
    text, final = _consume(model.stream(request))
    check(final, "tell me a joke please", text=text)
    check(model.run(request), "tell me a joke please")
    bad_text = genai.GenerationRequest()
    bad_text.prompt = "What is this audio?"
    with pytest.raises((RuntimeError, ValueError)):
      model.run(bad_text)
    bad_image = genai.GenerationRequest()
    bad_image.images = [np.zeros((1, 1, 3), dtype=np.uint8)]
    with pytest.raises((RuntimeError, ValueError)):
      model.run(bad_image)
    model.close()
    model.close()
