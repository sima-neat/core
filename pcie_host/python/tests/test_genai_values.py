"""Value binding checks; no card or daemon is opened."""
import numpy as np
import pytest
import json
from pyneatpcie import genai


def test_connection_address_override():
  connection = genai.ConnectionOptions()
  assert connection.card_id == 0
  assert connection.card_host == ""
  assert connection.user == "sima"
  connection.user = "root"
  assert connection.user == "root"
  connection.card_id = 1
  assert connection.card_host == ""
  connection.card_host = "192.168.1.42"
  assert connection.card_host == "192.168.1.42"


def test_explicit_request_values():
  request = genai.GenerationRequest()
  request.prompt = "Describe this image"
  request.images = [np.zeros((2, 3, 3), dtype=np.uint8)]
  request.max_new_tokens = 32
  assert request.prompt == "Describe this image"
  assert request.max_new_tokens == 32
  assert len(request.images) == 1


def test_in_memory_vlm_images():
  image = np.zeros((2, 3, 3), dtype=np.uint8)
  for value in (genai.GenerationRequest(), genai.ChatMessage()):
    value.images = [image]
    assert len(value.images) == 1
    assert value.images[0].shape == [2, 3, 3]
    assert not hasattr(value, "image_files")
    with pytest.raises(AttributeError):
      value.image_files = ["image.jpg"]


def test_asr_and_tools():
  request = genai.GenerationRequest()
  request.audio = np.zeros(160, dtype=np.float32)
  request.sample_rate = 16000
  request.asr_task = genai.ASRTask.Transcribe
  request.language = "en"
  request.tools = [{"type": "function", "function": {"name": "lookup"}}]
  assert request.tools[0]["function"]["name"] == "lookup"
  assert request.audio is not None


@pytest.mark.parametrize("model_path", ["missing-model", "/nonexistent/neat-model"])
def test_host_model_path_validation(tmp_path, monkeypatch, model_path):
  monkeypatch.chdir(tmp_path)
  connection = genai.ConnectionOptions()
  # Reject before opening the PCIe service or launching a card worker.
  with pytest.raises(ValueError, match="existing host directory"):
    genai.GenAIModel(model_path, connection)


@pytest.mark.parametrize("case,expected", [
    ("missing_draft", "one target and one draft"),
    ("missing_target", "missing target model"),
    ("single_target", "pass its parent directory"),
])
def test_speculative_pair_validation(tmp_path, case, expected):
  for name, is_draft in [("target", False), ("draft", True)]:
    if case == "missing_draft" and is_draft:
      continue
    if case == "missing_target" and not is_draft:
      continue
    root = tmp_path / name
    (root / "devkit").mkdir(parents=True)
    (root / "elf_files").mkdir()
    config = {"lm_cfg": {"speculative_decoding_cfg": {"is_draft": is_draft}}}
    (root / "devkit/vlm_config.json").write_text(json.dumps(config))
  path = tmp_path / "target" if case == "single_target" else tmp_path
  # Invalid pairs must fail locally, before opening transport or launching a worker.
  with pytest.raises(RuntimeError, match=expected):
    genai.GenAIModel(str(path))
