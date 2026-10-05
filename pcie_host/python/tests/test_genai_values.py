"""Value binding checks; no card or daemon is opened."""
import numpy as np
import pytest
from pyneatpcie import genai


def test_connection_address_override():
  connection = genai.ConnectionOptions()
  assert connection.card_id == 0
  assert connection.card_host == ""
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
