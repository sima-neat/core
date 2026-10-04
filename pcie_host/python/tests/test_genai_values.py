"""Value binding checks; no card or daemon is opened."""
import numpy as np
from pyneatpcie import genai


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
