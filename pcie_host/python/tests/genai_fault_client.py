#!/usr/bin/env python3
"""Connected-card GenAI client used by the Windows fault-injection harness."""

import argparse
import os
import time
from pathlib import Path

from pyneatpcie import genai
from genai_admission_contract import is_injected_transport_failure


def connection():
  result = genai.ConnectionOptions()
  result.card_host = os.environ["SIMAPCIE_CARD_HOST"]
  result.card_id = int(os.environ.get("SIMAPCIE_CARD_ID", "0"))
  result.user = os.environ.get("SIMAPCIE_USER", "sima")
  result.ssh_key = os.environ.get("SIMAPCIE_SSH_KEY", "")
  result.startup_timeout_ms = int(os.environ.get("SIMAPCIE_READINESS_TIMEOUT_MS", "180000"))
  result.request_timeout_ms = int(os.environ.get("SIMAPCIE_GENAI_REQUEST_TIMEOUT_MS", "120000"))
  if os.environ.get("SIMAPCIE_GENAI_MEDIA_DIRECTORY"):
    result.media_directory = os.environ["SIMAPCIE_GENAI_MEDIA_DIRECTORY"]
  if os.environ.get("SIMAPCIE_GENAI_MEDIA_SERVE_ROOT"):
    result.media_serve_root = os.environ["SIMAPCIE_GENAI_MEDIA_SERVE_ROOT"]
  return result


def model_path():
  root = Path(os.environ["SIMAPCIE_GENAI_MODELS_PATH"])
  name = os.environ.get(
      "SIMA_TEST_LLIMA_TEXT_MODEL", "Qwen2.5-0.5B-Instruct-Autoround-a16w4")
  path = root / name
  if not (path / "devkit" / "vlm_config.json").is_file():
    raise RuntimeError(f"Missing release GenAI model: {path}")
  return str(path)


def request(prompt, max_new_tokens):
  value = genai.GenerationRequest()
  value.prompt = prompt
  value.max_new_tokens = max_new_tokens
  return value


FAULT_WINDOW_SECONDS = 120


def wait_for_injected_failure():
  model = genai.GenAIModel(model_path(), connection())
  try:
    # A short answer can finish before the injected fault lands, so keep a
    # stream active until the fault is observed or the window closes.
    deadline = time.monotonic() + FAULT_WINDOW_SECONDS
    ready = False
    while time.monotonic() < deadline:
      stream = model.stream(request("Count from one to one thousand.", 2048))
      try:
        first = stream.next()
        if first is not None and not first.is_final:
          if not ready:
            print("GENAI_FAULT_READY", flush=True)
            ready = True
          while stream.next() is not None:
            pass
      except RuntimeError as error:
        if not ready or not is_injected_transport_failure(error):
          raise RuntimeError(
              f"GenAI fault client observed an unrelated failure: {error}") from error
        print(f"GENAI_FAULT_OBSERVED: {error}", flush=True)
        return
      if not ready:
        raise RuntimeError("Fault client did not establish an active GenAI stream")
    raise RuntimeError("GenAI stream completed instead of observing the injected fault")
  finally:
    model.close()


def smoke():
  model = genai.GenAIModel(model_path(), connection())
  try:
    result = model.run(request("Reply with the word ready.", 32))
    if not result.text.strip():
      raise RuntimeError("GenAI recovery smoke test produced no text")
    print("GENAI_RECOVERY_PASS", flush=True)
  finally:
    model.close()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("mode", choices=("wait-for-failure", "smoke"))
  args = parser.parse_args()
  if args.mode == "wait-for-failure":
    wait_for_injected_failure()
  else:
    smoke()


if __name__ == "__main__":
  main()
