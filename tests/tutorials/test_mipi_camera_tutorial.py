"""Hardware-gated smoke tests for the MIPI camera tutorial.

These tests are skipped by default. Enable them on a Modalix DevKit with:

  SIMA_NEAT_TEST_MIPI_CAMERA=1
  SIMA_NEAT_TUTORIAL_MODEL_MIPI=/path/to/model.tar.gz
  SIMA_NEAT_TUTORIAL_MIPI_DECODE=yolov9seg   # optional BoxDecode pass
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TUTORIAL = (
    REPO_ROOT
    / "tutorials"
    / "023_run_mipi_camera_model"
    / "run_mipi_camera_model.py"
)

ENABLED = os.environ.get("SIMA_NEAT_TEST_MIPI_CAMERA") == "1"
MODEL = os.environ.get("SIMA_NEAT_TUTORIAL_MODEL_MIPI")
BOXDECODE = os.environ.get("SIMA_NEAT_TUTORIAL_MIPI_DECODE", "").strip()
TIMEOUT_SEC = int(os.environ.get("SIMA_NEAT_MIPI_CAMERA_TEST_TIMEOUT_SEC", "120"))
PULL_TIMEOUT_MS = os.environ.get("SIMA_NEAT_MIPI_CAMERA_PULL_TIMEOUT_MS", "15000")


pytestmark = pytest.mark.skipif(
    not ENABLED,
    reason="set SIMA_NEAT_TEST_MIPI_CAMERA=1 to run MIPI camera hardware tests",
)


def _require_pyneat():
  if importlib.util.find_spec("pyneat") is None:
    pytest.skip("pyneat is not importable")
  import pyneat  # pylint: disable=import-outside-toplevel

  return pyneat


def test_installed_pyneat_exposes_camera_input() -> None:
  pyneat = _require_pyneat()
  assert hasattr(pyneat, "CameraInputOptions")
  assert hasattr(pyneat.nodes, "camera_input")

  opt = pyneat.CameraInputOptions()
  opt.allow_cpu_fallback = True
  node = pyneat.nodes.camera_input(opt)
  assert node.kind() == "CameraInput"
  assert node.input_role() == pyneat.InputRole.Source


def test_camera_input_output_smoke() -> None:
  pyneat = _require_pyneat()

  camera = pyneat.CameraInputOptions()
  camera.width = int(os.environ.get("SIMA_NEAT_MIPI_CAMERA_WIDTH", "1920"))
  camera.height = int(os.environ.get("SIMA_NEAT_MIPI_CAMERA_HEIGHT", "1080"))
  camera.framerate_num = int(os.environ.get("SIMA_NEAT_MIPI_CAMERA_FPS", "30"))
  camera.framerate_den = 1
  camera.format = os.environ.get("SIMA_NEAT_MIPI_CAMERA_FORMAT", "NV12")
  camera.buffer_name = "camera0"
  camera.allow_cpu_fallback = True

  graph = pyneat.Graph("mipi_camera_output_smoke")
  graph.add(pyneat.nodes.camera_input(camera))
  graph.add(pyneat.nodes.output("frames"))

  run = graph.build()
  sample = run.pull(timeout_ms=int(PULL_TIMEOUT_MS))
  assert sample is not None, f"camera output timed out; last_error={run.last_error()}"
  tensors = list(sample.tensors)
  assert tensors, "camera output sample did not contain tensors"


def _run_tutorial(decode: str) -> subprocess.CompletedProcess[str]:
  if not MODEL:
    pytest.skip("set SIMA_NEAT_TUTORIAL_MODEL_MIPI to run the camera model tutorial")
  env = os.environ.copy()
  for name in (
      "PYTHONPATH",
      "LD_LIBRARY_PATH",
      "GST_PLUGIN_PATH",
      "GST_PLUGIN_PATH_1_0",
      "GST_PLUGIN_SYSTEM_PATH",
      "GST_PLUGIN_SYSTEM_PATH_1_0",
  ):
    env.pop(name, None)
  cmd = [
      sys.executable,
      str(TUTORIAL),
      "--model",
      MODEL,
      "--frames",
      "2",
      "--decode",
      decode,
      "--pull-timeout-ms",
      PULL_TIMEOUT_MS,
  ]
  return subprocess.run(
      cmd,
      cwd=str(REPO_ROOT),
      env=env,
      capture_output=True,
      text=True,
      timeout=TIMEOUT_SEC,
  )


def test_mipi_camera_model_raw_outputs() -> None:
  result = _run_tutorial("none")
  assert result.returncode == 0, (
      f"raw MIPI model tutorial failed with {result.returncode}\n"
      f"--- STDOUT ---\n{result.stdout}\n"
      f"--- STDERR ---\n{result.stderr}"
  )
  assert "output_timeout" not in result.stdout


def test_mipi_camera_model_boxdecode_outputs() -> None:
  if not BOXDECODE:
    pytest.skip("set SIMA_NEAT_TUTORIAL_MIPI_DECODE to run the BoxDecode path")
  result = _run_tutorial(BOXDECODE)
  assert result.returncode == 0, (
      f"BoxDecode MIPI model tutorial failed with {result.returncode}\n"
      f"--- STDOUT ---\n{result.stdout}\n"
      f"--- STDERR ---\n{result.stderr}"
  )
  assert "output_timeout" not in result.stdout
