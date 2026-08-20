"""Model-backed regression for the wide-CBlock LightGlue DetessDequant route."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

KEYPOINT_COUNT = 600
DESCRIPTOR_DIM = 256
POSITION_DIM = 32
OUTPUT_EXTENT = KEYPOINT_COUNT + 1
MODEL_ENV = "SIMA_TEST_LIGHTGLUE_MODEL"
METRICS_PREFIX = "LIGHTGLUE_METRICS="


def _model_path() -> Path:
  value = os.environ.get(MODEL_ENV, "")
  if not value:
    pytest.skip(f"{MODEL_ENV} is not set")
  path = Path(value)
  if not path.is_file():
    raise AssertionError(f"{MODEL_ENV} does not name a file: {path}")
  return path


def _deterministic_identical_inputs() -> list[np.ndarray]:
  rng = np.random.default_rng(42)
  angles = rng.uniform(-np.pi, np.pi, size=(KEYPOINT_COUNT, POSITION_DIM)).astype(
      np.float32
  )
  positions = np.stack((np.cos(angles), np.sin(angles))).astype(np.float32)

  descriptors = rng.standard_normal((KEYPOINT_COUNT, DESCRIPTOR_DIM)).astype(
      np.float32
  )
  descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True) + np.float32(
      1e-8
  )
  descriptors = descriptors.reshape(1, 1, KEYPOINT_COUNT, DESCRIPTOR_DIM)
  return [positions, descriptors, positions.copy(), descriptors.copy()]


def _tensor(value: np.ndarray):
  import pyneat

  contiguous = np.array(value, dtype=np.float32, copy=True, order="C")
  return pyneat.Tensor.from_numpy(contiguous, copy=True)


def _run_worker(model_path: Path, target: str, output_path: Path) -> dict[str, object]:
  import pyneat

  options = pyneat.ModelOptions()
  options.preprocess.kind = pyneat.InputKind.Tensor
  if target != "AUTO":
    options.processcvu.post_run_target = target

  model = pyneat.Model(str(model_path), options)
  input_shapes = [list(spec.shape) for spec in model.input_specs()]
  expected_inputs = [
      [2, KEYPOINT_COUNT, POSITION_DIM],
      [1, KEYPOINT_COUNT, DESCRIPTOR_DIM],
      [2, KEYPOINT_COUNT, POSITION_DIM],
      [1, KEYPOINT_COUNT, DESCRIPTOR_DIM],
  ]
  if input_shapes != expected_inputs:
    raise AssertionError(
        f"unexpected LightGlue input contract: {input_shapes}, expected {expected_inputs}"
    )

  outputs = model.run(
      [_tensor(value) for value in _deterministic_identical_inputs()],
      timeout_ms=30000,
  )
  if len(outputs) != 1:
    raise AssertionError(f"LightGlue returned {len(outputs)} outputs, expected one")

  scores = np.array(outputs[0].to_numpy(), dtype=np.float32, copy=True, order="C")
  expected_shape = (1, OUTPUT_EXTENT, OUTPUT_EXTENT)
  if scores.shape != expected_shape:
    raise AssertionError(
        f"unexpected LightGlue output shape: {scores.shape}, expected {expected_shape}"
    )
  if not np.isfinite(scores).all():
    raise AssertionError("LightGlue output contains non-finite values")

  np.save(output_path, scores, allow_pickle=False)
  body = scores[0, :KEYPOINT_COUNT, :KEYPOINT_COUNT]
  diagonal = np.diag(body)
  off_diagonal = body[~np.eye(KEYPOINT_COUNT, dtype=bool)]
  row_best = np.argmax(body, axis=1)
  column_best = np.argmax(body, axis=0)
  indices = np.arange(KEYPOINT_COUNT)
  identity_mutual = (row_best == indices) & (column_best == indices)
  return {
      "target": target,
      "diagonal_margin": float(diagonal.mean() - off_diagonal.mean()),
      "diagonal_row_maxima": int(np.count_nonzero(row_best == indices)),
      "identity_mutual_matches": int(np.count_nonzero(identity_mutual)),
      "minimum": float(scores.min()),
      "maximum": float(scores.max()),
  }


def _worker_main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", type=Path, required=True)
  parser.add_argument("--target", choices=("AUTO", "EV74"), required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  metrics = _run_worker(args.model, args.target, args.output)
  print(f"{METRICS_PREFIX}{json.dumps(metrics, sort_keys=True)}")
  return 0


def _run_isolated(model_path: Path, target: str, output_path: Path) -> dict[str, object]:
  worker_env = os.environ.copy()
  extract_root = None
  if "SIMA_MPK_EXTRACT_ROOT" not in worker_env:
    extract_root = model_path.parent / f".lightglue-extract-{target.lower()}"
    extract_root.mkdir(parents=True, exist_ok=True)
    worker_env["SIMA_MPK_EXTRACT_ROOT"] = str(extract_root)
  command = [
      sys.executable,
      str(Path(__file__).resolve()),
      "--model",
      str(model_path),
      "--target",
      target,
      "--output",
      str(output_path),
  ]
  try:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        env=worker_env,
        text=True,
        timeout=120,
    )
  finally:
    if extract_root is not None:
      shutil.rmtree(extract_root, ignore_errors=True)
  if completed.returncode != 0:
    raise AssertionError(
        f"LightGlue {target} worker failed with {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
  lines = [
      line.removeprefix(METRICS_PREFIX)
      for line in completed.stdout.splitlines()
      if line.startswith(METRICS_PREFIX)
  ]
  if len(lines) != 1:
    raise AssertionError(
        f"LightGlue {target} worker did not emit one metrics record:\n"
        f"{completed.stdout}"
    )
  return json.loads(lines[0])


def test_lightglue_model_wide_cblock_matches_on_auto_and_ev74(tmp_path):
  if platform.machine().lower() not in {"aarch64", "arm64"}:
    pytest.skip("LightGlue MPK regression requires a Modalix AArch64 target")

  model_path = _model_path()
  outputs = {}
  metrics = {}
  for target in ("AUTO", "EV74"):
    output_path = tmp_path / f"lightglue-{target.lower()}.npy"
    metrics[target] = _run_isolated(model_path, target, output_path)
    outputs[target] = np.load(output_path, allow_pickle=False)

  for target in ("AUTO", "EV74"):
    result = metrics[target]
    assert result["diagonal_margin"] >= 1.0, result
    assert result["diagonal_row_maxima"] >= 550, result
    assert result["identity_mutual_matches"] >= 550, result
    assert result["minimum"] < result["maximum"], result

  assert np.array_equal(outputs["AUTO"], outputs["EV74"]), (
      "AUTO/A65 and explicit EV74 produced different LightGlue score matrices"
  )


if __name__ == "__main__":
  raise SystemExit(_worker_main())
