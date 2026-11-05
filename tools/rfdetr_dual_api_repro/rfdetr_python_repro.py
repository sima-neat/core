#!/usr/bin/env python3
"""RF-DETR pyneat repro: load model with A65 processcvu targets and run random tensor inputs."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pyneat


def main() -> int:
  model_path = Path(
      os.environ.get(
          "RFDETR_MODEL_PATH",
          "/mnt/nfs/sima-neat/tmp/drive_model/rfdetr_576_simplified_transformer_after_gather_base_mpk.tar.gz",
      )
  )
  timeout_ms = int(os.environ.get("RFDETR_TIMEOUT_MS", "60000"))
  run_target = os.environ.get("RFDETR_PROCESSCVU_TARGET", "A65")

  print(f"[python] model={model_path} exists={model_path.exists()}", flush=True)
  if model_path.exists():
    print(f"[python] model_size={model_path.stat().st_size}", flush=True)
  print(f"[python] processcvu pre/post target={run_target}", flush=True)

  opt = pyneat.ModelOptions()
  opt.preprocess.kind = pyneat.InputKind.Tensor
  opt.preprocess.enable = pyneat.AutoFlag.Off
  opt.processcvu.pre_run_target = run_target
  opt.processcvu.post_run_target = run_target

  try:
    model = pyneat.Model(str(model_path), opt)
    print("[python] Model loaded", flush=True)
  except Exception as exc:  # noqa: BLE001 - repro should report exact failure.
    print(f"[python] Model load failed: {type(exc).__name__}: {exc}", flush=True)
    return 2

  try:
    print(f"[python] input_specs={model.input_specs()}", flush=True)
  except Exception as exc:  # noqa: BLE001
    print(f"[python] input_specs failed: {type(exc).__name__}: {exc}", flush=True)

  inputs = [
      np.random.random((1, 36, 36, 256)).astype(np.float32),
      np.random.random((1, 300, 4)).astype(np.float32),
  ]
  print(f"[python] running inputs={[(x.shape, str(x.dtype)) for x in inputs]}", flush=True)
  try:
    outputs = model.run(inputs, timeout_ms=timeout_ms)
  except Exception as exc:  # noqa: BLE001
    print(f"[python] run failed: {type(exc).__name__}: {exc}", flush=True)
    return 3

  print(f"[python] run ok type={type(outputs)}", flush=True)
  if isinstance(outputs, list):
    print(f"[python] output_count={len(outputs)}", flush=True)
    for i, output in enumerate(outputs):
      print(f"[python] output[{i}]={output}", flush=True)
      if hasattr(output, "shape"):
        print(f"[python] output[{i}].shape={list(output.shape)} dtype={output.dtype}", flush=True)
  else:
    print(f"[python] output={outputs}", flush=True)
  return 0


if __name__ == "__main__":
  sys.exit(main())
