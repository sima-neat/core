#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

try:
  import pyneat
except ImportError:
  sys.exit("pyneat is not installed. Follow the installation guide.")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import python_utils as tu


def main(argv: list[str]) -> int:
  try:
    import numpy as np
  except Exception:
    return tu.skip("numpy is required")

  if tu.has_flag(argv, "--help"):
    print(f"Usage: {argv[0]} [--width <w>] [--height <h>]")
    return 0

  # Why: runtime markers make intent explicit without requiring external docs.
  # Why: parity/scorecard tooling relies on stable, machine-parseable checkpoints.
  tu.step("input_contract", "parse flags and establish deterministic defaults")
  tu.step("run_mode_choice", "exercise the chapter's primary runtime path")
  tu.why("understand the contract first: inputs, run mode, and outputs")
  tu.tradeoff("prefer deterministic samples and stable contracts over production realism")
  tu.failure_mode("runtime/plugin issues should degrade to runtime_fallback without losing observability")
  tu.interpret_output("use CHECK markers plus SIGNATURE fields to validate behavior and parity")
  tu.step("output_contract", "emit checks and machine-parseable signature")
# Framework map: model session graph run output contract.
  tu.check("strict_flag_available", isinstance(tu.strict_mode(), bool),
           "strict-mode guard is observable")

  width = tu.parse_int(argv, "--width", 128)
  height = tu.parse_int(argv, "--height", 96)

  # CORE LOGIC
  arr = np.full((height, width, 3), 17, dtype=np.uint8)
  t = pyneat.Tensor.from_numpy(arr, copy=True, image_format=pyneat.PixelFormat.RGB)
  arr2 = t.to_numpy(copy=True)
  print(f"numpy roundtrip shape: {arr2.shape}")

  try:
    import torch

    torch_tensor = torch.full((height, width, 3), 9, dtype=torch.uint8)
    t2 = pyneat.Tensor.from_torch(torch_tensor, copy=True, image_format=pyneat.PixelFormat.RGB)
    torch_back = t2.to_torch(copy=True)
    print(f"torch roundtrip shape: {tuple(torch_back.shape)}")
  except Exception as exc:
    print(f"torch path skipped: {exc}")
  # END CORE LOGIC

  tu.check("tutorial_completed", True, "main path reached end without exception")
  tu.signature({
      "tutorial": "008",
      "lang": "py",
      "flow": "numpy_torch_tensor_io",
      "run_mode": "sync",
      "output_kind": "sample_or_tensor",
      "tensor_rank": -1,
      "field_count": -1,
  })

  print("[OK] 008_numpy_torch_tensor_io")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
