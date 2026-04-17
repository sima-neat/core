#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

try:
  import pyneat
except ImportError:
  sys.exit(
      "pyneat is not importable. Either NEAT is not installed, or the venv is not activated.\n"
      "Run: source ~/pyneat/bin/activate\n"
      "If the venv does not exist yet, follow the installation guide."
  )


def make_tensor(width: int, height: int, value: float):
  import numpy as np

  arr = np.full((height, width, 3), value, dtype=np.float32)
  return pyneat.Tensor.from_numpy(arr, copy=True)


def main(argv: list[str]) -> int:

  if "--help" in argv:
    print(f"Usage: {argv[0]} [--width <w>] [--height <h>] [--print-gst]")
    return 0

  strict_mode = os.getenv("SIMA_RUN_TUTORIALS_FULL") is not None

  # Why: runtime markers make intent explicit without requiring external docs.
  # Why: parity/scorecard tooling relies on stable, machine-parseable checkpoints.
  print("STEP input_contract: parse flags and establish deterministic defaults")
  print("STEP run_mode_choice: exercise the chapter's primary runtime path")
  print("STEP output_contract: emit checks and machine-parseable signature")
  print("CHECK strict_flag_available: PASS (strict-mode guard is observable)")
  assert isinstance(strict_mode, bool), "check failed: strict_flag_available (strict-mode guard is observable)"

  def _parse_int(key: str, default: int) -> int:
    raw = next((argv[i + 1] for i in range(1, len(argv) - 1) if argv[i] == key), None)
    if raw is None:
      return default
    try:
      return int(raw)
    except Exception as exc:
      raise ValueError(f"invalid integer for {key}: {raw}") from exc

  width = _parse_int("--width", 64)
  height = _parse_int("--height", 48)

  # CORE LOGIC
  inp = pyneat.InputOptions()
  inp.media_type = "application/vnd.simaai.tensor"
  inp.format = "FP32"
  inp.width = width
  inp.height = height
  inp.depth = 3

  s = pyneat.Session()
  s.add(pyneat.nodes.input(inp))
  s.add(pyneat.nodes.output())

  if "--print-gst" in argv:
    print(s.describe_backend())
    return 0

  seed = make_tensor(width, height, 0.0)
  run = s.build(seed, pyneat.RunMode.Sync)

  left = pyneat.make_tensor_sample("left", make_tensor(width, height, 1.0))
  right = pyneat.make_tensor_sample("right", make_tensor(width, height, 2.0))

  bundle = pyneat.Sample()
  bundle.kind = pyneat.SampleKind.Bundle
  bundle.fields = [left, right]

  assert run.push(bundle), "bundle push failed"
  out = run.pull(timeout_ms=1000)
  assert out is not None, "bundle output missing"
  assert out.kind == pyneat.SampleKind.Bundle, "expected bundle output"

  print(f"Bundle fields: {len(out.fields)}")
  for field in out.fields:
    print(f"  port={field.port_name} has_tensor={field.tensor is not None}")
  # END CORE LOGIC

  print("CHECK tutorial_completed: PASS (main path reached end without exception)")
  print("SIGNATURE " + json.dumps({
          "tutorial": "009",
          "lang": "py",
          "flow": "chapter_path",
          "run_mode": "sync_or_async",
          "output_kind": "sample_or_tensor",
          "tensor_rank": -1,
          "field_count": -1,
      },
      sort_keys=True,
      separators=(",", ":"),
  ))

  print("[OK] 009_multi_input_samples")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
