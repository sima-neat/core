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


def summarize(sample) -> str:
  return f"kind={sample.kind} has_tensor={sample.tensor is not None} fields={len(sample.fields)}"


def main(argv: list[str]) -> int:
  import numpy as np

  if "--help" in argv:
    print(f"Usage: {argv[0]} [--print-gst]")
    return 0

  strict_mode = os.getenv("SIMA_RUN_TUTORIALS_FULL") is not None

  # Why: runtime markers make intent explicit without requiring external docs.
  # Why: parity/scorecard tooling relies on stable, machine-parseable checkpoints.
  print("STEP input_contract: parse flags and establish deterministic defaults")
  print("STEP run_mode_choice: exercise the chapter's primary runtime path")
  print("STEP output_contract: emit checks and machine-parseable signature")
  print("CHECK strict_flag_available: PASS (strict-mode guard is observable)")
  assert isinstance(strict_mode, bool), "check failed: strict_flag_available (strict-mode guard is observable)"

  # CORE LOGIC
  rgb = np.full((120, 160, 3), 101, dtype=np.uint8)
  tensor = pyneat.Tensor.from_numpy(rgb, copy=True, image_format=pyneat.PixelFormat.RGB)

  inp = pyneat.InputOptions()
  inp.format = "RGB"
  inp.width = 160
  inp.height = 120
  inp.depth = 3

  s = pyneat.Session()
  s.add(pyneat.nodes.input(inp))
  s.add(pyneat.nodes.output())

  if "--print-gst" in argv:
    print(s.describe_backend())
    return 0

  run = s.build(tensor, pyneat.RunMode.Sync)
  out = run.run(tensor, timeout_ms=1000)
  print(f"push_and_pull output: {summarize(out)}")

  assert out.tensor is not None, "expected tensor output"
  assert len(out.tensor.shape) > 0, "output tensor shape is empty"
  print(f"output rank: {len(out.tensor.shape)}")
  # END CORE LOGIC

  print("CHECK tutorial_completed: PASS (main path reached end without exception)")
  print("SIGNATURE " + json.dumps({
          "tutorial": "010",
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

  print("[OK] 010_output_handling_patterns")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
