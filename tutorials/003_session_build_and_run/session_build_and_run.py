#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

try:
  import pyneat
except ImportError:
  sys.exit("pyneat is not importable. Either NEAT is not installed, or the venv is not activated.\nRun: source ~/pyneat/bin/activate\nIf the venv does not exist yet, follow the installation guide.")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import python_utils as tu


# CORE LOGIC
def build_session(width: int, height: int):
  inp = pyneat.InputOptions()
  inp.format = "RGB"
  inp.width = width
  inp.height = height
  inp.depth = 3
  inp.is_live = False
  inp.do_timestamp = True

  s = pyneat.Session()
  s.add(pyneat.nodes.input(inp))
  s.add(pyneat.nodes.output())
  return s
# END CORE LOGIC

def main(argv: list[str]) -> int:
  import numpy as np

  if tu.has_flag(argv, "--help"):
    print(f"Usage: {argv[0]} [--width <w>] [--height <h>] [--print-gst]")
    return 0

  # Why: runtime markers make intent explicit without requiring external docs.
  # Why: parity/scorecard tooling relies on stable, machine-parseable checkpoints.
  tu.step("input_contract", "parse flags and establish deterministic defaults")
  tu.step("run_mode_choice", "exercise the chapter's primary runtime path")
  tu.step("output_contract", "emit checks and machine-parseable signature")
  tu.check("strict_flag_available", isinstance(tu.strict_mode(), bool),
           "strict-mode guard is observable")

  width = tu.parse_int(argv, "--width", 320)
  height = tu.parse_int(argv, "--height", 240)

  s = build_session(width, height)
  if tu.has_flag(argv, "--print-gst"):
    print(s.describe_backend())
    return 0

  # CORE LOGIC
  rgb = np.full((height, width, 3), 33, dtype=np.uint8)
  t = pyneat.Tensor.from_numpy(rgb, copy=True, image_format=pyneat.PixelFormat.RGB)
  run = s.build(t, pyneat.RunMode.Sync)
  out = run.run(t, timeout_ms=1000)
  tu.ensure(out.tensor is not None, "missing output tensor")
  # END CORE LOGIC

  print(f"Output rank: {len(out.tensor.shape)}")
  tu.check("tutorial_completed", True, "main path reached end without exception")
  tu.signature({
      "tutorial": "003",
      "lang": "py",
      "flow": "chapter_path",
      "run_mode": "sync_or_async",
      "output_kind": "sample_or_tensor",
      "tensor_rank": -1,
      "field_count": -1,
  })

  print("[OK] 003_session_build_and_run")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
