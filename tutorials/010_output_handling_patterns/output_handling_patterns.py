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


def summarize(sample) -> str:
  return f"kind={sample.kind} has_tensor={sample.tensor is not None} fields={len(sample.fields)}"


def main(argv: list[str]) -> int:
  import numpy as np

  if tu.has_flag(argv, "--help"):
    print(f"Usage: {argv[0]} [--print-gst]")
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
  tu.check("strict_flag_available", isinstance(tu.strict_mode(), bool),
           "strict-mode guard is observable")

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

  if tu.has_flag(argv, "--print-gst"):
    print(s.describe_backend())
    return 0

  run = s.build(tensor, pyneat.RunMode.Sync)
  out = run.run(tensor, timeout_ms=1000)
  print(f"push_and_pull output: {summarize(out)}")

  tu.ensure(out.tensor is not None, "expected tensor output")
  tu.ensure(len(out.tensor.shape) > 0, "output tensor shape is empty")
  print(f"output rank: {len(out.tensor.shape)}")
  # END CORE LOGIC

  tu.check("tutorial_completed", True, "main path reached end without exception")
  tu.signature({
      "tutorial": "010",
      "lang": "py",
      "flow": "chapter_path",
      "run_mode": "sync_or_async",
      "output_kind": "sample_or_tensor",
      "tensor_rank": -1,
      "field_count": -1,
  })

  print("[OK] 010_output_handling_patterns")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
