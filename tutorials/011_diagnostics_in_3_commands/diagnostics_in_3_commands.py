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


def main(argv: list[str]) -> int:
  import numpy as np

  if tu.has_flag(argv, "--help"):
    print(f"Usage: {argv[0]} [--print-gst]")
    return 0

  # Why: runtime markers make intent explicit without requiring external docs.
  # Why: parity/scorecard tooling relies on stable, machine-parseable checkpoints.
  tu.step("input_contract", "parse flags and establish deterministic defaults")
  tu.step("run_mode_choice", "exercise the chapter's primary runtime path")
  tu.step("output_contract", "emit checks and machine-parseable signature")
  tu.check("strict_flag_available", isinstance(tu.strict_mode(), bool),
           "strict-mode guard is observable")

  # CORE LOGIC
  rgb = np.full((96, 128, 3), 77, dtype=np.uint8)
  t = pyneat.Tensor.from_numpy(rgb, copy=True, image_format=pyneat.PixelFormat.RGB)

  inp = pyneat.InputOptions()
  inp.format = "RGB"
  inp.width = 128
  inp.height = 96
  inp.depth = 3

  s = pyneat.Session()
  s.add(pyneat.nodes.input(inp))
  s.add(pyneat.nodes.output())

  if tu.has_flag(argv, "--print-gst"):
    print(s.describe_backend())
    return 0

  # Command 1: validate
  rep = s.validate()
  print(f"validate.error_code: {rep.error_code}")

  # Command 2: run with metrics
  ropt = pyneat.RunOptions()
  ropt.enable_metrics = True
  ropt.output_memory = pyneat.OutputMemory.Owned
  run = s.build(t, pyneat.RunMode.Sync, ropt)
  out = run.run(t, timeout_ms=1000)
  tu.ensure(out.tensor is not None, "missing output tensor")

  # Command 3: diagnostics
  stats = run.stats()
  print(f"stats.inputs_enqueued={stats.inputs_enqueued} outputs_pulled={stats.outputs_pulled}")
  print(f"report.size={len(run.report())}")
  print(f"diag_summary={run.diagnostics_summary()}")
  # END CORE LOGIC

  tu.check("tutorial_completed", True, "main path reached end without exception")
  tu.signature({
      "tutorial": "011",
      "lang": "py",
      "flow": "chapter_path",
      "run_mode": "sync_or_async",
      "output_kind": "sample_or_tensor",
      "tensor_rank": -1,
      "field_count": -1,
  })

  print("[OK] 011_diagnostics_in_3_commands")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
