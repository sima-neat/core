#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import python_utils as tu


def make_tensor(neat, width: int, height: int, value: float):
  import numpy as np

  arr = np.full((height, width, 3), value, dtype=np.float32)
  return neat.Tensor.from_numpy(arr, copy=True)


def main(argv: list[str]) -> int:
  neat = tu.import_pyneat()

  if tu.has_flag(argv, "--help"):
    print(f"Usage: {argv[0]} [--width <w>] [--height <h>] [--print-gst]")
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

  width = tu.parse_int(argv, "--width", 64)
  height = tu.parse_int(argv, "--height", 48)

  # CORE LOGIC
  inp = neat.InputOptions()
  inp.media_type = "application/vnd.simaai.tensor"
  inp.format = "FP32"
  inp.width = width
  inp.height = height
  inp.depth = 3

  s = neat.Session()
  s.add(neat.nodes.input(inp))
  s.add(neat.nodes.output())

  if tu.has_flag(argv, "--print-gst"):
    print(s.describe_backend())
    return 0

  seed = make_tensor(neat, width, height, 0.0)
  run = s.build(seed, neat.RunMode.Sync)

  left = neat.make_tensor_sample("left", make_tensor(neat, width, height, 1.0))
  right = neat.make_tensor_sample("right", make_tensor(neat, width, height, 2.0))

  bundle = neat.Sample()
  bundle.kind = neat.SampleKind.Bundle
  bundle.fields = [left, right]

  tu.ensure(run.push(bundle), "bundle push failed")
  out = run.pull(timeout_ms=1000)
  tu.ensure(out is not None, "bundle output missing")
  tu.ensure(out.kind == neat.SampleKind.Bundle, "expected bundle output")

  print(f"Bundle fields: {len(out.fields)}")
  for field in out.fields:
    print(f"  port={field.port_name} has_tensor={field.tensor is not None}")
  # END CORE LOGIC

  tu.check("tutorial_completed", True, "main path reached end without exception")
  tu.signature({
      "tutorial": "009",
      "lang": "py",
      "flow": "chapter_path",
      "run_mode": "sync_or_async",
      "output_kind": "sample_or_tensor",
      "tensor_rank": -1,
      "field_count": -1,
  })

  print("[OK] 009_multi_input_samples")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
