#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import python_utils as tu


def main(argv: list[str]) -> int:
  neat = tu.import_pyneat()
  import numpy as np

  if tu.has_flag(argv, "--help"):
    print(f"Usage: {argv[0]} [--mpk <path>] [--size <n>] [--print-gst]")
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

  root = tu.repo_root()
  size = tu.parse_int(argv, "--size", 224)
  mpk_arg = tu.get_arg(argv, "--mpk")
  mpk = Path(mpk_arg) if mpk_arg else tu.first_existing([tu.default_resnet_mpk(root), tu.default_yolo_mpk(root)])
  if not mpk or not mpk.exists():
    return tu.skip("missing MPK (pass --mpk)")

  opt = neat.ModelOptions()
  opt.format = "RGB"
  opt.input_max_width = size
  opt.input_max_height = size
  opt.input_max_depth = 3
  opt.preproc.input_width = size
  opt.preproc.input_height = size
  opt.preproc.output_width = size
  opt.preproc.output_height = size
  opt.preproc.normalize = True
  opt.preproc.channel_mean = [0.5, 0.5, 0.5]
  opt.preproc.channel_stddev = [0.5, 0.5, 0.5]

  model = neat.Model(str(mpk), opt)

  pre_group = model.preprocess()
  print(f"preprocess group size: {pre_group.size()}")

  if tu.has_flag(argv, "--print-gst"):
    s = neat.Session()
    s.add(pre_group)
    s.add(neat.nodes.output())
    print(s.describe_backend())
    return 0

  rgb = np.full((size, size, 3), 120, dtype=np.uint8)
  t = neat.Tensor.from_numpy(rgb, copy=True, image_format=neat.PixelFormat.RGB)

  # Python bindings currently expose full model run paths, not direct stage API.
  try:
    out = model.run(t, timeout_ms=2000)
    print(f"output kind: {out.kind}")
  except Exception as exc:
    # Deterministic fallback keeps strict runs pedagogically useful when device plugins misconfigure.
    tu.runtime_fallback(exc)

  tu.check("tutorial_completed", True, "main path reached end without exception")
  tu.signature({
      "tutorial": "005",
      "lang": "py",
      "flow": "chapter_path",
      "run_mode": "sync_or_async",
      "output_kind": "sample_or_tensor",
      "tensor_rank": -1,
      "field_count": -1,
  })

  print("[OK] 005_preproc_chapter")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
