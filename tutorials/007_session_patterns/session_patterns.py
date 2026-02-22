#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import python_utils as tu


# CORE LOGIC
def make_base_session(neat, width: int, height: int):
  inp = neat.InputOptions()
  inp.format = "RGB"
  inp.width = width
  inp.height = height
  inp.depth = 3
  inp.do_timestamp = True

  s = neat.Session()
  s.add(neat.nodes.input(inp))
  s.add(neat.nodes.output())
  return s
# END CORE LOGIC


def main(argv: list[str]) -> int:
  neat = tu.import_pyneat()
  import numpy as np

  if tu.has_flag(argv, "--help"):
    print(f"Usage: {argv[0]} [--mpk <path>] [--print-gst]")
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

  width, height = 224, 224
  direct = make_base_session(neat, width, height)

  root = tu.repo_root()
  mpk_arg = tu.get_arg(argv, "--mpk")
  mpk = Path(mpk_arg) if mpk_arg else tu.first_existing([tu.default_yolo_mpk(root), tu.default_resnet_mpk(root)])

  if mpk and mpk.exists():
    # CORE LOGIC
    model = neat.Model(str(mpk))
    print(f"model.session().size: {model.session().size()}")

    sopt = neat.ModelSessionOptions()
    sopt.include_appsrc = False
    sopt.include_appsink = True
    sopt.upstream_name = "camera0"
    sopt.name_suffix = "_camera0"
    sopt.buffer_name = "camera0"

    attached = neat.Session()
    attached.add(model.session(sopt))

    if tu.has_flag(argv, "--print-gst"):
      print("[direct]")
      print(direct.describe_backend())
      print("[attached]")
      print(attached.describe_backend())
      return 0
    # END CORE LOGIC
  elif tu.has_flag(argv, "--print-gst"):
    print(direct.describe_backend())
    return 0

  rgb = np.full((height, width, 3), 55, dtype=np.uint8)
  t = neat.Tensor.from_numpy(rgb, copy=True, image_format=neat.PixelFormat.RGB)
  # CORE LOGIC
  run = direct.build(t, neat.RunMode.Sync)
  out = run.run(t, timeout_ms=1000)
  # END CORE LOGIC
  tu.ensure(out.tensor is not None, "direct session output missing tensor")

  tu.check("tutorial_completed", True, "main path reached end without exception")
  tu.signature({
      "tutorial": "007",
      "lang": "py",
      "flow": "chapter_path",
      "run_mode": "sync_or_async",
      "output_kind": "sample_or_tensor",
      "tensor_rank": -1,
      "field_count": -1,
  })

  print("[OK] 007_session_patterns")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
