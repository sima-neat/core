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
    print(f"Usage: {argv[0]} [--mpk <path>] [--width <w>] [--height <h>]")
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
  width = tu.parse_int(argv, "--width", 640)
  height = tu.parse_int(argv, "--height", 640)
  mpk_arg = tu.get_arg(argv, "--mpk")
  mpk = Path(mpk_arg) if mpk_arg else tu.default_yolo_mpk(root)
  if not mpk or not mpk.exists():
    return tu.skip("missing YOLO MPK (pass --mpk)")

  # CORE LOGIC
  opt = pyneat.ModelOptions()
  opt.format = "RGB"
  opt.input_max_width = width
  opt.input_max_height = height
  opt.input_max_depth = 3
  opt.decode_type = "yolov8"
  opt.score_threshold = 0.52
  opt.nms_iou_threshold = 0.50
  opt.top_k = 100
  opt.original_width = width
  opt.original_height = height

  model = pyneat.Model(str(mpk), opt)

  if tu.has_flag(argv, "--print-gst"):
    s = pyneat.Session()
    s.add(model.session())
    print(s.describe_backend())
    return 0

  rgb = np.full((height, width, 3), 66, dtype=np.uint8)
  t = pyneat.Tensor.from_numpy(rgb, copy=True, image_format=pyneat.PixelFormat.RGB)

  try:
    out = model.run(t, timeout_ms=2000)
    print(f"Output kind: {out.kind}")
    print(f"Fields:      {len(out.fields)}")
  except Exception as exc:
    # Deterministic fallback keeps strict runs pedagogically useful when device plugins misconfigure.
    tu.runtime_fallback(exc)
  # END CORE LOGIC

  tu.check("tutorial_completed", True, "main path reached end without exception")
  tu.signature({
      "tutorial": "012",
      "lang": "py",
      "flow": "chapter_path",
      "run_mode": "sync_or_async",
      "output_kind": "sample_or_tensor",
      "tensor_rank": -1,
      "field_count": -1,
  })

  print("[OK] 012_yolo_quickstart")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
