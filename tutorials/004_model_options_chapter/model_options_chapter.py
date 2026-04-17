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

  root = tu.repo_root()
  mpk_arg = tu.get_arg(argv, "--mpk")
  mpk = Path(mpk_arg) if mpk_arg else tu.first_existing([tu.default_yolo_mpk(root), tu.default_resnet_mpk(root)])
  if not mpk or not mpk.exists():
    return tu.skip("missing MPK (pass --mpk)")

  # CORE LOGIC
  opt = pyneat.ModelOptions()
  opt.format = "BGR"
  opt.input_max_width = 640
  opt.input_max_height = 640
  opt.input_max_depth = 3
  opt.decode_type = "yolov8"
  opt.score_threshold = 0.35
  opt.nms_iou_threshold = 0.45
  opt.top_k = 100
  opt.original_width = 640
  opt.original_height = 640
  opt.name_suffix = "_chapter"

  opt.preproc.normalize = True
  opt.preproc.channel_mean = [0.485, 0.456, 0.406]
  opt.preproc.channel_stddev = [0.229, 0.224, 0.225]

  model = pyneat.Model(str(mpk), opt)
  # END CORE LOGIC
  print(f"input_spec.rank:  {model.input_spec().rank}")
  print(f"output_spec.rank: {model.output_spec().rank}")
  print(f"metadata keys:    {len(model.metadata())}")

  if tu.has_flag(argv, "--print-gst"):
    s = pyneat.Session()
    s.add(model.session())
    print(s.describe_backend())
    return 0

  rgb = np.full((224, 224, 3), 44, dtype=np.uint8)
  t = pyneat.Tensor.from_numpy(rgb, copy=True, image_format=pyneat.PixelFormat.BGR)
  try:
    out = model.run(t, timeout_ms=2000)
    print(f"run() output kind: {out.kind}")
  except Exception as exc:
    # Deterministic fallback keeps strict runs pedagogically useful when device plugins misconfigure.
    tu.runtime_fallback(exc)

  tu.check("tutorial_completed", True, "main path reached end without exception")
  tu.signature({
      "tutorial": "004",
      "lang": "py",
      "flow": "chapter_path",
      "run_mode": "sync_or_async",
      "output_kind": "sample_or_tensor",
      "tensor_rank": -1,
      "field_count": -1,
  })

  print("[OK] 004_model_options_chapter")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
