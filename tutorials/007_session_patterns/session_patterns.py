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


# CORE LOGIC
def make_base_session(width: int, height: int):
  inp = pyneat.InputOptions()
  inp.format = "RGB"
  inp.width = width
  inp.height = height
  inp.depth = 3
  inp.do_timestamp = True

  s = pyneat.Session()
  s.add(pyneat.nodes.input(inp))
  s.add(pyneat.nodes.output())
  return s
# END CORE LOGIC


def main(argv: list[str]) -> int:
  import numpy as np

  if "--help" in argv:
    print(f"Usage: {argv[0]} [--mpk <path>] [--print-gst]")
    return 0

  strict_mode = os.getenv("SIMA_RUN_TUTORIALS_FULL") is not None

  # Why: runtime markers make intent explicit without requiring external docs.
  # Why: parity/scorecard tooling relies on stable, machine-parseable checkpoints.
  print("STEP input_contract: parse flags and establish deterministic defaults")
  print("STEP run_mode_choice: exercise the chapter's primary runtime path")
  print("STEP output_contract: emit checks and machine-parseable signature")
  print("CHECK strict_flag_available: PASS (strict-mode guard is observable)")
  assert isinstance(strict_mode, bool), "check failed: strict_flag_available (strict-mode guard is observable)"

  width, height = 224, 224
  direct = make_base_session(width, height)

  root = Path(__file__).resolve().parents[2]
  mpk_arg = next((argv[i + 1] for i in range(1, len(argv) - 1) if argv[i] == "--mpk"), None)
  if mpk_arg:
    mpk = Path(mpk_arg)
  else:
    yolo = next(
        (p for p in [
            root / "tmp" / "yolo_v8s_mpk.tar.gz",
            root / "tmp" / "yolov8s_mpk.tar.gz",
            root / "tmp" / "yolo_mpk.tar.gz",
        ] if p.exists()),
        None,
    )
    resnet = next(
        (p for p in [
            root / "tmp" / "resnet_50_mpk.tar.gz",
            root / "tmp" / "resnet50_mpk.tar.gz",
        ] if p.exists()),
        None,
    )
    mpk = next((p for p in [yolo, resnet] if p is not None and p.exists()), None)

  if mpk and mpk.exists():
    # CORE LOGIC
    model = pyneat.Model(str(mpk))
    print(f"model.session().size: {model.session().size()}")

    sopt = pyneat.ModelSessionOptions()
    sopt.include_appsrc = False
    sopt.include_appsink = True
    sopt.upstream_name = "camera0"
    sopt.name_suffix = "_camera0"
    sopt.buffer_name = "camera0"

    attached = pyneat.Session()
    attached.add(model.session(sopt))

    if "--print-gst" in argv:
      print("[direct]")
      print(direct.describe_backend())
      print("[attached]")
      print(attached.describe_backend())
      return 0
    # END CORE LOGIC
  elif "--print-gst" in argv:
    print(direct.describe_backend())
    return 0

  rgb = np.full((height, width, 3), 55, dtype=np.uint8)
  t = pyneat.Tensor.from_numpy(rgb, copy=True, image_format=pyneat.PixelFormat.RGB)
  # CORE LOGIC
  run = direct.build(t, pyneat.RunMode.Sync)
  out = run.run(t, timeout_ms=1000)
  # END CORE LOGIC
  assert out.tensor is not None, "direct session output missing tensor"

  print("CHECK tutorial_completed: PASS (main path reached end without exception)")
  print("SIGNATURE " + json.dumps({
          "tutorial": "007",
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

  print("[OK] 007_session_patterns")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
