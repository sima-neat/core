#!/usr/bin/env python3
"""Two ways to plug a Model into a Session: direct vs. attached via ModelSessionOptions.

Usage:
  python3 plug_model_into_pipeline.py --mpk /path/to/yolo_v8s.tar.gz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
  import pyneat
except ImportError:
  sys.exit(
      "pyneat is not importable. Either Neat is not installed, or the venv is not activated.\n"
      "Run: source ~/pyneat/bin/activate"
  )


def main(argv: list[str]) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--mpk", type=Path, required=True)
  args = ap.parse_args(argv[1:])

  model = pyneat.Model(str(args.mpk))

  # Pattern A: drop the model's own session group directly into a Session.
  # CORE LOGIC
  direct = pyneat.Session()
  direct.add(model.session())
  # END CORE LOGIC
  print(f"direct_session_size={model.session().size()}")

  # Pattern B: configure the session options for an attached upstream (e.g. a camera).
  sopt = pyneat.ModelSessionOptions()
  sopt.include_appsrc = False
  sopt.include_appsink = True
  sopt.upstream_name = "camera0"
  sopt.name_suffix = "_camera0"
  sopt.buffer_name = "camera0"

  # CORE LOGIC
  attached = pyneat.Session()
  attached.add(model.session(sopt))
  # END CORE LOGIC
  print(f"attached_session_built=True")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
