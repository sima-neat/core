#!/usr/bin/env python3
"""Two ways to plug a Model into a Graph: direct vs. attached via ModelRouteOptions.

Usage:
  python3 plug_model_into_pipeline.py --model /path/to/yolo_v8s.tar.gz
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
  ap.add_argument("--model", type=Path, required=True)
  args = ap.parse_args(argv[1:])

  model = pyneat.Model(str(args.model))

  # Pattern A: ask model.graph() to include explicit public Input/Output boundaries.
  # STEP model-graph
  runnable_opt = pyneat.ModelRouteOptions()
  runnable_opt.include_input = True
  runnable_opt.include_output = True
  # CORE LOGIC
  direct = pyneat.Graph()
  direct.add(model.graph(runnable_opt))
  # END CORE LOGIC
  # END STEP
  print("direct_graph_backend=")
  print(direct.describe_backend())

  # Pattern B: configure the route options for an attached upstream (e.g. a camera).
  # STEP route-options
  sopt = pyneat.ModelRouteOptions()
  sopt.include_input = False
  sopt.include_output = True
  sopt.upstream_name = "camera0"
  sopt.name_suffix = "_camera0"
  sopt.buffer_name = "camera0"
  # END STEP

  # CORE LOGIC
  # STEP attached-graph
  attached = pyneat.Graph()
  attached.add(model.graph(sopt))
  # END CORE LOGIC
  # END STEP
  print(f"attached_graph_built=True")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
