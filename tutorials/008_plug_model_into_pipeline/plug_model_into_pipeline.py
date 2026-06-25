#!/usr/bin/env python3
"""Two ways to plug a Model into a Graph: direct vs. attached via ModelRouteOptions.

Usage:
  python3 plug_model_into_pipeline.py --model /path/to/yolo_v8s.tar.gz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

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

  width = 224
  height = 224
  rgb = np.full((height, width, 3), 80, dtype=np.uint8)
  tensor = pyneat.Tensor.from_numpy(rgb, copy=True, image_format=pyneat.PixelFormat.RGB)

  inp = pyneat.InputOptions()
  inp.format = pyneat.Format.RGB
  inp.width = width
  inp.height = height
  inp.depth = 3

  direct = pyneat.Graph()
  direct.add(pyneat.nodes.input(inp))
  direct.add(pyneat.nodes.output())
  direct_outputs = direct.run([tensor])
  if not direct_outputs:
    raise RuntimeError("direct Graph output missing tensor")
  print(f"direct_rank={len(direct_outputs[0].shape)}")

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
