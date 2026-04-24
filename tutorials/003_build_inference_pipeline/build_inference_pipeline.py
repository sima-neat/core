#!/usr/bin/env python3
"""Compose a Session from input + output nodes, build a Run, invoke it once.

Usage:
  python3 session_build_and_run.py [--width 320] [--height 240]
"""
from __future__ import annotations

import argparse
import sys

try:
  import pyneat
except ImportError:
  sys.exit(
      "pyneat is not importable. Either NEAT is not installed, or the venv is not activated.\n"
      "Run: source ~/pyneat/bin/activate"
  )

import numpy as np


def main(argv: list[str]) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--width", type=int, default=320)
  ap.add_argument("--height", type=int, default=240)
  args = ap.parse_args(argv[1:])

  inp = pyneat.InputOptions()
  inp.format = "RGB"
  inp.width = args.width
  inp.height = args.height
  inp.depth = 3
  inp.do_timestamp = True

  # CORE LOGIC
  session = pyneat.Session()
  session.add(pyneat.nodes.input(inp))
  session.add(pyneat.nodes.output())

  rgb = np.full((args.height, args.width, 3), 33, dtype=np.uint8)
  tensor = pyneat.Tensor.from_numpy(rgb, copy=True, image_format=pyneat.PixelFormat.RGB)

  run = session.build(tensor, pyneat.RunMode.Sync)
  sample = run.run(tensor, timeout_ms=1000)
  # END CORE LOGIC

  print(f"output_rank={len(sample.tensor.shape)}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
