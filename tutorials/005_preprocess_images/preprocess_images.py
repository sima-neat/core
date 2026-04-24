#!/usr/bin/env python3
"""Configure preproc knobs on ModelOptions and inspect the preprocess() node group.

Usage:
  python3 preproc_chapter.py --mpk /path/to/yolo_v8s.tar.gz [--size 224]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
  ap.add_argument("--mpk", type=Path, required=True)
  ap.add_argument("--size", type=int, default=224)
  args = ap.parse_args(argv[1:])

  opt = pyneat.ModelOptions()
  opt.format = "RGB"
  opt.input_max_width = args.size
  opt.input_max_height = args.size
  opt.input_max_depth = 3
  opt.preproc.input_width = args.size
  opt.preproc.input_height = args.size
  opt.preproc.output_width = args.size
  opt.preproc.output_height = args.size
  opt.preproc.normalize = True
  opt.preproc.channel_mean = [0.5, 0.5, 0.5]
  opt.preproc.channel_stddev = [0.5, 0.5, 0.5]

  # CORE LOGIC
  model = pyneat.Model(str(args.mpk), opt)
  preproc_group = model.preprocess()
  print(f"preproc_group_size={preproc_group.size()}")
  # END CORE LOGIC

  rgb = np.full((args.size, args.size, 3), 120, dtype=np.uint8)
  tensor = pyneat.Tensor.from_numpy(rgb, copy=True, image_format=pyneat.PixelFormat.RGB)
  sample = model.run(tensor, timeout_ms=2000)
  print(f"output_kind={sample.kind}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
