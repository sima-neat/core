#!/usr/bin/env python3
"""Configure preproc knobs on ModelOptions and inspect the preprocess() Graph fragment.

Usage:
  python3 preprocess_images.py --model /path/to/resnet_50.tar.gz [--size 224]
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

import numpy as np


def main(argv: list[str]) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--model", type=Path, required=True)
  ap.add_argument("--size", type=int, default=224)
  args = ap.parse_args(argv[1:])

  # STEP configure-preproc
  opt = pyneat.ModelOptions()
  opt.preprocess.kind = pyneat.InputKind.Image
  opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.RGB
  opt.preprocess.input_max_width = args.size
  opt.preprocess.input_max_height = args.size
  opt.preprocess.input_max_depth = 3
  opt.preprocess.resize.enable = pyneat.AutoFlag.On
  opt.preprocess.resize.width = args.size
  opt.preprocess.resize.height = args.size
  opt.preprocess.normalize.enable = pyneat.AutoFlag.On
  opt.preprocess.normalize.mean = [0.5, 0.5, 0.5]
  opt.preprocess.normalize.stddev = [0.5, 0.5, 0.5]
  # END STEP

  # CORE LOGIC
  # STEP load-model
  model = pyneat.Model(str(args.model), opt)
  # END STEP
  # STEP inspect-preproc
  preproc_graph = model.preprocess()
  print("preproc_graph=ready")
  print(preproc_graph.describe())
  # END STEP
  # END CORE LOGIC

  rgb = np.full((args.size, args.size, 3), 120, dtype=np.uint8)
  tensor = pyneat.Tensor.from_numpy(rgb, copy=True, image_format=pyneat.PixelFormat.RGB)
  outputs = model.run([tensor], timeout_ms=2000)
  print(f"output_count={len(outputs)}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
