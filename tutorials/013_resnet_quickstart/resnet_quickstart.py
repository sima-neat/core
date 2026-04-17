#!/usr/bin/env python3
"""End-to-end ResNet-50 classification: build ModelOptions, run once, print top1.

Usage:
  python3 resnet_quickstart.py --mpk /path/to/resnet_50.tar.gz [--image /path/to.jpg]
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
from PIL import Image


def load_image(path: Path | None, size: int) -> np.ndarray:
  if path is None:
    return np.full((size, size, 3), 99, dtype=np.uint8)
  return np.asarray(Image.open(path).convert("RGB").resize((size, size)), dtype=np.uint8)


def main(argv: list[str]) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--mpk", type=Path, required=True)
  ap.add_argument("--image", type=Path)
  ap.add_argument("--size", type=int, default=224)
  args = ap.parse_args(argv[1:])

  opt = pyneat.ModelOptions()
  opt.media_type = "video/x-raw"
  opt.format = "RGB"
  opt.input_max_width = args.size
  opt.input_max_height = args.size
  opt.input_max_depth = 3
  opt.preproc.normalize = True
  opt.preproc.channel_mean = [0.485, 0.456, 0.406]
  opt.preproc.channel_stddev = [0.229, 0.224, 0.225]

  model = pyneat.Model(str(args.mpk), opt)
  image = load_image(args.image, size=args.size)
  sample = model.run(image, timeout_ms=2000)

  top1 = int(np.argmax(sample.tensor.to_numpy().reshape(-1)))
  print(f"top1={top1}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
