#!/usr/bin/env python3
"""Run a ResNet-50 model on an image in three lines of pyneat.

Usage:
  python3 model_in_5_minutes.py --mpk /path/to/resnet_50.tar.gz [--image /path/to.jpg]
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
import cv2


def load_image(path: Path | None, size: int) -> np.ndarray:
  if path is None:
    return np.full((size, size, 3), 99, dtype=np.uint8)
  bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
  if bgr is None:
    raise RuntimeError(f"failed to read image: {path}")
  if bgr.shape[0] != size or bgr.shape[1] != size:
    bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
  return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main(argv: list[str]) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--mpk", type=Path, required=True)
  ap.add_argument("--image", type=Path)
  args = ap.parse_args(argv[1:])

  # CORE LOGIC
  # The three-line Neat story:
  model = pyneat.Model(str(args.mpk))
  image = load_image(args.image, size=224)
  sample = model.run(image, timeout_ms=2000)
  # END CORE LOGIC

  top1 = int(np.argmax(sample.tensor.to_numpy().reshape(-1)))
  print(f"top1={top1}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
