#!/usr/bin/env python3
"""Run a ResNet-50 model on an image in three lines of pyneat.

Usage:
  python3 run_your_first_model.py --model /path/to/resnet_50.tar.gz [--image /path/to.jpg]
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


def build_options(size: int) -> pyneat.ModelOptions:
  opt = pyneat.ModelOptions()
  opt.preprocess.kind = pyneat.InputKind.Image
  opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.RGB
  opt.preprocess.input_max_width = size
  opt.preprocess.input_max_height = size
  opt.preprocess.input_max_depth = 3
  opt.preprocess.preset = pyneat.NormalizePreset.ImageNet
  return opt


def main(argv: list[str]) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--model", type=Path, required=True)
  ap.add_argument("--image", type=Path)
  args = ap.parse_args(argv[1:])

  # CORE LOGIC
  # The three-line Neat story:
  # STEP load-model
  model = pyneat.Model(str(args.model), build_options(224))
  # END STEP
  # STEP prepare-input
  image = load_image(args.image, size=224)
  tensor = pyneat.Tensor.from_numpy(image, copy=True, image_format=pyneat.PixelFormat.RGB)
  # END STEP
  # STEP run-inference
  outputs = model.run([tensor], timeout_ms=2000)
  # END STEP
  # END CORE LOGIC

  top1 = int(np.argmax(outputs[0].to_numpy().reshape(-1)))
  print(f"top1={top1}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
