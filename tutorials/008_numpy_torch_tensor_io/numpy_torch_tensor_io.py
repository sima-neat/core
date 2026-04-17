#!/usr/bin/env python3
"""Round-trip pyneat.Tensor through numpy and (optionally) torch.

Usage:
  python3 numpy_torch_tensor_io.py [--width 128] [--height 96]
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
  ap.add_argument("--width", type=int, default=128)
  ap.add_argument("--height", type=int, default=96)
  args = ap.parse_args(argv[1:])

  # numpy round-trip: HWC uint8 RGB -> pyneat.Tensor -> numpy.
  arr = np.full((args.height, args.width, 3), 17, dtype=np.uint8)
  tensor = pyneat.Tensor.from_numpy(arr, copy=True, image_format=pyneat.PixelFormat.RGB)
  arr_back = tensor.to_numpy(copy=True)
  print(f"numpy_roundtrip_shape={arr_back.shape}")

  # torch round-trip (skipped gracefully if torch isn't installed).
  try:
    import torch
  except ImportError:
    print("torch_roundtrip_skipped=True")
    return 0

  th = torch.full((args.height, args.width, 3), 9, dtype=torch.uint8)
  tensor2 = pyneat.Tensor.from_torch(th, copy=True, image_format=pyneat.PixelFormat.RGB)
  th_back = tensor2.to_torch(copy=True)
  print(f"torch_roundtrip_shape={tuple(th_back.shape)}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
