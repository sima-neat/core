#!/usr/bin/env python3
"""Push a Bundle Sample containing two named tensor fields and read them back.

Usage:
  python3 multi_input_samples.py [--width 64] [--height 48]
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


def make_fp32_tensor(width: int, height: int, value: float):
  arr = np.full((height, width, 3), value, dtype=np.float32)
  return pyneat.Tensor.from_numpy(arr, copy=True)


def main(argv: list[str]) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--width", type=int, default=64)
  ap.add_argument("--height", type=int, default=48)
  args = ap.parse_args(argv[1:])

  inp = pyneat.InputOptions()
  inp.media_type = "application/vnd.simaai.tensor"
  inp.format = "FP32"
  inp.width = args.width
  inp.height = args.height
  inp.depth = 3

  session = pyneat.Session()
  session.add(pyneat.nodes.input(inp))
  session.add(pyneat.nodes.output())

  seed = make_fp32_tensor(args.width, args.height, 0.0)
  run = session.build(seed, pyneat.RunMode.Sync)

  bundle = pyneat.Sample()
  bundle.kind = pyneat.SampleKind.Bundle
  bundle.fields = [
      pyneat.make_tensor_sample("left", make_fp32_tensor(args.width, args.height, 1.0)),
      pyneat.make_tensor_sample("right", make_fp32_tensor(args.width, args.height, 2.0)),
  ]

  run.push(bundle)
  out = run.pull(timeout_ms=1000)

  print(f"bundle_fields={len(out.fields)}")
  for field in out.fields:
    print(f"  port={field.port_name} has_tensor={field.tensor is not None}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
