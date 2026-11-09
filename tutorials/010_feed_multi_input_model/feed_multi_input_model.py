#!/usr/bin/env python3
"""Push a Bundle Sample containing two named tensor fields and read the two fields back.

Usage:
  python3 feed_multi_input_model.py [--width 64] [--height 48]
"""
from __future__ import annotations

import argparse
import sys

try:
  import pyneat
except ImportError:
  sys.exit(
      "pyneat is not importable. Either Neat is not installed, or the venv is not activated.\n"
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

  # STEP configure-tensor-input
  inp = pyneat.InputOptions()
  inp.payload_type = pyneat.PayloadType.Tensor
  inp.format = pyneat.Format.FP32
  inp.width = args.width
  inp.height = args.height
  inp.depth = 3
  # END STEP

  # STEP build-seed-run
  graph = pyneat.Graph()
  graph.add(pyneat.nodes.input(inp))
  graph.add(pyneat.nodes.output())

  seed = make_fp32_tensor(args.width, args.height, 0.0)
  run = graph.build([seed])
  # END STEP

  # CORE LOGIC
  # STEP make-bundle
  fields = [
      pyneat.make_tensor_sample("left", make_fp32_tensor(args.width, args.height, 1.0)),
      pyneat.make_tensor_sample("right", make_fp32_tensor(args.width, args.height, 2.0)),
  ]
  # END STEP

  # STEP push-and-read
  # run() pushes the bundle and returns the output Sample, a Bundle of the two fields in order,
  # as Run::run(Sample) does in C++. Python does not expose the fields' port_name yet (it comes
  # back empty), so the fields are printed by position.
  out = run.run(fields, timeout_ms=1000)
  # END STEP
  # END CORE LOGIC

  if out.kind != pyneat.SampleKind.Bundle or len(out.fields) != 2:
    raise RuntimeError("expected a two-field bundle output")
  print(f"bundle_fields={len(out.fields)}")
  for i, field in enumerate(out.fields):
    tensors = [field.tensor] if field.tensor is not None else list(field.tensors)
    if not tensors:
      raise RuntimeError(f"field {i} has no tensor")
    print(f"  field={i} shape={list(tensors[0].shape)}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
