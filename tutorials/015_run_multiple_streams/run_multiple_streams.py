#!/usr/bin/env python3
"""Build a multistream public Graph with graphs.combine(..., ByFrame).

Usage:
  python3 run_multiple_streams.py [--streams 8] [--frames 4]
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


def make_rgb_sample(stream_id: str, frame_id: int):
  arr = (np.arange(6 * 8 * 3, dtype=np.uint8) % 255).reshape(6, 8, 3)
  sample = pyneat.Sample()
  sample.kind = pyneat.SampleKind.Tensor
  sample.tensor = pyneat.Tensor.from_numpy(arr, copy=True, image_format=pyneat.PixelFormat.RGB)
  sample.frame_id = frame_id
  sample.stream_id = stream_id
  return sample


def main(argv: list[str]) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--streams", type=int, default=8)
  ap.add_argument("--frames", type=int, default=4)
  args = ap.parse_args(argv[1:])

  # CORE LOGIC
  # STEP build-combine-graph
  graph = pyneat.graphs.combine(["left", "right"], "combined", pyneat.CombinePolicy.ByFrame)
  expected = args.streams * args.frames
  run = graph.build()
  # END STEP
  # END CORE LOGIC

  received = 0
  first_fields = -1
  for frame in range(args.frames):
    for sid in range(args.streams):
      logical_frame = frame * args.streams + sid

      # STEP push-streams
      if not run.push("left", [make_rgb_sample(str(sid), logical_frame)]):
        raise RuntimeError(f"left push failed: {run.last_error()}")
      if not run.push("right", [make_rgb_sample(str(sid), logical_frame)]):
        raise RuntimeError(f"right push failed: {run.last_error()}")
      # END STEP

      # STEP pull-bundles
      bundle = run.pull("combined", 2000)
      if bundle is None:
        raise RuntimeError(f"timed out waiting for combined output: {run.last_error()}")
      fields = len(bundle.fields)
      if fields != 2:
        raise RuntimeError("joined bundle should contain two fields")
      if first_fields < 0:
        first_fields = fields
      received += 1
      if received <= 4:
        print(f"bundle stream={bundle.stream_id} fields={fields}")
      # END STEP

  run.close()

  if received != expected:
    raise RuntimeError(f"expected={expected} received={received}")
  if first_fields != 2:
    raise RuntimeError("join should emit a two-field bundle")
  print(f"received={received} fields={first_fields}")
  print("[OK] 015_run_multiple_streams")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
