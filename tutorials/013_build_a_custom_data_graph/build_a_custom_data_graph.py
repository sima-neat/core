#!/usr/bin/env python3
"""Build a minimal public pyneat.Graph and push/pull one tensor Sample.

Usage:
  python3 build_a_custom_data_graph.py
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


def make_rgb_sample():
  arr = (np.arange(8 * 8 * 3, dtype=np.uint8) % 255).reshape(8, 8, 3)
  sample = pyneat.Sample()
  sample.kind = pyneat.SampleKind.Tensor
  sample.tensor = pyneat.Tensor.from_numpy(arr, copy=True, image_format=pyneat.PixelFormat.RGB)
  sample.stream_id = "graph"
  sample.frame_id = 42
  sample.pts_ns = 123456789
  return sample


def main(argv: list[str]) -> int:
  argparse.ArgumentParser(description=__doc__).parse_args(argv[1:])

  # CORE LOGIC
  # STEP compose-graph
  graph = pyneat.Graph()
  graph.add(pyneat.nodes.input("image"))
  graph.add(pyneat.nodes.output("out"))
  # END STEP
  # STEP connect-endpoints
  graph.connect("image", "out")
  # END STEP

  # STEP build-and-push
  run = graph.build()
  if not run.push("image", [make_rgb_sample()]):
    raise RuntimeError(f"push failed: {run.last_error()}")
  # END STEP
  # STEP pull-and-verify
  out = run.pull("out", 2000)
  run.close()
  if out is None:
    raise RuntimeError("graph produced no output")
  # END STEP
  # END CORE LOGIC

  print(f"stream_id={out.stream_id} frame_id={out.frame_id} pts_ns={out.pts_ns}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
