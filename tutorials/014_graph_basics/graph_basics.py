#!/usr/bin/env python3
"""Build a two-node pyneat.graph.Graph and push/pull one tensor Sample.

Usage:
  python3 graph_basics.py
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


def make_rgb_sample():
  arr = (np.arange(8 * 8 * 3, dtype=np.uint8) % 255).reshape(8, 8, 3)
  sample = pyneat.Sample()
  sample.kind = pyneat.SampleKind.Tensor
  sample.tensor = pyneat.Tensor.from_numpy(arr, copy=True, image_format=pyneat.PixelFormat.RGB)
  sample.stream_id = "graph"
  sample.frame_id = -1
  return sample


def main(argv: list[str]) -> int:
  argparse.ArgumentParser(description=__doc__).parse_args(argv[1:])

  graph = pyneat.graph.Graph()
  pipe = graph.add(pyneat.graph.nodes.pipeline_node(pyneat.nodes.video_convert(), "convert"))
  stamp = graph.add(pyneat.graph.nodes.stamp_frame_id("stamp"))
  graph.connect(pipe, stamp)

  run = pyneat.graph.GraphSession(graph).build()
  run.push(pipe, make_rgb_sample())
  out = run.pull(stamp, 2000)
  run.stop()

  print(f"stream_id={out.stream_id} frame_id={out.frame_id}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
