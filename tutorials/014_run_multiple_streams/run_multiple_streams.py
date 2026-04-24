#!/usr/bin/env python3
"""Build a multistream graph: stamp -> scheduler -> fan-out -> join.

Usage:
  python3 graph_multistream.py [--streams 8] [--frames 4]
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
  graph = pyneat.graph.Graph()
  stamp = graph.add(pyneat.graph.nodes.stamp_frame_id("stamp"))

  sched_opt = pyneat.graph.nodes.StreamSchedulerOptions()
  sched_opt.per_stream_queue = 2
  sched_opt.drop_policy = pyneat.graph.nodes.StreamDropPolicy.DropOldest
  sched = graph.add(pyneat.graph.nodes.stream_scheduler(sched_opt, "sched"))

  fan = graph.add(pyneat.graph.nodes.fan_out(["image", "bbox"], "fan"))
  join = graph.add(pyneat.graph.nodes.join_bundle(["image", "bbox"], "join", "bundle"))

  graph.connect(stamp, sched)
  graph.connect(sched, fan)
  graph.connect(fan, join, "image", "image")
  graph.connect(fan, join, "bbox", "bbox")

  run = pyneat.graph.GraphSession(graph).build()
  # END CORE LOGIC
  for frame in range(args.frames):
    for sid in range(args.streams):
      run.push(stamp, make_rgb_sample(str(sid), frame))

  expected = args.streams * args.frames
  received = 0
  for _ in range(expected):
    if run.pull(join, 2000) is not None:
      received += 1
  run.stop()

  print(f"expected={expected} received={received}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
