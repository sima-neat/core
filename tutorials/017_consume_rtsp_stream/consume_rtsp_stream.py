#!/usr/bin/env python3
"""Consume a live H.264 RTSP stream via the RtspDecodedInput Graph fragment.

The fragment handles RTSP connect, depacketize, and H.264 decode — you hand it a
URL and pull decoded frames. This chapter is about the input fragment only.

Usage:
  python3 consume_rtsp_stream.py --url rtsp://host/path [--frames 5]
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


def main(argv: list[str]) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--url", required=True, help="RTSP URL to consume")
  ap.add_argument("--frames", type=int, default=5, help="Frames to pull")
  args = ap.parse_args(argv[1:])

  # CORE LOGIC
  # Configure RtspDecodedInputOptions, build a Graph whose only stages are
  # the RTSP group and an output node, and pull decoded frames.
  rtsp_opt = pyneat.RtspDecodedInputOptions()
  rtsp_opt.url = args.url
  rtsp_opt.tcp = True

  s = pyneat.Graph()
  s.add(pyneat.groups.rtsp_decoded_input(rtsp_opt))
  s.add(pyneat.nodes.output())
  run = s.build(pyneat.RunOptions())

  for i in range(args.frames):
    sample = run.pull(timeout_ms=5000)
    if sample is None or sample.tensor is None:
      print(f"frame={i} rtsp_timeout")
      break
    print(f"frame={i} shape={list(sample.tensor.shape)}")
  # END CORE LOGIC
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
