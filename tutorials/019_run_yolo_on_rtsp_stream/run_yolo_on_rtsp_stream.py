#!/usr/bin/env python3
"""Run YOLOv8 on decoded frames from a live RTSP stream.

Usage:
  python3 run_yolo_on_rtsp_stream.py --url rtsp://host/path --mpk /path/to/yolo_v8s.tar.gz [--frames 5]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
  import pyneat
except ImportError:
  sys.exit(
      "pyneat is not importable. Either NEAT is not installed, or the venv is not activated.\n"
      "Run: source ~/pyneat/bin/activate"
  )


def main(argv: list[str]) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--url", required=True, help="RTSP URL to consume")
  ap.add_argument("--mpk", type=Path, required=True, help="Path to YOLOv8 MPK")
  ap.add_argument("--frames", type=int, default=5, help="Frames to process")
  args = ap.parse_args(argv[1:])

  # CORE LOGIC
  # Session 1: decode the RTSP stream into BGR frames.
  ro = pyneat.RtspDecodedInputOptions()
  ro.url = args.url
  ro.tcp = True
  ro.out_format = "BGR"
  ro.output_caps.enable = True
  ro.output_caps.format = "BGR"

  rtsp = pyneat.Session()
  rtsp.add(pyneat.groups.rtsp_decoded_input(ro))
  rtsp.add(pyneat.nodes.output())
  rtsp_run = rtsp.build(pyneat.RunOptions())

  # Session 2: YOLOv8 model.
  mopt = pyneat.ModelOptions()
  mopt.format = "BGR"
  mopt.decode_type = "yolov8"
  mopt.input_max_width = 1920
  mopt.input_max_height = 1080
  mopt.input_max_depth = 3
  model = pyneat.Model(str(args.mpk), mopt)

  # Pull live frames, feed each to the model.
  for i in range(args.frames):
    frame = rtsp_run.pull(timeout_ms=5000)
    if frame is None or frame.tensor is None:
      print(f"frame={i} rtsp_timeout")
      break
    out = model.run(frame.tensor, timeout_ms=2000)
    bbox_bytes = out.tensor.shape[0] if out.tensor is not None else "na"
    print(f"frame={i} fields={len(out.fields)} bbox_bytes={bbox_bytes}")

  rtsp_run.close()
  # END CORE LOGIC
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
