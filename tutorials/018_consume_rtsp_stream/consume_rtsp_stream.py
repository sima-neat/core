#!/usr/bin/env python3
"""Consume a live H.264 or H.265 RTSP stream via RtspDecodedInput.

The fragment handles RTSP connect, codec-specific depacketize/parse, and
hardware decode. This chapter is about the input fragment only.

Usage:
  python3 consume_rtsp_stream.py --url rtsp://host/path
    [--codec h264|avc|h265|hevc] [--source-fps 30] [--frames 5]
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
    ap.add_argument(
        "--codec",
        choices=("h264", "avc", "h265", "hevc"),
        default="h264",
        help="Encoded stream codec (default: h264)",
    )
    ap.add_argument(
        "--source-fps",
        type=int,
        default=-1,
        help="Known source cadence; recommended for H.265",
    )
    ap.add_argument("--frames", type=int, default=5, help="Frames to pull")
    args = ap.parse_args(argv[1:])
    if args.source_fps != -1 and args.source_fps <= 0:
        ap.error("--source-fps must be positive")

    # CORE LOGIC
    # STEP configure-rtsp
    # Configure the URL, codec, source cadence, and RTSP transport.
    rtsp_opt = pyneat.RtspDecodedInputOptions()
    rtsp_opt.url = args.url
    rtsp_opt.codec = {
        "h264": pyneat.RtspCodec.H264,
        "avc": pyneat.RtspCodec.AVC,
        "h265": pyneat.RtspCodec.H265,
        "hevc": pyneat.RtspCodec.HEVC,
    }[args.codec]
    rtsp_opt.source_fps = args.source_fps
    rtsp_opt.tcp = True
    # END STEP

    # STEP compose-graph
    # Build a Graph whose only stages are the RTSP group and an output node.
    graph = pyneat.Graph()
    graph.add(pyneat.groups.rtsp_decoded_input(rtsp_opt))
    graph.add(pyneat.nodes.output())
    run = graph.build(pyneat.RunOptions())
    # END STEP

    # STEP pull-frames
    for i in range(args.frames):
        frame_sample = run.pull(timeout_ms=5000)
        if frame_sample is None or not frame_sample.tensors:
            print(f"frame={i} rtsp_timeout")
            break
        frame_tensor = frame_sample.tensors[0]
        print(f"frame={i} shape={list(frame_tensor.shape)}")
    # END STEP
    # END CORE LOGIC
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
