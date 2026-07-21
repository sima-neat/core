#!/usr/bin/env python3
"""Probe RTSP cadence from encoded access-unit timestamps."""

from __future__ import annotations

import argparse
import statistics

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument(
        "--codec", choices=("h264", "avc", "h265", "hevc"), required=True
    )
    args = parser.parse_args()

    Gst.init(None)
    encoding = "H265" if args.codec in ("h265", "hevc") else "H264"
    depay = "rtph265depay" if encoding == "H265" else "rtph264depay"
    elementary_parser = "h265parse" if encoding == "H265" else "h264parse"
    escaped_url = args.url.replace("\\", "\\\\").replace('"', '\\"')
    pipeline = Gst.parse_launch(
        f'rtspsrc location="{escaped_url}" protocols=tcp latency=100 name=src '
        f"src. ! application/x-rtp,media=video,encoding-name={encoding} "
        f"! {depay} ! {elementary_parser} "
        "! appsink name=sink sync=false max-buffers=16 drop=false"
    )
    sink = pipeline.get_by_name("sink")
    if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
        raise RuntimeError("failed to start the RTSP FPS probe")

    timestamps: list[int] = []
    try:
        for _ in range(10):
            sample = sink.emit("try-pull-sample", 5 * Gst.SECOND)
            if sample is None:
                raise RuntimeError("timed out while probing RTSP FPS")
            timestamps.append(sample.get_buffer().pts)
    finally:
        pipeline.set_state(Gst.State.NULL)

    deltas = [
        current - previous
        for previous, current in zip(timestamps, timestamps[1:])
        if previous != Gst.CLOCK_TIME_NONE and current > previous
    ]
    if not deltas:
        raise RuntimeError("failed to derive RTSP FPS from encoded timestamps")
    fps = round(Gst.SECOND / statistics.median(deltas))
    if fps <= 0:
        raise RuntimeError("failed to probe a positive RTSP source FPS")
    print(fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
