#!/usr/bin/env python3
"""Probe RTSP cadence from stream metadata reported by ffprobe."""

from __future__ import annotations

import argparse
import subprocess
from fractions import Fraction


def fps_from_rate(value: str) -> int:
    try:
        fps = float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return 0
    return int(round(fps)) if fps > 0 else 0


def probe_source_fps(url: str) -> int:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-rw_timeout",
        "5000000",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate",
        "-of",
        "default=nw=1",
        url,
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe is required when --source-fps is omitted") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("timed out while probing RTSP source FPS") from exc

    if result.returncode != 0:
        raise RuntimeError("ffprobe failed to probe RTSP source FPS")

    values: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value

    fps = fps_from_rate(values.get("avg_frame_rate", "")) or fps_from_rate(
        values.get("r_frame_rate", "")
    )
    if fps <= 0:
        raise RuntimeError("ffprobe did not report a positive RTSP source FPS")
    return fps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument(
        "--codec", choices=("h264", "avc", "h265", "hevc"), required=True
    )
    args = parser.parse_args()

    print(probe_source_fps(args.url))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
