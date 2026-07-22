#!/usr/bin/env python3
"""Generate deterministic encoded media fixtures for codec perf tests."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "command failed: "
            + " ".join(cmd)
            + "\nstdout:\n"
            + proc.stdout
            + "\nstderr:\n"
            + proc.stderr
        )


def ensure_tool(name: str) -> None:
    proc = subprocess.run(
        ["bash", "-lc", f"command -v {name} >/dev/null 2>&1"], check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(f"required tool '{name}' not found in PATH")


def make_h264(path: Path, width: int, height: int, fps: int, duration_s: int) -> None:
    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={duration_s}:size={width}x{height}:rate={fps}",
            "-vf",
            "format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-x264-params",
            f"keyint={fps}:min-keyint={fps}:scenecut=0",
            "-bsf:v",
            "filter_units=remove_types=6",
            "-f",
            "h264",
            str(path),
        ]
    )


def make_h265(path: Path, width: int, height: int, fps: int, duration_s: int) -> None:
    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={duration_s}:size={width}x{height}:rate={fps}",
            "-vf",
            "format=yuv420p",
            "-c:v",
            "libx265",
            "-preset",
            "ultrafast",
            "-profile:v",
            "main",
            "-x265-params",
            f"keyint={fps}:min-keyint={fps}:scenecut=0:bframes=0:log-level=error",
            "-bsf:v",
            "filter_units=remove_types=39|40",
            "-f",
            "hevc",
            str(path),
        ]
    )


def verify_h265(path: Path, width: int, height: int, frame_count: int) -> None:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,profile,pix_fmt,width,height,has_b_frames,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"failed to probe H.265 fixture {path}:\n{proc.stderr}")

    streams = json.loads(proc.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"expected one H.265 stream in fixture: {path}")

    stream = streams[0]
    expected = {
        "codec_name": "hevc",
        "profile": "Main",
        "pix_fmt": "yuv420p",
        "width": width,
        "height": height,
        "has_b_frames": 0,
        "nb_read_frames": str(frame_count),
    }
    mismatches = [
        f"{key}={stream.get(key)!r} (expected {value!r})"
        for key, value in expected.items()
        if stream.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(f"invalid H.265 fixture {path}: " + ", ".join(mismatches))


def needs_generation(path: Path, force: bool) -> bool:
    return force or not path.exists() or path.stat().st_size <= 0


def generate(output_dir: Path, force: bool, width: int, height: int, fps: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    h264 = output_dir / f"h264_{width}x{height}_{fps}fps_no_sei.h264"
    h265 = output_dir / f"h265_{width}x{height}_{fps}fps_no_sei.h265"
    generate_h264 = needs_generation(h264, force)
    generate_h265 = needs_generation(h265, force)

    if generate_h264 or generate_h265:
        ensure_tool("ffmpeg")
    if generate_h264:
        make_h264(h264, width, height, fps, duration_s=1)
        print(f"generated codec perf fixture: {h264}")
    else:
        print(f"codec perf fixture already exists: {h264}")
    if generate_h265:
        make_h265(h265, width, height, fps, duration_s=1)
        print(f"generated codec perf fixture: {h265}")
    else:
        print(f"codec perf fixture already exists: {h265}")

    ensure_tool("ffprobe")
    verify_h265(h265, width, height, frame_count=fps)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate codec perf fixtures")
    parser.add_argument("--output-dir", required=True, help="fixture output directory")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    generate(Path(args.output_dir), args.force, args.width, args.height, args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
