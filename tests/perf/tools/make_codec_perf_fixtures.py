#!/usr/bin/env python3
"""Generate deterministic encoded media fixtures for codec perf tests."""

from __future__ import annotations

import argparse
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
    proc = subprocess.run(["bash", "-lc", f"command -v {name} >/dev/null 2>&1"], check=False)
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
            "-f",
            "h264",
            str(path),
        ]
    )


def generate(output_dir: Path, force: bool, width: int, height: int, fps: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    h264 = output_dir / f"h264_{width}x{height}_{fps}fps.h264"
    if h264.exists() and h264.stat().st_size > 0 and not force:
        print(f"codec perf fixture already exists: {h264}")
        return

    ensure_tool("ffmpeg")
    make_h264(h264, width, height, fps, duration_s=1)

    if h264.stat().st_size <= 0:
        raise RuntimeError(f"generated fixture is empty: {h264}")
    print(f"generated codec perf fixture: {h264}")


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
