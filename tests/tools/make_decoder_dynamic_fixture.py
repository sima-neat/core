#!/usr/bin/env python3
"""Generate deterministic-ish H264 stream fixture with dynamic resolution."""

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


def make_segment(path: Path, width: int, height: int) -> None:
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
            f"testsrc=duration=1:size={width}x{height}:rate=30",
            "-vf",
            "format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-x264-params",
            "keyint=30:min-keyint=30:scenecut=0",
            "-f",
            "h264",
            str(path),
        ]
    )


def generate(output: Path, force: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0 and not force:
        print(f"decoder fixture already exists: {output}")
        return

    ensure_tool("ffmpeg")

    part1 = output.parent / "dynamic_caps_320x240.h264"
    part2 = output.parent / "dynamic_caps_640x360.h264"

    make_segment(part1, 320, 240)
    make_segment(part2, 640, 360)

    with output.open("wb") as out:
        out.write(part1.read_bytes())
        out.write(part2.read_bytes())

    for part in (part1, part2):
        part.unlink(missing_ok=True)

    if output.stat().st_size <= 0:
        raise RuntimeError(f"generated fixture is empty: {output}")

    print(f"generated decoder fixture: {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate decoder dynamic-caps fixture")
    parser.add_argument(
        "--output",
        default="tests/assets/decoder/dynamic_caps.h264",
        help="output H264 path",
    )
    parser.add_argument("--force", action="store_true", help="regenerate even if output exists")
    args = parser.parse_args()

    generate(Path(args.output), args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
