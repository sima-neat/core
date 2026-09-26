#!/usr/bin/env python3
"""Generate deterministic encoded media fixtures for codec perf tests."""

from __future__ import annotations

import argparse
import hashlib
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
    for codec, source, regenerated in (("h264", h264, generate_h264),
                                        ("h265", h265, generate_h265)):
        reference = output_dir / f"{codec}_no_b.nv12"
        if needs_generation(reference, force or regenerated):
            run(["ffmpeg", "-y", "-v", "error", "-i", str(source),
                 "-frames:v", "12", "-pix_fmt", "nv12", "-f", "rawvideo", str(reference)])
    make_accuracy_fixtures(output_dir, force)


def make_accuracy_fixtures(output_dir: Path, force: bool) -> None:
    """Keep short ordering/layout fixtures separate from no-B perf baselines."""
    from PIL import Image

    recipe = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    stamp = output_dir / "accuracy-recipe.sha256"
    force = force or not stamp.exists() or stamp.read_text() != recipe
    width, height, frames = 160, 96, 12
    for codec in ("h264", "h265"):
        path = output_dir / f"{codec}_ipb.mp4"
        reference = path.with_suffix(".nv12")
        if needs_generation(path, force) or needs_generation(reference, force):
            options = ["-x264-params", "bframes=2:b-adapt=0:scenecut=0"]
            if codec == "h265":
                options = ["-x265-params",
                           "bframes=2:b-adapt=0:scenecut=0:info=0:pools=1:frame-threads=1",
                           "-tag:v", "hvc1"]
            run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                 f"testsrc2=size={width}x{height}:rate=10", "-frames:v", str(frames),
                 "-c:v", "libx264" if codec == "h264" else "libx265",
                 "-preset", "medium", "-g", "12", "-pix_fmt", "yuv420p",
                 *options, "-avoid_negative_ts", "make_zero", str(path)])
            run(["ffmpeg", "-y", "-v", "error", "-i", str(path),
                 "-pix_fmt", "nv12", "-f", "rawvideo", str(reference)])
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
            "-show_entries", "frame=pict_type", "-of", "json", str(path)], text=True))
        types = [frame["pict_type"] for frame in probe["frames"]]
        if len(types) != frames or not {"I", "P", "B"}.issubset(types):
            raise RuntimeError(f"{path}: expected {frames} actual I/P/B pictures, got {types}")
        if reference.stat().st_size != frames * width * height * 3 // 2:
            raise RuntimeError(f"{reference}: incomplete software reference")

    for frame in range(4):
        path = output_dir / f"jpeg422_{frame}.jpg"
        reference = path.with_suffix(".i420")
        if needs_generation(path, force) or needs_generation(reference, force):
            rgb = bytes(channel for y in range(height) for x in range(width)
                        for channel in (30 + frame * 40 if y % 2 else 225 - frame * 40,
                                        30 + (x // 16 % 4) * 10 + (y // 8 % 4) * 45 + frame * 8,
                                        225 - frame * 40 if y % 2 else 30 + frame * 40))
            Image.frombytes("RGB", (width, height), rgb).save(path, quality=90, subsampling=1)
            raw = subprocess.check_output([
                "ffmpeg", "-v", "error", "-i", str(path), "-pix_fmt", "yuvj422p",
                "-f", "rawvideo", "-"])
            luma = width * height
            if len(raw) != luma * 2:
                raise RuntimeError(f"{path}: incomplete 4:2:2 reference")
            converted = bytearray(raw[:luma])
            stride = width // 2
            for plane in (luma, luma + luma // 2):
                for y in range(0, height, 2):
                    row = plane + y * stride
                    converted.extend((raw[row + x] + raw[row + stride + x] + 1) >> 1
                                     for x in range(stride))
            reference.write_bytes(converted)
        with Image.open(path) as image:
            sampling = [(component[1], component[2]) for component in image.layer]
            if image.size != (width, height) or sampling != [(2, 1), (1, 1), (1, 1)]:
                raise RuntimeError(f"{path}: incorrect JPEG dimensions/sampling: {sampling}")
        if reference.stat().st_size != width * height * 3 // 2:
            raise RuntimeError(f"{reference}: incomplete I420 reference")
    references = [(output_dir / f"jpeg422_{frame}.i420").read_bytes() for frame in range(4)]
    for index, reference in enumerate(references):
        for other in references[index + 1:]:
            if max(abs(a - b) for a, b in zip(reference, other)) <= 6:
                raise RuntimeError("JPEG reference frames are indistinguishable at tolerance 3")
    stamp.write_text(recipe)


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
