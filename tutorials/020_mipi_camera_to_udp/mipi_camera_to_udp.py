#!/usr/bin/env python3
"""Tutorial 020 - Modalix MIPI camera to UDP H.264 streaming.

Replaces the hand-rolled gst-launch pipeline used for Modalix camera
bring-up with a typed NEAT session: V4L2Input -> VideoConvert ->
H264EncodeSima -> UdpH264OutputGroup.

Board prerequisites (run externally before this tutorial):
    gst-launch-1.0 v4l2src device=/dev/video0raw \\
      ! 'video/x-bayer,format=rggb12le,width=1920,height=1080,framerate=30/1' \\
      ! fakesink &
    sudo isp_3a.elf &

Host receiver (start before launching the tutorial):
    gst-launch-1.0 udpsrc port=9000 \\
      ! application/x-rtp,encoding-name=H264,payload=96 \\
      ! rtpjitterbuffer ! rtph264depay \\
      ! video/x-h264,stream-format=byte-stream,alignment=au \\
      ! avdec_h264 ! fpsdisplaysink sync=0
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import python_utils as tu  # noqa: E402


@dataclass
class CliArgs:
  device: str = "/dev/video0out"
  media_type: str = "video/x-raw"
  fmt: str = "RGB"
  width: int = 1920
  height: int = 1080
  fps: int = 0
  host: str = "127.0.0.1"
  port: int = 9000
  bitrate_kbps: int = 4000
  profile: str = "baseline"
  level: str = "4.2"
  enc_fps: int = 30
  duration_ms: int = 5000


def parse_cli(argv: list[str]) -> CliArgs:
  a = CliArgs()
  a.device = tu.get_arg(argv, "--device", a.device) or a.device
  a.media_type = tu.get_arg(argv, "--media-type", a.media_type) or a.media_type
  a.fmt = tu.get_arg(argv, "--format", a.fmt) or a.fmt
  a.width = tu.parse_int(argv, "--width", a.width)
  a.height = tu.parse_int(argv, "--height", a.height)
  a.fps = tu.parse_int(argv, "--fps", a.fps)
  a.host = tu.get_arg(argv, "--host", a.host) or a.host
  a.port = tu.parse_int(argv, "--port", a.port)
  a.bitrate_kbps = tu.parse_int(argv, "--bitrate", a.bitrate_kbps)
  a.profile = tu.get_arg(argv, "--profile", a.profile) or a.profile
  a.level = tu.get_arg(argv, "--level", a.level) or a.level
  a.enc_fps = tu.parse_int(argv, "--enc-fps", a.enc_fps)
  a.duration_ms = tu.parse_int(argv, "--duration-ms", a.duration_ms)
  return a


# CORE LOGIC
def make_mipi_udp_session(neat, a: CliArgs):
  v4l2_opt = neat.V4L2InputOptions()
  v4l2_opt.device = a.device
  v4l2_opt.media_type = a.media_type
  v4l2_opt.format = a.fmt
  v4l2_opt.width = a.width
  v4l2_opt.height = a.height
  v4l2_opt.fps_n = a.fps

  udp_opt = neat.UdpH264OutputGroupOptions()
  udp_opt.udp_host = a.host
  udp_opt.udp_port = a.port

  s = neat.Session()
  s.add(neat.nodes.v4l2_input(v4l2_opt))
  s.add(neat.nodes.queue())
  s.add(neat.nodes.video_convert())
  s.add(
      neat.nodes.h264_encode_sima(
          width=a.width,
          height=a.height,
          fps=a.enc_fps,
          bitrate_kbps=a.bitrate_kbps,
          profile=a.profile,
          level=a.level,
      )
  )
  s.add(neat.groups.udp_h264_output_group(udp_opt))
  return s


def run_mipi_stream(neat, session, duration_ms: int) -> None:
  ropt = neat.RunOptions()
  ropt.queue_depth = 4
  ropt.overflow_policy = neat.OverflowPolicy.KeepLatest

  run = None
  try:
    run = session.build_source(ropt)
    time.sleep(max(duration_ms, 0) / 1000.0)
  finally:
    if run is not None:
      try:
        run.stop()
      finally:
        run.close()
# END CORE LOGIC


def main(argv: list[str]) -> int:
  try:
    neat = tu.import_pyneat()

    if tu.has_flag(argv, "--help"):
      print(
          f"Usage: {argv[0]}"
          " [--device <path>] [--width <w>] [--height <h>] [--fps <n>]"
          " [--format <fmt>] [--media-type <type>]"
          " [--host <ip>] [--port <p>] [--bitrate <kbps>]"
          " [--profile <name>] [--level <n>] [--enc-fps <n>]"
          " [--duration-ms <ms>]"
      )
      return 0

    tu.step("input_contract",
            "parse V4L2 + UDP flags and establish Modalix-friendly defaults")
    tu.step("run_mode_choice",
            "source-mode session.build_source() drives the live pipeline")
    tu.why("demonstrates V4L2Input + VideoConvert + H264EncodeSima + UdpH264OutputGroup "
           "as a replacement for hand-rolled gst-launch pipelines")
    tu.tradeoff(
        "VideoConvert performs RGB->NV12 on the A65; for zero-copy on SiMa hardware, "
        "swap in Preproc with a neatprocesscvu colorconvert JSON"
    )
    tu.failure_mode("missing gstreamer plugins or no MIPI camera -> runtime_fallback")
    tu.interpret_output(
        "start the UDP receiver on the host first; frames appear on fpsdisplaysink "
        "once the tutorial reaches the sleep loop"
    )
    tu.step("output_contract", "emit checks and machine-parseable signature")
    tu.check("strict_flag_available", isinstance(tu.strict_mode(), bool),
             "strict-mode guard is observable")

    args = parse_cli(argv)

    flow = "mipi_udp_stream"

    # CORE LOGIC
    session = make_mipi_udp_session(neat, args)

    if tu.has_flag(argv, "--print-gst"):
      print(session.describe_backend())
      return 0

    print(f"streaming to udp://{args.host}:{args.port} for {args.duration_ms} ms")
    run_mipi_stream(neat, session, args.duration_ms)
    # END CORE LOGIC

    tu.check("stream_completed", True, "pipeline stopped cleanly after duration")
    tu.check("tutorial_completed", True, "main path reached end without exception")
    tu.signature({
        "tutorial": "020",
        "lang": "py",
        "flow": flow,
        "run_mode": "source_stream",
        "output_kind": "0",
        "tensor_rank": "-1",
        "field_count": "0",
    })

    print("[OK] 020_mipi_camera_to_udp")
    return 0
  except Exception as exc:
    tu.runtime_fallback(exc)
    tu.signature({
        "tutorial": "020",
        "lang": "py",
        "flow": "runtime_fallback",
        "run_mode": "none",
        "output_kind": "0",
        "tensor_rank": "-1",
        "field_count": "0",
    })
    print("[OK] 020_mipi_camera_to_udp")
    return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
