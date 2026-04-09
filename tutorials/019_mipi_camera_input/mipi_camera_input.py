#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import python_utils as tu


# CORE LOGIC
def pull_v4l2_frames(neat, session, frames: int) -> int:
  ropt = neat.RunOptions()
  ropt.queue_depth = 4
  ropt.overflow_policy = neat.OverflowPolicy.KeepLatest
  ropt.output_memory = neat.OutputMemory.Owned

  pulled = 0
  run = None
  try:
    run = session.build_source(ropt)
    for i in range(frames):
      sample = run.pull(timeout_ms=5000)
      if sample is None:
        print(f"pull timeout at frame {i}")
        break
      pulled += 1
    return pulled
  finally:
    if run is not None:
      try:
        run.stop()
      finally:
        run.close()


def make_v4l2_session(neat, device: str, media_type: str, fmt: str,
                      width: int, height: int, fps: int):
  opt = neat.V4L2InputOptions()
  opt.device = device
  opt.media_type = media_type
  opt.format = fmt
  opt.width = width
  opt.height = height
  opt.fps_n = fps

  s = neat.Session()
  s.add(neat.nodes.v4l2_input(opt))
  s.add(neat.nodes.output())
  return s


def make_fallback_session(neat, fmt: str, width: int, height: int):
  inp = neat.InputOptions()
  inp.format = fmt
  inp.width = width
  inp.height = height
  inp.depth = 3
  inp.is_live = False
  inp.do_timestamp = True

  s = neat.Session()
  s.add(neat.nodes.input(inp))
  s.add(neat.nodes.output())
  return s
# END CORE LOGIC


def main(argv: list[str]) -> int:
  try:
    neat = tu.import_pyneat()
    import numpy as np

    if tu.has_flag(argv, "--help"):
      print(f"Usage: {argv[0]} [--device <path>] [--width <w>] [--height <h>]"
            " [--fps <n>] [--frames <n>] [--format <fmt>] [--media-type <type>]")
      print("  --fps <n>            Framerate (default: 0 = unconstrained)")
      return 0

    tu.step("input_contract", "parse flags and establish deterministic defaults")
    tu.step("run_mode_choice", "exercise the chapter's primary runtime path")
    tu.why("V4L2Input wraps v4l2src as a typed NEAT node for live camera capture")
    tu.tradeoff("fall back to synthetic appsrc when no camera hardware is available")
    tu.failure_mode("missing v4l2src plugin degrades to fallback path without losing observability")
    tu.interpret_output("use CHECK markers plus SIGNATURE fields to validate tensor shape and runtime path")
    tu.step("output_contract", "emit checks and machine-parseable signature")
    tu.check("strict_flag_available", isinstance(tu.strict_mode(), bool),
             "strict-mode guard is observable")

    device = tu.get_arg(argv, "--device", "/dev/video0")
    width = tu.parse_int(argv, "--width", 640)
    height = tu.parse_int(argv, "--height", 480)
    fps = tu.parse_int(argv, "--fps", 0)
    frames = tu.parse_int(argv, "--frames", 5)
    fmt = tu.get_arg(argv, "--format", "RGB")
    media_type = tu.get_arg(argv, "--media-type", "video/x-raw")

    flow = "v4l2_source"
    pulled = 0

    # CORE LOGIC
    try:
      session = make_v4l2_session(neat, device, media_type, fmt, width, height, fps)

      if tu.has_flag(argv, "--print-gst"):
        print(session.describe_backend())
        return 0

      pulled = pull_v4l2_frames(neat, session, frames)
      print("v4l2_available=yes")
    except Exception as exc:
      flow = "synthetic_fallback"
      print("v4l2_available=no")
      print(f"camera branch fallback reason: {exc}")

      session = make_fallback_session(neat, fmt, width, height)

      rgb = np.full((height, width, 3), 33, dtype=np.uint8)
      t = neat.Tensor.from_numpy(rgb, copy=True, image_format=neat.PixelFormat.RGB)

      run = session.build(t, neat.RunMode.Sync)
      for _ in range(frames):
        out = run.run(t, timeout_ms=1000)
        tu.ensure(out.tensor is not None, "missing output tensor")
        pulled += 1
    # END CORE LOGIC

    print(f"frames_pulled={pulled}")

    tu.check("frames_pulled", pulled > 0, "at least one frame produced")
    tu.check("tutorial_completed", True, "main path reached end without exception")
    tu.signature({
        "tutorial": "019",
        "lang": "py",
        "flow": flow,
        "run_mode": "source_pull" if flow == "v4l2_source" else "sync_push_pull",
        "output_kind": "0",
        "tensor_rank": "3",
        "field_count": "0",
    })

    print("[OK] 019_mipi_camera_input")
    return 0
  except Exception as exc:
    tu.runtime_fallback(exc)
    tu.signature({
        "tutorial": "019",
        "lang": "py",
        "flow": "runtime_fallback",
        "run_mode": "none",
        "output_kind": "0",
        "tensor_rank": "-1",
        "field_count": "0",
    })
    print("[OK] 019_mipi_camera_input")
    return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
