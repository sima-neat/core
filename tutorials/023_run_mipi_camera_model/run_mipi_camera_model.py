#!/usr/bin/env python3
"""Run a model from a MIPI/libcamera camera source.

Usage:
  python3 run_mipi_camera_model.py --model /path/to/model.tar.gz [--frames 5]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
  import pyneat
except ImportError:
  sys.exit(
      "pyneat is not importable. Either Neat is not installed, or the venv is not activated.\n"
      "Run: source ~/pyneat/bin/activate"
  )


def decode_type_from_token(token: str):
  v = token.strip().lower().replace("-", "")
  if v in ("", "none", "raw"):
    return pyneat.BoxDecodeType.Unspecified
  if v == "yolo":
    return pyneat.BoxDecodeType.Yolo
  if v == "yolov5":
    return pyneat.BoxDecodeType.YoloV5
  if v == "yolov8":
    return pyneat.BoxDecodeType.YoloV8
  if v == "yolov8seg":
    return pyneat.BoxDecodeType.YoloV8Seg
  if v == "yolov9":
    return pyneat.BoxDecodeType.YoloV9
  if v == "yolov9seg":
    return pyneat.BoxDecodeType.YoloV9Seg
  raise ValueError(f"unsupported --decode token: {token}")


def model_options_for_camera(camera: pyneat.CameraInputOptions, decode_type) -> pyneat.ModelOptions:
  options = pyneat.ModelOptions()
  options.preprocess.kind = pyneat.InputKind.Image
  options.preprocess.input_max_width = int(camera.width)
  options.preprocess.input_max_height = int(camera.height)
  options.preprocess.input_max_depth = 3
  options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
  options.preprocess.color_convert.output_format = pyneat.PreprocessColorFormat.RGB
  options.preprocess.resize.enable = pyneat.AutoFlag.On
  options.preprocess.resize.width = 640
  options.preprocess.resize.height = 640
  options.preprocess.resize.mode = pyneat.ResizeMode.Letterbox
  options.preprocess.resize.pad_value = 114
  options.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
  options.advanced_execution.preprocess_target = "EV74"
  options.decode_type = decode_type
  if decode_type == pyneat.BoxDecodeType.Unspecified:
    options.inference_terminal.mla_only = True
  else:
    options.advanced_execution.postprocess_target = "EV74"
    options.score_threshold = 0.25
    options.nms_iou_threshold = 0.45
    options.top_k = 100
  return options


def main(argv: list[str]) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--model", type=Path, required=True)
  ap.add_argument("--frames", type=int, default=5)
  ap.add_argument("--width", type=int, default=1920)
  ap.add_argument("--height", type=int, default=1080)
  ap.add_argument("--fps", type=int, default=30)
  ap.add_argument("--camera-name", default="")
  ap.add_argument("--decode", default="none", help="none, yolov8, yolov8seg, yolov9, ...")
  ap.add_argument("--pull-timeout-ms", type=int, default=15000)
  ap.add_argument("--strict-zero-copy", action="store_true")
  ap.add_argument("--print-backend", action="store_true")
  args = ap.parse_args(argv[1:])

  if not args.model.exists():
    raise FileNotFoundError(f"model archive not found: {args.model}")
  if args.frames <= 0:
    raise ValueError("--frames must be positive")
  if args.pull_timeout_ms <= 0:
    raise ValueError("--pull-timeout-ms must be positive")

  # CORE LOGIC
  # STEP configure-camera
  camera = pyneat.CameraInputOptions()
  camera.width = args.width
  camera.height = args.height
  camera.framerate_num = args.fps
  camera.framerate_den = 1
  camera.format = "NV12"
  camera.buffer_name = "camera0"
  camera.allow_cpu_fallback = not args.strict_zero_copy
  if args.camera_name:
    camera.camera_name = args.camera_name
  # END STEP

  # STEP configure-model
  decode_type = decode_type_from_token(args.decode)
  model = pyneat.Model(str(args.model), model_options_for_camera(camera, decode_type))

  route = pyneat.ModelRouteOptions()
  route.include_input = False
  route.include_output = True
  route.upstream_name = camera.buffer_name
  route.buffer_name = camera.buffer_name
  route.name_suffix = "_camera0"
  route.advanced_execution.preprocess_target = "EV74"
  if decode_type != pyneat.BoxDecodeType.Unspecified:
    route.advanced_execution.postprocess_target = "EV74"
  # END STEP

  # STEP compose-graph
  graph = pyneat.Graph("mipi_camera_model")
  graph.add(pyneat.nodes.camera_input(camera))
  graph.add(model.graph(route))

  if args.print_backend:
    print(graph.describe_backend(False))

  run = graph.build()
  # END STEP

  # STEP pull-output
  for i in range(args.frames):
    sample = run.pull(timeout_ms=args.pull_timeout_ms)
    if sample is None:
      msg = f"frame={i} output_timeout timeout_ms={args.pull_timeout_ms}"
      last_error = run.last_error()
      if last_error:
        msg += f" last_error={last_error}"
      print(msg)
      return 2
    tensors = list(sample.tensors)
    print(f"frame={i} tensors={len(tensors)}", end="")
    if tensors:
      print(f" first_shape={list(tensors[0].shape)}", end="")
    print()
  # END STEP
  # END CORE LOGIC

  print("[OK] 023_run_mipi_camera_model")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
