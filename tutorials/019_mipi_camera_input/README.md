# 019 MIPI Camera Input

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Intermediate |
| Estimated Read Time | 10-15 minutes |
| Labels | camera, v4l2, mipi |

## Concept
This tutorial demonstrates how to use `V4L2Input` to capture live frames
from a V4L2 camera device. It covers USB cameras and Modalix MIPI
post-ISP streams.

When a V4L2 device is available, the tutorial builds a source pipeline
that captures frames and pulls them as tensors. When no camera is
present (CI, desktop without camera), it falls back to a synthetic
appsrc path that exercises the same output contract.

**Modalix MIPI note:** When targeting `/dev/video0out`, the raw sensor
keepalive (`v4l2src device=/dev/video0raw ! fakesink &`) and ISP 3A
controller (`sudo isp_3a.elf`) must be running externally before
starting this tutorial.

## Learning Process
1. Configure `V4L2InputOptions` with device, caps, and format.
2. Build a source-mode session (no appsrc push — the pipeline self-drives).
3. Pull frames and inspect the output tensor shape.
4. Understand the fallback path when hardware is unavailable.

Use `--fps 0` to omit the framerate caps field when the device advertises a
different rate than expected. On Modalix `/dev/video0out`, this is often the
safest default.

## Troubleshooting
- If `/dev/video0out` times out after a previous NEAT or GStreamer run, first
  run `bash /usr/bin/fix_devkit_runtime.sh`, then restart the raw keepalive
  pipeline and `sudo isp_3a.elf`.
- If the stream still does not recover, reboot the board before treating the
  issue as a `V4L2Input` caps regression. We have seen stale Modalix runtime
  state survive process exit even when the generated `v4l2src` fragment is
  correct.

## What To Observe
- `v4l2_available=yes/no` indicates whether a real camera was detected.
- `CHECK` lines validate tensor shape matches the requested caps.
- `SIGNATURE` summarizes the runtime path taken.

## Run
```bash
./tutorial_v2_019_mipi_camera_input
./tutorial_v2_019_mipi_camera_input --device /dev/video0out --width 1920 --height 1080 --fps 0
python3 tutorials/019_mipi_camera_input/mipi_camera_input.py
python3 tutorials/019_mipi_camera_input/mipi_camera_input.py --device /dev/video0out --width 1920 --height 1080 --fps 0
```

## Source Files
- C++: `tutorials/019_mipi_camera_input/mipi_camera_input.cpp`
- Python: `tutorials/019_mipi_camera_input/mipi_camera_input.py`
