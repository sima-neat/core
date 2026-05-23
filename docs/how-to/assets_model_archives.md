---
title: Assets and Model Archive Setup
description: Model archives and sample assets
sidebar_position: 2
---

# Assets & model archive setup

This page describes where tutorials/tests look for model archives (`.tar.gz`) and
sample assets, and how to provide them locally.

## Ensure `sima-cli` is on PATH

Some tests invoke `sima-cli` from non-interactive shells. Use this once after
installing `sima-cli`:

```bash
SIMA_CLI_BIN_DIR="<path-to-sima-cli-bin>"
grep -Fqx "export PATH=\"${SIMA_CLI_BIN_DIR}:\$PATH\"" ~/.bashrc || echo "export PATH=\"${SIMA_CLI_BIN_DIR}:\$PATH\"" >> ~/.bashrc
source ~/.bashrc
```

Then verify:

```bash
/bin/sh -c 'command -v sima-cli'
```

## Model archive locations and environment variables

Extraction/runtime placement knobs:
- `SIMA_MPK_EXTRACT_ROOT=<dir>` sets the base extract directory.
- `SIMA_MPK_CLEANUP_EXTRACTED=0` preserves extracted `proc_*` model data after process exit.
- `SIMA_MPK_EXTRACT_GC_STALE_PROC=0` disables dead-`proc_*` cleanup on startup.

### ResNet50

Search order:
1) `SIMA_RESNET50_TAR` (per-model override)
2) `SIMA_MODEL_TAR` (shared fallback for model-archive tests/examples)
3) `tmp/resnet_50.tar.gz`
4) Local files moved into `tmp/` if found:
   - `resnet_50.tar.gz`
   - `resnet-50.tar.gz`

Download (if `sima-cli` is available):
```
sima-cli modelzoo get resnet_50
```

### YOLOv8 (v8s)

Search order:
1) `SIMA_YOLO_TAR` (per-model override)
2) `SIMA_MODEL_TAR` (shared fallback for model-archive tests/examples)
3) `tmp/yolo_v8s.tar.gz`
4) Common local names (moved into `tmp/` if found):
   - `yolo_v8s.tar.gz`
   - `yolo-v8s.tar.gz`
   - `yolov8s.tar.gz`
   - `yolov8_s.tar.gz`

Download (if `sima-cli` is available):
```
sima-cli modelzoo get yolo_v8s
```

## Sample images

Default image candidates used in tutorials/tests:
- `tmp/coco_sample.jpg` (downloaded if missing)
- `test.jpg`
- `tests/assets/preproc_dynamic/ilena_488.jpg`

You can override the COCO image URL used by tests with:
```
SIMA_COCO_URL=<custom_url>
```

## Where tests download to

Tests and examples generally place downloaded assets under `tmp/` in the repo
root. Tutorials will **skip** gracefully if required assets are missing.

## Troubleshooting

- If a tutorial prints `SKIP: missing ...`, provide the asset or pass a flag
  (e.g., `--model <path>`, `--image <path>`).
- If `sima-cli` is unavailable, set the env vars to point to local model archives.
