---
title: Assets and MPK Setup
description: Model packs and sample assets
sidebar_position: 2
---

# Assets & MPK setup

This page describes where tutorials/tests look for model packs (MPKs) and
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

## MPK locations and environment variables

### ResNet50

Search order:
1) `SIMA_RESNET50_TAR` (absolute or relative path)
2) `tmp/resnet_50_mpk.tar.gz`
3) Local files moved into `tmp/` if found:
   - `resnet_50_mpk.tar.gz`
   - `resnet-50_mpk.tar.gz`

Download (if `sima-cli` is available):
```
sima-cli modelzoo -v 2.0.0 get resnet_50
```

### YOLOv8 (v8s)

Search order:
1) `SIMA_YOLO_TAR`
2) `tmp/yolo_v8s_mpk.tar.gz`
3) Common local names (moved into `tmp/` if found):
   - `yolo_v8s_mpk.tar.gz`
   - `yolo-v8s_mpk.tar.gz`
   - `yolov8s_mpk.tar.gz`
   - `yolov8_s_mpk.tar.gz`

Download (if `sima-cli` is available):
```
sima-cli modelzoo -v 2.0.0 get yolo_v8s
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
  (e.g., `--mpk <path>`, `--image <path>`).
- If `sima-cli` is unavailable, set the env vars to point to local MPKs.
