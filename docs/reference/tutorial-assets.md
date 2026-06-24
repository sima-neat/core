---
title: Tutorial Assets and Model Archives
description: Source-tree asset lookup, model archive overrides, and debug environment variables for Neat tutorials and tests
sidebar_position: 7
slug: /reference/tutorial-assets
---

# Tutorial Assets and Model Archives

Most tutorial users only need `--model <path>` and `--image <path>`. This page
is for source-tree runs, tests, and repeatable debug sessions where you want to
control where model archives and sample assets come from.

Keep this material out of beginner tutorials. It is useful, but it is not the
first step.

## Model archive overrides

Use these environment variables when repo-local tests or helper scripts need a
model archive path without changing every command:

| Variable | Use |
| --- | --- |
| `SIMA_RESNET50_TAR` | Per-model override for ResNet-50 tutorials and tests. |
| `SIMA_MODEL_TAR` | Shared fallback model archive used by model-backed tests and examples. |
| `SIMA_YOLO_TAR` | Per-model override for YOLO-style detection tutorials and tests when supported by the script. |

Tutorial commands still accept explicit flags. Prefer `--model <path>` for a
single run; use environment variables when many commands should share the same
artifact.

<ShellCommand prompt="sdk-or-devkit">
export SIMA_RESNET50_TAR=/path/to/resnet_50.tar.gz
export SIMA_YOLO_TAR=/path/to/yolo_v8s.tar.gz
</ShellCommand>

## Common source-tree paths

Repo-local tests and examples commonly look under `tmp/` in the repo root. The
most common paths are:

| Asset | Common source-tree path |
| --- | --- |
| ResNet-50 model archive | `tmp/resnet_50.tar.gz` |
| YOLOv8s model archive | `tmp/yolo_v8s.tar.gz` |
| COCO sample image | `tmp/coco_sample.jpg` |

Installed tutorials can run from the extras folder instead. In that flow, pass
model and image paths explicitly if they are not already under `/tmp`.

## Model extraction controls

Neat extracts model archives before running them. These variables are mainly for
inspection and cleanup control during debugging:

| Variable | Effect |
| --- | --- |
| `SIMA_MPK_EXTRACT_ROOT=<dir>` | Sets the base directory for extracted model data. |
| `SIMA_MPK_CLEANUP_EXTRACTED=0` | Keeps extracted `proc_*` model data after process exit. |
| `SIMA_MPK_EXTRACT_GC_STALE_PROC=0` | Disables cleanup of stale `proc_*` directories on startup. |

Use them when you need to inspect generated config files or compare extracted
artifacts across runs. Leave the defaults on for normal tutorial work.

## Sample image overrides

Some source-tree tests use a COCO sample image and can download it if missing.
Override the URL when your environment needs a local mirror:

<ShellCommand prompt="sdk-or-devkit">
export SIMA_COCO_URL=https://example.com/path/to/coco_sample.jpg
</ShellCommand>

Prefer `--image <path>` for one tutorial run. Use `SIMA_COCO_URL` when a test
suite or CI job should fetch from a controlled location.

## Triage missing assets

If a tutorial prints `SKIP: missing ...` or fails to open a model:

1. Check the path shown in the message.
2. Pass the asset explicitly with the tutorial flag.
3. If you run source-tree tests, set the matching environment variable.
4. If the asset came from the Model Zoo, confirm the downloaded filename and
   point the tutorial at the actual `.tar.gz` file.

Do not unpack the model archive by hand for normal runs. Neat expects the
compiled archive path and handles extraction itself.
