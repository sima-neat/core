---
title: Before You Run Tutorials
description: Prerequisites, model archives, command contexts, and quick checks for Neat tutorials
sidebar_position: 2
slug: /tutorials/before-you-run
---

# Before You Run Tutorials

Use this page once before your first tutorial. It gets the path, model archive,
and command context sorted out so the tutorial can stay focused on the API.

## Check what you need

Before you run a chapter, make sure you have:

- Palette SDK or a target DevKit with Neat installed.
- `pyneat` available for Python tutorials.
- The tutorial extras folder if you want the prebuilt binaries under `lib/` or
  the installed tutorial sources under `share/`.
- A compiled model archive (`.tar.gz`, often called an MPK) for model-backed
  tutorials. Use the Model Zoo or compile your own model with the Model
  Compiler.

No mystery wires. If a command below does not match where you are running, stop
and switch context before copying it.

## Pick the right command context

| Command kind | Run from | Prompt context |
| --- | --- | --- |
| Python tutorial from the installed extras folder | The Neat install or extras root that contains `share/` | `sdk-or-devkit` |
| Prebuilt C++ tutorial | The Neat install or extras root that contains `lib/` | `sdk-or-devkit` |
| Rebuild a C++ tutorial from source | The repo root or extras root that contains `build.sh` | `sdk-or-devkit` |
| Run a repo binary through `dk` | Palette SDK container | `sdk-container`; the command executes on the DevKit |
| DevKit-only Python environment setup | DevKit shell | `devkit` |

Most tutorial pages show three commands:

1. Python from `share/sima-neat/tutorials/...`.
2. C++ prebuilt from `lib/sima-neat/tutorials/...`.
3. C++ rebuilt with `./build.sh --target ...`.

Run each command from the directory stated in the tutorial. Relative paths in the
examples are intentional.

## Verify the basics

Check that `sima-cli` is visible before you try to fetch model archives:

<ShellCommand prompt="sdk-or-devkit">
command -v sima-cli
</ShellCommand>

For Python tutorials, check that `pyneat` imports in the environment you will use:

<ShellCommand prompt="sdk-or-devkit">
python3 -c "import pyneat; print('pyneat ok')"
</ShellCommand>

If you run Python directly on a DevKit, activate the DevKit virtual environment
first:

<ShellCommand prompt="devkit">
source ~/pyneat/bin/activate
</ShellCommand>

## Prepare model archives

Model-backed tutorials use fixed example paths so commands stay short:

| Tutorial family | Example path used in commands |
| --- | --- |
| Classification tutorials | `/tmp/resnet_50.tar.gz` |
| Detection tutorials | `/tmp/yolo_v8s.tar.gz` |

Download the needed model from the Model Zoo when available:

<ShellCommand prompt="sdk-or-devkit">
sima-cli modelzoo get resnet_50
</ShellCommand>

<ShellCommand prompt="sdk-or-devkit">
sima-cli modelzoo get yolo_v8s
</ShellCommand>

Model Zoo filenames and download locations can vary. Use the actual tarball path
with `--model <path>`, or create a symlink to the tutorial path:

<ShellCommand prompt="sdk-or-devkit">
ln -sf /path/to/resnet_50_mpk.tar.gz /tmp/resnet_50.tar.gz
</ShellCommand>

<ShellCommand prompt="sdk-or-devkit">
ln -sf /path/to/yolo_v8s_mpk.tar.gz /tmp/yolo_v8s.tar.gz
</ShellCommand>

Then check the path you need.

<ShellCommand prompt="sdk-or-devkit">
ls -lh /tmp/resnet_50.tar.gz
</ShellCommand>

<ShellCommand prompt="sdk-or-devkit">
ls -lh /tmp/yolo_v8s.tar.gz
</ShellCommand>

## Know what a successful run looks like

Tutorial output is small by design. A successful run usually prints one or two
summary lines, such as `top1=...`, `outputs_pulled=...`, or `detections=...`.
C++ tutorials also print `[OK] ...` at the end.

Some chapters intentionally accept more than one valid output shape or label.
For example, detection tutorials may print raw output heads when the current
runtime route returns raw tensors instead of decoded BBOX data. The chapter
explains the expected variants instead of pretending all packs behave the same.

## When paths still fight back

If a tutorial reports a missing model or asset:

1. Pass the path explicitly with the tutorial's `--model` or `--image` flag.
2. Check that you are running from the directory the tutorial command expects.
3. Use the reference page for source-tree asset overrides when you are running
   tests or repo-local examples.

See [Tutorial Assets and Model Archives](/reference/tutorial-assets) for the
source-tree environment variables, and [Troubleshooting](/reference/troubleshooting)
for symptom-first fixes.
