# 001 Run Your First Model

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Beginner |
| Estimated Read Time | <5 minutes |
| Model | resnet_50 |
| Labels | model, inference, foundations |

## Concept

Load a compiled ResNet-50 model archive, feed it an image, and read the top-1 class in three lines of Python. This is the shortest path from "I have a model archive" to "I have a prediction."

A compiled model is a deployable `.tar.gz` model archive containing an MPK inference contract that Neat can load and execute on the target device. It contains the model artifacts and runtime metadata needed for inference — you provide input, call `run()`, and read outputs.

**APIs introduced**
- `pyneat.Model(model_path)` — load the compiled model.
- `model.run(input, timeout_ms)` — synchronous inference; returns a `Sample` with the model's output tensor.
  - `timeout_ms` is the max wall time (ms) to wait for output. `-1` (the default) blocks indefinitely; any `> 0` value throws on timeout so stalls surface loudly. Prefer a finite value (e.g. `2000`) in production code, `-1` only when you trust the runtime to always produce output.

**When to use this**
Fastest way to verify an model archive loads and runs on hardware. For throughput, batching, or live streams, move on to chapter 002.

**Prerequisites**
None — this is the entry chapter.

**References**
- [Model](/reference/programming-model/model)

## Learning Process
1. Set up runtime inputs: parse CLI args, locate the compiled ResNet50 model archive, and prepare sample input data.
2. Build the minimal model execution path for one model and one input stream.
3. Run synchronous inference to keep behavior deterministic and easy to debug.

## Run

**Python:**
```bash
python3 share/sima-neat/tutorials/001_run_your_first_model/run_your_first_model.py \
  --model /tmp/resnet_50.tar.gz
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_001_run_your_first_model \
  --model /tmp/resnet_50.tar.gz
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_001_run_your_first_model
./build/tutorials-standalone/tutorial_001_run_your_first_model \
  --model /tmp/resnet_50.tar.gz
```

To integrate this chapter's C++ source into your own project with a custom `CMakeLists.txt` (no extras folder required), see [How to Run Tutorials](/tutorials#compile-a-copy-yourself) on the landing page.

## In Practice

Where the tutorials and tests look for model archives (`.tar.gz`) and sample assets, and how to provide them locally. This is the prerequisite for every model-backed tutorial.

### Ensure `sima-cli` is on PATH

Some tests invoke `sima-cli` from non-interactive shells. Use this once after installing `sima-cli`:

```bash
SIMA_CLI_BIN_DIR="<path-to-sima-cli-bin>"
grep -Fqx "export PATH=\"${SIMA_CLI_BIN_DIR}:\$PATH\"" ~/.bashrc || echo "export PATH=\"${SIMA_CLI_BIN_DIR}:\$PATH\"" >> ~/.bashrc
source ~/.bashrc
```

Then verify:

```bash
/bin/sh -c 'command -v sima-cli'
```

### Model archive locations and environment variables

Extraction/runtime placement knobs:
- `SIMA_MPK_EXTRACT_ROOT=<dir>` sets the base extract directory.
- `SIMA_MPK_CLEANUP_EXTRACTED=0` preserves extracted `proc_*` model data after process exit.
- `SIMA_MPK_EXTRACT_GC_STALE_PROC=0` disables dead-`proc_*` cleanup on startup.

#### ResNet50

Search order:
1. `SIMA_RESNET50_TAR` (per-model override)
2. `SIMA_MODEL_TAR` (shared fallback for model-archive tests/examples)
3. `tmp/resnet_50.tar.gz`
4. Local files moved into `tmp/` if found: `resnet_50.tar.gz`, `resnet-50.tar.gz`

Download (if `sima-cli` is available):
```bash
sima-cli modelzoo get resnet_50
```

#### YOLOv8 (v8s)

Search order:
1. `SIMA_YOLO_TAR` (per-model override)
2. `SIMA_MODEL_TAR` (shared fallback for model-archive tests/examples)
3. `tmp/yolo_v8s.tar.gz`
4. Common local names (moved into `tmp/` if found): `yolo_v8s.tar.gz`, `yolo-v8s.tar.gz`, `yolov8s.tar.gz`, `yolov8_s.tar.gz`

Download (if `sima-cli` is available):
```bash
sima-cli modelzoo get yolo_v8s
```

### Sample images

Default image candidates used in tutorials/tests:
- `tmp/coco_sample.jpg` (downloaded if missing)
- `test.jpg`
- `tests/assets/preproc_dynamic/ilena_488.jpg`

You can override the COCO image URL used by tests with:
```bash
SIMA_COCO_URL=<custom_url>
```

### Where tests download to

Tests and examples generally place downloaded assets under `tmp/` in the repo root. Tutorials will **skip** gracefully if required assets are missing.

### Troubleshooting assets

- If a tutorial prints `SKIP: missing ...`, provide the asset or pass a flag (e.g., `--model <path>`, `--image <path>`).
- If `sima-cli` is unavailable, set the env vars to point to local model archives.

## Source Files
- C++: `tutorials/001_run_your_first_model/run_your_first_model.cpp`
- Python: `tutorials/001_run_your_first_model/run_your_first_model.py`
