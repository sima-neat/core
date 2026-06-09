# 001 Run Your First Model

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Beginner |
| Estimated Read Time | <5 minutes |
| Model | resnet_50 |
| Labels | model, inference, foundations |

## Concept

Load a compiled ResNet-50 archive, feed it an image, and read the top-1 class — the shortest path from "I have a model archive" to "I have a prediction."

## Walkthrough

This is the entry chapter. The goal is the smallest possible end-to-end inference: take a compiled model, hand it one image, and print the predicted class index. No graphs, no threads, no streaming — just the three calls that every Neat program is built on.

A *compiled model* is a deployable `.tar.gz` archive containing an MPK inference contract: the model artifacts plus the runtime metadata Neat needs to execute it on the target device. You don't unpack it or wire up stages yourself — you point Neat at the archive, give it input, and read the output. By the end you will have run inference in three lines and printed a `top1=` class index.

### Load the model {#step-load-model}

The first line turns a path-on-disk into a live, runnable `Model`: construction loads the archive and prepares it for execution.

**C++:** You pass `build_options(size)` as a second argument to declare the input contract this model expects — RGB color, `224×224`, and the ImageNet normalization ResNet-50 was trained with. Declaring it here tells the runtime how to turn a raw image into the tensor the model wants.

**Python:** The preprocessing defaults are sensible, so `pyneat.Model(path)` takes just the archive path — no options object needed for this chapter.

### Prepare the input {#step-prepare-input}

Next we produce exactly one image to classify. If you pass `--image`, it is read, resized to `224×224`, and converted to RGB to match the input contract; otherwise we synthesize a solid gray frame so the full load → run → read path still runs end to end without needing an asset on hand.

**C++:** The frame is a `cv::Mat`, produced by `load_rgb(...)` or as a gray placeholder.

**Python:** The frame is a NumPy array built by `load_image(...)` (OpenCV under the hood).

### Run inference and read the result {#step-run-inference}

The third line does the actual work: `run()` takes the input and a `timeout_ms`, executes the model synchronously, and returns the output. `timeout_ms` is the maximum wall-clock time to wait — `2000` ms here means "fail loudly if the device hasn't produced output in two seconds" rather than hanging forever. (Passing `-1` blocks indefinitely; prefer a finite value in real code.) We then reduce the output to a single class index with `argmax` and print `top1=`.

**C++:** `run()` returns a `TensorList`; read the first tensor's bytes via `map_read()`.

**Python:** `run()` returns a `Sample`; `sample.tensor.to_numpy()` hands you a NumPy array to `argmax` over.

That's the whole story. Everything in later chapters — async, pipelines, custom graphs — is built on these same three moves: construct, feed, read.

## Run

Run it and you should see the predicted class index printed to stdout. Run the **Python** and **C++ (prebuilt)** commands from the **Neat install root** (the directory that contains `share/` and `lib/`); run the **build from source** commands from the **repo root**.

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

Expected output (the exact index depends on the image):

```text
top1=285
[OK] 001_run_your_first_model
```

To integrate this chapter's C++ source into your own project with a custom `CMakeLists.txt` (no extras folder required), see [How to Run Tutorials](/tutorials#compile-a-copy-yourself) on the landing page.

For throughput, batching, or live streams, continue to chapter 002. Reference: [Model](/develop-apps/development-workflow/model).

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
