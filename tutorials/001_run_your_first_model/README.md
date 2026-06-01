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
- [Model](/getting-started/programming-model/model)

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

## Source Files
- C++: `tutorials/001_run_your_first_model/run_your_first_model.cpp`
- Python: `tutorials/001_run_your_first_model/run_your_first_model.py`
