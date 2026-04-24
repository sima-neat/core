# 001 Run Your First Model

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Beginner |
| Estimated Read Time | <5 minutes |
| Labels | model, mpk, inference, foundations |

## Concept

Load a compiled ResNet-50 MPK, feed it an image, and read the top-1 class in three lines of Python. This is the shortest path from "I have a model package" to "I have a prediction."

A compiled model is a deployable model package (`.tar.gz`, often called an MPK) that NEAT can load and execute on the target device. It contains the model artifacts and runtime metadata needed for inference — you provide input, call `run()`, and read outputs.

**APIs introduced**
- `pyneat.Model(mpk_path)` — load the compiled model.
- `model.run(input, timeout_ms)` — synchronous inference; returns a `Sample` with the model's output tensor.

**When to use this**
Fastest way to verify an MPK loads and runs on hardware. For throughput, batching, or live streams, move on to chapter 002.

**Prerequisites**
None — this is the entry chapter.

**References**
- [Model](/getting-started/programming-model/model)

## Learning Process
1. Set up runtime inputs: parse CLI args, locate the compiled ResNet50 MPK, and prepare sample input data.
2. Build the minimal model execution path for one model and one input stream.
3. Run synchronous inference to keep behavior deterministic and easy to debug.
4. Inspect top-1 predictions and validation output (`CHECK`, `SIGNATURE`, `[OK]`).

## What To Observe
- `CHECK ...` lines should indicate contract and runtime validation outcomes.
- `SIGNATURE ...` output should summarize machine-parseable chapter behavior.
- `[OK] ...` indicates the chapter flow completed successfully.

## Run

Fetch the ResNet-50 MPK once: `sima-cli modelzoo -v 2.0.0 get resnet_50`.

**Python:**
```bash
python3 /usr/share/sima-neat/tutorials/001_run_your_first_model/run_your_first_model.py \
  --mpk /path/to/resnet_50.tar.gz
```

**C++:**
```bash
/usr/lib/sima-neat/tutorials/tutorial_v2_001_run_your_first_model \
  --mpk /path/to/resnet_50.tar.gz
```

To compile this chapter's C++ source in your own project with a custom `CMakeLists.txt` (no `sima-neat-extras.deb` required), see [How to Run Tutorials](/tutorials/v2#compile-a-copy-yourself) on the landing page.

## Source Files
- C++: `tutorials/001_run_your_first_model/run_your_first_model.cpp`
- Python: `tutorials/001_run_your_first_model/run_your_first_model.py`
