# 002 Run Inference Asynchronously

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Beginner |
| Estimated Read Time | 10-15 minutes |
| Model | resnet_50 |
| Labels | async, push-pull, throughput, runtime |

## Concept

Feed a model from a producer thread while consuming predictions from another — decouple input and output for real throughput. Same ResNet path as chapter 001, now async.

In a synchronous loop, one thread blocks while waiting for each result. That underutilizes compute when input production and output consumption can overlap. Async execution improves throughput by decoupling:
- `push(...)` feeds inputs as they become ready.
- `pull(...)` consumes outputs independently.

**APIs introduced**
- `pyneat.Session()` + `session.add(model.session())` — compose a model into a runnable session.
- `session.build(sample, pyneat.RunMode.Async)` — produce an async `Run` handle.
- `run.push(frame)`, `run.pull(timeout_ms)`, `run.close_input()` — the producer/consumer pair.

**When to use this**
Camera streams, batch processing, or any pipeline where inputs arrive faster than a one-at-a-time synchronous loop can handle.

**Prerequisites**
Chapter 001. Familiarity with `pyneat.Model` and `model.run()` is assumed.

**References**
- [Session](/getting-started/programming-model/session)
- [Pipeline](/getting-started/programming-model/pipeline)

## Learning Process
1. Prepare runtime inputs: parse CLI args, load ResNet50 MPK, and construct local input samples.
2. Build the async run path and split responsibilities between producer `push(...)` and consumer `pull(...)`.
3. Observe queue-driven behavior and verify throughput-oriented execution.
4. Validate results with top-1 output, async stats, and stable tutorial signature.

## Run

**Python:**
```bash
python3 share/sima-neat/tutorials/002_run_inference_async/run_inference_async.py \
  --mpk /tmp/resnet_50_mpk.tar.gz --n 4
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_002_run_inference_async \
  --mpk /tmp/resnet_50_mpk.tar.gz --n 4
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_002_run_inference_async
./build/tutorials-standalone/tutorial_002_run_inference_async \
  --mpk /tmp/resnet_50_mpk.tar.gz --n 4
```

To integrate this chapter's C++ source into your own project with a custom `CMakeLists.txt` (no extras folder required), see [How to Run Tutorials](/tutorials#compile-a-copy-yourself) on the landing page.

## Source Files
- C++: `tutorials/002_run_inference_async/run_inference_async.cpp`
- Python: `tutorials/002_run_inference_async/run_inference_async.py`
