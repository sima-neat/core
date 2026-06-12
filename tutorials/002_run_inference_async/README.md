# 002 Run Inference Asynchronously

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Beginner |
| Estimated Read Time | 10-15 minutes |
| Model | resnet_50 |
| Labels | async, push-pull, throughput, runtime |

## Concept

Feed a model from a producer thread while consuming predictions from another, decoupling input and output for real throughput. Same ResNet path as chapter 001, now async.

## Walkthrough

Chapter 001 ran a model with a single synchronous call: hand it one frame, block until the result comes back. That is simple, but it wastes compute — the thread that produces inputs and the thread that consumes outputs are the same thread, so they can never overlap. This chapter keeps the exact same ResNet-50 model and turns it into a throughput-oriented pipeline by splitting those two jobs.

The mechanism is the async `Run`: you `build()` the model into a `Graph` in `Async` mode, then drive it with two independent calls — `push(...)` from a producer and `pull(...)` from a consumer. By the end you will have a producer thread feeding frames as fast as the runtime accepts them while the main thread pulls predictions out, and a final `pushed=N pulled=N` line proving nothing was lost.

### Load the model {#step-load-model}

We start exactly as in chapter 001 — construct a `Model` from the archive — but here we also declare a `RouteOptions` with `include_input` and `include_output` set. Those flags tell the model to expose its own input and output boundaries when it is composed into a graph, so the surrounding pipeline can push frames in and pull tensors out.

### Build the async pipeline {#step-build-async}

A `Model` is not directly drivable with push/pull; a `Run` is. We wrap the model in a fresh `Graph` via `graph.add(model.graph(route_opt))`, then `build(...)` it with a representative frame. Passing the sample frame lets `build()` negotiate concrete tensor shapes up front. The returned `Run` is the handle both threads will share.

### Push frames from a producer {#step-push-frames}

The producer's only job is to feed inputs. We spawn a thread that loops over the prepared frames, calls `push(...)` for each, and then calls `close_input()` to signal that no more frames are coming — that signal is what lets the consumer know when to stop. Because the producer runs independently, it does not wait for any result before sending the next frame.

**C++:** A `std::thread` runs the loop; an atomic `pushed` counter and a `producer_done` flag are updated as it goes so the main thread can observe progress without a lock.

**Python:** A `threading.Thread` named `frame_producer` runs the loop; the consumer later checks `thread.is_alive()` to detect completion.

### Pull results on the consumer {#step-pull-results}

The main thread consumes. It loops calling `pull(timeout_ms=2000)`, which returns the next available output or nothing if none arrived within the timeout. On an empty pull we check whether the producer has finished — if so we stop, otherwise we keep waiting. Each real result is reduced to a top-1 class index and printed. After the loop we join the producer and confirm `pushed == pulled`.

**C++:** `pull()` returns an `optional<Sample>`; extract tensors with `tensors_from_sample(...)` before reading bytes.

**Python:** `pull()` returns a `Sample` or `None`; `sample.tensor.to_numpy()` hands you the array to `argmax` over.

## Run

Run it and you should see one `top1=` line per frame followed by a push/pull tally. Run the **Python** and **C++ (prebuilt)** commands from the **Neat install root** (the directory that contains `share/` and `lib/`); run the **build from source** commands from the **repo root**.

**Python:**
```bash
python3 share/sima-neat/tutorials/002_run_inference_async/run_inference_async.py \
  --model /tmp/resnet_50.tar.gz --n 4
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_002_run_inference_async \
  --model /tmp/resnet_50.tar.gz --n 4
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_002_run_inference_async
./build/tutorials-standalone/tutorial_002_run_inference_async \
  --model /tmp/resnet_50.tar.gz --n 4
```

Expected output (the exact indices depend on the image; the C++ build adds a `pushed=...` field, the Python build prints only `pulled=...`):

```text
top1=285
top1=285
top1=285
top1=285
pushed=4 pulled=4
[OK] 002_run_inference_async
```

To integrate this chapter's C++ source into your own project with a custom `CMakeLists.txt` (no extras folder required), see [How to Run Tutorials](/tutorials#compile-a-copy-yourself) on the landing page.

## In Practice

This chapter uses the async push/pull surface. To measure the same model with deterministic synthetic inputs, continue to [Benchmark Your Model](/tutorials/benchmark-your-model). For the full build-vs-run and sync-vs-async model plus the complete `RunOptions` surface, see [Build an Inference Pipeline](/tutorials/build-inference-pipeline). For queue depth, overflow policy, and measurement under load, see [Tune Throughput and Queue Depth](/tutorials/tune-throughput-and-queues).

## Source Files
- C++: `tutorials/002_run_inference_async/run_inference_async.cpp`
- Python: `tutorials/002_run_inference_async/run_inference_async.py`
