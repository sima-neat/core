# 003 Benchmark Your Model

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Beginner |
| Estimated Read Time | 5-10 minutes |
| Model | resnet_50 |
| Labels | benchmark, synthetic, latency, throughput, power |

## Concept

Run a compiled model with deterministic synthetic tensors and print the headline latency, throughput, power, and energy numbers returned by `Model::benchmark()`.

## Walkthrough

Chapters 001 and 002 showed how to run a model once and then how to drive it asynchronously. This chapter answers the next practical question: "How fast does this model run on the device?" The benchmark API is intentionally small. You load the model, choose how many samples to measure, call `benchmark(...)`, and read the returned `BenchmarkReport`.

The benchmark uses the model's `input_specs()` to create deterministic synthetic inputs. That makes it useful for a quick model smoke benchmark and for comparing compiled model variants, but it is not a camera benchmark. It does not include camera decode, real preprocessing variability, dynamic input sizes, or data-dependent postprocessing behavior.

### Load the model {#step-load-model}

Start with the same compiled `.tar.gz` archive used by the earlier model tutorials. No image is needed because the benchmark creates synthetic tensors from the model's declared input specs.

**C++:** Construct `simaai::neat::Model` from the archive path.

**Python:** Construct `pyneat.Model` from the archive path.

### Run the benchmark {#step-run-benchmark}

Call `benchmark(samples)`. The API warms up the async model runner, measures an async push/pull window, prints a summary to stdout, and returns the same headline values in a `BenchmarkReport`.

The sample count is the number of measured synthetic inputs. Use a larger number for steadier throughput and power numbers; use a smaller number when you only want a quick smoke check.

### Read the report {#step-read-report}

The returned report keeps only the headline fields most users need: average end-to-end latency in milliseconds, throughput in frames per second, average board power in watts when available, and measured energy in joules when available.

Power telemetry depends on board support. If the runtime cannot sample power rails on the current target, the benchmark still reports latency and throughput and leaves the power fields at zero.

## Run

Run it and you should see the benchmark summary printed by `benchmark()`, followed by the same values printed from the returned report. Run the **Python** and **C++ (prebuilt)** commands from the **Neat install root** (the directory that contains `share/` and `lib/`); run the **build from source** commands from the **repo root**.

**Python:**
```bash
python3 share/sima-neat/tutorials/003_benchmark_your_model/benchmark_your_model.py \
  --model /tmp/resnet_50.tar.gz --samples 100
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_003_benchmark_your_model \
  --model /tmp/resnet_50.tar.gz --samples 100
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_003_benchmark_your_model
./build/tutorials-standalone/tutorial_003_benchmark_your_model \
  --model /tmp/resnet_50.tar.gz --samples 100
```

Expected output (exact numbers depend on the model, board, and current load; the C++ build also prints the trailing `[OK]` line):

```text
NEAT Benchmark
Input: synthetic
Samples: 100
Latency:      12.4 ms
FPS:          80.6
Power avg:    2.3 W
Energy:       2.8 J
report_latency_ms=12.4
report_fps=80.6
report_avg_power_watts=2.3
report_energy_joules=2.8
[OK] 003_benchmark_your_model
```

To integrate this chapter's C++ source into your own project with a custom `CMakeLists.txt` (no extras folder required), see [How to Run Tutorials](/tutorials#compile-a-copy-yourself) on the landing page.

## In Practice

Use this benchmark when you want a quick answer for a compiled model archive: does it run, what is the measured async throughput, and what are the headline board power numbers on this target?

For application performance, benchmark the real pipeline too. Synthetic model input is deliberately stable, so it does not represent camera jitter, codec cost, real preprocessing, host scheduling under load, or downstream application logic. For queue-depth and backpressure tuning with a hand-built async run, see [Tune Throughput and Queue Depth](/tutorials/016-tune-throughput-and-queues).

`Model::benchmark()` requires concrete `input_specs()` dimensions. If an input shape is dynamic or non-concrete, the benchmark fails clearly instead of guessing a shape.

## Source Files
- C++: `tutorials/003_benchmark_your_model/benchmark_your_model.cpp`
- Python: `tutorials/003_benchmark_your_model/benchmark_your_model.py`
