# 003 Benchmark Your Model

## Metadata
| Field | Value |
| --- | --- |
| Category | Models & Inference |
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

Call `benchmark(samples)`. The API warms up the model, measures single-flight latency over `samples` inputs, then measures throughput in a separate async push/pull pass, prints a summary to stdout, and returns the same headline values in a `BenchmarkReport`.

The throughput pass runs until it has measured at least `samples` inputs and 30 seconds have passed, so its power and temperature readings are steady. Set `BenchmarkOptions::min_duration_ms` to change that floor, or to 0 for a quick smoke check that stops after `samples` inputs.

Detection models whose route ends in BoxDecode can also use `BenchmarkOptions`. Set
`original_width`, `original_height`, and `resize_mode` to describe the source-image geometry that
BoxDecode uses when it maps detections from model coordinates. The synthetic tensor remains
model-shaped:

```cpp
simaai::neat::BenchmarkOptions options;
options.num_samples = 100;
options.original_width = 1920;
options.original_height = 1080;
options.resize_mode = simaai::neat::ResizeMode::Letterbox;
auto report = model.benchmark(options);
```

Python exposes the same fields through `pyneat.BenchmarkOptions`. Set both original dimensions or
omit both; when omitted, the benchmark infers geometry from the resolved model route. Per-run
benchmark geometry takes precedence over deprecated BoxDecode geometry in `ModelOptions`.

### Read the report {#step-read-report}

The returned report starts with the headline fields most users need: average end-to-end latency in milliseconds, throughput in frames per second, average and peak SoM power in watts, and the energy in joules needed to process `samples` inputs. `throughput_duration_ms` reports how long the throughput pass ran.

Power comes from Sentinel, the board telemetry daemon that the Neat installer sets up. `report.telemetry` also carries rail power, die and board temperatures, CPU utilization, and MLA memory for the throughput pass; see [Measure model execution](/develop-apps/development-workflow/model#power-and-board-telemetry) for each field. If Sentinel is not running or is recording another trace, the benchmark still reports latency and throughput, leaves the power fields at zero, and sets `report.telemetry.status` to the reason.

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
Compiled batch: 1
Measurement detail: e2e+throughput-only
Latency: 1.95 ms
FPS:     1458.19 inferences/s
Batch throughput: 1458.19 batches/s
Throughput window: 30.08 s, 43866 inputs
Power avg:    13.57 W
Power peak:   14.06 W
Energy:       0.93 J for 100 inputs
power_mla_watts: 3.28 W avg, 3.50 W max
die_temperature_max_c: 51.08 C avg, 51.98 C max
cpu_usage_pct: 12.74 % avg, 13.51 % max
mla_memory_mib: 29.65 MiB avg, 29.65 MiB max
cma_memory_mib: 6023.09 MiB avg, 6033.17 MiB max
linux_memory_used_mib: 681.10 MiB avg, 698.57 MiB max
Telemetry:    Sentinel, 15 samples, trace 20261014T135442.014Z-neat-benchmark-20261014t135441-939z
report_latency_ms=1.94755
report_fps=1458.19
report_avg_power_watts=13.5708
report_energy_joules=0.930664
report_peak_power_watts=14.0625
report_throughput_duration_ms=30082.5
report_telemetry_status=ok
[OK] 003_benchmark_your_model
```

To integrate this chapter's C++ source into your own project with a custom `CMakeLists.txt` (no extras folder required), see [How to Run Tutorials](/tutorials#compile-a-copy-yourself) on the landing page.

## In Practice

Use this benchmark when you want a quick answer for a compiled model archive: does it run, what is the measured async throughput, and what are the headline board power numbers on this target?

For application performance, benchmark the real pipeline too. Synthetic model input is deliberately stable, so it does not represent camera jitter, codec cost, real preprocessing, host scheduling under load, or downstream application logic. For queue-depth and backpressure tuning with a hand-built async run, see [Tune Throughput and Queue Depth](/tutorials/tune-throughput-and-queues).

`Model::benchmark()` requires concrete `input_specs()` dimensions. If an input shape is dynamic or non-concrete, the benchmark fails clearly instead of guessing a shape.

## Source Files
- C++: `tutorials/003_benchmark_your_model/benchmark_your_model.cpp`
- Python: `tutorials/003_benchmark_your_model/benchmark_your_model.py`
