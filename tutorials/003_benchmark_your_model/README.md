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

Call `benchmark(samples)`. The API warms up the model, measures single-flight latency, then measures throughput in a separate async push/pull pass, prints a summary to stdout, and returns the same headline values in a `BenchmarkReport`.

Each pass runs until it has measured at least `samples` inputs and 10 seconds have passed, so the throughput pass's power and temperature readings cover several samples. Set `BenchmarkOptions::min_duration_ms` to change that floor, or to 0 for a quick smoke check that stops each pass after `samples` inputs.

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

Power comes from Sentinel, the board telemetry daemon that the Neat installer sets up. `report.telemetry` also carries rail power, die and board temperatures, CPU utilization, and MLA memory for the throughput pass; see [Measure model execution](/develop-apps/development-workflow/model#power-and-board-telemetry) for each field. If Sentinel is not running, the benchmark still reports latency and throughput, leaves the power fields at zero, and sets `report.telemetry.status` to the reason.

### Advanced: MLA layer and stage stats {#step-read-mla-stats}

Two opt-in options run `mla-rt` after the benchmark to time the compiled MLA code. Both are
disabled by default and need an idle devkit with no other MLA workloads. Add
`--include-mla-layer-stats` or `--include-mla-stage-stats` to either tutorial command, or enable them
through the API:

```cpp
simaai::neat::BenchmarkOptions options;
options.include_mla_layer_stats = true;
options.include_mla_stage_stats = true;
auto report = model.benchmark(options);
```

```python
options = pyneat.BenchmarkOptions()
options.include_mla_layer_stats = True
options.include_mla_stage_stats = True
report = model.benchmark(options)
```

Read `report.mla_layer_stats.layers` for each compiled MLA layer's zero-based `stage_index`, `name`,
and `latency_us`: the hardware run time from one profiling pass that runs that layer on its own.
Zero-time no-op entries are retained. Layer latencies do not add up to the stage latency or the
headline latency.

Read `report.mla_stage_stats.stages` for each MLA stage's `stage_index` and `latency_us`: the mean
hardware time of 200 uninterrupted runs of that stage. Stage latency includes MLA computation
and the program's DRAM reads and writes. It excludes model loading, host and EV74 processing,
and pipeline overhead.

If `mla-rt` is missing, fails, or times out, `benchmark()` still returns the headline measurements,
leaves the list empty, and sets `report.mla_layer_stats.status` or `report.mla_stage_stats.status`
to the reason. Each status is `"not requested"` or `"ok"` otherwise. After a timed-out or crashed
`mla-rt` run, MLA may still hold unfinished work, so later profiling in the same process is skipped
with that reason. Keep the devkit idle until MLA recovery is verified.

### Save the report {#step-save-report}

Call `report.save(path)` to write the complete report as YAML or JSON, including MLA layer and
stage stats when requested:

```cpp
auto path = report.save("results/model/yolo_benchmark.yaml");
auto json_path = report.save("results/model/yolo_benchmark.json");
```

```python
path = report.save("results/model/yolo_benchmark.yaml")
json_path = report.save("results/model/yolo_benchmark.json")
```

The `.json` extension selects JSON; `.yaml` and all other endings use YAML. The filename is preserved.

Give a file path including the filename. Missing parent directories are created, an existing
file is replaced, and the written path is returned. Saving is explicit; `benchmark()` does
not create a report file automatically.

For either tutorial command, append `--report results/model/yolo_benchmark.yaml`. Combine
it with `--include-mla-layer-stats` or `--include-mla-stage-stats` to save the headline measurements
together with the MLA timings.

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
./build.sh --all --no-doc
./build/tutorials/tutorial_003_benchmark_your_model \
  --model /tmp/resnet_50.tar.gz --samples 100
```

Expected output (exact numbers depend on the model, board, and current load; the C++ build also prints the trailing `[OK]` line):

```text
NEAT Benchmark
Input: synthetic
Samples: 100
Compiled batch: 1
Measurement detail: e2e+throughput-only
Latency: 1.97 ms
FPS:     1457.04 inferences/s
Batch throughput: 1457.04 batches/s
Throughput window: 30.09 s, 43849 inputs
Power:         13.19 W avg, 13.81 W peak (SoM)
MLA power:     3.05 W avg, 3.25 W max
Energy:        0.91 J for 100 inputs
Die temp:      46.34 °C avg, 47.56 °C max
CPU usage:     12.38 % avg, 13.22 % max
MLA memory:    29.65 MiB
CMA used:      6035.02 MiB avg, 6044.38 MiB max
Linux memory:  682.88 MiB avg, 697.41 MiB max
Telemetry:     Sentinel main:766066b74f49, 15 samples
report_latency_ms=1.96852
report_fps=1457.04
report_avg_power_watts=13.1875
report_energy_joules=0.905085
report_peak_power_watts=13.8125
report_throughput_duration_ms=30094.5
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
