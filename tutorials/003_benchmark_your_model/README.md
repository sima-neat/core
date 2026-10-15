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

### Advanced: per-layer stats {#step-read-layer-stats}

Layer profiling is an advanced opt-in option. `BenchmarkOptions.include_layer_stats` defaults
to `false`; ordinary benchmarks keep the existing headline measurements and do not run a layer
profiling pass. Add `--include-layer-stats` to either tutorial command to request it, or enable
it through `BenchmarkOptions`:

```cpp
simaai::neat::BenchmarkOptions options;
options.include_layer_stats = true;
auto report = model.benchmark(options);
```

```python
options = pyneat.BenchmarkOptions()
options.include_layer_stats = True
report = model.benchmark(options)
```

Iterate over `report.layer_stats` to print each compiled layer's `stage_index`, `name`, and
`total_execution_time_us`. The examples retain compiler order and zero-time no-op entries.
`stage_index` follows zero-based executable stage order, including A65 stages.

Layer collection runs after the latency and throughput runners have closed. NEAT reuses the
extracted MPK, selects each MLA ELF from the manifest, and pairs it with `<elf_stem>_stats.yaml`.
It copies the input YAML unchanged into a temporary directory and runs
`mla-rt --connect mip,ma -vs -N --per-layer-stats <copied_yaml> <elf>`. The generated
`_output.yaml` is validated and read into the report, then the temporary files are removed.

Use an idle devkit with no other active MLA workloads: this command accesses MLA directly.
`mla-rt` must be installed and available in `PATH`. Profiling uses its default input buffers
and has a 120-second timeout per MLA stage. With the default `false`, `report.layer_stats` is
empty and the tutorial prints only the headline values. With the advanced flag, output looks
like this (illustrative values):

```text
Layer stats (execution time in microseconds):
  stage_index=0 name=MLA_0/conv total_execution_time_us=161.95
  stage_index=0 name=MLA_0/noop total_execution_time_us=0
  stage_index=2 name=MLA_1/conv total_execution_time_us=42.5
```

Profiling requires MLA-RT with the MIP-to-kernel handoff fix. Verified on eLxr B2121
with MLA-RT `2.2.0~git202610140145.603a8c6-2121` (commit `603a8c6`). With that fix,
you can resume ordinary inference immediately after `benchmark()` returns; no manual reset
or delay is needed. Older builds can leave the hardware in a state that causes subsequent
kernel-driven inference to fail.

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
Telemetry:     Sentinel, 15 samples, trace 20261014T144839.984Z-neat-benchmark-20261014t144839-905z
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

## Save the report

Call `report.save("results/model/yolo_benchmark.yaml")` in C++ or Python to write YAML to that
exact filename. Missing parent directories are created; an existing file is atomically replaced.
The returned path is the path you supplied. The tutorial accepts the same path through `--report`.
Saving is explicit; `benchmark()` does not create a report file automatically.

Saved reports also preserve latency percentiles, throughput-window metadata, and Sentinel telemetry.

The YAML contains `latency_ms`, `fps`, `avg_power_watts`, `energy_joules`, and `layer_stats`.
Each `LayerStats` record has `name`, `stage_index`, and `total_execution_time_us` (hardware
`run_time` in microseconds). Compiler-generated MLA names may fuse or split original ONNX
operators. `stage_index` is zero-based model execution order across MLA and A65 executable
stages, not archive file order. A single-stage model uses `0`; MLA–A65–MLA uses `0` and `2`
for the MLA layer records. Zero-time no-op entries are retained. Layer profiling and the headline measurements
use separate passes and inputs, so their times should not be equated.

Missing profiling files, process failures, timeouts, and invalid generated stats raise an
exception. The compiler's input YAML is preserved, and temporary profiling files are cleaned
up on success and failure.

## Source Files

- C++: `tutorials/003_benchmark_your_model/benchmark_your_model.cpp`
- Python: `tutorials/003_benchmark_your_model/benchmark_your_model.py`
