# 003 Benchmark your model

## Metadata
| Field | Value |
| --- | --- |
| Category | Models & Inference |
| Difficulty | Beginner |
| Estimated Read Time | 5 minutes |
| Model | resnet_50 |
| Labels | benchmark, synthetic, latency, throughput, power |

## Concept

Measure a compiled model's latency, throughput, power, and energy with `model.benchmark()`. The benchmark creates synthetic inputs, so you do not need an image or dataset.

## Walkthrough

### Load the model {#step-load-model}

Load your compiled `.tar.gz` archive with `simaai::neat::Model` in C++ or `pyneat.Model` in Python. The model must declare concrete input dimensions in `input_specs()`.

### Run the benchmark {#step-run-benchmark}

Call `model.benchmark()` for the default settings, or `model.benchmark(100)` to choose the sample count. The API warms up the model, measures sequential latency and asynchronous throughput, prints a summary, and returns a `BenchmarkReport`.

More samples can make throughput and power measurements steadier.

### Read the report {#step-read-report}

| Field | Meaning | Unit |
| --- | --- | --- |
| `latency_ms` | Mean latency after warmup | ms |
| `fps` | Logical inference throughput | inferences/s |
| `avg_power_watts` | Average board power during the throughput measurement | W |
| `energy_joules` | Total energy during the throughput measurement | J |

Power and energy stay zero if board power telemetry is unavailable.

In your Python code, you can also save the report:

```python
model = pyneat.Model("model.tar.gz")
report = model.benchmark()
report.save_json("benchmark.json")
```

Use `report.to_json()` to get JSON text, or `print(report)` to display the report again. The JSON marks unavailable measurements explicitly instead of treating them as valid zeros.

## Run

Run the Python and prebuilt C++ examples from the Neat install root, which contains `share/` and `lib/`. Run the source build commands from the repository root. Both examples print the benchmark summary and then the returned report fields.

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

Example report fields, with illustrative numbers:

```text
report_latency_ms=12.4
report_fps=80.6
report_avg_power_watts=2.3
report_energy_joules=2.8
```

For a custom C++ project, see [How to Run Tutorials](/tutorials#compile-a-copy-yourself).

## In Practice

Compare results using the same model settings and device. Record the model archive and sample count with saved reports.

Synthetic inputs do not represent camera timing or data-dependent processing. Measure your real application pipeline separately. For asynchronous queue tuning, see [Tune Throughput and Queue Depth](/tutorials/tune-throughput-and-queues).

For BoxDecode source-image dimensions and other benchmark options, see [Measure model execution](/develop-apps/development-workflow/model#measure-model-execution).

## Source Files
- C++: `tutorials/003_benchmark_your_model/benchmark_your_model.cpp`
- Python: `tutorials/003_benchmark_your_model/benchmark_your_model.py`
