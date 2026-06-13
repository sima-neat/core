# Benchmark Measurement Architecture Implementation Plan

This document is the implementation contract for upgrading Neat benchmark and Graph measurement tooling.

The goal is ambitious but surgical: build a trustworthy latency/throughput measurement foundation without doing a broad, speculative runtime rewrite.

## Grounding / external references

These references shape the design:

- OpenTelemetry traces model work using trace/span/parent relationships. This validates the long-term direction of stable lineage identity, but we will not implement a full OTel-style tracing system in this patch.
  - https://opentelemetry.io/docs/concepts/signals/traces/
- GStreamer latency tracing is event-based and explicit about source/sink/pipeline/element measurement semantics. This validates explicit event boundaries and declared latency semantics.
  - https://gstreamer.freedesktop.org/documentation/coretracers/latency.html
- NVIDIA Triton Perf Analyzer reports throughput and latency from explicit measurement windows and metric definitions. This validates numerator/denominator/window reporting and same-window measurement contracts.
  - https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/docs/measurements_metrics.html
- MLPerf Inference separates benchmark scenarios because latency/throughput semantics depend on workload shape. This validates separating model benchmarks from Graph/application benchmarks.
  - https://docs.mlcommons.org/inference/index_gh/

## Non-negotiable design principles

1. Every reported metric must say what it measured.
   - No bare `latency` without start/stop semantics.
   - No bare `FPS` without numerator, denominator, and measurement window.

2. Public benchmark values must come from the same measured window that produced them.
   - Do not compare against a separate later single-flight or burst oracle run.
   - Separate windows are useful diagnostics, not acceptance truth.

3. Graph E2E latency is queue-inclusive unless explicitly proven otherwise.
   - Current graph timing is public graph push/admission to public-output pull.
   - If the timestamp is captured before `graph_push()`, name the metric honestly as public-push-to-public-pull residency.
   - In queued/burst workloads it includes queue residency and application pull timing.
   - It must not be presented as model service latency.

4. Correlation quality must be visible and separate from workload completeness.
   - If timing count does not match output count, mark correlation partial.
   - If samples are unkeyed or misses occur, warn and mark correlation unreliable.
   - If drops occur, mark survivor bias/workload incompleteness, but do not invalidate correctly correlated latency samples.

5. Model and Graph benchmarks are separate contracts.
   - `Model::benchmark()` is synthetic controlled model-route benchmarking.
   - `Run::start_measurement()` / Graph export is application-window measurement.

6. The macro `LatencyTracker` remains diagnostic-only.
   - It is host-observed, tap-dependent, queue-inclusive, and weakly keyed.
   - Do not promote it to canonical runtime telemetry.

---

# Current worktree caution

The current worktree contains unrelated dirty files. Benchmark work must not accidentally absorb them.

Do not touch or justify with this plan unless explicitly requested:

- Python `Format` API changes:
  - `python/src/module.cpp`
  - `python/tests/test_api_surface.py`
  - `python/tests/test_format_runtime.py`
  - `python/tests/test_real_model_fixtures.py`
- Tensor segment/output-override fixes:
  - `src/pipeline/internal/OutputTensorOverride.h`
  - `src/pipeline/internal/SimaMemApi.h`
  - `src/pipeline/internal/TensorUtil.h`
  - `src/pipeline/tensor/TensorUtil.cpp`
  - `tests/unit_testing/unit_tensor_set_meta_route_contract_test.cpp`
- Temporary/diagnostic files:
  - `tests/unit_testing/unit_move_semantics_proof_test.cpp`
  - `core_issue_332/`

This plan touches only benchmark/measurement/report/test/docs files listed below.

Before implementation and before final summary, run:

```bash
git diff --name-only
```

and explicitly classify changed files as:

- benchmark-plan/measurement changes,
- pre-existing unrelated dirty files,
- accidental changes to revert.

The benchmark patch must not include accidental unrelated files.

---

# Freeze-critical implementation scope

This section is the release-freeze contract. The broader patch list below remains useful, but the implementation before freeze must prioritize the following items.

## Must land before freeze

1. **Remove flaky YOLO independent-oracle comparisons.**
   - `yolov8_model_benchmark_metrics_test.cpp` must become a real-model benchmark smoke/health test.
   - Do not compare `Model::benchmark()` against separate single-flight or burst wall-clock windows.

2. **Add a same-window `Model::benchmark()` projection contract.**
   - Add an internal helper that builds `BenchmarkReport` from the exact latency and throughput `MeasureReport`s produced by `Model::benchmark()`.
   - Unit-test the helper directly.

3. **Stamp stable public graph ingress sequence identity.**
   - Add a run-local sequence counter.
   - Fill `orig_input_seq/input_seq` when missing, even if `frame_id` exists.
   - Use the stamped sample for both graph push and graph timing entry.

4. **Add measurement quality/status without public ABI breakage.**
   - API/ABI is a release concern. Do not grow public `MeasureReport` unless ABI owners explicitly approve.
   - Prefer computing quality in internal helper code and exporting it through JSON/text/tooling.
   - If adding public fields is approved, keep them additive and documented.

5. **Add canonical JSON semantics for graph E2E and throughput.**
   - Graph E2E must declare unit, status, semantics, source, correlation reliability, and survivor bias.
   - Throughput must declare numerator, denominator, logical batch size, and logical inferences/s.
   - Keep flat fields as compatibility aliases.

6. **Add minimal zero-copy/materialization semantics.**
   - We claim zero-copy benchmarks, so reports must say what output memory/materialization semantics were in effect.
   - Full copy/map timing can come later, but metadata-only materialization status is freeze-critical.

7. **Update visualizer labels.**
   - Visualizer output is important for customer/internal interpretation.
   - It must not label queue-inclusive E2E as generic latency.
   - Unavailable metrics must render as N/A/status, not measured zero.

## Defer until after freeze

- Full OpenTelemetry-style trace/span lineage.
- Full materialization copy/map timing and byte accounting.
- Macro `LatencyTracker` architectural promotion. It remains diagnostic-only.
- Broad schema-version bump unless required.
- Removing flat JSON compatibility fields.

## ABI-safe rule for this release

Because API/ABI is a concern, default to this layering:

```text
Public structs remain layout-compatible where possible.
Internal helpers compute measurement quality/materialization metadata.
JSON/text/tooling expose the new semantics.
Public fields are added only with explicit ABI approval.
```

Concretely, prefer helper functions like:

```cpp
namespace simaai::neat::runtime {
struct MeasureQualityView {
  std::string end_to_end_status;
  bool end_to_end_correlation_reliable = false;
  bool survivor_biased = false;
  std::vector<std::string> warnings;
};

MeasureQualityView compute_measure_quality(const MeasureReport& report);
}
```

over adding fields directly to `MeasureReport`.

If `MeasureReport` fields are added, update this plan and call out ABI acceptance explicitly.

---

# Patch 1: Stable graph ingress identity

## Problem

Graph sample timing currently keys by:

```text
stream_id + orig_input_seq/input_seq/frame_id
```

If public graph ingress receives `TensorList`, `cv::Mat`, or an unkeyed `Sample`, timing may be unkeyed or output correlation may miss. Even when `frame_id` exists, it can repeat across looping sources, reconnects, generated test streams, or fanout/reuse. Relying on `frame_id` as the primary timing key is therefore not strong enough for benchmark correlation.

## Code changes

### File: `src/pipeline/runtime/RunCore.h`

Add a run-local public graph input sequence counter to `runtime::RunCore`.

```cpp
std::atomic<std::int64_t> next_public_graph_input_seq{0};
```

Placement: near other runtime counters such as `inputs_enqueued`, `inputs_pushed`, or near graph sample timing fields.

Rationale: the sequence is internal to one `Run`; it does not change public API or ABI-visible `Sample` layout.

### File: `src/pipeline/runtime/RunPush.cpp`

Add an internal helper near `push_graph_samples_to_endpoint`:

```cpp
Sample stamp_public_graph_ingress_sample(runtime::RunCore& core, const Sample& in) {
  Sample out = in;

  if (out.stream_id.empty()) {
    out.stream_id = "default";
  }

  // Always prefer a run-local public-ingress sequence over frame_id fallback when sequence fields
  // are absent. frame_id can repeat across RTSP reconnects, looping sources, generated tests,
  // and fanout/reuse; seq fields are the stable correlation key for this Run.
  if (out.orig_input_seq < 0 && out.input_seq < 0) {
    const auto seq = core.next_public_graph_input_seq.fetch_add(1, std::memory_order_relaxed);
    out.orig_input_seq = seq;
    out.input_seq = seq;
  } else if (out.orig_input_seq < 0 && out.input_seq >= 0) {
    out.orig_input_seq = out.input_seq;
  } else if (out.input_seq < 0 && out.orig_input_seq >= 0) {
    out.input_seq = out.orig_input_seq;
  }

  return out;
}
```

Important: do not require `frame_id < 0` before stamping sequence fields. If seq fields are absent, stamp them even when `frame_id` exists.

Then change `push_graph_samples_to_endpoint(...)` from using `msg` directly to using stamped sample consistently:

Current shape:

```cpp
for (const auto& msg : msgs) {
  core.inputs_enqueued.fetch_add(1, std::memory_order_relaxed);
  const auto entry_at = std::chrono::steady_clock::now();
  runtime::trace_graph_message_event(..., msg, endpoint_name);
  if (!core.graph_push(..., msg, ...)) return false;
  core.record_graph_sample_entry(endpoint_name, msg, entry_at);
  core.inputs_pushed.fetch_add(1, std::memory_order_relaxed);
}
```

Desired shape:

```cpp
for (const auto& msg : msgs) {
  Sample stamped = stamp_public_graph_ingress_sample(core, msg);

  core.inputs_enqueued.fetch_add(1, std::memory_order_relaxed);
  const auto entry_at = std::chrono::steady_clock::now();

  runtime::trace_graph_message_event(..., stamped, endpoint_name);

  if (!core.graph_push(endpoint.node, endpoint.port,
                       endpoint.port != simaai::neat::graph::kInvalidPort,
                       stamped,
                       graph_router_options_for_push(core, block))) {
    return false;
  }

  core.record_graph_sample_entry(endpoint_name, stamped, entry_at);
  core.inputs_pushed.fetch_add(1, std::memory_order_relaxed);
}
```

Important: use the stamped sample for both `graph_push` and `record_graph_sample_entry`. If entry and output use different metadata, correlation remains fragile.

### File: `src/pipeline/runtime/GraphSampleTiming.cpp`

No broad rewrite. Keep current key selection:

```cpp
orig_input_seq -> input_seq -> frame_id
```

But after Patch 1, public graph ingress should almost always have `orig_input_seq`/`input_seq`.

Optional small improvement: when key is missing, include a more actionable debug message under `SIMA_SAMPLE_TIMING_DEBUG`, but do not add logging by default.

## Tests

### File: `tests/unit_testing/unit_benchmark_measurement_contract_test.cpp`

Keep or strengthen existing test:

```cpp
benchmark_style_tensorlist_push_must_collect_latency_for_every_output()
```

Required assertions:

```cpp
report.counters.outputs_pulled == kSamples
report.end_to_end.count == kSamples
report.graph_sample_timing_unkeyed == 0
report.graph_sample_timing_misses == 0
report.latency_samples_collected == true
```

Add a new explicit test if not already covered:

```cpp
unkeyed_public_graph_sample_gets_run_local_identity()
```

Test shape:

1. Build a small graph-backed `Run`.
2. Push `TensorList` or a `Sample` with no `frame_id`, no `input_seq`, no `orig_input_seq`, no `stream_id`.
3. Pull output inside one `MeasureScope`.
4. Assert exactly one latency sample, zero unkeyed, zero misses.

---

# Patch 2: Honest timing boundaries and measurement quality/status

## Problem

A report can currently contain latency samples while correlation is incomplete. That makes averages look authoritative when they are partial. Also, if the timestamp is taken before `graph_push()` but the metric is described as graph-entry latency, the metric boundary is misleading: it may include public push blocking/admission time.

## Code changes

### File: `include/pipeline/Run.h`

ABI-safe default: do not grow `MeasureReport` unless release ABI owners approve. Keep `MeasureReport` layout stable when possible and compute quality through internal helper functions.

Preferred helper shape in an internal/runtime header or `.cpp` local helper:

```cpp
struct MeasureQualityView {
  std::string end_to_end_status = "unavailable"; // collected | partial | unavailable
  bool end_to_end_correlation_reliable = false;
  bool survivor_biased = false;
  std::vector<std::string> warnings;
};

MeasureQualityView compute_measure_quality(const MeasureReport& report);
```

Only if ABI approval is explicit, extend `MeasureReport` additively with:

```cpp
std::string end_to_end_status = "unavailable"; // collected | partial | unavailable
bool end_to_end_correlation_reliable = false;
bool survivor_biased = false;             // true when drops mean latency covers surviving outputs only
```

Keep using existing `MeasureReport::warnings` when fields are added. Do not add a second public warnings vector unless needed.

### File: `src/pipeline/runtime/RunMeasure.cpp`

Add a helper near report construction helpers:

```cpp
void finalize_end_to_end_quality(MeasureReport& report) {
  const auto pulled = report.counters.outputs_pulled;
  const auto samples = report.end_to_end.count;

  if (samples == 0) {
    report.end_to_end_status = pulled == 0 ? "unavailable_no_outputs" : "unavailable_no_samples";
    report.end_to_end_correlation_reliable = false;
    if (pulled > 0) {
      report.warnings.push_back("End-to-end timing unavailable: outputs were pulled but no latency samples were correlated");
    }
    return;
  }

  bool correlation_reliable = true;

  if (samples != pulled) {
    correlation_reliable = false;
    report.warnings.push_back(
        "End-to-end timing partial: collected " + std::to_string(samples) +
        " latency samples for " + std::to_string(pulled) + " pulled outputs");
  }
  if (report.graph_sample_timing_unkeyed > 0) {
    correlation_reliable = false;
    report.warnings.push_back(
        "Graph sample timing had " + std::to_string(report.graph_sample_timing_unkeyed) +
        " unkeyed graph entries");
  }
  if (report.graph_sample_timing_misses > 0) {
    correlation_reliable = false;
    report.warnings.push_back(
        "Graph sample timing had " + std::to_string(report.graph_sample_timing_misses) +
        " output correlation misses");
  }

  report.survivor_biased = report.counters.inputs_dropped > 0 || report.counters.outputs_dropped > 0;
  if (report.survivor_biased) {
    report.warnings.push_back(
        "Measured window included dropped inputs/outputs; latency describes surviving correlated outputs only");
  }

  report.end_to_end_status = correlation_reliable ? "collected" : "partial";
  report.end_to_end_correlation_reliable = correlation_reliable;
}
```

Important: drops should set `survivor_biased`, but should not by themselves make correctly correlated surviving-output latency unreliable.

Call it in `MeasureScope::stop()` after:

```cpp
report.end_to_end = summarize_samples(...);
report.frame_gap = summarize_samples(...);
report.latency_samples_collected = ...;
report.graph_sample_timing_unkeyed = ...;
report.graph_sample_timing_misses = ...;
```

and before returning `report`.

### File: `src/pipeline/runtime/RunMeasure.cpp` text output

In `MeasureReport::to_text()`, print:

```text
E2E status            : collected|partial|unavailable
E2E correlation       : reliable|partial
Survivor biased       : yes|no
```

When warnings exist, print them under a `Warnings` section.

### File: `src/pipeline/runtime/GraphMetricJson.cpp`

Change graph E2E JSON helper to expose status/reliability.

Preferred implementation: change signature from:

```cpp
json graph_e2e_json(const MeasureLatencyStats& stats, bool graph_backed,
                    std::string_view empty_status);
```

to:

```cpp
json graph_e2e_json(const MeasureReport& report, bool graph_backed,
                    std::string_view empty_status);
```

Then use:

```cpp
const auto& stats = report.end_to_end;
const std::string status = !report.end_to_end_status.empty()
                             ? report.end_to_end_status
                             : (stats.count > 0 ? "collected" : std::string(empty_status));
```

Emit:

```json
{
  "available": stats.count > 0,
  "status": status,
  "correlation_reliable": report.end_to_end_correlation_reliable,
  "survivor_biased": report.survivor_biased,
  "unit": "milliseconds",
  "scope": "graph_application",
  "source": "runtime_graph_sample_timing",
  "semantics": "queue_inclusive_public_graph_push_to_public_output_pull",
  "interpretation": "Single-flight loops approximate per-input latency; async burst/queued windows include queue wait and should be presented as queue residency, not standalone latency.",
  "count": ...,
  "avg_ms": ...,
  "p50_ms": ...,
  "p95_ms": ...,
  "warnings": report.warnings
}
```

### File: `src/pipeline/runtime/GraphMetricJson.h`

Update declaration accordingly.

### File: `src/pipeline/runtime/GraphRunExport.cpp`

Update caller from:

```cpp
run_json["graph_e2e_latency_ms"] = runtime::graph_e2e_json(
    report.end_to_end, graph_backed, ...);
```

to:

```cpp
run_json["graph_e2e_latency_ms"] = runtime::graph_e2e_json(
    report, graph_backed, ...);
```

Also export in `measurement` block:

```json
"end_to_end_status": report.end_to_end_status,
"end_to_end_correlation_reliable": report.end_to_end_correlation_reliable,
"survivor_biased": report.survivor_biased
```

## Tests

### File: `tests/unit_testing/unit_benchmark_measurement_contract_test.cpp`

Add a synthetic `MeasureReport` quality unit if possible without device runtime, or validate with a measured graph:

- Good case:
  - `end_to_end.count == outputs_pulled`
  - unkeyed/misses zero
  - status `collected`
  - reliable true

- Partial case:
  - intentionally unkeyed/missing correlation or construct report through helper if helper is accessible
  - status `partial`
  - reliable false
  - warning contains `partial` or `misses`

If direct helper is not public, add a test via report JSON export in `phaseA4_run_export_test.cpp`.

---

# Patch 3: Canonical metric envelopes and throughput auditability

## Problem

Some export fields are flat compatibility aliases. They are useful but not self-describing enough for customer benchmark tooling.

## Code changes

### File: `src/pipeline/runtime/GraphMetricJson.cpp`

Current `throughput_json(...)` already emits many useful fields. Extend it to include:

```json
"batches_per_s": outputs_per_s,
"throughput_batches_per_s": outputs_per_s,
"status": valid_window ? "collected" : "invalid_window",
"available": valid_window,
"scope": "measured_window" or passed-in scope,
"warnings": []
```

If scope is needed, update function signature:

```cpp
json throughput_json(std::uint64_t output_pulls, double elapsed_s,
                     int logical_batch_size, std::string_view scope = "measured_window");
```

Emit clear semantics:

```json
"semantics": "public_output_pulls_per_second",
"multi_output_semantics": "each successful public pull increments outputs_pulled once"
```

Do not delete flat aliases yet:

```json
"throughput_fps"
"outputs_per_s"
"throughput_batches_per_s"
"throughput_inferences_per_s"
```

But ensure they match nested values. Keep schema-v1 validation backward-compatible where necessary: old v1 payloads may omit nested `throughput`, but newly generated golden/export tests must require it. If we choose to require nested `throughput` for all v1 payloads, explicitly treat that as a schema tightening decision.

### File: `src/pipeline/runtime/GraphRunExport.cpp`

For measured-window export:

```cpp
const json throughput = runtime::throughput_json(
    report.counters.outputs_pulled,
    report.elapsed_s,
    report.options.logical_batch_size,
    "measured_window");
```

Set compatibility aliases from nested values only:

```cpp
graph_metrics["throughput_fps"] = throughput.value("outputs_per_s", 0.0);
graph_metrics["outputs_per_s"] = throughput.value("outputs_per_s", 0.0);
graph_metrics["throughput_batches_per_s"] = throughput.value("batches_per_s", 0.0);
graph_metrics["throughput_inferences_per_s"] = throughput.value("logical_inferences_per_s", 0.0);
```

For run-lifetime export, make sure `graph_metrics.throughput` is also present, not only flat `throughput_fps`.

### File: `tests/perf/tools/graph_run_schema.py`

Add or strengthen validators:

```python
def validate_throughput(value: Any, context: str) -> None:
    data = _mapping(value, context)
    _require(data, (
        "unit",
        "semantics",
        "numerator_counter",
        "numerator_value",
        "denominator",
        "denominator_seconds",
        "outputs_per_s",
        "logical_batch_size",
        "logical_inferences_per_s",
        "multi_output_semantics",
        "available",
        "status",
    ), context)
```

In `validate_graph_metrics(...)`, require nested throughput:

```python
_require(data, ("measurement_scope", "throughput_counting", "elapsed_seconds", "throughput_fps", "throughput"), context)
validate_throughput(data["throughput"], f"{context}.throughput")
```

Check alias consistency if present:

```python
if "outputs_per_s" in data:
    _number(data["outputs_per_s"], ...)
```

Do not reject old files outside tests unless schema version is bumped. For schema v1 compatibility, validator may allow missing nested throughput for old exports, but new golden tests must require it.

### File: `tests/perf/tools/test_graph_run_schema.py`

Update `valid_payload()` graph metrics to include nested `throughput`.

Add negative tests:

```python
def test_graph_metrics_requires_throughput_numerator_denominator(self):
    bad = valid_payload()
    del bad["run"]["graph_metrics"]["throughput"]["numerator_counter"]
    with self.assertRaises(schema.SchemaError):
        schema.validate_graph_run(bad)
```

```python
def test_graph_e2e_latency_requires_units_and_semantics(self):
    bad = valid_payload()
    del bad["run"]["graph_e2e_latency_ms"]["unit"]
    with self.assertRaises(schema.SchemaError):
        schema.validate_graph_run(bad)
```

### File: `tests/graph_migration/unified/phaseA4_run_export_test.cpp`

For both `run_to_json(run, opt, ...)` and `run_to_json(run, measured, opt, ...)`, assert:

```cpp
const auto& gm = json.at("run").at("graph_metrics");
require(gm.contains("throughput"), "graph metrics should include throughput object");
require(gm.at("throughput").at("numerator_counter").get<std::string>() == "outputs_pulled", ...);
require(gm.at("throughput").at("denominator").get<std::string>() == "elapsed_seconds", ...);
require(gm.at("throughput_fps").get<double>() ==
        gm.at("throughput").at("outputs_per_s").get<double>(), ...);
```

For measured export, assert:

```cpp
require(run.at("graph_e2e_latency_ms").at("unit") == "milliseconds", ...);
require(run.at("graph_e2e_latency_ms").at("semantics").find("queue_inclusive") != std::string::npos, ...);
require(run.at("graph_e2e_latency_ms").contains("status"), ...);
require(run.at("graph_e2e_latency_ms").contains("correlation_reliable"), ...);
require(run.at("graph_e2e_latency_ms").contains("survivor_biased"), ...);
```

---

# Patch 3.5: Minimal output materialization semantics (freeze-critical because zero-copy benchmarks are claimed)

## Problem

A benchmark can look fast because outputs were pulled zero-copy but never CPU-materialized. Full materialization timing can come later, but the report must at least say what output memory mode and materialization semantics were in effect. This is freeze-critical because we claim zero-copy benchmarks. It is also important because output segment/materialization has been a correctness-sensitive area.

## Code changes

### File: `include/pipeline/Run.h`

ABI-safe default: avoid adding public `MeasureReport` fields unless approved. Prefer computing output materialization metadata in JSON export from available `RunOptions`/pipeline options.

If ABI approval exists, add a small additive reporting struct or fields to `MeasureReport`:

```cpp
std::string output_memory_mode;              // owned | zero-copy | unknown
std::string materialization_semantics;       // owned_output_copy | lazy_zero_copy | not_tracked
bool materialization_timing_available = false;
```

Set these from `RunOptions::output_memory` / pipeline output options when the measurement starts or stops. Do not attempt full copy timing in this patch unless the hooks already exist.

### File: `src/pipeline/runtime/GraphRunExport.cpp`

Emit under `measurement` or `graph_metrics`:

```json
"output_materialization": {
  "available": true,
  "status": "metadata_only",
  "output_memory_mode": "owned|zero-copy|unknown",
  "semantics": "owned_output_copy|lazy_zero_copy|not_tracked",
  "timing_available": false,
  "note": "Timing of lazy map/clone materialization is not tracked in this report."
}
```

## Tests

Add export/schema assertions that the block exists and has `status`, `output_memory_mode`, and `semantics`. Do not assert copy timings yet.

---

# Patch 4: Model benchmark projection contract

## Problem

`Model::benchmark()` currently mixes implementation and public result projection. Tests have used source-string checks or separate oracle runs. Both are weak.

## Code changes

### File: `src/model/internal/ModelInternal.h`

Expose an internal helper for tests under `SIMA_NEAT_INTERNAL`:

```cpp
BenchmarkReport build_benchmark_report_from_measurements(const MeasureReport& latency_report,
                                                          const MeasureReport& throughput_report,
                                                          int expected_samples);
```

Namespace:

```cpp
namespace simaai::neat::internal { ... }
```

### File: `src/model/Model.cpp`

Implement helper near `Model::benchmark()`:

```cpp
BenchmarkReport internal::build_benchmark_report_from_measurements(
    const MeasureReport& latency_report,
    const MeasureReport& throughput_report,
    int expected_samples) {
  if (expected_samples <= 0) {
    throw std::runtime_error("Model::benchmark: expected_samples must be > 0");
  }
  if (latency_report.counters.outputs_pulled != static_cast<std::uint64_t>(expected_samples)) {
    throw std::runtime_error("Model::benchmark: latency measured output count mismatch");
  }
  if (latency_report.end_to_end.count != static_cast<std::size_t>(expected_samples)) {
    throw std::runtime_error("Model::benchmark: latency measurement did not collect one sample per output");
  }
  if (!latency_report.end_to_end_correlation_reliable && !latency_report.end_to_end_status.empty()) {
    throw std::runtime_error("Model::benchmark: latency measurement correlation is partial/unreliable");
  }
  if (latency_report.survivor_biased) {
    throw std::runtime_error("Model::benchmark: latency measurement is survivor-biased due to drops");
  }
  if (latency_report.options.logical_batch_size > 0 && throughput_report.options.logical_batch_size > 0 &&
      latency_report.options.logical_batch_size != throughput_report.options.logical_batch_size) {
    throw std::runtime_error("Model::benchmark: latency/throughput logical batch sizes disagree");
  }
  const double expected_inferences = throughput_report.throughput_batches_per_s *
      static_cast<double>(std::max(1, throughput_report.options.logical_batch_size));
  if (std::isfinite(expected_inferences) && std::isfinite(throughput_report.throughput_inferences_per_s) &&
      std::abs(expected_inferences - throughput_report.throughput_inferences_per_s) >
          std::max(1e-6, std::abs(expected_inferences) * 1e-6)) {
    throw std::runtime_error("Model::benchmark: throughput inferences/s does not match batches/s * logical_batch_size");
  }
  if (throughput_report.counters.outputs_pulled != static_cast<std::uint64_t>(expected_samples)) {
    throw std::runtime_error("Model::benchmark: throughput measured output count mismatch");
  }

  BenchmarkReport out;
  out.latency_ms = latency_report.end_to_end.avg_ms;
  out.fps = throughput_report.throughput_inferences_per_s;

  if (throughput_report.power.enabled && throughput_report.power.samples > 0) {
    out.avg_power_watts = throughput_report.power.total_avg_watts;
    out.energy_joules = throughput_report.power.energy_joules;
  }
  return out;
}
```

Note: if adding `end_to_end_reliable` in Patch 2, use it. If Patch 4 lands before Patch 2, only validate counts and sample availability, then add reliability once available.

Change `Model::benchmark(int num_samples)`:

1. Store latency `MeasureReport latency_measured`.
2. Store throughput `MeasureReport throughput_measured`.
3. Replace direct assignments:

Current:

```cpp
report.latency_ms = measured.end_to_end.avg_ms;
...
report.fps = measured.throughput_inferences_per_s;
```

Desired:

```cpp
BenchmarkReport report = internal::build_benchmark_report_from_measurements(
    latency_measured, throughput_measured, num_samples);
```

Preserve existing console output:

```text
Latency: ... ms
FPS: ... inferences/s
Batch throughput: ... batches/s
Power: unavailable or values
```

### File: `tests/unit_testing/unit_benchmark_measurement_contract_test.cpp`

Remove brittle source-string guard:

```cpp
model_benchmark_wrapper_must_publish_measured_logical_fps_and_real_latency()
```

Replace with behavioral projection test:

```cpp
void model_benchmark_projection_uses_same_window_reports() {
  MeasureReport latency;
  latency.counters.outputs_pulled = 3;
  latency.outputs = 3;
  latency.end_to_end.count = 3;
  latency.end_to_end.avg_ms = 7.5;
  latency.end_to_end_status = "collected";
  latency.end_to_end_reliable = true;

  MeasureReport throughput;
  throughput.counters.outputs_pulled = 3;
  throughput.outputs = 3;
  throughput.throughput_batches_per_s = 100.0;
  throughput.options.logical_batch_size = 4;
  throughput.throughput_inferences_per_s = 400.0;

  const auto report = simaai::neat::internal::build_benchmark_report_from_measurements(
      latency, throughput, 3);

  require(report.latency_ms == 7.5, ...);
  require(report.fps == 400.0, ...);
}
```

Add negative tests:

```cpp
projection_rejects_missing_latency_samples()
projection_rejects_latency_output_count_mismatch()
projection_rejects_throughput_output_count_mismatch()
projection_rejects_partial_latency_if_reliability_fields_available()
projection_rejects_survivor_biased_latency()
projection_rejects_logical_batch_mismatch()
projection_rejects_inference_throughput_arithmetic_mismatch()
```

### File: `tests/e2e_pipelines/obj_detection/yolov8_model_benchmark_metrics_test.cpp`

Make this a real-device smoke/health test only.

Remove:

- `make_synthetic_tensor(...)`
- `make_synthetic_inputs(...)`
- `make_keyed_sample(...)`
- `make_model_run_options(...)`
- `make_model_measure_options(...)`
- `measure_model_single_flight(...)`
- `measure_model_burst(...)`
- `require_measurement_report_is_complete(...)` if no longer used
- comparisons of `Model::benchmark()` against separate single/burst windows

Keep:

```cpp
const simaai::neat::BenchmarkReport report = model.benchmark(kBenchmarkSamples);
require(std::isfinite(report.latency_ms));
require(report.latency_ms > 0.0);
require(std::isfinite(report.fps));
require(report.fps > 0.0);
require(report.avg_power_watts >= 0.0);
require(report.energy_joules >= 0.0);
```

Add a comment explaining why no separate oracle is used.

---

# Patch 5: Documentation/tutorial wording

## Problem

Benchmark docs must not imply app-level latency or ambiguous FPS.

## Code changes

### File: `tutorials/003_benchmark_your_model/README.md`

Update wording to something like:

```md
`Model::benchmark()` is a synthetic model-route benchmark. It creates deterministic tensors from the model input contract. The headline latency is measured in a warmed-up single-flight loop and is queue-inclusive graph-entry-to-public-pull for that controlled model route. The headline FPS is logical inferences per second from a separate async throughput measurement window. For compiled batch size N, logical inferences/s = batches/s × N.
```

Update sample output to include:

```text
Compiled batch: N
Latency: X ms
FPS: Y inferences/s
Batch throughput: Z batches/s
Power: unavailable
```

Add warning:

```md
For full application performance, use `Run::start_measurement()` or graph-run export. Graph E2E latency is queue-inclusive and includes queue residency and public pull timing in async/burst workloads.
```

### File: `include/model/Model.h`

The existing `BenchmarkReport` comment is close. Ensure it states:

- synthetic inputs
- latency is warmed-up single-flight
- FPS is logical inferences/s from separate async throughput window
- power from throughput window only

No major API change.

### File: `include/pipeline/Run.h`

`MeasureReport` comment should explicitly say:

```text
end_to_end is queue-inclusive graph-entry/public-push to public-pull residency. In async/burst windows this is queue residency/app-visible E2E, not model service latency.
```

---

# Patch 6: Macro `LatencyTracker` labeling only

## Problem

The macro tracker is useful but easy to overinterpret.

## Code changes

### File: `tests/e2e_pipelines/macro/macro_mixed_multistream_multimodel_tput_test.cpp`

In `LatencyTracker::to_json()`, add fields:

```json
"canonical": false,
"semantics": "host_observed_app_tap_queue_inclusive_diagnostic",
"timing_source": "std::chrono::steady_clock at graph Map/output/send taps",
"not_model_service_latency": true,
"overhead_note": "Enabling latency_profile adds Map taps, mutex/hash bookkeeping, and can perturb throughput. Do not compare latency-profile throughput directly with baseline throughput."
```

Keep current stage stats.

Do not add trace/span identity here.

Optional cheap improvements:

- Add `duplicate_key_marks` counter if a key already has the stage set and a later mark arrives.
- Add `unmatched_entries` count in JSON.

Do not turn this into runtime architecture.

---

# Patch 7: Visualizer/tool display (freeze-critical for interpretation)

## Problem

Tools should not display queue-inclusive E2E as generic latency. This is freeze-critical because the visualizer is part of how benchmark results are interpreted.

## Code changes

### File: `tools/visualize_graph_run.py`

Find cards/table labels that render:

```text
Latency
E2E latency
Throughput
```

Change labels to prefer nested metric semantics:

```text
Graph E2E residency (queue-inclusive)
Throughput (public output pulls/s)
Logical inference throughput
```

When a metric has:

```json
"available": false
```

render:

```text
N/A (status/reason)
```

not `0`.

### Test file, if present:

Search:

```bash
find tests -name '*visual*' -o -name '*visualizer*'
```

Add/extend visualizer test to assert output contains:

```text
queue-inclusive
outputs_pulled
elapsed_seconds
N/A
```

---

# Patch 8: Cleanup/deletion

## Delete/weaken now

### File: `tests/e2e_pipelines/obj_detection/yolov8_model_benchmark_metrics_test.cpp`

Delete independent oracle helpers and comparisons as described in Patch 4.

### File: `tests/unit_testing/unit_benchmark_measurement_contract_test.cpp`

Delete source-string guard after projection helper exists.

### File: `tests/unit_testing/unit_measurement_consistency_test.cpp`

Review hard timing assumptions. Keep structural invariants. Weaken or remove host-load-sensitive checks such as:

- inverse throughput within exactly 1 ms of latency
- frame gap within exactly 1 ms
- fixed max push-call timing
- strict element-sum ratios

Only change these if they fail or if they are clearly non-contractual. Do not churn unnecessarily.

## Do not delete now

- Flat JSON compatibility fields.
- Existing `MeasureReport::end_to_end` field.
- Macro `LatencyTracker`.

---

# Validation plan

## Local Python/schema

```bash
python3 tests/perf/tools/test_graph_run_schema.py
```

## C++ focused build targets

Targets to build:

```text
unit_benchmark_measurement_contract_test
graph_migration_phaseA4_run_export_test
yolov8_model_benchmark_metrics_test
```

## DevKit rule

Before running any built binary:

```bash
file <path-to-binary>
```

If it reports aarch64/ARM aarch64, run through DevKit:

```bash
dk /home/docker/sima-cli/tmp/devkit_env_exec.sh <binary> [args...]
```

or existing wrappers:

```bash
tools/run-on-devkit <binary> [args...]
tools/ctest-on-devkit <regex> [extra ctest args...]
```

Do not stop at `Exec format error`; use the DevKit path.

## Expected test behavior

- Unit/schema tests should be deterministic.
- YOLO benchmark test should no longer fail from independent window variance.
- If YOLO benchmark fails, it should indicate real benchmark/runtime/model failure, not oracle mismatch.

---

# Acceptance criteria

Implementation is acceptable only when all of these are true:

1. Public graph ingress samples get stable identity if caller did not provide one.
2. `MeasureReport` distinguishes collected/partial/unavailable E2E timing.
3. JSON graph E2E latency declares unit, scope, source, semantics, status, reliability.
4. Throughput JSON declares numerator, denominator, outputs/s, logical batch size, logical inferences/s.
5. Flat throughput aliases match nested throughput values.
6. `Model::benchmark()` public fields are built from same-window reports via a testable helper.
7. YOLO E2E benchmark test no longer compares independent timing windows.
8. Macro latency profile is labeled diagnostic-only.
9. Docs explain model benchmark vs Graph/app measurement.
10. Unrelated dirty worktree files are not included in the benchmark patch.

---

# Suggested implementation order

1. Patch 1: stable ingress identity.
2. Patch 2: measurement quality/status.
3. Patch 3: JSON envelopes and throughput auditability.
4. Patch 4: model benchmark projection helper and tests.
5. Patch 5: docs/tutorial wording.
6. Patch 6: macro diagnostic labeling.
7. Patch 7: visualizer semantics.
8. Patch 8: cleanup of obsolete/flaky tests.

If time is short, minimum viable high-quality subset is:

1. Stable ingress identity that stamps seq fields even when `frame_id` exists.
2. Honest E2E boundary semantics and measurement quality/status, implemented ABI-safely if needed.
3. Separate correlation reliability from survivor bias.
4. Model benchmark projection helper.
5. Remove YOLO independent oracle comparisons.
6. JSON schema for graph E2E/throughput semantics.
7. Minimal output materialization semantics because zero-copy benchmarks are claimed.
8. Visualizer labeling for queue-inclusive E2E and unavailable metrics.


# Implementation log — 2026-06-12

Started implementation of the freeze-critical subset with ABI caveats:

- Added run-local public graph ingress sequence stamping in `RunCore`/`RunPush` so public graph samples without sequence identity get stable `orig_input_seq`/`input_seq` before graph entry timing.
- Added ABI-safe internal measurement-quality view in `GraphMetricJson` and exported JSON quality fields instead of changing public `MeasureReport` layout.
- Added explicit throughput semantics blocks for run-lifetime and measured-window exports.
- Added metadata-only output materialization reporting so zero-copy/owned benchmark claims are labeled even before copy/map timing exists.
- Added a testable internal `Model::benchmark()` projection helper and rewired `Model::benchmark()` to project public headline metrics only from same-window `MeasureReport`s.
- Removed the fragile YOLO benchmark independent-oracle comparison in favor of real-model health validation plus unit-level measurement contract tests.
- Updated graph-run schema checks and visualizer labels for output-pull throughput, logical inference throughput, queue-inclusive E2E, correlation reliability, and output materialization.

Validation performed:

- `python3 -m py_compile tests/perf/tools/graph_run_schema.py tests/perf/tools/test_graph_run_schema.py tools/visualize_graph_run.py`
- `python3 tests/perf/tools/test_graph_run_schema.py`
- `cmake --build build-codex-graph-aarch64 --target unit_benchmark_measurement_contract_test graph_migration_phaseA4_run_export_test -j4`
- `cmake --build build-codex-graph-aarch64 --target yolov8_model_benchmark_metrics_test -j4`

Board verification performed with `devkit-run` on `sima@192.168.1.246` after confirming the built binaries are ARM aarch64 with `file`:

- `unit_benchmark_measurement_contract_test` — PASS on DevKit.
- `graph_migration_phaseA4_run_export_test` — PASS on DevKit.
- `yolov8_model_benchmark_metrics_test /workspace/core_graph_changes` — PASS on DevKit; sample output reported latency `8.10136 ms` and `446.578` inferences/s for the batch-1 YOLO fixture.
- `macro_mixed_multistream_multimodel_tput_test --branch-mode encoded-only --streams 1 --rtsp-servers 1 --iters 2 ...` — PASS on DevKit; wrote `/tmp/macro_encoded_smoke.json` with `status=PASS`, `expected_outputs=2`, `total_outputs=2`, `raw_outputs=2`.

DevKit execution note: the exported `devkit-run` shell helper attempted `chmod +x` on NFS-mounted build artifacts and failed with `Operation not permitted`. For verification, the tests were launched through a temporary Python wrapper under `/workspace/core_graph_changes/tmp/` using `devkit-run`; the wrapper did not chmod the target and set only the NEAT/GStreamer runtime paths needed to avoid scanning unrelated system Python shared objects. The temporary wrapper was removed after verification.

## API simplification after review

The benchmark measurement-detail API must stay deliberately small for this release.

Do **not** add a new public benchmark options struct for this. The accepted surface is:

```cpp
// Default/no-arg path: E2E latency + throughput only; no plugin tracing.
auto scope = run.start_measurement();
auto scope2 = run.start_measurement(/*include_plugin_latency=*/false);

// Deep-analysis path: include plugin/kernel latency tracing.
auto scope3 = run.start_measurement(/*include_plugin_latency=*/true);

// Model benchmark mirrors the same boolean.
model.benchmark();
model.benchmark(100, /*include_plugin_latency=*/false);
model.benchmark(100, /*include_plugin_latency=*/true);
model.benchmark(/*include_plugin_latency=*/true); // uses default sample count
```

Implementation notes:

- `MeasureOptions` remains available for advanced knobs, but default construction is now the
  customer-safe path: E2E latency/throughput only.
- `MeasureOptions::include_plugin_latency` is the gate that guarantees profiling has zero plugin
  effect when false.
- Plugin timing is opt-in via the boolean overloads or by setting
  `MeasureOptions::include_plugin_latency = true` explicitly.
- No `BenchmarkOptions` struct / enum is used for this release.

Verification added/performed:

- `unit_benchmark_measurement_contract_test` checks the default/e2e-only path reports plugin
  latency `off/none` and collects no plugin rows.
- Built ARM64 targets with `cmake --build build-codex-graph-aarch64 --target
  unit_benchmark_measurement_contract_test yolov8_model_benchmark_metrics_test`.
- Ran on DevKit with `devkit-run`:
  - `unit_benchmark_measurement_contract_test` PASS.
  - `yolov8_model_benchmark_metrics_test` PASS and stdout shows
    `Measurement detail: e2e+throughput-only`.

Follow-up simplification:

- Removed the attempted `BenchmarkOptions` / `BenchmarkMeasurementDetail` public surface.
- Added only boolean overloads:
  - `Run::start_measurement(bool include_plugin_latency)`
  - `Model::Runner::start_measurement(bool include_plugin_latency)`
  - `Model::benchmark(int num_samples = 100, bool include_plugin_latency = false)`
  - `Model::benchmark(bool include_plugin_latency)` to avoid the `benchmark(true)` -> `num_samples=1` footgun.
- Updated Python binding to expose `model.benchmark(num_samples=100, include_plugin_latency=False)`.

## Final freeze hardening — 2026-06-13

Applied the four release-blocking polish items:

1. **Latency profile quality is explicit.** `LatencyTracker` now emits `quality.status`, minimum correlated sample policy, per-stage correlated counts/coverage, and warnings. Console output prints the quality state so short smoke tests cannot be mistaken for trustworthy latency profiles.
2. **Macro throughput headline is target-normalized.** The macro benchmark report now publishes `headline_fps = target_normalized_output_fps` with semantics, keeps aggregate output FPS as secondary, and warns when excess async/multi-output samples make aggregate FPS misleading.
3. **Measured-window graph export artifact is saved.** `graph_migration_phaseA4_run_export_test` writes `run_export_measured.neat.graph_run.json` and validates that the saved artifact preserves `graph_e2e_latency_ms`, measured-window throughput scope, plugin metrics, and output materialization semantics.
4. **Zero-copy/materialization claims are caveated.** `output_materialization` now carries `claim_status`, `claim_scope`, `runtime_resolved_mode_available=false`, `copy_map_timing_available=false`, and warnings. Schema tests and the visualizer surface the caveat instead of implying copy/map timing was measured.

Validation performed after the final changes:

- `python3 -m py_compile tests/perf/tools/graph_run_schema.py tests/perf/tools/test_graph_run_schema.py tools/visualize_graph_run.py`
- `python3 tests/perf/tools/test_graph_run_schema.py` — 12 tests PASS.
- `cmake --build build-codex-graph-aarch64 --target unit_benchmark_measurement_contract_test graph_migration_phaseA4_run_export_test macro_mixed_multistream_multimodel_tput_test yolov8_model_benchmark_metrics_test -j4` — build PASS.
- DevKit (`devkit-run`, ARM aarch64 binaries confirmed with `file`):
  - `unit_benchmark_measurement_contract_test` — PASS.
  - `graph_migration_phaseA4_run_export_test` — PASS.
  - `yolov8_model_benchmark_metrics_test /workspace/core_graph_changes` — PASS; stdout reports `Measurement detail: e2e+throughput-only`.
  - `macro_mixed_multistream_multimodel_tput_test --branch-mode encoded-only --streams 1 --rtsp-servers 1 --iters 2 --latency-profile true ...` — PASS; stdout reports `target_normalized_fps` and latency `quality=insufficient_correlated_samples`, as expected for a tiny encoded-only smoke run.

DevKit logs are under `/workspace/core_graph_changes/tmp/final_benchmark_changes_20260613T004509Z`.
