# Graph Metrics Implementation Plan

## Goal

Create one customer-facing graph performance artifact with this attribution model:

- **Graph-level headline averages**
  - Throughput (`throughput_fps`)
  - Board/rail power (`PowerSummary`: avg/min/max watts, energy)
- **Node/plugin latency only**
  - Public/lowered graph nodes receive latency summaries when attributable.
  - Plugin/kernel execution latency is nested under the owning node when attribution is unambiguous.
  - Ambiguous plugin timings are retained under `plugin_metrics_unattributed` with a reason.
- **No node/plugin power attribution**
  - Current power telemetry is board/rail PMBus data, not per-node power.
  - Per-node power would be misleading unless produced by isolated experiments or a future attribution model.

This plan has been revised after subagent review. The major corrections are:

1. Public connected `Graph::build(..., RunOptions)` does **not** reliably propagate `RunOptions::enable_metrics` or `RunOptions::power_monitor` into graph segment runtimes today; fix this first.
2. `Run::diag_snapshot()` is insufficient for graph-backed runs; graph metrics must walk `ExecutionGraphRuntime::pipelines[*]->run_core`.
3. `NodeReport.index -> pipe.seg.node_ids[index]` is not safe because graph runtime can inject boundary `Input`/`Output` nodes into materialized pipeline nodes; add an explicit materialized-node mapping.
4. Existing element/stage timing counters are cumulative lifetime counters; measured-window node latency requires baseline/delta snapshots or must be labeled `run_lifetime`.
5. `MeasureReport::plugin_latency` currently drops attribution fields; keep full structured profiler aggregates/events.
6. Profiler events are process-global and not Run-scoped; either enforce one active profiler or add run/graph IDs in profiler ABI.
7. Visualizer/schema must support both public `p*` and lowered `n*` node IDs.

---

## Current code inventory

### Run/graph-level metrics

Files:

- `include/pipeline/Run.h`
- `include/pipeline/RuntimeMetrics.h`
- `src/pipeline/runtime/RunReport.cpp`
- `src/pipeline/RuntimeMetrics.cpp`

Existing surfaces:

- `Run::stats()` / `RunStats`
  - Counters and push-to-output avg/min/max latency.
- `Run::measurement_summary()` / `RunMeasurementSummary`
  - `RunStats`, `InputStreamStats`, `elapsed_seconds`, `throughput_fps`, `PowerSummary`.
- `Run::metrics()` / `RuntimeMetrics`
  - Current throughput formula: `outputs_pulled / elapsed_seconds` over run lifetime.
- `Run::report()`
  - Human-readable diagnostics, run stats, input stats, optional power.

Important limitation:

- `Run::diag_snapshot()` currently reads only `core_->pipeline.stream.diag_ctx()`.
- For graph-backed `Run` objects, per-segment diagnostics live under `core->graph_execution().pipelines[*]->run_core`, not the top-level `core_->pipeline`.

### Timed measurement window

Files:

- `include/pipeline/Run.h`
- `src/pipeline/runtime/RunMeasure.cpp`

Existing surfaces:

- `Run::start_measurement()`
- `MeasureScope::stop()`
- `MeasureReport::to_text()`

Current behavior:

- The scope is observer-style: it measures from `start_measurement()` until caller calls `stop()`.
- `MeasureOptions::warmup_ms` and `duration_ms` are metadata/options today; `MeasureScope` does not enforce them by itself.
- `MeasureScope` deltas `RunStats` counters and collects measurement-only end-to-end/frame-gap latency arrays.
- It attaches `LatencyProfiler` when `MeasureOptions::include_plugin_latency` is true.

Important limitations:

- Element/stage/pad timings in `DiagCtx` are cumulative; `MeasureScope` does not baseline/delta them.
- Therefore measured-window **node latency** is not currently available from existing counters.
- `MeasureReport::plugin_latency` currently stores only `name`, `calls`, `avg/min/max` and loses backend/kernel/stage/slot/total attribution fields.

### Power telemetry

Files:

- `include/pipeline/PowerTelemetry.h`
- `src/pipeline/PowerTelemetry.cpp`
- `src/pipeline/runtime/Run.cpp`
- `src/pipeline/runtime/RunCoreGraphStart.cpp`
- `src/pipeline/runtime/RunCoreGraphStop.cpp`

Existing behavior:

- `RunOptions::enable_board_power()` sets `RunOptions::power_monitor`.
- Linear/single-pipeline `Run` starts `PowerMonitor` from `RunOptions::power_monitor`.
- Graph runtime starts graph-level power from `core->graph_options.power_monitor`.

Board/validation caveat:

- Power telemetry is known to be unreliable on the current DVT board available for this work.
- The same power path is expected to work on SOMs, so we should still implement the graph-level power plumbing and JSON/schema surfaces.
- Local/DVT validation should be limited to:
  - option propagation,
  - monitor construction/start/stop behavior,
  - graceful unavailable/error reporting,
  - JSON/schema shape.
- Do **not** gate the implementation on observed DVT wattage correctness, and do **not** calibrate or special-case customer semantics based on DVT readings.

Important blocker:

- Public connected graph build paths appear to set `start_opt.run_options = opt` but do not reliably copy `opt.power_monitor` into `start_opt.graph_options.power_monitor`, nor `opt` into `start_opt.graph_options.pipeline` for segment runs.
- Fix this before documenting customer API as `RunOptions opt; opt.enable_board_power(); opt.enable_metrics = true; graph.build(opt);`.

### Graph export

Files:

- `include/pipeline/RunExport.h`
- `src/pipeline/runtime/GraphRunExport.cpp`
- `schemas/graph_run_v1.schema.json`
- `tests/perf/tools/graph_run_schema.py`
- `tools/visualize_graph_run.py`

Existing export includes:

- `graph.public_view`
- `graph.lowered_view`
- `graph.nodes`, `graph.edges`, `graph.pipeline_segments`
- `run.stats.latency_ms`
- `run.elapsed_seconds`
- `run.throughput_fps`
- optional `run.power`

Current gap:

- Export does not attach per-node or per-plugin latency to graph nodes.

### Node-to-element mapping

Files:

- `include/pipeline/GraphReport.h`
- `src/pipeline/internal/Diagnostics.h`
- `src/pipeline/graph/GraphBuild.cpp`

Existing `NodeReport` fields:

```cpp
struct NodeReport {
  int index = -1;
  std::string kind;
  std::string user_label;
  std::string backend_fragment;
  std::vector<std::string> elements;
};
```

Population happens in `GraphBuild.cpp` when building the pipeline string:

- `nr.index = i`
- `nr.kind = nodes[i]->kind()`
- `nr.user_label = nodes[i]->user_label()`
- `nr.elements = frag.element_names`

Important blocker:

- `NodeReport.index` is an index in the **materialized pipeline node list**, not guaranteed to be the same as `PipelineSegmentPlan::node_ids` index.
- Graph runtime may inject synthetic boundary `Input`/`Output` nodes into materialized pipeline nodes.
- Therefore the implementation must add an explicit materialized-node-to-runtime-node mapping instead of assuming index equality.

### Element/stage timing

Files:

- `src/pipeline/internal/Diagnostics.h`
- `src/pipeline/graph/GraphBuild.cpp`
- `src/pipeline/runtime/RunReport.cpp`
- `src/pipeline/gst/GstDiagnosticsUtil.cpp`

Existing diagnostics:

- `DiagCtx::stage_timings`
- `DiagCtx::element_timings`
- `DiagCtx::element_flows`
- `DiagCtx::element_pad_timings`

Timing probe attachment:

- `attach_stage_timing_probes(...)` is env-gated by `SIMA_GST_STAGE_TIMINGS`.
- `attach_element_timing_probes(...)` is env-gated by `SIMA_GST_ELEMENT_TIMINGS`.
- `InputStreamOptions::enable_timings` is set from `RunOptions::enable_metrics`, but current probe helpers still require env vars.

Required change:

- `RunOptions::enable_metrics = true` must force timing probes without env vars.
- Env vars can remain as debug overrides.

### Graph runtime actor/stage telemetry

File:

- `src/graph/GraphRunReport.cpp`

Existing internal helper:

- `append_graph_runtime_groups(...)`

Current limitation:

- This helper is in an anonymous namespace and belongs to internal `graph::GraphRun`, not public pipeline `Run` export.
- If public graph export needs graph-stage actor timings (`avg_on_input_us`, mailbox waits, route time), refactor this into shared code or duplicate carefully.

### Plugin/kernel profiler

Files:

- `include/pipeline/LatencyProfiler.h`
- `src/pipeline/profiler/LatencyProfiler.cpp`
- `src/pipeline/profiler/LatencyProfilerSerialize.cpp`
- `deps/headers/neat/profiler/profiler_events.h`
- `deps/headers/neat/profiler/profiler_scoped_timer.h`

Existing profiler model:

- `ProfilerKernelInvocation` has:
  - `backend`
  - `phase`
  - `physical_input_index`
  - `output_slot`
  - `frame_id`
  - `request_id`
  - `kernel_name`
  - `stage_name`
  - segments/bytes
- `ProfilerKernelAggregate` currently groups by:
  - `backend`
  - `kernel_name`
  - `stage_name`
  - `physical_input_index`
  - `output_slot`

Important blockers/risks:

- `ProfilerKernelAggregate` omits `phase`, so load/build/exec/post events can be mixed.
- Profiler events are process-global, not Run-scoped. Concurrent runs/profilers can intermix events.
- Existing `SimaNeatProfilerEvent` does not have explicit `run_id`, `pipeline_segment_id`, `runtime_node_id`, `public_node_id`, or `gst_element_name`.
- `SIMA_NEAT_PROFILER_STAGE_NAME_LEN` is small (32 bytes), so encoding stable IDs into `stage_name` is risky.
- Header copies can drift; reviewers noted `deps/headers/neat/profiler/profiler_events.h` may be stale compared with internals/core headers.

Required robust fix:

- Add versioned profiler ABI fields for stable attribution, or enforce a documented ABI migration path.
- Do not rely on `stage_name` prefixes for final robust mapping.

---

## Target JSON shape

Keep existing `sima.neat.graph_run` v1 fields for compatibility. Add optional structured fields under `run`.

Preferred new fields:

```json
{
  "run": {
    "graph_metrics": {
      "measurement_scope": "run_lifetime",
      "throughput_counting": "all_pulled_outputs",
      "elapsed_seconds": 10.0,
      "outputs_pulled": 1000,
      "throughput_fps": 100.0,
      "power": {
        "enabled": true,
        "samples": 100,
        "duration_seconds": 10.0,
        "total_avg_watts": 5.6,
        "total_min_watts": 5.1,
        "total_max_watts": 6.2,
        "energy_joules": 56.0,
        "rails": []
      }
    },
    "node_metrics": [
      {
        "node_id": "n3",
        "public_node_ids": ["p2"],
        "pipeline_segment_id": 0,
        "segment_local_index": 1,
        "materialized_node_index": 2,
        "kind": "ModelFragment",
        "label": "resnet50",
        "user_label": "resnet50",
        "source": "element_timing",
        "latency_semantics": "sum_element_residency",
        "aggregation": "run_lifetime",
        "latency_ms": {
          "samples": 1000,
          "total": 2300.0,
          "avg": 2.3
        },
        "elements": [
          {
            "segment_id": 0,
            "name": "n2_neatprocessmla",
            "latency_ms": {
              "samples": 1000,
              "total": 1900.0,
              "avg": 1.9,
              "min": 1.6,
              "max": 2.4
            }
          }
        ],
        "plugins": [
          {
            "backend": "MLA",
            "phase": "Run",
            "kernel_name": "infer",
            "stage_name": "n2_neatprocessmla",
            "pipeline_segment_id": 0,
            "runtime_node_id": 3,
            "physical_input_index": 0,
            "output_slot": 0,
            "calls": 1000,
            "latency_ms": {
              "total": 1700.0,
              "avg": 1.7,
              "min": 1.4,
              "max": 2.2
            }
          }
        ]
      }
    ],
    "plugin_metrics_unattributed": []
  }
}
```

Compatibility notes:

- Existing `run.throughput_fps` and `run.power` remain for old consumers.
- `run.graph_metrics` is the preferred/headline customer location.
- Use the same canonical `PowerSummary` shape everywhere possible. Prefer reusing `power_summary_to_json()` for `graph_metrics.power` and keeping old compact `run.power` for compatibility only if required.

---

## Attribution semantics

### Graph throughput

Default semantics:

```text
throughput_counting = all_pulled_outputs
throughput_fps = outputs_pulled / elapsed_seconds
```

For multi-output graphs, document that this counts all successful public pulls. If customers need per-output throughput later, add a separate `output_metrics[]` section. Do not overload node metrics with FPS.

### Graph power

- One graph-level `PowerSummary`.
- For measured-window exports, use a measurement-local monitor or a correctly delimited power delta.
- No power fields in `node_metrics` or plugin metrics.
- Validation on DVT board should verify plumbing and error handling only; numerical wattage correctness must be validated on SOM hardware where the board power telemetry is reliable.

### Node latency

Current first milestone semantics:

```text
latency_semantics = sum_element_residency
aggregation = run_lifetime
```

Why:

- Existing element timing is residency from sink-arrival to src-emit and includes backpressure.
- Summing child elements can overcount overlapping work.
- Node-level `min`/`max` is not mathematically correct when combining multiple element counters; keep exact min/max at element level and only expose node total/avg unless a correct critical-path model is added.

Measured-window node latency requires Step 7 below.

### Plugin latency

Plugin latency semantics:

```text
latency_semantics = profiler_kernel_event_duration
```

Requirements before robust attribution:

- `phase` included in aggregation.
- `pipeline_segment_id` and `runtime_node_id` available from profiler events or a reliable side mapping.
- Empty/ambiguous mappings go to `plugin_metrics_unattributed`.

---

## Implementation plan

### Step 0: Fix public connected graph option propagation

Why first:

- Without this, customer APIs `RunOptions::enable_metrics` and `enable_board_power()` do not reliably apply to connected graph runs.

Files to inspect/change:

- `src/pipeline/graph/GraphBuildInput.cpp`
- `src/pipeline/graph/GraphBuildSource.cpp`
- `src/pipeline/runtime/RunCoreGraphStart.cpp`
- graph runtime option structs around `GraphRuntimeOptions` / start options

Required behavior:

- Public `Graph::build(const RunOptions& opt)` and seeded build variants must copy:

```cpp
start_opt.run_options = opt;
start_opt.graph_options.pipeline = opt;
start_opt.graph_options.power_monitor = opt.power_monitor;
start_opt.graph_options.pipeline.power_monitor = PowerMonitorOptions{}; // no per-segment duplicate
```

or `RunCoreGraphStart` must explicitly fall back to `core->opt` when `graph_options.pipeline/power_monitor` is unset.

Initial implementation status:

- Added `runtime::graph_runtime_options_from_run_options(...)` to derive graph runtime options
  from public `RunOptions`.
- Public source and seeded `Graph::build(...)` paths now populate `start_opt.graph_options`.
- The helper treats `RunOptions::power_monitor` as graph-level telemetry and clears the segment
  copy to avoid N+1 board monitors on connected graphs.
- Legacy/internal `graph::GraphRunOptions` graph-level power also clears segment power when the
  graph-level monitor is enabled.

Acceptance tests:

- Connected graph with `RunOptions::enable_metrics = true` produces segment `element_timings` without env vars after Step 2.
- Connected graph with `RunOptions::enable_board_power()` starts exactly one graph-level monitor, not per-segment duplicate board monitors.

### Step 1: Add graph metric public structs

Prefer new header:

- `include/pipeline/GraphMetrics.h`

Keep it free of internal `DiagCtx` types.

Proposed structs:

```cpp
struct MetricLatencySummary {
  std::uint64_t samples = 0;
  double total_ms = 0.0;
  double avg_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
  bool has_min_max = false;
};

struct PluginLatencyMetric {
  std::string backend;
  std::string phase;
  std::string kernel_name;
  std::string stage_name;
  int pipeline_segment_id = -1;
  int runtime_node_id = -1;
  std::vector<std::string> public_node_ids;
  std::int32_t physical_input_index = -1;
  std::int32_t output_slot = -1;
  std::uint64_t calls = 0;
  MetricLatencySummary latency;
  std::string mapping_error;
};

struct ElementLatencyMetric {
  int pipeline_segment_id = -1;
  std::string element_name;
  MetricLatencySummary latency;
  std::uint64_t missed_in = 0;
  std::uint64_t missed_out = 0;
};

struct NodeLatencyMetric {
  std::string node_id;                 // lowered graph id, e.g. n3
  int runtime_node_id = -1;
  std::vector<std::string> public_node_ids;
  int pipeline_segment_id = -1;
  int segment_local_index = -1;
  int materialized_node_index = -1;
  std::string kind;
  std::string label;
  std::string user_label;
  std::string source;                  // element_timing, graph_stage, none, mixed
  std::string latency_semantics;       // sum_element_residency, graph_stage_on_input, none
  std::string aggregation;             // run_lifetime, measured_window
  MetricLatencySummary latency;
  std::vector<ElementLatencyMetric> elements;
  std::vector<PluginLatencyMetric> plugins;
};

struct GraphMetricSummary {
  std::string measurement_scope;       // run_lifetime or measured_window
  std::string throughput_counting;     // all_pulled_outputs initially
  double elapsed_seconds = 0.0;
  std::uint64_t outputs_pulled = 0;
  double throughput_fps = 0.0;
  PowerSummary power;
};

struct GraphMetricsReport {
  GraphMetricSummary graph;
  std::vector<NodeLatencyMetric> nodes;
  std::vector<PluginLatencyMetric> unattributed_plugins;
};
```

### Step 2: Make timing probes controllable by `RunOptions::enable_metrics`

Files:

- `src/pipeline/graph/GraphBuild.cpp`
- `src/pipeline/graph/GraphDetail.h`
- call sites in `GraphBuildInput.cpp`, `GraphBuildSource.cpp`, `GraphValidate.cpp`

Change signatures:

```cpp
void attach_stage_timing_probes(GstElement* pipeline,
                                const std::shared_ptr<DiagCtx>& diag,
                                bool force_enable = false);

void attach_element_timing_probes(GstElement* pipeline,
                                  const std::shared_ptr<DiagCtx>& diag,
                                  bool force_enable = false);
```

Implementation:

```cpp
if (!force_enable && !env_bool("SIMA_GST_ELEMENT_TIMINGS", false)) return;
```

Call with `stream_opt.enable_timings` or the resolved run option for actual runs. Keep validation env-gated unless validation options explicitly request timings.

Acceptance criterion:

- No env vars required for `RunOptions::enable_metrics = true` to populate element timings.

Initial implementation status:

- `attach_stage_timing_probes(...)` and `attach_element_timing_probes(...)` now accept an
  option-driven enable flag while preserving the legacy env-var behavior.
- Source/input build paths pass `stream_opt.enable_timings`, which is derived from
  `RunOptions::enable_metrics`.
- Graph validation call sites remain env-gated through the default flag.
- Added a focused metrics-report test case that builds a simple `Input -> VideoConvert -> Output`
  run with `RunOptions::enable_metrics = true` and checks that element timing counters are present
  without relying on `SIMA_GST_ELEMENT_TIMINGS`.

### Step 3: Add explicit materialized-node mapping

Problem:

- `NodeReport.index` cannot safely map to `PipelineSegmentPlan::node_ids[index]`.

Preferred change:

Extend `NodeReport` or add an internal parallel mapping.

Option A: extend public `NodeReport` carefully:

```cpp
struct NodeReport {
  int index = -1;
  int runtime_node_id = -1;       // new: lowered graph node id when known
  int pipeline_segment_id = -1;   // new
  bool compiler_generated = false;
  ...
};
```

Option B: keep `NodeReport` stable and add internal-only mapping to `DiagCtx`:

```cpp
struct NodeElementAttribution {
  int materialized_node_index = -1;
  int runtime_node_id = -1;
  int pipeline_segment_id = -1;
  std::vector<std::string> elements;
};

std::vector<NodeElementAttribution> DiagCtx::node_attribution;
```

Recommended: Option B if API compatibility is a concern; Option A if exposing runtime IDs in reports is acceptable.

Initial implementation status:

- Chose an internal mapping, leaving public `NodeReport` stable for now.
- Added `runtime::MaterializedNodeAttribution` with explicit roles:
  - `SegmentNode`
  - `InjectedInput`
  - `InjectedOutput`
- Added `PipelineSegmentPlan::materialized_node_attribution`.
- Added helpers:
  - `attributed_runtime_node_for_segment_node(...)`
  - `make_materialized_node_attribution(...)`
- Graph runtime materialization now populates the mapping after inserting graph boundary
  `Input`/`Output` nodes and before the segment is used for pipeline execution.
- Injected boundary nodes intentionally keep `runtime_node = graph::kInvalidNode`, so later
  metrics code cannot silently shift timings onto the wrong compiled node.
- Added focused coverage in `graph_migration_phase3_metrics_report_test` for the injected
  input/output mapping helper.

Where to populate:

- In graph segment materialization, where the runtime converts `PipelineSegmentPlan` to materialized `nodes` vector before calling pipeline build.
- Build and pass a vector parallel to the materialized `nodes` vector:

```cpp
std::vector<int> materialized_runtime_node_ids;
```

Rules:

- Original segment nodes map to their `seg.node_ids[local]`.
- Injected boundary `Input`/`Output` map to `-1` or a specific boundary role.
- Use `(pipeline_segment_id, element_name)` as the element attribution key.

Acceptance tests:

- Segment with injected boundary `Input` has correct mapping for original nodes.
- Segment with injected terminal `Output` does not shift original node IDs.
- Two segments with same element names do not collide because key includes segment ID.

### Step 4: Build run-lifetime node metrics first

New implementation file:

- `src/pipeline/runtime/GraphMetrics.cpp`

Header:

- `include/pipeline/GraphMetrics.h`

Internal API:

```cpp
GraphMetricsReport build_graph_metrics_report_run_lifetime(
    const Run& run, const RuntimeMetricsOptions& opt = RuntimeMetricsOptions{});
```

Initial implementation status:

- Added `include/pipeline/GraphMetrics.h`.
- Added `src/pipeline/runtime/GraphMetrics.cpp`.
- Implemented `GraphMetricsReport` with:
  - `graph_metrics` from existing `Run::metrics(...)`,
  - `aggregation = "run_lifetime"`,
  - `latency_semantics = "sum_element_residency"`,
  - `throughput_counting = "all_pulled_outputs"`,
  - lowered/runtime `node_metrics`.
- Linear runs aggregate from the top-level `DiagCtx::node_reports` and `element_timings`.
- Graph runs walk `core->graph_execution().pipelines[*]->run_core`, use each segment's
  `materialized_node_attribution`, and skip injected boundary nodes.
- Node latency currently folds element residency totals into node-level totals. This is explicitly
  run-lifetime residency aggregation, not measured-window latency yet.
- Node metrics now retain exact per-element residency summaries (`elements[*].latency_ms`) in
  addition to the node-level aggregate.
- Added focused coverage in `graph_migration_phase3_metrics_report_test`.

Implementation outline:

1. Get `runtime::RunCore` via `run_internal::core(run)`.
2. Populate graph-level metrics from `run.metrics(RuntimeMetricsOptions{.include_power = true})`.
3. If no `graph_execution_`, build a single linear pipeline node metric set from top-level `core->pipeline.stream.diag_ctx()`.
4. If `graph_execution_` exists:
   - Walk `core->graph_execution().pipelines`.
   - For each built `pipe.run_core`, get its `DiagCtx` and `diag_snapshot()`.
   - Use explicit materialized mapping from Step 3 to create `NodeLatencyMetric` skeletons.
   - Map element timings to nodes by `(segment_id, element_name)`.
   - Attach public IDs via reverse mapping from `plan.public_nodes[*].runtime_node`.
5. For lazy/unbuilt segments:
   - Emit node skeletons with `source = "none"` and no latency, or omit metrics depending on `RunExportOptions::include_empty_node_metrics` (optional).

Node latency aggregation:

- Fold element totals into node total/avg.
- Do not expose node min/max unless a node has exactly one element or a correct critical-path computation exists.
- Keep exact min/max at `elements[*].latency_ms`.

### Step 5: Public/lowered node reverse mapping

Use `ExecutionGraphPlan::public_nodes` from graph export code.

Rules:

- `public_node.runtime_node == runtime_node_id` -> add `p<public_id>` to `public_node_ids`.
- Many public nodes may map to one runtime node; store a vector.
- If no public node maps, leave `public_node_ids` empty.
- For model fragments, optionally use fragment provenance/ranges for display metadata, but do not fabricate public IDs unless mapping is explicit.

Visualizer must index node metrics by both:

- lowered node ID (`n*`)
- each public node ID (`p*`)
- possibly public node's `runtime_node` field in graph JSON

Initial implementation status:

- `GraphMetricsReport` now carries `public_node_ids` for runtime/lowered nodes by reversing
  `ExecutionGraphPlan::public_nodes[*].runtime_node`.
- `GraphRunExport` serializes both lowered `node_id`/`runtime_node` and `public_node_ids`.
- The visualizer indexes metrics by lowered IDs (`n*`), public IDs (`p*`), and public-view
  `runtime_node` fallback.

### Step 6: Extend plugin profiler structures before plugin JSON

Files:

- `include/pipeline/LatencyProfiler.h`
- `src/pipeline/profiler/LatencyProfiler.cpp`
- `src/pipeline/runtime/RunMeasure.cpp`

Changes:

1. Add `phase` to `ProfilerKernelAggregate`.
2. Group by `(backend, phase, kernel_name, stage_name, physical_input_index, output_slot)`.
3. Preserve `total_ms` in measurement output.
4. Replace or extend `MeasurePluginLatency`:

```cpp
struct MeasurePluginLatency {
  std::string backend;
  std::string phase;
  std::string kernel_name;
  std::string stage_name;
  std::int32_t physical_input_index = -1;
  std::int32_t output_slot = -1;
  std::uint64_t calls = 0;
  double total_ms = 0.0;
  double avg_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
};
```

or store:

```cpp
std::vector<ProfilerKernelAggregate> plugin_latency;
```

Recommended: use `ProfilerKernelAggregate` or a struct with the same fields to avoid data loss.

Concurrency policy:

- Short-term: document/enforce one active `LatencyProfiler` at a time with a guard.
- Robust: add run/graph ID fields to profiler events.

Initial implementation status:

- Added `phase` to `ProfilerKernelAggregate`.
- Profiler aggregation now keys by `(backend, phase, kernel_name, stage_name,
  physical_input_index, output_slot)` so load/build/run/post events are no longer folded together.
- `MeasurePluginLatency` now preserves `backend`, `phase`, `kernel_name`, `stage_name`,
  `physical_input_index`, `output_slot`, and `total_ms` in addition to the old display `name`,
  calls, avg/min/max fields.
- Measurement text output now includes phase and total latency.
- The profiler is still process-global; robust run/graph attribution still belongs to Step 9.

### Step 7: Add measured-window diagnostic deltas

Do not claim measured-window node latency until this exists.

Files:

- `src/pipeline/runtime/RunMeasure.cpp`
- `src/pipeline/runtime/GraphMetrics.cpp`

Add to `MeasureScope::Impl`:

```cpp
GraphDiagnosticsSnapshot before_diag;
GraphDiagnosticsSnapshot after_diag;
```

Where `GraphDiagnosticsSnapshot` is an internal flattened snapshot of:

- per segment element timing counters
- per segment pad timing counters
- per segment stage timing counters
- graph stage actor telemetry if included

Delta rules:

- `samples` and `total_us` can be subtracted.
- `min/max` cannot be exactly subtracted from cumulative counters.
  - For measured window, either omit min/max or add measurement-window min/max counters/reset support.
- Mark measured-window latency fields with `aggregation = "measured_window"`.

Alternative:

- Reset timing counters at measurement start. This is simpler but more invasive and unsafe if other readers expect lifetime counters.
- Prefer before/after snapshots.

### Step 8: Measurement-local power monitor

Problem:

- Run-level `PowerMonitor` may include warmup or pre-window time.

Change `MeasureScope::Impl`:

```cpp
std::unique_ptr<PowerMonitor> power_monitor;
```

At `Run::start_measurement()`:

- If `opt.include_power`:
  - Determine options source:
    - linear run: `core_->opt.power_monitor`
    - graph run: `core_->graph_options.power_monitor`
  - If enabled, start a measurement-local `PowerMonitor`.

At `MeasureScope::stop()`:

- Stop measurement-local monitor and use its summary.
- Fall back to `run.power_summary()` only when no measurement-local monitor exists.

### Step 9: Robust profiler ABI attribution

Short-term best effort:

- Map plugin `stage_name` to `(segment_id, element_name)` only when exact and unique.
- Treat empty, duplicate, or ambiguous stage names as unattributed.

Robust required ABI migration:

Add versioned fields to profiler events:

```cpp
std::uint32_t event_size;
std::uint32_t event_version;
std::uint64_t run_id_hash;          // or fixed run UUID hash
std::int32_t pipeline_segment_id;
std::int32_t runtime_node_id;
std::int32_t public_node_id;        // optional; vector not possible in event
char gst_element_name[...];
```

Do not rely on stable ID prefixes in `stage_name`; reviewers found length/ABI risk.

Synchronize header copies:

- `deps/headers/neat/profiler/profiler_events.h`
- `deps/headers/usr/include/neat/profiler/profiler_events.h`
- any internals/core profiler header copy

### Step 10: JSON export overloads

Files:

- `include/pipeline/RunExport.h`
- `src/pipeline/runtime/GraphRunExport.cpp`

Prefer overloads, not a non-owning pointer in `RunExportOptions`:

```cpp
std::string run_to_json(const Run& run,
                        const RunExportOptions& opt = {},
                        std::string* err = nullptr);

std::string run_to_json(const Run& run,
                        const MeasureReport& report,
                        const RunExportOptions& opt = {},
                        std::string* err = nullptr);

bool save_run_json(const Run& run,
                   const std::string& path,
                   const RunExportOptions& opt = {},
                   std::string* err = nullptr);

bool save_run_json(const Run& run,
                   const MeasureReport& report,
                   const std::string& path,
                   const RunExportOptions& opt = {},
                   std::string* err = nullptr);
```

Extend options:

```cpp
struct RunExportOptions {
  std::string label;
  bool include_metrics = true;
  bool include_power = true;
  bool include_node_metrics = true;
  bool include_plugin_metrics = true;
  bool include_empty_node_metrics = true;
  int indent = 2;
  std::vector<std::pair<std::string, std::string>> metadata;
};
```

Also extend `RunAutoExportOptions` with node/plugin flags where applicable.

Initial implementation status:

- `run_to_json(...)` / `save_run_json(...)` now include `run.graph_metrics`,
  `run.node_metrics`, and an empty `run.plugin_metrics_unattributed` array.
- `run.graph_metrics` is graph-level only: measurement scope, throughput counting semantics,
  elapsed seconds, output counters, throughput FPS, counters, and optional canonical
  `PowerSummary` JSON.
- `run.node_metrics` carries only latency attribution: lowered/public IDs, segment, kind/label,
  element names, node `latency_ms`, and per-element `latency_ms`.
- Existing compatibility fields (`run.throughput_fps`, compact `run.power`, `run.stats`) remain.
- Plugin metrics are intentionally schema/export placeholders until profiler attribution work in
  Steps 6 and 9 is implemented.
- Added overloads that accept a `MeasureReport`:
  - `run_to_json(const Run&, const MeasureReport&, ...)`
  - `save_run_json(const Run&, const MeasureReport&, ...)`
- The measured overload marks `run.graph_metrics.measurement_scope = "measured_window"` and exports
  structured measurement-window plugin rows under `plugin_metrics_unattributed` with the preserved
  backend/phase/kernel/stage/slot/total fields. They remain unattributed until Step 9 adds reliable
  node IDs.

### Step 11: Schema and validator

Files:

- `schemas/graph_run_v1.schema.json`
- `tests/perf/tools/graph_run_schema.py`
- `tests/perf/tools/test_graph_run_schema.py`

Add real optional definitions, not just permissive objects:

- `graph_metrics`
- `latency_summary`
- `node_metric`
- `element_metric`
- `plugin_metric`
- `power_summary`

Validation rules:

- `graph_metrics.throughput_fps` numeric.
- `graph_metrics.power` object uses canonical `PowerSummary` shape.
- `node_metrics[*].node_id` string.
- `node_metrics[*].latency_ms` object if present.
- Node/plugin metrics must not include power fields.
- `plugin_metrics_unattributed` array of plugin metrics with `mapping_error` allowed/expected.

Keep `additionalProperties: true` for backward-compatible evolution, but validate known fields.

Initial implementation status:

- Added optional schema definitions for `graph_metrics`, `node_metric`, `element_metric`,
  `plugin_metric`, `latency_summary`, `counter_summary`, and canonical `power_summary`.
- Extended the dependency-free validator and schema tests to validate known metric fields and to
  reject node/plugin metric objects that contain `power`.
- Extended the export integration test to assert the new graph/node/plugin metric JSON surfaces.

### Step 12: Visualizer

File:

- `tools/visualize_graph_run.py`

Add:

1. Header cards:
   - throughput FPS
   - avg watts
   - energy joules
   - measurement scope
   - throughput counting semantics
2. Node annotations:
   - for lowered view: lookup by `node.id` (`n*`)
   - for public view: lookup by public ID (`p*`) and by `runtime_node`
3. Detail tables:
   - node metrics table
   - plugin metrics nested or adjacent
   - unattributed plugin warning section

Correct mapping:

```python
by_lowered = {m["node_id"]: m for m in node_metrics}
by_public = {}
for m in node_metrics:
    for pid in m.get("public_node_ids", []):
        by_public[pid] = m
```

For public nodes, if no direct `p*` match, use `node.get("runtime_node")` to look up lowered metric.

Initial implementation status:

- Added offline HTML metric cards for throughput, elapsed time, measurement scope, throughput
  counting semantics, and optional power.
- Graph nodes with attributable metrics are highlighted and annotated with latency text.
- Added a node-metrics detail table.
- Extended the visualizer test to cover metric rendering while keeping the output self-contained.

### Step 13: Python bindings and docs alignment

File:

- `python/src/module.cpp`

Current gaps to fix:

- Bind `RunOptions::enable_board_power()` and board-specific helpers.
- Bind `Run::metrics()` / `Run::metrics_report()` or at least `metrics_report()` text/JSON.
- Bind `Run::measurement_summary()` / `Run::power_summary()` if structs are exposed.
- Bind `RunReportOptions.include_power`.
- Bind `RunDiagSnapshot.element_pad_timings`.
- Bind new `RunExportOptions` flags.

Docs currently mention Python `enable_board_power()`/`run.metrics()` shape; if bindings lag, update docs or bindings so customer docs are true.

---

## Test plan

### Unit tests

1. **Option propagation for connected graphs**
   - New test or extend graph build tests.
   - Build connected graph with `RunOptions::enable_metrics = true` and power enabled.
   - Assert graph runtime options receive metrics/power.
   - For power, assert configuration/plumbing, not DVT board numerical wattage.

2. **Materialized node mapping**
   - New `unit_graph_metrics_mapping_test.cpp`.
   - Include injected boundary `Input`/`Output` cases.
   - Include duplicate element names across segment IDs.

3. **Latency aggregation semantics**
   - Verify node aggregate uses total/samples and does not fabricate min/max for multi-element nodes.
   - Verify element min/max retained.

4. **Plugin aggregate preserves fields**
   - Add/extend profiler tests.
   - Verify aggregate key includes `phase`.
   - Verify `MeasureReport` retains backend/kernel/stage/slot/total.

5. **No power in node/plugin metrics**
   - Schema/unit test that node/plugin metric objects do not contain power fields.

### Integration tests

1. **Run export node metrics**
   - Extend `tests/graph_migration/unified/phaseA4_run_export_test.cpp` or add new test.
   - Build `Input -> Output`, push/pull, export.
   - Assert `run.graph_metrics`, `run.node_metrics`, existing compatibility fields.

2. **Connected multi-segment graph export**
   - Verify per-segment metrics and no element-name collisions.

3. **Measured export**
   - Use `MeasureScope` after measured-window delta support.
   - Assert `measurement_scope = measured_window` and no lifetime/warmup node contamination.

4. **Visualizer**
   - Extend `tests/perf/tools/test_graph_run_visualizer.py`.
   - Test public and lowered views with metrics.

5. **Schema validator**
   - Extend `tests/perf/tools/test_graph_run_schema.py` with graph/node/plugin metric payloads.

### DevKit/runtime execution rules

Many built binaries are `aarch64`. Before executing any built binary/test:

```bash
file <path>
```

If it reports `aarch64` or `ARM aarch64`, do not run locally. Use devkit path when available:

```bash
dk /home/docker/sima-cli/tmp/devkit_env_exec.sh <binary> [args...]
```

or repo wrappers if present:

```bash
tools/run-on-devkit <binary> [args...]
tools/ctest-on-devkit <regex> [extra ctest args...]
```

Validation performed for the initial implementation:

- Built:
  - `cmake --build /workspace/core_graph_changes/build --target graph_migration_phase3_metrics_report_test -j2`
  - `cmake --build /workspace/core_graph_changes/build --target graph_migration_phaseA4_run_export_test -j2`
- Confirmed both focused binaries are `ARM aarch64` with `file`.
- Ran both focused tests on DevKit through `devkit-run`:
  - `graph_migration_phase3_metrics_report_test`
  - `graph_migration_phaseA4_run_export_test`
- Ran host-side schema/visualizer checks:
  - `python3 -m json.tool schemas/graph_run_v1.schema.json`
  - `python3 -m pytest tests/perf/tools/test_graph_run_schema.py tests/perf/tools/test_graph_run_visualizer.py`

If `dk` appears unavailable, check `/usr/local/bin/dk` and source `/root/.devkit-sync.rc` before concluding it is missing.

---

## Revised implementation order

1. Fix public connected graph `RunOptions` propagation for metrics/power.
2. Force timing probes from `RunOptions::enable_metrics` without env vars.
3. Add explicit materialized-node-to-runtime-node attribution mapping.
4. Add `GraphMetrics` structs and run-lifetime graph/node metric builder.
5. Extend JSON export/schema/validator/visualizer for run-lifetime graph/node metrics.
6. Preserve structured profiler aggregates in `MeasureReport`; add `phase` to aggregation.
7. Add best-effort plugin attribution with strict ambiguity handling.
8. Add measured-window diagnostic deltas for node latency.
9. Add measurement-local `PowerMonitor` for measured-window graph power.
10. Add versioned profiler ABI fields for robust plugin-to-node attribution.
11. Add measured-window export overloads using `MeasureReport`.
12. Update Python bindings/docs.

---

## Minimal first milestone

A safe first customer milestone should **not** claim measured-window node/plugin attribution yet.

Deliver:

- `RunOptions::enable_metrics = true` works for connected graph segment timing.
- `save_run_json(run, ...)` emits:
  - `run.graph_metrics` with run-lifetime graph throughput and graph power.
  - `run.node_metrics` with run-lifetime node latency from element timings.
  - `latency_semantics = sum_element_residency`.
  - no node/plugin power fields.
- Visualizer shows graph throughput/power and node latency in both public/lowered views.
- Plugin metrics are either absent or best-effort with unapplied events preserved under `plugin_metrics_unattributed`.

Milestone 2:

- Structured plugin aggregate preservation and best-effort attribution.

Milestone 3:

- Measured-window graph/node/plugin report with diagnostic deltas and measurement-local power.

Milestone 4:

- Robust profiler ABI stable IDs and Run-scoped events.

---

## Acceptance criteria

A customer can save one graph-run artifact and answer:

1. What was headline graph throughput?
2. What was graph-level average/min/max power and energy?
3. Which graph node had the largest latency under the stated latency semantics?
4. Which plugin/kernel timings were attached to each node?
5. Which plugin/kernel timings could not be attributed and why?

Correctness constraints:

- Throughput/power appear at graph level only.
- Node/plugin sections contain latency only.
- Node latency semantics are explicit.
- Existing `sima.neat.graph_run` consumers remain compatible.
- Ambiguous plugin attribution is not silently attached or dropped.
- Public and lowered graph visualizations both show metrics when available.
