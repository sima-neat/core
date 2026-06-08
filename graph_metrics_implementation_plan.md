# Customer-Ready Graph Metrics and Visualization Implementation Plan

## Status

This file **replaces** the older graph-metrics implementation plan. It reflects the code and artifacts produced so far, plus the subagent review of the customer gaps.

Current implementation is useful internally, but **not customer-ready**. The main problems are:

1. The visualizer has no customer topology view. It chooses raw `public_view` first and raw `lowered_view` second.
2. Branched public graphs show invalid boundary/pass-through public nodes. The customer should see `Input -> Branch -> Outputs`, not duplicated `camera_left -> camera_left` edges.
3. DETR currently renders from fallback/runtime metrics topology because the exported graph lacks named public topology.
4. Throughput is counted as `outputs_pulled / elapsed`, which is correct as an output-pull metric but misleading if labeled as model FPS or batches/s on multi-output graphs.
5. Measured-window JSON can mix lifetime fields with measured-window fields because `run_to_json(run, report, ...)` starts from lifetime JSON and overwrites only part of `run`.
6. Graph end-to-end latency is absent for graph-backed runs but currently looks like `N/A` without a structured availability reason.
7. Node/plugin/edge timing semantics are diagnostic and non-additive; the UI and schema do not make this strong enough.
8. Diagnostic edge percentiles and max fields are currently ambiguous: p50/p95 may be placeholder zeros; max may be lifetime high-water.
9. Power is graph-level board rail telemetry only. DVT board power is not reliable, but SOM power should remain implemented and clearly marked.
10. Customer validation is not automated: DETR, multi-stream, branch/fanout, and people-tracker coverage are currently ad-hoc.
11. Some current implementation files are untracked and must be included or removed before release.

The rest of this document gives an implementation plan with enough concrete file/function details to avoid hand-waving during implementation.

Concrete execution companion: [`graph_metrics_specific_implementation.md`](graph_metrics_specific_implementation.md).

Subagent hardening incorporated after review:

- Exact path timing must not mix `steady_clock` with LTTng timestamps. Graph E2E can use
  `steady_clock`; graph-entry→node, inter-plugin gap, and output-tail must use LTTng-only boundary
  and plugin/edge timestamps.
- Graph sinks must carry edge identity with an internal `RuntimeSinkQueueMsg`; do not use a
  single `sink_node -> edge_index` map.
- Runtime message IDs must match adopted plugin tracepoints: `orig_input_seq`, else `input_seq`,
  else `frame_id`, else `0`/missing. Do not hash message IDs.
- Trace hot paths must check `message_trace_enabled` before constructing trace args or copying
  strings.
- `trace_graph_id_hash` is atomic, just like `trace_run_id_hash`.
- Visualizer edge overlays are list/multi-stream aware, redaction applies to HTML/SVG/topology JSON,
  and tests cover Combine/JoinBundle as well as Branch/FanOut.

---

## Customer-facing semantics and invariants

### Metric ownership model

| Metric class | Scope | Meaning | Additive? | Report location |
|---|---:|---|---:|---|
| Throughput | Graph | Output pulls per second over run lifetime or measured window | No | top cards + `run.graph_metrics.throughput` |
| Logical throughput | Graph | output-pull throughput × logical batch size | No | top cards + `run.graph_metrics.throughput` |
| Power | Graph | board/SOM rail average power over the selected window | No | top cards + `run.graph_metrics.power` |
| Graph E2E latency | Graph | public push to public pull when identity mapping is available | No | top cards + `run.graph_e2e_latency_ms` |
| Node latency | Runtime/customer node | GStreamer element residency summary; diagnostic | No | node cards/table |
| Plugin latency | Plugin instance inside node | plugin/kernel execution span; diagnostic | No | nested plugin rows or unattributed table |
| Edge/message latency | Edge | queue/handoff/transport time; diagnostic | No | edge labels/details table |
| Path/timeline timing | Graph path | graph-entry to node arrival, inter-plugin gap, and output tail when sample identity exists | No | timeline table + optional path overlay |

Hard rules:

- Do **not** put power under node/plugin/edge rows.
- Do **not** add edge/message latency to plugin, node, or graph E2E latency.
- Do **not** sum multiple lowered-edge latencies into one customer edge. Use representative display such as max avg and show details.
- Do **not** render unavailable timing as `0.000 ms`.
- Do **not** call output-pull throughput “FPS” or “model FPS” unless one output pull equals one frame/inference for that graph.
- Do **not** maintain two independent calculations for the same customer number. New structured fields are the source of truth; old compatibility fields are copied from them.
- Do **not** infer per-sample timing by nearest timestamp. If `message_id`, `orig_input_seq`, `input_seq`, `frame_id`, or `stream_id` cannot correlate the sample, mark the path timing unavailable.

Compatibility rule:

- Schema v1 can keep old fields such as `throughput_fps`, `outputs_per_s`, and `measurement.end_to_end`, but only as a compatibility shim.
- Implementation must calculate canonical values once in typed helpers such as `throughput_json()`, `graph_e2e_json()`, and `power_status_json()`.
- Compatibility fields must be assigned from those canonical objects so they cannot drift.
- Any legacy calculation path that duplicates the canonical math should be deleted or replaced by a wrapper.

---

## Target JSON shape

Keep schema name/version stable:

```json
{
  "schema": "sima.neat.graph_run",
  "schema_version": 1,
  "graph": {},
  "run": {}
}
```

Add optional structured fields. Existing old fields remain only as compatibility aliases copied
from the new canonical fields; customer tools should prefer the new fields.

### `graph.customer_view`

Add a canonical customer topology view. It hides synthetic public boundary/pass-through nodes and preserves Branch/Combine/Model/Input/Output concepts.

```json
"graph": {
  "customer_view": {
    "topology_source": "customer_view_v1",
    "nodes": [
      {
        "id": "c0",
        "kind": "Input",
        "role": "input",
        "label": "camera_left",
        "endpoint_name": "camera_left",
        "public_node_ids": ["p0"],
        "lowered_node_ids": ["n0"],
        "runtime_node_ids": [0],
        "compiler_generated": false,
        "mapping_status": "mapped",
        "source": {
          "kind": "app_push",
          "uri": null,
          "endpoint": "camera_left",
          "details": {}
        }
      },
      {
        "id": "c1",
        "kind": "Branch",
        "role": "router",
        "label": "camera_left branch",
        "endpoint_name": null,
        "public_node_ids": ["p1"],
        "lowered_node_ids": ["n9"],
        "runtime_node_ids": [9],
        "compiler_generated": true,
        "mapping_status": "mapped"
      }
    ],
    "edges": [
      {
        "id": "ce0",
        "from": "c0",
        "to": "c1",
        "kind": "data",
        "label": "camera_left → branch",
        "from_endpoint": "camera_left",
        "to_endpoint": null,
        "public_edge_ids": ["pe2"],
        "lowered_edge_ids": ["e0"],
        "metric_policy": "single_lowered_edge",
        "mapping_status": "mapped"
      },
      {
        "id": "ce1",
        "from": "c1",
        "to": "c2",
        "kind": "branch",
        "label": "branch → left_preview",
        "from_endpoint": null,
        "to_endpoint": "left_preview",
        "public_edge_ids": ["pe0", "pe3"],
        "lowered_edge_ids": ["e1"],
        "metric_policy": "single_lowered_edge",
        "mapping_status": "mapped"
      }
    ],
    "mapping": {
      "customer_to_public": {"c0": ["p0"], "c1": ["p1"]},
      "public_to_customer": {"p0": "c0", "p1": "c1"},
      "customer_to_lowered": {"c0": ["n0"], "c1": ["n9"]},
      "lowered_to_customer": {"n0": "c0", "n9": "c1"},
      "customer_edge_to_public": {"ce0": ["pe2"]},
      "customer_edge_to_lowered": {"ce0": ["e0"]}
    }
  }
}
```

#### Customer node fields

Required:

```json
{
  "id": "c0",
  "kind": "Input|Output|Model|Branch|Combine|Operator|RuntimeNode",
  "role": "input|output|compute|router|fallback",
  "label": "string",
  "endpoint_name": "string|null",
  "public_node_ids": ["p0"],
  "lowered_node_ids": ["n0"],
  "runtime_node_ids": [0],
  "compiler_generated": false,
  "mapping_status": "mapped|partial|unresolved|fallback"
}
```

Optional:

```json
{
  "source": {},
  "sink": {},
  "model": {},
  "mapping_error": "specific reason"
}
```

#### Customer edge fields

Required:

```json
{
  "id": "ce0",
  "from": "c0",
  "to": "c1",
  "kind": "data|branch|join|endpoint|fallback",
  "label": "camera_left → left_preview",
  "from_endpoint": "camera_left|null",
  "to_endpoint": "left_preview|null",
  "public_edge_ids": ["pe0"],
  "lowered_edge_ids": ["e0"],
  "metric_policy": "single_lowered_edge|non_additive_representative_max|none",
  "mapping_status": "mapped|partial|unresolved|fallback"
}
```

If a public/customer edge cannot map to lowered runtime edges, do not leave ambiguity. Emit:

```json
{
  "mapping_status": "unresolved",
  "mapping_error": "public edge pe4 has invalid runtime endpoints and no reachable lowered path",
  "lowered_edge_ids": []
}
```

### `run.graph_metrics.window`

Add a window block so measured-window vs lifetime is unambiguous:

```json
"run": {
  "graph_metrics": {
    "measurement_scope": "measured_window",
    "aggregation": "measured_window",
    "window": {
      "scope": "measured_window",
      "start_source": "Run::start_measurement",
      "stop_source": "MeasureScope::stop",
      "elapsed_seconds": 2.000,
      "requested_duration_ms": 10000,
      "requested_warmup_ms": 1000,
      "duration_enforced_by_scope": false,
      "warmup_iterations_excluded": 0
    }
  }
}
```

For run-lifetime export:

```json
"window": {
  "scope": "run_lifetime",
  "start_source": "RunCore::created_at",
  "stop_source": "last_pull_or_output_or_now",
  "duration_enforced_by_scope": false
}
```

### `run.graph_metrics.throughput`

Keep compatibility fields:

```json
"throughput_fps": 41.43,
"outputs_per_s": 41.43,
"throughput_batches_per_s": 41.43
```

Add explicit denominator block:

```json
"throughput": {
  "unit": "output_pulls_per_second",
  "numerator_counter": "outputs_pulled",
  "numerator_value": 10,
  "denominator": "elapsed_seconds",
  "denominator_seconds": 0.241,
  "outputs_per_s": 41.43,
  "logical_batch_size": 1,
  "logical_inferences_per_s": 41.43,
  "multi_output_semantics": "each successful public pull increments outputs_pulled once"
}
```

For multi-output graphs, the visualizer must label this as `Output pulls/s`. If `graph.named_outputs.size() > 1`, show a warning hint:

```text
This graph has multiple public outputs; output-pull throughput counts every successful named/default pull.
```

### `run.graph_e2e_latency_ms`

Add explicit availability metadata. For current graph-backed runs without identity correlation:

```json
"graph_e2e_latency_ms": {
  "available": false,
  "status": "unavailable_graph_e2e_not_instrumented",
  "source": "none",
  "semantics": "public_push_to_public_pull",
  "count": 0
}
```

For linear runs with existing pending-time samples:

```json
"graph_e2e_latency_ms": {
  "available": true,
  "status": "collected",
  "source": "linear_pending_times",
  "semantics": "input_thread_push_to_output_ready",
  "count": 10,
  "avg_ms": 3.2,
  "p50_ms": 3.1,
  "p90_ms": 3.8,
  "p95_ms": 4.0,
  "p99_ms": 4.5,
  "max_ms": 4.7
}
```

Status enum:

```text
collected
no_samples
unavailable_graph_e2e_not_instrumented
unavailable_missing_sample_identity
unavailable_multi_input_lineage_ambiguous
unavailable_multi_output_lineage_ambiguous
```

### Node metric fields

Add or enforce:

```json
{
  "node_id": "n1",
  "source": "gst_element_timing",
  "status": "collected|no_samples|timing_disabled|unavailable_no_node_attribution",
  "latency_semantics": "sum_element_residency_delta",
  "aggregation": "measured_window",
  "non_additive": true,
  "latency_ms": {
    "samples": 10,
    "total_ms": 80.84,
    "avg_ms": 8.084,
    "min_ms": 0.0,
    "max_ms": 0.0,
    "min_max_available": false
  }
}
```

Visualizer label:

```text
Node avg 8.084 ms (element residency, non-additive)
```

Tooltip/details text:

```text
Sum of GStreamer element residency; sink-arrival to src-emit; can include backpressure and overlap. Do not sum with plugin or edge latency.
```

### Plugin metric fields

Add or enforce:

```json
{
  "name": "CVU:r123.g456.s0.n1.n1_quanttess_1",
  "backend": "CVU",
  "phase": "Run",
  "kernel_name": "processcvu_dispatcher",
  "gst_element_name": "n1_quanttess_1",
  "plugin_instance_id": "r123.g456.s0.n1.n1_quanttess_1",
  "source": "lttng",
  "timing_semantics": "inclusive_processcvu_dispatcher_device_window",
  "attribution_source": "lttng_v2_identity",
  "reliable": true,
  "non_additive": true,
  "mapping_error": null,
  "calls": 10,
  "latency_ms": {
    "samples": 10,
    "total_ms": 78.42,
    "avg_ms": 7.842,
    "min_ms": 7.1,
    "max_ms": 8.6,
    "min_max_available": true
  }
}
```

Semantic labels:

| `timing_semantics` | Customer label |
|---|---|
| `inclusive_processcvu_dispatcher_device_window` | CVU dispatcher/device window; not pure kernel math |
| `inclusive_processmla_dispatcher_device_window` | MLA dispatcher/device window; not pure kernel math |
| `backend_inner_event` | backend inner runtime event |
| `host_build_io` | host-side IO construction |
| `host_post_fixup` | host-side post/fixup |
| `diagnostic_plugin_event` | diagnostic plugin event |

### Edge metric fields

Add or enforce availability flags in `latency_ms`:

```json
{
  "edge_id": "e1",
  "from_node": "n9",
  "to_node": "n1",
  "samples": 25,
  "latency_ms": {
    "samples": 25,
    "total_ms": 0.781,
    "avg_ms": 0.031,
    "min_ms": 0.0,
    "max_ms": 0.049,
    "p50_ms": 0.0,
    "p95_ms": 0.0,
    "min_max_available": false,
    "percentiles_available": false,
    "max_semantics": "run_lifetime_high_water"
  },
  "source": "diagnostics",
  "timing_semantics": "queue_residence",
  "attribution_source": "stage_mailbox",
  "non_additive": true,
  "reliable": true
}
```

Visualizer rules:

- If `percentiles_available == false`, display p50/p95 as `N/A`, not `0.000 ms`.
- If `max_semantics == "run_lifetime_high_water"`, label max column as `Max (lifetime high-water)` or add a warning marker.
- Edge graph labels use avg only and include source/semantic, e.g. `queue avg 0.031 ms`.

### Path/timeline timing fields

The user-visible graph needs more than per-node latency: it must explain when a sample reaches each
plugin/node and how much time is spent between plugins. Add a separate timeline block. It is
separate from `node_metrics`, `plugin_metrics`, and `edge_metrics` because it uses per-sample
identity and is not always available.

Target JSON:

```json
"path_timing": {
  "available": true,
  "status": "collected",
  "source": "lttng_message_events",
  "aggregation": "measured_window",
  "identity": {
    "primary_key": "run_id_hash+graph_id_hash+message_id+edge_id",
    "fallback_key": "run_id_hash+stream_id+orig_input_seq+edge_id",
    "used_public_fields": ["stream_id", "orig_input_seq", "input_seq", "frame_id"],
    "sample_identity_source": "Sample/GstSimaSampleMeta"
  },
  "node_arrival_ms": [
    {
      "customer_node_id": "c1",
      "lowered_node_id": "n1",
      "runtime_node_id": 1,
      "label": "QuantTess",
      "semantics": "graph_entry_to_node_input_arrival",
      "source_edge_ids": ["ce0"],
      "lowered_edge_ids": ["e0"],
      "latency_ms": {
        "samples": 10,
        "avg_ms": 0.038,
        "p50_ms": 0.035,
        "p95_ms": 0.050,
        "max_ms": 0.055
      }
    }
  ],
  "inter_plugin_gap_ms": [
    {
      "customer_edge_id": "ce1",
      "lowered_edge_id": "e1",
      "from_plugin_instance_id": "r123.g456.s0.n1.n1_quanttess_1",
      "to_plugin_instance_id": "r123.g456.s1.n2.n2_model_1",
      "semantics": "upstream_plugin_end_to_downstream_plugin_start",
      "latency_ms": {
        "samples": 10,
        "avg_ms": 0.045,
        "p50_ms": 0.043,
        "p95_ms": 0.061,
        "max_ms": 0.066
      }
    }
  ],
  "output_tail_ms": [
    {
      "customer_output_id": "c4",
      "label": "detections",
      "semantics": "last_plugin_end_to_public_pull",
      "latency_ms": {
        "samples": 10,
        "avg_ms": 0.010
      }
    }
  ]
}
```

Unavailable JSON must be explicit:

```json
"path_timing": {
  "available": false,
  "status": "unavailable_sample_identity_missing",
  "source": "diagnostics",
  "reason": "diagnostic aggregate edge counters cannot compute graph-entry-to-node arrival",
  "aggregation": "measured_window"
}
```

Status enum:

```text
collected
partial
diagnostic_aggregate_only
unavailable_sample_identity_missing
unavailable_message_trace_disabled
unavailable_no_trace_capable_plugins
unavailable_lineage_ambiguous
failed
```

Visualizer rules:

- If available, draw optional timeline chips on nodes:
  - input/first compute: `entry→node avg 0.038 ms`
  - inter-node edge: `gap avg 0.045 ms`
  - output: `tail avg 0.010 ms`
- If unavailable, show a single visible message: `Timeline unavailable: <status>`.
- Never derive graph-entry-to-node timing by summing edge averages from different samples.
- Never hide N/A for first/last endpoint nodes; label them `endpoint / not timed` and show path timing separately when available.

### Power fields

Power is graph-level only. Use:

```json
"power": {
  "enabled": true,
  "status": "collected_with_errors",
  "scope": "board_rail_power_during_measured_window",
  "source": "pmbus_board_rails",
  "attribution": "graph_level_only",
  "samples": 3,
  "duration_seconds": 2.0,
  "total_avg_watts": 5.1,
  "total_min_watts": 4.9,
  "total_max_watts": 5.4,
  "energy_joules": 10.2,
  "rails": []
}
```

Status enum:

```text
disabled_by_options
not_configured_on_run
enabled_no_samples
unavailable_all_rails_failed
collected
collected_with_errors
```

DVT/SOM rule:

- DVT: do not validate numeric power correctness; validate status/shape/no crash.
- SOM: validate `samples > 0`, `total_avg_watts > 0`, rails if available.

### Metrics sources

Make `metrics_sources` structured, not bare strings:

```json
"metrics_sources": {
  "throughput": {
    "status": "collected",
    "source": "run_counters",
    "reliable": true
  },
  "node_latency": {
    "status": "collected",
    "source": "gst_element_timing",
    "semantics": "sum_element_residency_delta"
  },
  "plugin_latency": {
    "status": "collected",
    "source": "lttng",
    "trace_loss_detected": false
  },
  "edge_message_latency": {
    "status": "collected",
    "source": "diagnostics",
    "trace_loss_detected": false
  },
  "path_timing": {
    "status": "diagnostic_aggregate_only",
    "source": "diagnostics",
    "reliable": true
  },
  "power": {
    "status": "disabled_by_options",
    "source": "pmbus_board_rails",
    "attribution": "graph_level_only"
  }
}
```

---

## Implementation plan

### Phase 0: Repository hygiene and release blockers

Files currently untracked but required by current implementation:

```text
src/pipeline/runtime/EdgeMetrics.{h,cpp}
src/pipeline/runtime/LttngMetricsCollector.h
src/pipeline/runtime/LttngTraceParser.cpp
src/pipeline/runtime/LttngTraceSession.cpp
src/pipeline/runtime/TraceAttribution.{h,cpp}
src/pipeline/gst/TraceIdentity.{h,cpp}
tests/graph_migration/unified/phaseA4_branched_report_demo.cpp
tests/graph_migration/unified/phaseA4_customer_view_combine_export_test.cpp
```

Action:

- Add them intentionally or remove references before any release branch.
- Acceptance: clean checkout builds without relying on untracked local files.
- Failure condition: `RunMeasure.cpp` includes a missing header after a fresh clone.

Also carry forward these release blockers:

- People-tracker metrics teardown aborts must be fixed or explicitly scoped out.
- Any remaining ABI change in `include/pipeline/Run.h` must be handled by ABI bump or documented SDK rebuild requirement; avoid changing `include/graph/StageExecutor.h`.
- Current DETR artifacts use fallback topology; strict customer gate should fail until real topology is exported.

### Phase 1: Export `graph.customer_view`

Primary files:

```text
src/pipeline/runtime/CustomerGraphView.h        (new private header)
src/pipeline/runtime/CustomerGraphView.cpp      (new implementation)
src/pipeline/runtime/GraphRunExport.cpp         (wire into JSON export only)
```

Build-system fact:

- `core_graph_changes/CMakeLists.txt` uses `file(GLOB_RECURSE SIMANEAT_ALL_SOURCES ...)`, so new
  `.cpp` files under `src/` are picked up after CMake reconfigure. There is no per-source CMake
  list to edit for `CustomerGraphView.cpp`.
- Still verify the generated build includes the new object after configure; do not assume an
  already-configured build directory automatically notices new files.

Implementation rule:

- Build the customer view from `runtime::ExecutionGraphPlan`, not by reparsing JSON as the
  primary source. The public/lowered JSON already contains useful identity blocks, but the stable
  source of truth is the typed plan:
  - `plan.public_nodes`
  - `plan.public_edges`
  - `plan.edges`
  - `plan.pipeline_segments`
  - `plan.stage_nodes`
  - `plan.named_inputs`
  - `plan.named_outputs`
- JSON helper maps are acceptable only as a convenience for copying already-rendered identity
  blocks such as `source`, `sink`, and `model`.
- This avoids a fragile exporter that serializes topology to JSON and then parses its own strings
  back into graph IDs.

Public/private function split:

```cpp
// src/pipeline/runtime/CustomerGraphView.h
#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/runtime/ExecutionGraphPlan.h"
#include <nlohmann/json.hpp>

namespace simaai::neat::runtime {

nlohmann::ordered_json customer_view_to_json(const ExecutionGraphPlan& plan,
                                             const nlohmann::ordered_json& public_view,
                                             const nlohmann::ordered_json& lowered_view);

nlohmann::ordered_json fallback_customer_view_from_lowered(
    const nlohmann::ordered_json& lowered_view,
    std::string topology_source,
    std::vector<std::string> warnings);

} // namespace simaai::neat::runtime
```

`GraphRunExport.cpp` should only:

- include `pipeline/runtime/CustomerGraphView.h`;
- call `runtime::customer_view_to_json(...)` from `graph_topology_to_json()`;
- call `runtime::fallback_customer_view_from_lowered(...)` from fallback topology export.

Functions to change in `GraphRunExport.cpp`:

```text
public_view_to_json(const runtime::ExecutionGraphPlan& plan)
graph_topology_to_json(const runtime::RunCore& core)
ensure_graph_topology_from_node_metrics(json& root)
```

Add these implementation structs in `CustomerGraphView.cpp`, not in `GraphRunExport.cpp`:

```cpp
struct CustomerNodeBuild {
  std::string id;
  std::string kind;
  std::string role;
  std::string label;
  std::string endpoint_name;
  std::vector<std::string> public_node_ids;
  std::vector<std::string> lowered_node_ids;
  std::vector<std::int32_t> runtime_node_ids;
  bool compiler_generated = false;
  std::string mapping_status = "mapped";
  std::string mapping_error;
  json source;
  json sink;
  json model;
};

struct CustomerEdgeBuild {
  std::string id;
  std::string from;
  std::string to;
  std::string kind;
  std::string label;
  std::string from_endpoint;
  std::string to_endpoint;
  std::vector<std::string> public_edge_ids;
  std::vector<std::string> lowered_edge_ids;
  std::string metric_policy = "none";
  std::string mapping_status = "mapped";
  std::string mapping_error;
};
```

Use these helper names. Keep the JSON parsing helpers as compatibility helpers only; the typed
`PlanTopologyIndex` path below is the primary implementation path:

```cpp
std::optional<graph::NodeId> runtime_node_id_from_ref(const json& value);
std::unordered_map<std::string, json> nodes_by_id(const json& nodes);
std::unordered_map<std::string, json> edges_by_id(const json& edges);
std::unordered_map<std::string, std::vector<std::string>> lowered_out_edges(const json& lowered_edges);
std::unordered_map<std::string, std::vector<std::string>> lowered_in_edges(const json& lowered_edges);
std::vector<std::string> shortest_lowered_edge_path(
    const json& lowered_edges,
    std::string from_node,
    std::string to_node);
json customer_view_to_json(const runtime::ExecutionGraphPlan& plan,
                           const json& public_view,
                           const json& lowered_view);
```

Preferred typed index:

```cpp
struct PlanTopologyIndex {
  std::unordered_map<graph::NodeId, std::vector<std::size_t>> lowered_out_edges;
  std::unordered_map<graph::NodeId, std::vector<std::size_t>> lowered_in_edges;
  std::unordered_map<std::size_t, std::vector<std::size_t>> public_out_edges;
  std::unordered_map<std::size_t, std::vector<std::size_t>> public_in_edges;
  std::unordered_map<graph::NodeId, std::size_t> runtime_to_stage_node;
  std::unordered_map<graph::NodeId, std::size_t> runtime_to_segment;
  std::unordered_map<graph::NodeId, std::string> runtime_label;
  std::unordered_map<graph::NodeId, std::string> runtime_kind;
};

PlanTopologyIndex build_plan_topology_index(const ExecutionGraphPlan& plan) {
  PlanTopologyIndex idx;
  for (std::size_t e = 0; e < plan.edges.size(); ++e) {
    const auto& edge = plan.edges[e];
    if (edge.from != graph::kInvalidNode) idx.lowered_out_edges[edge.from].push_back(e);
    if (edge.to != graph::kInvalidNode) idx.lowered_in_edges[edge.to].push_back(e);
  }
  for (std::size_t e = 0; e < plan.public_edges.size(); ++e) {
    const auto& edge = plan.public_edges[e];
    if (edge.from < plan.public_nodes.size()) {
      idx.public_out_edges[edge.from].push_back(e);
    }
    if (edge.to < plan.public_nodes.size()) {
      idx.public_in_edges[edge.to].push_back(e);
    }
  }
  for (std::size_t i = 0; i < plan.stage_nodes.size(); ++i) {
    const auto& st = plan.stage_nodes[i];
    idx.runtime_to_stage_node[st.node_id] = i;
    idx.runtime_label[st.node_id] =
        st.node_id < plan.node_labels.size() ? plan.node_labels[st.node_id] : std::string{};
    idx.runtime_kind[st.node_id] = "stage";
  }
  for (std::size_t i = 0; i < plan.pipeline_segments.size(); ++i) {
    const auto& seg = plan.pipeline_segments[i];
    for (const graph::NodeId node_id : seg.node_ids) {
      idx.runtime_to_segment[node_id] = i;
      idx.runtime_label[node_id] =
          node_id < plan.node_labels.size() ? plan.node_labels[node_id] : std::string{};
      idx.runtime_kind[node_id] = "pipeline_segment";
    }
  }
  return idx;
}
```

Use typed edge IDs only at export boundary:

```cpp
std::string runtime_node_json_id(graph::NodeId id) {
  return id == graph::kInvalidNode ? "invalid" : "n" + std::to_string(id);
}
std::string lowered_edge_json_id(std::size_t edge_index) {
  return "e" + std::to_string(edge_index);
}
std::string public_node_json_id(std::size_t public_index) {
  return "p" + std::to_string(public_index);
}
std::string public_edge_json_id(std::size_t public_edge_index) {
  return "pe" + std::to_string(public_edge_index);
}
```

#### Deterministic endpoint ordering

`ExecutionGraphPlan::named_inputs` and `named_outputs` are maps in the current code, but the plan
should not rely on their container iteration order for customer visuals. Add a local helper that
uses public-node order as the primary order, then falls back to `(endpoint name, node, port, kind)`
for stability:

```cpp
constexpr std::size_t npos = static_cast<std::size_t>(-1);

struct NamedEndpointRef {
  std::string name;
  Endpoint endpoint;
  std::size_t public_node_index = npos;
};

std::vector<NamedEndpointRef>
ordered_named_endpoints(const ExecutionGraphPlan& plan, bool inputs) {
  const auto& map = inputs ? plan.named_inputs : plan.named_outputs;
  std::vector<NamedEndpointRef> out;
  out.reserve(map.size());
  for (const auto& [name, endpoint] : map) {
    NamedEndpointRef ref{name, endpoint, npos};
    for (std::size_t i = 0; i < plan.public_nodes.size(); ++i) {
      const auto& pn = plan.public_nodes[i];
      if ((inputs && pn.input_endpoint) || (!inputs && pn.output_endpoint)) {
        if (pn.endpoint_name == name ||
            pn.runtime_node == endpoint.node) {
          ref.public_node_index = i;
          break;
        }
      }
    }
    out.push_back(std::move(ref));
  }
  std::stable_sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
    if (a.public_node_index != b.public_node_index) return a.public_node_index < b.public_node_index;
    if (a.name != b.name) return a.name < b.name;
    if (a.endpoint.node != b.endpoint.node) return a.endpoint.node < b.endpoint.node;
    if (a.endpoint.port != b.endpoint.port) return a.endpoint.port < b.endpoint.port;
    return static_cast<int>(a.endpoint.kind) < static_cast<int>(b.endpoint.kind);
  });
  return out;
}
```

This helper feeds only layout/export ordering. It must not mutate `ExecutionGraphPlan`.

#### Lowered path lookup

Implement shortest-path on typed `plan.edges`, not on JSON. This is needed for customer edges that
collapse public pass-through nodes or map one customer edge to multiple lowered runtime edges.

```cpp
std::vector<std::size_t> shortest_lowered_edge_path(const ExecutionGraphPlan& plan,
                                                    const PlanTopologyIndex& idx,
                                                    graph::NodeId from,
                                                    graph::NodeId to) {
  if (from == graph::kInvalidNode || to == graph::kInvalidNode) return {};
  if (from == to) return {};
  struct Prev { graph::NodeId prev = graph::kInvalidNode; std::size_t edge = npos; };
  std::queue<graph::NodeId> q;
  std::unordered_map<graph::NodeId, Prev> prev;
  q.push(from);
  prev[from] = {};
  while (!q.empty()) {
    const auto n = q.front();
    q.pop();
    const auto out_it = idx.lowered_out_edges.find(n);
    if (out_it == idx.lowered_out_edges.end()) continue;
    for (const std::size_t eidx : out_it->second) {
      const auto next = plan.edges[eidx].to;
      if (next == graph::kInvalidNode || prev.count(next)) continue;
      prev[next] = Prev{n, eidx};
      if (next == to) {
        std::vector<std::size_t> path;
        for (auto cur = to; cur != from; cur = prev[cur].prev) path.push_back(prev[cur].edge);
        std::reverse(path.begin(), path.end());
        return path;
      }
      q.push(next);
    }
  }
  return {};
}
```

`npos` above is a local `constexpr std::size_t npos = static_cast<std::size_t>(-1);`.

#### Customer nodes, routers, and serializers

Keep all temporary state in `CustomerGraphView.cpp` so `GraphRunExport.cpp` remains only a JSON
assembly file:

```cpp
bool has_runtime_node(const PublicGraphNodePlan& node) {
  return node.runtime_node != graph::kInvalidNode;
}

std::string node_role(const PublicGraphNodePlan& node) {
  if (node.input_endpoint) return "input";
  if (node.output_endpoint) return "output";
  return "compute";
}

std::string customer_label(const PublicGraphNodePlan& node,
                           const PlanTopologyIndex& idx) {
  if (!node.label.empty()) return node.label;
  if (!node.endpoint_name.empty()) return node.endpoint_name;
  const auto it = idx.runtime_label.find(node.runtime_node);
  if (it != idx.runtime_label.end() && !it->second.empty()) return it->second;
  return node.kind.empty() ? ("node " + std::to_string(node.id)) : node.kind;
}

CustomerNodeBuild make_public_customer_node(const PublicGraphNodePlan& node,
                                            const PlanTopologyIndex& idx,
                                            const json* public_node_json) {
  CustomerNodeBuild out;
  out.id = "c" + std::to_string(node.id);
  out.kind = node.input_endpoint ? "Input" : node.output_endpoint ? "Output" :
             node.kind.empty() ? "Operator" : node.kind;
  out.role = node_role(node);
  out.label = customer_label(node, idx);
  out.endpoint_name = node.endpoint_name;
  out.public_node_ids = {public_node_json_id(node.id)};
  if (has_runtime_node(node)) {
    out.lowered_node_ids = {runtime_node_json_id(node.runtime_node)};
    out.runtime_node_ids = {static_cast<std::int32_t>(node.runtime_node)};
  }
  if (public_node_json != nullptr) {
    if (public_node_json->contains("source")) out.source = (*public_node_json)["source"];
    if (public_node_json->contains("sink")) out.sink = (*public_node_json)["sink"];
    if (public_node_json->contains("model")) out.model = (*public_node_json)["model"];
  }
  return out;
}
```

Router detection uses public edges for intent and lowered edges for metric mapping. Do not classify
router nodes from labels alone:

```cpp
enum class RouterKind { Branch, Combine, PassThrough, None };

RouterKind classify_invalid_public_node(std::size_t public_index,
                                         const PlanTopologyIndex& idx) {
  const auto out = idx.public_out_edges.find(public_index);
  const auto in = idx.public_in_edges.find(public_index);
  const std::size_t out_degree = out == idx.public_out_edges.end() ? 0 : out->second.size();
  const std::size_t in_degree = in == idx.public_in_edges.end() ? 0 : in->second.size();
  if (out_degree > 1 && in_degree >= 1) return RouterKind::Branch;
  if (in_degree > 1 && out_degree >= 1) return RouterKind::Combine;
  if (in_degree == 1 && out_degree == 1) return RouterKind::PassThrough;
  return RouterKind::None;
}

std::vector<std::size_t> nearest_valid_public_nodes(const ExecutionGraphPlan& plan,
                                                    const PlanTopologyIndex& idx,
                                                    std::size_t start,
                                                    bool reverse) {
  std::vector<std::size_t> found;
  std::queue<std::size_t> q;
  std::unordered_set<std::size_t> seen;
  q.push(start);
  seen.insert(start);
  while (!q.empty()) {
    const std::size_t cur = q.front();
    q.pop();
    const auto& edge_map = reverse ? idx.public_in_edges : idx.public_out_edges;
    const auto it = edge_map.find(cur);
    if (it == edge_map.end()) continue;
    for (const std::size_t peidx : it->second) {
      const auto& pe = plan.public_edges[peidx];
      const std::size_t next = reverse ? pe.from : pe.to;
      if (next >= plan.public_nodes.size() || !seen.insert(next).second) continue;
      if (has_runtime_node(plan.public_nodes[next])) {
        found.push_back(next);
      } else {
        q.push(next);
      }
    }
  }
  std::stable_sort(found.begin(), found.end());
  return found;
}
```

For lowered router matching, use the shortest paths between valid public endpoints. Pick a concrete
runtime node only when all required paths agree; otherwise keep the router visible but mark it
`partial` and let its edges carry the lowered paths.

```cpp
std::optional<graph::NodeId>
find_branch_runtime_node(const ExecutionGraphPlan& plan,
                         const PlanTopologyIndex& idx,
                         graph::NodeId upstream,
                         const std::vector<graph::NodeId>& downstreams,
                         const std::unordered_map<std::string, json>& lowered_nodes_by_id) {
  std::unordered_map<graph::NodeId, std::size_t> hit_count;
  std::unordered_map<graph::NodeId, std::size_t> distance_sum;
  for (const graph::NodeId downstream : downstreams) {
    const auto path = shortest_lowered_edge_path(plan, idx, upstream, downstream);
    std::size_t dist = 0;
    for (const std::size_t eidx : path) {
      const graph::NodeId n = plan.edges[eidx].from;
      hit_count[n] += 1;
      distance_sum[n] += dist++;
    }
  }
  std::vector<graph::NodeId> candidates;
  for (const auto& [node, count] : hit_count) {
    const auto out_it = idx.lowered_out_edges.find(node);
    const std::size_t out_degree =
        out_it == idx.lowered_out_edges.end() ? 0U : out_it->second.size();
    if (count == downstreams.size() && out_degree > 1U) {
      candidates.push_back(node);
    }
  }
  auto fanout_score = [&](graph::NodeId n) {
    const auto json_it = lowered_nodes_by_id.find(runtime_node_json_id(n));
    const std::string kind =
        json_it == lowered_nodes_by_id.end() ? "" : json_it->second.value("kind", "");
    return (kind == "FanOut" || kind == "Tee" || kind == "Fork") ? 0 : 1;
  };
  std::stable_sort(candidates.begin(), candidates.end(), [&](auto a, auto b) {
    if (fanout_score(a) != fanout_score(b)) return fanout_score(a) < fanout_score(b);
    if (distance_sum[a] != distance_sum[b]) return distance_sum[a] < distance_sum[b];
    return a < b;
  });
  if (candidates.empty()) return std::nullopt;
  return candidates.front();
}
```

Combine uses the same algorithm in reverse:

```cpp
std::optional<graph::NodeId>
find_combine_runtime_node(const ExecutionGraphPlan& plan,
                          const PlanTopologyIndex& idx,
                          const std::vector<graph::NodeId>& upstreams,
                          graph::NodeId downstream,
                          const std::unordered_map<std::string, json>& lowered_nodes_by_id) {
  std::unordered_map<graph::NodeId, std::size_t> hit_count;
  std::unordered_map<graph::NodeId, std::size_t> distance_sum;
  for (const graph::NodeId upstream : upstreams) {
    const auto path = shortest_lowered_edge_path(plan, idx, upstream, downstream);
    std::size_t dist = 0;
    for (const std::size_t eidx : path) {
      const graph::NodeId n = plan.edges[eidx].to;
      hit_count[n] += 1;
      distance_sum[n] += dist++;
    }
  }
  std::vector<graph::NodeId> candidates;
  for (const auto& [node, count] : hit_count) {
    const auto in_it = idx.lowered_in_edges.find(node);
    const std::size_t in_degree =
        in_it == idx.lowered_in_edges.end() ? 0U : in_it->second.size();
    if (count == upstreams.size() && in_degree > 1U) {
      candidates.push_back(node);
    }
  }
  auto join_score = [&](graph::NodeId n) {
    const auto json_it = lowered_nodes_by_id.find(runtime_node_json_id(n));
    const std::string kind =
        json_it == lowered_nodes_by_id.end() ? "" : json_it->second.value("kind", "");
    return (kind == "JoinBundle" || kind == "Join" || kind == "Combine") ? 0 : 1;
  };
  auto in_degree = [&](graph::NodeId n) {
    const auto it = idx.lowered_in_edges.find(n);
    return it == idx.lowered_in_edges.end() ? 0U : it->second.size();
  };
  std::stable_sort(candidates.begin(), candidates.end(), [&](auto a, auto b) {
    if (join_score(a) != join_score(b)) return join_score(a) < join_score(b);
    if (in_degree(a) != in_degree(b)) return in_degree(a) > in_degree(b);
    if (distance_sum[a] != distance_sum[b]) return distance_sum[a] < distance_sum[b];
    return a < b;
  });
  if (candidates.empty()) return std::nullopt;
  return candidates.front();
}
```

The JSON serializer must explicitly emit `null` for absent optional strings so the UI does not have
to distinguish missing from intentionally unavailable:

```cpp
json customer_node_to_json(const CustomerNodeBuild& n) {
  json out;
  out["id"] = n.id;
  out["kind"] = n.kind;
  out["role"] = n.role;
  out["label"] = n.label;
  out["endpoint_name"] = n.endpoint_name.empty() ? json(nullptr) : json(n.endpoint_name);
  out["public_node_ids"] = n.public_node_ids;
  out["lowered_node_ids"] = n.lowered_node_ids;
  out["runtime_node_ids"] = n.runtime_node_ids;
  out["compiler_generated"] = n.compiler_generated;
  out["mapping_status"] = n.mapping_status;
  if (!n.mapping_error.empty()) out["mapping_error"] = n.mapping_error;
  if (!n.source.is_null()) out["source"] = n.source;
  if (!n.sink.is_null()) out["sink"] = n.sink;
  if (!n.model.is_null()) out["model"] = n.model;
  return out;
}

json customer_edge_to_json(const CustomerEdgeBuild& e) {
  json out;
  out["id"] = e.id;
  out["from"] = e.from;
  out["to"] = e.to;
  out["kind"] = e.kind;
  out["label"] = e.label;
  out["from_endpoint"] = e.from_endpoint.empty() ? json(nullptr) : json(e.from_endpoint);
  out["to_endpoint"] = e.to_endpoint.empty() ? json(nullptr) : json(e.to_endpoint);
  out["public_edge_ids"] = e.public_edge_ids;
  out["lowered_edge_ids"] = e.lowered_edge_ids;
  out["metric_policy"] = e.metric_policy;
  out["mapping_status"] = e.mapping_status;
  if (!e.mapping_error.empty()) out["mapping_error"] = e.mapping_error;
  return out;
}
```

#### Deterministic customer-view construction algorithm

1. Build maps:

```cpp
public_node_by_id[p.id]
public_edge_by_id[pe.id]
lowered_node_by_id[n.id]
lowered_edge_by_id[e.id]
```

2. Classify public nodes:

```cpp
bool valid_runtime = runtime_node != "invalid" && runtime_node != "" && runtime_node != null;
bool is_input = input_endpoint == true;
bool is_output = output_endpoint == true;
```

3. Create endpoint/compute customer nodes for all valid public nodes:

```text
kind = public.kind
role = input_endpoint ? "input" : output_endpoint ? "output" : "compute"
public_node_ids = [p*]
lowered_node_ids = [runtime_node]
runtime_node_ids = parsed runtime node id
```

4. Treat invalid public nodes as virtual topology helpers:

- invalid node with out-degree > 1: Branch router candidate.
- invalid node with in-degree > 1: Combine router candidate.
- invalid node with one in and one out: pass-through; do not show.

5. Map Branch router to lowered generated node:

For invalid branch public node `pB`:

- Identify nearest upstream valid public node by walking reverse public edges until `runtime_node` is valid.
- Identify downstream valid public nodes by walking forward public edges until runtime nodes are valid.
- Let upstream lowered node be `u` and downstream lowered nodes be `d[]`.
- Find lowered node `r` such that:
  - it is reachable from `u`, and
  - it reaches all `d[]`, and
  - lowered out-degree > 1, or lowered kind is `FanOut`.
- Prefer exact lowered kind `FanOut`; else choose first common divergence by shortest path distance.
- Exclude valid public endpoint/runtime nodes as router candidates unless the lowered node kind is
  an explicit generated router (`FanOut`, `Tee`, `Fork`, `JoinBundle`, `Join`, `Combine`).
- If no distinct lowered router is found, keep the customer router virtual with
  `mapping_status="partial"`, `mapping_error="No distinct lowered router node found; customer router is virtual."`,
  and empty `lowered_node_ids`; customer edges still carry lowered paths.
- Emit customer node kind `Branch`, role `router`, lowered_node_ids `[r]`.

6. Map Combine router similarly:

- Many upstream valid public nodes, one downstream valid public node.
- Prefer lowered kind `JoinBundle` or lowered in-degree > 1.
- Emit customer node kind `Combine`, role `router`.

7. Create customer edges:

- Input/compute -> Branch uses lowered path from upstream lowered node to router lowered node.
- Branch -> output/compute uses lowered path from router lowered node to downstream lowered node.
- Compute -> compute with no router uses public `runtime_edges` when available; else shortest lowered path.
- Degree-1 invalid pass-through nodes collapse into the edge and are recorded in `public_edge_ids`.

8. Set `metric_policy`:

```cpp
if lowered_edge_ids.empty(): "none"
else if lowered_edge_ids.size() == 1: "single_lowered_edge"
else: "non_additive_representative_max"
```

9. Add mapping objects:

```json
"customer_to_public": {"c0": ["p0"]}
"public_to_customer": {"p0": "c0"}
"customer_to_lowered": {"c0": ["n0"]}
"lowered_to_customer": {"n0": "c0"}
"customer_edge_to_public": {"ce0": ["pe0", "pe1"]}
"customer_edge_to_lowered": {"ce0": ["e0", "e1"]}
```

10. If customer view cannot be built:

- still emit `customer_view` with `topology_source = "node_metrics_fallback"` or `"lowered_fallback"`.
- put a warning under `customer_view.warnings[]`.

#### Wire-in details

In `graph_topology_to_json()`:

```cpp
graph["public_view"] = public_view_to_json(plan);
graph["public_view"]["topology_source"] = "execution_plan_public_view";
// Existing code then fills graph["nodes"], graph["edges"], and graph["pipeline_segments"].
graph["lowered_view"] = {
  {"topology_source", "execution_plan_lowered_view"},
  {"nodes", graph["nodes"]},
  {"edges", graph["edges"]},
  {"pipeline_segments", graph["pipeline_segments"]},
};
graph["customer_view"] = customer_view_to_json(plan, graph["public_view"],
                                               graph["lowered_view"]);
```

In `ensure_graph_topology_from_node_metrics()`:

```cpp
if (using fallback) {
  graph["customer_view"] = fallback_customer_view_from_lowered(
      graph["lowered_view"],
      "node_metrics_fallback",
      {"Fallback topology reconstructed from node metrics; not source customer topology."});
}
```

### Phase 2: Fix measured-window JSON semantics

Primary files:

```text
include/pipeline/Run.h
src/pipeline/runtime/RunMeasure.cpp
src/pipeline/runtime/GraphMetricJson.h          (new private helper header)
src/pipeline/runtime/GraphMetricJson.cpp        (new canonical JSON calculations)
src/pipeline/runtime/GraphRunExport.cpp
src/pipeline/runtime/EdgeMetrics.cpp
src/pipeline/runtime/GraphMetrics.cpp
schemas/graph_run_v1.schema.json
```

Build-system fact:

- `GraphMetricJson.cpp` is also picked up by the existing recursive CMake source glob after
  reconfigure. The implementation still needs one include in `GraphRunExport.cpp`; do not duplicate
  helper functions in anonymous namespaces after moving them.

#### Canonical JSON helper module

Create one private helper module for customer-facing graph numbers. This is how we avoid lifetime
and measured-window calculations drifting apart.

```cpp
// src/pipeline/runtime/GraphMetricJson.h
#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/PowerTelemetry.h"
#include "pipeline/Run.h"
#include "pipeline/GraphMetrics.h"
#include <nlohmann/json.hpp>

namespace simaai::neat::runtime {

nlohmann::ordered_json throughput_json(std::uint64_t output_pulls,
                                       double elapsed_s,
                                       std::uint32_t logical_batch_size);

nlohmann::ordered_json window_json(std::string_view scope,
                                   double elapsed_s,
                                   int requested_duration_ms,
                                   int requested_warmup_ms,
                                   std::uint64_t warmup_iterations_excluded);

nlohmann::ordered_json power_status_json(const PowerSummary& power,
                                         bool export_requested,
                                         bool monitor_configured,
                                         std::string_view scope);

nlohmann::ordered_json graph_e2e_json(const MeasureLatencyStats& stats,
                                      bool graph_backed,
                                      std::string_view status_if_empty);

nlohmann::ordered_json path_timing_to_json(const MeasurePathTiming& timing,
                                           bool trace_loss_detected);

} // namespace simaai::neat::runtime
```

Implementation details:

```cpp
nlohmann::ordered_json throughput_json(std::uint64_t output_pulls,
                                       double elapsed_s,
                                       std::uint32_t logical_batch_size) {
  const bool valid_window = elapsed_s > 0.0 && std::isfinite(elapsed_s);
  const double outputs_per_s = valid_window ? static_cast<double>(output_pulls) / elapsed_s : 0.0;
  const std::uint32_t batch = logical_batch_size == 0 ? 1U : logical_batch_size;
  return {
      {"unit", "output_pulls_per_second"},
      {"semantics", "public_output_pulls_per_second"},
      {"numerator_counter", "outputs_pulled"},
      {"numerator_value", output_pulls},
      {"denominator", "elapsed_seconds"},
      {"denominator_seconds", elapsed_s},
      {"outputs_pulled", output_pulls},
      {"elapsed_seconds", elapsed_s},
      {"outputs_per_s", outputs_per_s},
      {"logical_batch_size", batch},
      {"logical_inferences_per_s", outputs_per_s * static_cast<double>(batch)},
      {"multi_output_semantics", "each successful public pull increments outputs_pulled once"},
      {"available", valid_window},
      {"status", valid_window ? "collected" : "invalid_window"},
  };
}
```

`power_status_json()` is the only place that emits power status strings:

```cpp
const auto rail_power_successes = sum(power.rails[*].power_w.samples);
const auto rail_power_errors = sum(power.rails[*].power_w.errors);
if (!export_requested) status = "disabled_by_options";
else if (!monitor_configured) status = "not_configured_on_run";
else if (power.samples == 0 && rail_power_errors == 0) status = "enabled_no_samples";
else if (power.samples == 0 && rail_power_errors > 0) status = "unavailable_all_rails_failed";
else if (rail_power_errors > 0) status = "collected_with_errors";
else status = "collected";
```

The helper should always add:

```json
{
  "scope": "board_rail_power_during_measured_window",
  "attribution": "graph_level_only",
  "customer_note": "Power is board/SOM rail telemetry. DVT board readings may be unavailable or unreliable."
}
```

Replace existing local helpers in `GraphRunExport.cpp`:

```text
disabled_power_json()
manual throughput_fps / outputs_per_s assignments
manual measured-window power status branches
manual graph_e2e_latency_ms JSON assembly
```

Compatibility fields remain, but are copied from canonical helpers:

```cpp
const auto throughput = runtime::throughput_json(report.outputs_pulled,
                                                report.elapsed_s,
                                                report.options.logical_batch_size);
graph_metrics["throughput"] = throughput;
graph_metrics["throughput_fps"] = throughput["outputs_per_s"]; // compatibility only
graph_metrics["outputs_per_s"] = throughput["outputs_per_s"];  // compatibility only
```

#### Add missing measured counters

In `include/pipeline/Run.h`, `struct MeasureReport`:

```cpp
std::uint64_t inputs_enqueued = 0;
std::uint64_t outputs_ready = 0;
bool trace_loss_detected = false;
```

In `src/pipeline/runtime/RunMeasure.cpp`, `MeasureScope::stop()`:

```cpp
report.inputs_enqueued = measured.inputs_enqueued;
report.outputs_ready = measured.outputs_ready;
// After LTTng parse succeeds:
if (parsed.trace_loss_detected) {
  report.trace_loss_detected = true;
}
```

#### Emit measured fields without lifetime mixing

In `run_to_json(const Run&, const MeasureReport&, ...)`:

- Keep `run.stats` for compatibility but add `run.stats_scope = "run_lifetime"`.
- Add `run.measured_stats` for deltas.
- Add `run.graph_metrics.window` and `run.graph_metrics.throughput`.

Snippet:

```cpp
run_json["stats_scope"] = "run_lifetime";
run_json["measured_stats"] = {
  {"inputs_enqueued", report.inputs_enqueued},
  {"inputs_dropped", report.inputs_dropped},
  {"inputs_pushed", report.inputs_pushed},
  {"outputs_ready", report.outputs_ready},
  {"outputs_pulled", report.outputs_pulled},
  {"outputs_dropped", report.outputs_dropped},
};

json& gm = run_json["graph_metrics"];
gm["window"] = {
  {"scope", "measured_window"},
  {"start_source", "Run::start_measurement"},
  {"stop_source", "MeasureScope::stop"},
  {"elapsed_seconds", report.elapsed_s},
  {"requested_duration_ms", report.options.duration_ms},
  {"requested_warmup_ms", report.options.warmup_ms},
  {"duration_enforced_by_scope", false},
  {"warmup_iterations_excluded", report.warmup_iterations},
};
gm["throughput"] = throughput_json(report.outputs_pulled, report.elapsed_s,
                                    report.options.logical_batch_size);
```

#### Graph E2E latency availability

In measured JSON export:

```cpp
const std::shared_ptr<const runtime::RunCore> core = run_internal::core(run);
const bool graph_backed = core && core->graph_execution_;
json e2e;
e2e["semantics"] = "public_push_to_public_pull";
if (report.end_to_end.count > 0) {
  e2e["available"] = true;
  e2e["status"] = "collected";
  e2e["source"] = graph_backed ? "graph_sample_identity" : "linear_pending_times";
  e2e["count"] = report.end_to_end.count;
  e2e["avg_ms"] = report.end_to_end.avg_ms;
  e2e["p50_ms"] = report.end_to_end.p50_ms;
  e2e["p90_ms"] = report.end_to_end.p90_ms;
  e2e["p95_ms"] = report.end_to_end.p95_ms;
  e2e["p99_ms"] = report.end_to_end.p99_ms;
  e2e["max_ms"] = report.end_to_end.max_ms;
} else {
  e2e["available"] = false;
  e2e["status"] = graph_backed ?
      "unavailable_graph_e2e_not_instrumented" : "no_samples";
  e2e["source"] = "none";
  e2e["count"] = 0;
}
run_json["graph_e2e_latency_ms"] = std::move(e2e);
```

Graph-backed E2E implementation goes in these files:

```text
src/pipeline/runtime/RunPush.cpp
src/pipeline/runtime/RunPull.cpp
src/pipeline/runtime/RunCore.h
src/pipeline/runtime/RunMeasure.cpp
src/pipeline/runtime/GraphRunExport.cpp
```

Use existing public sample identity first. Do **not** add a public `Sample` ABI field unless the
ABI bump is already accepted. The existing fields are enough for phase 1:

```text
Sample::stream_id
Sample::frame_id
Sample::input_seq
Sample::orig_input_seq
GstSimaSampleMeta fields: stream-id, frame-id, input-seq, orig-input-seq
PipelineSegmentRuntime::GraphTransport::SampleIdentity
```

Add private timing state in `RunCore.h`:

```cpp
enum class GraphSampleTimingKeyKind { OrigInputSeq, InputSeq, FrameId };

struct GraphSampleIdentityKey {
  std::string stream_id;
  GraphSampleTimingKeyKind kind = GraphSampleTimingKeyKind::OrigInputSeq;
  std::int64_t value = -1;
};

struct GraphSampleIdentityKeyHash {
  std::size_t operator()(const GraphSampleIdentityKey& k) const noexcept;
};

struct GraphSampleTimingState {
  std::chrono::steady_clock::time_point graph_entry_at;
  std::string input_endpoint;
  std::uint64_t generation = 0;
  bool ambiguous = false;
};

struct GraphSampleTimingOrderEntry {
  GraphSampleIdentityKey key;
  std::uint64_t generation = 0;
};

struct GraphSampleTimingEvent {
  std::string endpoint;
  std::string stream_id;
  GraphSampleTimingKeyKind key_kind = GraphSampleTimingKeyKind::OrigInputSeq;
  std::int64_t key_value = -1;
  double timestamp_s = 0.0; // steady-clock seconds for graph E2E diagnostics only, not LTTng path timing.
};

std::mutex graph_sample_timing_mu;
std::unordered_map<GraphSampleIdentityKey, GraphSampleTimingState,
                   GraphSampleIdentityKeyHash> graph_sample_timing_by_key;
std::deque<GraphSampleTimingOrderEntry> graph_sample_timing_order;
std::size_t graph_sample_timing_capacity = 4096;
std::uint64_t graph_sample_timing_generation = 0;
std::atomic<std::uint64_t> graph_sample_timing_unkeyed{0};
std::atomic<std::uint64_t> graph_sample_timing_misses{0};
std::chrono::steady_clock::time_point measurement_started_at{};
std::vector<GraphSampleTimingEvent> measurement_graph_entries;
std::vector<GraphSampleTimingEvent> measurement_graph_pulls;
```

Key equality rules:

1. Prefer `(stream_id, orig_input_seq)` when `orig_input_seq >= 0`.
2. Else use `(stream_id, input_seq)` when `input_seq >= 0`.
3. Else use `(stream_id, frame_id)` when `frame_id >= 0`.
4. Else do not insert; export status `unavailable_sample_identity_missing`.
5. Endpoint is stored in `GraphSampleTimingState`, not the key. If two pending samples share the
   same identity key, mark the state ambiguous instead of guessing.

Make those rules explicit in one helper. Do not encode fallback priority inside `operator==`,
because unordered-map equality must be stable for an already-built key:

```cpp
std::optional<GraphSampleIdentityKey>
make_graph_sample_identity_key(const Sample& sample) {
  const std::string stream = sample.stream_id.empty() ? "default" : sample.stream_id;
  if (sample.orig_input_seq >= 0) {
    return GraphSampleIdentityKey{stream, GraphSampleTimingKeyKind::OrigInputSeq,
                                  sample.orig_input_seq};
  }
  if (sample.input_seq >= 0) {
    return GraphSampleIdentityKey{stream, GraphSampleTimingKeyKind::InputSeq, sample.input_seq};
  }
  if (sample.frame_id >= 0) {
    return GraphSampleIdentityKey{stream, GraphSampleTimingKeyKind::FrameId, sample.frame_id};
  }
  return std::nullopt;
}
```

The hash combines `endpoint`, `stream_id`, `kind`, and `value`. `GraphSampleTimingState` should
store the original endpoint and timestamp only; the key already stores identity. This prevents a
bug where two samples with the same frame ID but different streams collide.

Stamp graph entry in `push_graph_samples_to_endpoint()` before `core.graph_push()`. Change the
helper signature so named-call sites pass the public endpoint name instead of losing it:

```cpp
bool push_graph_samples_to_endpoint(runtime::RunCore& core,
                                    const runtime::Endpoint& endpoint,
                                    std::string_view endpoint_name,
                                    const Sample& msgs,
                                    bool block);
```

Call-site updates:

- `RunCore::push_named_samples(input_name, ...)`: pass `input_name`.
- `Run::push(name, ...)` route-processor branches: pass `input_name`.
- default-input `RunCore::push_samples(...)`: pass `default_input_name(core)`:

```cpp
std::string default_input_name(const runtime::RunCore& core) {
  if (!core.graph_execution_ || !core.graph_execution_->plan.default_input) return {};
  const auto& def = *core.graph_execution_->plan.default_input;
  for (const auto& [name, ep] : core.graph_execution_->plan.named_inputs) {
    if (ep.node == def.node && ep.port == def.port && ep.kind == def.kind) return name;
  }
  return "default";
}
```

Stamp:

```cpp
const auto entry = std::chrono::steady_clock::now();
const bool ok = core.graph_push(...);
if (!ok) return false;
core.record_graph_sample_entry(std::string(endpoint_name), msg, entry);
```

Add `RunCore::record_graph_sample_entry()` as an internal method that:

- builds `GraphSampleIdentityKey` using the rules above;
- inserts the timestamp;
- evicts oldest entries beyond `graph_sample_timing_capacity`;
- increments a diagnostic counter for `graph_sample_timing_unkeyed` when the sample has no usable identity.

Concrete entry implementation:

```cpp
void RunCore::record_graph_sample_entry(std::string_view endpoint,
                                        const Sample& sample,
                                        std::chrono::steady_clock::time_point at) {
  auto key = make_graph_sample_identity_key(sample);
  if (!key) {
    graph_sample_timing_unkeyed.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  {
    std::lock_guard<std::mutex> lock(graph_sample_timing_mu);
    const std::uint64_t gen = ++graph_sample_timing_generation;
    auto [it, inserted] = graph_sample_timing_by_key.emplace(
        *key, GraphSampleTimingState{.graph_entry_at = at,
                                     .input_endpoint = std::string(endpoint),
                                     .generation = gen});
    if (!inserted) {
      it->second.ambiguous = true;
    } else {
      graph_sample_timing_order.push_back(GraphSampleTimingOrderEntry{*key, gen});
    }
    while (graph_sample_timing_order.size() > graph_sample_timing_capacity) {
      const auto old = graph_sample_timing_order.front();
      graph_sample_timing_order.pop_front();
      auto old_it = graph_sample_timing_by_key.find(old.key);
      if (old_it != graph_sample_timing_by_key.end() &&
          old_it->second.generation == old.generation) {
        graph_sample_timing_by_key.erase(old_it);
      }
    }
  }
  std::lock_guard<std::mutex> timing_lock(latency_mu);
  if (measurement_active) {
    measurement_graph_entries.push_back(GraphSampleTimingEvent{
        .endpoint = std::string(endpoint),
        .stream_id = key->stream_id,
        .key_kind = key->kind,
        .key_value = key->value,
        .timestamp_s = std::chrono::duration<double>(at - measurement_started_at).count()});
  }
}
```

Record graph output in both graph pull paths in `RunPull.cpp`:

```cpp
if (sample.has_value()) {
  const auto pull_at = std::chrono::steady_clock::now();
  core.record_graph_sample_output(endpoint_name, *sample, pull_at);
}
```

Endpoint-name sources:

- `RunCore::pull_named_output(output_name, ...)`: use `output_name`.
- `RunCore::pull(...)` default graph output: use `default_output_name(core)`, implemented like
  `default_input_name()` by matching `plan.default_output` against `plan.named_outputs`.

`record_graph_sample_output()` should:

- lookup by `(stream_id, orig_input_seq/input_seq/frame_id)`; endpoint is not part of the key;
- if the pending state is marked ambiguous, do not guess; count a miss/ambiguous lineage;
- compute public push-to-public pull latency;
- append to `measurement_latencies_ms` only while `measurement_active` is true;
- erase the matched pending entry;
- increment miss counters when no pending entry matches.

Avoid lock-order regressions: do the map lookup under `graph_sample_timing_mu`, release it, then
append latency under `latency_mu`. Existing pull code already uses `latency_mu` for pull timing.

```cpp
void RunCore::record_graph_sample_output(std::string_view output_endpoint,
                                         const Sample& sample,
                                         std::chrono::steady_clock::time_point pull_at) {
  std::optional<GraphSampleTimingState> state;
  auto key = make_graph_sample_identity_key(sample);
  if (!key) {
    graph_sample_timing_misses.fetch_add(1, std::memory_order_relaxed);
    return;
  }

  {
    std::lock_guard<std::mutex> lock(graph_sample_timing_mu);
    auto it = graph_sample_timing_by_key.find(*key);
    if (it != graph_sample_timing_by_key.end()) {
      if (it->second.ambiguous) {
        graph_sample_timing_misses.fetch_add(1, std::memory_order_relaxed);
        return;
      }
      state = it->second;
      graph_sample_timing_by_key.erase(it);
    }
  }

  if (!state) {
    graph_sample_timing_misses.fetch_add(1, std::memory_order_relaxed);
    return;
  }

  const double ms =
      std::chrono::duration<double, std::milli>(pull_at - state->graph_entry_at).count();
  std::lock_guard<std::mutex> lock(latency_mu);
  // update lifetime latency_count/mean/min/max here, using the same online update used by
  // the existing linear Run path; push into measurement_latencies_ms only when active.
  if (measurement_active) {
    measurement_latencies_ms.push_back(ms);
    if (auto key = make_graph_sample_identity_key(sample)) {
      measurement_graph_pulls.push_back(GraphSampleTimingEvent{
          .endpoint = std::string(output_endpoint),
          .stream_id = key->stream_id,
          .key_kind = key->kind,
          .key_value = key->value,
          .timestamp_s = std::chrono::duration<double>(pull_at - measurement_started_at).count()});
    }
  }
}
```

When a graph has multiple inputs, do **not** retry with arbitrary input endpoints. Use only exact
lineage carried by `orig_input_seq`/`stream_id`. If that is not enough, report the miss.

`RunMeasure.cpp` must bracket these vectors with the same lock as existing latency samples:

```cpp
// Run::start_measurement(), under core_->latency_mu
core_->measurement_latencies_ms.clear();
core_->measurement_frame_gaps_ms.clear();
core_->measurement_graph_entries.clear();
core_->measurement_graph_pulls.clear();
core_->measurement_started_at = impl->start;
core_->measurement_output_timing_init = false;
core_->measurement_active = true;

// MeasureScope::stop(), under st->latency_mu
st->measurement_active = false;
latency_samples = st->measurement_latencies_ms;
graph_entry_events = st->measurement_graph_entries;
graph_pull_events = st->measurement_graph_pulls;
st->measurement_latencies_ms.clear();
st->measurement_graph_entries.clear();
st->measurement_graph_pulls.clear();
```

`MeasureScope::stop()` then passes `graph_entry_events` and `graph_pull_events` to
`build_path_timing()` after parsing LTTng. If no exact LTTng message events were collected, these
vectors still support graph E2E latency but not per-node arrival, so `path_timing.status` remains
`diagnostic_aggregate_only`.

Concrete stop flow:

```cpp
if (parsed.parsed && impl_->run->core_ && impl_->run->core_->graph_execution_ &&
    impl_->options.include_message_latency) {
  report.path_timing =
      runtime::build_path_timing(impl_->run->core_->graph_execution_->plan,
                                 parsed,
                                 graph_entry_events,
                                 graph_pull_events);
} else {
  report.path_timing.available = false;
  report.path_timing.status = report.edge_latency.empty()
      ? "unavailable_message_trace_disabled"
      : "diagnostic_aggregate_only";
  report.path_timing.source = report.edge_latency.empty() ? "none" : "diagnostics";
  report.path_timing.warnings.push_back(
      "Exact path timing requires MeasureOptions.include_message_latency=true.");
}
```

For multi-input/multi-output:

- If the output sample has `orig_input_seq` and `stream_id`, use it.
- If a Join/Combine produces a bundle, use the canonical field identity already restored by
  `InputStreamPull.cpp`/`OutputTensorOverride.h`.
- If identity is ambiguous, do not guess. Emit
  `unavailable_multi_input_lineage_ambiguous` or `unavailable_multi_output_lineage_ambiguous`.

#### Path/timeline and between-plugin timing implementation

This is the concrete implementation for “time since graph entry to QuantTess” and “transport time
between plugins.”

Files:

```text
include/pipeline/Run.h
src/pipeline/runtime/RunMeasure.cpp
src/pipeline/runtime/LttngTraceParser.cpp
src/pipeline/runtime/PathTimingBuilder.h       (new private helper)
src/pipeline/runtime/PathTimingBuilder.cpp     (new implementation)
src/pipeline/runtime/GraphRunExport.cpp
src/pipeline/runtime/EdgeMetrics.cpp
src/pipeline/runtime/RunCore.h
src/pipeline/runtime/RunPush.cpp
src/pipeline/runtime/RunPull.cpp
```

Extend `MeasureReport`:

```cpp
struct MeasurePathStat {
  std::uint64_t samples = 0;
  double avg_ms = 0.0;
  double p50_ms = 0.0;
  double p95_ms = 0.0;
  double max_ms = 0.0;
  bool reliable = true;
};

struct MeasurePathNodeArrival {
  std::string customer_node_id; // "c3" when customer view mapping is known.
  std::string lowered_node_id;  // "n7".
  std::int32_t runtime_node_id = -1;
  std::string plugin_instance_id;
  std::string stream_id;
  std::string semantics = "graph_entry_to_first_node_observation";
  MeasurePathStat latency;
};

struct MeasurePathInterPluginGap {
  std::string customer_edge_id; // "ce4" when customer edge mapping is known.
  std::string lowered_edge_id;  // "e8".
  std::string from_customer_node_id;
  std::string to_customer_node_id;
  std::int32_t from_runtime_node_id = -1;
  std::int32_t to_runtime_node_id = -1;
  std::string from_plugin_instance_id;
  std::string to_plugin_instance_id;
  std::string stream_id;
  std::string semantics = "upstream_plugin_end_to_downstream_plugin_start";
  MeasurePathStat latency;
};

struct MeasurePathOutputTail {
  std::string output_endpoint;
  std::string customer_output_node_id;
  std::string lowered_edge_id;
  std::string stream_id;
  std::string semantics = "last_observed_work_to_public_pull";
  MeasurePathStat latency;
};

struct MeasurePathTiming {
  bool available = false;
  std::string status;
  std::string source;
  std::vector<std::string> warnings;
  std::vector<MeasurePathNodeArrival> node_arrival;
  std::vector<MeasurePathInterPluginGap> inter_plugin_gap;
  std::vector<MeasurePathOutputTail> output_tail;
};

MeasurePathTiming path_timing;
bool trace_loss_detected = false;
std::uint64_t graph_sample_timing_unkeyed = 0;
std::uint64_t graph_sample_timing_misses = 0;
```

Do not try to reconstruct inter-plugin gaps from aggregated `MeasurePluginLatency`; aggregation
throws away start/end timestamps. In `LttngTraceParser.cpp`, keep raw paired spans long enough to
compute path rows:

```cpp
struct ParsedPluginSpan {
  double start_s = 0.0;
  double end_s = 0.0;
  MeasurePluginLatency metric_identity; // one-call identity fields populated
  std::uint64_t run_id_hash = 0;
  std::uint64_t graph_id_hash = 0;
  std::string message_id;
  std::string stream_id;
  std::int64_t frame_id = -1;
  std::int64_t input_seq = -1;
  std::int64_t orig_input_seq = -1;
};

struct ParsedEdgeSpan {
  double start_s = 0.0;
  double end_s = 0.0;
  MeasureEdgeLatency metric_identity; // one-call identity fields populated
  TraceGraphMessageEventType start_type = TraceGraphMessageEventType::EdgeSrcPush;
  TraceGraphMessageEventType end_type = TraceGraphMessageEventType::EdgeSinkRecv;
  std::uint64_t run_id_hash = 0;
  std::uint64_t graph_id_hash = 0;
  std::string message_id;
  std::string stream_id;
  std::int64_t frame_id = -1;
  std::int64_t input_seq = -1;
  std::int64_t orig_input_seq = -1;
};
```

Add internal vectors to `LttngParseResult` or a private parser result object:

```cpp
std::vector<ParsedPluginSpan> raw_plugin_spans;
std::vector<ParsedEdgeSpan> raw_edge_spans;
```

Then:

1. Pair raw events into raw spans.
2. Build path/timeline rows from raw spans.
3. Aggregate raw spans into `MeasurePluginLatency` / `MeasureEdgeLatency` for the existing tables.

Do the row building in a separate private helper, not inside `GraphRunExport.cpp`:

```cpp
// src/pipeline/runtime/PathTimingBuilder.h
#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/Run.h"
#include "pipeline/runtime/ExecutionGraphPlan.h"
#include "LttngMetricsCollector.h"
#include <vector>

namespace simaai::neat::runtime {

MeasurePathTiming build_path_timing(const ExecutionGraphPlan& plan,
                                    const pipeline_internal::LttngParseResult& parsed,
                                    const std::vector<GraphSampleTimingEvent>& graph_entries,
                                    const std::vector<GraphSampleTimingEvent>& graph_pulls);

} // namespace simaai::neat::runtime
```

`GraphSampleTimingEvent` is the internal runtime struct defined in the `RunCore.h` timing state
section above and captured when `record_graph_sample_entry()` and `record_graph_sample_output()`
run during an active measurement.

The builder's sample key:

```cpp
struct TraceSampleKey {
  std::uint64_t run_id_hash = 0;
  std::uint64_t graph_id_hash = 0;
  std::string stream_id;
  std::string message_id; // preferred when present
  GraphSampleTimingKeyKind fallback_kind = GraphSampleTimingKeyKind::OrigInputSeq;
  std::int64_t fallback_value = -1;
};
```

Rules:

- Prefer exact `message_id` for LTTng edge/plugin pairing.
- Else use `(run_id_hash, graph_id_hash, stream_id, orig_input_seq)`.
- Else use `(run_id_hash, graph_id_hash, stream_id, input_seq)`.
- Else use `(run_id_hash, graph_id_hash, stream_id, frame_id)`.
- If two rows match the same fallback key and same edge/node, mark the affected path row
  `reliable=false` and add `AMBIGUOUS_SAMPLE_LINEAGE` to `path_timing.warnings`.

Use two sources:

1. `source="lttng_message_events"` when `MeasureOptions.include_message_latency=true`.
   - Exact per-message timing.
   - Parser already recognizes `sima_neat_edge:message`; finish provider/emitter integration if
     events are not emitted by the target plugins/runtime path.
   - Pair using `run_id_hash + graph_id_hash + edge_id + stream_id + message_id + span_kind`.
   - Fallback key only if `message_id` is missing:
     `run_id_hash + stream_id + orig_input_seq + edge_id`.
   - Treat `message_id == 0` as missing; adopted plugin tracepoints use zero for missing identity.
2. `source="diagnostics"` when only low-overhead counters exist.
   - Can populate `edge_metrics` average queue/transport labels.
   - Must set `path_timing.status="diagnostic_aggregate_only"` because diagnostics cannot compute
     graph-entry-to-node arrival for the same sample.

Fix the edge pairing key before adding more event sites. Current parser-style FIFO pairing by
`edge_id + message_id` is not safe once each edge emits both transport and queue-residence spans:

```cpp
std::string edge_span_kind(const RawEvent& ev) {
  const int type = parse_i32(ev.fields, "event_type", -1);
  if (type == 0 || type == 1) return "edge_transport";  // EdgeSrcPush -> EdgeSinkRecv
  if (type == 2 || type == 3) return "queue_residence"; // QueueIn -> QueueOut
  return "unknown";
}

std::string edge_pair_key(const RawEvent& ev) {
  return join_key({field_string(ev.fields, "run_id_hash"),
                   field_string(ev.fields, "graph_id_hash"),
                   field_string(ev.fields, "edge_id"),
                   field_string(ev.fields, "stream_id"),
                   field_string(ev.fields, "message_id"),
                   edge_span_kind(ev)});
}
```

Use `std::deque` for pending plugin/edge starts and `pop_front()`; do not use
`vector.erase(begin())` in high-volume message traces. Extend parser APIs to filter by
`expected_graph_id_hash` as well as run hash.

Without this, an event sequence `EdgeSrcPush, QueueIn, QueueOut, EdgeSinkRecv` can incorrectly pair
`QueueOut` with `EdgeSrcPush`, producing bogus EV/transport times.

Exact message event emitter details:

```text
src/pipeline/runtime/ExecutionGraphRuntime.h
src/pipeline/runtime/PipelineSegmentRuntime.h
src/pipeline/runtime/RunCoreGraphStart.cpp
src/pipeline/runtime/RunCore.h
src/pipeline/runtime/EdgeRouter.h
src/pipeline/runtime/EdgeRouter.cpp
src/pipeline/runtime/TraceMessageEvents.h
src/pipeline/runtime/TraceMessageEvents.cpp
```

Add a tiny trace helper layer. All call sites must be cheap when message tracing is disabled:

```cpp
// src/pipeline/runtime/TraceMessageEvents.h
#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/GraphOptions.h"
#include "graph/GraphTypes.h"
#include <cstddef>
#include <cstdint>
#include <string>

namespace simaai::neat::runtime {

struct ExecutionGraphRuntime;

constexpr std::size_t invalid_edge_index() {
  return static_cast<std::size_t>(-1);
}

enum class TraceGraphMessageEventType : std::uint8_t {
  EdgeSrcPush = 0,
  EdgeSinkRecv = 1,
  QueueIn = 2,
  QueueOut = 3,
  Drop = 4,
  GraphEntry = 5,
  GraphOutputPull = 6,
};

struct TraceGraphMessageArgs {
  std::uint64_t run_id_hash = 0;
  std::uint64_t graph_id_hash = 0;
  std::size_t edge_index = invalid_edge_index();
  std::int32_t src_runtime_node_id = -1;
  std::int32_t dst_runtime_node_id = -1;
  std::string stream_id;
  std::int64_t frame_id = -1;
  std::int64_t input_seq = -1;
  std::int64_t orig_input_seq = -1;
  std::int64_t pts_ns = -1;
  std::uint64_t message_id = 0;
};

bool graph_message_trace_enabled(const ExecutionGraphRuntime* execution) noexcept;

TraceGraphMessageArgs make_trace_graph_message_args(const ExecutionGraphRuntime* execution,
                                                    std::size_t edge_index,
                                                    const Sample& sample) noexcept;

void trace_graph_message_event(TraceGraphMessageEventType type,
                               const TraceGraphMessageArgs& args) noexcept;

void trace_graph_message_event(TraceGraphMessageEventType type,
                               const ExecutionGraphRuntime* execution,
                               std::size_t edge_index,
                               const Sample& sample) noexcept;

} // namespace simaai::neat::runtime
```

`TraceMessageEvents.cpp` fast path:

```cpp
void trace_graph_message_event(TraceGraphMessageEventType type,
                               const ExecutionGraphRuntime* execution,
                               std::size_t edge_index,
                               const Sample& sample) noexcept {
  if (!graph_message_trace_enabled(execution)) return;
  if (edge_index == invalid_edge_index()) return;
  trace_graph_message_event(type, make_trace_graph_message_args(execution, edge_index, sample));
}

void trace_graph_message_event(TraceGraphMessageEventType type,
                               const TraceGraphMessageArgs& in) noexcept {
  sima_neat_edge_message_args args{};
  args.event_type = static_cast<std::uint32_t>(type);
  args.run_id_hash = in.run_id_hash;
  args.graph_id_hash = in.graph_id_hash;
  args.message_id = in.message_id;
  args.edge_id = static_cast<std::int32_t>(in.edge_index); // parser/export canonicalizes to "eN"
  args.src_runtime_node_id = in.src_runtime_node_id;
  args.dst_runtime_node_id = in.dst_runtime_node_id;
  args.stream_id = in.stream_id.c_str();
  args.frame_id = in.frame_id;
  args.input_seq = in.input_seq;
  args.orig_input_seq = in.orig_input_seq;
  args.pts_ns = in.pts_ns;
  tracepoint_sima_neat_edge_message(&args);
}
```

Provider style:

- Use the existing V2 tracepoint package style from `lttng_plugin_latency_implementation_plan.md`:
  one `sima_neat_edge:message` event with a struct-pointer payload.
- Do not define tracepoint providers inside dlopen-able plugins; add/verify the provider in
  `libsimaaitrace.so` and compile the core adapter as a no-op when the provider header is absent.
- Keep `edge_id` numeric in the tracepoint (`int32_t`) to avoid hot-path string allocation; convert
  to canonical lowered JSON ID `"e<edge_id>"` in the parser/export.
- `make_numeric_message_id(sample)` must match adopted plugin tracepoints: `orig_input_seq`, else
  `input_seq`, else `frame_id`, else `0` for missing identity. Pair with `stream_id`; do not hash.

`make_trace_graph_message_args()` fills runtime endpoints from the typed plan so the parser does not
infer them from element names:

```cpp
TraceGraphMessageArgs make_trace_graph_message_args(const ExecutionGraphRuntime* execution,
                                                    std::size_t edge_index,
                                                    const Sample& sample) noexcept {
  TraceGraphMessageArgs out;
  out.run_id_hash = execution->trace_run_id_hash.load(std::memory_order_acquire);
  out.graph_id_hash = execution->trace_graph_id_hash.load(std::memory_order_acquire);
  out.edge_index = edge_index;
  if (edge_index < execution->plan.edges.size()) {
    out.src_runtime_node_id = static_cast<std::int32_t>(execution->plan.edges[edge_index].from);
    out.dst_runtime_node_id = static_cast<std::int32_t>(execution->plan.edges[edge_index].to);
  }
  out.stream_id = sample.stream_id;
  out.frame_id = sample.frame_id;
  out.input_seq = sample.input_seq;
  out.orig_input_seq = sample.orig_input_seq;
  out.pts_ns = sample.pts_ns;
  out.message_id = make_numeric_message_id(sample);
  return out;
}
```

No-op behavior:

- If the LTTng provider headers are not available in this build configuration, compile
  `TraceMessageEvents.cpp` as a no-op with the same ABI.
- If providers are available but `include_message_latency=false`, the only cost should be one
  relaxed atomic load and one branch per event site.

Add edge identity to runtime routing:

```cpp
struct DownstreamTarget {
  enum class Kind { StageGroup, PipelineInput, GraphSink };
  Kind kind = Kind::StageGroup;
  std::size_t index = 0;
  graph::PortId port = graph::kInvalidPort;
  std::size_t edge_index = invalid_edge_index();
};
```

In `RunCoreGraphStart.cpp`, where adjacency is populated from `plan.edges[edge_index]`, set
`target.edge_index = edge_index` for StageGroup, PipelineInput, and GraphSink targets.

Do **not** extend public `graph::StageMsg` just to carry metrics metadata. Fact from the current
code: `StageMsg` lives in `include/graph/StageExecutor.h` and is the object handed to
`StageExecutor::on_input()`, so adding metrics-only fields there leaks runtime internals into the
public stage API. Keep this surgical with internal queue wrappers.

```cpp
struct RuntimeStageQueueMsg {
  PortId in_port = kInvalidPort;
  Sample sample;
  std::size_t edge_index = invalid_edge_index(); // internal metrics routing identity
};

struct RuntimePipelineQueueMsg {
  Sample sample;
  std::size_t edge_index = invalid_edge_index(); // internal metrics routing identity
};

struct RuntimeSinkQueueMsg {
  Sample sample;
  std::size_t edge_index = invalid_edge_index(); // internal metrics routing identity
};

using GraphSinkQueue = graph::runtime::BlockingQueue<RuntimeSinkQueueMsg>;
```

Exact internal type placement:

- Put `RuntimeStageQueueMsg` in `ExecutionGraphRuntime.h`, near `StageRuntime`, because only the
  graph stage workers should see it.
- Put `RuntimePipelineQueueMsg` in `PipelineSegmentRuntime.h`, before
  `PipelineSegmentRuntime::GraphTransport`, because `GraphTransport::input_queue` owns the type.
- Change graph sinks to internal `GraphSinkQueue` as well. Do **not** recover edge identity with a
  `sink_node -> edge_index` map; one sink can have multiple incoming edges.

Add trace toggles to runtime state:

```cpp
struct ExecutionGraphRuntime {
  // Existing plan/runtime fields remain unchanged.
  std::atomic<bool> message_trace_enabled{false};
  std::atomic<std::uint64_t> trace_run_id_hash{0};
  std::atomic<std::uint64_t> trace_graph_id_hash{0};
  std::unordered_map<graph::NodeId, std::shared_ptr<GraphSinkQueue>> sinks;
};
```

`trace_graph_message_event()` reads these fields; it must not reach through `RunCore`.

`RunMeasure.cpp` toggling:

```cpp
const bool effective_message_lttng =
    opt.include_message_latency &&
    message_source != MetricsTraceSource::Off &&
    trace_source_requests_lttng(message_source);
// pass effective_message_lttng into make_lttng_options so sima_neat_edge:* is not enabled
// when the effective message source is Off.

// After LTTng collector starts successfully and only when include_message_latency is active.
if (core_ && core_->graph_execution_ && effective_message_lttng) {
  auto& ge = *core_->graph_execution_;
  ge.trace_run_id_hash.store(impl->trace_context.run_id_hash, std::memory_order_relaxed);
  ge.trace_graph_id_hash.store(impl->trace_context.graph_id_hash, std::memory_order_relaxed);
  ge.message_trace_enabled.store(true, std::memory_order_release);
}

// In MeasureScope::stop(), before stop_and_destroy()/parse and also in destructor cleanup paths.
if (impl_->run && impl_->run->core_ && impl_->run->core_->graph_execution_) {
  impl_->run->core_->graph_execution_->message_trace_enabled.store(false,
                                                                   std::memory_order_release);
}
```

If LTTng start fails in Auto mode, leave `message_trace_enabled=false` and set
`report.path_timing.status="diagnostic_aggregate_only"`; never emit partial exact timeline rows
from a failed trace session.

After parse, do not leave `message_latency_status="collected"` unless message spans were parsed:

```cpp
if (impl_->options.include_message_latency &&
    parsed.raw_edge_spans.empty() &&
    parsed.edge_metrics.empty() &&
    parsed.edge_metrics_unattributed.empty()) {
  report.message_latency_status = "unavailable";
  report.warnings.push_back(
      "LTTng message latency requested but no sima_neat_edge:message spans were collected");
}
```

Concurrency rule from current code:

- `LttngTraceSession.cpp` has a process-global active-collector guard. Keep it.
- A single multi-stream `Run` should use one collector and distinguish stream/plugin instances with
  `stream_id`, `plugin_instance_id`, `runtime_node_id`, and `edge_id`.
- Two simultaneous `Run::start_measurement()` calls that both require LTTng should not silently mix
  traces. In Auto mode, the second run falls back to diagnostics with a warning:
  `LTTNG_COLLECTOR_BUSY: exact plugin/message attribution disabled for this measurement`.
  In forced `MetricsTraceSource::Lttng`, throw the existing start failure.

Concrete code changes:

- In `src/pipeline/runtime/ExecutionGraphRuntime.h`, change `StageRuntime` to own an internal
  `BlockingQueue<RuntimeStageQueueMsg> inbox` instead of `graph::runtime::StageMailbox mailbox`.
  Remove the `graph/runtime/StageMailbox.h` include if nothing else uses it in that file.
- Replace all current references:
  - `stage->mailbox.inbox.close()` -> `stage->inbox.close()`
  - `stage.mailbox.inbox.push(...)` -> `stage.inbox.push(...)`
  - `stage->mailbox.inbox.stats()` -> `stage->inbox.stats()`
  - `st.mailbox.inbox.pop(...)` -> `st.inbox.pop(...)`
  Current files include `RunCore.cpp`, `EdgeMetrics.cpp`, `RunCoreGraphStart.cpp`, and
  `ExecutionGraphRuntime.h`.
- In `StageRuntime` constructor:

```cpp
explicit StageRuntime(std::size_t capacity = 0) : inbox(capacity) {}
```

- In `RunCoreGraphStart.cpp::start_stage_workers()`, pop `RuntimeStageQueueMsg`, emit queue-out
  trace using `queued.edge_index`, then construct the public
  `graph::StageMsg{queued.in_port, std::move(queued.sample)}` for the executor.
- In `PipelineSegmentRuntime::GraphTransport`, change `input_queue` from
  `std::shared_ptr<BlockingQueue<Sample>>` to
  `std::shared_ptr<BlockingQueue<RuntimePipelineQueueMsg>>`.
- In `RunCoreGraphStart.cpp::start_pipeline_push_thread()`, pop the wrapper, emit queue-out trace
  using `edge_index`, then move `wrapper.sample` into the existing sanitize/push path.
- In direct public `RunCore::graph_push()` paths where no lowered edge exists yet, set
  `edge_index = invalid` and rely on graph-entry timing, not edge timing.

Change the dispatch signatures only in private runtime files:

```cpp
// EdgeRouter.h
std::function<bool(std::size_t, graph::PortId, Sample&&, std::size_t edge_index)>
    dispatch_to_stage_group;

// RunCore.h
bool graph_dispatch_to_stage_group(std::size_t group_index,
                                   graph::PortId port,
                                   Sample&& sample,
                                   std::size_t edge_index,
                                   const EdgeRouterOptions& options);
```

`EdgeRouter::dispatch_to_target()` changes:

```cpp
if (target.kind == DownstreamTarget::Kind::StageGroup) {
  const bool trace =
      graph_message_trace_enabled(runtime_) && target.edge_index != invalid_edge_index();
  TraceGraphMessageArgs trace_args;
  if (trace) {
    trace_args = make_trace_graph_message_args(runtime_, target.edge_index, sample);
    trace_graph_message_event(TraceGraphMessageEventType::EdgeSrcPush, trace_args);
  }
  const bool ok = callbacks.dispatch_to_stage_group(target.index, target.port,
                                                    std::move(sample), target.edge_index);
  if (trace) {
    trace_graph_message_event(ok ? TraceGraphMessageEventType::QueueIn
                                 : TraceGraphMessageEventType::Drop,
                              trace_args);
  }
  return ok;
}
// PipelineInput branch:
RuntimePipelineQueueMsg queued{.sample = std::move(sample), .edge_index = target.edge_index};
const bool trace =
    graph_message_trace_enabled(runtime_) && target.edge_index != invalid_edge_index();
TraceGraphMessageArgs trace_args;
if (trace) {
  trace_args = make_trace_graph_message_args(runtime_, target.edge_index, queued.sample);
  trace_graph_message_event(TraceGraphMessageEventType::EdgeSrcPush, trace_args);
}
const bool ok = input_queue->push(std::move(queued), options.push_timeout_ms);
if (trace) {
  trace_graph_message_event(ok ? TraceGraphMessageEventType::QueueIn
                               : TraceGraphMessageEventType::Drop,
                            trace_args);
}
return ok;
```

Sink route changes:

```cpp
// EdgeRouter::push_to_sink():
RuntimeSinkQueueMsg queued{.sample = std::move(sample), .edge_index = target.edge_index};
const bool trace =
    graph_message_trace_enabled(runtime_) && target.edge_index != invalid_edge_index();
TraceGraphMessageArgs trace_args;
if (trace) {
  trace_args = make_trace_graph_message_args(runtime_, target.edge_index, queued.sample);
}
if (trace) {
  trace_graph_message_event(TraceGraphMessageEventType::EdgeSrcPush, trace_args);
}
const bool ok = sink_queue->push(std::move(queued), options.push_timeout_ms);
if (trace) {
  trace_graph_message_event(ok ? TraceGraphMessageEventType::QueueIn
                               : TraceGraphMessageEventType::Drop,
                            trace_args);
}
return ok;
```

`RunCoreGraphStart.cpp` call-site changes:

```cpp
// make_edge_router_callbacks()
callbacks.dispatch_to_stage_group = [core](std::size_t group_index,
                                           PortId port,
                                           Sample&& sample,
                                           std::size_t edge_index) {
  return core->graph_dispatch_to_stage_group(group_index, port, std::move(sample),
                                             edge_index,
                                             core->graph_options.router_options());
};

// allocate pipeline input queue
runtime->transport.input_queue =
    std::make_shared<graph::runtime::BlockingQueue<RuntimePipelineQueueMsg>>(
        core->graph_options.edge_queue);

// build_adjacency_and_sinks(), inside for (std::size_t eidx = 0; ...)
outs.push_back(DownstreamTarget{DownstreamTarget::Kind::StageGroup,
                                it_stage->second,
                                e.to_port,
                                eidx});
outs.push_back(DownstreamTarget{DownstreamTarget::Kind::PipelineInput,
                                it_pipe->second,
                                e.to_port,
                                eidx});
outs.push_back(DownstreamTarget{DownstreamTarget::Kind::GraphSink,
                                static_cast<std::size_t>(e.to),
                                e.to_port,
                                eidx});
```

Stage worker pop path:

```cpp
RuntimeStageQueueMsg queued;
if (!st.inbox.pop(queued, core->graph_options.pull_timeout_ms)) {
  st.telemetry.mailbox_pop_miss.fetch_add(1, std::memory_order_relaxed);
  continue;
}
trace_graph_message_event(TraceGraphMessageEventType::QueueOut, &execution,
                          queued.edge_index, queued.sample);
trace_graph_message_event(TraceGraphMessageEventType::EdgeSinkRecv, &execution,
                          queued.edge_index, queued.sample);
StageMsg msg{queued.in_port, std::move(queued.sample)};
```

Pipeline push-thread pop path:

```cpp
RuntimePipelineQueueMsg queued;
if (!pipe.transport.input_queue->pop(queued, core->graph_options.pull_timeout_ms)) {
  pipe.transport.telemetry.push_thread_pop_miss.fetch_add(1, std::memory_order_relaxed);
  continue;
}
trace_graph_message_event(TraceGraphMessageEventType::QueueOut, &execution,
                          queued.edge_index, queued.sample);
trace_graph_message_event(TraceGraphMessageEventType::EdgeSinkRecv, &execution,
                          queued.edge_index, queued.sample);
Sample sample = std::move(queued.sample);
```

Graph pull path:

```cpp
RuntimeSinkQueueMsg queued;
if (sink_queue->pop(queued, timeout_ms)) {
  trace_graph_message_event(TraceGraphMessageEventType::QueueOut, graph_execution_.get(),
                            queued.edge_index, queued.sample);
  trace_graph_message_event(TraceGraphMessageEventType::EdgeSinkRecv, graph_execution_.get(),
                            queued.edge_index, queued.sample);
  Sample sample = std::move(queued.sample);
}
```

This is more surgical than changing `StageMsg`: it changes only graph runtime internals and leaves
customer StageExecutor source compatibility intact.

Emitter points:

```cpp
// EdgeRouter.cpp, before and after queue/mailbox/sink push:
const auto trace_args = make_trace_graph_message_args(runtime_, target.edge_index, sample);
trace_graph_message_event(TraceGraphMessageEventType::EdgeSrcPush, trace_args);
// After successful enqueue:
trace_graph_message_event(TraceGraphMessageEventType::QueueIn, trace_args);
// On failed enqueue:
trace_graph_message_event(TraceGraphMessageEventType::Drop, trace_args);

// RunCoreGraphStart.cpp stage worker, after internal inbox pop succeeds:
trace_graph_message_event(TraceGraphMessageEventType::QueueOut,
                          &execution, queued.edge_index, queued.sample);
trace_graph_message_event(TraceGraphMessageEventType::EdgeSinkRecv,
                          &execution, queued.edge_index, queued.sample);

// RunCoreGraphStart.cpp pipeline push thread, after input_queue pop succeeds:
trace_graph_message_event(TraceGraphMessageEventType::QueueOut,
                          &execution, queued.edge_index, queued.sample);
trace_graph_message_event(TraceGraphMessageEventType::EdgeSinkRecv,
                          &execution, queued.edge_index, queued.sample);

// RunCore::graph_pull()/pull_named_output(), after sink queue pop succeeds:
trace_graph_message_event(TraceGraphMessageEventType::QueueOut,
                          &execution, incoming_edge_index, out);
trace_graph_message_event(TraceGraphMessageEventType::EdgeSinkRecv,
                          &execution, incoming_edge_index, out);
```

`TraceMessageEvents.cpp` should be compiled to a no-op when LTTng providers are unavailable and
should check the measurement options/trace-enabled flag before doing any expensive string work.
Use existing `Sample` identity fields (`stream_id`, `frame_id`, `input_seq`, `orig_input_seq`) and
emit numeric `edge_index`; parser/export converts it to canonical lowered edge ID `e<edge_index>`.

For inter-plugin gap from LTTng:

```cpp
inter_plugin_gap_ms =
    downstream_plugin_span.start_timestamp_s - upstream_plugin_span.end_timestamp_s;
```

Pairing requirements:

- Same `run_id_hash`.
- Same `graph_id_hash`.
- Same `stream_id`.
- Same `orig_input_seq` or same `message_id`.
- A customer/lowered edge connects upstream runtime node to downstream runtime node.
- If multiple upstream candidates match, choose the one connected by `edge_id`; otherwise mark
  `unavailable_lineage_ambiguous`.

For node arrival from graph entry:

```cpp
node_arrival_ms =
    first_observed_node_input_or_plugin_start_timestamp_s - graph_entry_timestamp_s;
```

`graph_entry_timestamp_s` must come from the LTTng `GraphEntry` event emitted by
`TraceMessageEvents`, not from `std::chrono::steady_clock`. Do not subtract C++ steady-clock
timestamps from LTTng timestamps.

The first observed node input timestamp preference order:

1. `sima_neat_edge:message` `EDGE_SINK_RECV` for that lowered edge.
2. Downstream plugin span START for that node.
3. Diagnostic edge aggregate is **not allowed** for this field.

For output tail:

```cpp
output_tail_ms = public_pull_timestamp - last_plugin_end_or_output_edge_receive_timestamp;
```

`public_pull_timestamp` must come from the LTTng `GraphOutputPull` event. The steady-clock
graph-entry/pull timestamps are only for `graph_e2e_latency_ms`.

If public pull timestamp is missing or sample identity does not match, omit that row and add a
warning; do not synthesize it.

Export:

```cpp
run_json["path_timing"] = path_timing_to_json(report.path_timing);
run_json["metrics_sources"]["path_timing"] = {
  {"status", report.path_timing.status},
  {"source", report.path_timing.source},
  {"reliable", !report.trace_loss_detected}
};
```

`path_timing_to_json()` should serialize empty/unavailable explicitly:

```cpp
json path_timing_to_json(const MeasurePathTiming& timing, bool trace_loss_detected) {
  json out;
  out["available"] = timing.available;
  out["status"] = timing.status.empty() ? "unavailable_message_trace_disabled" : timing.status;
  out["source"] = timing.source.empty() ? "none" : timing.source;
  out["aggregation"] = "measured_window";
  out["warnings"] = timing.warnings;
  out["reliable"] = !trace_loss_detected && all_path_rows_reliable(timing);
  out["node_arrival_ms"] = serialize_node_arrivals(timing.node_arrival);
  out["inter_plugin_gap_ms"] = serialize_inter_plugin_gaps(timing.inter_plugin_gap);
  out["output_tail_ms"] = serialize_output_tails(timing.output_tail);
  if (!timing.available && !out.contains("reason")) {
    out["reason"] = first_warning_or_status(timing);
  }
  return out;
}
```

`all_path_rows_reliable()` is a three-vector loop over `node_arrival`, `inter_plugin_gap`, and
`output_tail`, checking `row.latency.reliable`.

Tests:

- Synthetic LTTng text: two plugins on one edge should produce one inter-plugin gap row.
- Branch/fanout: same `orig_input_seq` on two edge IDs should produce two separate edge/timeline
  rows, not one merged row.
- Missing `message_id` and missing `orig_input_seq` should make `path_timing.available=false`.
- Diagnostic-only run should show edge labels but timeline status `diagnostic_aggregate_only`.

#### Edge latency availability flags

Normalize edge identity before export. Today `MeasureEdgeLatency::edge_id` is a string, while LTTng
message tracepoints may carry a numeric edge index. The JSON/visualizer must use canonical lowered
edge IDs:

```cpp
std::string canonical_lowered_edge_id(std::string raw) {
  raw = trim(raw);
  auto all_digits = [](const std::string& s, std::size_t start = 0) {
    return start < s.size() &&
           std::all_of(s.begin() + static_cast<std::ptrdiff_t>(start), s.end(),
                       [](unsigned char c) { return std::isdigit(c) != 0; });
  };
  if (raw.empty()) return {};
  if (raw.size() > 1 && raw[0] == 'e' && all_digits(raw, 1)) {
    return raw;
  }
  if (all_digits(raw)) {
    return "e" + raw;
  }
  return raw; // diagnostic synthetic ids remain as-is
}
```

Use it in:

```text
src/pipeline/runtime/LttngTraceParser.cpp::edge_metric_from_span()
src/pipeline/runtime/EdgeMetrics.cpp::snapshot_graph_queue_latencies()
src/pipeline/runtime/GraphRunExport.cpp::edge_metric_to_json()
```

`edge_metric_to_json()` should emit both:

```json
"edge_id": "e1",
"lowered_edge_id": "e1"
```

When the ID is diagnostic-only (`n1_element->n2_element:sink`, `seg0:input_queue`), emit:

```json
"edge_id": "seg0:input_queue",
"lowered_edge_id": null,
"mapping_status": "diagnostic_unmapped"
```

In `edge_metric_to_json()` in `GraphRunExport.cpp`, add:

```cpp
latency["percentiles_available"] = e.source == "lttng";
latency["min_max_available"] = e.source == "lttng";
latency["max_semantics"] = e.source == "diagnostics" ?
    "run_lifetime_high_water" : "measured_window_sample_max";
```

If diagnostics rows do not have real percentiles, keep p50/p95 numeric fields for compatibility but visualizer must use the availability flag.

#### Power status split

Replace ambiguous `requested_no_samples` in `disabled_power_json()` call sites with one of:

```text
not_configured_on_run
enabled_no_samples
unavailable_all_rails_failed
disabled_by_options
```

Implementation detail:

- If `RunExportOptions.include_power == false` or `MeasureOptions.include_power == false`:
  `disabled_by_options`.
- If include_power true but no `RunOptions.power_monitor.enabled`: `not_configured_on_run`.
- If monitor enabled and `samples == 0`: `enabled_no_samples`.
- If monitor has rail errors and no valid rail samples: `unavailable_all_rails_failed`.
- If monitor has some samples but some rails failed: `collected_with_errors`.
- If valid samples: `collected`.

Measured export should pass:

```cpp
const bool power_export_requested = opt.include_power && report.options.include_power;
const bool monitor_configured = report.power.enabled;
```

### Phase 3: Rewrite visualizer around customer view

Primary file:

```text
tools/visualize_graph_run.py
```

#### CLI additions

Current CLI:

```bash
visualize_graph_run.py input -o output --view auto|public|lowered
```

New CLI:

```bash
python3 tools/visualize_graph_run.py run.json \
  --view auto|customer|public|lowered \
  -o run.html \
  --svg run.svg \
  --topology-json run.customer_topology.json \
  --summary \
  --summary-json run.summary.json \
  --validate \
  --strict \
  --redact customer \
  --fail-on-fallback \
  --fail-on-unattributed
```

Implementation details:

- `--view auto` order:
  1. `graph.customer_view`
  2. `graph.public_view`
  3. `graph.lowered_view`
  4. `graph.nodes`
  5. node metrics fallback
- `--svg PATH`: write graph-only standalone SVG using same CSS embedded in CDATA.
- `--topology-json PATH`: write normalized selected topology plus overlay metrics. This is useful for debugging layout/mapping.
- `--summary`: print `view=customer nodes=... edges=... topology_source=... warnings=...`.
- `--summary-json PATH`: same data in JSON.
- `--validate`: import `tests/perf/tools/graph_run_schema.py` if available and validate before render.
- `--strict`: enable stricter validator checks.
- `--redact customer`: deep-redact payload before selecting topology/overlay. Redaction applies to
  visible HTML labels, SVG, topology JSON, summary JSON, and raw JSON/details.
- `--fail-on-fallback`: non-zero exit when selected view source is `node_metrics_fallback`.
- `--fail-on-unattributed`: non-zero exit if plugin/edge unattributed rows exist.

Redaction order:

```python
render_payload = _redact_payload(payload, args.redact)
selected = _select_view(render_payload, args.view)
overlay = _metric_overlay(render_payload, selected)
```

#### New internal data model

Add small Python structures or dictionaries:

```python
SelectedTopology = {
  "view": "customer",
  "source": "customer_view_v1",
  "nodes": [...],
  "edges": [...],
  "warnings": [...],
  "mapping": {...}
}

MetricOverlay = {
  "node_by_id": {...},
  "edge_by_lowered_id": {...},
  "edge_by_customer_id": {...},
  "plugin_by_node": {...},
  "path_node_arrival_by_customer_node": {...},
  "path_gap_by_customer_edge": {...},
  "path_output_tail_by_customer_output": {...}
}
```

Functions to add/replace:

```python
def _select_view(payload, requested_view) -> SelectedTopology
def _normalize_customer_view(view) -> SelectedTopology
def _metric_overlay(payload, selected) -> MetricOverlay
def _path_timing_overlay(payload, selected) -> dict[str, Any]
def _customer_edge_metric_label(edge, overlay) -> str
def _customer_node_timeline_label(node, overlay) -> str
def _layout_dag(nodes, edges) -> tuple[positions, width, height]
def _render_svg(selected, overlay) -> str
def _write_svg(path, svg, css) -> None
def _redact_payload(payload, mode) -> dict
```

#### Edge metric overlay rules

Current `_edge_metric_lookup()` maps `(from_node, to_node)` only. Replace with:

1. Build `edge_metrics_by_lowered_edge_id: dict[str, list[metric]]` from
   `run.edge_metrics[*].lowered_edge_id` first. Fall back to `run.edge_metrics[*].edge_id` only
   when it matches `^e[0-9]+$`. Values are lists because multi-stream can have multiple rows per
   lowered edge.
2. Also map by `(from_node, to_node, timing_semantics)` for legacy artifacts.
3. For customer edge:

```python
lowered_ids = edge.get("lowered_edge_ids", [])
metrics = []
for eid in lowered_ids:
    metrics.extend(by_id.get(eid, []))
if not metrics and edge has from/to lowered nodes:
    metrics = lookup_by_from_to(...)
```

4. Label:

```python
if len(metrics) == 0:
    return "handoff N/A" if edge.get("mapping_status") != "mapped" else ""
if len(metrics) == 1:
    return f"{semantic_short(metrics[0])} avg {fmt(avg_ms)}"
else:
    max_avg = max(metric.avg_ms for metric in metrics)
    streams = sorted({m.get("stream_id") for m in metrics if m.get("stream_id")})
    return f"handoff max avg {fmt(max_avg)} / {len(metrics)} rows / {len(streams)} streams"
```

5. Never sum metrics.

#### DAG layout algorithm

Replace `_parallel_pair_layout()` and `_default_layout()`.

Pseudo-code:

```python
def _layout_dag(nodes, edges):
    node_order = {id: i}
    comps = weakly_connected_components(nodes, edges)
    y_offset = margin
    for comp in comps:
        dag_edges, back_edges = break_cycles(comp.edges)
        rank = {input_like_node: 0}
        for node in topo_order(dag_edges):
            for dst in out[node]:
                rank[dst] = max(rank.get(dst, 0), rank[node] + 1)
        order_by_rank = initial_order(rank, node_order)
        for _ in range(4):
            barycentric_sweep_down(order_by_rank, dag_edges)
            barycentric_sweep_up(order_by_rank, dag_edges)
        if comp contains router nodes:
            place router at average y of successors/predecessors where possible
        assign x/y
        y_offset += comp_height + component_gap
```

Stable ordering:

- Input endpoints first in `graph.named_inputs` order.
- Output endpoints in `graph.named_outputs` order.
- Otherwise original JSON order.

Coordinates:

```python
node_w = 220
node_h = 92
x_gap = 90
y_gap = 42
margin = 40
x = margin + rank * (node_w + x_gap)
y = component_y + lane * (node_h + y_gap)
```

Edges:

- normal edges: cubic Bezier left-to-right.
- same-rank edges: dogleg down/up.
- parallel edges: offset by `[-8, 0, +8]`.
- back edges: dashed red/amber style and label `cycle/back-edge`.

Acceptance for branch artifact:

- Default HTML shows 3 components/lanes with ranks input -> branch -> output.
- No raw invalid public nodes are visible in customer view.
- Lowered debug view still shows generated `FanOut` nodes.

#### Visual labels

Customer node max 4 lines:

1. `label` or `endpoint_name`
2. `kind` / role (`Input`, `Branch`, `MLA infer`, `Output`)
3. metric line:
   - `avg 8.084 ms / n=10`
   - `endpoint / not timed`
   - `no samples`
4. mapping hint:
   - `maps p0→n0`
   - router: `fanout n9`

Lowered node max 4 lines:

1. `n9 FanOut`
2. `fanout0 [generated]`
3. metric or queue summary
4. segment/stage info

Metric line generation:

```python
if node role in {"input", "output"} and no node metric:
    "endpoint / not timed"
elif node metric status == "collected":
    f"avg {avg_ms:.3f} ms / n={samples}"
elif top plugin exists:
    f"plugin {backend} avg {avg_ms:.3f} ms"
else:
    "no samples"
```

Timeline line generation:

```python
arrival = overlay.path_node_arrival_by_customer_node.get(node.id)
tail = overlay.path_output_tail_by_customer_output.get(node.id)
if arrival:
    f"entry→node {fmt(arrival.avg_ms)} / n={arrival.samples}"
elif tail:
    f"tail {fmt(tail.avg_ms)} / n={tail.samples}"
elif payload.run.path_timing.available is False:
    ""  # show one global warning instead of repeating on every node
```

Edge timeline label:

```python
gap = overlay.path_gap_by_customer_edge.get(edge.id)
if gap:
    return f"gap avg {fmt(gap.avg_ms)}"
return _customer_edge_metric_label(edge, overlay)
```

The visualizer must prefer exact `path_timing.inter_plugin_gap_ms` over diagnostic aggregate edge
labels. If only diagnostic edge labels are available, keep the label as `queue avg ...` or
`transport avg ...`, never `gap`.

Latency table cells must obey availability flags:

```python
def _latency_cell(latency, key, availability_key=None):
    if latency.get("samples", 0) == 0:
        return "N/A"
    if availability_key and latency.get(availability_key) is False:
        return "N/A"
    return _fmt(latency.get(key), " ms")
```

#### Cards

Update `_render_metric_cards()` cards:

- `Output pulls/s`: use `graph_metrics.throughput.outputs_per_s` if present.
- `Logical inf/s`: use `graph_metrics.throughput.logical_inferences_per_s`.
- `Measured outputs`: from `measurement.outputs` or `graph_metrics.outputs_pulled`.
- `Elapsed`: from `graph_metrics.window.elapsed_seconds` or `graph_metrics.elapsed_seconds`.
- `E2E p50/p95`: show only if `graph_e2e_latency_ms.available == true`; otherwise `E2E unavailable` with status.
- `Power`: graph-level only; show status, avg W, energy, samples when collected.
- `Plugin latency`: status/source.
- `Edge/message`: status/source.
- `Timeline`: show `collected`, `partial`, or explicit unavailable status from `run.path_timing`.

#### Tables

Add or improve:

1. **Graph semantics summary** table:
   - measurement scope
   - throughput formula
   - E2E availability/status
   - node latency semantics
   - edge latency semantics
   - power attribution

2. **Customer edge details** table:
   - customer edge id
   - label
   - lowered edge ids
   - metric policy
   - displayed metric
   - mapping status/error

3. **Timeline/path timing** table:
   - row type: `entry_to_node`, `inter_plugin_gap`, `output_tail`
   - customer node/edge id
   - lowered edge id or plugin instance ids
   - samples
   - avg/p50/p95/max
   - source
   - identity key used
   - status/error

4. **Unavailability summary**:
   - If `path_timing.available=false`, show status/reason once above graph.
   - If graph E2E is unavailable but path timing exists, say:
     `Per-hop timeline is available; public push→pull E2E is unavailable because <status>.`

5. Existing node/plugin/edge tables updated to:
   - show `N/A` for unavailable percentiles.
   - show reliability warning when `reliable == false`.
   - show `max_semantics` in edge table.

#### Privacy/redaction

Add:

```python
def _redact_payload(payload, mode):
    if mode != "customer": return payload
    redact keys: "argv", "hostname", "pid", "uri", "source_path", "path"
    redact strings containing:
      "/workspace", "/home/", "rtsp://", "file://", "udp://"
    replacement: "<redacted>"
```

HTML behavior:

- `--redact customer`: raw JSON details uses redacted payload.
- Default: include warning before raw JSON:

```text
Raw JSON can include hostnames, argv, paths, and stream URIs. Use --redact customer before sharing externally.
```

### Phase 4: Schema and strict validator

Files:

```text
schemas/graph_run_v1.schema.json
tests/perf/tools/graph_run_schema.py
tests/perf/tools/test_graph_run_schema.py
```

#### Schema additions

Add optional `graph.customer_view`:

```json
"customer_view": {"$ref": "#/$defs/customer_view"}
```

Add optional `run.path_timing`:

```json
"path_timing": {"$ref": "#/$defs/path_timing"}
```

Add defs:

```json
"customer_view": {
  "type": "object",
  "required": ["topology_source", "nodes", "edges", "mapping"],
  "properties": {
    "topology_source": {"type": "string"},
    "nodes": {"type": "array", "items": {"$ref": "#/$defs/customer_node"}},
    "edges": {"type": "array", "items": {"$ref": "#/$defs/customer_edge"}},
    "mapping": {"$ref": "#/$defs/customer_mapping"},
    "warnings": {"type": "array", "items": {"type": "string"}}
  },
  "additionalProperties": true
}
```

Add path timing def:

```json
"path_timing": {
  "type": "object",
  "required": ["available", "status", "source", "aggregation"],
  "properties": {
    "available": {"type": "boolean"},
    "status": {"type": "string"},
    "source": {"type": "string"},
    "reason": {"type": "string"},
    "aggregation": {"type": "string"},
    "warnings": {"type": "array", "items": {"type": "string"}},
    "identity": {"type": "object"},
    "node_arrival_ms": {
      "type": "array",
      "items": {"$ref": "#/$defs/path_timing_row"}
    },
    "inter_plugin_gap_ms": {
      "type": "array",
      "items": {"$ref": "#/$defs/path_timing_row"}
    },
    "output_tail_ms": {
      "type": "array",
      "items": {"$ref": "#/$defs/path_timing_row"}
    }
  },
  "additionalProperties": true
},
"path_timing_row": {
  "type": "object",
  "required": ["semantics", "latency_ms"],
  "properties": {
    "customer_node_id": {"type": "string"},
    "customer_edge_id": {"type": "string"},
    "customer_output_id": {"type": "string"},
    "lowered_node_id": {"type": "string"},
    "runtime_node_id": {"type": "integer"},
    "lowered_edge_id": {"type": "string"},
    "stream_id": {"type": "string"},
    "from_plugin_instance_id": {"type": "string"},
    "to_plugin_instance_id": {"type": "string"},
    "semantics": {"type": "string"},
    "source": {"type": "string"},
    "status": {"type": "string"},
    "latency_ms": {"$ref": "#/$defs/latency_summary"}
  },
  "additionalProperties": true
}
```

Customer node/edge ID patterns:

```json
"id": {"type": "string", "pattern": "^c[0-9]+$"}
"edge id": {"type": "string", "pattern": "^ce[0-9]+$"}
```

Enums to enforce in Python strict validator, not necessarily JSON schema for compatibility:

```python
MEASUREMENT_SCOPES = {"run_lifetime", "measured_window"}
EDGE_SEMANTICS = {"edge_transport", "queue_residence", "boundary_flow"}
POWER_STATUS = {
  "disabled_by_options", "not_configured_on_run", "enabled_no_samples",
  "unavailable_all_rails_failed", "collected", "collected_with_errors"
}
PATH_TIMING_STATUS = {
  "collected", "partial", "diagnostic_aggregate_only",
  "unavailable_sample_identity_missing", "unavailable_message_trace_disabled",
  "unavailable_no_trace_capable_plugins", "unavailable_lineage_ambiguous",
  "unavailable_multi_input_lineage_ambiguous",
  "unavailable_multi_output_lineage_ambiguous",
  "failed"
}
SOURCE_STATUS = {"off", "collected", "unavailable", "failed"}
```

#### Strict validator CLI

Update `tests/perf/tools/graph_run_schema.py` so it can run as a CLI:

```bash
python3 tests/perf/tools/graph_run_schema.py --strict --customer artifact.json
```

CLI behavior:

```python
parser.add_argument("input", type=Path)
parser.add_argument("--strict", action="store_true")
parser.add_argument("--customer", action="store_true")
parser.add_argument("--allow-private", action="store_true")
```

Strict checks:

- Lowered node IDs unique.
- Lowered edge IDs unique.
- Lowered edge endpoints reference existing lowered nodes.
- Public node IDs unique.
- Public edge endpoints reference existing public nodes.
- Public `runtime_edges` reference existing lowered edge IDs unless edge is virtual.
- Customer node IDs unique.
- Customer edge endpoints reference existing customer nodes.
- Customer mappings reference existing public/lowered/customer IDs.
- `run.node_metrics[*].node_id` exists in lowered nodes unless fallback allowed.
- `run.node_metrics[*].public_node_ids[]` exist in public/customer mapping when public view is present.
- `run.edge_metrics[*].lowered_edge_id`, when non-null, exists in lowered edges.
- Numeric-only LTTng edge IDs are rejected in strict exported JSON; parser/export must canonicalize them to `eN`.
- `plugin.calls == plugin.latency_ms.samples` when both present.
- `edge.samples == edge.latency_ms.samples` when both present.
- `path_timing.available == true` requires at least one row in `node_arrival_ms`,
  `inter_plugin_gap_ms`, or `output_tail_ms`.
- `path_timing.available == false` requires non-empty `status` and `reason` or `status != collected`.
- `path_timing.inter_plugin_gap_ms[*].customer_edge_id` references a customer edge when customer view exists.
- `path_timing.node_arrival_ms[*].customer_node_id` references a customer node when customer view exists.
- No negative counters/durations/samples/power values.
- No `power` object under node/plugin/edge metrics.
- `plugin_latency.status == "collected"` requires plugin rows or warning explaining no trace-capable plugins.
- `edge_message_latency.status == "collected"` requires non-empty `edge_metrics`.

Customer gate checks:

- Fail on `graph.customer_view.topology_source == "node_metrics_fallback"` unless explicitly allowed.
- Fail on unredacted private strings unless `--allow-private`:
  - `rtsp://`
  - `file://`
  - `/workspace`
  - `/home/`
  - absolute model paths
  - hostnames and argv in raw identity blocks.

### Phase 5: Tests

#### Existing tests to extend

`tests/graph_migration/unified/phaseA4_run_export_test.cpp`

Add:

```cpp
require(json.at("graph").contains("customer_view"), "export should include customer_view");
const auto& cv = json.at("graph").at("customer_view");
require(cv.at("topology_source").get<std::string>() != "", "customer_view source required");
require(customer_view_has_node(cv, "image"), "customer_view should contain image");
require(customer_view_has_node(cv, "classes"), "customer_view should contain classes");
require(customer_view_has_edge(cv, "image", "classes"), "customer_view should connect image/classes");
```

Add graph integrity assertions:

```cpp
require(unique_ids(graph.at("lowered_view").at("nodes"), "id"), "lowered node ids unique");
require(all_lowered_edges_resolve(graph.at("lowered_view")), "lowered edge endpoints resolve");
require(graph_metric_formula_ok(json.at("run").at("graph_metrics")),
        "throughput must equal outputs_pulled / elapsed_seconds within tolerance");
```

Add no-power-under-node/plugin/edge assertions:

```cpp
for (const auto& node : measured_json.at("run").value("node_metrics", json::array())) {
  require(!node.contains("power"), "node metrics must not contain power");
  for (const auto& plugin : node.value("plugins", json::array())) {
    require(!plugin.contains("power"), "plugin metrics must not contain power");
  }
}
for (const auto& edge : measured_json.at("run").value("edge_metrics", json::array())) {
  require(!edge.contains("power"), "edge metrics must not contain power");
}
```

`tests/graph_migration/unified/phase3_metrics_report_test.cpp`

Add measured counter and min/max assertions:

```cpp
require(report.outputs_pulled == 1, "measured outputs_pulled delta must count pulls");
require(report.inputs_pushed == 1, "measured inputs_pushed delta must count pushes");
require(report.inputs_enqueued == 1, "inputs_enqueued delta must be exposed");
for (const auto& node : report.node_metrics) {
  require(node.latency.total_ms >= 0.0, "node total latency non-negative");
  if (node.latency.samples > 0) {
    require(close_enough(node.latency.avg_ms,
                         node.latency.total_ms / double(node.latency.samples)),
            "node avg must equal total/samples");
  }
  require(!node.latency.min_max_available,
          "measured-window node min/max must not be fabricated from cumulative counters");
}
```

#### New C++ tests

Add `tests/graph_migration/unified/phaseA4_customer_view_export_test.cpp`.

Topology:

```cpp
camera_left  -> Branch(left_preview, left_model)
camera_right -> Branch(right_preview, right_model)
metadata     -> Branch(meta_debug, meta_archive)
```

Assertions:

```cpp
const auto& cv = json.at("graph").at("customer_view");
require(count_kind(cv, "Input") == 3, "customer_view should have 3 inputs");
require(count_kind(cv, "Branch") == 3, "customer_view should have 3 branch routers");
require(count_kind(cv, "Output") == 6, "customer_view should have 6 outputs");
require(cv.at("edges").size() == 9, "customer_view should have input->branch and branch->output edges");
require(no_customer_invalid_runtime(cv), "customer view must not expose invalid runtime nodes");
require(branch_nodes_map_to_fanout(cv, json.at("graph").at("lowered_view")),
        "branch customer nodes should map to lowered FanOut nodes");
require(all_customer_edges_have_policy(cv), "customer edges need metric_policy");
```

Register near existing Phase A4 tests in `tests/CMakeLists.txt`.

Add or rename the current demo to a contract test:

```text
tests/graph_migration/unified/phaseA4_branched_export_contract_test.cpp
```

Assertions:

- `graph.mode == "connected"`.
- named inputs exactly `camera_left,camera_right,metadata`.
- named outputs exactly `left_preview,left_model,right_preview,right_model,meta_debug,meta_archive`.
- lowered view has 3 `FanOut` nodes with `compiler_generated=true`.
- measured edge metrics size is 9 when metrics are enabled.
- every edge metric samples equals frame count.
- public edges have mapped runtime edges or explicit virtual marker.

#### Python fixtures

Add fixtures:

```text
tests/perf/fixtures/graph_run/detr_measured_window.graph_run.json
tests/perf/fixtures/graph_run/branched_report_demo.graph_run.json
tests/perf/fixtures/graph_run/synthetic_multistream.graph_run.json
tests/perf/fixtures/graph_run/bad_dangling_edge.graph_run.json
tests/perf/fixtures/graph_run/bad_privacy.graph_run.json
```

Fixture update policy:

- Generated from DevKit runs.
- Checked into repo only after redacting paths/hostnames/argv.
- Keep small: 10-25 frames, no raw trace dumps.

#### Python schema tests

In `tests/perf/tools/test_graph_run_schema.py`, add:

```python
def test_rejects_duplicate_lowered_node_ids(self): ...
def test_rejects_dangling_lowered_edge_endpoint(self): ...
def test_rejects_public_edge_to_missing_public_node(self): ...
def test_rejects_public_runtime_edge_reference_to_missing_lowered_edge(self): ...
def test_rejects_metric_node_not_in_topology_strict(self): ...
def test_rejects_numeric_only_exported_edge_id_strict(self): ...
def test_rejects_edge_sample_mismatch(self): ...
def test_rejects_collected_plugin_status_without_rows(self): ...
def test_edge_metrics_must_not_contain_power(self): ...
def test_customer_gate_rejects_fallback_topology(self): ...
def test_customer_gate_rejects_private_strings_without_allow_private(self): ...
```

#### Python visualizer tests

In `tests/perf/tools/test_graph_run_visualizer.py`, add:

```python
def test_auto_selects_customer_view(self):
    ... assert "Topology view: customer_view_v1" in body

def test_branched_customer_view_has_three_ranks(self):
    ... assert x coordinates include exactly input, branch, output ranks

def test_svg_output(self):
    ... run --svg graph.svg and assert graph.svg starts with XML/SVG and has no HTML

def test_topology_json_output(self):
    ... run --topology-json selected.json and assert selected view is customer

def test_unavailable_latency_not_zero(self):
    ... assert "E2E unavailable" in body and ">0.000 ms<" not in E2E card

def test_timeline_status_rendered_once(self):
    ... assert "Timeline unavailable" in body and body.count("Timeline unavailable") == 1

def test_timeline_gap_overrides_diagnostic_edge_label(self):
    ... assert "gap avg" in body and "queue avg" not in selected customer edge label

def test_redaction(self):
    ... run --redact customer and assert no /workspace, /home/, rtsp://, argv, hostname

def test_strict_validate_bad_json_fails(self):
    ... subprocess returns non-zero
```

#### Performance regression gates

Facts from current code:

- `BlockingQueue<T>` already timestamps enqueue/dequeue and updates residence atomics on every
  push/pop.
- LTTng message events are controlled by `MeasureOptions.include_message_latency` and are off by
  default.

Add explicit gates so this stays a performance feature, not a performance tax:

```text
include/graph/runtime/BlockingQueue.h
src/pipeline/runtime/RunMeasure.cpp
src/pipeline/runtime/EdgeMetrics.cpp
tests/perf/tools/test_metrics_overhead.py
tests/graph_migration/unified/phaseA4_metrics_overhead_smoke_test.cpp
```

Concrete queue-timing gate:

```cpp
// include/graph/runtime/BlockingQueue.h
enum class QueueTelemetryLevel : std::uint8_t {
  CountersOnly,
  Timing,
};

mutable std::atomic<QueueTelemetryLevel> telemetry_level_{QueueTelemetryLevel::CountersOnly};

void set_telemetry_level(QueueTelemetryLevel level) noexcept {
  telemetry_level_.store(level, std::memory_order_release);
}

bool timing_enabled() const noexcept {
  return telemetry_level_.load(std::memory_order_acquire) == QueueTelemetryLevel::Timing;
}
```

Keep cheap counters/high-watermark always on. Guard `steady_clock::now()` and residence timing:

```cpp
using TimePoint = std::chrono::steady_clock::time_point;
const bool timing = timing_enabled();
const auto t0 = timing ? std::chrono::steady_clock::now() : TimePoint{};
...
if (timing) record_push_wait(t0);
queue_.push_back(QueueEntry{std::move(item), timing ? std::chrono::steady_clock::now()
                                                    : TimePoint{}, timing});
...
if (entry.has_enqueue_time) record_residence(entry.enqueue_time);
```

`QueueEntry` becomes:

```cpp
struct QueueEntry {
  T value;
  std::chrono::steady_clock::time_point enqueue_time{};
  bool has_enqueue_time = false;
};
```

`BlockingQueue<T>::Stats` must include a `timing_enabled` or `telemetry_level` field so
`EdgeMetrics.cpp` can export `status="timing_disabled"` instead of zero latency when timing is off.

Runtime control:

```cpp
void set_graph_queue_timing_enabled(RunCore& core, bool enabled) {
  const auto level = enabled ? QueueTelemetryLevel::Timing : QueueTelemetryLevel::CountersOnly;
  if (core.graph_execution_) {
    for (auto& st : core.graph_execution_->stages) st->inbox.set_telemetry_level(level);
    for (auto& pipe : core.graph_execution_->pipelines) {
      if (pipe && pipe->transport.input_queue) pipe->transport.input_queue->set_telemetry_level(level);
    }
    for (auto& [node, sink] : core.graph_execution_->sinks) {
      if (sink) sink->set_telemetry_level(level);
    }
  }
}
```

`Run::start_measurement()` enables queue timing only when
`MeasureOptions.include_edge_latency || MeasureOptions.include_message_latency`. `MeasureScope::stop()`
disables it again before returning. Existing diagnostic counters remain available with timing off;
latency fields must mark `status="timing_disabled"` instead of exporting zeros.

Required checks:

- Default measurement with `include_message_latency=false` does not enable `sima_neat_edge:*`.
- `include_message_latency=true` is the only mode that emits exact message trace events.
- Queue residence stats do not call `steady_clock::now()` when queue timing is disabled and do not
  allocate per message beyond the queue entry already stored.
- DETR throughput with default metrics is within the agreed tolerance of no-metrics baseline.
- Branch/FanOut throughput with customer-view export does not regress relative to existing
  diagnostics-only export.
- Combine/JoinBundle customer view exposes 2 inputs -> Combine -> 1 output, maps Combine to
  JoinBundle when present, and output-pull throughput counts emitted bundles/frames, not input
  pushes.

Acceptance thresholds:

```text
default metrics overhead: <= agreed tolerance, target <= 2% on representative graph runs
exact message tracing overhead: measured and documented, not part of default customer path
visualizer runtime: < 1s for checked-in DETR/branch fixtures, < 5s for 500-node synthetic graph
HTML size: no raw trace dumps embedded; only aggregated metrics/topology
```

### Phase 6: Real app validation

#### DETR

Use current DETR app, but do not accept fallback topology as customer-ready.

Pass assertions:

```text
schema passes strict validator
graph.customer_view.topology_source != node_metrics_fallback
graph.customer_view.topology_source == customer_view_v1 for customer-ready artifacts
graph.named_inputs non-empty
graph.named_outputs non-empty
customer/lowered nodes include Input -> QuantTess -> MLA/ModelFragment -> DetessDequant -> Postproc -> Output when present
run.graph_metrics.outputs_pulled >= 30
run.graph_metrics.throughput.outputs_per_s > 0
plugin latency either contains expected backends or status explains unavailable/partial
edge metrics non-empty or status explains unavailable
path_timing is collected when include_message_latency=true, otherwise diagnostic_aggregate_only or unavailable with reason
QuantTess customer node shows either node latency or entry→node timeline label; endpoint nodes are labelled endpoint/not timed
postproc plugin either has attributed plugin row or explicit unavailable/no-trace-capable status
power on DVT is disabled/not_configured/enabled_no_samples, not a failure
no unexpected warnings
no intermittent preflight timeout across repeated run
```

Known failure to carry forward:

```text
DETR customer cleanup/preflight run previously had timeout waiting for output.
```

#### Multi-stream object detector

Pass assertions:

```text
process exits 0
one JSON per detector graph/runtime if needed
outputs_pulled >= number_of_streams
throughput explicitly labelled output-pulls/s
plugin latency disabled or serialized safely for concurrent runs
if one Run owns multiple streams, one LTTng collector may trace all streams; rows must separate by stream_id and plugin_instance_id
if multiple Run measurement scopes are active in one process, only one may own LTTng at a time; others must fail clearly or fall back to diagnostics, never silently mix traces
no process-global profiler/LTTng conflict
no heap corruption on close
```

#### Branch/FanOut

Use in-tree C++ branch contract test, not temp app:

```text
tests/graph_migration/unified/phaseA4_customer_view_export_test.cpp
```

Do not use `/home/docker/sima-cli/tmp/branched_graph_metrics/main.cpp` as release evidence because it is temporary and was modified during debugging.

Pass assertions:

```text
3 inputs
3 Branch nodes
6 outputs
9 customer edges
output pulls = frames * 6
branch nodes map to FanOut/Tee/Fork when lowered router exists, otherwise mapping_status=partial
```

#### Combine/JoinBundle

Add and use:

```text
tests/graph_migration/unified/phaseA4_customer_view_combine_export_test.cpp
```

Pass assertions:

```text
2 inputs
1 Combine node
1 output
Combine maps to JoinBundle/Join/Combine when lowered router exists, otherwise mapping_status=partial
output pulls = frames
graph E2E unavailable/partial is explicit if multi-input lineage is ambiguous
```

#### People-tracker

Release blocker until clean:

```text
1-stream plugin metrics exits 0
2-stream no-plugin metrics exits 0
all JSON files are written
no "Aborted"
no "malloc(): mismatching next->prev_size"
no "LatencyProfiler: another profiler is already active"
```

### Phase 7: Docs, CLI install, and deployment

Docs to add:

```text
docs/how-to/graph-run-visualization.md
docs/reference/graph_run_v1.md
```

App docs to update:

```text
DETR README / object-detection app README
multi-stream app README
```

Required doc topics:

- How to enable metrics:

```cpp
neat::RunOptions opt;
opt.enable_metrics = true;
// Power only on SOM/customer power validation:
// opt.enable_board_power();
auto run = graph.build(opt);

neat::MeasureOptions m;
m.include_plugin_latency = true;
m.include_edge_latency = true;
m.include_message_latency = false; // low-overhead default; true for exact per-message timeline
m.include_power = false; // DVT default
m.logical_batch_size = 1;
auto scope = run.start_measurement(m);
// app pushes/pulls here
auto report = scope.stop();
neat::save_run_json(run, report, "graph_metrics.json");
```

- How to render:

```bash
sima-neat-visualize-graph-run graph_metrics.json \
  --view auto \
  -o graph_metrics.html \
  --svg graph_metrics.svg \
  --validate --strict
```

- Interpretations:
  - Throughput is output-pulls/s by default.
  - Multi-output graphs count each successful named/default pull; do not call it frame FPS unless
    the app pulls exactly one output per frame/inference.
  - Node latency is element residency and non-additive.
  - Plugin latency is plugin/kernel span and non-additive.
  - Edge latency is handoff/queue/transport and non-additive.
  - Exact path/timeline timing requires sample identity plus message tracing; otherwise the report
    shows aggregate edge diagnostics and marks timeline unavailable/diagnostic-only.
  - Graph E2E may be unavailable for graph-backed/multi-output runs until identity correlation exists.
  - Power is graph-level board/SOM rail average only.
  - DVT board power is not reliable; use SOM for customer power validation.

- Privacy:
  - Raw JSON may include hostnames, argv, absolute paths, RTSP/UDP URIs.
  - Use `--redact customer` before sharing externally.

Install/deployment:

- Install visualizer as a console script or tool wrapper:

```text
sima-neat-visualize-graph-run
sima-neat-validate-graph-run
```

- Install schema:

```text
share/sima-neat/schemas/graph_run_v1.schema.json
```

- Add package smoke test:

```bash
sima-neat-validate-graph-run --strict sample.graph_run.json
sima-neat-visualize-graph-run sample.graph_run.json -o sample.html --svg sample.svg
```

ABI:

- Minimize public ABI/API changes.
- `include/graph/StageExecutor.h` must **not** change for metrics-only edge identity; use internal
  runtime queue wrappers as described above.
- `include/pipeline/GraphOptions.h::Sample` must **not** gain a new trace/sample ID in this phase;
  reuse existing `stream_id`, `frame_id`, `input_seq`, and `orig_input_seq`.
- Decide before coding:
  - Option A, ABI bump accepted: add typed public `MeasurePathTiming` fields to
    `include/pipeline/Run.h::MeasureReport`.
  - Option B, no ABI bump: keep path timing private and export it only through JSON.
  For this report-tool work, prefer Option B unless customer C++ access is explicitly required.
- Prefer placing large new structs behind internal/export helpers where possible so future metrics
  fields can be added to JSON/schema without repeatedly growing public ABI structs.
- Customer validation must use matched headers, core library, runtime plugins, and app binaries.

---

## Acceptance criteria

### Graph visualization

- Branch artifact default HTML shows customer topology: 3 inputs -> 3 Branch routers -> 6 outputs.
- DETR renders with real named topology; if fallback is used, customer strict gate fails and HTML warns visibly.
- `--view lowered` still shows compiler-generated runtime nodes for debugging.
- Edge labels never sum non-additive metrics.
- Unavailable percentiles/E2E latency show `N/A` or explicit unavailable status, not `0.000 ms`.
- Timeline/path timing shows graph-entry-to-node, inter-plugin gap, and output-tail only when
  sample identity supports it; otherwise it shows one explicit unavailable reason.
- No node/plugin/edge power rows.

### Metrics correctness

- Measured JSON has `run.measured_stats` and `run.graph_metrics.window`.
- Throughput card says `Output pulls/s`, not generic FPS.
- Multi-output graph shows multi-output throughput caveat.
- Power status distinguishes disabled/not configured/enabled no samples/unavailable/collected.
- Diagnostic edge max is labeled as lifetime high-water when applicable.
- Exact between-plugin timing uses LTTng/message identity; diagnostic aggregate counters are never
  relabeled as exact timeline gaps.
- Default customer metrics path keeps exact per-message tracing off and passes the agreed overhead
  gate before release.

### Validation

- Strict validator catches duplicate IDs, dangling edges, bad runtime references, metrics pointing to missing nodes, and private strings under customer gate.
- Visualizer tests cover customer view, branch layout, SVG export, topology JSON export, redaction, and strict validation failure.
- Visualizer/schema tests cover `path_timing` collected, partial, and unavailable cases.
- C++ export tests cover simple, branched, and fallback topologies.

### Real app confidence

- DETR measured run passes strict customer gate.
- Multi-stream run exits cleanly and exports valid JSON(s).
- People-tracker metrics either exits cleanly or remains a documented release blocker.
- Clean checkout builds with all required files tracked.

---

## Suggested implementation order

1. Track/remove untracked files and get clean build inputs.
2. Add `graph.customer_view` export and branch customer-view C++ contract test.
3. Add strict schema/validator referential-integrity checks.
4. Rewrite visualizer view selection and DAG layout; add `--svg`, `--topology-json`, `--summary`.
5. Fix throughput/window/E2E/power/status JSON semantics.
6. Implement path/timeline timing status and exact LTTng-message path rows where events exist.
7. Update visualizer cards/tables/labels for new semantics.
8. Add redaction and customer strict gates.
9. Add DETR/multi-stream/branch fixtures and Python tests.
10. Run DevKit app validation.
11. Write docs and package/install smoke tests.
