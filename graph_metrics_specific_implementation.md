# Graph Metrics Specific Implementation Spec

This document is the concrete implementation companion to `graph_metrics_implementation_plan.md`.
It is written so implementation can proceed file-by-file without re-deciding architecture.

Do **not** compile or run until every code item in sections A-G is implemented. After that, use the
validation order in section H.

---

## Non-negotiable invariants

1. **Power is graph-level only.** Never add power under node/plugin/edge rows.
2. **Throughput is graph-level output-pulls/s.** Do not label it model FPS unless one public pull equals one frame/inference.
3. **Node/plugin/edge/path timings are non-additive diagnostics.** Never sum them into graph latency.
4. **No public `StageMsg` change.** Edge identity is carried only by private runtime queue wrappers.
5. **No public `Sample` trace-id addition in this phase.** Use existing `stream_id`, `frame_id`, `input_seq`, `orig_input_seq`.
6. **Unavailable timing is never `0.000 ms`.** Emit `available=false` or availability flags.
7. **Per-sample path timing must use identity.** Do not pair by nearest timestamp.
8. **DVT power can be unavailable/unreliable.** Keep SOM power implementation and expose status clearly.
9. **Default metrics path must stay low-overhead.** Exact message tracing and queue residence timing
   are opt-in during measurement, not always-on runtime tax.

### ABI decision before coding

This spec currently adds `MeasurePathTiming` structs to public `include/pipeline/Run.h`. That is
acceptable only if the branch is allowed to change the SDK ABI or all consuming apps/plugins are
rebuilt together. If ABI must remain stable, do **not** add those public structs; instead:

- keep path timing in private runtime structs,
- serialize it only through `GraphMetricJson::path_timing_to_json(...)`,
- expose status through JSON reports, not `MeasureReport`.

Do not start implementation until this ABI decision is explicit in the PR/branch notes.

### Subagent hardening corrections already applied to this spec

The subagent review found five correctness issues that are now requirements:

1. **No clock-domain mixing.** `GraphSampleTimingEvent`/`steady_clock` is only for graph E2E.
   Exact path timing must use LTTng timestamps for graph-entry, edge/node/plugin, and public-pull
   events.
2. **Sink queues carry edge identity too.** Do not recover sink edge identity from
   `sink_node -> edge_index`; use `RuntimeSinkQueueMsg`.
3. **Message IDs match adopted plugin tracepoints.** Runtime message ID is
   `orig_input_seq`, else `input_seq`, else `frame_id`, else `0` for missing. Do not hash.
4. **Trace fast path does not allocate.** Check `graph_message_trace_enabled()` before constructing
   `TraceGraphMessageArgs`.
5. **Queue timing defaults off.** `BlockingQueue` starts in `CountersOnly`; measurement enables
   timing only when edge/message latency is requested.

---

## A. Add canonical metric JSON helpers

### A1. Create `src/pipeline/runtime/GraphMetricJson.h`

```cpp
#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/GraphMetrics.h"
#include "pipeline/PowerTelemetry.h"
#include "pipeline/Run.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string_view>

namespace simaai::neat::runtime {

nlohmann::ordered_json throughput_json(std::uint64_t output_pulls,
                                       double elapsed_s,
                                       int logical_batch_size);

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
                                      std::string_view empty_status);

nlohmann::ordered_json path_timing_to_json(const MeasurePathTiming& timing,
                                           bool trace_loss_detected);

std::string canonical_lowered_edge_id(std::string raw);

} // namespace simaai::neat::runtime
```

### A2. Create `src/pipeline/runtime/GraphMetricJson.cpp`

Implement exactly these semantics:

```cpp
nlohmann::ordered_json throughput_json(std::uint64_t output_pulls,
                                       double elapsed_s,
                                       int logical_batch_size) {
  const bool valid_window = elapsed_s > 0.0 && std::isfinite(elapsed_s);
  const double outputs_per_s = valid_window ? static_cast<double>(output_pulls) / elapsed_s : 0.0;
  const int batch = logical_batch_size <= 0 ? 1 : logical_batch_size;
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

`power_status_json()` status rules:

```cpp
std::uint64_t rail_power_successes = 0;
std::uint64_t rail_power_errors = 0;
for (const PowerRailSummary& rail : power.rails) {
  rail_power_successes += rail.power_w.samples;
  rail_power_errors += rail.power_w.errors;
}

std::string status;
if (!export_requested) {
  status = "disabled_by_options";
} else if (!monitor_configured) {
  status = "not_configured_on_run";
} else if (power.samples == 0 && rail_power_errors == 0) {
  status = "enabled_no_samples";
} else if (power.samples == 0 && rail_power_errors > 0) {
  status = "unavailable_all_rails_failed";
} else if (rail_power_errors > 0) {
  status = "collected_with_errors";
} else {
  status = "collected";
}
```

Every power JSON object must contain:

```json
"attribution": "graph_level_only"
```

and this note:

```text
Power is board/SOM rail telemetry. DVT board readings may be unavailable or unreliable.
```

`canonical_lowered_edge_id()` must normalize:

- `"7"` -> `"e7"`
- `"e7"` -> `"e7"`
- empty -> empty string
- diagnostic synthetic IDs unchanged

---

## B. Add customer topology export

### B1. Create `src/pipeline/runtime/CustomerGraphView.h`

```cpp
#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/runtime/ExecutionGraphPlan.h"

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

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

### B2. Create `src/pipeline/runtime/CustomerGraphView.cpp`

Use `using json = nlohmann::ordered_json;` and keep all build structs private to this file.

Required private helpers:

```cpp
constexpr std::size_t npos = static_cast<std::size_t>(-1);

std::string runtime_node_json_id(graph::NodeId id);
std::string lowered_edge_json_id(std::size_t edge_index);
std::string public_node_json_id(std::size_t public_index);
std::string public_edge_json_id(std::size_t public_edge_index);

bool has_runtime_node(const PublicGraphNodePlan& node);
std::string node_role(const PublicGraphNodePlan& node);
json customer_node_to_json(const CustomerNodeBuild& node);
json customer_edge_to_json(const CustomerEdgeBuild& edge);
```

Required build structs:

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

Required typed topology index:

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
```

`build_plan_topology_index()` must iterate current real fields:

- `plan.edges[e].from`
- `plan.edges[e].to`
- `plan.public_edges[e].from`
- `plan.public_edges[e].to`
- `plan.stage_nodes[i].node_id`
- every `plan.pipeline_segments[i].node_ids[j]`

Do **not** use non-existent fields such as `from_node_index`, `runtime_port`, or `segment.runtime_node`.

Required customer construction algorithm:

1. Create one customer node for every public node where `runtime_node != graph::kInvalidNode`.
2. For invalid public nodes:
   - out-degree > 1 -> Branch router
   - in-degree > 1 -> Combine router
   - in-degree == 1 and out-degree == 1 -> collapse pass-through into edge
3. For Branch:
   - find nearest upstream valid public node by reverse public BFS
   - find all nearest downstream valid public nodes by forward public BFS
   - find lowered router candidate by shortest lowered paths from upstream to each downstream
   - prefer lowered kind `FanOut`, `Tee`, or `Fork`; otherwise prefer first common divergence with out-degree > 1
4. For Combine:
   - find many upstream valid public nodes and one downstream valid node
   - prefer lowered kind `JoinBundle`, `Join`, or `Combine`; otherwise in-degree > 1
5. Create customer edges:
   - valid public node -> router
   - router -> valid public node
   - valid public node -> valid public node
   - collapsed pass-through public edge IDs included in `public_edge_ids`
6. For each customer edge:
   - `lowered_edge_ids` from explicit `PublicGraphEdgePlan::runtime_edge_indices` if present
   - otherwise from typed shortest lowered edge path
   - `metric_policy = "none"` if no lowered edges
   - `metric_policy = "single_lowered_edge"` if exactly one
   - `metric_policy = "non_additive_representative_max"` if more than one
7. Emit mapping objects:
   - `customer_to_public`
   - `public_to_customer`
   - `customer_to_lowered`
   - `lowered_to_customer`
   - `customer_edge_to_public`
   - `customer_edge_to_lowered`

`fallback_customer_view_from_lowered()` must copy lowered nodes/edges into a customer-like view with:

```json
"topology_source": "node_metrics_fallback"
"mapping_status": "fallback"
"warnings": ["Fallback topology reconstructed from node metrics; not source customer topology."]
```

---

## C. Wire customer view and canonical metrics into exporter

### C1. Modify `src/pipeline/runtime/GraphRunExport.cpp` includes

Add:

```cpp
#include "pipeline/runtime/CustomerGraphView.h"
#include "pipeline/runtime/GraphMetricJson.h"
```

### C2. Modify `graph_topology_to_json(const runtime::RunCore& core)`

After `graph["public_view"] = public_view_to_json(plan);` add:

```cpp
graph["public_view"]["topology_source"] = "execution_plan_public_view";
```

When building lowered view, include topology source:

```cpp
graph["lowered_view"] = {
    {"topology_source", "execution_plan_lowered_view"},
    {"nodes", graph["nodes"]},
    {"edges", graph["edges"]},
    {"pipeline_segments", graph["pipeline_segments"]},
};

graph["customer_view"] = runtime::customer_view_to_json(plan, graph["public_view"],
                                                         graph["lowered_view"]);
```

### C3. Modify fallback topology export

In `ensure_graph_topology_from_node_metrics(json& root)`, whenever fallback lowered view is created,
add:

```cpp
graph["customer_view"] = runtime::fallback_customer_view_from_lowered(
    graph["lowered_view"],
    "node_metrics_fallback",
    {"Fallback topology reconstructed from node metrics; not source customer topology."});
```

### C4. Replace measured-window graph metrics assembly

In `run_to_json(const Run&, const MeasureReport&, ...)`, replace the manual `run_json["graph_metrics"] = {...}` block with:

```cpp
json graph_metrics;
graph_metrics["measurement_scope"] = "measured_window";
graph_metrics["aggregation"] = "measured_window";
graph_metrics["latency_semantics"] = "sum_element_residency_delta";
graph_metrics["throughput_counting"] = "all_pulled_outputs";
graph_metrics["window"] = runtime::window_json("measured_window", report.elapsed_s,
                                                 report.options.duration_ms,
                                                 report.options.warmup_ms,
                                                 report.warmup_iterations);
const json throughput = runtime::throughput_json(report.outputs_pulled, report.elapsed_s,
                                                 report.options.logical_batch_size);
graph_metrics["throughput"] = throughput;
graph_metrics["elapsed_seconds"] = report.elapsed_s;
graph_metrics["outputs_pulled"] = report.outputs_pulled;
graph_metrics["throughput_fps"] = throughput.value("outputs_per_s", 0.0);
graph_metrics["outputs_per_s"] = throughput.value("outputs_per_s", 0.0);
graph_metrics["throughput_batches_per_s"] = throughput.value("outputs_per_s", 0.0);
graph_metrics["counters"] = {
    {"inputs_enqueued", report.inputs_enqueued},
    {"inputs_dropped", report.inputs_dropped},
    {"inputs_pushed", report.inputs_pushed},
    {"outputs_ready", report.outputs_ready},
    {"outputs_pulled", report.outputs_pulled},
    {"outputs_dropped", report.outputs_dropped},
};
graph_metrics["power"] = runtime::power_status_json(report.power,
                                                     opt.include_power,
                                                     report.power.enabled,
                                                     "board_rail_power_during_measured_window");
run_json["graph_metrics"] = std::move(graph_metrics);
```

### C5. Update `edge_metric_to_json()`

At the start:

```cpp
const std::string lowered_id = runtime::canonical_lowered_edge_id(e.edge_id);
const bool lowered_mapped = !lowered_id.empty() && lowered_id.size() > 1 && lowered_id[0] == 'e' &&
                            std::all_of(lowered_id.begin() + 1, lowered_id.end(),
                                        [](unsigned char c) { return std::isdigit(c) != 0; });
```

Emit:

```cpp
out["edge_id"] = e.edge_id.empty() ? json(nullptr) : json(lowered_id.empty() ? e.edge_id : lowered_id);
out["lowered_edge_id"] = lowered_mapped ? json(lowered_id) : json(nullptr);
out["mapping_status"] = lowered_mapped ? "mapped" : "diagnostic_unmapped";
```

Inside `latency_ms`, add:

```cpp
{"percentiles_available", e.source == "lttng"},
{"min_max_available", e.source == "lttng"},
{"max_semantics", e.source == "diagnostics" ? "run_lifetime_high_water"
                                             : "measured_window_sample_max"},
```

### C6. Add graph E2E and path timing export

After graph metrics:

```cpp
const std::shared_ptr<const runtime::RunCore> core_ptr = run_internal::core(run);
const bool graph_backed = core_ptr && core_ptr->graph_execution_;
run_json["graph_e2e_latency_ms"] = runtime::graph_e2e_json(
    report.end_to_end,
    graph_backed,
    graph_backed ? "unavailable_graph_e2e_not_instrumented" : "no_samples");
run_json["path_timing"] = runtime::path_timing_to_json(report.path_timing,
                                                        report.trace_loss_detected);
```

---

## D. Add graph E2E sample timing

### D1. Modify `include/pipeline/Run.h`

Add measured counters to `MeasureReport`:

```cpp
std::uint64_t inputs_enqueued = 0;
std::uint64_t outputs_ready = 0;
bool trace_loss_detected = false;
std::uint64_t graph_sample_timing_unkeyed = 0;
std::uint64_t graph_sample_timing_misses = 0;
MeasurePathTiming path_timing;
```

Add path timing public structs before `MeasureReport`:

```cpp
struct MeasurePathStat {
  std::uint64_t samples = 0;
  double avg_ms = 0.0;
  double p50_ms = 0.0;
  double p95_ms = 0.0;
  double max_ms = 0.0;
  bool reliable = true;
};

struct MeasurePathIdentity {
  std::string primary_key;
  std::string fallback_key;
  std::vector<std::string> used_public_fields;
  std::string sample_identity_source;
};

struct MeasurePathNodeArrival {
  std::string customer_node_id;
  std::string lowered_node_id;
  std::int32_t runtime_node_id = -1;
  std::string plugin_instance_id;
  std::string stream_id;
  std::string semantics = "graph_entry_to_first_node_observation";
  MeasurePathStat latency;
};

struct MeasurePathInterPluginGap {
  std::string customer_edge_id;
  std::string lowered_edge_id;
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
  std::string reason;
  std::string aggregation = "measured_window";
  std::vector<std::string> warnings;
  MeasurePathIdentity identity;
  std::vector<MeasurePathNodeArrival> node_arrival;
  std::vector<MeasurePathInterPluginGap> inter_plugin_gap;
  std::vector<MeasurePathOutputTail> output_tail;
};
```

### D2. Modify `src/pipeline/runtime/RunCore.h`

Add includes:

```cpp
#include <deque>
#include <unordered_map>
```

Add private timing types inside `namespace simaai::neat::runtime` before `RunCore`:

```cpp
enum class GraphSampleTimingKeyKind { OrigInputSeq, InputSeq, FrameId };

struct GraphSampleIdentityKey {
  std::string stream_id;
  GraphSampleTimingKeyKind kind = GraphSampleTimingKeyKind::OrigInputSeq;
  std::int64_t value = -1;
  bool operator==(const GraphSampleIdentityKey& other) const noexcept;
};

struct GraphSampleIdentityKeyHash {
  std::size_t operator()(const GraphSampleIdentityKey& key) const noexcept;
};

struct GraphSampleTimingState {
  std::chrono::steady_clock::time_point graph_entry_at{};
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
  double timestamp_s = 0.0;
};
```

Add `RunCore` methods:

```cpp
void record_graph_sample_entry(std::string_view endpoint,
                               const Sample& sample,
                               std::chrono::steady_clock::time_point at);

void record_graph_sample_output(std::string_view endpoint,
                                const Sample& sample,
                                std::chrono::steady_clock::time_point at);
```

Add `RunCore` state near existing `latency_mu` state:

```cpp
std::mutex graph_sample_timing_mu;
std::unordered_map<GraphSampleIdentityKey, GraphSampleTimingState, GraphSampleIdentityKeyHash>
    graph_sample_timing_by_key;
std::deque<GraphSampleTimingOrderEntry> graph_sample_timing_order;
std::size_t graph_sample_timing_capacity = 4096;
std::uint64_t graph_sample_timing_generation = 0;
std::atomic<std::uint64_t> graph_sample_timing_unkeyed{0};
std::atomic<std::uint64_t> graph_sample_timing_misses{0};
std::chrono::steady_clock::time_point measurement_started_at{};
std::vector<GraphSampleTimingEvent> measurement_graph_entries;
std::vector<GraphSampleTimingEvent> measurement_graph_pulls;
```

### D3. Create `src/pipeline/runtime/GraphSampleTiming.cpp`

Implement:

- `GraphSampleIdentityKey::operator==`
- `GraphSampleIdentityKeyHash`
- local `make_graph_sample_identity_key(const Sample&)`
- `RunCore::record_graph_sample_entry`
- `RunCore::record_graph_sample_output`

Key rules:

1. Use `std::string stream = sample.stream_id.empty() ? "default" : sample.stream_id;`.
2. Prefer `orig_input_seq`, then `input_seq`, then `frame_id`.
3. If all three are negative, return `std::nullopt` and increment `graph_sample_timing_unkeyed`.
4. Do not include endpoint in the primary key; endpoint is state/diagnostic metadata.
5. If a duplicate identity key arrives while an older sample is still pending, mark the state
   `ambiguous=true` instead of guessing.
6. Entry function lock order: `graph_sample_timing_mu`, release, then `latency_mu`.
7. Output function lock order: `graph_sample_timing_mu`, release, then `latency_mu`.

Entry implementation shape:

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
    auto [it, inserted] =
        graph_sample_timing_by_key.emplace(*key, GraphSampleTimingState{at, std::string(endpoint), gen});
    if (!inserted) {
      it->second.ambiguous = true;
    } else {
      graph_sample_timing_order.push_back(GraphSampleTimingOrderEntry{*key, gen});
    }
    while (graph_sample_timing_order.size() > graph_sample_timing_capacity) {
      const auto old = graph_sample_timing_order.front();
      graph_sample_timing_order.pop_front();
      auto old_it = graph_sample_timing_by_key.find(old.key);
      if (old_it != graph_sample_timing_by_key.end() && old_it->second.generation == old.generation) {
        graph_sample_timing_by_key.erase(old_it);
      }
    }
  }
  std::lock_guard<std::mutex> lock(latency_mu);
  if (measurement_active) {
    measurement_graph_entries.push_back(GraphSampleTimingEvent{
        std::string(endpoint), key->stream_id, key->kind, key->value,
        std::chrono::duration<double>(at - measurement_started_at).count()});
  }
}
```

Output implementation shape:

```cpp
void RunCore::record_graph_sample_output(std::string_view endpoint,
                                         const Sample& sample,
                                         std::chrono::steady_clock::time_point at) {
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
  const double ms = std::chrono::duration<double, std::milli>(at - state->graph_entry_at).count();
  std::lock_guard<std::mutex> lock(latency_mu);
  ++latency_count;
  if (!latency_init) {
    latency_mean_ms = latency_min_ms = latency_max_ms = ms;
    latency_init = true;
  } else {
    latency_mean_ms += (ms - latency_mean_ms) / static_cast<double>(latency_count);
    latency_min_ms = std::min(latency_min_ms, ms);
    latency_max_ms = std::max(latency_max_ms, ms);
  }
  if (measurement_active) {
    measurement_latencies_ms.push_back(ms);
    measurement_graph_pulls.push_back(GraphSampleTimingEvent{
        std::string(endpoint), key->stream_id, key->kind, key->value,
        std::chrono::duration<double>(at - measurement_started_at).count()});
  }
}
```

### D4. Modify `RunPush.cpp`

Change helper signature:

```cpp
bool push_graph_samples_to_endpoint(runtime::RunCore& core,
                                    const runtime::Endpoint& endpoint,
                                    std::string_view endpoint_name,
                                    const Sample& msgs,
                                    bool block)
```

Around `core.graph_push(...)`, record only after success so failed pushes do not leak pending
timing entries:

```cpp
const auto entry_at = std::chrono::steady_clock::now();
const bool ok = core.graph_push(endpoint.node, endpoint.port,
                                endpoint.port != simaai::neat::graph::kInvalidPort,
                                msg,
                                graph_router_options_for_push(core, block));
if (!ok) return false;
core.record_graph_sample_entry(endpoint_name, msg, entry_at);
trace_graph_message_event(TraceGraphMessageEventType::GraphEntry,
                          core.graph_execution_.get(),
                          invalid_edge_index(),
                          msg);
```

Add local helper:

```cpp
std::string default_input_name(const runtime::RunCore& core) {
  if (!core.graph_execution_ || !core.graph_execution_->plan.default_input) return "default";
  const auto& def = *core.graph_execution_->plan.default_input;
  for (const auto& [name, ep] : core.graph_execution_->plan.named_inputs) {
    if (ep.node == def.node && ep.port == def.port && ep.kind == def.kind) return name;
  }
  return "default";
}
```

Update all call sites:

- named `Run::push(name, ...)` and `Run::try_push(name, ...)`: pass `input_name`
- `RunCore::push_named_samples`: pass `input_name`
- default `RunCore::push_samples`: pass `default_input_name(*this)`
- route-processor branches must preserve the same endpoint name instead of going through default when named

### D5. Modify `RunPull.cpp`

Add local helper:

```cpp
std::string default_output_name(const runtime::RunCore& core) {
  if (!core.graph_execution_ || !core.graph_execution_->plan.default_output) return "default";
  const auto& def = *core.graph_execution_->plan.default_output;
  for (const auto& [name, ep] : core.graph_execution_->plan.named_outputs) {
    if (ep.node == def.node && ep.port == def.port && ep.kind == def.kind) return name;
  }
  return "default";
}
```

In graph-backed `RunCore::pull()` after `graph_pull()` succeeds and before moving to `out`:

```cpp
Sample value = std::move(*sample);
const auto pull_at = std::chrono::steady_clock::now();
st->record_graph_sample_output(default_output_name(*st), value, pull_at);
out = std::move(value);
```

In `RunCore::pull_named_output()` after `graph_pull()` succeeds:

```cpp
Sample value = std::move(*sample);
const auto pull_at = std::chrono::steady_clock::now();
record_graph_sample_output(output_name, value, pull_at);
trace_graph_message_event(TraceGraphMessageEventType::GraphOutputPull,
                          graph_execution_.get(),
                          invalid_edge_index(),
                          value);
out = std::move(value);
```

### D6. Modify `RunMeasure.cpp`

In `MeasureScope::stop()` after counter delta:

```cpp
report.inputs_enqueued = measured.inputs_enqueued;
report.outputs_ready = measured.outputs_ready;
```

When measurement starts under `core_->latency_mu`, also clear/set:

```cpp
core_->measurement_graph_entries.clear();
core_->measurement_graph_pulls.clear();
core_->measurement_started_at = impl->start;
```

When measurement stops under `st->latency_mu`, copy and clear:

```cpp
graph_entry_events = st->measurement_graph_entries;
graph_pull_events = st->measurement_graph_pulls;
st->measurement_graph_entries.clear();
st->measurement_graph_pulls.clear();
report.graph_sample_timing_unkeyed = st->graph_sample_timing_unkeyed.load(std::memory_order_relaxed);
report.graph_sample_timing_misses = st->graph_sample_timing_misses.load(std::memory_order_relaxed);
```

---

## E. Add internal edge identity and message trace events

### E1. Create `src/pipeline/runtime/TraceMessageEvents.h`

```cpp
#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/GraphOptions.h"
#include "graph/GraphTypes.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace simaai::neat::runtime {

struct ExecutionGraphRuntime;

constexpr std::size_t invalid_edge_index() { return static_cast<std::size_t>(-1); }

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
  std::string_view stream_id;
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

### E2. Create `src/pipeline/runtime/TraceMessageEvents.cpp`

Requirements:

- if LTTng provider headers are unavailable, compile all trace emitters as no-op
- do not define tracepoint providers inside dlopen-able plugins
- add/verify `sima_neat_edge:message` provider in `libsimaaitrace.so`
- `TraceMessageEvents.cpp` must use a guarded adapter header and compile as no-op when unavailable
- fast path returns before allocating strings when `message_trace_enabled=false`
- edge ID emitted to tracepoint is numeric `edge_index`
- `GraphEntry` and `GraphOutputPull` boundary events are allowed to use `edge_index=-1`; edge
  events still require a valid edge index
- parser/export canonicalizes numeric edge ID to `eN`

`make_trace_graph_message_args()` must fill `src_runtime_node_id` and `dst_runtime_node_id` from `execution->plan.edges[edge_index]` when `edge_index` is valid.

`make_numeric_message_id(sample)` must match the adopted plugin tracepoints
(`processmla`, `processcvu`, `objectdecode`): use original input sequence, then input sequence,
then frame ID, else zero for missing identity. Do **not** hash; hashed IDs would not match plugin
spans.

```cpp
std::uint64_t make_numeric_message_id(const Sample& sample) {
  if (sample.orig_input_seq >= 0) return static_cast<std::uint64_t>(sample.orig_input_seq);
  if (sample.input_seq >= 0) return static_cast<std::uint64_t>(sample.input_seq);
  if (sample.frame_id >= 0) return static_cast<std::uint64_t>(sample.frame_id);
  return 0;
}
```

Parser pairing must treat `message_id == 0` as missing.

### E3. Modify `ExecutionGraphRuntime.h`

Add include:

```cpp
#include "pipeline/runtime/TraceMessageEvents.h"
```

Add wrapper:

```cpp
struct RuntimeStageQueueMsg {
  simaai::neat::graph::PortId in_port = simaai::neat::graph::kInvalidPort;
  Sample sample;
  std::size_t edge_index = invalid_edge_index();
};

struct RuntimeSinkQueueMsg {
  Sample sample;
  std::size_t edge_index = invalid_edge_index();
};

using GraphSinkQueue = simaai::neat::graph::runtime::BlockingQueue<RuntimeSinkQueueMsg>;
```

Modify `DownstreamTarget`:

```cpp
std::size_t edge_index = invalid_edge_index();
```

Modify `StageRuntime`:

```cpp
explicit StageRuntime(std::size_t capacity = 0) : inbox(capacity) {}
simaai::neat::graph::runtime::BlockingQueue<RuntimeStageQueueMsg> inbox;
```

Remove `graph::runtime::StageMailbox mailbox` from `StageRuntime`.

Replace all current mailbox references:

```cpp
stage->mailbox.inbox.close()  -> stage->inbox.close()
stage.mailbox.inbox.push(...) -> stage.inbox.push(...)
stage->mailbox.inbox.stats() -> stage->inbox.stats()
st.mailbox.inbox.pop(...)    -> st.inbox.pop(...)
```

Current files to update include:

```text
src/pipeline/runtime/RunCore.cpp
src/pipeline/runtime/EdgeMetrics.cpp
src/pipeline/runtime/RunCoreGraphStart.cpp
src/pipeline/runtime/ExecutionGraphRuntime.h
```

Add trace atomics to `ExecutionGraphRuntime` and change the existing `sinks` map:

```cpp
std::atomic<bool> message_trace_enabled{false};
std::atomic<std::uint64_t> trace_run_id_hash{0};
std::atomic<std::uint64_t> trace_graph_id_hash{0};
std::unordered_map<simaai::neat::graph::NodeId, std::shared_ptr<GraphSinkQueue>> sinks;
```

### E4. Modify `PipelineSegmentRuntime.h`

Add wrapper before `PipelineSegmentRuntime`:

```cpp
struct RuntimePipelineQueueMsg {
  Sample sample;
  std::size_t edge_index = invalid_edge_index();
};
```

Change `GraphTransport::input_queue` to:

```cpp
std::shared_ptr<simaai::neat::graph::runtime::BlockingQueue<RuntimePipelineQueueMsg>> input_queue;
```

### E5. Modify `EdgeRouter.h/.cpp`

Change callback:

```cpp
std::function<bool(std::size_t, simaai::neat::graph::PortId, Sample&&, std::size_t)>
    dispatch_to_stage_group;
```

In `EdgeRouter::dispatch_to_target()`:

- StageGroup:
  - create `TraceGraphMessageArgs` before moving sample
  - emit `EdgeSrcPush`
  - call callback with `target.edge_index`
  - emit `QueueIn` if success, `Drop` if fail
- PipelineInput:
  - wrap sample in `RuntimePipelineQueueMsg`
  - emit `EdgeSrcPush` before push
  - emit `QueueIn` after successful push, `Drop` on fail
- GraphSink:
  - `push_to_sink()` must receive or be passed the exact `edge_index`
  - push `RuntimeSinkQueueMsg{std::move(sample), edge_index}`
  - same event rules as above

All three paths must use this fast-path pattern so disabled tracing does not copy strings:

```cpp
const bool trace =
    graph_message_trace_enabled(runtime_) && target.edge_index != invalid_edge_index();
TraceGraphMessageArgs trace_args;
if (trace) {
  trace_args = make_trace_graph_message_args(runtime_, target.edge_index, sample);
  trace_graph_message_event(TraceGraphMessageEventType::EdgeSrcPush, trace_args);
}

const bool ok = enqueue_or_dispatch(...);

if (trace) {
  trace_graph_message_event(ok ? TraceGraphMessageEventType::QueueIn
                               : TraceGraphMessageEventType::Drop,
                            trace_args);
}
```

### E6. Modify `RunCore.h`

Change signature:

```cpp
bool graph_dispatch_to_stage_group(std::size_t group_index,
                                   simaai::neat::graph::PortId port,
                                   Sample&& sample,
                                   std::size_t edge_index,
                                   const EdgeRouterOptions& options);
```

### E7. Modify `RunCoreGraphStart.cpp`

In `make_edge_router_callbacks()`:

```cpp
callbacks.dispatch_to_stage_group = [core](std::size_t group_index,
                                           PortId port,
                                           Sample&& sample,
                                           std::size_t edge_index) {
  return core->graph_dispatch_to_stage_group(group_index, port, std::move(sample), edge_index,
                                             core->graph_options.router_options());
};
```

When allocating pipeline input queue:

```cpp
runtime->transport.input_queue =
    std::make_shared<simaai::neat::graph::runtime::BlockingQueue<RuntimePipelineQueueMsg>>(
        core->graph_options.edge_queue);
```

In `build_adjacency_and_sinks()` set edge index in all target initializers:

```cpp
DownstreamTarget{DownstreamTarget::Kind::StageGroup, it_stage->second, e.to_port, eidx}
DownstreamTarget{DownstreamTarget::Kind::PipelineInput, it_pipe->second, e.to_port, eidx}
DownstreamTarget{DownstreamTarget::Kind::GraphSink, static_cast<std::size_t>(e.to), e.to_port, eidx}
```

When adding a sink target:

```cpp
execution.sinks[e.to] =
    std::make_shared<GraphSinkQueue>(core->graph_options.edge_queue);
```

Stage worker pop path:

```cpp
RuntimeStageQueueMsg queued;
if (!st.inbox.pop(queued, core->graph_options.pull_timeout_ms)) {
  st.telemetry.mailbox_pop_miss.fetch_add(1, std::memory_order_relaxed);
  atomic_add_max(st.telemetry.mailbox_pop_wait_ns, st.telemetry.mailbox_pop_wait_max_ns,
                 elapsed_ns_since(pop_start));
  continue;
}
trace_graph_message_event(TraceGraphMessageEventType::QueueOut, &execution,
                          queued.edge_index, queued.sample);
trace_graph_message_event(TraceGraphMessageEventType::EdgeSinkRecv, &execution,
                          queued.edge_index, queued.sample);
StageMsg msg{queued.in_port, std::move(queued.sample)};
```

Pipeline push thread pop path:

```cpp
RuntimePipelineQueueMsg queued;
if (!pipe.transport.input_queue->pop(queued, core->graph_options.pull_timeout_ms)) {
  pipe.transport.telemetry.push_thread_pop_miss.fetch_add(1, std::memory_order_relaxed);
  atomic_add_max(pipe.transport.telemetry.push_thread_pop_wait_ns,
                 pipe.transport.telemetry.push_thread_pop_wait_max_ns,
                 elapsed_ns_since(pop_start));
  continue;
}
trace_graph_message_event(TraceGraphMessageEventType::QueueOut, &execution,
                          queued.edge_index, queued.sample);
trace_graph_message_event(TraceGraphMessageEventType::EdgeSinkRecv, &execution,
                          queued.edge_index, queued.sample);
Sample sample = std::move(queued.sample);
```

`graph_dispatch_to_stage_group()` implementation must push `RuntimeStageQueueMsg` into `st.inbox`.

`RunCore::graph_pull()` must pop `RuntimeSinkQueueMsg`, emit `QueueOut` and `EdgeSinkRecv` using
`queued.edge_index`, then return `std::move(queued.sample)`.

### E8. Modify `RunMeasure.cpp` trace toggles

After LTTng collector starts successfully and only when `include_message_latency=true`:

```cpp
if (core_ && core_->graph_execution_ && opt.include_message_latency &&
    message_source != MetricsTraceSource::Off) {
  auto& ge = *core_->graph_execution_;
  ge.trace_run_id_hash.store(impl->trace_context.run_id_hash, std::memory_order_relaxed);
  ge.trace_graph_id_hash.store(impl->trace_context.graph_id_hash, std::memory_order_relaxed);
  ge.message_trace_enabled.store(true, std::memory_order_release);
}
```

In `MeasureScope::stop()` and destructor cleanup before destroying the trace session:

```cpp
if (impl_->run && impl_->run->core_ && impl_->run->core_->graph_execution_) {
  impl_->run->core_->graph_execution_->message_trace_enabled.store(false,
                                                                   std::memory_order_release);
}
```

If LTTng start fails in Auto mode, leave `message_trace_enabled=false` and set path timing to diagnostics-only/unavailable.

---

## F. LTTng parser and path builder

### F1. Modify `LttngMetricsCollector.h`

Add raw span structs:

```cpp
struct ParsedPluginSpan {
  double start_s = 0.0;
  double end_s = 0.0;
  MeasurePluginLatency metric_identity;
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
  MeasureEdgeLatency metric_identity;
  std::uint32_t start_type = 0;
  std::uint32_t end_type = 1;
  std::uint64_t run_id_hash = 0;
  std::uint64_t graph_id_hash = 0;
  std::string message_id;
  std::string stream_id;
  std::int64_t frame_id = -1;
  std::int64_t input_seq = -1;
  std::int64_t orig_input_seq = -1;
};
```

Add to `LttngParseResult`:

```cpp
std::vector<ParsedPluginSpan> raw_plugin_spans;
std::vector<ParsedEdgeSpan> raw_edge_spans;
```

### F2. Modify `LttngTraceParser.cpp`

Change edge pair key to include span kind:

```cpp
std::string edge_span_kind(const RawEvent& ev) {
  const int type = parse_i32(ev.fields, "event_type", -1);
  if (type == 0 || type == 1) return "edge_transport";
  if (type == 2 || type == 3) return "queue_residence";
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

If `message_id` is `""` or `"0"`, fallback key fields are `orig_input_seq`, then `input_seq`,
then `frame_id`; if those are missing, mark the event unattributed rather than pairing by time.

Use `std::deque` for pending starts:

```cpp
std::map<std::string, std::deque<SpanStart>> plugin_starts;
std::map<std::string, std::deque<EdgeStart>> edge_starts;
```

Then consume with `pop_front()`; do not use `vector.erase(begin())` in high-volume message traces.

Extend parser entrypoint to filter by graph hash as well as run hash:

```cpp
LttngParseResult parse_lttng_trace_text(const std::string& text,
                                        std::uint64_t expected_run_id_hash,
                                        std::uint64_t expected_graph_id_hash,
                                        bool allow_pipeline_fallback);
```

When plugin span pairs successfully:

- push one `ParsedPluginSpan` into `result.raw_plugin_spans`
- then aggregate as today

When edge span pairs successfully:

- canonicalize numeric edge ID through `runtime::canonical_lowered_edge_id()` or equivalent
- push one `ParsedEdgeSpan` into `result.raw_edge_spans`
- then aggregate as today

### F3. Create `src/pipeline/runtime/PathTimingBuilder.h/.cpp`

Header:

```cpp
#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "LttngMetricsCollector.h"
#include "pipeline/Run.h"
#include "pipeline/runtime/ExecutionGraphPlan.h"
#include "pipeline/runtime/RunCore.h"

namespace simaai::neat::runtime {

MeasurePathTiming build_path_timing(const ExecutionGraphPlan& plan,
                                    const pipeline_internal::LttngParseResult& parsed,
                                    const std::vector<GraphSampleTimingEvent>& graph_entries,
                                    const std::vector<GraphSampleTimingEvent>& graph_pulls);

} // namespace simaai::neat::runtime
```

Builder algorithm:

1. If `parsed.raw_edge_spans.empty()` and `parsed.raw_plugin_spans.empty()`, return unavailable with status `unavailable_message_trace_disabled`.
2. Build sample keys from message ID first, fallback to `(stream_id, orig_input_seq/input_seq/frame_id)`.
3. For node arrival:
   - graph entry timestamp from LTTng `GraphEntry` events, not `steady_clock`
   - first matching edge sink receive or first plugin start for the runtime node
   - aggregate by runtime node ID
4. For inter-plugin gap:
   - for each lowered edge `eN`, find upstream plugin span ending before downstream plugin span starts for same sample key
   - compute `downstream.start_s - upstream.end_s`
   - aggregate by lowered edge ID and plugin instance IDs
5. For output tail:
   - public pull timestamp from LTTng `GraphOutputPull` events, not `steady_clock`
   - last matching plugin end or output edge receive timestamp
   - aggregate by output endpoint
6. If ambiguous multiple candidates exist for same key/edge, mark row reliable=false and add warning `AMBIGUOUS_SAMPLE_LINEAGE`.
7. Set `available=true` if any row exists, `status="collected"` if all reliable else `"partial"`, `source="lttng_message_events"`.

`graph_entries` and `graph_pulls` are passed only to expose steady-clock graph E2E diagnostics and
lineage warnings. They must never be subtracted from LTTng timestamps.

### F4. Hook path builder in `RunMeasure.cpp`

Build LTTng options from the **effective** message source, not raw
`MeasureOptions.include_message_latency`:

```cpp
const bool effective_message_lttng =
    opt.include_message_latency &&
    message_source != MetricsTraceSource::Off &&
    trace_source_requests_lttng(message_source);
lttng_opt.enable_message_events = effective_message_lttng;
```

After successful LTTng parse and after edge/plugin metrics are moved into report:

```cpp
if (impl_->options.include_message_latency && impl_->run->core_ &&
    impl_->run->core_->graph_execution_) {
  report.path_timing = runtime::build_path_timing(
      impl_->run->core_->graph_execution_->plan,
      parsed,
      graph_entry_events,
      graph_pull_events);
}
```

If exact message latency is disabled:

```cpp
report.path_timing.available = false;
report.path_timing.status = report.edge_latency.empty()
    ? "unavailable_message_trace_disabled"
    : "diagnostic_aggregate_only";
report.path_timing.source = report.edge_latency.empty() ? "none" : "diagnostics";
report.path_timing.warnings.push_back(
    "Exact path timing requires MeasureOptions.include_message_latency=true.");
```

After parse, do not leave `message_latency_status="collected"` unless actual message spans were
collected:

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

---

## G. Queue telemetry performance gate

Modify `include/graph/runtime/BlockingQueue.h`.

Add enum:

```cpp
enum class QueueTelemetryLevel : std::uint8_t {
  CountersOnly,
  Timing,
};
```

Add member:

```cpp
mutable std::atomic<QueueTelemetryLevel> telemetry_level_{QueueTelemetryLevel::CountersOnly};
```

Add methods:

```cpp
void set_telemetry_level(QueueTelemetryLevel level) noexcept {
  telemetry_level_.store(level, std::memory_order_release);
}

bool timing_enabled() const noexcept {
  return telemetry_level_.load(std::memory_order_acquire) == QueueTelemetryLevel::Timing;
}
```

Guard `steady_clock::now()` calls so counters/high-watermark stay on but timing is disabled by
default. This is a deliberate change from current `BlockingQueue<T>`, which timestamps every
push/pop today.

Add `telemetry_level` or `timing_enabled` to `BlockingQueue<T>::Stats` so `EdgeMetrics.cpp` can
export `status="timing_disabled"` instead of fake zero latency.

`Run::start_measurement()` enables queue timing only if:

```cpp
opt.include_edge_latency || opt.include_message_latency
```

Ordering:

```cpp
set_graph_queue_timing_enabled(*core, opt.include_edge_latency || opt.include_message_latency);
impl->before_graph_queue_metrics = snapshot_graph_queue_latencies(*this);
impl->start = Clock::now();
```

On stop, take the final snapshot before disabling timing:

```cpp
auto after_queue = snapshot_graph_queue_latencies(*impl_->run);
set_graph_queue_timing_enabled(*core, false);
```

---

## H. Visualizer/schema/tests implementation

### H1. Visualizer

Modify `tools/visualize_graph_run.py`.

CLI choices:

```python
parser.add_argument("--view", choices=("auto", "customer", "public", "lowered"), default="auto")
parser.add_argument("--svg", type=Path)
parser.add_argument("--topology-json", type=Path)
parser.add_argument("--summary", action="store_true")
parser.add_argument("--summary-json", type=Path)
parser.add_argument("--validate", action="store_true")
parser.add_argument("--strict", action="store_true")
parser.add_argument("--redact", choices=("none", "customer"), default="none")
parser.add_argument("--fail-on-fallback", action="store_true")
parser.add_argument("--fail-on-unattributed", action="store_true")
```

Replace `_edge_metric_lookup()` with ID-based lookup:

1. map `run.edge_metrics[*].lowered_edge_id`
2. fallback to `edge_id` only when it matches `^e[0-9]+$`
3. customer edge label uses `edge.lowered_edge_ids`
4. metrics map values are **lists**, not single rows, because multi-stream can have multiple rows
   per lowered edge
5. if multiple lowered metrics, use max avg and show row/stream counts, never sum

Concrete lookup:

```python
edge_metrics_by_lowered_id: dict[str, list[Mapping[str, Any]]] = {}
for metric in _as_list(run.get("edge_metrics")):
    eid = metric.get("lowered_edge_id") or metric.get("edge_id")
    eid = _canonical_lowered_edge_id(eid)
    if eid:
        edge_metrics_by_lowered_id.setdefault(eid, []).append(metric)
```

Customer label:

```python
metrics = []
for eid in edge.get("lowered_edge_ids", []):
    metrics.extend(edge_metrics_by_lowered_id.get(eid, []))
if not metrics:
    return "handoff N/A" if edge.get("mapping_status") != "mapped" else ""
values = [avg_ms(m) for m in metrics if avg_ms(m) is not None]
streams = sorted({m.get("stream_id") for m in metrics if m.get("stream_id")})
return f"handoff max avg {fmt(max(values))} / {len(metrics)} rows / {len(streams)} streams"
```

Add DAG layout function `_layout_dag(nodes, edges)` with ranks from inputs to outputs. Customer `--view auto` should prefer `graph.customer_view`, then public, then lowered.

Timeline overlay priority:

1. `path_timing.inter_plugin_gap_ms` label `gap avg ...`
2. else diagnostic edge label `queue avg ...` or `transport avg ...`
3. never call diagnostic edge label `gap`

Redaction must happen before selecting topology/overlay and must apply to HTML, SVG,
`--topology-json`, and raw JSON:

```python
render_payload = _redact_payload(payload, args.redact)
selected = _select_view(render_payload, args.view)
overlay = _metric_overlay(render_payload, selected)
```

Use deep redaction for keys/patterns:

```python
PRIVATE_KEYS = {"argv", "hostname", "pid", "uri", "source_path", "path"}
PRIVATE_PATTERNS = ("/workspace", "/home/", "rtsp://", "file://", "udp://")
```

Latency cells must respect availability flags:

```python
def _latency_cell(latency, key, availability_key=None):
    if latency.get("samples", 0) == 0:
        return "N/A"
    if availability_key and latency.get(availability_key) is False:
        return "N/A"
    return _fmt(latency.get(key), " ms")
```

### H2. Schema

Modify `schemas/graph_run_v1.schema.json`:

- add `graph.customer_view`
- add `run.path_timing`
- add `lowered_edge_id` to edge metrics
- add `percentiles_available`, `min_max_available`, `max_semantics` to edge latency summary

Modify `tests/perf/tools/graph_run_schema.py` strict checks:

Add concrete API/CLI:

```python
def validate_graph_run(data, *, strict=False, customer=False, allow_private=False) -> None:
    validate_graph_run_shape(data)
    if strict:
        validate_referential_integrity(data)
    if customer:
        validate_customer_gate(data, allow_private=allow_private)
```

CLI:

```python
parser.add_argument("input", type=Path)
parser.add_argument("--strict", action="store_true")
parser.add_argument("--customer", action="store_true")
parser.add_argument("--allow-private", action="store_true")
```

- customer edge endpoints must exist
- customer mappings must reference existing IDs
- numeric-only exported edge IDs fail strict validation
- path timing rows must reference customer nodes/edges when customer view exists
- no power object under node/plugin/edge metrics
- fallback customer topology fails with `--customer` unless explicitly allowed

### H3. Tests

Add/extend:

```text
tests/graph_migration/unified/phaseA4_run_export_test.cpp
tests/graph_migration/unified/phaseA4_branched_report_demo.cpp
tests/perf/tools/test_graph_run_schema.py
tests/perf/tools/test_graph_run_visualizer.py
tests/graph_migration/unified/phaseA4_metrics_overhead_smoke_test.cpp
```

Required assertions:

- `graph.customer_view` exists
- Branch graph has visible `Branch` nodes and no invalid public pass-through nodes in customer view
- measured JSON has `run.measured_stats` and `run.graph_metrics.window`
- E2E unavailable is explicit, not zero
- edge percentiles unavailable render as `N/A`
- exact path timing with synthetic LTTng produces inter-plugin gap row
- diagnostic-only run has `path_timing.status="diagnostic_aggregate_only"`
- visualizer `--view customer --strict --fail-on-fallback` fails fallback topology
- default metrics overhead target <= 2% on representative graph
- add `tests/graph_migration/unified/phaseA4_customer_view_combine_export_test.cpp`
- Combine/JoinBundle assertions: 2 inputs, 1 Combine, 1 output, Combine maps to JoinBundle,
  output pulls equal emitted bundles/frames, not input pushes
- visualizer layout tests assert semantic rank ordering, not exact pixel coordinates
- redaction tests verify HTML, SVG, topology JSON, and raw JSON

---

## I. Validation order after implementation

Do this only after A-H are implemented.

1. Reconfigure build so new `.cpp` files from `file(GLOB_RECURSE SIMANEAT_ALL_SOURCES ...)` are picked up.
2. Build library and unit tests.
3. Before executing any built binary/test, run `file <path>`.
4. If binary is aarch64, run through devkit path, not locally:

```bash
dk /home/docker/sima-cli/tmp/devkit_env_exec.sh <binary> [args...]
```

or repo wrapper:

```bash
tools/ctest-on-devkit <regex> --output-on-failure
```

5. Run schema/visualizer Python tests locally.
6. Generate DETR report with:

```cpp
MeasureOptions m;
m.include_plugin_latency = true;
m.include_edge_latency = true;
m.include_message_latency = false;
```

Expected DETR:

- graph-level throughput card is output-pulls/s
- power card says disabled/not configured/enabled-no-samples on DVT, not failure
- node/plugin latencies visible for QuantTess/Detess/Postproc where attribution exists
- `path_timing.status` is diagnostic-only or unavailable because exact message tracing is off

7. Generate exact trace run with:

```cpp
m.include_message_latency = true;
```

Expected exact run:

- `run.path_timing.available=true` or `partial`
- inter-plugin gap rows exist where plugin spans and edge IDs line up
- no bogus global/process attribution in multi-stream; rows separate by `stream_id` and `plugin_instance_id`
