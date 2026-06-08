# LTTng Graph Metrics Implementation Plan

## Status and intent

This is the revised implementation plan after sub-agent review. The plan is intentionally focused on a small, maintainable implementation:

1. **Graph-level throughput and power** stay graph-level averages.
2. **Execution latency** is reported per Node/Plugin.
3. **Inter-plugin/message latency** is reported separately as edge/queue/handoff diagnostics.

The customer report must not mix these semantics. Plugin latency is execution time. Edge/message latency is handoff/queue/transport time. Throughput and power remain graph-level aggregates.

The implementation moves the customer-facing plugin latency path away from the legacy process-global NEAT profiler and toward LTTng-UST, while keeping the codebase small by:

- keeping LTTng internals private to runtime code;
- extracting one shared attribution helper instead of duplicating mapping logic;
- using existing pad/queue diagnostics first for edge latency;
- making exact per-message LTTng edge tracing opt-in/targeted because it can be high volume;
- not using legacy `transmit=true` KPI bus messages for metrics collection;
- deleting or compile-guarding legacy profiler paths after LTTng parity is validated.

## External references checked

- LTTng docs: https://lttng.org/docs/v2.15/
- LTTng-UST man page: https://lttng.org/man/3/lttng-ust/v2.15/
- LTTng channel command: https://lttng.org/man/1/lttng-enable-channel/v2.15/
- LTTng tracking command: https://lttng.org/man/1/lttng-track/v2.15/
- Babeltrace2: https://babeltrace.org/docs/v2.0/man1/babeltrace2-convert.1/

## Local facts driving the plan

DevKit has the required runtime tooling:

```bash
command -v lttng       # /usr/bin/lttng
command -v babeltrace2 # /usr/bin/babeltrace2
ls /usr/lib/aarch64-linux-gnu/libsimaaitrace.so
```

Existing tracepoint headers are in:

```text
core_graph_changes/deps/headers/usr/include/simaai/trace/pipeline_new_tp.h
core_graph_changes/deps/headers/usr/include/simaai/trace/pipeline_tp.h
core_graph_changes/deps/headers/usr/include/simaai/trace/remote_core_tp.h
core_graph_changes/deps/headers/usr/include/simaai/trace/common_macros.h
```

Existing providers exported by `libsimaaitrace.so` include:

```text
pipeline:A65
pipeline:CVU
pipeline:MLA
pipeline:AllegroDecoder
pipeline:AllegroEncoder
pipeline:PCIeSrc
pipeline:PCIeSink
remote_core:EV74
remote_core:MLA
simaaipipeline:plugin_trace
```

Existing adopters include:

```text
internals/gst_plugins/processmla/gstneatprocessmla.cpp
internals/gst_plugins/processcvu/gstneatprocesscvu.cpp
internals/gst_plugins/templates/aggregator/agg_template.h
internals/gst_plugins/*decoder* / *encoder* / pcie / A65 postprocess plugins
```

Current `pipeline:*` payload is useful but insufficient for robust attribution:

```text
event_type, frame_id, plugin_id, stream_id
```

It has no root run identity, graph id, node id, edge id, or plugin-instance id. Therefore, current `pipeline:*` can be a diagnostic/single-run fallback, but **v2 identity is required for production-quality multi-stream/multi-run attribution**.

Current custom profiler limitations:

- `LatencyProfiler` is process-global.
- `profiler_events.cpp` uses a process-global ring.
- The C ABI lacks stable run/node identity and a size/version field.
- Single-stream tests produced JSON/HTML but aborted in teardown with heap corruption.

Decision:

- Customer metrics use LTTng.
- Legacy profiler/KPI paths are not customer paths.
- Legacy code is removed or compile-guarded after LTTng parity.

## Requirements

### Customer semantics

| Metric class | Scope | Meaning | Additive? | Report location |
|---|---:|---|---:|---|
| Throughput | Graph | Average outputs/inferences per second over the measured window | No | Top summary card |
| Power | Graph | Average rail power over the measured window; DVT may be unavailable/untrusted | No | Top summary card |
| Node latency | Node | Runtime node latency summary | No | Node cards/table |
| Plugin latency | Plugin inside node | Execution span inside one plugin/kernel | No | Plugin rows under node or unattributed diagnostics |
| Edge/message latency | Edge between plugins/nodes | Queue/handoff/transport time between upstream output and downstream input | No | Separate edge table/graph edge labels |

Do not add edge/message latency to plugin latency or graph latency.

### Functional requirements

1. Plugin attribution must distinguish plugin instances across streams, graph segments, and concurrent Runs.
2. Ambiguous rows must be kept in `*_unattributed` with reason codes; never guess by nearest timestamp.
3. Existing JSON remains backward compatible.
4. Edge/message timing must be possible without turning every plugin into a large instrumentation project.
5. Metrics collection must not require power to work. DVT power can report `unavailable` while SOM power remains implemented.

### Performance requirements

1. Disabled tracepoints remain near-zero overhead via `tracepoint_enabled`/`tracepoint_if_enabled` and a separate `trace-enabled` property.
2. Plugin span collection targets less than 3-5% throughput impact versus graph metrics only.
3. Exact edge/message tracing is opt-in or sampled because it can multiply event volume.
4. Parse CTF traces after the measured window, never during it.

## Architecture overview

### Minimal file layout

Public API changes only in existing public headers:

```text
include/pipeline/Run.h
include/pipeline/RunExport.h           # only if export options need one extra toggle
```

Private runtime implementation:

```text
src/pipeline/runtime/LttngMetricsCollector.h
src/pipeline/runtime/LttngMetricsCollector.cpp
src/pipeline/runtime/LttngTraceSession.cpp
src/pipeline/runtime/LttngTraceParser.cpp
src/pipeline/runtime/TraceAttribution.h
src/pipeline/runtime/TraceAttribution.cpp
src/pipeline/runtime/EdgeMetrics.cpp
src/pipeline/gst/TraceIdentity.h
src/pipeline/gst/TraceIdentity.cpp
```

Internals trace provider and plugin helpers:

```text
internals/.../trace/sima_neat_metrics_tp.h       # exact location depends on trace package layout
internals/.../trace/sima_neat_metrics_provider.c
internals/gst_plugins/common/TraceIdentityProps.h # shared property install/set/get helper
```

Avoid adding a public `include/pipeline/Lttng*.h`. Unit tests can include private headers directly.

### One collector facade

Use one runtime facade instead of scattered LTTng calls:

```cpp
class LttngMetricsCollector {
public:
  LttngMetricsCollector(const InternalMetricsTraceOptions& opt,
                        TraceIdentityContext trace_context,
                        GraphTopologyForAttribution topology);
  bool available(std::string* reason) const;
  bool start(std::string* err);
  bool stop_and_destroy(std::string* err);
  LttngParseResult parse(std::string* err) const;
  void cleanup_noexcept();
};
```

`RunMeasure.cpp` owns this through `MeasureScope::Impl` and does not know LTTng command details.

## Public API and app config

### Keep public API small

File: `include/pipeline/Run.h`

Add a small enum:

```cpp
enum class MetricsTraceSource {
  Auto,  // Prefer LTTng. If unavailable, report unavailable; do not use legacy profiler.
  Off,
  Lttng // Require LTTng; fail clearly if unavailable.
};
```

Modify `MeasureOptions`:

```cpp
struct MeasureOptions {
  int duration_ms = 10000;
  int warmup_ms = 1000;
  int timeout_ms = 5000;

  // Plugin execution spans. Auto means LTTng when available, otherwise unavailable.
  bool include_plugin_latency = true;
  MetricsTraceSource plugin_latency_source = MetricsTraceSource::Auto;

  // Low-overhead inter-plugin / queue / handoff summary. Uses existing diagnostics by default.
  bool include_edge_latency = true;

  // Exact per-message edge tracing. High-volume; off unless requested by app/support.
  bool include_message_latency = false;
  MetricsTraceSource message_latency_source = MetricsTraceSource::Auto;

  // Trace retention is the only user-facing LTTng knob.
  bool retain_metrics_trace = false;
  std::string metrics_trace_dir; // empty => private mkdtemp under report dir or /tmp

  bool include_power = true;
  std::string title = "NEAT measurement";
  std::string model;
  std::string input;
  std::string placement;
  int logical_batch_size = 1;
};
```

Do not expose provider toggles, contexts, subbuffer sizes, or remote-core selection publicly. Put those under internal options and debug env vars.

### Add report fields

```cpp
struct MeasureEdgeLatency {
  std::string edge_id;
  std::string name;
  std::string from_node_id;
  std::string to_node_id;
  std::int32_t from_runtime_node_id = -1;
  std::int32_t to_runtime_node_id = -1;
  std::string from_element_name;
  std::string to_element_name;
  std::string from_plugin_instance_id;
  std::string to_plugin_instance_id;
  std::string stream_id;

  std::uint64_t samples = 0;
  double total_ms = 0.0;
  double avg_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
  double p50_ms = 0.0;
  double p95_ms = 0.0;

  std::string source;             // diagnostics | lttng
  std::string timing_semantics;   // queue_residence | edge_transport | inter_plugin_gap | pad_wait
  std::string attribution_source; // graph_edge_identity | element_link | unattributed
  std::string mapping_error;
  bool non_additive = true;
  bool reliable = true;
};
```

Extend existing `MeasurePluginLatency`:

```cpp
std::string stream_id;
std::string plugin_instance_id;
std::string source;             // lttng
std::string attribution_source; // lttng_v2_identity | lttng_element_name | unattributed
std::string mapping_error;
bool reliable = true;
```

Extend `MeasureReport`:

```cpp
std::string plugin_latency_status;   // off | collected | unavailable | failed
std::string plugin_latency_source;   // lttng | none
std::string message_latency_status;  // off | collected | unavailable | failed
std::string message_latency_source;  // diagnostics | lttng | none
std::string metrics_trace_dir;       // set only if retained
std::vector<std::string> warnings;
std::vector<MeasureEdgeLatency> edge_latency;
std::vector<MeasurePluginLatency> plugin_latency_unattributed;
std::vector<MeasureEdgeLatency> edge_latency_unattributed;
```

### App config

Use one simple schema in apps:

```yaml
metrics:
  enable: true
  include_plugin_latency: true
  plugin_latency_source: auto     # auto | off | lttng
  include_edge_latency: true      # low-overhead edge/queue summary
  include_message_latency: false  # exact LTTng per-message edge timing; high-volume
  message_latency_source: auto    # auto | off | lttng
  retain_metrics_trace: false
  metrics_trace_dir: ""
  include_power: true
```

App parser helper:

```cpp
MetricsTraceSource parse_metrics_trace_source(std::string s) {
  s = lower_copy(trim_copy(s));
  if (s == "auto") return MetricsTraceSource::Auto;
  if (s == "off") return MetricsTraceSource::Off;
  if (s == "lttng") return MetricsTraceSource::Lttng;
  throw std::runtime_error("metrics trace source must be one of [auto, off, lttng]");
}
```

No public `legacy_profiler` option. If a developer still needs it temporarily, gate it with an internal build flag/env var, not customer config.

## LTTng session lifecycle

### Correct command sequence

Every command must pass `--session=<name>`. Do not rely on LTTng's global current session.

For PID `${PID}`, session `${SESSION}`, output `${TRACE_DIR}`:

```bash
lttng create "$SESSION" --output="$TRACE_DIR"
lttng enable-channel --userspace --session="$SESSION" \
  --discard --blocking-timeout=0 --subbuf-size=1M --num-subbuf=8 sima-neat-metrics
lttng add-context --userspace --session="$SESSION" --channel=sima-neat-metrics \
  --type=vpid --type=vtid
lttng track --userspace --session="$SESSION" --vpid="$PID"
lttng enable-event --userspace --session="$SESSION" --channel=sima-neat-metrics 'sima_neat_plugin:*'
# Compatibility/single-run diagnostic fallback only:
lttng enable-event --userspace --session="$SESSION" --channel=sima-neat-metrics 'pipeline:*'
# Debug only, disabled by default until correlation exists:
# lttng enable-event --userspace --session="$SESSION" --channel=sima-neat-metrics 'remote_core:*'
lttng start "$SESSION"
# measured window
lttng stop "$SESSION"
lttng destroy "$SESSION"
babeltrace2 --color=never --no-delta --clock-seconds --names=context,payload "$TRACE_DIR"
```

When using `execvp`, arguments are literal; do not include shell quotes in `argv`.

Cleanup rules:

- `stop -> destroy -> parse`.
- Never use `--no-wait`.
- `destroy` flushes/frees the session while keeping trace files.
- If `retain_metrics_trace=false`, delete only the private `mkdtemp` trace subdir created by the collector. Never delete a user-provided parent directory.
- On parse failure, retain the trace if possible and add a warning with the path.

### Internal options

Private only:

```cpp
struct InternalMetricsTraceOptions {
  std::string session_name;
  std::string trace_dir;
  bool retain_trace = false;
  bool enable_pipeline_fallback = true;
  bool enable_remote_core_debug = false;
  bool enable_message_events = false;
  bool require_v2_identity_for_node_attribution = true;
  bool allow_element_name_fallback = true;
  int subbuf_size_kb = 1024;
  int num_subbuf = 8;
  bool add_procname_context = false; // off by default for overhead
};
```

Debug env vars:

```text
SIMA_NEAT_METRICS_TRACE_SOURCE=auto|off|lttng
SIMA_NEAT_METRICS_RETAIN_TRACE=1
SIMA_NEAT_METRICS_TRACE_DIR=/tmp/sima-neat-traces
SIMA_NEAT_LTTNG_SUBBUF_SIZE_KB=1024
SIMA_NEAT_LTTNG_NUM_SUBBUF=8
SIMA_NEAT_LTTNG_ENABLE_REMOTE_CORE=0
SIMA_NEAT_LTTNG_ENABLE_PIPELINE_FALLBACK=1
```

### Overlapping measurements

Phase 1 `pipeline:*` fallback cannot isolate two overlapping `MeasureScope`s in the same process because VPID tracking is process-wide and `pipeline:*` has no run identity.

Rules:

- If only fallback events are available and an overlapping measurement is detected, plugin attribution is disabled with a warning in `auto`; `lttng` mode fails clearly.
- V2 events with `run_id_hash` are filtered to the measured root Run.
- Future process-wide collector can multiplex overlapping Runs by `run_id_hash`, but the first implementation can use one active LTTng collector guard per process.

## Trace identity and plugin instrumentation

### Root trace identity

Use the root graph/run id, not segment-local run ids. Add:

```cpp
struct TraceIdentityContext {
  std::uint64_t run_id_hash = 0;   // stable hash of root Run/GraphRun id
  std::uint64_t graph_id_hash = 0; // stable hash of public graph id/version/topology
  std::uint32_t pipeline_segment_id = UINT32_MAX;
};
```

Add to `RunCoreStartOptions`:

```cpp
TraceIdentityContext trace_identity;
```

When graph runtime starts segment pipelines:

```cpp
start_opt.trace_identity.run_id_hash = core->run_id_hash;
start_opt.trace_identity.graph_id_hash = stable_graph_id_hash(execution.plan);
start_opt.trace_identity.pipeline_segment_id = pipe.seg.id;
```

Use FNV-1a 64-bit or another fixed hash. Do not use `std::hash`.

### Element trace identity

```cpp
struct ElementTraceIdentity {
  std::string element_name;
  std::uint64_t run_id_hash = 0;
  std::uint64_t graph_id_hash = 0;
  std::uint32_t pipeline_segment_id = UINT32_MAX;
  std::int32_t runtime_node_id = -1;
  std::int32_t public_node_id = -1;
  std::string plugin_instance_id;
};
```

`plugin_instance_id` must be deterministic and unique within a root run:

```text
r<run_hash>.g<graph_hash>.s<segment>.n<runtime_node>.e<element_name>
```

Core injection helper:

```cpp
void apply_trace_identity(GstElement* pipeline,
                          const std::vector<ElementTraceIdentity>& identities,
                          bool enable_trace) {
  for (const auto& id : identities) {
    GstElement* element = gst_bin_get_by_name(GST_BIN(pipeline), id.element_name.c_str());
    if (!element) continue;
    GObjectClass* klass = G_OBJECT_GET_CLASS(element);

    auto set_if = [&](const char* prop, auto value) {
      if (g_object_class_find_property(klass, prop)) {
        g_object_set(G_OBJECT(element), prop, value, nullptr);
      }
    };

    set_if("trace-enabled", enable_trace ? TRUE : FALSE);
    set_if("trace-run-id-hash", static_cast<guint64>(id.run_id_hash));
    set_if("trace-graph-id-hash", static_cast<guint64>(id.graph_id_hash));
    set_if("trace-segment-id", static_cast<guint>(id.pipeline_segment_id));
    set_if("trace-runtime-node-id", static_cast<gint>(id.runtime_node_id));
    set_if("trace-public-node-id", static_cast<gint>(id.public_node_id));
    set_if("trace-plugin-instance-id", id.plugin_instance_id.c_str());

    gst_object_unref(element);
  }
}
```

Call after pipeline construction and before `PLAYING`/push.

### Do not use `transmit=true` for LTTng

Existing plugins often gate current tracepoints and KPI bus messages with `self->transmit`. Metrics must not set `transmit=true`, because that revives legacy KPI messages and side effects.

Add a separate GObject property to instrumented plugins:

```cpp
gboolean trace_enabled = FALSE;
```

Property install:

```cpp
g_object_class_install_property(
    gobject_class,
    PROP_TRACE_ENABLED,
    g_param_spec_boolean("trace-enabled", "Trace enabled",
                         "Enable LTTng metrics tracepoints only; does not emit KPI bus messages",
                         FALSE,
                         static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
```

Usage pattern:

```cpp
const bool lttng_trace = self->priv->trace_enabled;
const bool legacy_kpi = self->transmit;

if (lttng_trace) {
  sima_neat_plugin_span_args a = make_span_args(...);
  tracepoint_sima_neat_plugin_span_start(&a);
  // Optionally also emit current pipeline:* compatibility event while Phase 1 parser exists.
}

if (legacy_kpi) {
  // Existing gst_message_new_application("kpi", ...) behavior only.
}
```

Migration rule:

```text
transmit=true       => legacy bus KPI messages only
trace-enabled=true  => LTTng metrics tracepoints only
```

## V2 tracepoint provider

Use one provider package, preferably in `libsimaaitrace.so`, with unique provider names:

```text
sima_neat_plugin:plugin_span
sima_neat_edge:message
sima_neat_metrics:window
sima_neat_metrics:node_def
sima_neat_metrics:edge_def
```

Do not statically embed provider definitions into dlopen-able GStreamer plugins.

### Provider API style

Use the same LTTng-UST API style as the existing trace package, or migrate the full package consistently. Do not mix incompatible styles accidentally.

The current headers use compatibility `TRACEPOINT_EVENT`/`ctf_*` macros. If retained, require compatibility API 0 in the trace provider build. If migrated, migrate to `LTTNG_UST_TRACEPOINT_EVENT` / `lttng_ust_field_*` across the new provider.

### Struct-pointer payload to avoid TP_ARGS limits

Do not use a long list of `TP_ARGS`. Use a struct pointer.

```c
struct sima_neat_plugin_span_args {
  uint32_t event_type;      // START=0, END=1
  uint64_t run_id_hash;
  uint64_t graph_id_hash;
  uint32_t pipeline_segment_id;
  int32_t runtime_node_id;
  int32_t public_node_id;
  const char* plugin_instance_id;
  const char* element_name;
  const char* stream_id;
  const char* backend;
  const char* phase;
  const char* kernel_name;
  uint64_t frame_id;
  uint64_t request_id;
  uint64_t message_id;
};
```

Tracepoint snippet:

```c
TRACEPOINT_EVENT(
  sima_neat_plugin,
  plugin_span,
  TP_ARGS(const struct sima_neat_plugin_span_args*, a),
  TP_FIELDS(
    ctf_integer(uint32_t, event_type, a->event_type)
    ctf_integer(uint64_t, run_id_hash, a->run_id_hash)
    ctf_integer(uint64_t, graph_id_hash, a->graph_id_hash)
    ctf_integer(uint32_t, pipeline_segment_id, a->pipeline_segment_id)
    ctf_integer(int32_t, runtime_node_id, a->runtime_node_id)
    ctf_integer(int32_t, public_node_id, a->public_node_id)
    ctf_string(plugin_instance_id, a->plugin_instance_id ? a->plugin_instance_id : "")
    ctf_string(element_name, a->element_name ? a->element_name : "")
    ctf_string(stream_id, a->stream_id ? a->stream_id : "")
    ctf_string(backend, a->backend ? a->backend : "")
    ctf_string(phase, a->phase ? a->phase : "")
    ctf_string(kernel_name, a->kernel_name ? a->kernel_name : "")
    ctf_integer(uint64_t, frame_id, a->frame_id)
    ctf_integer_hex(uint64_t, request_id, a->request_id)
    ctf_integer(uint64_t, message_id, a->message_id)
  )
)
```

Performance rule: cache element/plugin names and strings outside hot paths where possible. In high-rate traces, prefer numeric IDs plus one `node_def`/`edge_def` metadata event per run.

## LTTng parser

DevKit lacks Python `bt2`, so phase 1 parser uses `babeltrace2` CLI text output.

Command:

```bash
babeltrace2 --color=never --no-delta --clock-seconds --names=context,payload "$TRACE_DIR"
```

Parser responsibilities:

```cpp
struct LttngRawEvent {
  double timestamp_s = 0.0;
  std::string provider;
  std::string event_name;
  std::uint64_t vpid = 0;
  std::uint64_t vtid = 0;
  std::map<std::string, std::string> fields;
};

struct LttngParseResult {
  bool parsed = false;
  bool trace_loss_detected = false;
  std::vector<MeasurePluginLatency> plugin_metrics;
  std::vector<MeasurePluginLatency> plugin_metrics_unattributed;
  std::vector<MeasureEdgeLatency> edge_metrics;
  std::vector<MeasureEdgeLatency> edge_metrics_unattributed;
  std::vector<std::string> warnings;
};
```

Parsing helpers:

```cpp
std::optional<double> parse_timestamp_s(std::string_view line);
std::optional<std::pair<std::string, std::string>> parse_provider_event(std::string_view line);
std::map<std::string, std::string> parse_context_and_payload_fields(std::string_view line);
```

Rules:

- Parse by field names, not by column positions.
- Tolerate formatting variations and enum renderings.
- Do not include `vtid` in pairing keys for queue/message latency; message events can cross threads.
- Detect lost events/discarded packets if visible in Babeltrace output and mark affected metrics unreliable.
- For current `remote_core:*`, do not use LTTng event timestamp for kernel duration; use `custom_timestamp` only after units are validated. Keep remote-core disabled by default until request correlation is implemented.

### Plugin span pairing

V2 key:

```text
run_id_hash + graph_id_hash + pipeline_segment_id + runtime_node_id
+ plugin_instance_id + backend + phase + kernel_name + stream_id
+ frame_id + request_id + message_id
```

Fallback `pipeline:*` key:

```text
vpid + provider + event_name + plugin_id + stream_id + frame_id
```

Fallback is marked `attribution_source=lttng_element_name` or unattributed. It is not robust for same-process concurrent Runs.

Filtering:

```cpp
if (span.run_id_hash != 0 && span.run_id_hash != expected_root_run_id_hash) {
  drop_with_warning("span belongs to a different Run");
}
```

Aggregation key must include `plugin_instance_id`:

```text
backend + phase + kernel_name + plugin_instance_id + element_name + stream_id
+ run_id_hash + graph_id_hash + pipeline_segment_id + runtime_node_id + public_node_id
```

FIFO-pair duplicate starts only when the key is exact. Otherwise send to unattributed diagnostics.

## Shared attribution helper

Avoid third-copy mapping logic. Add:

```text
src/pipeline/runtime/TraceAttribution.h
src/pipeline/runtime/TraceAttribution.cpp
```

Core structs:

```cpp
struct PluginAttributionResult {
  const GraphNodeMetrics* node = nullptr;
  std::string attribution_source; // lttng_v2_identity | lttng_element_name | unattributed
  std::string mapping_error;
};

PluginAttributionResult attribute_plugin_latency(
    const MeasurePluginLatency& plugin,
    const std::vector<GraphNodeMetrics>& nodes);
```

Rules:

1. V2 exact identity wins:

```cpp
node.pipeline_segment_id == plugin.pipeline_segment_id &&
node.runtime_node_id == plugin.runtime_node_id
```

2. If public id exists, validate it is compatible with `node.public_node_ids`.
3. Element-name fallback only when exactly one node owns the element.
4. Ambiguous/missing matches go to `plugin_latency_unattributed` with `mapping_error`.
5. Set `attribution_source` at match time so export never loses provenance.

Use this helper from `RunMeasure.cpp` and `GraphRunExport.cpp`.

## Inter-plugin / message / edge timing

This is a separate metric class, not a plugin row.

### Customer definition

Edge/message latency measures the time for a sample/buffer/message to move from an upstream plugin/node output to a downstream input. It may include queueing, scheduling, backpressure, and transport overhead. It is diagnostic and non-additive; it is not graph throughput, not power, and not plugin execution time.

### Phase A: low-overhead existing diagnostics

Use existing telemetry first:

- `RunDiagSnapshot::element_pad_timings` for pad queue-wait/inter-arrival diagnostics.
- `BoundaryFlowStats` for graph boundary flow and last in/out wall times.
- `BlockingQueue::Stats` for graph actor mailbox/input/sink wait/backpressure.

Extend `BlockingQueue::Stats` minimally to measure queue residence:

```cpp
struct Stats {
  ...
  std::uint64_t residence_count = 0;
  std::uint64_t residence_ns = 0;
  std::uint64_t max_residence_ns = 0;
};
```

Implementation detail for `BlockingQueue<T>`:

```cpp
struct QueueEntry {
  T value;
  std::chrono::steady_clock::time_point enqueue_time;
};
std::deque<QueueEntry> queue_;
```

On push, store `enqueue_time = steady_clock::now()`. On pop, compute residence and update atomics. This changes only the queue internals, not `Sample` or public queue API.

Convert these into `MeasureEdgeLatency` rows with `source="diagnostics"` and timing semantics:

```text
queue_residence     # BlockingQueue enqueue -> dequeue
pad_wait            # GStreamer pad queue wait summary
inter_arrival       # pad inter-arrival timing
boundary_flow       # boundary in/out timestamps, if reliable
```

### Phase B: exact LTTng message/edge tracing

Only enable when `include_message_latency=true`. Install core-owned probes on selected graph edges/queues; do not instrument every pad by default.

Preferred event:

```c
struct sima_neat_edge_message_args {
  uint32_t event_type; // EDGE_SRC_PUSH=0, EDGE_SINK_RECV=1, QUEUE_IN=2, QUEUE_OUT=3, DROP=4
  uint64_t run_id_hash;
  uint64_t graph_id_hash;
  uint64_t message_id;
  uint32_t pipeline_segment_id;
  int32_t edge_id;
  int32_t src_runtime_node_id;
  int32_t dst_runtime_node_id;
  const char* src_plugin_instance_id;
  const char* dst_plugin_instance_id;
  const char* src_element;
  const char* dst_element;
  const char* src_pad;
  const char* dst_pad;
  const char* stream_id;
  int64_t frame_id;
  int64_t input_seq;
  int64_t orig_input_seq;
  uint64_t pts_ns;
  uint64_t bytes;
  uint64_t buffer_addr; // fallback diagnostic only
};
```

Pairing key:

```text
run_id_hash + graph_id_hash + pipeline_segment_id + edge_id + message_id
```

Fallback key only if `message_id` is absent:

```text
run_id_hash + stream_id + frame_id + orig_input_seq + edge_id
```

If both `message_id` and `edge_id` are missing, mark unattributed. Never infer by nearest timestamp.

Metrics computed:

```text
queue_residence_ms  = QUEUE_OUT - QUEUE_IN
edge_transport_ms   = EDGE_SINK_RECV - EDGE_SRC_PUSH
inter_plugin_gap_ms = downstream plugin START - upstream plugin END
```

Use LTTng event timestamps for wall-clock latency. Use Gst PTS/DTS only for correlation/debug, never latency math.

### Message identity

Preferred: add a stable `trace_sample_id` / `message_id` at source and propagate through transforms.

```cpp
struct Sample {
  ...
  std::uint64_t trace_sample_id = 0; // zero means absent/unassigned
};
```

When converting to GStreamer buffers, write metadata:

```cpp
gst_structure_set(s,
  "trace-run-id-hash", G_TYPE_UINT64, run_id_hash,
  "trace-sample-id", G_TYPE_UINT64, sample.trace_sample_id,
  nullptr);
```

If buffers are copied/batched/split and metadata is lost, rows become unattributed or best-effort; do not guess.

### Edge report JSON

Separate top-level sections:

```json
"edge_message_latency": {
  "requested": true,
  "status": "collected",
  "source": "diagnostics",
  "warnings": []
},
"edge_metrics": [
  {
    "edge_id": "s0:n2->n3",
    "from_node": "p2",
    "to_node": "p3",
    "stream_id": "stream0",
    "samples": 120,
    "latency_ms": {"avg": 0.42, "p50": 0.31, "p95": 0.91, "max": 1.7},
    "timing_semantics": "queue_residence",
    "source": "diagnostics",
    "attribution_source": "graph_edge_identity",
    "non_additive": true
  }
],
"edge_metrics_unattributed": []
```

Reason codes:

```text
MISSING_MESSAGE_ID
EDGE_MATCHED_MULTIPLE_LINKS
EDGE_NOT_IN_GRAPH_TOPOLOGY
UNMATCHED_EDGE_EMIT
UNMATCHED_EDGE_RECEIVE
TRACE_LOSS_DETECTED
NEGATIVE_LATENCY
```

## RunMeasure integration

Add to `MeasureScope::Impl`:

```cpp
std::unique_ptr<pipeline_internal::LttngMetricsCollector> lttng_metrics;
std::optional<RunDiagSnapshot> before_diag;
std::optional<GraphMetricsReport> before_graph_metrics;
```

Start order:

```cpp
impl->before = stats();
impl->before_diag = diag_snapshot();
impl->before_graph_metrics = build_graph_metrics_report_run_lifetime(...);

const bool need_lttng = resolve_lttng_needed(options);
if (need_lttng) {
  impl->lttng_metrics = std::make_unique<LttngMetricsCollector>(...);
  std::string err;
  if (!impl->lttng_metrics->start(&err)) {
    if (options.plugin_latency_source == MetricsTraceSource::Lttng ||
        options.message_latency_source == MetricsTraceSource::Lttng) {
      throw std::runtime_error("LTTng metrics start failed: " + err);
    }
    report_warning_later("LTTng metrics unavailable: " + err);
    impl->lttng_metrics.reset();
  }
}

impl->start = Clock::now();
core_->measurement_active = true;
```

Stop order:

```cpp
core_->measurement_active = false;
impl->end = Clock::now();
impl->after = stats();
auto after_diag = diag_snapshot();
auto after_graph_metrics = build_graph_metrics_report_run_lifetime(...);

if (impl->lttng_metrics) {
  std::string err;
  if (impl->lttng_metrics->stop_and_destroy(&err)) {
    auto parsed = impl->lttng_metrics->parse(&err);
    merge_lttng_results(parsed, report);
  } else {
    report.plugin_latency_status = "failed";
    report.warnings.push_back("LTTng stop/destroy failed: " + err);
  }
  impl->lttng_metrics->cleanup_noexcept();
}

if (options.include_edge_latency) {
  append_edge_latency_from_diag_delta(*impl->before_diag, after_diag, &report);
}

attribute_plugin_latency_to_nodes(&report);
attribute_edge_latency_to_graph_edges(&report);
```

No `LatencyProfiler` construction in Auto/Lttng.

## JSON/schema/visualizer

### JSON export

File: `src/pipeline/runtime/GraphRunExport.cpp`

Add provenance:

```json
"metrics_sources": {
  "throughput": "run_measure_window",
  "power": "board_power_monitor",
  "node_latency": "graph_runtime_metrics",
  "plugin_latency": {"status": "collected", "source": "lttng"},
  "edge_message_latency": {"status": "collected", "source": "diagnostics"}
}
```

Plugin rows include:

```json
"stream_id": "stream0",
"plugin_instance_id": "r123.g456.s0.n3.eprocessmla0",
"source": "lttng",
"attribution_source": "lttng_v2_identity",
"mapping_error": null,
"reliable": true
```

Edge rows live separately under `edge_metrics` and `edge_metrics_unattributed`.

### Schema

File: `schemas/graph_run_v1.schema.json`

Add optional fields with `additionalProperties: true` retained for compatibility:

```json
"edge_metrics": {"type": "array"},
"edge_metrics_unattributed": {"type": "array"},
"metrics_sources": {"type": "object"}
```

### HTML visualizer

File: `tools/visualize_graph_run.py`

Add:

- top cards: throughput, power, end-to-end latency unchanged;
- node/plugin section: plugin execution spans only;
- separate section titled `Inter-plugin message / edge latency`;
- note: `Measures handoff/queue/transport time between nodes/plugins. Do not add to plugin execution latency or graph latency.`

Columns for edge table:

```text
Edge | From | To | Stream | Samples | Avg | P50 | P95 | Max | Semantics | Source | Mapping/warning
```

Optional graph rendering: label SVG edges with `avg/p95` edge latency, not node labels.

## Legacy removal plan

The implementation should remove legacy from the customer metrics path, not add new dependencies on it.

### Immediate changes

- `RunMeasure.cpp`: never construct `LatencyProfiler` for `Auto` or `Lttng`.
- `MeasureOptions` docs: replace `NEAT profiler` wording with `LTTng plugin execution spans`.
- Do not parse legacy `gst_message_new_application("kpi", ...)` into `MeasureReport`.
- Metrics path sets `trace-enabled=true`, never `transmit=true`.
- If LTTng is unavailable in `Auto`: `plugin_latency_status="unavailable"` and add warning.
- If LTTng is unavailable in `Lttng`: throw before measurement starts.

### Compile guard or deletion after parity

```text
include/pipeline/LatencyProfiler.h            -> deprecate, then remove or guard
src/pipeline/profiler/LatencyProfiler.cpp     -> guard behind SIMA_ENABLE_LEGACY_PROFILER
internals/core/profiler/profiler_events.cpp   -> guard/remove
libsimaaineatprofiler.so preload in pyneat     -> remove unless debug flag enabled
CMake NeatInternals::simaaineatprofiler link   -> optional/dev-only
profiler_events.h users in plugins/dispatcher  -> remove or guard
```

Build flag:

```cmake
option(SIMA_ENABLE_LEGACY_PROFILER "Enable deprecated process-global NEAT profiler" OFF)
```

If disabled, customer code still builds and reports LTTng/unavailable status cleanly.

## CMake/build changes

Core runtime additions:

```cmake
target_sources(sima_neat_objects PRIVATE
  src/pipeline/runtime/LttngMetricsCollector.cpp
  src/pipeline/runtime/LttngTraceParser.cpp
  src/pipeline/runtime/LttngTraceSession.cpp
  src/pipeline/runtime/TraceAttribution.cpp
  src/pipeline/runtime/EdgeMetrics.cpp
  src/pipeline/gst/TraceIdentity.cpp
)
```

Core does not link `liblttng-ust`; it shells out through `execvp` to `lttng`/`babeltrace2`.

Trace provider build:

```c
#define TRACEPOINT_CREATE_PROBES
#define TRACEPOINT_DEFINE
#include <simaai/trace/sima_neat_metrics_tp.h>
```

Add provider object to `libsimaaitrace.so` and install the header beside existing trace headers.

## Tests

### Parser/unit tests on host

```text
tests/perf/lttng_trace_parser_test.cpp
```

Cases:

1. V2 plugin start/end pairs to one `MeasurePluginLatency`.
2. Current `pipeline:*` pairs but is marked fallback/low confidence.
3. Duplicate starts pair FIFO only for exact keys; otherwise unattributed.
4. Missing end creates warning/unattributed.
5. `run_id_hash` mismatch is dropped/warned.
6. Same element name in two graph segments: fallback ambiguous; v2 identity correct.
7. Trace loss marker sets `reliable=false` and warning.
8. `sima_neat_edge:message` queue in/out computes queue residence.
9. Edge src/dst events compute edge transport.
10. Negative latency and missing message id go to unattributed diagnostics.

### Runtime/devkit tests

1. `trace-enabled=true`, `transmit=false`: LTTng events exist; no GST `kpi` bus messages emitted.
2. Multi-stream same frame id: plugin rows separated by `stream_id` and `plugin_instance_id`.
3. Two concurrent Runs in same process: v2 parser filters by `run_id_hash`; fallback disabled/unavailable.
4. Fan-out/fan-in edge timing: same message id on multiple edges remains separated by `edge_id`.
5. Legacy disabled: Auto/Lttng never constructs `LatencyProfiler`; `sima_neat_profiler_enabled()` remains false/zero if still present.
6. DVT power: report says unavailable/untrusted without failing the run.
7. SOM power: rail average appears in graph summary.

### Visualizer tests

1. Plugin metrics render under node/plugin section.
2. Edge metrics render in a separate edge table.
3. Throughput/power cards are unchanged when edge metrics are enabled.
4. Unattributed rows show reason codes and do not appear as node/plugin latency.

## Rollout phases

### Phase 1: private LTTng collector and safe fallback

- Add private collector/session/parser.
- Use correct LTTng command sequence with `--session` on every command.
- Parse current `pipeline:*` only as single-run diagnostic fallback.
- No legacy profiler in Auto/Lttng.
- Add trace retention cleanup.

Exit criteria: customer report succeeds with plugin status `collected` or `unavailable`; no teardown crash; no legacy profiler construction.

### Phase 2: v2 plugin identity

- Add `sima_neat_plugin:plugin_span` provider using struct-pointer args.
- Add shared trace identity property helper.
- Add `trace-enabled` property; do not set `transmit`.
- Inject root run/graph/segment/node/plugin-instance identity before PLAYING.
- Parser filters by root `run_id_hash`.

Exit criteria: multi-stream and same-process concurrent Run attribution is correct with v2 events.

### Phase 3: edge/message timing

- Add low-overhead `MeasureEdgeLatency` from existing diagnostics and `BlockingQueue` residence.
- Add exact `sima_neat_edge:message` events for selected edges when `include_message_latency=true`.
- Add `trace_sample_id/message_id` propagation where practical.
- Add separate JSON/HTML edge sections.

Exit criteria: report shows inter-plugin timing separately and never folds it into plugin latency.

### Phase 4: legacy cleanup and polish

- Guard/remove `LatencyProfiler` and `libsimaaineatprofiler` from default builds.
- Remove KPI-message parsing from metrics path.
- Docs explain LTTng status, DVT/SOM power caveat, edge semantics, trace retention.
- Apps default to simple `auto/off/lttng` config.

## Completion criteria

The report work is complete when:

1. Graph throughput and power are graph-level averages.
2. Node/plugin execution latency is per Node/Plugin with v2 attribution where available.
3. Edge/message latency is separate, non-additive, and clearly labeled.
4. Multi-stream plugin instances are distinguished by stream and plugin instance id.
5. Concurrent same-process Runs are filtered by root `run_id_hash` or fallback is disabled with warnings.
6. Auto mode never silently falls back to the legacy process-global profiler.
7. Metrics path never sets `transmit=true` and does not emit legacy KPI bus messages.
8. JSON schema validates and HTML is clean/customer-readable.
9. DVT power unavailability does not fail the report; SOM power remains implemented.
10. LTTng overhead is measured and acceptable; exact message tracing is opt-in/sampled.
