#pragma once

#include "pipeline/Run.h"

#include <cstdint>
#include <map>
#include <set>
#include <tuple>
#include <string>
#include <string_view>
#include <vector>

namespace simaai::neat::pipeline_internal {

std::uint64_t stable_trace_hash(std::string_view value);

struct TraceIdentityContext {
  std::uint64_t run_id_hash = 0;
  std::uint64_t graph_id_hash = 0;
  std::uint32_t pipeline_segment_id = UINT32_MAX;
};

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
  bool add_procname_context = false;
};

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

struct ParsedGraphMessageEvent {
  double timestamp_s = 0.0;
  std::uint32_t event_type = 0;
  std::uint64_t run_id_hash = 0;
  std::uint64_t graph_id_hash = 0;
  std::string endpoint;
  std::string stream_id;
  std::string message_id;
  std::int64_t frame_id = -1;
  std::int64_t input_seq = -1;
  std::int64_t orig_input_seq = -1;
};

// Offline selection uses the existing measured graph-entry identities. No new
// registry or work is introduced on the streaming/execution path.
enum class PluginSampleKeyKind {
  OrigInputSeq,
  InputSeq,
  FrameId,
  // Existing plugin_span wire schema projects the preferred correlation key to
  // message_id. Offline selection rejects any ambiguous cross-kind projection.
  MessageId
};
struct PluginSampleIdentity {
  std::string stream_id;
  PluginSampleKeyKind kind = PluginSampleKeyKind::OrigInputSeq;
  std::int64_t value = -1;
  bool operator<(const PluginSampleIdentity& other) const {
    return std::tie(stream_id, kind, value) < std::tie(other.stream_id, other.kind, other.value);
  }
};
struct PluginTraceSelection {
  bool require_session_identity = false;
  std::uint64_t measurement_epoch = 0;
  std::set<std::string> plugin_instance_ids;
  bool require_sample_identity = false;
  std::set<PluginSampleIdentity> samples;
  std::set<PluginSampleIdentity> ambiguous_samples;
};
// Duplicate graph entry keys are ambiguous, not two interchangeable observations.
void add_measured_plugin_sample(PluginTraceSelection* selection, PluginSampleIdentity sample);

struct LttngParseResult {
  bool parsed = false;
  bool trace_loss_detected = false;
  std::string trace_dir;
  std::vector<MeasurePluginLatency> plugin_metrics;
  std::vector<MeasurePluginLatencyPercentiles> plugin_metric_percentiles;
  std::vector<MeasurePluginLatency> plugin_metrics_unattributed;
  std::vector<MeasurePluginLatencyPercentiles> plugin_metric_percentiles_unattributed;
  std::vector<MeasureEdgeLatency> edge_metrics;
  std::vector<MeasureEdgeLatency> edge_metrics_unattributed;
  std::vector<ParsedPluginSpan> raw_plugin_spans;
  std::vector<ParsedEdgeSpan> raw_edge_spans;
  std::vector<ParsedGraphMessageEvent> raw_graph_events;
  std::vector<std::string> warnings;
};

class LttngMetricsCollector {
public:
  LttngMetricsCollector(InternalMetricsTraceOptions opt, TraceIdentityContext trace_context);
  ~LttngMetricsCollector() noexcept;

  LttngMetricsCollector(const LttngMetricsCollector&) = delete;
  LttngMetricsCollector& operator=(const LttngMetricsCollector&) = delete;

  bool available(std::string* reason) const;
  bool start(std::string* err);
  bool stop_and_destroy(std::string* err);
  LttngParseResult parse(std::string* err) const;
  LttngParseResult parse(const PluginTraceSelection& selection, std::string* err) const;
  void cleanup_noexcept();
  void retain_trace_for_debug() {
    opt_.retain_trace = true;
  }

  const std::string& session_name() const {
    return session_name_;
  }
  const std::string& trace_dir() const {
    return trace_dir_;
  }
  bool retained() const {
    return opt_.retain_trace;
  }

private:
  InternalMetricsTraceOptions opt_;
  TraceIdentityContext trace_context_;
  std::string session_name_;
  std::string trace_dir_;
  bool created_ = false;
  bool started_ = false;
  bool owns_trace_dir_ = false;
};

LttngParseResult parse_lttng_trace_text(const std::string& text, std::uint64_t expected_run_id_hash,
                                        std::uint64_t expected_graph_id_hash,
                                        bool allow_pipeline_fallback);

LttngParseResult parse_lttng_trace_text(const std::string& text, std::uint64_t expected_run_id_hash,
                                        std::uint64_t expected_graph_id_hash,
                                        bool allow_pipeline_fallback,
                                        const PluginTraceSelection& selection);

} // namespace simaai::neat::pipeline_internal
