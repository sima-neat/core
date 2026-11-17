#include "pipeline/runtime/LttngMetricsCollector.h"
#include "test_main.h"
#include "test_utils.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>

namespace {

std::string plugin_event(double timestamp_s, int event_type, int frame_id) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(6) << '[' << timestamp_s
      << "] sima_neat_plugin:plugin_span: { event_type = " << event_type
      << ", run_id_hash = 17, graph_id_hash = 23, pipeline_segment_id = 2"
         ", runtime_node_id = 5, public_node_id = 3"
         ", plugin_instance_id = \"r11.g17.s2.n5.boxdecode\""
         ", element_name = \"boxdecode\", backend = \"A65\", phase = \"Exec\""
         ", kernel_name = \"boxdecode_plugin_exclusive\", stream_id = \"stream-0\""
      << ", frame_id = " << frame_id << ", request_id = " << frame_id << ", message_id = \"message-"
      << frame_id << "\" }\n";
  return out.str();
}

void require_near(double actual, double expected, const std::string& label);

std::string selected_event(double timestamp_s, int event_type, int frame_id,
                           std::string instance = "r11.g17.s2.n5.boxdecode.m2",
                           std::int64_t orig_input_seq = -1, std::string stream = "stream-0") {
  auto event = plugin_event(timestamp_s, event_type, frame_id);
  const std::string old_instance = "r11.g17.s2.n5.boxdecode";
  event.replace(event.find(old_instance), old_instance.size(), instance);
  event.replace(event.find("stream-0"), 8, stream);
  event.insert(event.find(" }"), ", orig_input_seq = " + std::to_string(orig_input_seq));
  const std::string old_message = "\"message-" + std::to_string(frame_id) + "\"";
  event.replace(event.find(old_message), old_message.size(),
                std::to_string(orig_input_seq >= 0 ? orig_input_seq : frame_id));
  return event;
}

bool has_warning(const simaai::neat::pipeline_internal::LttngParseResult& parsed,
                 const std::string& prefix) {
  return std::any_of(parsed.warnings.begin(), parsed.warnings.end(),
                     [&](const auto& warning) { return warning.starts_with(prefix); });
}

void check_measured_selection() {
  using namespace simaai::neat::pipeline_internal;
  PluginTraceSelection selection;
  selection.require_session_identity = true;
  selection.measurement_epoch = 2;
  selection.plugin_instance_ids.insert("r11.g17.s2.n5.boxdecode.m2");
  selection.require_sample_identity = true;
  add_measured_plugin_sample(&selection, {"stream-0", PluginSampleKeyKind::OrigInputSeq, 10});
  add_measured_plugin_sample(&selection, {"stream-0", PluginSampleKeyKind::OrigInputSeq, 11});
  const std::string begin = selected_event(2.0, 0, 1, *selection.plugin_instance_ids.begin(), 10);
  const std::string end = selected_event(2.0005, 1, 1, *selection.plugin_instance_ids.begin(), 10);
  const std::string measured = begin + end;
  const auto parse = [&](const std::string& text) {
    return parse_lttng_trace_text(text, 17, 23, false, selection);
  };

  // A late old-session end and current-session warmup work are discarded BEFORE pairing.
  const auto filtered =
      parse(selected_event(1.0, 1, 1, "r11.g17.s2.n5.boxdecode.m1", 10) +
            selected_event(1.1, 0, 0, *selection.plugin_instance_ids.begin(), 0) +
            selected_event(1.2, 1, 0, *selection.plugin_instance_ids.begin(), 0) + measured);
  require(filtered.raw_plugin_spans.size() == 1 && filtered.plugin_metrics.size() == 1 &&
              filtered.plugin_metrics.front().calls == 1 &&
              filtered.plugin_metrics.front().reliable && filtered.warnings.empty(),
          "old epochs and warmup must not inflate counts or create unmatched warnings");
  require_near(filtered.plugin_metrics.front().avg_ms, 0.5, "selected-only latency");

  const auto old_only = parse(selected_event(2.0, 0, 1, "r11.g17.s2.n5.boxdecode.m1", 10) +
                              selected_event(2.1, 1, 1, "r11.g17.s2.n5.boxdecode.m1", 10));
  require(old_only.raw_plugin_spans.empty() && old_only.warnings.empty(),
          "repeated measurement must not admit a prior epoch's same-frame pair");
  const auto foreign_stream =
      parse(selected_event(2.0, 0, 1, *selection.plugin_instance_ids.begin(), 10, "other") +
            selected_event(2.1, 1, 1, *selection.plugin_instance_ids.begin(), 10, "other"));
  require(foreign_stream.raw_plugin_spans.empty() && foreign_stream.warnings.empty(),
          "equal sequence numbers on different streams are not interchangeable");
  const auto unknown = parse(measured + selected_event(2.1, 1, 1, "unknown.m2", 10));
  require(has_warning(unknown, "UNKNOWN_PLUGIN_SESSION_IDENTITY") &&
              !unknown.plugin_metrics.front().reliable,
          "unknown current-session identities must invalidate exact measurement");
  require(has_warning(parse(selected_event(2.0, 0, 1, "", 10)), "MISSING_PLUGIN_SESSION_IDENTITY"),
          "missing session identity must be diagnosed, not attributed by element name");
  require(has_warning(parse(selected_event(2.0, 0, 1, "boxdecode.mbad", 10)),
                      "INVALID_PLUGIN_SESSION_EPOCH"),
          "malformed session epoch must not silently pass as foreign");
  require(has_warning(parse(selected_event(2.0, 0, -1)), "MISSING_PLUGIN_SAMPLE_IDENTITY"),
          "missing invocation identity must be diagnosed");

  auto malformed = begin;
  malformed.replace(malformed.find("orig_input_seq = 10"), 19, "orig_input_seq = 10broken");
  require(has_warning(parse(malformed), "MISSING_PLUGIN_SAMPLE_IDENTITY"),
          "malformed original authority must not fall back to an apparently valid frame");
  auto malformed_run = begin;
  const std::string run_field = "run_id_hash = 17";
  malformed_run.replace(malformed_run.find(run_field), run_field.size(), "run_id_hash = 17broken");
  require(has_warning(parse(malformed_run), "MISSING_PLUGIN_RUN_IDENTITY"),
          "partially parsed run identities must not pass current-session validation");
  const auto incomplete =
      parse(measured + selected_event(2.2, 0, 2, *selection.plugin_instance_ids.begin(), 11));
  require(has_warning(incomplete, "UNMATCHED_PLUGIN_START") &&
              incomplete.plugin_metrics.size() == 1 && !incomplete.plugin_metrics.front().reliable,
          "early stop must expose a selected partial span without synthetic success");
  const auto duplicate = parse(measured + measured);
  require(has_warning(duplicate, "DUPLICATE_PLUGIN_START") &&
              duplicate.plugin_metrics.front().calls == 1 &&
              !duplicate.plugin_metrics.front().reliable,
          "duplicate same-invocation spans must not satisfy minimum sample count");

  // Input authority is also part of pairing: a reused frame/request/message ID is insufficient.
  const auto changed_input =
      parse(begin + selected_event(2.0005, 1, 1, *selection.plugin_instance_ids.begin(), 11));
  require(changed_input.raw_plugin_spans.empty() &&
              has_warning(changed_input, "UNMATCHED_PLUGIN_END") &&
              has_warning(changed_input, "UNMATCHED_PLUGIN_START"),
          "reused frame identity must not pair distinct input sequences");
  add_measured_plugin_sample(&selection, {"stream-0", PluginSampleKeyKind::OrigInputSeq, 10});
  const auto ambiguous = parse(measured);
  require(ambiguous.raw_plugin_spans.empty() &&
              has_warning(ambiguous, "AMBIGUOUS_MEASURED_PLUGIN_SAMPLE_IDENTITY"),
          "duplicate graph-entry authority is ambiguous and must not be counted");

  // Actual installed provider schema has numeric message_id, not orig/input fields.
  // Reuse the key kind already selected by graph entry, including zero and all fallbacks.
  for (const auto kind : {PluginSampleKeyKind::OrigInputSeq, PluginSampleKeyKind::InputSeq,
                          PluginSampleKeyKind::FrameId}) {
    PluginTraceSelection projected;
    projected.require_sample_identity = true;
    add_measured_plugin_sample(&projected, {"stream-0", kind, 0});
    const auto projection = parse_lttng_trace_text(
        selected_event(3.0, 0, 0) + selected_event(3.001, 1, 0), 17, 23, false, projected);
    require(projection.raw_plugin_spans.size() == 1 && projection.warnings.empty(),
            "all graph-entry identity kinds project to numeric message_id; zero is valid");
  }
  PluginTraceSelection collision;
  collision.require_sample_identity = true;
  add_measured_plugin_sample(&collision, {"stream-0", PluginSampleKeyKind::OrigInputSeq, 1});
  add_measured_plugin_sample(&collision, {"stream-0", PluginSampleKeyKind::FrameId, 1});
  const auto colliding = parse_lttng_trace_text(
      selected_event(3.0, 0, 1) + selected_event(3.001, 1, 1), 17, 23, false, collision);
  require(colliding.raw_plugin_spans.empty() &&
              has_warning(colliding, "AMBIGUOUS_PLUGIN_SAMPLE_IDENTITY"),
          "equal numeric message IDs with different source key kinds cannot be guessed");

  PluginTraceSelection normalized;
  normalized.require_sample_identity = true;
  add_measured_plugin_sample(&normalized, {"", PluginSampleKeyKind::FrameId, 1});
  const auto default_stream = parse_lttng_trace_text(
      selected_event(2.0, 0, 1, "legacy", -1, "") + selected_event(2.001, 1, 1, "legacy", -1, ""),
      17, 23, false, normalized);
  require(default_stream.raw_plugin_spans.size() == 1,
          "empty stream identity uses the existing graph-timing default normalization");
}

void require_near(double actual, double expected, const std::string& label) {
  require(std::abs(actual - expected) < 1e-8,
          label + ": expected=" + std::to_string(expected) + " actual=" + std::to_string(actual));
}

} // namespace

RUN_TEST("unit_lttng_plugin_percentiles_test", ([] {
           using simaai::neat::pipeline_internal::parse_lttng_trace_text;
           check_measured_selection();

           std::string trace;
           const double durations_ms[] = {0.1, 0.2, 0.3, 0.4, 0.5};
           for (int i = 0; i < 5; ++i) {
             const double start_s = 1.0 + static_cast<double>(i) * 0.01;
             trace += plugin_event(start_s, 0, i);
             trace += plugin_event(start_s + durations_ms[i] / 1000.0, 1, i);
           }

           const auto parsed = parse_lttng_trace_text(trace, 17, 23, false);
           require(parsed.parsed, "synthetic trace should parse");
           require(parsed.plugin_metrics.size() == 1,
                   "exact component identity should aggregate into one row");
           require(parsed.raw_plugin_spans.size() == 5,
                   "all exact per-call component spans should be retained");
           require(parsed.plugin_metric_percentiles.size() == 1,
                   "every aggregated plugin row should have one percentile sidecar");

           const auto& row = parsed.plugin_metrics.front();
           const auto& percentiles = parsed.plugin_metric_percentiles.front();
           require(row.backend == "A65" && row.phase == "Exec" &&
                       row.kernel_name == "boxdecode_plugin_exclusive" &&
                       row.stage_name == "boxdecode" &&
                       row.plugin_instance_id == "r11.g17.s2.n5.boxdecode",
                   "component identity fields should remain exact");
           require(row.calls == 5, "component sample count should equal paired spans");
           require(row.reliable, "loss-free paired trace should be reliable");
           require(percentiles.available, "per-call duration percentiles should be available");
           require_near(row.avg_ms, 0.3, "average");
           require_near(percentiles.p50_ms, 0.3, "p50");
           require_near(percentiles.p95_ms, 0.48, "p95");
           require_near(percentiles.p99_ms, 0.496, "p99");
           require_near(row.max_ms, 0.5, "maximum");

           const auto with_loss =
               parse_lttng_trace_text("discarded events\n" + trace, 17, 23, false);
           require(with_loss.trace_loss_detected, "trace loss marker should be retained");
           require(with_loss.plugin_metrics.size() == 1 &&
                       !with_loss.plugin_metrics.front().reliable,
                   "trace loss should invalidate component reliability");
         }));
