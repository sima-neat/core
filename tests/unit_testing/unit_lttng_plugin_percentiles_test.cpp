#include "pipeline/runtime/LttngMetricsCollector.h"
#include "test_main.h"
#include "test_utils.h"

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

void require_near(double actual, double expected, const std::string& label) {
  require(std::abs(actual - expected) < 1e-8,
          label + ": expected=" + std::to_string(expected) + " actual=" + std::to_string(actual));
}

} // namespace

RUN_TEST("unit_lttng_plugin_percentiles_test", ([] {
           using simaai::neat::pipeline_internal::parse_lttng_trace_text;

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
