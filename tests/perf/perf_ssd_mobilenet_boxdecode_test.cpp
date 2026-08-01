#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif

#include "model/Model.h"
#include "nodes/groups/ModelGroups.h"
#include "nodes/io/Input.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "perf_metrics_common.h"
#include "pipeline/Graph.h"

#include "e2e_pipelines/obj_detection/obj_detection_utils.h"
#include "e2e_pipelines/obj_detection/ssd_mobilenet_test_utils.h"
#include "test_utils.h"

#include <opencv2/imgcodecs.hpp>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;

class RedirectStdoutToStderr {
public:
  RedirectStdoutToStderr() : original_(std::cout.rdbuf(std::cerr.rdbuf())) {}

  RedirectStdoutToStderr(const RedirectStdoutToStderr&) = delete;
  RedirectStdoutToStderr& operator=(const RedirectStdoutToStderr&) = delete;

  ~RedirectStdoutToStderr() {
    restore();
  }

  void restore() {
    if (original_ != nullptr) {
      std::cout.rdbuf(original_);
      original_ = nullptr;
    }
  }

private:
  std::streambuf* original_;
};

struct ComponentIdentity {
  const char* component_id;
  const char* backend;
  const char* phase;
  const char* kernel_name;
  const char* stage_name;
};

std::vector<const simaai::neat::MeasurePluginLatencyWithPercentiles*>
select_exact_component_rows(const simaai::neat::MeasureReportWithPluginPercentiles& report,
                            const ComponentIdentity& identity) {
  std::vector<const simaai::neat::MeasurePluginLatencyWithPercentiles*> matches;
  for (const auto& row : report.plugin_latency) {
    const auto& metric = row.metric;
    if (metric.backend == identity.backend && metric.phase == identity.phase &&
        metric.kernel_name == identity.kernel_name && metric.stage_name == identity.stage_name &&
        metric.gst_element_name == identity.stage_name && !metric.plugin_instance_id.empty()) {
      matches.push_back(&row);
    }
  }
  return matches;
}

void validate_component_selection(
    const ComponentIdentity& identity,
    const std::vector<const simaai::neat::MeasurePluginLatencyWithPercentiles*>& rows) {
  if (rows.empty()) {
    throw std::runtime_error(std::string("PERF_HARNESS_ERROR:HARNESS_COMPONENT_MISSING: ") +
                             identity.component_id);
  }
  if (rows.size() != 1U) {
    throw std::runtime_error(std::string("PERF_HARNESS_ERROR:HARNESS_COMPONENT_AMBIGUOUS: ") +
                             identity.component_id + " matches=" + std::to_string(rows.size()));
  }
  const auto& row = rows.front()->metric;
  const auto& percentiles = rows.front()->percentiles;
  if (!row.reliable) {
    throw std::runtime_error(std::string("PERF_HARNESS_ERROR:HARNESS_COMPONENT_UNRELIABLE: ") +
                             identity.component_id);
  }
  if (row.calls == 0U) {
    throw std::runtime_error(std::string("PERF_HARNESS_ERROR:HARNESS_COMPONENT_SAMPLE_COUNT: ") +
                             identity.component_id + " samples=0");
  }
  if (!percentiles.available || !std::isfinite(percentiles.p50_ms) ||
      !std::isfinite(percentiles.p95_ms) || !std::isfinite(percentiles.p99_ms)) {
    throw std::runtime_error(
        std::string("PERF_HARNESS_ERROR:HARNESS_COMPONENT_PERCENTILES_MISSING: ") +
        identity.component_id);
  }
  if (!std::isfinite(row.avg_ms) || !std::isfinite(row.max_ms) || row.avg_ms < 0.0 ||
      row.max_ms < 0.0) {
    throw std::runtime_error(std::string("PERF_HARNESS_ERROR:HARNESS_SCHEMA_INVALID: ") +
                             identity.component_id + " has invalid latency values");
  }
}

} // namespace

int main(int argc, char** argv) {
  try {
    using namespace simaai::neat;

    // The performance harness parses stdout as one JSON document. Route graph/runtime progress
    // messages to stderr until all measurements and validations are complete.
    RedirectStdoutToStderr progress_output;

    const fs::path root = (argc > 1) ? fs::path(argv[1]) : fs::current_path();
    const int warmup = sima_perf::env_int("SIMA_SSD_PERF_WARMUP", 25);
    const int iterations = sima_perf::env_int("SIMA_PERF_ITERS", 1000);
    const auto config = sima_ssd_mobilenet_test::ModelConfig{};
    const std::string archive = sima_ssd_mobilenet_test::resolve_ssd_mobilenet_tar_or_skip(root);
    const cv::Mat image = sima_ssd_mobilenet_test::load_coco_people_image_or_skip(root);

    Model model(archive, sima_ssd_mobilenet_test::make_model_options(config));
    Graph graph;
    graph.add(nodes::Input());
    graph.add(nodes::groups::Preprocess(model));
    graph.add(nodes::groups::Infer(model));
    graph.add(nodes::SimaBoxDecode(model, BoxDecodeType::Ssd, config.score_threshold,
                                   config.nms_iou, config.top_k));
    graph.add(nodes::Output());

    auto make_sample = [&]() {
      return Sample{Sample::from_image(image, ImageSpec::PixelFormat::BGR, TensorMemory::EV74)};
    };

    const auto startup_t0 = sima_perf::Clock::now();
    Run run = graph.build_seeded_internal(make_sample(), RunMode::Sync);
    const auto startup_t1 = sima_perf::Clock::now();

    for (int i = 0; i < warmup; ++i) {
      const Sample outputs = run.run(make_sample(), 30000);
      if (outputs.empty()) {
        throw std::runtime_error("SSD performance warmup returned no output");
      }
    }

    MeasureOptions measure_options;
    measure_options.title = "SSD-MobileNetV2 full graph and BoxDecode";
    measure_options.model = "ssd-mobile-300-v1";
    measure_options.input = std::to_string(image.cols) + "x" + std::to_string(image.rows) + " BGR";
    measure_options.placement = "CVU + MLA + on-device BoxDecode";
    measure_options.warmup_ms = 0;
    measure_options.timeout_ms = 30000;
    measure_options.include_plugin_latency = true;
    measure_options.include_edge_latency = false;
    measure_options.include_power = false;
    auto measurement = run.start_measurement(measure_options);

    std::vector<double> latencies_ms;
    latencies_ms.reserve(static_cast<std::size_t>(iterations));
    PowerMonitor power_monitor(sima_perf::power_options_from_env());
    power_monitor.start();
    const auto run_t0 = sima_perf::Clock::now();
    for (int i = 0; i < iterations; ++i) {
      const auto t0 = sima_perf::Clock::now();
      const Sample outputs = run.run(make_sample(), 30000);
      const auto t1 = sima_perf::Clock::now();
      if (outputs.empty()) {
        throw std::runtime_error("SSD performance run returned no output at iteration " +
                                 std::to_string(i));
      }
      latencies_ms.push_back(sima_perf::elapsed_ms(t0, t1));

      // Keep validation outside the per-sample latency, but inside the measured run, so corrupt
      // output cannot produce attractive performance numbers.
      std::vector<std::uint8_t> payload;
      std::string error;
      if (!objdet::extract_bbox_payload(outputs.front(), i, payload, error)) {
        throw std::runtime_error("SSD performance output validation failed: " + error);
      }
      (void)objdet::parse_boxes_strict(payload, image.cols, image.rows, config.top_k, false);
    }
    const auto run_t1 = sima_perf::Clock::now();
    power_monitor.stop();
    const MeasureReportWithPluginPercentiles measured = measurement.stop_with_plugin_percentiles();
    const MeasureReport& report = measured.report;
    run.stop();

    if (report.outputs != static_cast<std::size_t>(iterations) ||
        !report.latency_samples_collected) {
      throw std::runtime_error("SSD performance measurement did not correlate every output");
    }
    if (report.plugin_latency_status != "collected") {
      throw std::runtime_error("SSD BoxDecode plugin latency was not collected (status=" +
                               report.plugin_latency_status + ")");
    }
    constexpr ComponentIdentity kBackendComponent{"boxdecode_backend", "A65", "Run",
                                                  "boxdecode_backend", "boxdecode"};
    constexpr ComponentIdentity kExclusiveComponent{"boxdecode_plugin_exclusive", "A65", "Exec",
                                                    "boxdecode_plugin_exclusive", "boxdecode"};
    auto backend_rows = select_exact_component_rows(measured, kBackendComponent);
    auto exclusive_rows = select_exact_component_rows(measured, kExclusiveComponent);
    validate_component_selection(kBackendComponent, backend_rows);
    validate_component_selection(kExclusiveComponent, exclusive_rows);

    sima_perf::PerfMetrics metrics;
    const double elapsed_s = sima_perf::elapsed_seconds(run_t0, run_t1);
    metrics.throughput = elapsed_s > 0.0 ? static_cast<double>(iterations) / elapsed_s : 0.0;
    metrics.p50 = sima_perf::percentile(latencies_ms, 50.0);
    metrics.p95 = sima_perf::percentile(latencies_ms, 95.0);
    metrics.startup = sima_perf::elapsed_ms(startup_t0, startup_t1);
    metrics.rss_peak_kb = sima_perf::rss_peak_kb();
    metrics.input_drop_count = report.counters.inputs_dropped;
    metrics.output_drop_count = report.counters.outputs_dropped;

    const PowerSummary power = power_monitor.summary();
    const std::vector<sima_perf::ComponentLatencySelection> component_latency = {
        {kBackendComponent.component_id, std::move(backend_rows)},
        {kExclusiveComponent.component_id, std::move(exclusive_rows)},
    };
    progress_output.restore();
    sima_perf::emit_metrics_json("ssd_mobilenet_boxdecode", iterations, metrics, "sync", &power,
                                 &report, &component_latency);
    return 0;
  } catch (const SkipTest& error) {
    std::cerr << "perf_ssd_mobilenet_boxdecode_test skipped: " << error.what() << "\n";
    return skip_long_test(error.what());
  } catch (const std::exception& error) {
    if (is_dispatcher_unavailable(error.what())) {
      return skip_long_test("dispatcher unavailable");
    }
    std::cerr << "perf_ssd_mobilenet_boxdecode_test exception: " << error.what() << "\n";
    return 1;
  }
}
