#include "pipeline/RuntimeMetrics.h"
#include "test_main.h"
#include "test_utils.h"

RUN_TEST("unit_runtime_metrics_test", ([] {
           simaai::neat::RuntimeMetrics metrics;
           metrics.source_kind = "run";
           metrics.source_name = "unit";
           metrics.elapsed_seconds = 2.0;
           metrics.throughput_fps = 10.0;
           metrics.latency.avg_ms = 3.0;
           metrics.latency.min_ms = 1.0;
           metrics.latency.max_ms = 5.0;
           metrics.counters.outputs_pulled = 20;
           metrics.groups.push_back({"diagnostic", {{"samples", 20.0, "count"}}});

           const std::string text = simaai::neat::format_runtime_metrics(
               metrics, simaai::neat::RuntimeMetricsFormat::Text);
           require_contains(text, "throughput_fps=10", "text metrics throughput");
           require_contains(text, "Latency:", "text metrics latency");

           const std::string compact = simaai::neat::format_runtime_metrics(
               metrics, simaai::neat::RuntimeMetricsFormat::CompactText);
           require_contains(compact, "RuntimeMetrics", "compact metrics header");

           const std::string json = simaai::neat::format_runtime_metrics(
               metrics, simaai::neat::RuntimeMetricsFormat::Json);
           require_contains(json, "\"source_kind\": \"run\"", "json metrics source");
           require_contains(json, "\"throughput_fps\": 10.000000", "json metrics throughput");
         }));
