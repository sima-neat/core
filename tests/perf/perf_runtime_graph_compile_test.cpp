#include "graph/Compiler.h"
#include "graph/nodes/PipelineNode.h"
#include "nodes/common/Output.h"
#include "nodes/common/VideoConvert.h"
#include "nodes/io/Input.h"
#include "perf_metrics_common.h"

#include <memory>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

int main() {
  try {
    using namespace simaai::neat;
    constexpr std::size_t node_count = 128;
    constexpr std::size_t segment_size = 8;
    graph::Graph input;
    std::vector<graph::NodeId> ids;
    std::unordered_set<graph::NodeId> boundaries;
    for (std::size_t i = 0; i < node_count; ++i) {
      auto node = i == 0                ? nodes::Input()
                  : i + 1 == node_count ? nodes::Output()
                                        : nodes::VideoConvert();
      ids.push_back(input.add(std::make_shared<graph::nodes::PipelineNode>(
          std::move(node), "node_" + std::to_string(i))));
      if (i != 0)
        input.connect(ids[i - 1], ids[i]);
      if (i != 0 && i % segment_size == 0)
        boundaries.insert(ids[i]);
    }
    graph::CompilerOptions options;
    options.root_input_specs.emplace(
        ids.front(),
        OutputSpec{
            .media_type = "video/x-raw", .format = "RGB", .width = 32, .height = 24, .depth = 3});
    graph::Compiler compiler;
    if (compiler.compile(input, options).pipelines.size() != 1)
      throw std::runtime_error("default graph did not merge into one segment");

    const auto validate = [&](const graph::CompiledGraph& compiled) {
      if (compiled.pipelines.size() != node_count / segment_size || !compiled.stages.empty() ||
          compiled.edges.size() != node_count - 1)
        throw std::runtime_error("compiler changed graph partitioning");
      for (std::size_t i = 0; i < compiled.pipelines.size(); ++i) {
        const auto& segment = compiled.pipelines[i];
        if (segment.node_ids.size() != segment_size ||
            segment.input_edges.size() != (i == 0 ? 0U : 1U) ||
            segment.output_edges.size() != (i + 1 == compiled.pipelines.size() ? 0U : 1U))
          throw std::runtime_error("compiler changed segment boundaries");
        for (std::size_t j = 0; j < segment_size; ++j)
          if (segment.node_ids[j] != ids[i * segment_size + j])
            throw std::runtime_error("compiler reordered pipeline nodes");
      }
    };
    const auto first_start = sima_perf::Clock::now();
    const auto first = compiler.compile(input, options, boundaries);
    const auto first_end = sima_perf::Clock::now();
    validate(first);
    for (int i = 0; i < 20; ++i)
      validate(compiler.compile(input, options, boundaries));

    const int iterations = sima_perf::env_int("SIMA_PERF_ITERS", 300);
    if (iterations <= 0)
      throw std::runtime_error("iterations must be positive");
    std::vector<double> durations;
    durations.reserve(static_cast<std::size_t>(iterations));
    for (int i = 0; i < iterations; ++i) {
      const auto start = sima_perf::Clock::now();
      const auto compiled = compiler.compile(input, options, boundaries);
      const auto end = sima_perf::Clock::now();
      durations.push_back(sima_perf::elapsed_ms(start, end));
      validate(compiled);
    }
    sima_perf::PerfMetrics metrics;
    const double total_ms = std::accumulate(durations.begin(), durations.end(), 0.0);
    metrics.throughput = total_ms > 0 ? iterations * 1000.0 / total_ms : 0;
    metrics.p50 = sima_perf::percentile(durations, 50.0);
    metrics.p95 = sima_perf::percentile(durations, 95.0);
    metrics.startup = sima_perf::elapsed_ms(first_start, first_end);
    metrics.rss_peak_kb = sima_perf::rss_peak_kb();
    sima_perf::emit_metrics_json("runtime_graph_compile", iterations, metrics, "compile");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "perf_runtime_graph_compile_test: " << error.what() << '\n';
    return 1;
  }
}
