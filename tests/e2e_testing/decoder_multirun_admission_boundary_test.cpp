#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif

#include "graphs/Fragments.h"
#include "nodes/common/Caps.h"
#include "nodes/common/Output.h"
#include "nodes/common/Queue.h"
#include "nodes/groups/RtspEncodedInput.h"
#include "nodes/io/Input.h"
#include "nodes/sima/SimaDecode.h"
#include "pipeline/Graph.h"
#include "pipeline/internal/Diagnostics.h"
#include "pipeline/runtime/RunCore.h"
#include "rtsp_probe_utils.h"
#include "test_utils.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr std::size_t kStreamCount = 4;
constexpr int kFramesPerStream = 5;
constexpr int kPullTimeoutMs = 20000;
constexpr int kWidth = 1280;
constexpr int kHeight = 720;
constexpr double kMinimumStreamRateRatio = 0.90;
constexpr double kMinimumAggregateRateRatio = 0.95;
constexpr auto kWarmupWindow = std::chrono::seconds(10);
constexpr auto kThroughputWindow = std::chrono::seconds(10);

std::string trim_copy(const std::string& value) {
  const std::size_t start = value.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) {
    return {};
  }
  const std::size_t end = value.find_last_not_of(" \t\r\n");
  return value.substr(start, end - start + 1U);
}

std::vector<std::string> split_urls(const std::string& value) {
  std::vector<std::string> urls;
  std::string current;
  for (const char c : value) {
    if (c == ',' || c == ';' || c == ' ' || c == '\t' || c == '\n' || c == '\r') {
      const std::string trimmed = trim_copy(current);
      if (!trimmed.empty()) {
        urls.push_back(trimmed);
      }
      current.clear();
      continue;
    }
    current.push_back(c);
  }
  const std::string trimmed = trim_copy(current);
  if (!trimmed.empty()) {
    urls.push_back(trimmed);
  }
  return urls;
}

std::vector<std::string> configured_urls() {
  if (const char* many = std::getenv("SIMANEAT_TEST_RTSP_H264_URLS"); many && *many) {
    return split_urls(many);
  }
  return {};
}

std::size_t count_occurrences(const std::string& value, const std::string& needle) {
  std::size_t count = 0;
  std::size_t pos = 0;
  while ((pos = value.find(needle, pos)) != std::string::npos) {
    ++count;
    pos += needle.size();
  }
  return count;
}

bool has_injected_input(const simaai::neat::runtime::PipelineSegmentPlan& segment) {
  return !segment.materialized_node_attribution.empty() &&
         segment.materialized_node_attribution.front().role ==
             simaai::neat::runtime::MaterializedNodeAttribution::Role::InjectedInput;
}

std::string output_name(std::size_t stream_index) {
  return "frame" + std::to_string(stream_index);
}

simaai::neat::Graph make_source(const std::string& url, int fps) {
  simaai::neat::nodes::groups::RtspEncodedInputOptions options;
  options.url = url;
  options.codec = simaai::neat::nodes::groups::RtspCodec::H264;
  options.source_fps = fps;
  options.drop_on_latency = true;
  options.fallback_h264_width = kWidth;
  options.fallback_h264_height = kHeight;
  options.fallback_h264_fps = fps;
  return simaai::neat::nodes::groups::RtspEncodedInput(options);
}

simaai::neat::Graph make_decoder(std::size_t stream_index, int fps) {
  simaai::neat::SimaDecodeOptions options;
  options.type = simaai::neat::SimaDecodeType::H264;
  options.raw_output = true;
  options.dec_width = kWidth;
  options.dec_height = kHeight;
  options.dec_fps = fps;

  simaai::neat::Graph graph("decoder_" + std::to_string(stream_index));
  graph.add(simaai::neat::nodes::SimaDecode(options));
  graph.add(
      simaai::neat::nodes::CapsRaw("NV12", kWidth, kHeight, fps, simaai::neat::CapsMemory::Any));
  graph.add(simaai::neat::nodes::Output(output_name(stream_index),
                                        simaai::neat::OutputOptions::Latest()));
  return graph;
}

simaai::neat::Graph make_single_stream_graph(const std::string& url, int fps) {
  simaai::neat::Graph graph;
  graph.add(make_source(url, fps));

  simaai::neat::SimaDecodeOptions options;
  options.type = simaai::neat::SimaDecodeType::H264;
  options.raw_output = true;
  options.dec_width = kWidth;
  options.dec_height = kHeight;
  options.dec_fps = fps;
  graph.add(simaai::neat::nodes::SimaDecode(options));
  graph.add(
      simaai::neat::nodes::CapsRaw("NV12", kWidth, kHeight, fps, simaai::neat::CapsMemory::Any));
  graph.add(simaai::neat::nodes::Output(simaai::neat::OutputOptions::Latest()));
  return graph;
}

simaai::neat::RunOptions run_options() {
  simaai::neat::RunOptions options;
  options.preset = simaai::neat::RunPreset::Realtime;
  options.queue_depth = 3;
  options.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
  options.output_memory = simaai::neat::OutputMemory::ZeroCopy;
  return options;
}

void require_decoded_frame(const simaai::neat::Sample& sample, const std::string& where) {
  require(simaai::neat::sample_payload_type(sample) == simaai::neat::PayloadType::Image,
          where + ": expected decoded image payload");
  const simaai::neat::TensorList tensors = simaai::neat::tensors_from_sample(sample, true);
  require(tensors.size() == 1U, where + ": expected one decoded tensor");
  const simaai::neat::Tensor& tensor = tensors.front();
  require(tensor.width() == kWidth && tensor.height() == kHeight,
          where + ": decoded dimensions changed");
  require(tensor.storage != nullptr, where + ": decoded tensor has no storage");
}

simaai::neat::PullStatus pull_output(simaai::neat::Run& run, const std::string& endpoint,
                                     int timeout_ms, simaai::neat::Sample& sample,
                                     simaai::neat::PullError* error) {
  return endpoint.empty() ? run.pull(timeout_ms, sample, error)
                          : run.pull(endpoint, timeout_ms, sample, error);
}

simaai::neat::Sample pull_frames(simaai::neat::Run& run, const std::string& endpoint,
                                 const std::string& where) {
  simaai::neat::Sample first_sample;
  for (int frame = 0; frame < kFramesPerStream; ++frame) {
    simaai::neat::Sample sample;
    simaai::neat::PullError error;
    const simaai::neat::PullStatus status =
        pull_output(run, endpoint, kPullTimeoutMs, sample, &error);
    require(status == simaai::neat::PullStatus::Ok,
            where + ": pull failed status=" + std::to_string(static_cast<int>(status)) +
                " error=" + error.message);
    require_decoded_frame(sample, where);
    if (frame == 0) {
      first_sample = sample;
    }
  }
  return first_sample;
}

void require_admitted_boundaries(const simaai::neat::Run& run, std::size_t expected_decoders,
                                 const std::string& where) {
  const auto core = simaai::neat::run_internal::core(run);
  require(core != nullptr, where + ": missing RunCore");
  require(core->decoder_admission && core->decoder_admission->active(),
          where + ": decoder graph started without an admission lease");

  std::size_t admitted_decoders = 0;
  if (core->graph_execution_) {
    for (const auto& pipeline : core->graph_execution_->pipelines) {
      require(pipeline != nullptr, where + ": missing pipeline segment");
      admitted_decoders +=
          count_occurrences(pipeline->last_pipeline, "decoder-admission-required=true");
    }
  } else {
    const auto diag = core->pipeline.stream.diag_ctx();
    require(diag != nullptr, where + ": simple pipeline has no diagnostics");
    admitted_decoders = count_occurrences(diag->pipeline_string, "decoder-admission-required=true");
  }

  require(admitted_decoders == expected_decoders,
          where + ": expected " + std::to_string(expected_decoders) + " admitted decoder(s), got " +
              std::to_string(admitted_decoders));
}

double measure_throughput(const std::vector<simaai::neat::Run*>& runs,
                          const std::vector<std::string>& endpoints,
                          const std::vector<int>& source_fps, std::chrono::seconds window,
                          const std::string& label, bool enforce_thresholds) {
  require(source_fps.size() == kStreamCount, label + ": missing source FPS values");
  std::array<std::size_t, kStreamCount> frame_counts{};
  std::array<std::string, kStreamCount> errors{};
  const auto start = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
  const auto deadline = start + window;
  std::vector<std::thread> drainers;
  drainers.reserve(kStreamCount);

  for (std::size_t i = 0; i < kStreamCount; ++i) {
    drainers.emplace_back([&, i]() {
      while (std::chrono::steady_clock::now() < start) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      while (std::chrono::steady_clock::now() < deadline) {
        simaai::neat::Sample sample;
        simaai::neat::PullError error;
        const auto status = pull_output(*runs[i], endpoints[i], 1000, sample, &error);
        if (status == simaai::neat::PullStatus::Timeout) {
          continue;
        }
        if (status != simaai::neat::PullStatus::Ok) {
          errors[i] = error.message;
          break;
        }
        ++frame_counts[i];
      }
    });
  }

  for (auto& drainer : drainers) {
    drainer.join();
  }
  const double seconds = std::chrono::duration<double>(window).count();

  std::size_t total_frames = 0;
  for (std::size_t i = 0; i < kStreamCount; ++i) {
    require(errors[i].empty(), label + " stream " + std::to_string(i) + " stopped: " + errors[i]);
    total_frames += frame_counts[i];
    const double stream_fps = static_cast<double>(frame_counts[i]) / seconds;
    std::cout << "[INFO] " << label << " stream=" << i << " fps=" << stream_fps << "\n";
    if (enforce_thresholds) {
      const double minimum_stream_fps =
          static_cast<double>(source_fps[i]) * kMinimumStreamRateRatio;
      require(stream_fps >= minimum_stream_fps,
              label + " stream " + std::to_string(i) + " regressed: measured " +
                  std::to_string(stream_fps) + " fps, expected at least " +
                  std::to_string(minimum_stream_fps) + " fps");
    }
  }
  const double aggregate_fps = static_cast<double>(total_frames) / seconds;
  std::cout << "[INFO] " << label << " aggregate_fps=" << aggregate_fps << "\n";
  if (enforce_thresholds) {
    double expected_aggregate_fps = 0.0;
    for (const int expected_fps : source_fps) {
      expected_aggregate_fps += static_cast<double>(expected_fps);
    }
    const double minimum_aggregate_fps = expected_aggregate_fps * kMinimumAggregateRateRatio;
    require(aggregate_fps >= minimum_aggregate_fps,
            label + " aggregate throughput regressed: measured " + std::to_string(aggregate_fps) +
                " fps, expected at least " + std::to_string(minimum_aggregate_fps) + " fps");
  }
  return aggregate_fps;
}

simaai::neat::Graph branch_leg(const std::string& input, const std::string& output) {
  simaai::neat::Graph graph(input);
  graph.add(simaai::neat::nodes::Input(input));
  graph.add(simaai::neat::nodes::Queue());
  graph.add(simaai::neat::nodes::Output(output));
  return graph;
}

void run_appsrc_branch_boundary(const simaai::neat::Sample& decoded) {
  simaai::neat::Graph source("decoded");
  source.add(simaai::neat::nodes::Input("decoded"));
  source.add(simaai::neat::nodes::Queue());

  simaai::neat::Graph branch =
      simaai::neat::graphs::Branch("decoded", {"model_input", "egress_input"});
  simaai::neat::Graph app("decoded_appsrc_branch_boundary");
  app.connect(source, branch);
  app.connect(branch, branch_leg("model_input", "model_output"));
  app.connect(branch, branch_leg("egress_input", "egress_output"));

  simaai::neat::Run run = app.build(run_options());
  require(run.push("decoded", simaai::neat::Sample{decoded}),
          "appsrc branch: decoded-frame push failed: " + run.last_error());
  for (const std::string endpoint : {"model_output", "egress_output"}) {
    simaai::neat::Sample sample;
    simaai::neat::PullError error;
    const auto status = run.pull(endpoint, kPullTimeoutMs, sample, &error);
    require(status == simaai::neat::PullStatus::Ok,
            "appsrc branch: " + endpoint + " pull failed: " + error.message);
    require_decoded_frame(sample, "appsrc branch " + endpoint);
  }

  const auto core = simaai::neat::run_internal::core(run);
  require(core != nullptr, "appsrc branch: missing RunCore");
  std::size_t injected_inputs = 0;
  for (const auto& pipeline : core->graph_execution().pipelines) {
    if (pipeline == nullptr || !has_injected_input(pipeline->seg)) {
      continue;
    }
    ++injected_inputs;
    const auto diag = pipeline->run_core ? pipeline->run_core->pipeline.stream.diag_ctx() : nullptr;
    require(diag != nullptr && !diag->boundaries.empty(),
            "appsrc branch: injected segment has no boundary diagnostics");
    require(diag->boundaries.front()->in_buffers.load() > 0U,
            "appsrc branch: injected appsrc did not receive a buffer");
    require(diag->boundaries.back()->out_buffers.load() > 0U,
            "appsrc branch: injected segment did not forward a buffer");
  }
  require(injected_inputs == 2U, "appsrc branch: expected two injected-input legs, got " +
                                     std::to_string(injected_inputs));
  run.close();
}

double run_combined_graph(const std::vector<std::string>& urls, const std::vector<int>& fps) {
  simaai::neat::Graph graph("combined_decoder_reference");
  for (std::size_t i = 0; i < kStreamCount; ++i) {
    graph.connect(make_source(urls[i], fps[i]), make_decoder(i, fps[i]));
  }

  simaai::neat::Run run = graph.build(run_options());
  for (std::size_t i = 0; i < kStreamCount; ++i) {
    (void)pull_frames(run, output_name(i), "combined stream " + std::to_string(i));
  }
  require_admitted_boundaries(run, kStreamCount, "combined graph");
  std::vector<simaai::neat::Run*> run_refs(kStreamCount, &run);
  std::vector<std::string> endpoints;
  for (std::size_t i = 0; i < kStreamCount; ++i) {
    endpoints.push_back(output_name(i));
  }
  (void)measure_throughput(run_refs, endpoints, fps, kWarmupWindow, "combined warmup", false);
  const double aggregate =
      measure_throughput(run_refs, endpoints, fps, kThroughputWindow, "combined", true);
  run.close();
  return aggregate;
}

double run_independent_graphs(const std::vector<std::string>& urls, const std::vector<int>& fps) {
  std::vector<simaai::neat::Run> runs;
  runs.reserve(kStreamCount);
  for (std::size_t i = 0; i < kStreamCount; ++i) {
    simaai::neat::Graph graph = make_single_stream_graph(urls[i], fps[i]);
    runs.push_back(graph.build(run_options()));
  }

  simaai::neat::Sample branch_sample;
  for (std::size_t i = 0; i < kStreamCount; ++i) {
    simaai::neat::Sample sample =
        pull_frames(runs[i], {}, "independent stream " + std::to_string(i));
    if (i == 0U) {
      branch_sample = std::move(sample);
    }
    require_admitted_boundaries(runs[i], 1U, "independent graph " + std::to_string(i));
  }
  std::vector<simaai::neat::Run*> run_refs;
  std::vector<std::string> endpoints(kStreamCount);
  for (auto& run : runs) {
    run_refs.push_back(&run);
  }
  (void)measure_throughput(run_refs, endpoints, fps, kWarmupWindow, "independent warmup", false);
  const double aggregate =
      measure_throughput(run_refs, endpoints, fps, kThroughputWindow, "independent", true);
  run_appsrc_branch_boundary(branch_sample);
  for (auto& run : runs) {
    run.close();
  }
  return aggregate;
}

void run_close_and_rebuild(const std::string& url, int fps) {
  simaai::neat::Graph graph = make_single_stream_graph(url, fps);
  simaai::neat::Run run = graph.build(run_options());
  (void)pull_frames(run, {}, "simple rebuild");
  require_admitted_boundaries(run, 1U, "simple rebuild");
  run.close();
}

} // namespace

int main() {
  try {
    setenv("SIMA_GST_RUN_INSERT_BOUNDARIES", "1", 1);
    setenv("SIMA_GST_BOUNDARY_PROBES", "1", 1);

    std::vector<std::string> urls = configured_urls();
    if (urls.size() < kStreamCount) {
      std::cout << "[SKIP] SIMANEAT_TEST_RTSP_H264_URLS must provide at least four URLs\n";
      return 77;
    }
    urls.resize(kStreamCount);

    std::vector<int> fps;
    fps.reserve(kStreamCount);
    for (const auto& url : urls) {
      const int source_fps = sima_test::probe_rtsp_source_fps(url);
      require(source_fps > 0, "decoder admission performance test could not probe source FPS");
      fps.push_back(source_fps);
    }

    const double combined_fps = run_combined_graph(urls, fps);
    const double independent_fps = run_independent_graphs(urls, fps);
    const double relative_delta =
        std::abs(independent_fps - combined_fps) / std::max(independent_fps, combined_fps);
    require(relative_delta <= 0.05,
            "independent .add() throughput differs from combined .connect() throughput by more "
            "than 5%: independent=" +
                std::to_string(independent_fps) + " combined=" + std::to_string(combined_fps));
    run_close_and_rebuild(urls.front(), fps.front());
    std::cout << "[OK] decoder_multirun_admission_boundary_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
