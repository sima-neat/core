#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif
#pragma once

#include "asset_utils.h"
#include "gst/GstInit.h"
#include "nodes/common/FileInput.h"
#include "nodes/common/JpegParse.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "nodes/sima/H264Parse.h"
#include "nodes/sima/SimaDecode.h"
#include "perf_metrics_common.h"
#include "pipeline/EncodedSampleUtil.h"
#include "pipeline/Graph.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace sima_codec_perf {

struct EncodedFrame {
  std::vector<std::uint8_t> bytes;
  std::string caps;
};

struct CodecPerfConfig {
  std::string scenario_id;
  std::string run_mode;
  simaai::neat::SimaDecodeType decode_type = simaai::neat::SimaDecodeType::H264;
  int width = 1280;
  int height = 720;
  int fps = 30;
};

struct CodecPerfPhaseResult {
  sima_perf::PerfMetrics metrics;
  simaai::neat::MeasureReport report;
};

inline simaai::neat::Sample pull_or_throw(simaai::neat::Run& run, int timeout_ms,
                                          const std::string& context) {
  simaai::neat::Sample sample;
  simaai::neat::PullError error;
  const simaai::neat::PullStatus status = run.pull(timeout_ms, sample, &error);
  if (status == simaai::neat::PullStatus::Ok) {
    return sample;
  }
  if (status == simaai::neat::PullStatus::Timeout) {
    throw std::runtime_error(context + ": timed out");
  }
  if (status == simaai::neat::PullStatus::Closed) {
    throw std::runtime_error(context + ": closed before output");
  }
  std::string message = error.message.empty() ? (context + ": pull failed") : error.message;
  if (!error.code.empty()) {
    message += " (code=" + error.code + ")";
  }
  throw std::runtime_error(message);
}

inline void require_decoded_sample(const simaai::neat::Sample& sample,
                                   const std::string& scenario_id,
                                   bool require_zero_copy_output = false) {
  if (simaai::neat::sample_payload_type(sample) != simaai::neat::PayloadType::Image) {
    throw std::runtime_error(scenario_id + ": expected decoded image output");
  }
  const simaai::neat::TensorList tensors = simaai::neat::tensors_from_sample(sample, true);
  if (tensors.size() != 1U) {
    throw std::runtime_error(scenario_id + ": expected one decoded tensor");
  }
  const simaai::neat::Tensor& tensor = tensors.front();
  if (tensor.storage == nullptr) {
    throw std::runtime_error(scenario_id + ": decoded tensor missing storage");
  }
  if (!require_zero_copy_output) {
    return;
  }
  if (sample.owned) {
    throw std::runtime_error(scenario_id + ": decoded output was copied into owned memory");
  }
  if (tensor.storage->kind != simaai::neat::StorageKind::GstSample) {
    throw std::runtime_error(scenario_id + ": decoded output is not GstSample-backed");
  }
}

inline std::vector<std::uint8_t> make_jpeg_bytes(int width, int height) {
  cv::Mat image(height, width, CV_8UC3);
  for (int y = 0; y < height; ++y) {
    auto* row = image.ptr<cv::Vec3b>(y);
    for (int x = 0; x < width; ++x) {
      row[x] = cv::Vec3b(static_cast<std::uint8_t>((x + y) & 0xFF),
                         static_cast<std::uint8_t>((2 * x + y) & 0xFF),
                         static_cast<std::uint8_t>((x + 2 * y) & 0xFF));
    }
  }

  std::vector<std::uint8_t> encoded;
  const std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 90};
  if (!cv::imencode(".jpg", image, encoded, params) || encoded.empty()) {
    throw std::runtime_error("failed to encode MJPEG perf fixture frame");
  }
  return encoded;
}

inline std::vector<EncodedFrame> make_mjpeg_frames(const CodecPerfConfig& config) {
  return {{
      make_jpeg_bytes(config.width, config.height),
      "image/jpeg,width=(int)" + std::to_string(config.width) + ",height=(int)" +
          std::to_string(config.height) + ",framerate=(fraction)" + std::to_string(config.fps) +
          "/1",
  }};
}

inline std::filesystem::path h264_fixture_path() {
  namespace fs = std::filesystem;
  if (const char* env = std::getenv("SIMA_TEST_CODEC_PERF_H264_FIXTURE"); env && *env) {
    return fs::path(env);
  }
  return sima_test::test_codec_perf_h264_fixture_path();
}

inline std::filesystem::path h265_fixture_path() {
  namespace fs = std::filesystem;
  if (const char* env = std::getenv("SIMA_TEST_CODEC_PERF_H265_FIXTURE"); env && *env) {
    return fs::path(env);
  }
  return sima_test::test_codec_perf_h265_fixture_path();
}

inline std::string h265_parser_fragment() {
  return "h265parse disable-passthrough=true config-interval=-1 ! capsfilter "
         "caps=\"video/x-h265,parsed=true,stream-format=(string)byte-stream,"
         "alignment=(string)au\"";
}

inline std::vector<EncodedFrame> extract_video_access_units(simaai::neat::SimaDecodeType type,
                                                            int max_frames) {
  namespace fs = std::filesystem;
  const bool is_h264 = type == simaai::neat::SimaDecodeType::H264;
  const fs::path fixture = is_h264 ? h264_fixture_path() : h265_fixture_path();
  const std::string codec = is_h264 ? "h264" : "h265";
  if (!fs::exists(fixture)) {
    throw std::runtime_error("missing " + codec + " decoder fixture: " + fixture.string());
  }

  simaai::neat::Graph graph(codec + "-access-unit-extract");
  graph.add(simaai::neat::nodes::FileInput(fixture.string()));

  if (is_h264) {
    simaai::neat::H264ParseOptions parse;
    parse.config_interval = 1;
    parse.enforce_caps = true;
    parse.alignment = simaai::neat::H264ParseOptions::Alignment::AU;
    parse.stream_format = simaai::neat::H264ParseOptions::StreamFormat::ByteStream;
    graph.add(simaai::neat::nodes::H264Parse(parse));
  } else {
    graph.add(simaai::neat::nodes::Custom(h265_parser_fragment()));
  }
  graph.add(simaai::neat::nodes::Output(simaai::neat::OutputOptions::EveryFrame(max_frames + 8)));

  simaai::neat::RunOptions run_options;
  run_options.output_memory = simaai::neat::OutputMemory::Owned;
  simaai::neat::Run run = graph.build(run_options);

  std::vector<EncodedFrame> frames;
  frames.reserve(static_cast<std::size_t>(max_frames));
  while (static_cast<int>(frames.size()) < max_frames) {
    simaai::neat::Sample sample;
    simaai::neat::PullError error;
    const simaai::neat::PullStatus status = run.pull(5000, sample, &error);
    if (status == simaai::neat::PullStatus::Closed) {
      break;
    }
    if (status == simaai::neat::PullStatus::Timeout) {
      throw std::runtime_error(codec + " fixture extraction timed out");
    }
    if (status == simaai::neat::PullStatus::Error) {
      throw std::runtime_error(error.message.empty() ? codec + " fixture extraction failed"
                                                     : error.message);
    }
    if (simaai::neat::sample_payload_type(sample) != simaai::neat::PayloadType::Encoded) {
      throw std::runtime_error(codec + " fixture extraction produced non-encoded sample");
    }
    const simaai::neat::TensorList tensors = simaai::neat::tensors_from_sample(sample, true);
    if (tensors.size() != 1U) {
      throw std::runtime_error(codec + " fixture extraction expected one tensor");
    }
    std::vector<std::uint8_t> bytes = tensors.front().copy_payload_bytes();
    if (bytes.empty()) {
      throw std::runtime_error(codec + " fixture extraction produced empty AU");
    }
    std::string caps = sample.caps_string;
    if (caps.empty()) {
      caps = is_h264 ? "video/x-h264,parsed=(boolean)true,stream-format=(string)byte-stream,"
                       "alignment=(string)au"
                     : "video/x-h265,parsed=(boolean)true,stream-format=(string)byte-stream,"
                       "alignment=(string)au";
    }
    frames.push_back({std::move(bytes), std::move(caps)});
  }

  run.stop();
  if (frames.empty()) {
    throw std::runtime_error(codec + " fixture extraction produced no access units");
  }
  return frames;
}

inline std::vector<EncodedFrame> extract_h264_access_units(int max_frames) {
  return extract_video_access_units(simaai::neat::SimaDecodeType::H264, max_frames);
}

inline std::vector<EncodedFrame> extract_h265_access_units(int max_frames) {
  return extract_video_access_units(simaai::neat::SimaDecodeType::H265, max_frames);
}

inline simaai::neat::Sample make_timed_encoded_sample(const EncodedFrame& frame,
                                                      const std::string& stream_id,
                                                      int64_t frame_id, int fps) {
  const int64_t duration_ns = 1000000000LL / std::max(1, fps);
  simaai::neat::Sample sample = simaai::neat::make_encoded_sample(
      frame.bytes, frame.caps, frame_id * duration_ns, frame_id * duration_ns, duration_ns);
  sample.frame_id = frame_id;
  sample.stream_id = stream_id;
  sample.stream_label = stream_id;
  return sample;
}

inline std::vector<simaai::neat::Sample>
make_sample_sequence(const std::vector<EncodedFrame>& frames, const CodecPerfConfig& config,
                     int count, int64_t first_frame_id = 0) {
  std::vector<simaai::neat::Sample> samples;
  samples.reserve(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    const int64_t frame_id = first_frame_id + i;
    samples.push_back(make_timed_encoded_sample(frames[static_cast<std::size_t>(i) % frames.size()],
                                                config.scenario_id, frame_id, config.fps));
  }
  return samples;
}

inline simaai::neat::Graph make_decode_graph(const CodecPerfConfig& config,
                                             const simaai::neat::Sample& seed, int max_buffers,
                                             bool live_input = false,
                                             bool decoder_zero_copy_output = false) {
  simaai::neat::Graph graph(config.scenario_id);

  simaai::neat::InputOptions input;
  input.payload_type = simaai::neat::PayloadType::Encoded;
  input.caps_override = seed.caps_string;
  input.is_live = live_input;
  input.do_timestamp = live_input;
  input.block = true;
  input.pool_max_buffers = std::max(2, max_buffers);
  input.memory_policy = simaai::neat::InputMemoryPolicy::SystemMemory;
  graph.add(simaai::neat::nodes::Input(input));

  if (config.decode_type == simaai::neat::SimaDecodeType::H264) {
    graph.custom("identity name=h264_segment silent=true single-segment=true");

    simaai::neat::H264ParseOptions parse;
    parse.config_interval = 1;
    parse.enforce_caps = true;
    parse.alignment = simaai::neat::H264ParseOptions::Alignment::AU;
    parse.stream_format = simaai::neat::H264ParseOptions::StreamFormat::ByteStream;
    graph.add(simaai::neat::nodes::H264Parse(parse));
  } else if (config.decode_type == simaai::neat::SimaDecodeType::H265) {
    graph.add(simaai::neat::nodes::Custom(h265_parser_fragment()));
  } else {
    graph.add(simaai::neat::nodes::JpegParse());
  }

  if (decoder_zero_copy_output) {
    const char* decoder_type = config.decode_type == simaai::neat::SimaDecodeType::H265    ? "h265"
                               : config.decode_type == simaai::neat::SimaDecodeType::MJPEG ? "mjpeg"
                                                                                           : "h264";
    std::ostringstream decoder;
    decoder << "neatdecoder name=codec_perf_decoder sima-allocator-type=2 dec-type=" << decoder_type
            << " dec-fmt=NV12 dec-width=" << config.width << " dec-height=" << config.height
            << " dec-fps=" << config.fps
            << " dec-ip-cnt=8 num-buffers=64 zero-copy-output=true ! capsfilter "
               "name=codec_perf_output_caps caps=\"video/x-raw,format=(string)NV12,width=(int)"
            << config.width << ",height=(int)" << config.height << "\"";
    graph.add(simaai::neat::nodes::Custom(decoder.str()));
  } else {
    simaai::neat::SimaDecodeOptions decode;
    decode.type = config.decode_type;
    decode.out_format = simaai::neat::FormatTag::NV12;
    decode.raw_output = false;
    decode.dec_width = config.width;
    decode.dec_height = config.height;
    decode.dec_fps = config.fps;
    decode.num_buffers = 64;
    graph.add(simaai::neat::nodes::SimaDecode(decode));
  }
  graph.add(simaai::neat::nodes::Output(simaai::neat::OutputOptions::EveryFrame(max_buffers)));
  return graph;
}

inline sima_perf::Clock::time_point
push_and_pull_burst(simaai::neat::Run& run, const std::vector<simaai::neat::Sample>& samples,
                    int expected_outputs, bool close_input, const std::string& context,
                    bool require_zero_copy_output) {
  if (expected_outputs == 0) {
    return sima_perf::Clock::now();
  }

  std::exception_ptr push_error;
  std::thread producer([&] {
    try {
      for (std::size_t pushed = 0; pushed < samples.size(); ++pushed) {
        if (!run.push(simaai::neat::Sample{samples[pushed]})) {
          throw std::runtime_error(context + ": push failed");
        }
      }
      if (close_input) {
        run.close_input();
      }
    } catch (...) {
      push_error = std::current_exception();
      try {
        run.close_input();
      } catch (...) {
      }
    }
  });

  std::exception_ptr pull_error;
  sima_perf::Clock::time_point outputs_done;
  try {
    int64_t first_output_frame_id = -1;
    for (int pulled = 0; pulled < expected_outputs; ++pulled) {
      simaai::neat::Sample out = pull_or_throw(run, 5000, context + ": pull");
      require_decoded_sample(out, context, require_zero_copy_output);
      if (pulled == 0) {
        first_output_frame_id = out.frame_id;
      }
      const int64_t expected_frame_id = first_output_frame_id + pulled;
      if (first_output_frame_id < 0 || out.frame_id != expected_frame_id) {
        throw std::runtime_error(context + ": decoded output frame sequence mismatch at output " +
                                 std::to_string(pulled) +
                                 " (expected=" + std::to_string(expected_frame_id) +
                                 ", actual=" + std::to_string(out.frame_id) + ")");
      }
    }
    outputs_done = sima_perf::Clock::now();
  } catch (...) {
    pull_error = std::current_exception();
    try {
      run.close_input();
    } catch (...) {
    }
  }

  if (producer.joinable()) {
    producer.join();
  }
  if (push_error) {
    std::rethrow_exception(push_error);
  }
  if (pull_error) {
    std::rethrow_exception(pull_error);
  }
  return outputs_done;
}

inline sima_perf::PerfMetrics
measure_decode_throughput(simaai::neat::Run& run, const std::vector<simaai::neat::Sample>& samples,
                          int expected_outputs, bool close_input,
                          bool require_zero_copy_output = false) {
  const auto run_t0 = sima_perf::Clock::now();
  const auto run_t1 = push_and_pull_burst(run, samples, expected_outputs, close_input,
                                          "codec perf throughput", require_zero_copy_output);

  sima_perf::PerfMetrics metrics;
  const double seconds = sima_perf::elapsed_seconds(run_t0, run_t1);
  metrics.throughput = seconds > 0.0 ? static_cast<double>(expected_outputs) / seconds : 0.0;
  return metrics;
}

inline sima_perf::PerfMetrics
measure_decode_residency(simaai::neat::Run& run, const std::vector<simaai::neat::Sample>& samples,
                         std::vector<double>& latencies_ms, bool close_input = true,
                         bool require_zero_copy_output = false) {
  latencies_ms.clear();
  latencies_ms.reserve(samples.size());

  const int iterations = static_cast<int>(samples.size());
  const auto run_t0 = sima_perf::Clock::now();
  try {
    int64_t first_output_frame_id = -1;
    for (int i = 0; i < iterations; ++i) {
      const auto frame_t0 = sima_perf::Clock::now();
      if (!run.push(simaai::neat::Sample{samples[static_cast<std::size_t>(i)]})) {
        throw std::runtime_error("codec perf measured push failed");
      }
      simaai::neat::Sample out = pull_or_throw(run, 5000, "codec perf measured pull");
      require_decoded_sample(out, "codec perf", require_zero_copy_output);
      if (i == 0) {
        first_output_frame_id = out.frame_id;
      }
      const int64_t expected_frame_id = first_output_frame_id + i;
      if (first_output_frame_id < 0 || out.frame_id != expected_frame_id) {
        throw std::runtime_error("codec perf: decoded output frame sequence mismatch at output " +
                                 std::to_string(i) +
                                 " (expected=" + std::to_string(expected_frame_id) +
                                 ", actual=" + std::to_string(out.frame_id) + ")");
      }
      latencies_ms.push_back(sima_perf::elapsed_ms(frame_t0, sima_perf::Clock::now()));
    }
    if (close_input) {
      run.close_input();
    }
  } catch (...) {
    try {
      run.close_input();
    } catch (...) {
    }
    throw;
  }
  const auto run_t1 = sima_perf::Clock::now();

  sima_perf::PerfMetrics metrics;
  const double seconds = sima_perf::elapsed_seconds(run_t0, run_t1);
  metrics.throughput = seconds > 0.0 ? static_cast<double>(iterations) / seconds : 0.0;
  metrics.p50 = sima_perf::percentile(latencies_ms, 50.0);
  metrics.p95 = sima_perf::percentile(latencies_ms, 95.0);
  return metrics;
}

inline simaai::neat::RunOptions codec_run_options(int queue_depth, bool zero_copy_output = false) {
  simaai::neat::RunOptions run_options;
  run_options.overflow_policy = simaai::neat::OverflowPolicy::Block;
  run_options.queue_depth = std::max(queue_depth, 8);
  run_options.output_memory =
      zero_copy_output ? simaai::neat::OutputMemory::ZeroCopy : simaai::neat::OutputMemory::Owned;
  if (zero_copy_output) {
    run_options.advanced.copy_input = false;
  }
  run_options.startup_preflight = false;
  return run_options;
}

inline CodecPerfPhaseResult
run_throughput_phase(const CodecPerfConfig& config, const simaai::neat::Sample& seed,
                     const std::vector<simaai::neat::Sample>& warmup, int warmup_outputs,
                     const std::vector<simaai::neat::Sample>& measured, int measured_outputs,
                     int output_buffers, double* startup_ms) {
  if (!warmup.empty()) {
    simaai::neat::Graph warmup_graph = make_decode_graph(config, seed, output_buffers, false, true);
    simaai::neat::Run warmup_run = warmup_graph.build(
        simaai::neat::Sample{seed}, codec_run_options(static_cast<int>(warmup.size()), true));
    const bool close_input = config.decode_type == simaai::neat::SimaDecodeType::MJPEG;
    push_and_pull_burst(warmup_run, warmup, warmup_outputs, close_input,
                        "codec perf throughput warmup", true);
    warmup_run.stop();
  }

  simaai::neat::Graph graph = make_decode_graph(config, seed, output_buffers, false, true);
  const auto startup_t0 = sima_perf::Clock::now();
  simaai::neat::Run run = graph.build(
      simaai::neat::Sample{seed},
      codec_run_options(static_cast<int>(std::max(warmup.size(), measured.size())), true));
  const auto startup_t1 = sima_perf::Clock::now();
  if (startup_ms != nullptr) {
    *startup_ms = sima_perf::elapsed_ms(startup_t0, startup_t1);
  }

  simaai::neat::MeasureOptions measure_options;
  measure_options.include_plugin_latency = false;
  measure_options.include_edge_latency = false;
  measure_options.include_power = false;
  auto measure_scope = run.start_measurement(measure_options);
  const bool close_input = config.decode_type == simaai::neat::SimaDecodeType::MJPEG;
  sima_perf::PerfMetrics metrics =
      measure_decode_throughput(run, measured, measured_outputs, close_input, true);
  const simaai::neat::MeasureReport report = measure_scope.stop();
  run.stop();
  return {.metrics = metrics, .report = report};
}

inline CodecPerfPhaseResult run_residency_phase(const CodecPerfConfig& config,
                                                const simaai::neat::Sample& seed,
                                                const std::vector<simaai::neat::Sample>& warmup,
                                                const std::vector<simaai::neat::Sample>& measured,
                                                int output_buffers) {
  simaai::neat::Graph graph = make_decode_graph(config, seed, output_buffers, true, true);
  simaai::neat::Run run = graph.build(simaai::neat::Sample{seed}, codec_run_options(8, true));

  std::vector<double> warmup_latencies_ms;
  measure_decode_residency(run, warmup, warmup_latencies_ms, false, true);

  simaai::neat::MeasureOptions measure_options;
  measure_options.include_plugin_latency = false;
  measure_options.include_edge_latency = false;
  measure_options.include_power = false;
  auto measure_scope = run.start_measurement(measure_options);

  std::vector<double> latencies_ms;
  sima_perf::PerfMetrics metrics =
      measure_decode_residency(run, measured, latencies_ms, true, true);
  const simaai::neat::MeasureReport report = measure_scope.stop();
  run.stop();
  return {.metrics = metrics, .report = report};
}

inline void emit_codec_metrics_json(const CodecPerfConfig& config, int iterations,
                                    const sima_perf::PerfMetrics& metrics,
                                    const simaai::neat::MeasureReport& throughput_report,
                                    const simaai::neat::MeasureReport* latency_report) {
  std::cout << std::fixed << std::setprecision(6) << "{\n"
            << "  \"scenario_id\": \"" << config.scenario_id << "\",\n"
            << "  \"iterations\": " << iterations << ",\n"
            << "  \"run_mode\": \"" << config.run_mode << "\",\n"
            << "  \"throughput\": " << metrics.throughput << ",\n"
            << "  \"p50\": " << metrics.p50 << ",\n"
            << "  \"p95\": " << metrics.p95 << ",\n"
            << "  \"startup\": " << metrics.startup << ",\n"
            << "  \"rss_peak_kb\": " << metrics.rss_peak_kb << ",\n"
            << "  \"input_drop_count\": " << metrics.input_drop_count << ",\n"
            << "  \"output_drop_count\": " << metrics.output_drop_count << ",\n"
            << "  \"measurement_modes\": {\n"
            << "    \"throughput\": \"async_burst_push_pull\",\n"
            << "    \"latency\": \""
            << (latency_report != nullptr ? "single_flight_push_pull_residency"
                                          : "unavailable_inter_frame_single_flight")
            << "\"\n"
            << "  },\n"
            << "  \"latency_available\": " << (latency_report != nullptr ? "true" : "false")
            << ",\n"
            << "  \"measure_report\": {\n"
            << "    \"schema\": \"sima.neat.codec_perf_phase_report\",\n"
            << "    \"memory_path\": {\n"
            << "      \"encoded_input\": \"plugin_input_pool_copy\",\n"
            << "      \"core_output\": \"zero_copy_gst_sample\",\n"
            << "      \"daemon_zero_copy_output\": true\n"
            << "    },\n"
            << "    \"throughput\": " << throughput_report.to_json(4);
  if (latency_report != nullptr) {
    std::cout << ",\n    \"latency\": " << latency_report->to_json(4) << "\n";
  } else {
    std::cout << ",\n    \"latency_unavailable_reason\": "
                 "\"buffered inter-frame decode requires multiple in-flight inputs\"\n";
  }
  std::cout << "  }\n"
            << "}\n";
  std::cout.flush();
}

inline int run_codec_decode_perf(const CodecPerfConfig& config,
                                 const std::vector<EncodedFrame>& frames) {
  try {
    simaai::neat::gst_init_once();
    const int iterations = sima_perf::env_int("SIMA_PERF_ITERS", 1000);
    const int warmup_iterations = sima_perf::env_int("SIMA_PERF_CODEC_WARMUP_ITERS", 200, 0);
    if (frames.empty()) {
      throw std::runtime_error(config.scenario_id + ": no encoded fixture frames");
    }

    const int output_buffers = 64;
    // H.26x keeps one or more decoded pictures until a later access unit arrives. Feed an
    // uncounted release window and stop timing at the requested output count.
    const int release_inputs =
        config.decode_type == simaai::neat::SimaDecodeType::MJPEG ? 0 : output_buffers;
    std::vector<simaai::neat::Sample> warmup = make_sample_sequence(
        frames, config, warmup_iterations == 0 ? 0 : warmup_iterations + release_inputs);
    std::vector<simaai::neat::Sample> measured = make_sample_sequence(
        frames, config, iterations + release_inputs, warmup_iterations + release_inputs);
    const simaai::neat::Sample& seed = warmup.empty() ? measured.front() : warmup.front();

    double startup_ms = 0.0;

    const CodecPerfPhaseResult throughput = run_throughput_phase(
        config, seed, warmup, warmup_iterations, measured, iterations, output_buffers, &startup_ms);
    std::optional<CodecPerfPhaseResult> latency;
    if (config.decode_type == simaai::neat::SimaDecodeType::MJPEG) {
      const std::vector<simaai::neat::Sample> latency_warmup(warmup.begin(),
                                                             warmup.begin() + warmup_iterations);
      const std::vector<simaai::neat::Sample> latency_measured(measured.begin(),
                                                               measured.begin() + iterations);
      latency = run_residency_phase(config, seed, latency_warmup, latency_measured, output_buffers);
    }

    sima_perf::PerfMetrics metrics = throughput.metrics;
    if (latency.has_value()) {
      metrics.p50 = latency->metrics.p50;
      metrics.p95 = latency->metrics.p95;
    } else {
      metrics.p50 = 0.0;
      metrics.p95 = 0.0;
    }
    metrics.startup = startup_ms;
    metrics.rss_peak_kb = sima_perf::rss_peak_kb();
    metrics.input_drop_count = throughput.report.counters.inputs_dropped;
    metrics.output_drop_count = throughput.report.counters.outputs_dropped;
    if (latency.has_value()) {
      metrics.input_drop_count += latency->report.counters.inputs_dropped;
      metrics.output_drop_count += latency->report.counters.outputs_dropped;
    }

    emit_codec_metrics_json(config, iterations, metrics, throughput.report,
                            latency.has_value() ? &latency->report : nullptr);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << config.scenario_id << " exception: " << e.what() << "\n";
    return 1;
  }
}

inline int run_mjpeg_decode_perf() {
  const CodecPerfConfig config{.scenario_id = "runtime_codec_mjpeg_decode",
                               .run_mode = "codec_mjpeg_decode",
                               .decode_type = simaai::neat::SimaDecodeType::MJPEG};
  return run_codec_decode_perf(config, make_mjpeg_frames(config));
}

inline int run_h264_decode_perf() {
  const CodecPerfConfig config{.scenario_id = "runtime_codec_h264_decode",
                               .run_mode = "codec_h264_decode",
                               .decode_type = simaai::neat::SimaDecodeType::H264};
  return run_codec_decode_perf(config, extract_h264_access_units(30));
}

inline int run_h265_decode_perf() {
  const CodecPerfConfig config{.scenario_id = "runtime_codec_h265_decode",
                               .run_mode = "codec_h265_decode",
                               .decode_type = simaai::neat::SimaDecodeType::H265};
  return run_codec_decode_perf(config, extract_h265_access_units(30));
}

} // namespace sima_codec_perf
