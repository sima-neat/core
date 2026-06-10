#include "model/Model.h"

#include "e2e_pipelines/obj_detection/yolov8_test_utils.h"
#include "test_utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kWarmupSamples = 50;
constexpr int kBenchmarkSamples = 100;
constexpr int kTimeoutMs = 120000;

std::size_t dtype_bytes(simaai::neat::TensorDType dtype) {
  using simaai::neat::TensorDType;
  switch (dtype) {
  case TensorDType::UInt8:
  case TensorDType::Int8:
    return 1U;
  case TensorDType::UInt16:
  case TensorDType::Int16:
  case TensorDType::BFloat16:
    return 2U;
  case TensorDType::Int32:
  case TensorDType::Float32:
    return 4U;
  case TensorDType::Float64:
    return 8U;
  }
  return 0U;
}

std::vector<int64_t> contiguous_strides(const std::vector<int64_t>& shape, std::size_t elem) {
  std::vector<int64_t> strides(shape.size(), 0);
  int64_t stride = static_cast<int64_t>(elem);
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides[static_cast<std::size_t>(i)] = stride;
    stride *= shape[static_cast<std::size_t>(i)];
  }
  return strides;
}

simaai::neat::Tensor make_synthetic_tensor(const simaai::neat::TensorSpec& spec,
                                           std::size_t input_index) {
  using namespace simaai::neat;

  require(!spec.shape.empty(), "synthetic model input: empty input spec shape");
  require(!spec.dtypes.empty(), "synthetic model input: empty input spec dtype list");
  require(spec.required_segments.empty() && spec.required_segment_names.empty(),
          "synthetic model input: segmented input specs are not supported");
  require(spec.image_format != ImageSpec::PixelFormat::NV12 &&
              spec.image_format != ImageSpec::PixelFormat::I420,
          "synthetic model input: planar input specs are not supported");

  const TensorDType dtype = spec.dtypes.front();
  const std::size_t elem = dtype_bytes(dtype);
  require(elem > 0U, "synthetic model input: unsupported dtype");

  std::vector<int64_t> shape;
  shape.reserve(spec.shape.size());
  std::uint64_t element_count = 1U;
  for (const int64_t dim : spec.shape) {
    require(dim > 0, "synthetic model input: non-concrete input dimension");
    shape.push_back(dim);
    const auto udim = static_cast<std::uint64_t>(dim);
    require(element_count <= std::numeric_limits<std::uint64_t>::max() / udim,
            "synthetic model input: shape is too large");
    element_count *= udim;
  }
  require(element_count <= std::numeric_limits<std::uint64_t>::max() / elem,
          "synthetic model input: byte size is too large");
  const auto bytes64 = element_count * static_cast<std::uint64_t>(elem);
  require(bytes64 <= static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()),
          "synthetic model input: byte size does not fit size_t");
  const std::size_t bytes = static_cast<std::size_t>(bytes64);

  auto storage = make_cpu_owned_storage(bytes);
  auto map = storage->map(MapMode::Write);
  require(map.data || bytes == 0U, "synthetic model input: failed to map storage");
  auto* out = static_cast<std::uint8_t*>(map.data);
  for (std::size_t i = 0; i < bytes; ++i) {
    out[i] = static_cast<std::uint8_t>((i + input_index * 17U) & 0xffU);
  }

  Tensor tensor;
  tensor.storage = std::move(storage);
  tensor.dtype = dtype;
  tensor.shape = std::move(shape);
  tensor.strides_bytes = contiguous_strides(tensor.shape, elem);
  tensor.device = {DeviceType::CPU, 0};
  tensor.read_only = true;
  if (spec.image_format.has_value()) {
    tensor.semantic.image = ImageSpec{*spec.image_format, ""};
    if (tensor.shape.size() == 2U) {
      tensor.layout = TensorLayout::HW;
      tensor.axis_semantics = {TensorAxisSemantic::H, TensorAxisSemantic::W};
    } else if (tensor.shape.size() == 3U) {
      tensor.layout = TensorLayout::HWC;
      tensor.axis_semantics = {TensorAxisSemantic::H, TensorAxisSemantic::W, TensorAxisSemantic::C};
    }
  }
  return tensor.cvu();
}

simaai::neat::TensorList make_synthetic_inputs(const simaai::neat::Model& model) {
  const std::vector<simaai::neat::TensorSpec> specs = model.input_specs();
  require(!specs.empty(), "synthetic model input: model.input_specs() returned no inputs");
  simaai::neat::TensorList inputs;
  inputs.reserve(specs.size());
  for (std::size_t i = 0; i < specs.size(); ++i) {
    inputs.push_back(make_synthetic_tensor(specs[i], i));
  }
  return inputs;
}

simaai::neat::Sample make_keyed_sample(const simaai::neat::TensorList& inputs, int frame_id,
                                       const std::string& stream_id) {
  simaai::neat::Sample sample = simaai::neat::sample_from_tensors(inputs);
  sample.frame_id = frame_id;
  sample.stream_id = stream_id;
  return sample;
}

simaai::neat::RunOptions make_model_run_options(bool include_power) {
  simaai::neat::RunOptions run_options;
  run_options.startup_preflight = false;
  if (include_power) {
    run_options.enable_board_power();
  }
  return run_options;
}

simaai::neat::MeasureOptions make_model_measure_options(const simaai::neat::Model& model,
                                                        const char* title) {
  simaai::neat::MeasureOptions opt;
  opt.duration_ms = kTimeoutMs;
  opt.warmup_ms = 0;
  opt.timeout_ms = kTimeoutMs;
  opt.include_plugin_latency = false;
  opt.include_edge_latency = false;
  opt.include_message_latency = false;
  opt.include_power = false;
  opt.logical_batch_size = model.compiled_batch_size();
  opt.title = title;
  opt.input = "synthetic";
  return opt;
}

void require_nonempty_output(const simaai::neat::Sample& out, const std::string& where) {
  require(!out.empty(), where + ": pull timed out or returned EOS");
  require(!simaai::neat::tensors_from_sample(out, true).empty(),
          where + ": output should contain tensors");
}

struct MeasuredModelRun {
  simaai::neat::MeasureReport report;
  double inverse_throughput_ms = 0.0;
};

MeasuredModelRun measure_model_single_flight(simaai::neat::Model& model,
                                             const simaai::neat::TensorList& inputs, int samples) {
  simaai::neat::Model::Runner runner =
      model.build(inputs, simaai::neat::Model::RouteOptions{}, make_model_run_options(false));
  for (int i = 0; i < kWarmupSamples; ++i) {
    require(runner.push(make_keyed_sample(inputs, 10'000 + i, "model-single-warmup")),
            "model single-flight warmup push failed");
    require_nonempty_output(runner.pull(kTimeoutMs), "model single-flight warmup");
  }

  simaai::neat::MeasureScope scope =
      runner.start_measurement(make_model_measure_options(model, "model single-flight"));
  for (int i = 0; i < samples; ++i) {
    require(runner.push(make_keyed_sample(inputs, i, "model-single-flight")),
            "model single-flight measured push failed");
    require_nonempty_output(runner.pull(kTimeoutMs), "model single-flight measured");
  }
  MeasuredModelRun measured;
  measured.report = scope.stop();
  runner.close();
  if (measured.report.throughput_batches_per_s > 0.0) {
    measured.inverse_throughput_ms = 1000.0 / measured.report.throughput_batches_per_s;
  }
  return measured;
}

MeasuredModelRun measure_model_burst(simaai::neat::Model& model,
                                     const simaai::neat::TensorList& inputs, int samples) {
  simaai::neat::Model::Runner runner =
      model.build(inputs, simaai::neat::Model::RouteOptions{}, make_model_run_options(false));
  for (int i = 0; i < kWarmupSamples; ++i) {
    require(runner.push(make_keyed_sample(inputs, 20'000 + i, "model-burst-warmup")),
            "model burst warmup push failed");
    require_nonempty_output(runner.pull(kTimeoutMs), "model burst warmup");
  }

  int pulled = 0;
  std::exception_ptr pull_error;
  simaai::neat::MeasureScope scope =
      runner.start_measurement(make_model_measure_options(model, "model burst"));
  std::thread pull_thread([&] {
    try {
      while (pulled < samples) {
        require_nonempty_output(runner.pull(kTimeoutMs), "model burst measured");
        ++pulled;
      }
    } catch (...) {
      pull_error = std::current_exception();
    }
  });

  try {
    for (int i = 0; i < samples; ++i) {
      require(runner.push(make_keyed_sample(inputs, i, "model-burst")),
              "model burst measured push failed");
    }
    runner.close_input();
    pull_thread.join();
  } catch (...) {
    runner.close_input();
    if (pull_thread.joinable()) {
      pull_thread.join();
    }
    runner.close();
    throw;
  }
  if (pull_error) {
    runner.close();
    std::rethrow_exception(pull_error);
  }

  MeasuredModelRun measured;
  measured.report = scope.stop();
  runner.close();
  if (measured.report.throughput_batches_per_s > 0.0) {
    measured.inverse_throughput_ms = 1000.0 / measured.report.throughput_batches_per_s;
  }
  return measured;
}

void require_measurement_report_is_complete(const simaai::neat::MeasureReport& report,
                                            int expected_samples, const std::string& name) {
  require(report.outputs == static_cast<std::size_t>(expected_samples),
          name + ": measured output count mismatch");
  require(report.counters.outputs_pulled == static_cast<std::uint64_t>(expected_samples),
          name + ": outputs_pulled count mismatch");
  require(report.end_to_end.count == static_cast<std::size_t>(expected_samples),
          name + ": expected one end-to-end latency sample per output, got " +
              std::to_string(report.end_to_end.count));
  require(report.graph_sample_timing_unkeyed == 0,
          name + ": keyed Model-path samples should not be unkeyed");
  require(report.graph_sample_timing_misses == 0,
          name + ": keyed Model-path samples should not miss graph timing correlation");
  require(report.latency_samples_collected, name + ": latency samples should be collected");
  require(std::isfinite(report.end_to_end.avg_ms) && report.end_to_end.avg_ms > 0.0,
          name + ": latency should be finite and positive");
  require(std::isfinite(report.throughput_batches_per_s) && report.throughput_batches_per_s > 0.0,
          name + ": batch throughput should be finite and positive");
  require(std::isfinite(report.throughput_inferences_per_s) &&
              report.throughput_inferences_per_s > 0.0,
          name + ": inference throughput should be finite and positive");
}

} // namespace

int main(int argc, char** argv) {
  try {
    const fs::path root = (argc > 1) ? fs::path(argv[1]) : fs::current_path();
    std::error_code ec;
    fs::create_directories(root / "tmp", ec);
    fs::current_path(root, ec);

    const std::string tar_gz = sima_yolov8_test::resolve_yolov8s_tar_or_skip(root);
    simaai::neat::Model model(tar_gz);

    const auto specs = model.input_specs();
    require(!specs.empty(), "yolov8 benchmark: model.input_specs() returned no inputs");
    const int compiled_batch_size = model.compiled_batch_size();
    require(compiled_batch_size > 0, "yolov8 benchmark: compiled batch size must be positive");
    std::cout << "[e2e_benchmark] compiled_batch_size=" << compiled_batch_size << "\n";

    const simaai::neat::TensorList synthetic_inputs = make_synthetic_inputs(model);

    const simaai::neat::BenchmarkReport report = model.benchmark(kBenchmarkSamples);
    std::cout << "[e2e_benchmark] latency_ms=" << report.latency_ms << "\n";
    std::cout << "[e2e_benchmark] fps=" << report.fps << "\n";
    std::cout << "[e2e_benchmark] avg_power_watts=" << report.avg_power_watts << "\n";
    std::cout << "[e2e_benchmark] energy_joules=" << report.energy_joules << "\n";

    require(std::isfinite(report.latency_ms), "yolov8 benchmark: latency must be finite");
    require(std::isfinite(report.fps), "yolov8 benchmark: FPS must be finite");
    require(std::isfinite(report.avg_power_watts), "yolov8 benchmark: power must be finite");
    require(std::isfinite(report.energy_joules), "yolov8 benchmark: energy must be finite");
    require(report.latency_ms > 0.0, "yolov8 benchmark: latency must be positive");
    require(report.fps > 0.0, "yolov8 benchmark: FPS must be positive");
    require(report.avg_power_watts >= 0.0, "yolov8 benchmark: power must be non-negative");
    require(report.energy_joules >= 0.0, "yolov8 benchmark: energy must be non-negative");

    const MeasuredModelRun single =
        measure_model_single_flight(model, synthetic_inputs, kBenchmarkSamples);
    std::cout << "[e2e_benchmark_single] latency_ms=" << single.report.end_to_end.avg_ms
              << " tput_batches_s=" << single.report.throughput_batches_per_s
              << " tput_inferences_s=" << single.report.throughput_inferences_per_s
              << " inverse_tput_ms=" << single.inverse_throughput_ms
              << " e2e_count=" << single.report.end_to_end.count << "\n";
    require_measurement_report_is_complete(single.report, kBenchmarkSamples,
                                           "yolov8 single-flight Model measurement");
    require(std::abs(single.inverse_throughput_ms - single.report.end_to_end.avg_ms) <= 1.0,
            "yolov8 single-flight Model measurement should have inverse throughput within 1 ms "
            "of e2e latency; inverse_tput_ms=" +
                std::to_string(single.inverse_throughput_ms) +
                " latency_ms=" + std::to_string(single.report.end_to_end.avg_ms));

    const MeasuredModelRun burst = measure_model_burst(model, synthetic_inputs, kBenchmarkSamples);
    std::cout << "[e2e_benchmark_burst] queue_residency_ms=" << burst.report.end_to_end.avg_ms
              << " tput_batches_s=" << burst.report.throughput_batches_per_s
              << " tput_inferences_s=" << burst.report.throughput_inferences_per_s
              << " inverse_tput_ms=" << burst.inverse_throughput_ms
              << " e2e_count=" << burst.report.end_to_end.count << "\n";
    require_measurement_report_is_complete(burst.report, kBenchmarkSamples,
                                           "yolov8 burst Model measurement");
    require(burst.report.end_to_end.avg_ms > burst.inverse_throughput_ms,
            "yolov8 burst Model measurement should expose pipelined throughput separately from "
            "per-frame latency");

    const double benchmark_latency_delta =
        std::abs(report.latency_ms - single.report.end_to_end.avg_ms);
    require(benchmark_latency_delta <= 1.0,
            "Model::benchmark latency should report warmed-up single-flight latency, not the "
            "queueing-inflated latency from its throughput burst; benchmark_latency_ms=" +
                std::to_string(report.latency_ms) +
                " single_flight_latency_ms=" + std::to_string(single.report.end_to_end.avg_ms) +
                " burst_queue_residency_ms=" + std::to_string(burst.report.end_to_end.avg_ms) +
                " delta_ms=" + std::to_string(benchmark_latency_delta));

    const double expected_benchmark_fps = burst.report.throughput_inferences_per_s;
    const double fps_delta = std::abs(report.fps - expected_benchmark_fps);
    const double fps_tolerance = std::max(10.0, expected_benchmark_fps * 0.25);
    require(fps_delta <= fps_tolerance,
            "Model::benchmark FPS should track logical inference throughput from the same "
            "Model-path burst measurement; benchmark_fps=" +
                std::to_string(report.fps) +
                " measured_inferences_s=" + std::to_string(expected_benchmark_fps) +
                " measured_batches_s=" + std::to_string(burst.report.throughput_batches_per_s));
    if (compiled_batch_size == 1) {
      std::cout << "[e2e_benchmark] note: this YOLO fixture is batch=1, so it cannot expose a "
                   "batches/s-vs-inferences/s FPS bug by numeric comparison.\n";
    }

    std::cout << "[OK] yolov8_model_benchmark_metrics_test passed\n";
    return 0;
  } catch (const SkipTest& e) {
    std::cout << "[SKIP] " << e.what() << "\n";
    return skip_long_test(e.what());
  } catch (const std::exception& e) {
    if (is_dispatcher_unavailable(e.what())) {
      return skip_long_test("dispatcher unavailable");
    }
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
}
