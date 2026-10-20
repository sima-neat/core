#include "model/Model.h"
#include "model/internal/MlaLayerStats.h"
#include "model/internal/ModelPack.h"
#include "model/internal/ModelFormat.h"
#include "model/internal/ModelInternal.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/internal/TelemetryMonitorInternal.h"
#include "pipeline/internal/TensorMath.h"

#include <glib.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

namespace simaai::neat {
namespace {

using internal::image_format_name;
using internal::preprocess_color_format_name;

Tensor make_synthetic_benchmark_tensor(const TensorSpec& spec, std::size_t input_index) {
  if (spec.shape.empty()) {
    throw std::runtime_error("Model::benchmark: input_specs() entry has empty shape");
  }
  if (spec.dtypes.empty()) {
    throw std::runtime_error("Model::benchmark: input_specs() entry has no dtype");
  }
  if (!spec.required_segments.empty() || !spec.required_segment_names.empty()) {
    throw std::runtime_error(
        "Model::benchmark: segmented input specs are not supported by the synthetic benchmark");
  }
  const TensorDType dtype = spec.dtypes.front();
  const std::size_t elem = pipeline_internal::dtype_bytes(dtype);
  if (elem == 0U) {
    throw std::runtime_error("Model::benchmark: unsupported input dtype");
  }

  std::vector<int64_t> shape;
  shape.reserve(spec.shape.size());
  for (const int64_t dim : spec.shape) {
    if (dim <= 0) {
      throw std::runtime_error(
          "Model::benchmark: input_specs() must provide concrete dimensions for synthetic model "
          "benchmark input");
    }
    shape.push_back(dim);
  }

  if (spec.image_format == ImageSpec::PixelFormat::NV12 ||
      spec.image_format == ImageSpec::PixelFormat::I420) {
    throw std::runtime_error(
        "Model::benchmark: planar image input specs are not supported by the synthetic benchmark");
  }

  std::uint64_t element_count = 1;
  for (const int64_t dim : shape) {
    const auto udim = static_cast<std::uint64_t>(dim);
    if (element_count > std::numeric_limits<std::uint64_t>::max() / udim) {
      throw std::runtime_error("Model::benchmark: synthetic input shape is too large");
    }
    element_count *= udim;
  }
  if (element_count > std::numeric_limits<std::uint64_t>::max() / elem) {
    throw std::runtime_error("Model::benchmark: synthetic input byte size is too large");
  }
  const auto bytes64 = element_count * static_cast<std::uint64_t>(elem);
  if (bytes64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error("Model::benchmark: synthetic input byte size is too large");
  }
  const std::size_t bytes = static_cast<std::size_t>(bytes64);

  auto storage = make_cpu_owned_storage(bytes);
  auto map = storage->map(MapMode::Write);
  if (!map.data && bytes != 0U) {
    throw std::runtime_error("Model::benchmark: failed to map synthetic input storage");
  }
  auto* out = static_cast<std::uint8_t*>(map.data);
  for (std::size_t i = 0; i < bytes; ++i) {
    out[i] = static_cast<std::uint8_t>((i + input_index * 17U) & 0xffU);
  }

  Tensor tensor;
  tensor.storage = std::move(storage);
  tensor.dtype = dtype;
  tensor.shape = std::move(shape);
  tensor.strides_bytes = pipeline_internal::contiguous_strides_bytes(tensor.shape, elem);
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

struct BenchmarkInputGeometry {
  int width = 0;
  int height = 0;
  explicit operator bool() const {
    return width > 0 && height > 0;
  }
};

int benchmark_dim(int64_t value) {
  return (value > 0 && value <= static_cast<int64_t>(std::numeric_limits<int>::max()))
             ? static_cast<int>(value)
             : 0;
}

BenchmarkInputGeometry benchmark_tensor_geometry(const Tensor& tensor) {
  auto wh = [](int w, int h) { return BenchmarkInputGeometry{w, h}; };
  if (tensor.axis_semantics.size() == tensor.shape.size()) {
    BenchmarkInputGeometry g;
    for (std::size_t i = 0; i < tensor.shape.size(); ++i) {
      if (tensor.axis_semantics[i] == TensorAxisSemantic::W)
        g.width = benchmark_dim(tensor.shape[i]);
      else if (tensor.axis_semantics[i] == TensorAxisSemantic::H)
        g.height = benchmark_dim(tensor.shape[i]);
    }
    if (g)
      return g;
  }

  const auto& s = tensor.shape;
  if (s.size() == 2U)
    return wh(benchmark_dim(s[1]), benchmark_dim(s[0]));
  if (s.size() == 3U) {
    const bool chw = s[0] > 0 && s[0] <= 4 && s[1] > 4 && s[2] > 4;
    return chw ? wh(benchmark_dim(s[2]), benchmark_dim(s[1]))
               : wh(benchmark_dim(s[1]), benchmark_dim(s[0]));
  }
  if (s.size() == 4U) {
    return (s[3] > 0 && s[3] <= 4) ? wh(benchmark_dim(s[2]), benchmark_dim(s[1]))
                                   : wh(benchmark_dim(s[3]), benchmark_dim(s[2]));
  }
  return {};
}

BenchmarkInputGeometry benchmark_model_geometry(const Tensor& tensor,
                                                const ResolvedPreprocessPlan& plan,
                                                const PreprocessRuntimeMeta& meta) {
  auto wh = [](int w, int h) { return BenchmarkInputGeometry{w, h}; };
  for (const auto g :
       {wh(meta.resized_width, meta.resized_height), wh(meta.scaled_width, meta.scaled_height),
        wh(plan.effective.resize.width, plan.effective.resize.height),
        wh(plan.mla_contract.width, plan.mla_contract.height)}) {
    if (g)
      return g;
  }
  for (const auto& c : plan.ingress_contracts) {
    if (auto g = wh(c.width, c.height))
      return g;
  }
  if (const BenchmarkInputGeometry tensor_geometry = benchmark_tensor_geometry(tensor);
      tensor_geometry) {
    return tensor_geometry;
  }
  return wh(meta.original_width, meta.original_height);
}

std::string benchmark_resize_mode(ResizeMode mode) {
  switch (mode) {
  case ResizeMode::Stretch:
    return "stretch";
  case ResizeMode::Letterbox:
    return "letterbox";
  case ResizeMode::Crop:
    return "crop";
  }
  return "stretch";
}

std::string benchmark_input_color(const Tensor& tensor, const ResolvedPreprocessPlan& plan) {
  if (tensor.semantic.image.has_value()) {
    if (const std::string token = internal::image_format_name(tensor.semantic.image->format);
        !token.empty())
      return token;
  }
  const std::string planned =
      internal::preprocess_color_format_name(plan.effective.color_convert.input_format);
  return planned.empty() ? "synthetic" : planned;
}

void validate_benchmark_options(const BenchmarkOptions& options) {
  if (options.num_samples <= 0) {
    throw std::runtime_error("Model::benchmark: num_samples must be > 0");
  }
  if (options.min_duration_ms < 0) {
    throw std::runtime_error("Model::benchmark: min_duration_ms must be >= 0");
  }
  if (options.original_width.has_value() != options.original_height.has_value()) {
    throw std::runtime_error(
        "Model::benchmark: original_width and original_height must be provided together");
  }
  if (options.original_width.has_value() &&
      (*options.original_width <= 0 || *options.original_height <= 0)) {
    throw std::runtime_error("Model::benchmark: original_width and original_height must be > 0");
  }
}

PreprocessRuntimeMeta
resolve_benchmark_preprocess_meta_impl(const Tensor& tensor, const ResolvedPreprocessPlan& plan,
                                       const internal::PreprocessContractFlags& flags,
                                       const BenchmarkOptions& options) {
  validate_benchmark_options(options);

  const PreprocessRuntimeMeta existing =
      tensor.semantic.preprocess.value_or(PreprocessRuntimeMeta{});
  BenchmarkInputGeometry model_geometry = benchmark_model_geometry(tensor, plan, existing);
  BenchmarkInputGeometry source_geometry{
      options.original_width.value_or(existing.original_width),
      options.original_height.value_or(existing.original_height)};
  if (!source_geometry) {
    source_geometry = model_geometry;
  }
  if (!model_geometry) {
    model_geometry = source_geometry;
  }
  if (!source_geometry || !model_geometry) {
    throw std::runtime_error(
        "Model::benchmark: unable to infer synthetic source geometry required by BoxDecode");
  }

  const ResizeMode resize_mode = options.resize_mode.value_or(
      plan.effective.resize.enable == AutoFlag::On ? plan.effective.resize.mode
                                                   : ResizeMode::Stretch);

  PreprocessMetaTemplate meta_template;
  meta_template.enabled = true;
  meta_template.target_width = model_geometry.width;
  meta_template.target_height = model_geometry.height;
  meta_template.scaled_width = model_geometry.width;
  meta_template.scaled_height = model_geometry.height;
  meta_template.resize_mode = benchmark_resize_mode(resize_mode);
  meta_template.color_in = benchmark_input_color(tensor, plan);
  meta_template.color_out =
      internal::preprocess_color_format_name(plan.effective.color_convert.output_format);
  if (meta_template.color_out.empty()) {
    meta_template.color_out = meta_template.color_in;
  }
  meta_template.axis_perm = plan.effective.layout_convert.perm;
  meta_template.normalize = plan.effective.normalize.enable == AutoFlag::On;
  meta_template.quantize = flags.quant_needed;
  meta_template.tessellate = flags.tess_needed;

  InputOptions input_options;
  input_options.preprocess_meta = std::move(meta_template);
  const auto meta = make_simaai_preprocess_meta_from_template(input_options, source_geometry.width,
                                                              source_geometry.height);
  if (!meta.has_value()) {
    throw std::runtime_error(
        "Model::benchmark: failed to create synthetic preprocess metadata for BoxDecode");
  }
  return *meta;
}

void attach_benchmark_preprocess_meta_for_boxdecode(const Model& model, TensorList& inputs,
                                                    const BenchmarkOptions& options) {
  if (internal::ModelAccess::resolved_post_kind(model) != internal::PostRouteStageKind::BoxDecode)
    return;

  const ResolvedPreprocessPlan plan = model.resolved_preprocess_plan();
  const internal::PreprocessContractFlags flags =
      internal::ModelAccess::preprocess_contract_flags(model);
  for (Tensor& tensor : inputs) {
    tensor.semantic.preprocess =
        internal::resolve_benchmark_preprocess_meta(tensor, plan, flags, options);
  }
}

TensorList make_synthetic_benchmark_inputs(const Model& model, const BenchmarkOptions& options) {
  const std::vector<TensorSpec> specs = model.input_specs();
  if (specs.empty()) {
    throw std::runtime_error("Model::benchmark: input_specs() returned no inputs");
  }
  TensorList inputs;
  inputs.reserve(specs.size());
  for (std::size_t i = 0; i < specs.size(); ++i) {
    inputs.push_back(make_synthetic_benchmark_tensor(specs[i], i));
  }
  attach_benchmark_preprocess_meta_for_boxdecode(model, inputs, options);
  return inputs;
}

MeasureOptions make_benchmark_measure_options(int timeout_ms, int logical_batch_size,
                                              bool include_plugin_latency,
                                              std::string title = "NEAT Benchmark") {
  MeasureOptions opt;
  opt.duration_ms = timeout_ms;
  opt.warmup_ms = 0;
  opt.timeout_ms = timeout_ms;
  opt.include_plugin_latency = include_plugin_latency;
  opt.include_edge_latency = false;
  opt.include_message_latency = false;
  opt.logical_batch_size = logical_batch_size > 0 ? logical_batch_size : 1;
  opt.title = std::move(title);
  opt.input = "synthetic";
  return opt;
}

void require_benchmark_output(const simaai::neat::Sample& out, const char* where) {
  if (out.empty()) {
    throw std::runtime_error(std::string("Model::benchmark: ") + where + " pull timed out");
  }
  (void)tensors_from_sample(out, true);
}

} // namespace

namespace internal {

PreprocessRuntimeMeta resolve_benchmark_preprocess_meta(const Tensor& tensor,
                                                        const ResolvedPreprocessPlan& plan,
                                                        const PreprocessContractFlags& flags,
                                                        const BenchmarkOptions& options) {
  return resolve_benchmark_preprocess_meta_impl(tensor, plan, flags, options);
}

Model::Options benchmark_build_model_options(const Model::Options& model_options) {
  Model::Options options = model_options;
  // Let per-run metadata override deprecated BoxDecode geometry.
  options.boxdecode_original_width = 0;
  options.boxdecode_original_height = 0;
  options.boxdecode_resize_mode.reset();
  return options;
}

} // namespace internal

BenchmarkReport internal::run_benchmark(Model& model, const BenchmarkOptions& options) {
  validate_benchmark_options(options);
  // This gate covers benchmark calls only; offline profiling requires an idle DevKit.
  static std::shared_mutex benchmark_gate;
  std::unique_lock exclusive(benchmark_gate, std::defer_lock);
  std::shared_lock shared(benchmark_gate, std::defer_lock);
  std::vector<LayerProfileStage> profile_stages;
  if (options.include_layer_stats) {
    exclusive.lock();
    require_owned_mla_layer_profiling();
    const auto& contract = ModelAccess::pack(model).mpk_contract();
    if (!contract) {
      throw std::runtime_error("Model::benchmark: layer profiling requires an MPK manifest");
    }
    profile_stages = resolve_layer_profile_stages(*contract);
  } else {
    shared.lock();
  }
  const int num_samples = options.num_samples;
  const bool include_plugin_latency = options.include_plugin_latency;

  constexpr int kWarmupSamples = 100;
  constexpr int kTimeoutMs = 120000;
  const TensorList inputs = make_synthetic_benchmark_inputs(model, options);
  const int logical_batch_size = model.compiled_batch_size();
  const Model::Options benchmark_model_options =
      internal::ModelAccess::resolved_post_kind(model) == internal::PostRouteStageKind::BoxDecode
          ? internal::benchmark_build_model_options(internal::ModelAccess::options(model))
          : internal::ModelAccess::options(model);

  RunOptions latency_run_options;
  latency_run_options.startup_preflight = false;

  MeasureReport latency_measured;
  MeasureReport throughput_measured;
  int latency_samples = 0;
  {
    Model::Runner latency_runner = ModelAccess::build_benchmark_runner(
        model, inputs, latency_run_options, benchmark_model_options);
    for (int i = 0; i < kWarmupSamples; ++i) {
      if (!latency_runner.push(inputs)) {
        latency_runner.close();
        throw std::runtime_error("Model::benchmark: latency warmup push failed");
      }
      Sample out = latency_runner.pull(kTimeoutMs);
      require_benchmark_output(out, "latency warmup");
    }

    MeasureScope measure = latency_runner.start_measurement(make_benchmark_measure_options(
        kTimeoutMs, logical_batch_size, include_plugin_latency, "NEAT Benchmark latency"));
    const auto min_end =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(options.min_duration_ms);
    while (latency_samples < num_samples || std::chrono::steady_clock::now() < min_end) {
      if (!latency_runner.push(inputs)) {
        latency_runner.close();
        throw std::runtime_error("Model::benchmark: latency measured push failed");
      }
      Sample out = latency_runner.pull(kTimeoutMs);
      require_benchmark_output(out, "latency measured");
      ++latency_samples;
    }

    latency_measured = measure.stop();
    latency_runner.close();
  }

  Telemetry telemetry;

  RunOptions throughput_run_options;
  throughput_run_options.startup_preflight = false;

  int pushed = 0;
  {
    Model::Runner runner = ModelAccess::build_benchmark_runner(
        model, inputs, throughput_run_options, benchmark_model_options);
    const bool defer_eos_until_outputs_drained =
        ModelAccess::uses_direct_mla_public_output_boundary(model);
    for (int i = 0; i < kWarmupSamples; ++i) {
      if (!runner.push(inputs)) {
        runner.close();
        throw std::runtime_error("Model::benchmark: throughput warmup push failed");
      }
      Sample out = runner.pull(kTimeoutMs);
      require_benchmark_output(out, "throughput warmup");
    }

    TelemetryMonitor telemetry_monitor;
    internal::TelemetryMonitorAccess::prepare_benchmark(telemetry_monitor);
    MeasureScope measure = runner.start_measurement(make_benchmark_measure_options(
        kTimeoutMs, logical_batch_size, include_plugin_latency, "NEAT Benchmark throughput"));
    internal::TelemetryMonitorAccess::arm_benchmark(telemetry_monitor);
    const auto min_end =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(options.min_duration_ms);

    // Pull only after the corresponding input has been pushed.
    std::mutex count_mu;
    std::condition_variable count_cv;
    bool pushing = true;
    int pulled = 0;
    std::exception_ptr pull_error;
    std::thread pull_thread([&] {
      try {
        for (;;) {
          {
            std::unique_lock<std::mutex> lock(count_mu);
            count_cv.wait(lock, [&] { return pulled < pushed || !pushing; });
            if (pulled == pushed) {
              break;
            }
          }
          Sample out = runner.pull(kTimeoutMs);
          require_benchmark_output(out, "throughput measured");
          ++pulled;
        }
      } catch (...) {
        {
          std::lock_guard<std::mutex> lock(count_mu);
          pull_error = std::current_exception();
        }
        // Unblock a producer when output draining fails.
        runner.close_input();
      }
    });
    const auto finish_pushing = [&] {
      {
        std::lock_guard<std::mutex> lock(count_mu);
        pushing = false;
      }
      count_cv.notify_one();
    };

    try {
      while (pushed < num_samples || std::chrono::steady_clock::now() < min_end) {
        if (!runner.push(inputs)) {
          throw std::runtime_error("Model::benchmark: throughput measured push failed");
        }
        {
          std::lock_guard<std::mutex> lock(count_mu);
          ++pushed;
          if (pull_error) {
            break;
          }
        }
        count_cv.notify_one();
      }
      finish_pushing();
      if (!defer_eos_until_outputs_drained) {
        runner.close_input();
      }
      pull_thread.join();
      if (defer_eos_until_outputs_drained) {
        runner.close_input();
      }
    } catch (...) {
      finish_pushing();
      runner.close_input();
      if (pull_thread.joinable()) {
        pull_thread.join();
      }
      runner.close();
      // Report the pull failure that closed the input.
      if (pull_error) {
        std::rethrow_exception(pull_error);
      }
      throw;
    }
    if (pull_error) {
      runner.close();
      std::rethrow_exception(pull_error);
    }
    if (pulled != pushed) {
      runner.close();
      throw std::runtime_error("Model::benchmark: async measured output count mismatch");
    }

    // Seal without joining or contacting Sentinel while the model timer runs.
    // The final poll and worker teardown belong after the throughput window.
    internal::TelemetryMonitorAccess::seal_benchmark(telemetry_monitor);
    throughput_measured = measure.stop();
    telemetry = telemetry_monitor.stop();
    runner.close();
  }

  BenchmarkReport report = internal::build_benchmark_report_from_measurements(
      latency_measured, throughput_measured, num_samples, latency_samples, pushed,
      std::move(telemetry));
  if (options.include_layer_stats) {
    report.layer_stats = collect_mla_layer_stats(profile_stages);
  }
  const double batch_throughput = throughput_measured.throughput_batches_per_s;
  const Telemetry& board = report.telemetry;

  const auto old_flags = std::cout.flags();
  const auto old_precision = std::cout.precision();
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "NEAT Benchmark\n";
  std::cout << "Input: synthetic\n";
  std::cout << "Samples: " << num_samples << "\n";
  std::cout << "Compiled batch: " << logical_batch_size << "\n";
  std::cout << "Measurement detail: "
            << (include_plugin_latency ? "e2e+throughput+plugin-latency" : "e2e+throughput-only")
            << "\n";
  std::cout << "Latency: " << report.latency_ms << " ms\n";
  std::cout << "Latency window: " << latency_measured.elapsed_s << " s, " << latency_samples
            << " inputs\n";
  std::cout << "FPS:     " << report.fps << " inferences/s\n";
  std::cout << "Batch throughput: " << batch_throughput << " batches/s\n";
  std::cout << "Throughput window: " << report.throughput_duration_ms / 1000.0 << " s, "
            << report.throughput_samples << " inputs\n";
  if (include_plugin_latency) {
    std::cout << "Plugin latency latency-run: " << latency_measured.plugin_latency_source << "/"
              << latency_measured.plugin_latency_status << " rows="
              << (latency_measured.plugin_latency.size() +
                  latency_measured.plugin_latency_unattributed.size())
              << "\n";
    std::cout << "Plugin latency throughput-run: " << throughput_measured.plugin_latency_source
              << "/" << throughput_measured.plugin_latency_status << " rows="
              << (throughput_measured.plugin_latency.size() +
                  throughput_measured.plugin_latency_unattributed.size())
              << "\n";
  }
  const auto print_metric = [](const char* label, const TelemetryStats& metric, const char* unit) {
    if (metric.samples == 0) {
      return;
    }
    std::cout << label << metric.mean << " " << unit;
    if (metric.minimum != metric.maximum) {
      std::cout << " avg, " << metric.maximum << " " << unit << " max";
    }
    std::cout << "\n";
  };
  if (board.power_som_watts.samples > 0) {
    std::cout << "Power:         " << report.avg_power_watts << " W avg, "
              << report.peak_power_watts << " W peak (SoM)\n";
    print_metric("MLA power:     ", board.power_mla_watts, "W");
    std::cout << "Energy:        " << report.energy_joules << " J for " << num_samples
              << " inputs\n";
  } else {
    std::cout << "Power: unavailable ("
              << (board.status == "ok" ? "Sentinel reported no SoM power" : board.status) << ")\n";
  }
  if (board.status == "ok") {
    print_metric("Die temp:      ", board.die_temperature_max_c, "°C");
    print_metric("CPU usage:     ", board.cpu_usage_pct, "%");
    print_metric("MLA memory:    ", board.mla_memory_mib, "MiB");
    print_metric("CMA used:      ", board.cma_memory_mib, "MiB");
    print_metric("Linux memory:  ", board.linux_memory_used_mib, "MiB");
    std::cout << "Telemetry:     Sentinel " << board.sentinel_version << ", " << board.samples
              << " samples\n";
  }
  std::cout.flags(old_flags);
  std::cout.precision(old_precision);

  return report;
}

namespace internal {

namespace {

void require_benchmark_measure_report_window(const MeasureReport& report, const char* label,
                                             int expected_samples) {
  if (expected_samples <= 0) {
    throw std::runtime_error("Model::benchmark: expected_samples must be > 0");
  }
  const auto expected_u64 = static_cast<std::uint64_t>(expected_samples);
  const auto expected_size = static_cast<std::size_t>(expected_samples);
  if (report.counters.outputs_pulled != expected_u64) {
    std::ostringstream oss;
    oss << "Model::benchmark: " << label << " measured output count mismatch"
        << " (expected=" << expected_samples
        << ", outputs_pulled=" << report.counters.outputs_pulled << ")";
    throw std::runtime_error(oss.str());
  }
  if (report.end_to_end.count != expected_size) {
    std::ostringstream oss;
    oss << "Model::benchmark: " << label << " measurement collected " << report.end_to_end.count
        << " end-to-end samples for " << expected_samples
        << " pulled outputs; cannot publish benchmark result";
    throw std::runtime_error(oss.str());
  }
  if (report.graph_sample_timing_unkeyed != 0U || report.graph_sample_timing_misses != 0U) {
    std::ostringstream oss;
    oss << "Model::benchmark: " << label << " measurement has unreliable graph sample correlation"
        << " (unkeyed=" << report.graph_sample_timing_unkeyed
        << ", misses=" << report.graph_sample_timing_misses << ")";
    throw std::runtime_error(oss.str());
  }
  if (report.counters.inputs_dropped != 0U || report.counters.outputs_dropped != 0U) {
    std::ostringstream oss;
    oss << "Model::benchmark: " << label << " measurement dropped inputs or outputs"
        << " (inputs_dropped=" << report.counters.inputs_dropped
        << ", outputs_dropped=" << report.counters.outputs_dropped << ")";
    throw std::runtime_error(oss.str());
  }
}

void require_finite_positive(double value, const char* field) {
  if (!std::isfinite(value) || value <= 0.0) {
    std::ostringstream oss;
    oss << "Model::benchmark: " << field << " must be finite and positive; got " << value;
    throw std::runtime_error(oss.str());
  }
}

} // namespace

BenchmarkReport build_benchmark_report_from_measurements(const MeasureReport& latency_report,
                                                         const MeasureReport& throughput_report,
                                                         int num_samples, int latency_samples,
                                                         int throughput_samples,
                                                         Telemetry telemetry) {
  require_benchmark_measure_report_window(latency_report, "latency", latency_samples);
  require_benchmark_measure_report_window(throughput_report, "throughput", throughput_samples);

  const int latency_batch =
      latency_report.options.logical_batch_size > 0 ? latency_report.options.logical_batch_size : 1;
  const int throughput_batch = throughput_report.options.logical_batch_size > 0
                                   ? throughput_report.options.logical_batch_size
                                   : 1;
  if (latency_batch != throughput_batch) {
    std::ostringstream oss;
    oss << "Model::benchmark: latency and throughput measurement windows disagree on logical "
           "batch size"
        << " (latency=" << latency_batch << ", throughput=" << throughput_batch << ")";
    throw std::runtime_error(oss.str());
  }

  require_finite_positive(latency_report.end_to_end.avg_ms, "latency_ms");
  require_finite_positive(throughput_report.throughput_batches_per_s, "throughput_batches_per_s");
  require_finite_positive(throughput_report.throughput_inferences_per_s,
                          "throughput_inferences_per_s");

  const double expected_inference_tput =
      throughput_report.throughput_batches_per_s * static_cast<double>(throughput_batch);
  const double tolerance = std::max(1e-9, std::abs(expected_inference_tput) * 1e-9);
  if (std::abs(throughput_report.throughput_inferences_per_s - expected_inference_tput) >
      tolerance) {
    std::ostringstream oss;
    oss << "Model::benchmark: throughput_inferences_per_s must equal batches/s multiplied by "
           "logical batch size"
        << " (batches_per_s=" << throughput_report.throughput_batches_per_s
        << ", logical_batch_size=" << throughput_batch
        << ", inferences_per_s=" << throughput_report.throughput_inferences_per_s << ")";
    throw std::runtime_error(oss.str());
  }

  BenchmarkReport out;
  out.latency_ms = latency_report.end_to_end.avg_ms;
  out.latency_p50_ms = latency_report.end_to_end.p50_ms;
  out.latency_p95_ms = latency_report.end_to_end.p95_ms;
  out.latency_p99_ms = latency_report.end_to_end.p99_ms;
  out.fps = throughput_report.throughput_inferences_per_s;
  out.throughput_samples = throughput_samples;
  out.throughput_duration_ms = throughput_report.elapsed_s * 1000.0;
  if (telemetry.power_som_watts.samples > 0) {
    out.avg_power_watts = telemetry.power_som_watts.mean;
    out.peak_power_watts = telemetry.power_som_watts.maximum;
    // Normalize energy to num_samples, independent of window length.
    out.energy_joules = out.avg_power_watts * static_cast<double>(num_samples) /
                        throughput_report.throughput_batches_per_s;
  }
  out.telemetry = std::move(telemetry);
  return out;
}

} // namespace internal

} // namespace simaai::neat
