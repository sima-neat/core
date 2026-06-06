#include "gst/GstHelpers.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "nodes/sima/VisualFrontend.h"
#include "pipeline/Graph.h"
#include "test_utils.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

int env_int_local(const char* key, int def) {
  const char* v = std::getenv(key);
  if (!v || !*v) {
    return def;
  }
  return std::atoi(v);
}

std::string env_string(const char* key, std::string def = {}) {
  const char* v = std::getenv(key);
  if (!v || !*v) {
    return def;
  }
  return std::string(v);
}

std::string shape_string(const std::vector<int64_t>& shape) {
  std::ostringstream os;
  os << "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i) {
      os << ",";
    }
    os << shape[i];
  }
  os << "]";
  return os.str();
}

std::string dtype_string(simaai::neat::TensorDType dtype) {
  switch (dtype) {
  case simaai::neat::TensorDType::UInt8:
    return "UInt8";
  case simaai::neat::TensorDType::Int8:
    return "Int8";
  case simaai::neat::TensorDType::UInt16:
    return "UInt16";
  case simaai::neat::TensorDType::Int16:
    return "Int16";
  case simaai::neat::TensorDType::Int32:
    return "Int32";
  case simaai::neat::TensorDType::BFloat16:
    return "BFloat16";
  case simaai::neat::TensorDType::Float32:
    return "Float32";
  case simaai::neat::TensorDType::Float64:
    return "Float64";
  }
  return "Unknown";
}

void stamp_route(simaai::neat::Tensor& tensor, const std::string& name, int logical,
                 int physical) {
  tensor.route.name = name;
  tensor.route.backend_name = name;
  tensor.route.segment_name = name;
  tensor.route.logical_index = logical;
  tensor.route.physical_index = physical;
  tensor.route.memory_index = 0;
  tensor.route.route_slot = logical;
  tensor.route.backend_output_index = logical;
}

simaai::neat::Tensor make_u8_tensor(int width, int height, const std::string& name, int logical,
                                    int physical, std::uint8_t salt) {
  std::vector<std::uint8_t> data(static_cast<std::size_t>(width) *
                                 static_cast<std::size_t>(height));
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      data[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
           static_cast<std::size_t>(x)] =
          static_cast<std::uint8_t>((x * 3 + y * 5 + salt) & 0xFF);
    }
  }
  auto tensor = simaai::neat::Tensor::from_vector(data, {height, width},
                                                  simaai::neat::TensorMemory::EV74);
  tensor.layout = simaai::neat::TensorLayout::HW;
  tensor.axis_semantics = {simaai::neat::TensorAxisSemantic::H,
                           simaai::neat::TensorAxisSemantic::W};
  stamp_route(tensor, name, logical, physical);
  return tensor;
}

simaai::neat::Tensor make_i32_points_tensor(int width, int height, int num_points,
                                            const std::string& name, int logical, int physical) {
  std::vector<std::int32_t> data(static_cast<std::size_t>(num_points) * 2U);
  for (int i = 0; i < num_points; ++i) {
    data[static_cast<std::size_t>(i) * 2U + 0U] =
        8 + ((i * 17) % std::max(1, width - 16));
    data[static_cast<std::size_t>(i) * 2U + 1U] =
        8 + ((i * 11) % std::max(1, height - 16));
  }
  auto tensor = simaai::neat::Tensor::from_vector(data, {num_points, 2},
                                                  simaai::neat::TensorMemory::EV74);
  tensor.layout = simaai::neat::TensorLayout::HW;
  tensor.axis_semantics = {simaai::neat::TensorAxisSemantic::H,
                           simaai::neat::TensorAxisSemantic::W};
  stamp_route(tensor, name, logical, physical);
  return tensor;
}

struct ExpectedOutput {
  std::string name;
  std::vector<int64_t> shape;
  simaai::neat::TensorDType dtype = simaai::neat::TensorDType::UInt8;
};

struct BenchCase {
  std::string name;
  std::string graph_name;
  int graph_id = 0;
  std::vector<std::string> input_names;
  simaai::neat::TensorList inputs;
  std::shared_ptr<simaai::neat::Node> node;
  std::vector<ExpectedOutput> expected_outputs;
};

struct TimingSummary {
  double avg_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
  double fps = 0.0;
};

TimingSummary summarize(const std::vector<double>& samples_ms) {
  TimingSummary out;
  if (samples_ms.empty()) {
    return out;
  }
  out.min_ms = *std::min_element(samples_ms.begin(), samples_ms.end());
  out.max_ms = *std::max_element(samples_ms.begin(), samples_ms.end());
  out.avg_ms =
      std::accumulate(samples_ms.begin(), samples_ms.end(), 0.0) /
      static_cast<double>(samples_ms.size());
  out.fps = out.avg_ms > 0.0 ? 1000.0 / out.avg_ms : 0.0;
  return out;
}

double element_avg_ms(const simaai::neat::Run& run, const std::string& element_name) {
  const simaai::neat::RunDiagSnapshot diag = run.diag_snapshot();
  for (const auto& timing : diag.element_timings) {
    if (timing.element_name == element_name && timing.samples > 0U) {
      return static_cast<double>(timing.total_us) / 1000.0 /
             static_cast<double>(timing.samples);
    }
  }
  return 0.0;
}

void validate_outputs(const BenchCase& c, const simaai::neat::TensorList& outputs) {
  require(outputs.size() == c.expected_outputs.size(),
          c.name + ": expected " + std::to_string(c.expected_outputs.size()) +
              " output tensors, got " + std::to_string(outputs.size()));
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    const auto& tensor = outputs[i];
    const auto& expected = c.expected_outputs[i];
    require(tensor.shape == expected.shape,
            c.name + ": output[" + std::to_string(i) + "] shape mismatch expected=" +
                shape_string(expected.shape) + " actual=" + shape_string(tensor.shape));
    require(tensor.dtype == expected.dtype,
            c.name + ": output[" + std::to_string(i) + "] dtype mismatch expected=" +
                dtype_string(expected.dtype) + " actual=" + dtype_string(tensor.dtype));
    require(tensor.dense_bytes_tight() > 0U,
            c.name + ": output[" + std::to_string(i) + "] has zero payload bytes");
  }
}

BenchCase make_feature_histogram_case(int width, int height, int num_buffers) {
  simaai::neat::FeatureHistogramOptions opt;
  opt.width = width;
  opt.height = height;
  opt.batch_size = 1;
  opt.num_buffers = num_buffers;
  opt.element_name = "feature_histogram";
  opt.input_name = "input_image";
  opt.output_name = "output_hist";

  BenchCase c;
  c.name = "FeatureHistogram";
  c.graph_name = "feature_histogram";
  c.graph_id = 235;
  c.input_names = {opt.input_name};
  c.inputs = {make_u8_tensor(width, height, opt.input_name, 0, 0, 13)};
  c.node = simaai::neat::nodes::FeatureHistogram(opt);
  c.expected_outputs = {{opt.output_name, {1, 256}, simaai::neat::TensorDType::Int32}};
  return c;
}

BenchCase make_grider_fast_case(int width, int height, int max_features, int num_buffers) {
  simaai::neat::GriderFastOptions opt;
  opt.width = width;
  opt.height = height;
  opt.batch_size = 1;
  opt.max_features = max_features;
  opt.grid_x = 8;
  opt.grid_y = 6;
  opt.threshold = 30;
  opt.min_px_dist = 10;
  opt.num_buffers = num_buffers;
  opt.element_name = "grider_fast";
  opt.input_name = "input_image";
  opt.output_name = "output_features";

  BenchCase c;
  c.name = "GriderFast";
  c.graph_name = "grider_fast";
  c.graph_id = 236;
  c.input_names = {opt.input_name};
  c.inputs = {make_u8_tensor(width, height, opt.input_name, 0, 0, 23)};
  c.node = simaai::neat::nodes::GriderFast(opt);
  c.expected_outputs = {
      {opt.output_name, {1, 1 + max_features * 3}, simaai::neat::TensorDType::Int32}};
  return c;
}

BenchCase make_track_descriptor_case(int width, int height, int max_features, int num_buffers) {
  simaai::neat::TrackDescriptorOptions opt;
  opt.width = width;
  opt.height = height;
  opt.batch_size = 1;
  opt.max_features = max_features;
  opt.grid_x = 8;
  opt.grid_y = 6;
  opt.threshold = 30;
  opt.min_px_dist = 10;
  opt.descriptor_words = 8;
  opt.num_buffers = num_buffers;
  opt.element_name = "track_descriptor";
  opt.input_name = "input_image";
  opt.features_output_name = "output_features";
  opt.descriptors_output_name = "output_descriptors";

  BenchCase c;
  c.name = "TrackDescriptor";
  c.graph_name = "track_descriptor";
  c.graph_id = 237;
  c.input_names = {opt.input_name};
  c.inputs = {make_u8_tensor(width, height, opt.input_name, 0, 0, 37)};
  c.node = simaai::neat::nodes::TrackDescriptor(opt);
  c.expected_outputs = {
      {opt.features_output_name, {1, 1 + max_features * 3}, simaai::neat::TensorDType::Int32},
      {opt.descriptors_output_name, {max_features, 8}, simaai::neat::TensorDType::Int32}};
  return c;
}

BenchCase make_track_klt_case(int width, int height, int num_points, int max_features,
                              int num_buffers) {
  simaai::neat::TrackKLTOptions opt;
  opt.width = width;
  opt.height = height;
  opt.num_points = num_points;
  opt.max_features = max_features;
  opt.detect_new_features = 0;
  opt.grid_x = 8;
  opt.grid_y = 6;
  opt.fast_threshold = 30;
  opt.min_px_dist = 10;
  opt.num_buffers = num_buffers;
  opt.element_name = "track_klt";
  opt.prev_image_name = "prev_image";
  opt.cur_image_name = "cur_image";
  opt.input_points_name = "input_points";
  opt.output_points_name = "output_points";
  opt.output_status_name = "output_status";
  opt.output_features_name = "output_features";

  BenchCase c;
  c.name = "TrackKLT";
  c.graph_name = "track_klt";
  c.graph_id = 238;
  c.input_names = {opt.prev_image_name, opt.cur_image_name, opt.input_points_name};
  c.inputs = {make_u8_tensor(width, height, opt.prev_image_name, 0, 0, 47),
              make_u8_tensor(width, height, opt.cur_image_name, 1, 1, 49),
              make_i32_points_tensor(width, height, num_points, opt.input_points_name, 2, 2)};
  c.node = simaai::neat::nodes::TrackKLT(opt);
  c.expected_outputs = {{opt.output_points_name, {num_points, 2},
                         simaai::neat::TensorDType::Float32},
                        {opt.output_status_name, {num_points, 1},
                         simaai::neat::TensorDType::Int32}};
  return c;
}

std::vector<BenchCase> make_cases() {
  const int width = env_int_local("SIMA_VISUAL_TPUT_WIDTH", 640);
  const int height = env_int_local("SIMA_VISUAL_TPUT_HEIGHT", 480);
  const int max_features = env_int_local("SIMA_VISUAL_TPUT_MAX_FEATURES", 256);
  const int num_points = env_int_local("SIMA_VISUAL_TPUT_NUM_POINTS", 128);
  const int num_buffers = env_int_local("SIMA_VISUAL_TPUT_NUM_BUFFERS", 4);

  std::vector<BenchCase> cases;
  cases.push_back(make_feature_histogram_case(width, height, num_buffers));
  cases.push_back(make_grider_fast_case(width, height, max_features, num_buffers));
  cases.push_back(make_track_descriptor_case(width, height, max_features, num_buffers));
  cases.push_back(make_track_klt_case(width, height, num_points, max_features, num_buffers));

  const std::string only = env_string("SIMA_VISUAL_TPUT_CASE");
  if (only.empty() || only == "all") {
    return cases;
  }

  std::vector<BenchCase> filtered;
  for (auto& c : cases) {
    if (c.graph_name == only || c.name == only) {
      filtered.push_back(std::move(c));
    }
  }
  if (filtered.empty()) {
    throw std::runtime_error("unknown SIMA_VISUAL_TPUT_CASE: " + only);
  }
  return filtered;
}

void print_tensor_summary(const simaai::neat::TensorList& outputs) {
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    const auto& t = outputs[i];
    std::cout << " output[" << i << "] name="
              << (t.route.segment_name.empty() ? "<empty>" : t.route.segment_name)
              << " shape=" << shape_string(t.shape) << " dtype=" << dtype_string(t.dtype)
              << " bytes=" << t.dense_bytes_tight() << "\n";
  }
}

bool run_case(const BenchCase& c, int warmup, int iterations, int timeout_ms, int queue_depth) {
  std::cout << "\n[case] " << c.name << " graph=" << c.graph_name << " id=" << c.graph_id
            << " inputs=" << c.inputs.size() << " warmup=" << warmup
            << " iterations=" << iterations << "\n";

  simaai::neat::Graph graph;

  simaai::neat::InputOptions src_opt;
  src_opt.payload_type = simaai::neat::PayloadType::Tensor;
  src_opt.format = simaai::neat::FormatTag::UINT8;
  src_opt.width = static_cast<int>(c.inputs.front().shape.size() >= 2 ? c.inputs.front().shape[1]
                                                                      : -1);
  src_opt.height = static_cast<int>(c.inputs.front().shape.size() >= 1 ? c.inputs.front().shape[0]
                                                                       : -1);
  src_opt.depth = 1;
  src_opt.is_live = true;
  src_opt.do_timestamp = true;
  src_opt.block = true;
  src_opt.use_simaai_pool = env_flag("SIMA_VISUAL_TPUT_USE_POOL", true);
  src_opt.pool_min_buffers = std::max(2, env_int_local("SIMA_VISUAL_TPUT_NUM_BUFFERS", 4));
  src_opt.pool_max_buffers = src_opt.pool_min_buffers;
  src_opt.memory_policy = simaai::neat::InputMemoryPolicy::Ev74;
  src_opt.buffer_name = c.input_names.empty() ? "input_tensor" : c.input_names.front();
  src_opt.max_bytes = 0;

  graph.add(simaai::neat::nodes::Input(src_opt));
  graph.add(c.node);

  simaai::neat::OutputOptions sink_opt;
  sink_opt.sync = false;
  sink_opt.drop = false;
  sink_opt.max_buffers = std::max(2, queue_depth);
  graph.add(simaai::neat::nodes::Output(sink_opt));

  simaai::neat::RunOptions run_opt;
  run_opt.output_memory = simaai::neat::OutputMemory::Owned;
  run_opt.enable_metrics = true;
  run_opt.queue_depth = queue_depth;
  run_opt.input_timeout_ms = timeout_ms;
  run_opt.startup_preflight = true;

  auto build_start = Clock::now();
  simaai::neat::Run run =
      graph.build(c.inputs, simaai::neat::RunMode::Async, run_opt);
  auto build_end = Clock::now();
  const double build_ms =
      std::chrono::duration<double, std::milli>(build_end - build_start).count();
  std::cout << " build_ms=" << std::fixed << std::setprecision(3) << build_ms << "\n";

  simaai::neat::TensorList last_outputs;
  for (int i = 0; i < warmup; ++i) {
    last_outputs = run.run(c.inputs, timeout_ms);
    validate_outputs(c, last_outputs);
  }

  std::vector<double> samples_ms;
  samples_ms.reserve(static_cast<std::size_t>(iterations));
  for (int i = 0; i < iterations; ++i) {
    const auto start = Clock::now();
    last_outputs = run.run(c.inputs, timeout_ms);
    const auto end = Clock::now();
    validate_outputs(c, last_outputs);
    samples_ms.push_back(std::chrono::duration<double, std::milli>(end - start).count());
  }

  const TimingSummary timing = summarize(samples_ms);
  const double plugin_ms = element_avg_ms(run, c.graph_name);
  const simaai::neat::RunStats stats = run.stats();
  const simaai::neat::InputStreamStats input_stats = run.input_stats();

  std::cout << " result=PASS avg_ms=" << timing.avg_ms << " min_ms=" << timing.min_ms
            << " max_ms=" << timing.max_ms << " fps=" << timing.fps
            << " processcvu_avg_ms=" << plugin_ms
            << " pushed=" << input_stats.push_count
            << " pulled=" << input_stats.pull_count
            << " outputs_ready=" << stats.outputs_ready
            << " outputs_pulled=" << stats.outputs_pulled << "\n";
  print_tensor_summary(last_outputs);
  run.close();
  return true;
}

} // namespace

int main() {
  try {
    require(simaai::neat::element_exists("neatprocesscvu"),
            "Missing SIMA processcvu plugin (neatprocesscvu).");

    const int warmup = env_int_local("SIMA_VISUAL_TPUT_WARMUP", 2);
    const int iterations = env_int_local("SIMA_VISUAL_TPUT_ITERS", 20);
    const int timeout_ms = env_int_local("SIMA_VISUAL_TPUT_TIMEOUT_MS", 30000);
    const int queue_depth = std::max(1, env_int_local("SIMA_VISUAL_TPUT_QUEUE_DEPTH", 1));
    const bool continue_on_fail = env_flag("SIMA_VISUAL_TPUT_CONTINUE_ON_FAIL", true);

    std::vector<BenchCase> cases = make_cases();
    int failures = 0;
    for (const auto& c : cases) {
      try {
        (void)run_case(c, warmup, iterations, timeout_ms, queue_depth);
      } catch (const std::exception& e) {
        ++failures;
        const std::string msg = e.what();
        std::cerr << "[case-fail] " << c.name << ": " << msg << "\n";
        if (!continue_on_fail) {
          throw;
        }
      }
    }

    if (failures != 0) {
      std::cerr << "[FAIL] visual_frontend_tput_test failures=" << failures << "/"
                << cases.size() << "\n";
      return 1;
    }
    std::cout << "\n[OK] visual_frontend_tput_test passed cases=" << cases.size() << "\n";
    return 0;
  } catch (const SkipTest& e) {
    return skip_test(e.what());
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    if (is_dispatcher_unavailable(msg)) {
      return skip_test("dispatcher unavailable: " + msg);
    }
    std::cerr << "[FAIL] " << msg << "\n";
    return 1;
  }
}
