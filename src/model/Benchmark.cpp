#include "model/Model.h"
#include "model/internal/BenchmarkInternal.h"
#include "model/internal/BenchmarkTelemetry.h"
#include "model/internal/ModelPack.h"
#include "model/internal/ModelFormat.h"
#include "model/internal/ModelInternal.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/internal/TensorMath.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <thread>
#include <utility>

#include <glib.h>
#include <yaml.h>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>

#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <nlohmann/json.hpp>

#include <cmath>
#include <charconv>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace simaai::neat {
namespace {

void require_finite_nonnegative(double value, const char* field) {
  if (!std::isfinite(value) || value < 0.0) {
    throw std::invalid_argument(std::string("BenchmarkReport::save: ") + field +
                                " must be finite and nonnegative");
  }
}

// JSON double-quoted strings are also YAML 1.2 scalars. Escaping keeps compiler layer names
// containing colons, quotes, newlines, or YAML-looking values unambiguously strings.
std::string yaml_string(const std::string& value) {
  const auto quoted = nlohmann::json(value).dump();
  std::string escaped;
  constexpr char hex[] = "0123456789abcdef";
  for (std::size_t i = 0; i < quoted.size(); ++i) {
    const auto byte = static_cast<unsigned char>(quoted[i]);
    if (byte == 0x7f) {
      escaped += "\\u007f";
    } else if (byte == 0xc2 && i + 1 < quoted.size() &&
               static_cast<unsigned char>(quoted[i + 1]) >= 0x80 &&
               static_cast<unsigned char>(quoted[i + 1]) <= 0x9f) {
      // YAML excludes C1 controls from literal scalars; NEL is also a line break in YAML 1.1.
      const auto control = static_cast<unsigned char>(quoted[++i]);
      escaped += "\\u00";
      escaped += hex[control >> 4];
      escaped += hex[control & 0xf];
    } else if (byte == 0xe2 && i + 2 < quoted.size() &&
               static_cast<unsigned char>(quoted[i + 1]) == 0x80 &&
               (static_cast<unsigned char>(quoted[i + 2]) == 0xa8 ||
                static_cast<unsigned char>(quoted[i + 2]) == 0xa9)) {
      escaped += static_cast<unsigned char>(quoted[i + 2]) == 0xa8 ? "\\u2028" : "\\u2029";
      i += 2;
    } else {
      escaped += quoted[i];
    }
  }
  return escaped;
}

std::string yaml_number(double value) {
  char buffer[64];
  const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value);
  if (result.ec != std::errc{}) {
    throw std::runtime_error("BenchmarkReport::save: could not format timing value");
  }
  std::string number(buffer, result.ptr);
  // YAML 1.1 readers such as PyYAML require a decimal point in an exponent's mantissa.
  const auto exponent = number.find_first_of("eE");
  if (exponent != std::string::npos && number.find('.') == std::string::npos) {
    number.insert(exponent, ".0");
  }
  return number;
}

// Keep floating-point telemetry compatible with YAML 1.1 readers too. A plain JSON dump
// spells small values as 1e-09, which PyYAML resolves as a string rather than a number.
void write_yaml_flow(std::ostream& out, const nlohmann::json& value) {
  if (value.is_number_float()) {
    out << yaml_number(value.get<double>());
  } else if (value.is_string()) {
    out << yaml_string(value.get_ref<const std::string&>());
  } else if (value.is_structured()) {
    const bool object = value.is_object();
    out << (object ? '{' : '[');
    bool first = true;
    for (const auto& item : value.items()) {
      if (!first)
        out << ',';
      first = false;
      if (object)
        out << yaml_string(item.key()) << ':';
      write_yaml_flow(out, item.value());
    }
    out << (object ? '}' : ']');
  } else {
    out << value.dump();
  }
}

nlohmann::json metric_json(const BenchmarkMetric& metric) {
  if (metric.samples < 0 || !std::isfinite(metric.mean) || !std::isfinite(metric.minimum) ||
      !std::isfinite(metric.maximum)) {
    throw std::invalid_argument("BenchmarkReport::save: invalid telemetry metric: " + metric.key);
  }
  return {{"key", metric.key},         {"label", metric.label}, {"unit", metric.unit},
          {"samples", metric.samples}, {"mean", metric.mean},   {"minimum", metric.minimum},
          {"maximum", metric.maximum}};
}

nlohmann::json telemetry_json(const BenchmarkTelemetry& telemetry) {
  if (telemetry.samples < 0 || telemetry.sample_interval_ms < 0)
    throw std::invalid_argument("BenchmarkReport::save: invalid telemetry sample counts");
  const auto metrics = [](const std::vector<BenchmarkMetric>& values) {
    auto result = nlohmann::json::array();
    for (const auto& value : values)
      result.push_back(metric_json(value));
    return result;
  };
  return {{"status", telemetry.status},
          {"trace_id", telemetry.trace_id},
          {"sentinel_version", telemetry.sentinel_version},
          {"sample_interval_ms", telemetry.sample_interval_ms},
          {"samples", telemetry.samples},
          {"power_som_watts", metric_json(telemetry.power_som_watts)},
          {"power_mla_watts", metric_json(telemetry.power_mla_watts)},
          {"power_rails_watts", metrics(telemetry.power_rails_watts)},
          {"die_temperatures_c", metrics(telemetry.die_temperatures_c)},
          {"die_temperature_max_c", metric_json(telemetry.die_temperature_max_c)},
          {"board_temperatures_c", metrics(telemetry.board_temperatures_c)},
          {"cpu_usage_pct", metric_json(telemetry.cpu_usage_pct)},
          {"cpu_cores_usage_pct", metrics(telemetry.cpu_cores_usage_pct)},
          {"mla_memory_mib", metric_json(telemetry.mla_memory_mib)},
          {"cma_memory_mib", metric_json(telemetry.cma_memory_mib)},
          {"linux_memory_used_mib", metric_json(telemetry.linux_memory_used_mib)}};
}

} // namespace

std::filesystem::path BenchmarkReport::save(const std::filesystem::path& file_path) const {
  const auto path = file_path.string();
  if (path.empty() || file_path.filename().empty() || path.find('\0') != std::string::npos) {
    throw std::invalid_argument("BenchmarkReport::save: provide a destination file path");
  }
  require_finite_nonnegative(latency_ms, "latency_ms");
  require_finite_nonnegative(latency_p50_ms, "latency_p50_ms");
  require_finite_nonnegative(latency_p95_ms, "latency_p95_ms");
  require_finite_nonnegative(latency_p99_ms, "latency_p99_ms");
  require_finite_nonnegative(peak_power_watts, "peak_power_watts");
  require_finite_nonnegative(throughput_duration_ms, "throughput_duration_ms");
  if (throughput_samples < 0)
    throw std::invalid_argument("BenchmarkReport::save: throughput_samples must be nonnegative");
  require_finite_nonnegative(fps, "fps");
  require_finite_nonnegative(avg_power_watts, "avg_power_watts");
  require_finite_nonnegative(energy_joules, "energy_joules");

  std::ostringstream yaml;
  yaml.imbue(std::locale::classic());
  yaml << "# NEAT benchmark report (YAML 1.2).\n"
       << "# Latency: ms; throughput: logical inferences/s; power: W; energy: J.\n"
       << "latency_ms: " << yaml_number(latency_ms) << '\n'
       << "latency_p50_ms: " << yaml_number(latency_p50_ms) << '\n'
       << "latency_p95_ms: " << yaml_number(latency_p95_ms) << '\n'
       << "latency_p99_ms: " << yaml_number(latency_p99_ms) << '\n'
       << "peak_power_watts: " << yaml_number(peak_power_watts) << '\n'
       << "throughput_samples: " << throughput_samples << '\n'
       << "throughput_duration_ms: " << yaml_number(throughput_duration_ms) << '\n'
       << "telemetry: ";
  write_yaml_flow(yaml, telemetry_json(telemetry));
  yaml << '\n'
       << "fps: " << yaml_number(fps) << '\n'
       << "avg_power_watts: " << yaml_number(avg_power_watts) << '\n'
       << "energy_joules: " << yaml_number(energy_joules) << '\n'
       << "# Layer times: hardware run_time in microseconds for compiled MLA layers.\n"
       << "# Separate profiling passes use mla-rt default input buffers. Their sum is not\n"
       << "# the headline latency; instrumentation and inputs differ from the NEAT runs.\n";
  if (layer_stats.empty()) {
    yaml << "layer_stats: []\n";
  } else {
    yaml << "layer_stats:\n";
    for (const auto& layer : layer_stats) {
      require_finite_nonnegative(layer.total_execution_time_us, "total_execution_time_us");
      yaml << "  - stage_index: " << layer.stage_index << '\n'
           << "    name: " << yaml_string(layer.name) << '\n'
           << "    total_execution_time_us: " << yaml_number(layer.total_execution_time_us) << '\n';
    }
  }
  const auto contents = yaml.str();
  if (!file_path.parent_path().empty()) {
    std::filesystem::create_directories(file_path.parent_path());
  }
  GError* error = nullptr;
  if (!g_file_set_contents_full(path.c_str(), contents.data(), contents.size(),
                                static_cast<GFileSetContentsFlags>(G_FILE_SET_CONTENTS_CONSISTENT |
                                                                   G_FILE_SET_CONTENTS_DURABLE),
                                0666, &error)) {
    const std::string detail = error ? error->message : "unknown write error";
    g_clear_error(&error);
    throw std::runtime_error("BenchmarkReport::save: cannot write '" + path + "': " + detail);
  }
  return file_path;
}

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
  // BenchmarkOptions describes a per-run buffer contract. Keep deprecated construction-time
  // BoxDecode geometry from becoming authoritative element properties and masking that metadata.
  options.boxdecode_original_width = 0;
  options.boxdecode_original_height = 0;
  options.boxdecode_resize_mode.reset();
  return options;
}

} // namespace internal

namespace {

[[noreturn]] void profile_error(const std::string& message) {
  throw std::runtime_error("Model::benchmark layer stats: " + message);
}

std::string read_stats_file(const std::filesystem::path& path) {
  constexpr std::size_t maximum = 8U * 1024U * 1024U;
  std::ifstream input(path, std::ios::binary);
  if (!input)
    profile_error("cannot read " + path.string());
  std::string text(maximum + 1, '\0');
  input.read(text.data(), static_cast<std::streamsize>(text.size()));
  text.resize(static_cast<std::size_t>(input.gcount()));
  if (input.bad())
    profile_error("failed reading " + path.string());
  if (text.size() > maximum)
    profile_error("stats YAML exceeds 8 MiB: " + path.string());
  return text;
}

struct StatsDocument {
  yaml_parser_t parser{};
  yaml_document_t document{};
  bool loaded = false;
  StatsDocument() {
    if (!yaml_parser_initialize(&parser))
      profile_error("cannot initialize YAML parser");
  }
  ~StatsDocument() {
    if (loaded)
      yaml_document_delete(&document);
    yaml_parser_delete(&parser);
  }
};

std::string stats_scalar(yaml_node_t* node) {
  if (!node || node->type != YAML_SCALAR_NODE)
    profile_error("expected a YAML scalar");
  return {reinterpret_cast<const char*>(node->data.scalar.value), node->data.scalar.length};
}

std::uint64_t stats_integer(const std::string& text) {
  std::uint64_t value{};
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size())
    profile_error("invalid layer ID or cycle counter: " + text);
  return value;
}

struct ProfileRow {
  std::string name;
  std::uint64_t start = 0;
  std::uint64_t end = 0;
  double run_time = 0;
};

std::map<std::uint64_t, ProfileRow> parse_stats_rows(const std::filesystem::path& path,
                                                     bool measured) {
  const std::string text = read_stats_file(path);
  StatsDocument yaml;
  yaml_parser_set_input_string(&yaml.parser, reinterpret_cast<const unsigned char*>(text.data()),
                               text.size());
  if (!yaml_parser_load(&yaml.parser, &yaml.document)) {
    profile_error("invalid YAML in " + path.string() + ": " +
                  (yaml.parser.problem ? yaml.parser.problem : "parse failed"));
  }
  yaml.loaded = true;
  auto* root = yaml_document_get_root_node(&yaml.document);
  if (!root || root->type != YAML_MAPPING_NODE)
    profile_error("stats YAML must be a mapping");
  std::map<std::uint64_t, ProfileRow> rows;
  for (auto* pair = root->data.mapping.pairs.start; pair != root->data.mapping.pairs.top; ++pair) {
    const auto id = stats_integer(stats_scalar(yaml_document_get_node(&yaml.document, pair->key)));
    auto* layer = yaml_document_get_node(&yaml.document, pair->value);
    if (!layer || layer->type != YAML_MAPPING_NODE)
      profile_error("layer must be a mapping");
    std::map<std::string, std::string> fields;
    for (auto* field = layer->data.mapping.pairs.start; field != layer->data.mapping.pairs.top;
         ++field) {
      const auto key = stats_scalar(yaml_document_get_node(&yaml.document, field->key));
      if (!fields.emplace(key, stats_scalar(yaml_document_get_node(&yaml.document, field->value)))
               .second)
        profile_error("duplicate layer field: " + key);
    }
    const auto required = [&](const char* key) -> const std::string& {
      const auto found = fields.find(key);
      if (found == fields.end())
        profile_error(std::string("missing ") + key + " on layer " + std::to_string(id));
      return found->second;
    };
    ProfileRow row;
    row.name = required("name");
    if (row.name.empty())
      profile_error("empty layer name");
    if (measured || (id != 0 && row.name.find("placeholder") == std::string::npos)) {
      row.start = stats_integer(required("start_cycle"));
      row.end = stats_integer(required("end_cycle"));
    }
    if (measured) {
      const auto& time = required("run_time");
      if (!time.ends_with("us"))
        profile_error("run_time must have microsecond units: " + time);
      const auto result = std::from_chars(time.data(), time.data() + time.size() - 2, row.run_time);
      if (result.ec != std::errc{} || result.ptr != time.data() + time.size() - 2 ||
          !std::isfinite(row.run_time) || row.run_time < 0)
        profile_error("invalid run_time: " + time);
    }
    if (!rows.emplace(id, std::move(row)).second)
      profile_error("duplicate layer ID");
  }
  if (rows.empty())
    profile_error("stats YAML contains no layers");
  // A second YAML document would otherwise be silently ignored by yaml_parser_load.
  yaml_document_t trailing{};
  if (!yaml_parser_load(&yaml.parser, &trailing))
    profile_error("invalid trailing YAML document");
  const bool extra = yaml_document_get_root_node(&trailing) != nullptr;
  yaml_document_delete(&trailing);
  if (extra)
    profile_error("stats YAML must contain exactly one document");
  return rows;
}

void validate_profile_input(const std::filesystem::path& path) {
  const auto rows = parse_stats_rows(path, false);
  std::uint64_t expected = 0;
  std::size_t layers = 0;
  for (const auto& [id, row] : rows) {
    if (id != expected++)
      profile_error("input layer IDs must be contiguous from zero");
    if (id != 0 && row.name.find("placeholder") == std::string::npos) {
      ++layers;
      if (row.start > INT32_MAX || row.end > INT32_MAX)
        profile_error("cycle range exceeds mla-rt's supported range");
    }
  }
  if (layers == 0)
    profile_error("input stats YAML has no profileable layers");
}

struct ProfileDirectory {
  std::filesystem::path path;
  ProfileDirectory() {
    GError* error = nullptr;
    gchar* directory = g_dir_make_tmp("neat-layer-stats-XXXXXX", &error);
    if (!directory) {
      const std::string detail = error ? error->message : "unknown error";
      g_clear_error(&error);
      profile_error("cannot create temporary directory: " + detail);
    }
    path = std::filesystem::absolute(directory);
    g_free(directory);
  }
  ~ProfileDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
};

struct ProfileChild {
  GPid pid = 0;
  GPid group = 0;
  int output = -1;
  int errors = -1;
  ~ProfileChild() {
    if (group > 0)
      (void)::kill(-group, SIGKILL);
    if (pid > 0) {
      while (::waitpid(pid, nullptr, 0) < 0 && errno == EINTR) {
      }
      g_spawn_close_pid(pid);
    }
    if (output >= 0)
      ::close(output);
    if (errors >= 0)
      ::close(errors);
  }
};

void profile_child_group(gpointer) {
  // Child setup uses only async-signal-safe operations (the caller can have GStreamer threads).
  if (::setpgid(0, 0) != 0)
    _exit(127);
}

std::string run_profile_process(const std::filesystem::path& mla_rt,
                                const std::filesystem::path& input_yaml,
                                const std::filesystem::path& executable,
                                std::chrono::milliseconds timeout) {
  if (timeout <= std::chrono::milliseconds::zero())
    profile_error("timeout must be positive");
  std::vector<std::string> args{
      mla_rt.string(),     "--connect",         "mip,ma",           "-vs", "-N",
      "--per-layer-stats", input_yaml.string(), executable.string()};
  std::vector<gchar*> argv;
  for (auto& arg : args)
    argv.push_back(arg.data());
  argv.push_back(nullptr);
  ProfileChild child;
  GError* error = nullptr;
  if (!g_spawn_async_with_pipes(nullptr, argv.data(), nullptr, G_SPAWN_DO_NOT_REAP_CHILD,
                                profile_child_group, nullptr, &child.pid, nullptr, &child.output,
                                &child.errors, &error)) {
    const std::string detail = error ? error->message : "unknown error";
    g_clear_error(&error);
    profile_error("cannot start mla-rt: " + detail);
  }
  child.group = child.pid;
  for (int fd : {child.output, child.errors}) {
    const int flags = ::fcntl(fd, F_GETFL);
    if (flags < 0 || ::fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0)
      profile_error("cannot configure mla-rt output capture");
  }
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  constexpr std::size_t max_diagnostics = 64U * 1024U;
  std::string diagnostics;
  bool exited = false;
  int status = 0;
  while (!exited || child.output >= 0 || child.errors >= 0) {
    if (std::chrono::steady_clock::now() >= deadline)
      profile_error("mla-rt timed out after " + std::to_string(timeout.count()) + " ms\n" +
                    diagnostics);
    pollfd descriptors[]{{child.output, POLLIN, 0}, {child.errors, POLLIN, 0}};
    const int polled = ::poll(descriptors, 2, 20);
    if (polled < 0 && errno != EINTR)
      profile_error("cannot poll mla-rt output");
    for (std::size_t i = 0; i < 2; ++i) {
      int& fd = i == 0 ? child.output : child.errors;
      if (fd < 0 || descriptors[i].revents == 0)
        continue;
      char buffer[4096];
      // A bounded drain prevents noisy output from postponing the timeout check indefinitely.
      for (int reads = 0; reads < 16; ++reads) {
        const auto bytes = ::read(fd, buffer, sizeof(buffer));
        if (bytes > 0) {
          diagnostics.append(buffer, static_cast<std::size_t>(bytes));
          if (diagnostics.size() > max_diagnostics)
            diagnostics.erase(0, diagnostics.size() - max_diagnostics);
        } else if (bytes == 0) {
          ::close(fd);
          fd = -1;
          break;
        } else if (errno == EINTR) {
          continue;
        } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
          break;
        } else {
          profile_error("cannot read mla-rt diagnostics");
        }
      }
    }
    if (!exited) {
      const auto waited = ::waitpid(child.pid, &status, WNOHANG);
      if (waited == child.pid) {
        exited = true;
        g_spawn_close_pid(child.pid);
        child.pid = 0;
      } else if (waited < 0 && errno != EINTR) {
        profile_error("cannot reap mla-rt");
      }
    }
  }
  child.group = 0;
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    const std::string result = WIFEXITED(status) ? "exit " + std::to_string(WEXITSTATUS(status))
                                                 : "signal " + std::to_string(WTERMSIG(status));
    profile_error("mla-rt failed (" + result + ")\n" + diagnostics);
  }
  return diagnostics;
}

} // namespace

std::vector<internal::LayerProfileStage>
internal::resolve_layer_profile_stages(const pipeline_internal::sima::MpkContract& contract) {
  std::vector<LayerProfileStage> stages;
  std::size_t stage_index = 0;
  for (const auto plugin_index : pipeline_internal::sima::plugins_in_execution_order(contract)) {
    const auto& plugin = contract.plugins.at(plugin_index);
    if (plugin.processor != "MLA" && plugin.processor != "A65")
      continue;
    const auto current_index = stage_index++;
    if (plugin.processor != "MLA")
      continue;
    const auto executable = pipeline_internal::sima::static_contract::resolve_mla_executable(
        contract.mpk_json_path, plugin.executable);
    if (executable.extension() != ".elf" || !std::filesystem::is_regular_file(executable))
      profile_error("missing MLA ELF for " + plugin.name + ": " + executable.string());
    const auto stats_yaml = std::filesystem::path(contract.mpk_json_path).parent_path() /
                            (executable.stem().string() + "_stats.yaml");
    if (!std::filesystem::is_regular_file(stats_yaml))
      profile_error("missing matching stats YAML: " + stats_yaml.string());
    validate_profile_input(stats_yaml);
    stages.push_back({current_index, executable, stats_yaml});
  }
  if (stages.empty())
    profile_error("model has no MLA stages to profile");
  return stages;
}

std::vector<LayerStats>
internal::collect_mla_layer_stats(const std::vector<LayerProfileStage>& stages,
                                  const std::filesystem::path& mla_rt,
                                  std::chrono::milliseconds timeout) {
  if (stages.empty())
    profile_error("model has no MLA stages to profile");
  std::vector<LayerStats> result;
  for (const auto& stage : stages) {
    validate_profile_input(stage.stats_yaml);
    const auto expected = parse_stats_rows(stage.stats_yaml, false);
    ProfileDirectory temporary;
    const auto input_yaml = temporary.path / stage.stats_yaml.filename();
    std::filesystem::copy_file(stage.stats_yaml, input_yaml);
    const auto diagnostics = run_profile_process(mla_rt, input_yaml, stage.executable, timeout);
    try {
      const auto output_yaml = temporary.path / (input_yaml.stem().string() + "_output.yaml");
      const auto actual = parse_stats_rows(output_yaml, true);
      std::size_t profiled = 0;
      bool has_nonzero_time = false;
      bool expects_work = false;
      for (const auto& [id, row] : expected) {
        // mla-rt omits layer zero and compiler placeholders. All real no-ops remain intact.
        if (id == 0 || row.name.find("placeholder") != std::string::npos)
          continue;
        ++profiled;
        const auto found = actual.find(id);
        if (found == actual.end() || found->second.name != row.name ||
            found->second.start != row.start || found->second.end != row.end)
          profile_error("mla-rt output does not match input layer " + std::to_string(id));
        const auto time = found->second.run_time;
        if (row.end < row.start && time != 0)
          profile_error("mla-rt returned nonzero time for a zero-cycle layer; update mla-rt");
        has_nonzero_time |= time > 0;
        expects_work |= row.end >= row.start;
        result.push_back({row.name, time, stage.stage_index});
      }
      if (actual.size() != profiled)
        profile_error("mla-rt output has unexpected layer records");
      if (expects_work && !has_nonzero_time)
        profile_error("mla-rt returned no hardware timings");
    } catch (const std::exception& error) {
      throw std::runtime_error(std::string(error.what()) + "\nmla-rt diagnostics:\n" + diagnostics);
    }
  }
  return result;
}

BenchmarkReport internal::run_benchmark(Model& model, const BenchmarkOptions& options) {
  validate_benchmark_options(options);
  // Benchmark calls can share the kernel scheduler, but direct MIP profiling must not overlap
  // another benchmark call in this process. Other applications require an idle devkit.
  static std::shared_mutex benchmark_gate;
  std::unique_lock exclusive(benchmark_gate, std::defer_lock);
  std::shared_lock shared(benchmark_gate, std::defer_lock);
  std::vector<LayerProfileStage> profile_stages;
  std::filesystem::path mla_rt;
  if (options.include_layer_stats) {
    exclusive.lock();
    const auto& contract = ModelAccess::pack(model).mpk_contract();
    if (!contract) {
      throw std::runtime_error("Model::benchmark: layer profiling requires an MPK manifest");
    }
    profile_stages = resolve_layer_profile_stages(*contract);
    gchar* executable = g_find_program_in_path("mla-rt");
    if (!executable) {
      throw std::runtime_error("Model::benchmark: mla-rt was not found in PATH");
    }
    mla_rt = executable;
    g_free(executable);
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
    for (int i = 0; i < num_samples; ++i) {
      if (!latency_runner.push(inputs)) {
        latency_runner.close();
        throw std::runtime_error("Model::benchmark: latency measured push failed");
      }
      Sample out = latency_runner.pull(kTimeoutMs);
      require_benchmark_output(out, "latency measured");
    }

    latency_measured = measure.stop();
    latency_runner.close();
  }

  const std::string trace_note =
      std::filesystem::path(ModelAccess::source_path(model)).filename().string();
  BenchmarkTelemetry telemetry;

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

    internal::SentinelTrace trace(trace_note);
    MeasureScope measure = runner.start_measurement(make_benchmark_measure_options(
        kTimeoutMs, logical_batch_size, include_plugin_latency, "NEAT Benchmark throughput"));
    const auto min_end =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(options.min_duration_ms);

    // The push loop decides how many inputs the window measures, so the pull thread only pulls
    // an output once its input has been pushed.
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
        // Nothing drains the outputs any more, so wake a push blocked on a full input queue.
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
      // A failed pull closes the input, which fails the push; report the cause.
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

    // End the window before stopping Sentinel: its stop request takes about 0.1 s.
    throughput_measured = measure.stop();
    telemetry = trace.finish();
    runner.close();
  }

  BenchmarkReport report = internal::build_benchmark_report_from_measurements(
      latency_measured, throughput_measured, num_samples, pushed, std::move(telemetry));
  if (options.include_layer_stats) {
    report.layer_stats = collect_mla_layer_stats(profile_stages, mla_rt);
  }
  const double batch_throughput = throughput_measured.throughput_batches_per_s;
  const BenchmarkTelemetry& board = report.telemetry;

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
  // One line per telemetry metric; metrics the board did not report are skipped. A metric that
  // stayed constant over the window prints a single value.
  const auto print_metric = [](const char* label, const BenchmarkMetric& metric, const char* unit) {
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
    std::cout << "Telemetry:     Sentinel, " << board.samples << " samples, trace "
              << board.trace_id << "\n";
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
                                                         int num_samples, int throughput_samples,
                                                         BenchmarkTelemetry telemetry) {
  require_benchmark_measure_report_window(latency_report, "latency", num_samples);
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
    // Energy for num_samples inputs at the measured rate keeps the value independent of how
    // long the throughput window ran.
    out.energy_joules = out.avg_power_watts * static_cast<double>(num_samples) /
                        throughput_report.throughput_batches_per_s;
  }
  out.telemetry = std::move(telemetry);
  return out;
}

} // namespace internal

} // namespace simaai::neat
