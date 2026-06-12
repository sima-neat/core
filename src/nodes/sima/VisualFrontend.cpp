#include "nodes/sima/VisualFrontend.h"

#include "gst/GstHelpers.h"
#include "pipeline/internal/contract/CompiledNodeContract.h"
#include "pipeline/internal/contract/ContractFacts.h"
#include "pipeline/internal/sima/TensorSemanticsUtil.h"
#include "pipeline/internal/sima/stagesemantics/ProcessCvuRuntimeConfigAdapter.h"
#include "pipeline/internal/sima/stagesemantics/ProcessCvuStageSemantics.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat {
namespace {

using RuntimeConfig = pipeline_internal::sima::stagesemantics::CompiledProcessCvuRuntimeConfig;
namespace tensorsemantics = pipeline_internal::sima::tensorsemantics;
namespace stagesemantics = pipeline_internal::sima::stagesemantics;

struct NativeVisualSpec {
  std::string graph_name;
  int graph_id = -1;
  std::vector<std::string> input_names;
  // Public/Neat logical tensor shapes.  Native EV74 visual kernels still receive
  // packed storage (for example B grayscale images as [B*H,W]), but users and
  // downstream contracts should see the batch as its own leading dimension.
  std::vector<std::vector<int>> input_shapes;
  std::vector<std::vector<int>> transport_input_shapes;
  std::vector<std::string> input_dtypes;
  std::vector<std::string> input_layouts;
  std::vector<std::string> runtime_output_names;
  std::vector<std::string> published_output_names;
  // Public/Neat logical output shapes; transport_output_shapes are the processcvu
  // buffer descriptors exposed to ConfigManager/dispatcher.
  std::vector<std::vector<int>> output_shapes;
  std::vector<std::vector<int>> transport_output_shapes;
  std::vector<std::string> output_dtypes;
  std::vector<std::string> output_layouts;
  std::vector<std::string> logical_output_layouts;
  std::string primary_output_name;
};

void require_positive(const char* owner, const char* field, int value) {
  if (value <= 0) {
    throw std::runtime_error(std::string(owner) + ": " + field + " must be positive");
  }
}

void require_non_negative(const char* owner, const char* field, int value) {
  if (value < 0) {
    throw std::runtime_error(std::string(owner) + ": " + field + " must be non-negative");
  }
}

void require_range(const char* owner, const char* field, int value, int min_value, int max_value) {
  if (value < min_value || value > max_value) {
    throw std::runtime_error(std::string(owner) + ": " + field + " is out of range");
  }
}

void require_non_empty(const char* owner, const char* field, const std::string& value) {
  if (value.empty()) {
    throw std::runtime_error(std::string(owner) + ": " + field + " must not be empty");
  }
}

void require_distinct(const char* owner, const std::vector<std::string>& names) {
  for (std::size_t i = 0; i < names.size(); ++i) {
    for (std::size_t j = i + 1U; j < names.size(); ++j) {
      if (names[i] == names[j]) {
        throw std::runtime_error(std::string(owner) + ": tensor port names must be distinct");
      }
    }
  }
}

constexpr int kVisualMaxDim = 4096;
constexpr int kFastMinDim = 7;
constexpr int kFastBriefMinDim = 31;
constexpr int kFastBriefMaxDim = 2048;
constexpr int kVisualMaxBatch = 64;
constexpr int kTrackerMaxLevel = 4;
constexpr int kTrackerMaxWinHalf = 15;
constexpr std::uint64_t kMaxBufferBytes = std::numeric_limits<std::uint32_t>::max();
constexpr std::uint64_t kMaxShapeDim = static_cast<std::uint64_t>(std::numeric_limits<int>::max());

std::uint64_t checked_mul_u64(const char* owner, const char* field, std::uint64_t a,
                              std::uint64_t b) {
  if (a != 0U && b > std::numeric_limits<std::uint64_t>::max() / a) {
    throw std::runtime_error(std::string(owner) + ": " + field + " overflows uint64");
  }
  return a * b;
}

void require_u32_buffer_bytes(const char* owner, const char* field, std::uint64_t bytes) {
  if (bytes == 0U || bytes > kMaxBufferBytes) {
    throw std::runtime_error(std::string(owner) + ": " + field +
                             " buffer size is outside the EV74 dispatcher envelope");
  }
}

int checked_shape_dim(const char* owner, const char* field, std::uint64_t value) {
  if (value == 0U || value > kMaxShapeDim) {
    throw std::runtime_error(std::string(owner) + ": " + field +
                             " shape dimension is outside int32 range");
  }
  return static_cast<int>(value);
}

int packed_batch_height(const char* owner, int height, int batch_size) {
  return checked_shape_dim(owner, "packed_batch_height",
                           checked_mul_u64(owner, "packed_batch_height",
                                           static_cast<std::uint64_t>(height),
                                           static_cast<std::uint64_t>(batch_size)));
}

std::uint64_t feature_list_ints(const char* owner, int max_features) {
  require_positive(owner, "max_features", max_features);
  return 1ULL + checked_mul_u64(owner, "feature_list_width",
                                static_cast<std::uint64_t>(max_features), 3ULL);
}

void require_image_shape(const char* owner, int width, int height, int min_dim, int max_dim) {
  require_range(owner, "width", width, min_dim, max_dim);
  require_range(owner, "height", height, min_dim, max_dim);
}

void require_batch(const char* owner, int batch_size) {
  require_range(owner, "batch_size", batch_size, 1, kVisualMaxBatch);
}

void require_common_options(const char* owner, const VisualFrontendCommonOptions& opt, int min_dim,
                            int max_dim) {
  require_image_shape(owner, opt.width, opt.height, min_dim, max_dim);
  require_batch(owner, opt.batch_size);
  require_range(owner, "debug", opt.debug, 0, 2);
  require_non_negative(owner, "num_buffers", opt.num_buffers);
  const auto image_bytes = checked_mul_u64(owner, "input_image_size",
                                           checked_mul_u64(owner, "input_image_size",
                                                           static_cast<std::uint64_t>(opt.width),
                                                           static_cast<std::uint64_t>(opt.height)),
                                           static_cast<std::uint64_t>(opt.batch_size));
  require_u32_buffer_bytes(owner, "input_image", image_bytes);
}

void require_feature_list_envelope(const char* owner, int max_features, int batch_size,
                                   std::uint64_t records_per_feature, bool descriptors) {
  const auto feature_ints = feature_list_ints(owner, max_features);
  (void)checked_shape_dim(owner, "feature_list_width", feature_ints);
  const auto feature_bytes = checked_mul_u64(
      owner, "output_features_size",
      checked_mul_u64(owner, "output_features_size", feature_ints, sizeof(std::int32_t)),
      static_cast<std::uint64_t>(batch_size));
  require_u32_buffer_bytes(owner, "output_features", feature_bytes);

  const auto scratch_ints =
      1ULL + checked_mul_u64(owner, "scratch_records", static_cast<std::uint64_t>(max_features),
                             records_per_feature);
  const auto scratch_bytes = checked_mul_u64(
      owner, "scratch_size",
      checked_mul_u64(owner, "scratch_size", 64ULL,
                      checked_mul_u64(owner, "scratch_size", scratch_ints, sizeof(std::int32_t))),
      static_cast<std::uint64_t>(batch_size));
  require_u32_buffer_bytes(owner, "scratch", scratch_bytes);

  if (descriptors) {
    const auto descriptor_rows =
        checked_mul_u64(owner, "descriptor_rows", static_cast<std::uint64_t>(batch_size),
                        static_cast<std::uint64_t>(max_features));
    (void)checked_shape_dim(owner, "descriptor_rows", descriptor_rows);
    const auto descriptor_bytes =
        checked_mul_u64(owner, "output_descriptors_size",
                        checked_mul_u64(owner, "output_descriptors_size", descriptor_rows, 8ULL),
                        sizeof(std::int32_t));
    require_u32_buffer_bytes(owner, "output_descriptors", descriptor_bytes);
  }
}

std::uint64_t track_klt_pyramid_bytes(const char* owner, int width, int height, int max_level) {
  std::uint64_t bytes = 0U;
  int lw = width;
  int lh = height;
  for (int level = 1; level <= max_level; ++level) {
    lw = (lw + 1) / 2;
    lh = (lh + 1) / 2;
    bytes += checked_mul_u64(owner, "pyramid_level_size", static_cast<std::uint64_t>(lw),
                             static_cast<std::uint64_t>(lh));
  }
  return checked_mul_u64(owner, "pyramid_size", bytes, 2ULL);
}

void require_track_klt_envelope(const TrackKLTOptions& opt) {
  constexpr const char* owner = "TrackKLT";
  require_image_shape(owner, opt.width, opt.height, kFastMinDim, kVisualMaxDim);
  require_batch(owner, opt.batch_size);
  require_positive(owner, "num_points", opt.num_points);
  require_range(owner, "win_half", opt.win_half, 1, kTrackerMaxWinHalf);
  require_positive(owner, "max_iters", opt.max_iters);
  require_range(owner, "max_level", opt.max_level, 0, kTrackerMaxLevel);
  require_range(owner, "detect_new_features", opt.detect_new_features, 0, 1);
  require_range(owner, "fast_threshold", opt.fast_threshold, 0, 255);
  require_positive(owner, "max_features", opt.max_features);
  require_positive(owner, "grid_x", opt.grid_x);
  require_positive(owner, "grid_y", opt.grid_y);
  require_non_negative(owner, "min_px_dist", opt.min_px_dist);
  require_range(owner, "debug", opt.debug, 0, 2);
  require_non_negative(owner, "num_buffers", opt.num_buffers);

  const auto image_bytes =
      checked_mul_u64(owner, "image_size",
                      checked_mul_u64(owner, "image_size", static_cast<std::uint64_t>(opt.width),
                                      static_cast<std::uint64_t>(opt.height)),
                      static_cast<std::uint64_t>(opt.batch_size));
  require_u32_buffer_bytes(owner, "prev_image", image_bytes);
  require_u32_buffer_bytes(owner, "cur_image", image_bytes);
  const auto point_count =
      checked_mul_u64(owner, "point_count", static_cast<std::uint64_t>(opt.num_points),
                      static_cast<std::uint64_t>(opt.batch_size));
  require_u32_buffer_bytes(
      owner, "input_points",
      checked_mul_u64(owner, "input_points_size", point_count, 2ULL * sizeof(std::int32_t)));
  require_u32_buffer_bytes(
      owner, "output_points",
      checked_mul_u64(owner, "output_points_size", point_count, 2ULL * sizeof(float)));
  require_u32_buffer_bytes(
      owner, "output_status",
      checked_mul_u64(owner, "output_status_size", point_count, sizeof(std::int32_t)));
  const auto feature_ints = feature_list_ints(owner, opt.max_features);
  (void)checked_shape_dim(owner, "feature_list_width", feature_ints);
  require_u32_buffer_bytes(owner, "output_features",
                           checked_mul_u64(owner, "output_features_size",
                                           checked_mul_u64(owner, "output_features_size",
                                                           feature_ints, sizeof(std::int32_t)),
                                           static_cast<std::uint64_t>(opt.batch_size)));
  const auto fast_scratch = checked_mul_u64(
      owner, "scratch_fast_size", 64ULL,
      checked_mul_u64(owner, "scratch_fast_size", feature_ints, sizeof(std::int32_t)));
  const auto scratch = checked_mul_u64(
      owner, "scratch_size",
      track_klt_pyramid_bytes(owner, opt.width, opt.height, opt.max_level) + fast_scratch,
      static_cast<std::uint64_t>(opt.batch_size));
  require_u32_buffer_bytes(owner, "scratch", scratch);
}

std::string default_element_name(int node_index, const char* suffix) {
  return "n" + std::to_string(node_index) + "_" + suffix;
}

std::string quoted_or_auto(const std::string& value) {
  return value.empty() ? std::string("<auto>") : ("\"" + value + "\"");
}

void append_common_summary(std::ostringstream& ss, const VisualFrontendCommonOptions& opt) {
  ss << "width=" << opt.width << ",height=" << opt.height << ",batch_size=" << opt.batch_size
     << ",debug=" << opt.debug << ",num_buffers=" << opt.num_buffers
     << ",element_name=" << quoted_or_auto(opt.element_name);
}

void append_fast_summary(std::ostringstream& ss, int threshold, int max_features, int grid_x,
                         int grid_y, int min_px_dist) {
  ss << ",threshold=" << threshold << ",max_features=" << max_features << ",grid=(" << grid_x << ","
     << grid_y << ")"
     << ",min_px_dist=" << min_px_dist;
}

std::uint64_t dtype_size_bytes(const std::string& raw) {
  std::string token = raw;
  std::transform(token.begin(), token.end(), token.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  if (token == "INT16" || token == "UINT16" || token == "FP16" || token == "BF16" ||
      token == "EVXX_BFLOAT16" || token == "EVXX_FLOAT16") {
    return 2U;
  }
  if (token == "INT32" || token == "UINT32" || token == "FP32" || token == "FLOAT32" ||
      token == "EVXX_FLOAT32") {
    return 4U;
  }
  return 1U;
}

std::uint64_t shape_bytes(const std::vector<int>& shape, const std::string& dtype) {
  std::uint64_t elems = 1U;
  for (const int dim : shape) {
    if (dim <= 0) {
      return 0U;
    }
    elems *= static_cast<std::uint64_t>(dim);
  }
  return elems * dtype_size_bytes(dtype);
}

sima_ev_tensor_desc build_contract_tensor_desc(const std::vector<int>& shape,
                                               const std::string& dtype, const std::string& layout,
                                               const char* owner) {
  sima_ev_tensor_desc desc{};
  std::string err;
  bool ok = false;
  if (layout.empty()) {
    ok = tensorsemantics::build_generic_dense_tensor_desc(
        shape, dtype, &desc, &err, "native_visual_tensor_output_missing",
        "native_visual_tensor_rank_invalid", "native_visual_tensor_dim_invalid",
        "native_visual_tensor_dtype_invalid", "native_visual_tensor_stride_output_missing");
  } else {
    ok = tensorsemantics::build_dense_tensor_desc(
        shape, dtype, layout, &desc, &err, "native_visual_tensor_output_missing",
        "native_visual_tensor_rank_invalid", "native_visual_tensor_dim_invalid",
        "native_visual_tensor_dtype_invalid", "native_visual_tensor_stride_output_missing");
  }
  if (!ok) {
    throw std::runtime_error(std::string(owner) + ": failed to build tensor descriptor" +
                             (err.empty() ? std::string() : ": " + err));
  }
  return desc;
}

RuntimeConfig make_runtime_base(const NativeVisualSpec& spec) {
  const auto& transport_input_shapes =
      spec.transport_input_shapes.empty() ? spec.input_shapes : spec.transport_input_shapes;
  const auto& transport_output_shapes =
      spec.transport_output_shapes.empty() ? spec.output_shapes : spec.transport_output_shapes;
  const auto& logical_output_layouts =
      spec.logical_output_layouts.empty() ? spec.output_layouts : spec.logical_output_layouts;
  if (spec.input_names.empty() || spec.input_names.size() != spec.input_shapes.size() ||
      spec.input_names.size() != transport_input_shapes.size() ||
      spec.input_names.size() != spec.input_dtypes.size() ||
      spec.input_names.size() != spec.input_layouts.size()) {
    throw std::runtime_error("native visual spec input arrays are inconsistent");
  }
  if (spec.runtime_output_names.empty() ||
      spec.runtime_output_names.size() != spec.output_shapes.size() ||
      spec.runtime_output_names.size() != transport_output_shapes.size() ||
      spec.runtime_output_names.size() != spec.output_dtypes.size() ||
      spec.runtime_output_names.size() != spec.output_layouts.size() ||
      spec.runtime_output_names.size() != logical_output_layouts.size()) {
    throw std::runtime_error("native visual spec output arrays are inconsistent");
  }

  RuntimeConfig runtime;
  runtime.graph_family = spec.graph_name;
  runtime.graph_name = spec.graph_name;
  runtime.graph_id = spec.graph_id;
  runtime.default_input_name = spec.input_names.front();
  runtime.runtime_input_names = spec.input_names;
  runtime.physical_input_names = spec.input_names;
  runtime.runtime_output_names = spec.runtime_output_names;
  runtime.physical_output_names = spec.runtime_output_names;
  runtime.published_output_names =
      spec.published_output_names.empty() ? spec.runtime_output_names : spec.published_output_names;
  runtime.primary_output_name = spec.primary_output_name.empty() ? spec.runtime_output_names.front()
                                                                 : spec.primary_output_name;
  runtime.primary_output_transport_kind =
      pipeline_internal::sima::ProcessCvuOutputTransportKind::Dense;
  runtime.primary_output_semantic_kind =
      pipeline_internal::sima::ProcessCvuOutputSemanticKind::Tensor;
  runtime.input_shapes = spec.input_shapes;
  runtime.output_shapes = spec.output_shapes;
  runtime.runtime_output_logical_shapes = spec.output_shapes;
  runtime.input_dtype = spec.input_dtypes.front();
  runtime.output_dtype = spec.output_dtypes.front();
  runtime.out_dtype = runtime.output_dtype;
  runtime.runtime_output_dtype_list = spec.output_dtypes;
  runtime.runtime_output_logical_layout_list = logical_output_layouts;
  runtime.runtime_output_transport_kind_list.assign(
      spec.runtime_output_names.size(),
      pipeline_internal::sima::ProcessCvuOutputTransportKind::Dense);
  runtime.runtime_output_semantic_kind_list.assign(
      spec.runtime_output_names.size(),
      pipeline_internal::sima::ProcessCvuOutputSemanticKind::Tensor);
  runtime.runtime_output_logical_index_list.reserve(spec.runtime_output_names.size());
  runtime.runtime_output_output_slot_list.reserve(spec.runtime_output_names.size());
  runtime.runtime_output_physical_index_list.reserve(spec.runtime_output_names.size());
  for (std::size_t i = 0; i < spec.runtime_output_names.size(); ++i) {
    runtime.runtime_output_logical_index_list.push_back(static_cast<int>(i));
    runtime.runtime_output_output_slot_list.push_back(static_cast<int>(i));
    runtime.runtime_output_physical_index_list.push_back(static_cast<int>(i));
  }

  runtime.input_tensors.reserve(spec.input_names.size());
  for (std::size_t i = 0; i < spec.input_names.size(); ++i) {
    runtime.input_tensors.push_back(
        build_contract_tensor_desc(transport_input_shapes[i], spec.input_dtypes[i],
                                   spec.input_layouts[i], spec.graph_name.c_str()));
  }
  runtime.output_tensors.reserve(spec.runtime_output_names.size());
  for (std::size_t i = 0; i < spec.runtime_output_names.size(); ++i) {
    runtime.output_tensors.push_back(
        build_contract_tensor_desc(transport_output_shapes[i], spec.output_dtypes[i],
                                   spec.output_layouts[i], spec.graph_name.c_str()));
  }
  return runtime;
}

NodeContractDefinition make_contract_definition(const std::string& node_kind,
                                                const std::vector<std::string>& input_names,
                                                const std::vector<std::string>& output_names) {
  NodeContractDefinition def;
  def.node_kind = node_kind;
  def.plugin_kind = "processcvu";
  for (const auto& name : input_names) {
    ContractPortSpec input;
    input.port_id = name;
    input.media_type = "application/vnd.simaai.tensor";
    input.required_segment_names = {name};
    def.inputs.push_back(std::move(input));
  }
  for (const auto& name : output_names) {
    ContractPortSpec output;
    output.port_id = name;
    output.media_type = "application/vnd.simaai.tensor";
    output.required_segment_names = {name};
    def.outputs.push_back(std::move(output));
  }
  return def;
}

bool compile_runtime_contract(const std::string& node_kind, const std::string& element_name,
                              const NodeContractDefinition& definition,
                              const RuntimeConfig& runtime, CompiledNodeContract* out,
                              std::string* err) {
  try {
    const auto compiled =
        stagesemantics::build_processcvu_compiled_contract_from_runtime_config(runtime);
    return stagesemantics::build_processcvu_node_contract(node_kind, element_name, element_name,
                                                          definition, compiled, out, err);
  } catch (const std::exception& ex) {
    if (err) {
      *err = ex.what();
    }
    return false;
  }
}

std::string processcvu_backend_fragment(const std::string& element_name, int num_buffers) {
  require_element("neatprocesscvu", "VisualFrontend::backend_fragment");
  std::ostringstream ss;
  ss << "neatprocesscvu name=" << element_name << " stage-id=" << element_name;
  if (num_buffers > 0) {
    ss << " num-buffers=" << num_buffers;
  }
  return ss.str();
}

OutputSpec tensor_output_spec(const std::string& format, const std::vector<int>& shape,
                              const std::string& dtype, const OutputSpec& input,
                              std::string layout = {}) {
  OutputSpec out;
  out.media_type = "application/vnd.simaai.tensor";
  out.format = format;
  out.dtype = dtype;
  out.layout = !layout.empty()
                   ? std::move(layout)
                   : (shape.size() == 2U ? "HW" : (shape.size() >= 3U ? "HWC" : std::string{}));
  out.memory = input.memory.empty() ? "SimaAI" : input.memory;
  out.certainty = SpecCertainty::Derived;
  out.note = "neatprocesscvu";
  out.byte_size = shape_bytes(shape, dtype);
  std::string layout_upper = out.layout;
  std::transform(layout_upper.begin(), layout_upper.end(), layout_upper.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  if (layout_upper == "HW" && shape.size() >= 2U) {
    out.height = shape[shape.size() - 2U];
    out.width = shape.back();
    out.depth = 1;
  } else if (layout_upper == "CHW" && shape.size() >= 3U) {
    out.depth = shape[shape.size() - 3U];
    out.height = shape[shape.size() - 2U];
    out.width = shape.back();
  } else if (shape.size() >= 3U) {
    out.height = shape[shape.size() - 3U];
    out.width = shape[shape.size() - 2U];
    out.depth = shape.back();
  } else if (shape.size() == 2U) {
    out.height = shape[0];
    out.width = shape[1];
    out.depth = 1;
  } else if (shape.size() == 1U) {
    out.height = 1;
    out.width = shape.front();
    out.depth = 1;
  }
  return out;
}

NativeVisualSpec spec_from_options(const FeatureHistogramOptions& opt) {
  require_common_options("FeatureHistogram", opt, 1, kVisualMaxDim);
  require_non_empty("FeatureHistogram", "input_name", opt.input_name);
  require_non_empty("FeatureHistogram", "output_name", opt.output_name);
  require_distinct("FeatureHistogram", {opt.input_name, opt.output_name});
  require_u32_buffer_bytes("FeatureHistogram", "output_hist",
                           256ULL * sizeof(std::int32_t) *
                               static_cast<std::uint64_t>(opt.batch_size));
  require_u32_buffer_bytes("FeatureHistogram", "scratch",
                           64ULL * 256ULL * sizeof(std::int32_t) *
                               static_cast<std::uint64_t>(opt.batch_size));
  NativeVisualSpec spec;
  spec.graph_name = "feature_histogram";
  spec.graph_id = 235;
  spec.input_names = {opt.input_name};
  spec.input_shapes = {{opt.batch_size, opt.height, opt.width}};
  spec.transport_input_shapes = {
      {packed_batch_height("FeatureHistogram", opt.height, opt.batch_size), opt.width}};
  spec.input_dtypes = {"UINT8"};
  spec.input_layouts = {"HW"};
  spec.runtime_output_names = {opt.output_name};
  spec.published_output_names = {opt.output_name};
  spec.output_shapes = {{opt.batch_size, 256}};
  spec.transport_output_shapes = spec.output_shapes;
  spec.output_dtypes = {"INT32"};
  spec.output_layouts = {"HW"};
  spec.logical_output_layouts = {"HW"};
  spec.primary_output_name = opt.output_name;
  return spec;
}

NativeVisualSpec spec_from_options(const GriderFastOptions& opt) {
  require_common_options("GriderFast", opt, kFastMinDim, kVisualMaxDim);
  require_positive("GriderFast", "max_features", opt.max_features);
  require_positive("GriderFast", "grid_x", opt.grid_x);
  require_positive("GriderFast", "grid_y", opt.grid_y);
  require_range("GriderFast", "threshold", opt.threshold, 0, 255);
  require_non_negative("GriderFast", "min_px_dist", opt.min_px_dist);
  require_non_empty("GriderFast", "input_name", opt.input_name);
  require_non_empty("GriderFast", "output_name", opt.output_name);
  require_distinct("GriderFast", {opt.input_name, opt.output_name});
  require_feature_list_envelope("GriderFast", opt.max_features, opt.batch_size, 3ULL, false);
  const auto feature_width = checked_shape_dim("GriderFast", "feature_list_width",
                                               feature_list_ints("GriderFast", opt.max_features));
  NativeVisualSpec spec;
  spec.graph_name = "grider_fast";
  spec.graph_id = 236;
  spec.input_names = {opt.input_name};
  spec.input_shapes = {{opt.batch_size, opt.height, opt.width}};
  spec.transport_input_shapes = {
      {packed_batch_height("GriderFast", opt.height, opt.batch_size), opt.width}};
  spec.input_dtypes = {"UINT8"};
  spec.input_layouts = {"HW"};
  spec.runtime_output_names = {opt.output_name};
  spec.published_output_names = {opt.output_name};
  spec.output_shapes = {{opt.batch_size, feature_width}};
  spec.transport_output_shapes = spec.output_shapes;
  spec.output_dtypes = {"INT32"};
  spec.output_layouts = {"HW"};
  spec.logical_output_layouts = {"HW"};
  spec.primary_output_name = opt.output_name;
  return spec;
}

NativeVisualSpec spec_from_options(const TrackDescriptorOptions& opt) {
  require_common_options("TrackDescriptor", opt, kFastBriefMinDim, kFastBriefMaxDim);
  require_positive("TrackDescriptor", "max_features", opt.max_features);
  require_positive("TrackDescriptor", "grid_x", opt.grid_x);
  require_positive("TrackDescriptor", "grid_y", opt.grid_y);
  require_range("TrackDescriptor", "threshold", opt.threshold, 0, 255);
  require_non_negative("TrackDescriptor", "min_px_dist", opt.min_px_dist);
  if (opt.descriptor_words != 8) {
    throw std::runtime_error(
        "TrackDescriptor: descriptor_words must be 8 for the current EV74 ABI");
  }
  require_non_empty("TrackDescriptor", "input_name", opt.input_name);
  require_non_empty("TrackDescriptor", "features_output_name", opt.features_output_name);
  require_non_empty("TrackDescriptor", "descriptors_output_name", opt.descriptors_output_name);
  require_distinct("TrackDescriptor",
                   {opt.input_name, opt.features_output_name, opt.descriptors_output_name});
  require_feature_list_envelope("TrackDescriptor", opt.max_features, opt.batch_size, 11ULL, true);
  const auto feature_width =
      checked_shape_dim("TrackDescriptor", "feature_list_width",
                        feature_list_ints("TrackDescriptor", opt.max_features));
  const auto descriptor_rows =
      checked_shape_dim("TrackDescriptor", "descriptor_rows",
                        checked_mul_u64("TrackDescriptor", "descriptor_rows",
                                        static_cast<std::uint64_t>(opt.batch_size),
                                        static_cast<std::uint64_t>(opt.max_features)));
  NativeVisualSpec spec;
  spec.graph_name = "track_descriptor";
  spec.graph_id = 237;
  spec.input_names = {opt.input_name};
  spec.input_shapes = {{opt.batch_size, opt.height, opt.width}};
  spec.transport_input_shapes = {
      {packed_batch_height("TrackDescriptor", opt.height, opt.batch_size), opt.width}};
  spec.input_dtypes = {"UINT8"};
  spec.input_layouts = {"HW"};
  spec.runtime_output_names = {opt.features_output_name, opt.descriptors_output_name};
  spec.published_output_names = spec.runtime_output_names;
  spec.output_shapes = {{opt.batch_size, feature_width},
                        {opt.batch_size, opt.max_features, opt.descriptor_words}};
  spec.transport_output_shapes = {{opt.batch_size, feature_width},
                                  {descriptor_rows, opt.descriptor_words}};
  spec.output_dtypes = {"INT32", "INT32"};
  spec.output_layouts = {"HW", "HW"};
  spec.logical_output_layouts = {"HW", "HW"};
  spec.primary_output_name = opt.features_output_name;
  return spec;
}

NativeVisualSpec spec_from_options(const TrackKLTOptions& opt) {
  require_track_klt_envelope(opt);
  require_non_empty("TrackKLT", "prev_image_name", opt.prev_image_name);
  require_non_empty("TrackKLT", "cur_image_name", opt.cur_image_name);
  require_non_empty("TrackKLT", "input_points_name", opt.input_points_name);
  require_non_empty("TrackKLT", "output_points_name", opt.output_points_name);
  require_non_empty("TrackKLT", "output_status_name", opt.output_status_name);
  require_non_empty("TrackKLT", "output_features_name", opt.output_features_name);
  require_distinct("TrackKLT",
                   {opt.prev_image_name, opt.cur_image_name, opt.input_points_name,
                    opt.output_points_name, opt.output_status_name, opt.output_features_name});
  const auto feature_width = checked_shape_dim("TrackKLT", "feature_list_width",
                                               feature_list_ints("TrackKLT", opt.max_features));
  const auto point_rows = checked_shape_dim(
      "TrackKLT", "point_rows",
      checked_mul_u64("TrackKLT", "point_rows", static_cast<std::uint64_t>(opt.batch_size),
                      static_cast<std::uint64_t>(opt.num_points)));
  NativeVisualSpec spec;
  spec.graph_name = "track_klt";
  spec.graph_id = 238;
  spec.input_names = {opt.prev_image_name, opt.cur_image_name, opt.input_points_name};
  spec.input_shapes = {{opt.batch_size, opt.height, opt.width},
                       {opt.batch_size, opt.height, opt.width},
                       {opt.batch_size, opt.num_points, 2}};
  spec.transport_input_shapes = {
      {packed_batch_height("TrackKLT", opt.height, opt.batch_size), opt.width},
      {packed_batch_height("TrackKLT", opt.height, opt.batch_size), opt.width},
      {point_rows, 2}};
  spec.input_dtypes = {"UINT8", "UINT8", "INT32"};
  spec.input_layouts = {"HW", "HW", "HW"};
  spec.runtime_output_names = {opt.output_points_name, opt.output_status_name,
                               opt.output_features_name};
  spec.published_output_names =
      opt.detect_new_features != 0
          ? spec.runtime_output_names
          : std::vector<std::string>{opt.output_points_name, opt.output_status_name};
  spec.output_shapes = {{opt.batch_size, opt.num_points, 2},
                        {opt.batch_size, opt.num_points, 1},
                        {opt.batch_size, feature_width}};
  spec.transport_output_shapes = {
      {point_rows, 2}, {point_rows, 1}, {opt.batch_size, feature_width}};
  spec.output_dtypes = {"FP32", "INT32", "INT32"};
  spec.output_layouts = {"HW", "HW", "HW"};
  spec.logical_output_layouts = {"HW", "HW", "HW"};
  spec.primary_output_name = opt.output_points_name;
  return spec;
}

RuntimeConfig make_runtime(const FeatureHistogramOptions& opt) {
  auto runtime = make_runtime_base(spec_from_options(opt));
  runtime.width = opt.width;
  runtime.height = opt.height;
  runtime.debug = opt.debug;
  runtime.batch_size = opt.batch_size;
  return runtime;
}

RuntimeConfig make_runtime(const GriderFastOptions& opt) {
  auto runtime = make_runtime_base(spec_from_options(opt));
  runtime.width = opt.width;
  runtime.height = opt.height;
  runtime.threshold = opt.threshold;
  runtime.max_features = opt.max_features;
  runtime.grid_x = opt.grid_x;
  runtime.grid_y = opt.grid_y;
  runtime.min_px_dist = opt.min_px_dist;
  runtime.debug = opt.debug;
  runtime.batch_size = opt.batch_size;
  return runtime;
}

RuntimeConfig make_runtime(const TrackDescriptorOptions& opt) {
  auto runtime = make_runtime_base(spec_from_options(opt));
  runtime.width = opt.width;
  runtime.height = opt.height;
  runtime.threshold = opt.threshold;
  runtime.max_features = opt.max_features;
  runtime.grid_x = opt.grid_x;
  runtime.grid_y = opt.grid_y;
  runtime.min_px_dist = opt.min_px_dist;
  runtime.descriptor_words = opt.descriptor_words;
  runtime.debug = opt.debug;
  runtime.batch_size = opt.batch_size;
  return runtime;
}

RuntimeConfig make_runtime(const TrackKLTOptions& opt) {
  auto runtime = make_runtime_base(spec_from_options(opt));
  runtime.width = opt.width;
  runtime.height = opt.height;
  runtime.num_points = opt.num_points;
  runtime.win_half = opt.win_half;
  runtime.max_iters = opt.max_iters;
  runtime.max_level = opt.max_level;
  runtime.detect_new_features = opt.detect_new_features;
  runtime.fast_threshold = opt.fast_threshold;
  runtime.max_features = opt.max_features;
  runtime.grid_x = opt.grid_x;
  runtime.grid_y = opt.grid_y;
  runtime.min_px_dist = opt.min_px_dist;
  runtime.debug = opt.debug;
  runtime.batch_size = opt.batch_size;
  return runtime;
}

} // namespace

FeatureHistogram::FeatureHistogram(FeatureHistogramOptions opt) : opt_(std::move(opt)) {}
GriderFast::GriderFast(GriderFastOptions opt) : opt_(std::move(opt)) {}
TrackDescriptor::TrackDescriptor(TrackDescriptorOptions opt) : opt_(std::move(opt)) {}
TrackKLT::TrackKLT(TrackKLTOptions opt) : opt_(std::move(opt)) {}

std::string FeatureHistogramOptions::summary() const {
  std::ostringstream ss;
  ss << "FeatureHistogramOptions(graph=feature_histogram,graph_id=235,";
  append_common_summary(ss, *this);
  ss << ",input=" << quoted_or_auto(input_name) << ",output=" << quoted_or_auto(output_name)
     << ",input_shape=[" << batch_size << "," << height << "," << width << "]"
     << ",output_shape=[" << batch_size << ",256])";
  return ss.str();
}

std::string GriderFastOptions::summary() const {
  std::ostringstream ss;
  ss << "GriderFastOptions(graph=grider_fast,graph_id=236,";
  append_common_summary(ss, *this);
  append_fast_summary(ss, threshold, max_features, grid_x, grid_y, min_px_dist);
  ss << ",input=" << quoted_or_auto(input_name) << ",output=" << quoted_or_auto(output_name)
     << ",input_shape=[" << batch_size << "," << height << "," << width << "]"
     << ",feature_shape=[" << batch_size << ","
     << (1LL + static_cast<long long>(max_features) * 3LL) << "])";
  return ss.str();
}

std::string TrackDescriptorOptions::summary() const {
  std::ostringstream ss;
  ss << "TrackDescriptorOptions(graph=track_descriptor,graph_id=237,";
  append_common_summary(ss, *this);
  append_fast_summary(ss, threshold, max_features, grid_x, grid_y, min_px_dist);
  ss << ",descriptor_words=" << descriptor_words << ",input=" << quoted_or_auto(input_name)
     << ",features=" << quoted_or_auto(features_output_name)
     << ",descriptors=" << quoted_or_auto(descriptors_output_name) << ",input_shape=[" << batch_size
     << "," << height << "," << width << "]"
     << ",feature_shape=[" << batch_size << ","
     << (1LL + static_cast<long long>(max_features) * 3LL) << "]"
     << ",descriptor_shape=[" << batch_size << "," << max_features << "," << descriptor_words
     << "])";
  return ss.str();
}

std::string TrackKLTOptions::summary() const {
  std::ostringstream ss;
  ss << "TrackKLTOptions(graph=track_klt,graph_id=238,width=" << width << ",height=" << height
     << ",batch_size=" << batch_size << ",num_points=" << num_points << ",win_half=" << win_half
     << ",max_iters=" << max_iters << ",max_level=" << max_level
     << ",detect_new_features=" << detect_new_features;
  append_fast_summary(ss, fast_threshold, max_features, grid_x, grid_y, min_px_dist);
  ss << ",debug=" << debug << ",num_buffers=" << num_buffers
     << ",element_name=" << quoted_or_auto(element_name)
     << ",prev_image=" << quoted_or_auto(prev_image_name)
     << ",cur_image=" << quoted_or_auto(cur_image_name)
     << ",input_points=" << quoted_or_auto(input_points_name)
     << ",output_points=" << quoted_or_auto(output_points_name)
     << ",output_status=" << quoted_or_auto(output_status_name)
     << ",output_features=" << quoted_or_auto(output_features_name) << ",image_shape=["
     << batch_size << "," << height << "," << width << "]"
     << ",input_points_shape=[" << batch_size << "," << num_points << ",2]"
     << ",output_points_shape=[" << batch_size << "," << num_points << ",2]"
     << ",output_status_shape=[" << batch_size << "," << num_points << ",1]";
  if (detect_new_features != 0) {
    ss << ",published_feature_shape=[" << batch_size << ","
       << (1LL + static_cast<long long>(max_features) * 3LL) << "]";
  } else {
    ss << ",published_feature_shape=<disabled>";
  }
  ss << ")";
  return ss.str();
}

NodeContractDefinition FeatureHistogram::contract_definition() const {
  const auto spec = spec_from_options(opt_);
  return make_contract_definition(kind(), spec.input_names, spec.published_output_names);
}

NodeContractDefinition GriderFast::contract_definition() const {
  const auto spec = spec_from_options(opt_);
  return make_contract_definition(kind(), spec.input_names, spec.published_output_names);
}

NodeContractDefinition TrackDescriptor::contract_definition() const {
  const auto spec = spec_from_options(opt_);
  return make_contract_definition(kind(), spec.input_names, spec.published_output_names);
}

NodeContractDefinition TrackKLT::contract_definition() const {
  const auto spec = spec_from_options(opt_);
  return make_contract_definition(kind(), spec.input_names, spec.published_output_names);
}

bool FeatureHistogram::compile_node_contract(const ContractCompileInput& input,
                                             CompiledNodeContract* out, std::string* err) const {
  const std::string name = element_names(input.node_index).front();
  return compile_runtime_contract(kind(), name, contract_definition(), make_runtime(opt_), out,
                                  err);
}

bool GriderFast::compile_node_contract(const ContractCompileInput& input, CompiledNodeContract* out,
                                       std::string* err) const {
  const std::string name = element_names(input.node_index).front();
  return compile_runtime_contract(kind(), name, contract_definition(), make_runtime(opt_), out,
                                  err);
}

bool TrackDescriptor::compile_node_contract(const ContractCompileInput& input,
                                            CompiledNodeContract* out, std::string* err) const {
  const std::string name = element_names(input.node_index).front();
  return compile_runtime_contract(kind(), name, contract_definition(), make_runtime(opt_), out,
                                  err);
}

bool TrackKLT::compile_node_contract(const ContractCompileInput& input, CompiledNodeContract* out,
                                     std::string* err) const {
  const std::string name = element_names(input.node_index).front();
  return compile_runtime_contract(kind(), name, contract_definition(), make_runtime(opt_), out,
                                  err);
}

void FeatureHistogram::apply_compiled_contract(const CompiledNodeContract&, std::string* err) {
  if (err)
    err->clear();
}
void GriderFast::apply_compiled_contract(const CompiledNodeContract&, std::string* err) {
  if (err)
    err->clear();
}
void TrackDescriptor::apply_compiled_contract(const CompiledNodeContract&, std::string* err) {
  if (err)
    err->clear();
}
void TrackKLT::apply_compiled_contract(const CompiledNodeContract&, std::string* err) {
  if (err)
    err->clear();
}

std::string FeatureHistogram::backend_fragment(int node_index) const {
  const auto names = element_names(node_index);
  return processcvu_backend_fragment(names.front(), opt_.num_buffers);
}
std::string GriderFast::backend_fragment(int node_index) const {
  const auto names = element_names(node_index);
  return processcvu_backend_fragment(names.front(), opt_.num_buffers);
}
std::string TrackDescriptor::backend_fragment(int node_index) const {
  const auto names = element_names(node_index);
  return processcvu_backend_fragment(names.front(), opt_.num_buffers);
}
std::string TrackKLT::backend_fragment(int node_index) const {
  const auto names = element_names(node_index);
  return processcvu_backend_fragment(names.front(), opt_.num_buffers);
}

std::vector<std::string> FeatureHistogram::element_names(int node_index) const {
  return {opt_.element_name.empty() ? default_element_name(node_index, "feature_histogram")
                                    : opt_.element_name};
}
std::vector<std::string> GriderFast::element_names(int node_index) const {
  return {opt_.element_name.empty() ? default_element_name(node_index, "grider_fast")
                                    : opt_.element_name};
}
std::vector<std::string> TrackDescriptor::element_names(int node_index) const {
  return {opt_.element_name.empty() ? default_element_name(node_index, "track_descriptor")
                                    : opt_.element_name};
}
std::vector<std::string> TrackKLT::element_names(int node_index) const {
  return {opt_.element_name.empty() ? default_element_name(node_index, "track_klt")
                                    : opt_.element_name};
}

OutputSpec FeatureHistogram::output_spec(const OutputSpec& input) const {
  const auto spec = spec_from_options(opt_);
  return tensor_output_spec("FEATURE_HISTOGRAM", spec.output_shapes.front(), "INT32", input);
}
OutputSpec GriderFast::output_spec(const OutputSpec& input) const {
  const auto spec = spec_from_options(opt_);
  return tensor_output_spec("FEATURE_LIST", spec.output_shapes.front(), "INT32", input);
}
OutputSpec TrackDescriptor::output_spec(const OutputSpec& input) const {
  const auto spec = spec_from_options(opt_);
  return tensor_output_spec("FEATURE_LIST", spec.output_shapes.front(), "INT32", input);
}
OutputSpec TrackKLT::output_spec(const OutputSpec& input) const {
  const auto spec = spec_from_options(opt_);
  return tensor_output_spec("TRACK_POINTS", spec.output_shapes.front(), "FP32", input, "HW");
}

} // namespace simaai::neat

namespace simaai::neat::nodes {
std::shared_ptr<simaai::neat::Node> FeatureHistogram(FeatureHistogramOptions opt) {
  return std::make_shared<simaai::neat::FeatureHistogram>(std::move(opt));
}
std::shared_ptr<simaai::neat::Node> GriderFast(GriderFastOptions opt) {
  return std::make_shared<simaai::neat::GriderFast>(std::move(opt));
}
std::shared_ptr<simaai::neat::Node> TrackDescriptor(TrackDescriptorOptions opt) {
  return std::make_shared<simaai::neat::TrackDescriptor>(std::move(opt));
}
std::shared_ptr<simaai::neat::Node> TrackKLT(TrackKLTOptions opt) {
  return std::make_shared<simaai::neat::TrackKLT>(std::move(opt));
}
} // namespace simaai::neat::nodes
