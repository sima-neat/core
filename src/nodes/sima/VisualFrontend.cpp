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
  std::vector<std::vector<int>> input_shapes;
  std::vector<std::string> input_dtypes;
  std::vector<std::string> input_layouts;
  std::vector<std::string> runtime_output_names;
  std::vector<std::string> published_output_names;
  std::vector<std::vector<int>> output_shapes;
  std::vector<std::string> output_dtypes;
  std::vector<std::string> output_layouts;
  std::string primary_output_name;
};

void require_positive(const char* owner, const char* field, int value) {
  if (value <= 0) {
    throw std::runtime_error(std::string(owner) + ": " + field + " must be positive");
  }
}

void require_range(const char* owner, const char* field, int value, int min_value, int max_value) {
  if (value < min_value || value > max_value) {
    throw std::runtime_error(std::string(owner) + ": " + field + " is out of range");
  }
}

std::string default_element_name(int node_index, const char* suffix) {
  return "n" + std::to_string(node_index) + "_" + suffix;
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
                                               const std::string& dtype,
                                               const std::string& layout,
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
  if (spec.input_names.empty() || spec.input_names.size() != spec.input_shapes.size() ||
      spec.input_names.size() != spec.input_dtypes.size() ||
      spec.input_names.size() != spec.input_layouts.size()) {
    throw std::runtime_error("native visual spec input arrays are inconsistent");
  }
  if (spec.runtime_output_names.empty() ||
      spec.runtime_output_names.size() != spec.output_shapes.size() ||
      spec.runtime_output_names.size() != spec.output_dtypes.size() ||
      spec.runtime_output_names.size() != spec.output_layouts.size()) {
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
  runtime.published_output_names = spec.published_output_names.empty()
                                       ? spec.runtime_output_names
                                       : spec.published_output_names;
  runtime.primary_output_name = spec.primary_output_name.empty() ? spec.runtime_output_names.front()
                                                                 : spec.primary_output_name;
  runtime.primary_output_transport_kind = pipeline_internal::sima::ProcessCvuOutputTransportKind::Dense;
  runtime.primary_output_semantic_kind = pipeline_internal::sima::ProcessCvuOutputSemanticKind::Tensor;
  runtime.input_shapes = spec.input_shapes;
  runtime.output_shapes = spec.output_shapes;
  runtime.runtime_output_logical_shapes = spec.output_shapes;
  runtime.input_dtype = spec.input_dtypes.front();
  runtime.output_dtype = spec.output_dtypes.front();
  runtime.out_dtype = runtime.output_dtype;
  runtime.runtime_output_dtype_list = spec.output_dtypes;
  runtime.runtime_output_logical_layout_list = spec.output_layouts;
  runtime.runtime_output_transport_kind_list.assign(
      spec.runtime_output_names.size(), pipeline_internal::sima::ProcessCvuOutputTransportKind::Dense);
  runtime.runtime_output_semantic_kind_list.assign(
      spec.runtime_output_names.size(), pipeline_internal::sima::ProcessCvuOutputSemanticKind::Tensor);
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
        build_contract_tensor_desc(spec.input_shapes[i], spec.input_dtypes[i], spec.input_layouts[i],
                                   spec.graph_name.c_str()));
  }
  runtime.output_tensors.reserve(spec.runtime_output_names.size());
  for (std::size_t i = 0; i < spec.runtime_output_names.size(); ++i) {
    runtime.output_tensors.push_back(build_contract_tensor_desc(
        spec.output_shapes[i], spec.output_dtypes[i], spec.output_layouts[i], spec.graph_name.c_str()));
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
    const auto compiled = stagesemantics::build_processcvu_compiled_contract_from_runtime_config(runtime);
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
                              const std::string& dtype, const OutputSpec& input) {
  OutputSpec out;
  out.media_type = "application/vnd.simaai.tensor";
  out.format = format;
  out.dtype = dtype;
  out.layout = shape.size() == 2U ? "HW" : (shape.size() >= 3U ? "HWC" : std::string{});
  out.memory = input.memory.empty() ? "SimaAI" : input.memory;
  out.certainty = SpecCertainty::Derived;
  out.note = "neatprocesscvu";
  out.byte_size = shape_bytes(shape, dtype);
  if (shape.size() >= 2U) {
    out.height = shape[0];
    out.width = shape[1];
  }
  if (!shape.empty()) {
    out.depth = shape.back();
  }
  return out;
}

NativeVisualSpec spec_from_options(const FeatureHistogramOptions& opt) {
  require_positive("FeatureHistogram", "width", opt.width);
  require_positive("FeatureHistogram", "height", opt.height);
  require_positive("FeatureHistogram", "batch_size", opt.batch_size);
  NativeVisualSpec spec;
  spec.graph_name = "feature_histogram";
  spec.graph_id = 235;
  spec.input_names = {opt.input_name};
  spec.input_shapes = {{opt.height, opt.width}};
  spec.input_dtypes = {"UINT8"};
  spec.input_layouts = {"HW"};
  spec.runtime_output_names = {opt.output_name};
  spec.published_output_names = {opt.output_name};
  spec.output_shapes = {{opt.batch_size, 256}};
  spec.output_dtypes = {"INT32"};
  spec.output_layouts = {"HW"};
  spec.primary_output_name = opt.output_name;
  return spec;
}

NativeVisualSpec spec_from_options(const GriderFastOptions& opt) {
  require_positive("GriderFast", "width", opt.width);
  require_positive("GriderFast", "height", opt.height);
  require_positive("GriderFast", "batch_size", opt.batch_size);
  require_positive("GriderFast", "max_features", opt.max_features);
  require_positive("GriderFast", "grid_x", opt.grid_x);
  require_positive("GriderFast", "grid_y", opt.grid_y);
  require_range("GriderFast", "threshold", opt.threshold, 0, 255);
  if (opt.min_px_dist < 0) {
    throw std::runtime_error("GriderFast: min_px_dist must be non-negative");
  }
  NativeVisualSpec spec;
  spec.graph_name = "grider_fast";
  spec.graph_id = 236;
  spec.input_names = {opt.input_name};
  spec.input_shapes = {{opt.height, opt.width}};
  spec.input_dtypes = {"UINT8"};
  spec.input_layouts = {"HW"};
  spec.runtime_output_names = {opt.output_name};
  spec.published_output_names = {opt.output_name};
  spec.output_shapes = {{opt.batch_size, 1 + opt.max_features * 3}};
  spec.output_dtypes = {"INT32"};
  spec.output_layouts = {"HW"};
  spec.primary_output_name = opt.output_name;
  return spec;
}

NativeVisualSpec spec_from_options(const TrackDescriptorOptions& opt) {
  require_positive("TrackDescriptor", "width", opt.width);
  require_positive("TrackDescriptor", "height", opt.height);
  require_positive("TrackDescriptor", "batch_size", opt.batch_size);
  require_positive("TrackDescriptor", "max_features", opt.max_features);
  require_positive("TrackDescriptor", "grid_x", opt.grid_x);
  require_positive("TrackDescriptor", "grid_y", opt.grid_y);
  require_range("TrackDescriptor", "threshold", opt.threshold, 0, 255);
  if (opt.min_px_dist < 0) {
    throw std::runtime_error("TrackDescriptor: min_px_dist must be non-negative");
  }
  if (opt.descriptor_words != 8) {
    throw std::runtime_error("TrackDescriptor: descriptor_words must be 8 for the current EV74 ABI");
  }
  NativeVisualSpec spec;
  spec.graph_name = "track_descriptor";
  spec.graph_id = 237;
  spec.input_names = {opt.input_name};
  spec.input_shapes = {{opt.height, opt.width}};
  spec.input_dtypes = {"UINT8"};
  spec.input_layouts = {"HW"};
  spec.runtime_output_names = {opt.features_output_name, opt.descriptors_output_name};
  spec.published_output_names = spec.runtime_output_names;
  spec.output_shapes = {{opt.batch_size, 1 + opt.max_features * 3},
                        {opt.batch_size * opt.max_features, opt.descriptor_words}};
  spec.output_dtypes = {"INT32", "INT32"};
  spec.output_layouts = {"HW", "HW"};
  spec.primary_output_name = opt.features_output_name;
  return spec;
}

NativeVisualSpec spec_from_options(const TrackKLTOptions& opt) {
  require_positive("TrackKLT", "width", opt.width);
  require_positive("TrackKLT", "height", opt.height);
  require_positive("TrackKLT", "num_points", opt.num_points);
  require_positive("TrackKLT", "win_half", opt.win_half);
  require_positive("TrackKLT", "max_iters", opt.max_iters);
  require_positive("TrackKLT", "max_features", opt.max_features);
  require_positive("TrackKLT", "grid_x", opt.grid_x);
  require_positive("TrackKLT", "grid_y", opt.grid_y);
  require_range("TrackKLT", "max_level", opt.max_level, 0, 4);
  require_range("TrackKLT", "detect_new_features", opt.detect_new_features, 0, 1);
  require_range("TrackKLT", "fast_threshold", opt.fast_threshold, 0, 255);
  if (opt.min_px_dist < 0) {
    throw std::runtime_error("TrackKLT: min_px_dist must be non-negative");
  }
  NativeVisualSpec spec;
  spec.graph_name = "track_klt";
  spec.graph_id = 238;
  spec.input_names = {opt.prev_image_name, opt.cur_image_name, opt.input_points_name};
  spec.input_shapes = {{opt.height, opt.width}, {opt.height, opt.width}, {opt.num_points, 2}};
  spec.input_dtypes = {"UINT8", "UINT8", "INT32"};
  spec.input_layouts = {"HW", "HW", "HW"};
  spec.runtime_output_names = {opt.output_points_name, opt.output_status_name,
                               opt.output_features_name};
  spec.published_output_names = opt.detect_new_features != 0
                                    ? spec.runtime_output_names
                                    : std::vector<std::string>{opt.output_points_name,
                                                               opt.output_status_name};
  spec.output_shapes = {{opt.num_points, 2}, {opt.num_points, 1}, {1 + opt.max_features * 3, 1}};
  spec.output_dtypes = {"FP32", "INT32", "INT32"};
  spec.output_layouts = {"HW", "HW", "HW"};
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
  runtime.batch_size = 1;
  return runtime;
}

} // namespace

FeatureHistogram::FeatureHistogram(FeatureHistogramOptions opt) : opt_(std::move(opt)) {}
GriderFast::GriderFast(GriderFastOptions opt) : opt_(std::move(opt)) {}
TrackDescriptor::TrackDescriptor(TrackDescriptorOptions opt) : opt_(std::move(opt)) {}
TrackKLT::TrackKLT(TrackKLTOptions opt) : opt_(std::move(opt)) {}

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
  return compile_runtime_contract(kind(), name, contract_definition(), make_runtime(opt_), out, err);
}

bool GriderFast::compile_node_contract(const ContractCompileInput& input, CompiledNodeContract* out,
                                       std::string* err) const {
  const std::string name = element_names(input.node_index).front();
  return compile_runtime_contract(kind(), name, contract_definition(), make_runtime(opt_), out, err);
}

bool TrackDescriptor::compile_node_contract(const ContractCompileInput& input,
                                            CompiledNodeContract* out, std::string* err) const {
  const std::string name = element_names(input.node_index).front();
  return compile_runtime_contract(kind(), name, contract_definition(), make_runtime(opt_), out, err);
}

bool TrackKLT::compile_node_contract(const ContractCompileInput& input, CompiledNodeContract* out,
                                     std::string* err) const {
  const std::string name = element_names(input.node_index).front();
  return compile_runtime_contract(kind(), name, contract_definition(), make_runtime(opt_), out, err);
}

void FeatureHistogram::apply_compiled_contract(const CompiledNodeContract&, std::string* err) {
  if (err) err->clear();
}
void GriderFast::apply_compiled_contract(const CompiledNodeContract&, std::string* err) {
  if (err) err->clear();
}
void TrackDescriptor::apply_compiled_contract(const CompiledNodeContract&, std::string* err) {
  if (err) err->clear();
}
void TrackKLT::apply_compiled_contract(const CompiledNodeContract&, std::string* err) {
  if (err) err->clear();
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
  return tensor_output_spec("FEATURE_HISTOGRAM", {opt_.batch_size, 256}, "INT32", input);
}
OutputSpec GriderFast::output_spec(const OutputSpec& input) const {
  return tensor_output_spec("FEATURE_LIST", {opt_.batch_size, 1 + opt_.max_features * 3}, "INT32",
                            input);
}
OutputSpec TrackDescriptor::output_spec(const OutputSpec& input) const {
  return tensor_output_spec("FEATURE_LIST", {opt_.batch_size, 1 + opt_.max_features * 3}, "INT32",
                            input);
}
OutputSpec TrackKLT::output_spec(const OutputSpec& input) const {
  return tensor_output_spec("TRACK_POINTS", {opt_.num_points, 2}, "FP32", input);
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
