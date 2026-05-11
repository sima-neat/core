/**
 * @file SessionBuildInput.cpp
 * @brief Session input-mode build/run methods.
 */

#include "SessionDetail.h"
#include "internal/SessionBuildInternal.h"
#include "pipeline/internal/UxLogging.h"

#include "gst/GstHelpers.h"
#include "gst/GstInit.h"

#include "builder/NodeGroup.h"
#include "builder/InputContractConfigurable.h"
#include "builder/OutputSpec.h"
#include "nodes/io/Input.h"
#include "nodes/io/RTSPInput.h"
#include "nodes/sima/Preproc.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/SessionError.h"
#include "pipeline/internal/GstDataAdapter.h"
#include "pipeline/internal/OutputTensorOverride.h"
#include "pipeline/internal/contract/ContractApply.h"
#include "pipeline/internal/contract/ContractCompiler.h"
#include "pipeline/internal/contract/ContractFacts.h"
#include "pipeline/internal/InputPolicy.h"
#include "pipeline/internal/RenderedMlaContractQuery.h"
#include "pipeline/internal/InputRouteProcessor.h"
#include "pipeline/internal/SampleUtil.h"
#include "pipeline/internal/sima/ContractRender.h"
#include "pipeline/internal/SyncBuild.h"
#include "pipeline/internal/TensorBufferEnvelope.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstddef>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include <gst/app/gstappsrc.h>

namespace simaai::neat {

namespace {
namespace rendered_stage_query = pipeline_internal::rendered_stage_query;

bool run_options_equal_for_cache_local(const RunOptions& a, const RunOptions& b);
std::string dtype_token_from_tensor(const Tensor& tensor);
std::string layout_token_from_tensor(const Tensor& tensor);
std::string layout_token_from_value(TensorLayout layout);

int max_num_buffers_in_pipeline_local(const std::string& pipeline) {
  const std::string key = "num-buffers=";
  int max_val = 0;
  int min_val = 0;
  size_t pos = 0;
  while ((pos = pipeline.find(key, pos)) != std::string::npos) {
    pos += key.size();
    int value = 0;
    bool found = false;
    while (pos < pipeline.size() && std::isdigit(static_cast<unsigned char>(pipeline[pos]))) {
      found = true;
      value = value * 10 + (pipeline[pos] - '0');
      ++pos;
    }
    if (found) {
      max_val = std::max(max_val, value);
      if (value > 0 && (min_val == 0 || value < min_val)) {
        min_val = value;
      }
    }
  }
  if (min_val == 1) {
    return 1;
  }
  return max_val;
}

const char* run_mode_name(RunMode mode) {
  return (mode == RunMode::Sync) ? "Sync" : "Async";
}

struct SessionBuildInputDebugFlags {
  bool build_mode_debug = env_bool("SIMA_BUILD_MODE_DEBUG", false);
  bool inputstream_debug = env_bool("SIMA_INPUTSTREAM_DEBUG", false);
  bool inputstream_warn = env_bool("SIMA_INPUTSTREAM_WARN", false);
  bool pipeline_state_debug = env_bool("SIMA_PIPELINE_STATE_DEBUG", false);
  bool preproc_debug_config = env_bool("SIMA_PREPROC_DEBUG_CONFIG", false);
  bool gst_enforce_names = env_bool("SIMA_GST_ENFORCE_NAMES", false);
  bool detess_override_debug =
      env_bool("SIMA_DETESS_OVERRIDE_DEBUG", false) || env_bool("SIMA_DETESS_DISPATCH_DEBUG", false);
  bool session_sync_cache_debug =
      env_bool("SIMA_SESSION_SYNC_CACHE_DEBUG", false) ||
      env_bool("SIMA_DETESS_DISPATCH_DEBUG", false) || env_bool("SIMA_DETESS_LIFECYCLE_DEBUG", false);
};

const SessionBuildInputDebugFlags& session_build_input_debug_flags() {
  static const SessionBuildInputDebugFlags flags;
  return flags;
}

bool build_mode_debug_enabled() {
  return session_build_input_debug_flags().build_mode_debug;
}

bool inputstream_debug_enabled_for_build() {
  return session_build_input_debug_flags().inputstream_debug;
}

bool inputstream_warn_or_debug_enabled_for_build() {
  const auto& flags = session_build_input_debug_flags();
  return flags.inputstream_warn || flags.inputstream_debug;
}

bool pipeline_state_debug_enabled_for_build() {
  return session_build_input_debug_flags().pipeline_state_debug;
}

bool preproc_debug_config_enabled() {
  return session_build_input_debug_flags().preproc_debug_config;
}

bool gst_enforce_names_enabled() {
  return session_build_input_debug_flags().gst_enforce_names;
}

TensorList tensor_list_from_mats(const std::vector<cv::Mat>& inputs, const InputOptions& opt,
                                 const char* where) {
  if (inputs.empty()) {
    throw std::runtime_error(std::string(where) + ": empty image list");
  }
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i].empty()) {
      throw std::runtime_error(std::string(where) + ": empty image input at index " +
                               std::to_string(i));
    }
  }
  TensorList out;
  out.reserve(inputs.size());
  const InputOptions normalized = pipeline_internal::normalize_shape_bounds(opt);
  for (const auto& input : inputs) {
    out.emplace_back(tensor_from_cv_mat(input, normalized, where));
  }
  return out;
}

bool input_options_expect_tensor_media(const InputOptions& opt) {
  return lower_copy(opt.media_type) == "application/vnd.simaai.tensor";
}

bool detess_override_debug_enabled() {
  return session_build_input_debug_flags().detess_override_debug;
}

bool session_sync_cache_debug_enabled() {
  return session_build_input_debug_flags().session_sync_cache_debug;
}

std::vector<const Tensor*> tensor_payload_from_sample(const Sample& sample) {
  std::vector<const Tensor*> out;
  if (sample_has_tensor_list(sample)) {
    out.reserve(sample.tensors.size());
    for (const auto& tensor : sample.tensors) {
      out.push_back(&tensor);
    }
    return out;
  }
  out.reserve(sample.fields.size());
  for (const auto& field : sample.fields) {
    if (!sample_has_tensor_list(field) || field.tensors.empty()) {
      continue;
    }
    out.push_back(&field.tensors.front());
  }
  return out;
}

const char* run_input_kind_name(RunInputKind kind) {
  switch (kind) {
  case RunInputKind::Mat:
    return "Mat";
  case RunInputKind::Tensor:
    return "Tensor";
  case RunInputKind::Sample:
    return "Sample";
  default:
    return "Unknown";
  }
}

const char* input_memory_policy_name(InputMemoryPolicy policy) {
  switch (policy) {
  case InputMemoryPolicy::Auto:
    return "auto";
  case InputMemoryPolicy::Ev74:
    return "ev74";
  case InputMemoryPolicy::Dms0:
    return "dms0";
  case InputMemoryPolicy::SystemMemory:
    return "system";
  }
  return "auto";
}

InputMemoryPolicy resolve_memory_policy_from_first_downstream_node(
    const std::vector<std::shared_ptr<Node>>& nodes) {
  if (nodes.size() <= 1U) {
    return InputMemoryPolicy::SystemMemory;
  }
  for (std::size_t i = 1U; i < nodes.size(); ++i) {
    const auto& node = nodes[i];
    if (!node) {
      continue;
    }
    const std::string kind = node->kind();
    if (kind == "Cast") {
      continue;
    }
    if (kind == "Preproc" || kind == "Quant" || kind == "Tess" || kind == "QuantTess") {
      return InputMemoryPolicy::Ev74;
    }
    if (kind == "ModelFragment") {
      return InputMemoryPolicy::Dms0;
    }
    return InputMemoryPolicy::SystemMemory;
  }
  return InputMemoryPolicy::SystemMemory;
}

std::string infer_first_effective_downstream_kind(const std::vector<std::shared_ptr<Node>>& nodes) {
  if (nodes.size() <= 1U) {
    return "<none>";
  }
  for (std::size_t i = 1U; i < nodes.size(); ++i) {
    const auto& node = nodes[i];
    if (!node) {
      continue;
    }
    const std::string kind = node->kind();
    if (kind == "Cast") {
      continue;
    }
    return kind;
  }
  return "<none>";
}

bool apply_auto_memory_policy_from_downstream(InputOptions& src_opt,
                                              const std::vector<std::shared_ptr<Node>>& nodes) {
  if (src_opt.memory_policy != InputMemoryPolicy::Auto) {
    return false;
  }
  const InputMemoryPolicy resolved = resolve_memory_policy_from_first_downstream_node(nodes);
  src_opt.memory_policy = resolved;
  src_opt.use_simaai_pool = (resolved != InputMemoryPolicy::SystemMemory);
  return true;
}

void maybe_log_build_mode(const char* where, RunMode mode, bool insert_queue2) {
  if (!build_mode_debug_enabled())
    return;
  std::fprintf(stderr, "[DBG] %s mode=%s(%d) async=%d sync=%d insert_queue2=%d\n",
               where ? where : "build", run_mode_name(mode), static_cast<int>(mode),
               static_cast<int>(RunMode::Async), static_cast<int>(RunMode::Sync),
               insert_queue2 ? 1 : 0);
}

bool preset_default_zero_copy(RunPreset preset) {
  switch (preset) {
  case RunPreset::Realtime:
    return true;
  case RunPreset::Balanced:
    return true;
  case RunPreset::Reliable:
    return false;
  }
  return false;
}

int preset_default_worker_poll_ms(RunPreset preset) {
  switch (preset) {
  case RunPreset::Realtime:
    return 10;
  case RunPreset::Balanced:
    return 20;
  case RunPreset::Reliable:
    return 30;
  }
  return 20;
}

int preset_default_stability_frames(RunPreset preset) {
  switch (preset) {
  case RunPreset::Realtime:
    return 1;
  case RunPreset::Balanced:
    return 2;
  case RunPreset::Reliable:
    return 2;
  }
  return 2;
}

int preset_default_queue_depth(RunPreset preset) {
  switch (preset) {
  case RunPreset::Realtime:
    return 2;
  case RunPreset::Balanced:
    return 4;
  case RunPreset::Reliable:
    return 8;
  }
  return 4;
}

OverflowPolicy preset_default_overflow_policy(RunPreset preset) {
  switch (preset) {
  case RunPreset::Realtime:
    return OverflowPolicy::KeepLatest;
  case RunPreset::Balanced:
    return OverflowPolicy::Block;
  case RunPreset::Reliable:
    return OverflowPolicy::Block;
  }
  return OverflowPolicy::Block;
}

RunOptions sync_run_defaults() {
  RunOptions opt;
  opt.preset = RunPreset::Reliable;
  opt.queue_depth = 1;
  opt.overflow_policy = OverflowPolicy::Block;
  opt.output_memory = OutputMemory::Owned;
  opt.enable_metrics = false;
  opt.advanced.copy_input = false;
  opt.advanced.max_input_bytes = 0;
  opt.advanced.sync_num_buffers_override = -1;
  if (env_bool("SIMA_DETESS_ZERO_COPY", false)) {
    opt.output_memory = OutputMemory::ZeroCopy;
  }
  return opt;
}

bool is_default_run_options(const RunOptions& opt) {
  const RunOptions d{};
  return run_options_equal_for_cache_local(opt, d) && !opt.on_input_drop;
}

bool effective_zero_copy_output(const RunOptions& opt) {
  switch (opt.output_memory) {
  case OutputMemory::ZeroCopy:
    return true;
  case OutputMemory::Owned:
    return false;
  case OutputMemory::Auto:
    return preset_default_zero_copy(opt.preset);
  }
  return preset_default_zero_copy(opt.preset);
}

bool resolve_prepare_output_cpu_visible(const RunOptions& opt, bool zero_copy) {
  if (!zero_copy) {
    return false;
  }
  const char* env = std::getenv("SIMA_PREPARE_OUTPUT_CPU_VISIBLE");
  if (env && *env) {
    return env_bool("SIMA_PREPARE_OUTPUT_CPU_VISIBLE", false);
  }
  return opt.advanced.prepare_output_cpu_visible;
}

int resolved_input_timeout_ms(const RunOptions& opt) {
  if (opt.input_timeout_ms >= 0) {
    return std::max(10, opt.input_timeout_ms);
  }
  return std::max(10, std::atoi(env_str("SIMA_GST_RUN_INPUT_TIMEOUT_MS", "10000").c_str()));
}

bool preset_default_insert_queue2(RunPreset preset) {
  switch (preset) {
  case RunPreset::Realtime:
    return false;
  case RunPreset::Balanced:
    return true;
  case RunPreset::Reliable:
    return true;
  }
  return true;
}

bool should_insert_async_queue2(RunMode mode, const RunOptions& opt) {
  if (mode != RunMode::Async)
    return false;
  const bool def_val = preset_default_insert_queue2(opt.preset);
  return env_bool("SIMA_ENABLE_ASYNC_QUEUE2", def_val);
}

int find_output_appsink_index_local(const std::vector<std::shared_ptr<Node>>& nodes) {
  int found = -1;
  for (size_t i = 0; i < nodes.size(); ++i) {
    if (!nodes[i])
      continue;
    if (nodes[i]->kind() != "Output")
      continue;
    if (found >= 0) {
      session_build_throw_session_error_simple(error_codes::kPipelineShape,
                                               "InvalidPipeline: multiple Output nodes found");
    }
    found = static_cast<int>(i);
  }
  return found;
}

bool has_output_appsink_local(const std::vector<std::shared_ptr<Node>>& nodes) {
  return find_output_appsink_index_local(nodes) >= 0;
}

std::optional<OutputTensorOverride>
build_detess_output_override(const std::vector<std::shared_ptr<Node>>& nodes) {
  if (!has_output_appsink_local(nodes))
    return std::nullopt;

  std::shared_ptr<Node> last;
  for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
    if (!*it)
      continue;
    if ((*it)->kind() == "Output")
      continue;
    last = *it;
    break;
  }
  if (!last || last->kind() != "DetessDequant")
    return std::nullopt;

  const NodeGroup group({last});
  const auto info = rendered_stage_query::detessdequant_output_info(group);
  if (info.outputs.empty())
    return std::nullopt;

  if (detess_override_debug_enabled()) {
    const std::size_t elem_bytes = pipeline_internal::dtype_bytes(info.dtype);
    std::fprintf(stderr,
                 "[detess-override] outputs=%zu elem_bytes=%zu dtype=%d layout=%d\n",
                 info.outputs.size(), elem_bytes, static_cast<int>(info.dtype),
                 static_cast<int>(info.layout));
    for (std::size_t i = 0; i < info.outputs.size(); ++i) {
      const auto& tensor = info.outputs[i];
      std::ostringstream shape_ss;
      shape_ss << "[";
      int64_t bytes = static_cast<int64_t>(elem_bytes);
      for (std::size_t j = 0; j < tensor.shape.size(); ++j) {
        if (j > 0U) {
          shape_ss << ",";
        }
        shape_ss << tensor.shape[j];
        bytes *= tensor.shape[j];
      }
      shape_ss << "]";
      std::fprintf(stderr,
                   "[detess-override]   output[%zu] shape=%s byte_offset=%lld bytes=%lld\n",
                   i, shape_ss.str().c_str(), static_cast<long long>(tensor.byte_offset),
                   static_cast<long long>(bytes));
    }
  }

  OutputTensorOverride out;
  out.outputs.reserve(info.outputs.size());
  for (const auto& tensor : info.outputs) {
    OutputTensorOverrideEntry entry;
    entry.shape = tensor.shape;
    entry.strides_bytes = tensor.stride_bytes;
    entry.byte_offset = tensor.byte_offset;
    entry.logical_output_index = static_cast<int>(out.outputs.size());
    entry.route_slot = entry.logical_output_index;
    entry.segment_name = "output" + std::to_string(entry.logical_output_index);
    entry.dtype = info.dtype;
    entry.layout = info.layout;
    out.outputs.push_back(std::move(entry));
  }
  return out;
}

TensorDType tensor_dtype_from_contract_token_local(std::string token) {
  std::transform(token.begin(), token.end(), token.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  constexpr const char* kPrefix = "EVXX_";
  if (token.rfind(kPrefix, 0U) == 0U) {
    token.erase(0U, std::char_traits<char>::length(kPrefix));
  }
  if (token.find("BFLOAT16") != std::string::npos || token.find("BF16") != std::string::npos) {
    return TensorDType::BFloat16;
  }
  if (token.find("FP64") != std::string::npos || token.find("FLOAT64") != std::string::npos) {
    return TensorDType::Float64;
  }
  if (token.find("FP32") != std::string::npos || token.find("FLOAT32") != std::string::npos) {
    return TensorDType::Float32;
  }
  if (token.find("UINT16") != std::string::npos) {
    return TensorDType::UInt16;
  }
  if (token.find("INT16") != std::string::npos) {
    return TensorDType::Int16;
  }
  if (token.find("INT32") != std::string::npos) {
    return TensorDType::Int32;
  }
  if (token.find("INT8") != std::string::npos) {
    return TensorDType::Int8;
  }
  if (token.find("UINT8") != std::string::npos) {
    return TensorDType::UInt8;
  }
  return TensorDType::UInt8;
}

std::string dtype_token_from_value(TensorDType dtype) {
  switch (dtype) {
  case TensorDType::UInt8:
    return "UINT8";
  case TensorDType::Int8:
    return "INT8";
  case TensorDType::UInt16:
    return "UINT16";
  case TensorDType::Int16:
    return "INT16";
  case TensorDType::Int32:
    return "INT32";
  case TensorDType::BFloat16:
    return "BF16";
  case TensorDType::Float32:
    return "FP32";
  case TensorDType::Float64:
    return "FP64";
  }
  return "UINT8";
}

std::string layout_token_from_tensor(const Tensor& tensor) {
  return layout_token_from_value(tensor.layout);
}

std::string dtype_token_from_tensor(const Tensor& tensor) {
  return dtype_token_from_value(tensor.dtype);
}

std::string dtype_token_from_spec(const SampleSpec& spec) {
  return dtype_token_from_value(spec.dtype);
}

std::string layout_token_from_value(TensorLayout layout) {
  switch (layout) {
  case TensorLayout::HW:
    return "HW";
  case TensorLayout::HWC:
    return "HWC";
  case TensorLayout::CHW:
    return "CHW";
  case TensorLayout::Unknown:
  default:
    return "UNKNOWN";
  }
}

std::string layout_token_from_spec(const SampleSpec& spec) {
  return layout_token_from_value(spec.layout);
}

std::string format_token_from_tensor(const Tensor& tensor) {
  if (tensor.semantic.tess.has_value()) {
    return upper_copy(tensor.semantic.tess->format);
  }
  if (tensor.semantic.image.has_value()) {
    switch (tensor.semantic.image->format) {
    case ImageSpec::PixelFormat::RGB:
      return "RGB";
    case ImageSpec::PixelFormat::BGR:
      return "BGR";
    case ImageSpec::PixelFormat::GRAY8:
      return "GRAY8";
    case ImageSpec::PixelFormat::NV12:
      return "NV12";
    case ImageSpec::PixelFormat::I420:
      return "I420";
    case ImageSpec::PixelFormat::UNKNOWN:
      break;
    }
  }
  return dtype_token_from_tensor(tensor);
}

InputContract input_contract_from_tensor(const Tensor& tensor) {
  InputContract out;
  if (tensor.semantic.image.has_value()) {
    out.media_type = "video/x-raw";
  } else {
    out.media_type = "application/vnd.simaai.tensor";
  }
  out.format = format_token_from_tensor(tensor);
  out.dtype = dtype_token_from_tensor(tensor);
  out.layout = layout_token_from_tensor(tensor);
  if (!tensor.shape.empty()) {
    if (tensor.layout == TensorLayout::HW) {
      out.height = tensor.shape.size() > 0 ? static_cast<int>(tensor.shape[0]) : 0;
      out.width = tensor.shape.size() > 1 ? static_cast<int>(tensor.shape[1]) : 0;
      out.depth = 1;
    } else if (tensor.layout == TensorLayout::CHW) {
      out.depth = tensor.shape.size() > 0 ? static_cast<int>(tensor.shape[0]) : 0;
      out.height = tensor.shape.size() > 1 ? static_cast<int>(tensor.shape[1]) : 0;
      out.width = tensor.shape.size() > 2 ? static_cast<int>(tensor.shape[2]) : 0;
    } else {
      out.height = tensor.shape.size() > 0 ? static_cast<int>(tensor.shape[0]) : 0;
      out.width = tensor.shape.size() > 1 ? static_cast<int>(tensor.shape[1]) : 0;
      out.depth = tensor.shape.size() > 2 ? static_cast<int>(tensor.shape[2]) : 0;
    }
  }
  if (tensor.semantic.quant.has_value()) {
    const auto& q = *tensor.semantic.quant;
    if (!q.scales.empty()) {
      out.q_scale = q.scales.front();
    }
    if (!q.zero_points.empty()) {
      out.q_zp = q.zero_points.front();
    }
  }
  return out;
}

#if defined(SIMA_WITH_OPENCV)
InputContract input_contract_from_mat(const cv::Mat& input) {
  InputContract out;
  out.media_type = "video/x-raw";
  out.width = input.cols;
  out.height = input.rows;
  out.depth = input.channels();
  out.format = (out.depth == 1) ? "GRAY8" : "BGR";
  return out;
}
#endif

InputContract input_contract_from_output_spec(const OutputSpec& spec) {
  InputContract out;
  out.media_type = spec.media_type;
  out.format = spec.format;
  out.dtype = spec.dtype;
  out.layout = spec.layout;
  out.width = spec.width;
  out.height = spec.height;
  out.depth = spec.depth;
  return out;
}

InputContract input_contract_from_sample_spec(const SampleSpec& spec) {
  InputContract out;
  out.media_type = spec.media_type;
  out.format = spec.format;
  out.dtype = dtype_token_from_spec(spec);
  out.layout = layout_token_from_spec(spec);
  if (spec.tensor_envelope_transport && spec.kind != SampleMediaKind::RawVideo) {
    return out;
  }
  out.width = spec.width;
  out.height = spec.height;
  out.depth = spec.depth;
  return out;
}

template <typename InputT> InputContract input_contract_from_input(const InputT& sample);

#if defined(SIMA_WITH_OPENCV)
template <> InputContract input_contract_from_input(const cv::Mat& sample) {
  return input_contract_from_mat(sample);
}
#endif

template <> InputContract input_contract_from_input(const Tensor& sample) {
  return input_contract_from_tensor(sample);
}

template <> InputContract input_contract_from_input(const Sample& sample) {
  if (sample_has_tensor_list(sample)) {
    return input_contract_from_sample_spec(derive_sample_spec_or_throw(sample));
  }
  InputContract out;
  out.media_type = sample.media_type;
  return out;
}

void apply_input_contract_to_nodes(const std::vector<std::shared_ptr<Node>>& nodes,
                                   const InputContract& seed_contract) {
  if (nodes.empty()) {
    return;
  }
  InputContract current = seed_contract;
  for (std::size_t i = 1; i < nodes.size(); ++i) {
    const auto& node = nodes[i];
    if (!node) {
      continue;
    }
    if (auto* configurable = dynamic_cast<InputContractConfigurable*>(node.get())) {
      std::string err;
      configurable->apply_input_contract(current, &err);
      if (!err.empty()) {
        throw std::runtime_error(err);
      }
    }
    if (auto* provider = dynamic_cast<OutputSpecProvider*>(node.get())) {
      OutputSpec in_spec;
      in_spec.media_type = current.media_type;
      in_spec.format = current.format;
      in_spec.dtype = current.dtype;
      in_spec.layout = current.layout;
      in_spec.width = current.width;
      in_spec.height = current.height;
      in_spec.depth = current.depth;
      current = input_contract_from_output_spec(provider->output_spec(in_spec));
    }
  }
}

void maybe_compile_build_result_contracts(
    BuildResult* build_result,
    std::vector<std::shared_ptr<Node>>* nodes,
    const SessionOptions& sess_opt,
    const InputContract& ingress_contract,
    const SampleSpec& ingress_spec,
    const std::optional<Sample>& ingress_sample,
    const char* where) {
  if (!build_result || !nodes) {
    return;
  }

  ContractCompileInput compile_input;
  compile_input.pipeline_label = where ? where : "Session::build(input)";
  compile_input.processcvu_requested_run_target = sess_opt.processcvu_requested_run_target;
  compile_input.processcvu = sess_opt.processcvu;
  compile_input.ingress.ingress_contract = ingress_contract;
  OutputSpec spec;
  spec.media_type = ingress_spec.media_type;
  spec.format = ingress_spec.format;
  spec.dtype = dtype_token_from_spec(ingress_spec);
  spec.layout = layout_token_from_spec(ingress_spec);
  spec.width = ingress_spec.width;
  spec.height = ingress_spec.height;
  spec.depth = ingress_spec.depth;
  compile_input.ingress.ingress_spec = spec;
  compile_input.ingress.ingress_sample = ingress_sample;
  session_build_compile_contracts(build_result, *nodes, compile_input, where, nodes);
}

std::optional<Sample> contract_compile_sample_from_input(const Sample& sample) {
  return sample;
}

template <typename InputT> std::optional<Sample> contract_compile_sample_from_input(const InputT&) {
  return std::nullopt;
}

std::optional<OutputTensorOverride>
build_mla_output_override(const std::vector<std::shared_ptr<Node>>& nodes) {
  if (!has_output_appsink_local(nodes))
    return std::nullopt;

  std::shared_ptr<Node> last;
  for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
    if (!*it)
      continue;
    if ((*it)->kind() == "Output")
      continue;
    last = *it;
    break;
  }
  if (!last || last->kind() != "ModelFragment")
    return std::nullopt;

  const NodeGroup group({last});
  const auto outputs = rendered_stage_query::mla_output_tensors(group);
  if (outputs.size() <= 1U) {
    return std::nullopt;
  }

  OutputTensorOverride out;
  out.outputs.reserve(outputs.size());
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    const auto& tensor = outputs[i];
    if (tensor.shape.empty()) {
      return std::nullopt;
    }
    OutputTensorOverrideEntry entry;
    entry.shape = tensor.shape;
    entry.strides_bytes = tensor.stride_bytes;
    entry.byte_offset = tensor.byte_offset;
    entry.memory_index = tensor.memory_index;
    entry.logical_output_index = static_cast<int>(i);
    entry.route_slot = static_cast<int>(i);
    entry.dtype = tensor_dtype_from_contract_token_local(tensor.data_type);
    entry.layout = tensor.layout;
    entry.format = tensor.output_format;
    entry.name = tensor.name.empty() ? ("output" + std::to_string(i)) : tensor.name;
    entry.segment_name = !tensor.segment_name.empty() ? tensor.segment_name : entry.name;
    out.outputs.push_back(std::move(entry));
  }

  if (detess_override_debug_enabled()) {
    std::fprintf(stderr, "[output-override] mla outputs=%zu\n", out.outputs.size());
    for (std::size_t i = 0; i < out.outputs.size(); ++i) {
      const auto& entry = out.outputs[i];
      std::ostringstream shape_ss;
      shape_ss << "[";
      for (std::size_t j = 0; j < entry.shape.size(); ++j) {
        if (j > 0U) {
          shape_ss << ",";
        }
        shape_ss << entry.shape[j];
      }
      shape_ss << "]";
      std::fprintf(stderr,
                   "[output-override]   output[%zu] name=%s mem=%d shape=%s byte_offset=%lld\n",
                   i, entry.name.c_str(), entry.memory_index, shape_ss.str().c_str(),
                   static_cast<long long>(entry.byte_offset));
    }
  }

  return out;
}

std::optional<OutputTensorOverride>
build_cast_output_override(const std::vector<std::shared_ptr<Node>>& nodes) {
  if (!has_output_appsink_local(nodes))
    return std::nullopt;

  std::shared_ptr<Node> last;
  std::shared_ptr<Node> upstream;
  for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
    if (!*it)
      continue;
    if ((*it)->kind() == "Output")
      continue;
    if (!last) {
      last = *it;
      continue;
    }
    upstream = *it;
    break;
  }
  if (!last || last->kind() != "Cast" || !upstream || upstream->kind() != "ModelFragment")
    return std::nullopt;

  const NodeGroup group({upstream});
  const auto outputs = rendered_stage_query::mla_output_tensors(group);
  if (outputs.size() <= 1U) {
    return std::nullopt;
  }

  OutputTensorOverride out;
  out.outputs.reserve(outputs.size());
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    const auto& tensor = outputs[i];
    if (tensor.shape.empty()) {
      return std::nullopt;
    }
    OutputTensorOverrideEntry entry;
    entry.shape = tensor.shape;
    entry.memory_index = tensor.memory_index;
    entry.logical_output_index = static_cast<int>(i);
    entry.route_slot = static_cast<int>(i);

    const TensorDType in_dtype = tensor_dtype_from_contract_token_local(tensor.data_type);
    TensorDType out_dtype = in_dtype;
    if (in_dtype == TensorDType::BFloat16) {
      out_dtype = TensorDType::Float32;
    } else if (in_dtype == TensorDType::Float32) {
      out_dtype = TensorDType::BFloat16;
    }
    entry.dtype = out_dtype;
    entry.layout = tensor.layout;
    entry.format = tensor.output_format;
    entry.name = tensor.name.empty() ? ("output" + std::to_string(i)) : tensor.name;
    entry.segment_name = !tensor.segment_name.empty() ? tensor.segment_name : entry.name;

    const std::size_t in_elem_bytes = pipeline_internal::dtype_bytes(in_dtype);
    const std::size_t out_elem_bytes = pipeline_internal::dtype_bytes(out_dtype);
    if (in_elem_bytes > 0U && out_elem_bytes > 0U) {
      entry.byte_offset = static_cast<int64_t>(
          (static_cast<std::uint64_t>(tensor.byte_offset) / in_elem_bytes) * out_elem_bytes);
      if (!tensor.stride_bytes.empty()) {
        entry.strides_bytes.reserve(tensor.stride_bytes.size());
        for (const auto stride : tensor.stride_bytes) {
          if (stride > 0) {
            entry.strides_bytes.push_back(static_cast<int64_t>(
                (static_cast<std::uint64_t>(stride) / in_elem_bytes) * out_elem_bytes));
          } else {
            entry.strides_bytes.push_back(stride);
          }
        }
      }
    } else {
      entry.byte_offset = tensor.byte_offset;
      entry.strides_bytes = tensor.stride_bytes;
    }
    out.outputs.push_back(std::move(entry));
  }

  if (detess_override_debug_enabled()) {
    std::fprintf(stderr, "[output-override] cast outputs=%zu\n", out.outputs.size());
  }

  return out;
}

bool has_rtsp_input_nodes(const std::vector<std::shared_ptr<Node>>& nodes) {
  for (const auto& node : nodes) {
    if (!node)
      continue;
    if (dynamic_cast<const RTSPInput*>(node.get()))
      return true;
  }
  return false;
}

void maybe_enable_rtsp_appsink_drop(InputStreamOptions& stream_opt,
                                    const std::vector<std::shared_ptr<Node>>& nodes) {
  if (!has_rtsp_input_nodes(nodes))
    return;
  if (env_bool("SIMA_RTSP_ALLOW_BACKPRESSURE", false))
    return;
  stream_opt.appsink_drop = true;
  if (stream_opt.appsink_max_buffers <= 0) {
    stream_opt.appsink_max_buffers = 1;
  }
}

void maybe_apply_detess_output_override(const std::vector<std::shared_ptr<Node>>& nodes,
                                        InputStreamOptions& stream_opt) {
  if (stream_opt.output_override.has_value())
    return;
  std::optional<OutputTensorOverride> override;
  if (!override.has_value()) {
    override = build_cast_output_override(nodes);
  }
  if (!override.has_value()) {
    override = build_mla_output_override(nodes);
  }
  if (override.has_value()) {
    stream_opt.output_override = std::move(*override);
  }
}

InputStreamOptions make_stream_options(const RunOptions& opt, RunMode mode) {
  InputStreamOptions stream_opt;
  const int queue_depth = (opt.queue_depth > 0) ? opt.queue_depth : 0;
  const bool zero_copy = effective_zero_copy_output(opt);
  stream_opt.appsink_sync = false;
  stream_opt.appsink_drop = (opt.overflow_policy != OverflowPolicy::Block);
  stream_opt.appsink_max_buffers = queue_depth;
  stream_opt.stability_frames = preset_default_stability_frames(opt.preset);
  stream_opt.max_input_bytes = opt.advanced.max_input_bytes;
  stream_opt.copy_output = !zero_copy;
  stream_opt.prepare_output_cpu_visible = resolve_prepare_output_cpu_visible(opt, zero_copy);
  // Honor opt.advanced.copy_input as the single source of truth for input
  // lifetime semantics. The previous "|| mode == RunMode::Sync" override
  // was a heuristic from when Sync-mode callers were assumed to pass
  // short-lived raw inputs; today, per-stage helpers (stages::Preproc/
  // Infer/MLA/Postprocess) and tensor handoffs hand us GstSample-backed
  // tensors whose holders own the segment-backed memory across the entire
  // Sync push/pull cycle. Forcing a copy in that path discards the
  // segment-backed memory, leaving downstream MLA with a SystemMemory
  // buffer that fails its zero-copy bind contract. The buffer adapter
  // (buffer_from_tensor_or_copy / wrap_cpu_video_zero_copy) already does
  // per-storage durability discrimination, so callers who genuinely need
  // a copy must request it via opt.advanced.copy_input = true.
  stream_opt.copy_input = opt.advanced.copy_input;
  stream_opt.reuse_input_buffer = false;
  stream_opt.on_input_drop = opt.on_input_drop;
  stream_opt.enable_timings = opt.enable_metrics;
  stream_opt.timeout_ms = resolved_input_timeout_ms(opt);
  const bool balanced_zero_copy_probe = (opt.preset == RunPreset::Balanced) && zero_copy;
  stream_opt.startup_preflight = balanced_zero_copy_probe;
  stream_opt.worker_poll_ms = preset_default_worker_poll_ms(opt.preset);
  return stream_opt;
}

void validate_shape_limits_or_throw(const InputStreamOptions::ResolvedShapeLimits& limits,
                                    const char* where) {
  const char* tag = where ? where : "Session::build(input)";
  const std::optional<std::string> err = pipeline_internal::validate_shape_limits(limits);
  if (!err.has_value())
    return;

  if (err->find("width") != std::string::npos) {
    session_build_throw_session_error_simple(
        error_codes::kInputShape, std::string(tag) + ": " + *err,
        "Set width <= max_width (or clear width to infer seed dynamically).");
  }
  if (err->find("height") != std::string::npos) {
    session_build_throw_session_error_simple(
        error_codes::kInputShape, std::string(tag) + ": " + *err,
        "Set height <= max_height (or clear height to infer seed dynamically).");
  }
  session_build_throw_session_error_simple(
      error_codes::kInputShape, std::string(tag) + ": " + *err,
      "Set depth <= max_depth (or clear depth to infer seed dynamically).");
}

InputStreamOptions::DynamicCapability detect_dynamic_capability(
    const std::vector<std::shared_ptr<Node>>& nodes, const InputOptions& src_opt,
    const SampleSpec& seed, InputStreamOptions::ShapePolicy policy) {
  if (policy == InputStreamOptions::ShapePolicy::LockedByCapsOverride) {
    return InputStreamOptions::DynamicCapability::StaticOnly;
  }

  if (seed.kind != SampleMediaKind::RawVideo) {
    return InputStreamOptions::DynamicCapability::StaticOnly;
  }

  const std::string media = lower_copy(src_opt.media_type.empty() ? seed.media_type : src_opt.media_type);
  if (media != "video/x-raw") {
    return InputStreamOptions::DynamicCapability::StaticOnly;
  }

  bool all_dynamic_after_input = true;
  for (std::size_t i = 1; i < nodes.size(); ++i) {
    if (!nodes[i])
      continue;
    if (nodes[i]->caps_behavior() != NodeCapsBehavior::Dynamic) {
      all_dynamic_after_input = false;
      break;
    }
  }
  if (all_dynamic_after_input) {
    return InputStreamOptions::DynamicCapability::FullyDynamic;
  }

  const Preproc* dynamic_preproc = nullptr;
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    auto* preproc = dynamic_cast<const Preproc*>(nodes[i].get());
    if (preproc && preproc->options().dynamic_input_dims) {
      dynamic_preproc = preproc;
      break;
    }
  }
  if (!dynamic_preproc) {
    return InputStreamOptions::DynamicCapability::StaticOnly;
  }

  const auto& pre_opt = dynamic_preproc->options();
  if (pre_opt.output_width() <= 0 || pre_opt.output_height() <= 0 ||
      pre_opt.output_img_type.empty()) {
    return InputStreamOptions::DynamicCapability::StaticOnly;
  }

  return InputStreamOptions::DynamicCapability::IngressDynamicCvuOnly;
}

const char* shape_policy_name(InputStreamOptions::ShapePolicy policy) {
  switch (policy) {
  case InputStreamOptions::ShapePolicy::BoundedDynamic:
    return "BoundedDynamic";
  case InputStreamOptions::ShapePolicy::ElasticDynamic:
    return "ElasticDynamic";
  case InputStreamOptions::ShapePolicy::LockedByCapsOverride:
    return "LockedByCapsOverride";
  }
  return "Unknown";
}

const char* dynamic_capability_name(InputStreamOptions::DynamicCapability capability) {
  switch (capability) {
  case InputStreamOptions::DynamicCapability::StaticOnly:
    return "StaticOnly";
  case InputStreamOptions::DynamicCapability::IngressDynamicCvuOnly:
    return "IngressDynamicCvuOnly";
  case InputStreamOptions::DynamicCapability::FullyDynamic:
    return "FullyDynamic";
  }
  return "Unknown";
}

const char* limit_origin_name(InputStreamOptions::LimitOrigin origin) {
  switch (origin) {
  case InputStreamOptions::LimitOrigin::Unset:
    return "unset";
  case InputStreamOptions::LimitOrigin::SeedInput:
    return "seed_input";
  case InputStreamOptions::LimitOrigin::UserSeed:
    return "user_seed";
  case InputStreamOptions::LimitOrigin::UserMax:
    return "user_max";
  }
  return "unset";
}

const char* byte_guard_origin_name(InputStreamOptions::ByteGuardOrigin origin) {
  switch (origin) {
  case InputStreamOptions::ByteGuardOrigin::Unset:
    return "unset";
  case InputStreamOptions::ByteGuardOrigin::User:
    return "user";
  case InputStreamOptions::ByteGuardOrigin::DerivedElasticDefault:
    return "derived_elastic_default";
  case InputStreamOptions::ByteGuardOrigin::DerivedBoundedEstimate:
    return "derived_bounded_estimate";
  }
  return "unset";
}

bool is_supported_raw_ingress_format(std::string fmt) {
  fmt = upper_copy(fmt);
  if (fmt == "GRAY")
    fmt = "GRAY8";
  if (fmt == "IYUV")
    fmt = "I420";
  return fmt == "RGB" || fmt == "BGR" || fmt == "GRAY8" || fmt == "NV12" || fmt == "I420";
}

int raw_depth_for_format(std::string fmt) {
  fmt = upper_copy(fmt);
  if (fmt == "GRAY" || fmt == "GRAY8")
    return 1;
  if (fmt == "RGB" || fmt == "BGR")
    return 3;
  return -1;
}

std::string canonical_raw_format(std::string fmt) {
  fmt = upper_copy(fmt);
  if (fmt == "GRAY")
    return "GRAY8";
  if (fmt == "IYUV")
    return "I420";
  return fmt;
}

void add_unique_ingress_format(std::vector<std::string>& out, const std::string& fmt) {
  std::string c = canonical_raw_format(fmt);
  if (c.empty() || !is_supported_raw_ingress_format(c))
    return;
  if (std::find(out.begin(), out.end(), c) == out.end())
    out.push_back(c);
}

OutputSpec make_ingress_spec_for_format(const SampleSpec& seed, const std::string& fmt) {
  OutputSpec in;
  in.media_type = "video/x-raw";
  in.format = canonical_raw_format(fmt);
  in.width = seed.width;
  in.height = seed.height;
  in.depth = raw_depth_for_format(in.format);
  in.layout.clear();
  in.dtype = "UInt8";
  in.memory = "SystemMemory";
  in.certainty = SpecCertainty::Authoritative;
  in.byte_size = expected_byte_size(in);
  return in;
}

bool output_contract_equal(const OutputSpec& a, const OutputSpec& b) {
  return a.media_type == b.media_type && a.format == b.format && a.width == b.width &&
         a.height == b.height && a.depth == b.depth && a.layout == b.layout &&
         a.dtype == b.dtype;
}

bool derive_downstream_contract_for_ingress_format(
    const std::vector<std::shared_ptr<Node>>& nodes, std::size_t start_idx, const SampleSpec& seed,
    const std::string& fmt, OutputSpec* out) {
  if (!out || seed.width <= 0 || seed.height <= 0)
    return false;
  if (start_idx >= nodes.size())
    return false;
  std::vector<std::shared_ptr<Node>> downstream(nodes.begin() + static_cast<std::ptrdiff_t>(start_idx),
                                                nodes.end());
  if (downstream.empty())
    return false;

  try {
    const NodeGroup group(downstream);
    *out = derive_output_spec(group, make_ingress_spec_for_format(seed, fmt));
    return !out->media_type.empty();
  } catch (...) {
    return false;
  }
}

bool detect_allow_ingress_format_renegotiation(
    const std::vector<std::shared_ptr<Node>>& nodes, const InputOptions& src_opt,
    const SampleSpec& seed, InputStreamOptions::ShapePolicy policy,
    InputStreamOptions::DynamicCapability capability) {
  if (policy == InputStreamOptions::ShapePolicy::LockedByCapsOverride)
    return false;
  if (capability != InputStreamOptions::DynamicCapability::IngressDynamicCvuOnly)
    return false;
  if (seed.kind != SampleMediaKind::RawVideo || seed.width <= 0 || seed.height <= 0)
    return false;

  const Preproc* dynamic_preproc = nullptr;
  std::size_t preproc_idx = 0;
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    auto* preproc = dynamic_cast<const Preproc*>(nodes[i].get());
    if (preproc && preproc->options().dynamic_input_dims) {
      dynamic_preproc = preproc;
      preproc_idx = i;
      break;
    }
  }
  if (!dynamic_preproc)
    return false;

  const auto& pre_opt = dynamic_preproc->options();
  if (pre_opt.output_width() <= 0 || pre_opt.output_height() <= 0 ||
      pre_opt.output_img_type.empty())
    return false;

  std::vector<std::string> formats;
  add_unique_ingress_format(formats, seed.format);
  add_unique_ingress_format(formats, src_opt.format);
  add_unique_ingress_format(formats, pre_opt.input_img_type);

  const std::string baseline = !formats.empty()
                                   ? formats.front()
                                   : canonical_raw_format(seed.format);
  if (!is_supported_raw_ingress_format(baseline))
    return false;

  if (baseline == "RGB")
    add_unique_ingress_format(formats, "BGR");
  else if (baseline == "BGR")
    add_unique_ingress_format(formats, "RGB");
  else if (baseline == "NV12")
    add_unique_ingress_format(formats, "I420");
  else if (baseline == "I420")
    add_unique_ingress_format(formats, "NV12");

  OutputSpec baseline_out;
  if (!derive_downstream_contract_for_ingress_format(nodes, preproc_idx, seed, baseline,
                                                     &baseline_out)) {
    return false;
  }

  bool saw_alternate = false;
  for (const auto& fmt : formats) {
    if (fmt == baseline)
      continue;
    OutputSpec alt_out;
    if (!derive_downstream_contract_for_ingress_format(nodes, preproc_idx, seed, fmt, &alt_out)) {
      continue;
    }
    saw_alternate = true;
    if (!output_contract_equal(baseline_out, alt_out)) {
      return false;
    }
  }

  return saw_alternate;
}

void add_build_adaptation_action(BuildAdaptationSummary& summary, const std::string& target,
                                 bool applied, const std::string& detail,
                                 const std::string& reason = {}) {
  BuildAdaptationAction action;
  action.target = target;
  action.applied = applied;
  action.detail = detail;
  action.reason = reason;
  summary.actions.push_back(std::move(action));
}

InputOptions seed_input_options_from_spec(const InputOptions& opt, const SampleSpec& seed) {
  InputOptions out = pipeline_internal::normalize_shape_bounds(opt);
  const InputOptions defaults{};
  if (!out.caps_override.empty()) {
    return out;
  }

  const bool default_media_seed =
      (out.media_type == defaults.media_type) && out.format.empty() && out.width <= 0 &&
      out.height <= 0 && out.depth <= 0;
  if ((out.media_type.empty() || default_media_seed) && !seed.media_type.empty()) {
    out.media_type = seed.media_type;
  }
  if (out.format.empty() && !seed.format.empty()) {
    out.format = seed.format;
  }

  if (seed.tensor_envelope_transport) {
    return out;
  }

  const auto limits = pipeline_internal::resolve_shape_limits(out, seed);
  if (out.width <= 0 && limits.seed_width > 0) {
    out.width = limits.seed_width;
  }
  if (out.height <= 0 && limits.seed_height > 0) {
    out.height = limits.seed_height;
  }
  if (out.depth <= 0 && limits.seed_depth > 0) {
    out.depth = limits.seed_depth;
  }

  if (out.max_width <= 0 && limits.max_width > 0) {
    out.max_width = limits.max_width;
  }
  if (out.max_height <= 0 && limits.max_height > 0) {
    out.max_height = limits.max_height;
  }
  if (out.max_depth <= 0 && limits.max_depth > 0) {
    out.max_depth = limits.max_depth;
  }

  return out;
}

std::vector<std::shared_ptr<Node>> replace_first_input_node_for_build(
    const std::vector<std::shared_ptr<Node>>& nodes, const InputOptions& seeded_input_opt) {
  std::vector<std::shared_ptr<Node>> patched = nodes;
  if (!patched.empty()) {
    patched.front() = nodes::Input(seeded_input_opt);
  }
  return patched;
}


RunOptions resolve_build_opt(RunMode mode, const RunOptions& opt) {
  RunOptions out = opt;
  const RunOptions defaults{};
  if (mode == RunMode::Sync) {
    const RunOptions sync_defaults = sync_run_defaults();
    if (out.preset == defaults.preset) {
      out.preset = sync_defaults.preset;
    }
    if (out.queue_depth == defaults.queue_depth) {
      out.queue_depth = sync_defaults.queue_depth;
    }
    if (out.overflow_policy == defaults.overflow_policy) {
      out.overflow_policy = sync_defaults.overflow_policy;
    }
    if (out.output_memory == defaults.output_memory) {
      out.output_memory = sync_defaults.output_memory;
    }
    if (out.enable_metrics == defaults.enable_metrics) {
      out.enable_metrics = sync_defaults.enable_metrics;
    }
    if (out.advanced.copy_input == defaults.advanced.copy_input) {
      out.advanced.copy_input = sync_defaults.advanced.copy_input;
    }
    if (out.advanced.max_input_bytes == defaults.advanced.max_input_bytes) {
      out.advanced.max_input_bytes = sync_defaults.advanced.max_input_bytes;
    }
    if (out.advanced.sync_num_buffers_override == defaults.advanced.sync_num_buffers_override) {
      out.advanced.sync_num_buffers_override = sync_defaults.advanced.sync_num_buffers_override;
    }
    if (out.advanced.prepare_output_cpu_visible == defaults.advanced.prepare_output_cpu_visible) {
      out.advanced.prepare_output_cpu_visible = sync_defaults.advanced.prepare_output_cpu_visible;
    }
  }
  if (out.queue_depth == defaults.queue_depth) {
    out.queue_depth = preset_default_queue_depth(out.preset);
  }
  if (out.overflow_policy == defaults.overflow_policy) {
    out.overflow_policy = preset_default_overflow_policy(out.preset);
  }
  return out;
}

BuildInputContext prepare_build_input_context(const std::vector<std::shared_ptr<Node>>& nodes,
                                              const SessionOptions& sess_opt, RunMode mode,
                                              const RunOptions& opt) {
  BuildInputContext ctx;
  ctx.mode = mode;
  const RunOptions requested_opt = resolve_build_opt(mode, opt);
  ctx.merged_opt = requested_opt;

  enforce_caps_behavior(nodes, "Session::build(input)");
  enforce_sink_last_if_present(nodes, "Session::build(input)");
  enforce_push_run_mode(nodes, "Session::build(input)");
  require_input_appsrc(nodes, "Session::build(input)", &ctx.src_node);

  ctx.stream_opt = make_stream_options(ctx.merged_opt, ctx.mode);
  maybe_enable_rtsp_appsink_drop(ctx.stream_opt, nodes);
  maybe_apply_detess_output_override(nodes, ctx.stream_opt);
  ctx.insert_queue2 = should_insert_async_queue2(ctx.mode, ctx.merged_opt);
  maybe_log_build_mode("Session::build(input)", ctx.mode, ctx.insert_queue2);

  ctx.sync_num_buffers_override = ctx.merged_opt.advanced.sync_num_buffers_override;
  ctx.name_transform = make_name_transform(sess_opt);
  return ctx;
}

void maybe_log_appsrc_state(GstElement* appsrc) {
  if (!appsrc || !inputstream_debug_enabled_for_build())
    return;
  gboolean block = FALSE;
  gint64 max_bytes = 0;
  guint64 max_buffers = 0;
  g_object_get(G_OBJECT(appsrc), "block", &block, "max-bytes", &max_bytes, "max-buffers",
               &max_buffers, nullptr);
  std::fprintf(stderr, "[DBG] appsrc block=%d max-bytes=%lld max-buffers=%llu\n",
               static_cast<int>(block), static_cast<long long>(max_bytes),
               static_cast<unsigned long long>(max_buffers));
}

void maybe_log_appsink_state(GstElement* appsink) {
  if (!appsink || !inputstream_debug_enabled_for_build())
    return;
  gboolean drop = FALSE;
  guint max_buffers = 0;
  gboolean sync = FALSE;
  g_object_get(G_OBJECT(appsink), "drop", &drop, "max-buffers", &max_buffers, "sync", &sync,
               nullptr);
  std::fprintf(stderr, "[DBG] appsink drop=%d max-buffers=%u sync=%d\n", static_cast<int>(drop),
               static_cast<unsigned int>(max_buffers), static_cast<int>(sync));
}

void maybe_log_pipeline_state(GstElement* pipeline, const char* where) {
  if (!pipeline || !pipeline_state_debug_enabled_for_build())
    return;
  GstState cur = GST_STATE_VOID_PENDING;
  GstState pend = GST_STATE_VOID_PENDING;
  gst_element_get_state(pipeline, &cur, &pend, 0);
  std::fprintf(stderr, "[DBG] %s state cur=%s pending=%s\n", where ? where : "pipeline",
               gst_element_state_get_name(cur), gst_element_state_get_name(pend));
}

void update_preproc_snapshot_format(const std::string& path, const std::string& format) {
  if (path.empty() || format.empty())
    return;
  std::ifstream in(path);
  if (!in.is_open())
    return;
  nlohmann::json j;
  try {
    in >> j;
  } catch (const std::exception&) {
    return;
  }
  in.close();
  if (j.contains("input_img_type") && j["input_img_type"].is_string() &&
      j["input_img_type"].get<std::string>() == format) {
    return;
  }
  j["input_img_type"] = format;
  std::ofstream out(path);
  if (!out.is_open())
    return;
  out << j.dump(4);
}

void install_input_contract_caps_change_callback(InputStream& stream,
                                                 const std::vector<std::shared_ptr<Node>>& nodes,
                                                 const SampleSpec& initial_spec) {
  std::vector<std::string> snapshot_paths;
  for (const auto& node : nodes) {
    auto* preproc = dynamic_cast<simaai::neat::Preproc*>(node.get());
    if (!preproc)
      continue;
    const std::string& snap = preproc->config_snapshot_path();
    if (!snap.empty())
      snapshot_paths.push_back(snap);
  }
  if (preproc_debug_config_enabled()) {
    for (const auto& path : snapshot_paths) {
      std::fprintf(stderr, "[DBG] preproc snapshot path=%s\n", path.c_str());
    }
  }
  std::string last_format = initial_spec.format;
  stream.set_on_caps_change(
      [nodes, snapshot_paths, last_format](const SampleSpec& old_spec,
                                           const SampleSpec& new_spec) mutable {
        (void)old_spec;

        apply_input_contract_to_nodes(nodes, input_contract_from_sample_spec(new_spec));

        if (new_spec.kind != SampleMediaKind::RawVideo)
          return;
        if (new_spec.format.empty())
          return;
        if (!last_format.empty() && new_spec.format == last_format)
          return;
        if (preproc_debug_config_enabled()) {
          std::fprintf(stderr,
                       "[DBG] preproc snapshot update format=%s width=%d height=%d depth=%d\n",
                       new_spec.format.c_str(), new_spec.width, new_spec.height, new_spec.depth);
        }
        for (const auto& path : snapshot_paths) {
          update_preproc_snapshot_format(path, new_spec.format);
        }
        last_format = new_spec.format;
      });
}

bool resolve_startup_preflight(const InputStreamOptions& opt) {
  const char* env = std::getenv("SIMA_INPUTSTREAM_PREFLIGHT_RUN");
  if (env && *env)
    return std::atoi(env) != 0;
  return opt.startup_preflight;
}

void validate_caps_override_or_throw(const InputOptions& src_opt, const char* where,
                                     const std::string& pipeline_hint) {
  if (src_opt.caps_override.empty())
    return;
  GstCaps* parsed = gst_caps_from_string(src_opt.caps_override.c_str());
  if (!parsed) {
    session_build_throw_session_error_simple(
        error_codes::kCaps,
        std::string(where ? where : "Session::build(input)") +
            ": invalid caps_override: " + src_opt.caps_override,
        "Provide a valid GstCaps string matching the input payload.", pipeline_hint);
  }
  gst_caps_unref(parsed);
}

std::string parse_named_element_for_error(const std::string& pipeline, const std::string& element) {
  const std::size_t elem_pos = pipeline.find(element);
  if (elem_pos == std::string::npos)
    return "";
  const std::size_t seg_end = pipeline.find('!', elem_pos);
  const std::size_t name_pos = pipeline.find("name=", elem_pos);
  if (name_pos == std::string::npos)
    return "";
  if (seg_end != std::string::npos && name_pos > seg_end)
    return "";
  std::size_t vpos = name_pos + 5;
  std::size_t vend = vpos;
  while (vend < pipeline.size()) {
    const char c = pipeline[vend];
    if (std::isspace(static_cast<unsigned char>(c)) || c == '!' || c == '"')
      break;
    ++vend;
  }
  if (vend <= vpos)
    return "";
  return pipeline.substr(vpos, vend - vpos);
}

std::string infer_error_node_name(const std::string& pipeline) {
  for (const char* element :
       {"neatdecoder", "neatencoder", "neatprocesscvu", "neatprocessmla", "neatboxdecode",
        "neatdequant", "neatdetess"}) {
    const std::string name = parse_named_element_for_error(pipeline, element);
    if (!name.empty())
      return name;
  }
  return parse_named_element_for_error(pipeline, "name=");
}

[[noreturn]] void throw_preflight_failure(const char* where, const std::string& pipeline,
                                          const std::shared_ptr<DiagCtx>& diag,
                                          const std::string& detail) {
  SessionReport rep = diag ? diag->snapshot_basic() : SessionReport{};
  rep.error_code = error_codes::kRuntimePull;

  std::ostringstream note;
  note << "where=" << (where ? where : "Session::build(input)_preflight")
       << " code=" << rep.error_code << " summary=GST ERROR";
  const std::string node = infer_error_node_name(pipeline);
  if (!node.empty()) {
    note << " node='" << node << "'";
  }
  note << " details='preflight failed: " << detail << "'";
  const std::string boundary = boundary_summary_local(diag);
  if (!boundary.empty()) {
    note << "\n" << boundary;
  }
  note << "\nHint: inspect node configuration/caps and upstream bus diagnostics.";
  rep.repro_note = note.str();
  throw SessionError(session_build_decorate_with_error_code(rep.error_code, rep.repro_note),
                     std::move(rep));
}

bool supports_single_sample_preflight(const std::string& pipeline) {
  const std::string lower = lower_copy(pipeline);
  if (lower.find("neatencoder") != std::string::npos)
    return false;
  return true;
}

template <typename InputT>
bool supports_single_sample_preflight_input(const InputT&) {
  return true;
}

template <typename InputT> struct BuildInputStreamTraits;

template <> struct BuildInputStreamTraits<cv::Mat> {
  static const char* trace_start() {
    return "build(input): start";
  }
  static const char* trace_parsed() {
    return "build(input): parsed";
  }
  static const char* trace_set_state_playing() {
    return "build(input): set_state PLAYING";
  }
  static const char* trace_playing() {
    return "build(input): playing";
  }
  static const char* trace_preflight_start() {
    return "build(input): preflight start";
  }
  static const char* trace_preflight_done() {
    return "build(input): preflight done";
  }
  static const char* trace_done() {
    return "build(input): done";
  }
  static const char* pipeline_state_label() {
    return "build(input)";
  }
  static const char* debug_label() {
    return "build(input)";
  }
  static bool dump_pipeline_string() {
    return false;
  }
  static bool warn_pipeline_state() {
    return true;
  }
  static void run_preflight(InputStream& stream, const cv::Mat& sample, int timeout_ms) {
    stream.push(sample);
    (void)stream.pull(timeout_ms);
  }
};

template <> struct BuildInputStreamTraits<simaai::neat::Tensor> {
  static const char* trace_start() {
    return "build(input neat): start";
  }
  static const char* trace_parsed() {
    return "build(input neat): parsed";
  }
  static const char* trace_set_state_playing() {
    return "build(input neat): set_state PLAYING";
  }
  static const char* trace_playing() {
    return "build(input neat): playing";
  }
  static const char* trace_preflight_start() {
    return "build(input neat): preflight start";
  }
  static const char* trace_preflight_done() {
    return "build(input neat): preflight done";
  }
  static const char* trace_done() {
    return "build(input neat): done";
  }
  static const char* pipeline_state_label() {
    return "build(input neat)";
  }
  static const char* debug_label() {
    return "build(input neat)";
  }
  static bool dump_pipeline_string() {
    return false;
  }
  static bool warn_pipeline_state() {
    return true;
  }
  static void run_preflight(InputStream& stream, const simaai::neat::Tensor& sample,
                            int timeout_ms) {
    stream.push(sample);
    (void)stream.pull(timeout_ms);
  }
};

template <> struct BuildInputStreamTraits<Sample> {
  static const char* trace_start() {
    return "build(input sample): start";
  }
  static const char* trace_parsed() {
    return "build(input sample): parsed";
  }
  static const char* trace_set_state_playing() {
    return "build(input sample): set_state PLAYING";
  }
  static const char* trace_playing() {
    return "build(input sample): playing";
  }
  static const char* trace_preflight_start() {
    return "build(input sample): preflight start";
  }
  static const char* trace_preflight_done() {
    return "build(input sample): preflight done";
  }
  static const char* trace_done() {
    return "build(input sample): done";
  }
  static const char* pipeline_state_label() {
    return "build(input)";
  }
  static const char* debug_label() {
    return "build(input sample)";
  }
  static bool dump_pipeline_string() {
    return true;
  }
  static bool warn_pipeline_state() {
    return false;
  }
  static void run_preflight(InputStream& stream, const Sample& sample, int timeout_ms) {
    Sample preflight_sample = pipeline_internal::canonicalize_tensor_transport_sample(sample);
    stream.push_message(preflight_sample);
    (void)stream.pull(timeout_ms);
  }
};

template <typename InputT>
InputStream run_input_stream_internal_typed(const std::vector<std::shared_ptr<Node>>& nodes,
                                            const std::shared_ptr<void>& guard, const void* owner,
                                            std::string& last_pipeline, const InputT& sample,
                                            const SessionOptions& sess_opt,
                                            const InputStreamOptions& opt,
                                            const NameTransform& name_transform, bool insert_queue2,
                                            int sync_num_buffers_override, bool sync_mode) {
  using Traits = BuildInputStreamTraits<InputT>;

  gst_init_once();
  enforce_push_run_mode(nodes, "Session::build(input)");
  trace_step(Traits::trace_start());

  const bool has_sink = session_build_has_output_appsink(nodes);
  enforce_sink_last_if_present(nodes, "Session::build(input)");

  const Input* src_node = nullptr;
  require_input_appsrc(nodes, "Session::build(input)", &src_node);
  const InputOptions normalized_input_opt = pipeline_internal::normalize_shape_bounds(src_node->options());
  const SampleSpec seed_spec =
      infer_input_spec(normalized_input_opt, sample, "Session::build(input)");
  InputOptions seeded_input_opt = seed_input_options_from_spec(src_node->options(), seed_spec);
  std::vector<std::shared_ptr<Node>> build_nodes =
      replace_first_input_node_for_build(nodes, seeded_input_opt);
  build_nodes = session_build_materialize_model_bound_nodes(build_nodes, sync_mode);
  apply_input_contract_to_nodes(build_nodes, input_contract_from_input(sample));

  require_element("appsrc", "Session::build(input)");
  if (has_sink) {
    require_element("appsink", "Session::build(input)");
    require_element("identity", "Session::build(input)");
  }

  const bool insert_boundaries =
      should_insert_boundaries_for_mode("SIMA_GST_RUN_INSERT_BOUNDARIES", false);

  BuildResult br =
      build_pipeline_full(build_nodes, insert_boundaries, "mysink", insert_queue2, name_transform, &sess_opt);
  maybe_compile_build_result_contracts(&br, &build_nodes, sess_opt,
                                       input_contract_from_input(sample),
                                       seed_spec, contract_compile_sample_from_input(sample),
                                       "Session::build(input)");
  if (sync_mode) {
    br.pipeline_string =
        session_build_clamp_sync_pipeline(std::move(br.pipeline_string), sync_num_buffers_override);
    br.pipeline_string = session_build_clamp_detess_num_buffers(std::move(br.pipeline_string),
                                                                sync_num_buffers_override);
    br.diag->pipeline_string = br.pipeline_string;
  }
  last_pipeline = br.pipeline_string;
  br.pipeline_string = last_pipeline;
  br.diag->pipeline_string = last_pipeline;
  session_build_enforce_mla_num_buffers(last_pipeline, "Session::build(input)", sync_mode);
  if (Traits::dump_pipeline_string()) {
    session_build_maybe_dump_pipeline_string(last_pipeline, "build_input");
  }
  enforce_mla_pipeline_guard("Session::build(input)", last_pipeline, owner);
  validate_caps_override_or_throw(src_node->options(), "Session::build(input)", last_pipeline);

  GstElement* pipeline = session_build_parse_pipeline_or_throw(br, "Session::build(input)");
  session_build_dump_pipeline_element_properties(pipeline);
  trace_step(Traits::trace_parsed());

  if (gst_enforce_names_enabled()) {
    enforce_names_contract(pipeline, br);
  }

  attach_boundary_probes(pipeline, br.diag);
  attach_stage_timing_probes(pipeline, br.diag);
  attach_element_timing_probes(pipeline, br.diag);
  attach_element_flow_probes(pipeline, br.diag);
  session_build_attach_debug_detess_input_probes(pipeline);
  session_build_attach_debug_appsink_probes(pipeline);
  session_build_attach_debug_all_buffer_probes(pipeline);
  session_build_attach_debug_element_buffer_probes(pipeline);

  GstElement* sink = nullptr;
  if (has_sink) {
    const std::string appsink_name = apply_name_transform(name_transform, "mysink");
    sink = gst_bin_get_by_name(GST_BIN(pipeline), appsink_name.c_str());
    if (!sink) {
      maybe_dump_dot(pipeline, "build_input_missing_mysink");
      stop_and_unref(pipeline);
      session_build_throw_session_error_simple(error_codes::kPipelineShape,
                                               "Session::build(input): appsink '" + appsink_name +
                                                   "' not found.\nPipeline:\n" + last_pipeline,
                                               "Add Output() as the last node.", last_pipeline);
    }
    session_build_configure_appsink_for_input_stream(sink, opt);
    session_build_configure_appsink_allocation_preference(sink, nodes);
  }

  const std::string appsrc_name = apply_name_transform(name_transform, "mysrc");
  GstElement* appsrc = gst_bin_get_by_name(GST_BIN(pipeline), appsrc_name.c_str());
  if (!appsrc) {
    if (sink)
      gst_object_unref(sink);
    stop_and_unref(pipeline);
    session_build_throw_session_error_simple(error_codes::kPipelineShape,
                                             "Session::build(input): appsrc '" + appsrc_name +
                                                 "' not found.\nPipeline:\n" + last_pipeline,
                                             "Add Input() as the first node.", last_pipeline);
  }

  SampleSpec spec = seed_spec;
  InputOptions src_opt = session_build_resolve_appsrc_options(normalized_input_opt, name_transform);
  const bool memory_policy_auto_applied =
      apply_auto_memory_policy_from_downstream(src_opt, build_nodes);
  const std::string first_effective_downstream_kind =
      infer_first_effective_downstream_kind(build_nodes);
  if (src_opt.media_type.empty()) {
    src_opt.media_type = seed_spec.media_type;
  }
  if (sync_mode && src_opt.pool_min_buffers == 1 && src_opt.pool_max_buffers == 2) {
    const int max_buffers = max_num_buffers_in_pipeline_local(last_pipeline);
    src_opt.pool_max_buffers = std::max(1, max_buffers);
  }

  InputStreamOptions stream_opt = opt;
  const std::size_t bounded_estimate_bytes =
      static_cast<std::size_t>(session_build_estimate_frame_bytes_limit(src_opt, spec));
  const auto resolved_input_policy =
      pipeline_internal::resolve_session_input_policy(src_opt, spec, stream_opt.max_input_bytes,
                                                     bounded_estimate_bytes);
  stream_opt.shape_policy = resolved_input_policy.shape_policy;
  stream_opt.shape_limits = resolved_input_policy.shape_limits;
  stream_opt.max_input_bytes = resolved_input_policy.max_input_bytes_guard;
  stream_opt.byte_guard_origin = resolved_input_policy.byte_guard_origin;
  validate_shape_limits_or_throw(stream_opt.shape_limits, "Session::build(input)");
  stream_opt.dynamic_capability =
      detect_dynamic_capability(nodes, src_opt, spec, stream_opt.shape_policy);
  stream_opt.allow_ingress_cvu_format_renegotiation = detect_allow_ingress_format_renegotiation(
      nodes, src_opt, spec, stream_opt.shape_policy, stream_opt.dynamic_capability);
  stream_opt.require_device_visible_input =
      (src_opt.memory_policy == InputMemoryPolicy::Ev74 ||
       src_opt.memory_policy == InputMemoryPolicy::Dms0);

  BuildAdaptationSummary adaptation;
  adaptation.shape_policy = shape_policy_name(stream_opt.shape_policy);
  adaptation.dynamic_capability = dynamic_capability_name(stream_opt.dynamic_capability);
  adaptation.seed_width = stream_opt.shape_limits.seed_width;
  adaptation.seed_height = stream_opt.shape_limits.seed_height;
  adaptation.seed_depth = stream_opt.shape_limits.seed_depth;
  adaptation.seed_width_origin = limit_origin_name(stream_opt.shape_limits.seed_width_origin);
  adaptation.seed_height_origin = limit_origin_name(stream_opt.shape_limits.seed_height_origin);
  adaptation.seed_depth_origin = limit_origin_name(stream_opt.shape_limits.seed_depth_origin);
  adaptation.max_width = stream_opt.shape_limits.max_width;
  adaptation.max_height = stream_opt.shape_limits.max_height;
  adaptation.max_depth = stream_opt.shape_limits.max_depth;
  adaptation.max_width_origin = limit_origin_name(stream_opt.shape_limits.max_width_origin);
  adaptation.max_height_origin = limit_origin_name(stream_opt.shape_limits.max_height_origin);
  adaptation.max_depth_origin = limit_origin_name(stream_opt.shape_limits.max_depth_origin);
  adaptation.allow_ingress_cvu_format_renegotiation =
      stream_opt.allow_ingress_cvu_format_renegotiation;

  {
    std::ostringstream detail;
    detail << "seed=" << adaptation.seed_width << "x" << adaptation.seed_height << "x"
           << adaptation.seed_depth << " max=" << adaptation.max_width << "x"
           << adaptation.max_height << "x" << adaptation.max_depth;
    add_build_adaptation_action(adaptation, "input_constraints", true, detail.str());
  }

  add_build_adaptation_action(
      adaptation, "dynamic_capability", true,
      std::string("shape_policy=") + adaptation.shape_policy + " capability=" +
          adaptation.dynamic_capability);

  add_build_adaptation_action(
      adaptation, "format_renegotiation_gate",
      stream_opt.allow_ingress_cvu_format_renegotiation,
      stream_opt.allow_ingress_cvu_format_renegotiation
          ? "ingress raw format changes allowed for this graph"
          : "ingress format changes are rebuild-only for this graph",
      stream_opt.allow_ingress_cvu_format_renegotiation
          ? std::string()
          : "downstream output contract could not be proven stable for format changes");

  if (inputstream_debug_enabled_for_build()) {
    std::fprintf(stderr,
                 "[DBG] build(input) shape-policy=%d capability=%d allow-fmt-reneg=%d "
                 "seed=%dx%dx%d max=%dx%dx%d\n",
                 static_cast<int>(stream_opt.shape_policy),
                 static_cast<int>(stream_opt.dynamic_capability),
                 stream_opt.allow_ingress_cvu_format_renegotiation ? 1 : 0,
                 stream_opt.shape_limits.seed_width, stream_opt.shape_limits.seed_height,
                 stream_opt.shape_limits.seed_depth, stream_opt.shape_limits.max_width,
                 stream_opt.shape_limits.max_height, stream_opt.shape_limits.max_depth);
  }

  src_opt.max_bytes = resolve_appsrc_max_bytes(src_opt, spec);

  adaptation.max_input_bytes_guard = stream_opt.max_input_bytes;
  adaptation.byte_guard_origin = byte_guard_origin_name(stream_opt.byte_guard_origin);
  add_build_adaptation_action(adaptation, "byte_guard", true,
                              std::string("max_input_bytes=") +
                                  std::to_string(stream_opt.max_input_bytes),
                              std::string());

  add_build_adaptation_action(
      adaptation, "appsrc_caps_seed", src_opt.caps_override.empty(),
      src_opt.caps_override.empty()
          ? std::string("caps derived from resolved seed input")
          : std::string("caps_override=") + src_opt.caps_override,
      src_opt.caps_override.empty() ? std::string()
                                    : std::string("caps_override is authoritative"));

  {
    std::ostringstream detail;
    detail << "policy=" << input_memory_policy_name(src_opt.memory_policy)
           << " use_simaai_pool=" << (src_opt.use_simaai_pool ? 1 : 0)
           << " first_downstream=" << first_effective_downstream_kind;
    add_build_adaptation_action(
        adaptation, "appsrc_memory_policy", true, detail.str(),
        memory_policy_auto_applied
            ? "auto policy resolved from first downstream node"
            : "policy already explicit (not auto-overridden)");
  }

  if (br.diag) {
    br.diag->has_build_adaptation = true;
    br.diag->build_adaptation = adaptation;
  }

  GstCaps* caps = nullptr;
  if (!src_opt.caps_override.empty()) {
    caps = gst_caps_from_string(src_opt.caps_override.c_str());
    if (!caps) {
      if (sink)
        gst_object_unref(sink);
      gst_object_unref(appsrc);
      stop_and_unref(pipeline);
      session_build_throw_session_error_simple(
          error_codes::kCaps,
          "Session::build(input): invalid caps_override: " + src_opt.caps_override,
          "Provide a valid GstCaps string matching the input payload.", last_pipeline);
    }
  } else {
    caps = caps_from_spec(spec);
  }
  gst_app_src_set_caps(GST_APP_SRC(appsrc), caps);
  gst_caps_unref(caps);

  configure_appsrc(appsrc, src_opt);
  maybe_log_appsrc_state(appsrc);
  if (sink)
    maybe_log_appsink_state(sink);

  try {
    trace_step(Traits::trace_set_state_playing());
    set_state_or_throw(pipeline, GST_STATE_PLAYING, "Session::build(input)", br.diag);
  } catch (...) {
    gst_object_unref(appsrc);
    if (sink)
      gst_object_unref(sink);
    stop_and_unref(pipeline);
    throw;
  }
  trace_step(Traits::trace_playing());
  maybe_log_pipeline_state(pipeline, Traits::pipeline_state_label());

  if (Traits::warn_pipeline_state() && inputstream_warn_or_debug_enabled_for_build()) {
    GstState cur = GST_STATE_VOID_PENDING;
    GstState pend = GST_STATE_VOID_PENDING;
    gst_element_get_state(pipeline, &cur, &pend, 0);
    if (cur < GST_STATE_PAUSED) {
      std::fprintf(stderr,
                   "[WARN] build(input): pipeline state is %s (pending %s); "
                   "streaming may stall until it reaches PAUSED/PLAYING.\n",
                   gst_element_state_get_name(cur), gst_element_state_get_name(pend));
    }
  }

  InputStream stream =
      InputStream::create(pipeline, appsrc, sink, spec, src_opt, stream_opt, br.diag, guard);
  install_input_contract_caps_change_callback(stream, build_nodes, spec);
  const int max_buffers = max_num_buffers_in_pipeline_local(last_pipeline);
  const bool want_preflight = resolve_startup_preflight(opt);
  const bool pipeline_preflight_supported = supports_single_sample_preflight(last_pipeline);
  const bool input_preflight_supported = supports_single_sample_preflight_input(sample);
  const bool preflight_supported = pipeline_preflight_supported && input_preflight_supported;
  if (want_preflight && sink && preflight_supported) {
    if (sync_mode && max_buffers > 1) {
      if (inputstream_debug_enabled_for_build()) {
        std::fprintf(stderr, "[DBG] %s: skip preflight (sync, num-buffers=%d)\n",
                     Traits::debug_label(), max_buffers);
      }
    } else {
      if (inputstream_debug_enabled_for_build()) {
        std::fprintf(stderr, "[DBG] %s: preflight push/pull\n", Traits::debug_label());
      }
      trace_step(Traits::trace_preflight_start());
      try {
        Traits::run_preflight(stream, sample, opt.timeout_ms);
      } catch (const SessionError&) {
        throw;
      } catch (const std::exception& e) {
        try {
          session_build_throw_if_bus_error_local(pipeline, br.diag,
                                                 "Session::build(input)_preflight");
        } catch (const SessionError&) {
          throw;
        }
        throw_preflight_failure("Session::build(input)_preflight", last_pipeline, br.diag,
                                e.what());
      }
      trace_step(Traits::trace_preflight_done());
    }
  } else if (want_preflight && !preflight_supported && inputstream_debug_enabled_for_build()) {
    if (!pipeline_preflight_supported) {
      std::fprintf(stderr, "[DBG] %s: skip preflight (pipeline requires warmup)\n",
                   Traits::debug_label());
    }
  }

  trace_step(Traits::trace_done());
  return stream;
}

InputStream run_input_stream_internal(const std::vector<std::shared_ptr<Node>>& nodes,
                                      const std::shared_ptr<void>& guard, const void* owner,
                                      std::string& last_pipeline, const cv::Mat& sample,
                                      const SessionOptions& sess_opt,
                                      const InputStreamOptions& opt,
                                      const NameTransform& name_transform, bool insert_queue2,
                                      int sync_num_buffers_override, bool sync_mode) {
  return run_input_stream_internal_typed(nodes, guard, owner, last_pipeline, sample, sess_opt, opt,
                                         name_transform, insert_queue2, sync_num_buffers_override,
                                         sync_mode);
}

InputStream run_input_stream_internal(const std::vector<std::shared_ptr<Node>>& nodes,
                                      const std::shared_ptr<void>& guard, const void* owner,
                                      std::string& last_pipeline,
                                      const simaai::neat::Tensor& sample,
                                      const SessionOptions& sess_opt,
                                      const InputStreamOptions& opt,
                                      const NameTransform& name_transform, bool insert_queue2,
                                      int sync_num_buffers_override, bool sync_mode) {
  return run_input_stream_internal_typed(nodes, guard, owner, last_pipeline, sample, sess_opt, opt,
                                         name_transform, insert_queue2, sync_num_buffers_override,
                                         sync_mode);
}

InputStream run_input_stream_internal(const std::vector<std::shared_ptr<Node>>& nodes,
                                      const std::shared_ptr<void>& guard, const void* owner,
                                      std::string& last_pipeline, const Sample& sample,
                                      const SessionOptions& sess_opt,
                                      const InputStreamOptions& opt,
                                      const NameTransform& name_transform, bool insert_queue2,
                                      int sync_num_buffers_override, bool sync_mode) {
  return run_input_stream_internal_typed(nodes, guard, owner, last_pipeline, sample, sess_opt, opt,
                                         name_transform, insert_queue2, sync_num_buffers_override,
                                         sync_mode);
}

template <typename InputT>
Sample run_sync_once_typed(Run& runner, const InputT& input, int timeout_ms) {
  if constexpr (std::is_same_v<std::decay_t<InputT>, cv::Mat>) {
    return sample_from_tensors(runner.run(std::vector<cv::Mat>{input}, timeout_ms));
  } else if constexpr (std::is_same_v<std::decay_t<InputT>, simaai::neat::Tensor>) {
    return sample_from_tensors(runner.run(TensorList{input}, timeout_ms));
  } else {
    SampleList outputs = runner.run(SampleList{input}, timeout_ms);
    if (outputs.empty()) {
      throw std::runtime_error("Session::run(input): SampleList run returned no outputs");
    }
    return std::move(outputs.front());
  }
}

bool is_run_timeout_exception(const std::exception& e) {
  const std::string_view msg = e.what();
  return msg.find("status=Timeout") != std::string_view::npos ||
         msg.find("timeout waiting for output") != std::string_view::npos;
}

template <typename InputT>
Sample run_sync_prefill_typed(Run& runner, const InputT& input, int timeout_ms, int count,
                              bool allow_startup_lag) {
  Sample out;
  bool saw_output = false;
  std::string last_timeout;
  int startup_lag_budget = allow_startup_lag ? std::max(1, count * 3) : 0;
  int target_pushes = std::max(1, count);
  for (int i = 0; i < target_pushes; ++i) {
    try {
      out = run_sync_once_typed(runner, input, timeout_ms);
      saw_output = true;
      continue;
    } catch (const std::exception& e) {
      if (!is_run_timeout_exception(e)) {
        throw;
      }
      last_timeout = e.what();
    }

    if (startup_lag_budget > 0) {
      if (inputstream_debug_enabled_for_build()) {
        std::fprintf(stderr,
                     "[DBG] Session::run(input): prefill startup-lag compensation "
                     "iteration=%d/%d remaining=%d\n",
                     i + 1, target_pushes, startup_lag_budget);
      }
      --startup_lag_budget;
      ++target_pushes;
      continue;
    }
    break;
  }

  if (!saw_output) {
    std::ostringstream oss;
    oss << "Session::run(input): prefill stage produced no output"
        << " pushes=" << target_pushes << " timeout_ms=" << timeout_ms;
    if (!last_timeout.empty()) {
      oss << ": " << last_timeout;
    }
    const std::string last_err = runner.last_error();
    if (!last_err.empty()) {
      oss << ": " << last_err;
    }
    session_build_throw_session_error_simple(
        error_codes::kRuntimePull, oss.str(),
        "Inspect Run::report()/SessionReport bus diagnostics.");
  }

  return out;
}

Sample run_sync_prefill(Run& runner, const cv::Mat& input, int timeout_ms, int count,
                        bool allow_startup_lag) {
  return run_sync_prefill_typed(runner, input, timeout_ms, count, allow_startup_lag);
}

Sample run_sync_prefill(Run& runner, const simaai::neat::Tensor& input, int timeout_ms, int count,
                        bool allow_startup_lag) {
  return run_sync_prefill_typed(runner, input, timeout_ms, count, allow_startup_lag);
}

Sample run_sync_prefill(Run& runner, const Sample& input, int timeout_ms, int count,
                        bool allow_startup_lag) {
  return run_sync_prefill_typed(runner, input, timeout_ms, count, allow_startup_lag);
}

bool run_options_equal_for_cache_local(const RunOptions& a, const RunOptions& b) {
  const auto rail_options_equal = [](const PowerMonitorOptions& lhs,
                                     const PowerMonitorOptions& rhs) {
    if (lhs.enabled != rhs.enabled || lhs.sample_interval_ms != rhs.sample_interval_ms ||
        lhs.profile != rhs.profile || lhs.rails.size() != rhs.rails.size()) {
      return false;
    }
    for (std::size_t i = 0; i < lhs.rails.size(); ++i) {
      const auto& l = lhs.rails[i];
      const auto& r = rhs.rails[i];
      if (l.name != r.name || l.i2c_bus != r.i2c_bus || l.i2c_addr != r.i2c_addr ||
          l.page != r.page || l.vout_exponent != r.vout_exponent ||
          l.iout_exponent != r.iout_exponent || l.pout_exponent != r.pout_exponent) {
        return false;
      }
    }
    return true;
  };
  return a.preset == b.preset && a.queue_depth == b.queue_depth &&
         a.overflow_policy == b.overflow_policy && a.output_memory == b.output_memory &&
         a.enable_metrics == b.enable_metrics && a.input_timeout_ms == b.input_timeout_ms &&
         a.advanced.copy_input == b.advanced.copy_input &&
         a.advanced.max_input_bytes == b.advanced.max_input_bytes &&
         a.advanced.sync_num_buffers_override == b.advanced.sync_num_buffers_override &&
         a.advanced.prepare_output_cpu_visible == b.advanced.prepare_output_cpu_visible &&
         rail_options_equal(a.power_monitor, b.power_monitor);
}

template <typename CachePtrT>
bool should_reuse_sync_cache(const CachePtrT& run_cache, RunInputKind kind, uint64_t nodes_version,
                             const RunOptions& run_opt, const CapKey& caps_key) {
  return run_cache && run_cache->input_kind == kind && run_cache->nodes_version == nodes_version &&
         run_options_equal_for_cache_local(run_cache->opt, run_opt) &&
         run_cache->caps_key == caps_key;
}

template <typename CachePtrT>
void set_sync_cache(CachePtrT& run_cache, Run runner, const CapKey& caps_key,
                    const RunOptions& run_opt, uint64_t nodes_version, RunInputKind kind) {
  using CacheType = typename std::decay_t<CachePtrT>::element_type;
  auto cache = std::make_unique<CacheType>();
  cache->runner = std::move(runner);
  cache->caps_key = caps_key;
  cache->opt = run_opt;
  cache->nodes_version = nodes_version;
  cache->input_kind = kind;
  run_cache = std::move(cache);
}

template <typename InputT, typename NodesVersionT, typename CachePtrT, typename BuildFn>
Sample run_sync_cached_input(const std::vector<std::shared_ptr<Node>>& nodes,
                             NodesVersionT& nodes_version, std::string& last_pipeline,
                             CachePtrT& run_cache, const InputT& input, const RunOptions& opt,
                             RunInputKind kind, BuildFn&& build_sync_runner) {
  gst_init_once();

  enforce_caps_behavior(nodes, "Session::run(input)");
  enforce_sink_last(nodes);
  enforce_push_run_mode(nodes, "Session::run(input)");

  const Input* src_node = nullptr;
  require_input_appsrc(nodes, "Session::run(input)", &src_node);

  const SampleSpec spec =
      infer_input_spec(pipeline_internal::normalize_shape_bounds(src_node->options()), input,
                       "Session::run(input)");
  const uint64_t version = nodes_version.load(std::memory_order_relaxed);
  const RunOptions run_opt = session_build_resolve_build_opt(RunMode::Sync, opt);
  const int timeout_ms = resolved_input_timeout_ms(run_opt);
  const bool reuse_cache =
      should_reuse_sync_cache(run_cache, kind, version, run_opt, spec.caps_key);

  if (session_sync_cache_debug_enabled()) {
    const bool cache_present = static_cast<bool>(run_cache);
    const bool cache_kind_match = cache_present && run_cache->input_kind == kind;
    const bool cache_version_match = cache_present && run_cache->nodes_version == version;
    const bool cache_opt_match =
        cache_present && run_options_equal_for_cache_local(run_cache->opt, run_opt);
    const bool cache_caps_match = cache_present && run_cache->caps_key == spec.caps_key;
    std::fprintf(
        stderr,
        "[sync-cache] pre kind=%s cache_present=%d reuse=%d nodes_version=%llu "
        "cache_nodes_version=%llu kind_match=%d version_match=%d opt_match=%d caps_match=%d "
        "timeout_ms=%d pipeline_len=%zu\n",
        run_input_kind_name(kind), cache_present ? 1 : 0, reuse_cache ? 1 : 0,
        static_cast<unsigned long long>(version),
        cache_present ? static_cast<unsigned long long>(run_cache->nodes_version) : 0ULL,
        cache_kind_match ? 1 : 0, cache_version_match ? 1 : 0, cache_opt_match ? 1 : 0,
        cache_caps_match ? 1 : 0, timeout_ms, last_pipeline.size());
  }

  if (!reuse_cache) {
    pipeline_internal::ScopedSyncBuild sync_guard(true);
    set_sync_cache(run_cache, std::forward<BuildFn>(build_sync_runner)(run_opt), spec.caps_key,
                   run_opt, version, kind);
    if (session_sync_cache_debug_enabled()) {
      std::fprintf(
          stderr,
          "[sync-cache] build_new kind=%s run_cache=%p runner_obj=%p nodes_version=%llu "
          "sync_prefill_warmed=%d\n",
          run_input_kind_name(kind), run_cache ? static_cast<void*>(run_cache.get()) : nullptr,
          run_cache ? static_cast<void*>(std::addressof(run_cache->runner)) : nullptr,
          static_cast<unsigned long long>(version),
          run_cache && run_cache->sync_prefill_warmed ? 1 : 0);
    }
  } else if (session_sync_cache_debug_enabled()) {
    std::fprintf(stderr,
                 "[sync-cache] reuse kind=%s run_cache=%p runner_obj=%p sync_prefill_warmed=%d\n",
                 run_input_kind_name(kind), run_cache ? static_cast<void*>(run_cache.get()) : nullptr,
                 run_cache ? static_cast<void*>(std::addressof(run_cache->runner)) : nullptr,
                 run_cache && run_cache->sync_prefill_warmed ? 1 : 0);
  }
  auto run_with_current_runner = [&]() -> Sample {
    const int max_buffers = max_num_buffers_in_pipeline_local(last_pipeline);
    if (session_sync_cache_debug_enabled()) {
      std::fprintf(stderr,
                   "[sync-cache] run kind=%s run_cache=%p runner_obj=%p max_buffers=%d timeout_ms=%d "
                   "prefill_warmed=%d\n",
                   run_input_kind_name(kind), run_cache ? static_cast<void*>(run_cache.get()) : nullptr,
                   run_cache ? static_cast<void*>(std::addressof(run_cache->runner)) : nullptr,
                   max_buffers, timeout_ms, run_cache && run_cache->sync_prefill_warmed ? 1 : 0);
    }
    if (max_buffers > 1) {
      const bool allow_startup_lag = !run_cache->sync_prefill_warmed;
      Sample out = run_sync_prefill(run_cache->runner, input, timeout_ms, max_buffers,
                                    allow_startup_lag);
      run_cache->sync_prefill_warmed = true;
      return out;
    }
    run_cache->sync_prefill_warmed = true;
    return run_sync_once_typed(run_cache->runner, input, timeout_ms);
  };

  try {
    return run_with_current_runner();
  } catch (const SessionError&) {
    if (!reuse_cache) {
      throw;
    }
    if (session_sync_cache_debug_enabled()) {
      std::fprintf(stderr,
                   "[sync-cache] rebuild_after_session_error kind=%s old_run_cache=%p old_runner_obj=%p\n",
                   run_input_kind_name(kind), run_cache ? static_cast<void*>(run_cache.get()) : nullptr,
                   run_cache ? static_cast<void*>(std::addressof(run_cache->runner)) : nullptr);
    }
    if (inputstream_debug_enabled_for_build()) {
      std::fprintf(stderr,
                   "[DBG] Session::run(input): cached sync runner failed; rebuilding and retrying\n");
    }
    pipeline_internal::ScopedSyncBuild sync_guard(true);
    set_sync_cache(run_cache, std::forward<BuildFn>(build_sync_runner)(run_opt), spec.caps_key,
                   run_opt, version, kind);
    if (session_sync_cache_debug_enabled()) {
      std::fprintf(stderr, "[sync-cache] rebuilt kind=%s new_run_cache=%p new_runner_obj=%p\n",
                   run_input_kind_name(kind), run_cache ? static_cast<void*>(run_cache.get()) : nullptr,
                   run_cache ? static_cast<void*>(std::addressof(run_cache->runner)) : nullptr);
    }
    return run_with_current_runner();
  } catch (const std::exception&) {
    if (!reuse_cache) {
      throw;
    }
    if (session_sync_cache_debug_enabled()) {
      std::fprintf(stderr,
                   "[sync-cache] rebuild_after_std_exception kind=%s old_run_cache=%p old_runner_obj=%p\n",
                   run_input_kind_name(kind), run_cache ? static_cast<void*>(run_cache.get()) : nullptr,
                   run_cache ? static_cast<void*>(std::addressof(run_cache->runner)) : nullptr);
    }
    if (inputstream_debug_enabled_for_build()) {
      std::fprintf(stderr,
                   "[DBG] Session::run(input): cached sync runner std::exception; rebuilding and retrying\n");
    }
    pipeline_internal::ScopedSyncBuild sync_guard(true);
    set_sync_cache(run_cache, std::forward<BuildFn>(build_sync_runner)(run_opt), spec.caps_key,
                   run_opt, version, kind);
    if (session_sync_cache_debug_enabled()) {
      std::fprintf(stderr, "[sync-cache] rebuilt kind=%s new_run_cache=%p new_runner_obj=%p\n",
                   run_input_kind_name(kind), run_cache ? static_cast<void*>(run_cache.get()) : nullptr,
                   run_cache ? static_cast<void*>(std::addressof(run_cache->runner)) : nullptr);
    }
    return run_with_current_runner();
  }
}

} // namespace

void session_build_apply_derived_input_contracts(std::vector<std::shared_ptr<Node>>* nodes) {
  if (!nodes || nodes->empty()) {
    return;
  }
  for (std::size_t i = 1; i < nodes->size(); ++i) {
    auto* configurable = dynamic_cast<InputContractConfigurable*>((*nodes)[i].get());
    if (!configurable) {
      continue;
    }

    const NodeGroup upstream_group(std::vector<std::shared_ptr<Node>>(nodes->begin(),
                                                                      nodes->begin() + i));
    OutputSpec upstream_spec;
    try {
      upstream_spec = derive_output_spec(upstream_group, {});
    } catch (...) {
      continue;
    }

    if (upstream_spec.width <= 0 || upstream_spec.height <= 0 || upstream_spec.format.empty()) {
      continue;
    }

    InputContract contract;
    contract.media_type = upstream_spec.media_type;
    contract.format = upstream_spec.format;
    contract.dtype = upstream_spec.dtype;
    contract.layout = upstream_spec.layout;
    contract.width = upstream_spec.width;
    contract.height = upstream_spec.height;
    contract.depth = upstream_spec.depth;

    std::string err;
    configurable->apply_input_contract(contract, &err);
    if (!err.empty()) {
      throw std::runtime_error(err);
    }
  }
}

Run Session::build(const std::vector<cv::Mat>& inputs, RunMode mode, const RunOptions& opt) {
  pipeline_internal::ux::ScopedVerboseContext verbose_ctx(opt_.verbose);
  pipeline_internal::ux::ProgressReporter progress(opt_.verbose, 4);
  progress.step("Initializing runtime...");
  gst_init_once();
  if (inputs.empty()) {
    throw std::runtime_error("Session::build(inputs): empty image list");
  }
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i].empty()) {
      throw std::runtime_error("Session::build(inputs): empty image input at index " +
                               std::to_string(i));
    }
  }
  const BuildInputContext ctx = session_build_prepare_build_input_context(nodes_, opt_, mode, opt);
  progress.step("Preparing input stream...");
  InputStreamOptions build_stream_opt = ctx.stream_opt;
  // build(inputs) uses the provided payload only to seed caps/spec. It must not
  // consume that same payload through startup preflight before the caller
  // explicitly pushes or runs.
  build_stream_opt.startup_preflight = false;
  InputOptions tensor_src_opt = pipeline_internal::normalize_shape_bounds(ctx.src_node->options());
  if (!input_options_expect_tensor_media(tensor_src_opt)) {
    if (inputs.size() != 1U) {
      throw std::runtime_error(
          "Session::build(inputs): raw-image ingress supports exactly one cv::Mat per inference item");
    }
    progress.step("Building session...");
    InputStream stream = session_build_run_input_stream_internal(
        nodes_, guard_, this, last_pipeline_, inputs.front(), opt_, build_stream_opt, ctx.name_transform,
        ctx.insert_queue2, ctx.sync_num_buffers_override, ctx.mode == RunMode::Sync);
    progress.done("Session ready");
    return Run::create(std::move(stream), ctx.merged_opt, build_stream_opt, ctx.mode, std::nullopt,
                       input_route_processor_);
  }
  TensorList tensors = tensor_list_from_mats(inputs, tensor_src_opt, "Session::build(inputs)");
  const Sample seed =
      input_route_processor_ ? input_route_processor_->process_tensors(tensors, "Session::build(inputs)")
                             : pipeline_internal::sample_from_tensors_for_input(tensors, tensor_src_opt);
  progress.step("Building session...");
  InputStream stream = session_build_run_input_stream_internal(
      nodes_, guard_, this, last_pipeline_, seed, opt_, build_stream_opt, ctx.name_transform,
      ctx.insert_queue2, ctx.sync_num_buffers_override, ctx.mode == RunMode::Sync);
  progress.done("Session ready");
  return Run::create(std::move(stream), ctx.merged_opt, build_stream_opt, ctx.mode, tensor_src_opt,
                     input_route_processor_);
}

Run Session::build(const TensorList& inputs, RunMode mode, const RunOptions& opt) {
  pipeline_internal::ux::ScopedVerboseContext verbose_ctx(opt_.verbose);
  pipeline_internal::ux::ProgressReporter progress(opt_.verbose, 4);
  progress.step("Initializing runtime...");
  gst_init_once();
  if (inputs.empty()) {
    throw std::runtime_error("Session::build(inputs): empty tensor list");
  }
  const BuildInputContext ctx = session_build_prepare_build_input_context(nodes_, opt_, mode, opt);
  progress.step("Preparing input stream...");
  InputStreamOptions build_stream_opt = ctx.stream_opt;
  build_stream_opt.startup_preflight = false;
  const InputOptions src_opt = pipeline_internal::normalize_shape_bounds(ctx.src_node->options());
  const Sample seed =
      input_route_processor_ ? input_route_processor_->seed_tensors(inputs, "Session::build(inputs)")
                             : pipeline_internal::sample_from_tensors_for_input(inputs, src_opt);
  progress.step("Building session...");
  InputStream stream = session_build_run_input_stream_internal(
      nodes_, guard_, this, last_pipeline_, seed, opt_, build_stream_opt, ctx.name_transform,
      ctx.insert_queue2, ctx.sync_num_buffers_override, ctx.mode == RunMode::Sync);
  const std::optional<InputOptions> tensor_src_opt =
      input_options_expect_tensor_media(src_opt) ? std::optional<InputOptions>(src_opt) : std::nullopt;
  progress.done("Session ready");
  return Run::create(std::move(stream), ctx.merged_opt, build_stream_opt, ctx.mode, tensor_src_opt,
                     input_route_processor_);
}

Run Session::build(const SampleList& inputs, RunMode mode, const RunOptions& opt) {
  pipeline_internal::ux::ScopedVerboseContext verbose_ctx(opt_.verbose);
  pipeline_internal::ux::ProgressReporter progress(opt_.verbose, 4);
  progress.step("Initializing runtime...");
  gst_init_once();
  if (inputs.empty()) {
    throw std::runtime_error("Session::build(inputs): empty sample list");
  }
  const BuildInputContext ctx = session_build_prepare_build_input_context(nodes_, opt_, mode, opt);
  progress.step("Preparing input stream...");
  InputStreamOptions build_stream_opt = ctx.stream_opt;
  build_stream_opt.startup_preflight = false;
  const Sample seed =
      input_route_processor_ ? input_route_processor_->seed_samples(inputs, "Session::build(inputs)")
                             : inputs.front();
  progress.step("Building session...");
  InputStream stream = session_build_run_input_stream_internal(
      nodes_, guard_, this, last_pipeline_, seed, opt_, build_stream_opt, ctx.name_transform,
      ctx.insert_queue2, ctx.sync_num_buffers_override, ctx.mode == RunMode::Sync);
  progress.done("Session ready");
  return Run::create(std::move(stream), ctx.merged_opt, build_stream_opt, ctx.mode, std::nullopt,
                     input_route_processor_);
}

TensorList Session::run(const std::vector<cv::Mat>& inputs, const RunOptions& opt) {
  (void)tensor_list_from_mats(inputs, InputOptions{}, "Session::run(inputs)");
  if (inputs.size() == 1U) {
    Sample out = run_sync_cached_input(
        nodes_, nodes_version_, last_pipeline_, run_cache_, inputs.front(), opt, RunInputKind::Mat,
        [this, &input = inputs.front()](const RunOptions& run_opt) {
          return build(std::vector<cv::Mat>{input}, RunMode::Sync, run_opt);
        });
    return tensors_from_sample(std::move(out), false);
  }
  Run runner = build(inputs, RunMode::Sync, opt);
  return runner.run(inputs, resolved_input_timeout_ms(opt));
}

TensorList Session::run(const TensorList& inputs, const RunOptions& opt) {
  if (inputs.empty()) {
    throw std::runtime_error("Session::run(inputs): empty tensor list");
  }
  if (inputs.size() == 1U) {
    Sample out = run_sync_cached_input(
        nodes_, nodes_version_, last_pipeline_, run_cache_, inputs.front(), opt,
        RunInputKind::Tensor,
        [this, &input = inputs.front()](const RunOptions& run_opt) {
          return build(TensorList{input}, RunMode::Sync, run_opt);
        });
    return tensors_from_sample(std::move(out), false);
  }
  Run runner = build(inputs, RunMode::Sync, opt);
  return runner.run(inputs, resolved_input_timeout_ms(opt));
}

SampleList Session::run(const SampleList& inputs, const RunOptions& opt) {
  if (inputs.empty()) {
    throw std::runtime_error("Session::run(inputs): empty sample list");
  }
  if (inputs.size() == 1U) {
    Sample out = run_sync_cached_input(
        nodes_, nodes_version_, last_pipeline_, run_cache_, inputs.front(), opt,
        RunInputKind::Sample,
        [this, &input = inputs.front()](const RunOptions& run_opt) {
          return build(SampleList{input}, RunMode::Sync, run_opt);
        });
    return SampleList{std::move(out)};
  }
  Run runner = build(inputs, RunMode::Sync, opt);
  return runner.run(inputs, resolved_input_timeout_ms(opt));
}

RunOptions session_build_apply_run_defaults(const RunOptions& opt, const SessionOptions& sess_opt) {
  (void)sess_opt;
  return opt;
}

RunOptions session_build_resolve_build_opt(RunMode mode, const RunOptions& opt) {
  return resolve_build_opt(mode, opt);
}

bool session_build_should_insert_async_queue2(RunMode mode, const RunOptions& opt) {
  return should_insert_async_queue2(mode, opt);
}

InputStreamOptions session_build_make_stream_options(const RunOptions& opt, RunMode mode) {
  return make_stream_options(opt, mode);
}

void session_build_maybe_enable_rtsp_appsink_drop(InputStreamOptions& stream_opt,
                                                  const std::vector<std::shared_ptr<Node>>& nodes) {
  maybe_enable_rtsp_appsink_drop(stream_opt, nodes);
}

void session_build_maybe_apply_detess_output_override(
    const std::vector<std::shared_ptr<Node>>& nodes, InputStreamOptions& stream_opt) {
  maybe_apply_detess_output_override(nodes, stream_opt);
}

BuildInputContext
session_build_prepare_build_input_context(const std::vector<std::shared_ptr<Node>>& nodes,
                                          const SessionOptions& sess_opt, RunMode mode,
                                          const RunOptions& opt) {
  return prepare_build_input_context(nodes, sess_opt, mode, opt);
}

InputStream session_build_run_input_stream_internal(
    const std::vector<std::shared_ptr<Node>>& nodes, const std::shared_ptr<void>& guard,
    const void* owner, std::string& last_pipeline, const cv::Mat& sample,
    const SessionOptions& sess_opt,
    const InputStreamOptions& opt, const NameTransform& name_transform, bool insert_queue2,
    int sync_num_buffers_override, bool sync_mode) {
  return run_input_stream_internal(nodes, guard, owner, last_pipeline, sample, sess_opt, opt, name_transform,
                                   insert_queue2, sync_num_buffers_override, sync_mode);
}

InputStream session_build_run_input_stream_internal(
    const std::vector<std::shared_ptr<Node>>& nodes, const std::shared_ptr<void>& guard,
    const void* owner, std::string& last_pipeline, const simaai::neat::Tensor& sample,
    const SessionOptions& sess_opt,
    const InputStreamOptions& opt, const NameTransform& name_transform, bool insert_queue2,
    int sync_num_buffers_override, bool sync_mode) {
  return run_input_stream_internal(nodes, guard, owner, last_pipeline, sample, sess_opt, opt, name_transform,
                                   insert_queue2, sync_num_buffers_override, sync_mode);
}

InputStream session_build_run_input_stream_internal(
    const std::vector<std::shared_ptr<Node>>& nodes, const std::shared_ptr<void>& guard,
    const void* owner, std::string& last_pipeline, const Sample& sample,
    const SessionOptions& sess_opt,
    const InputStreamOptions& opt, const NameTransform& name_transform, bool insert_queue2,
    int sync_num_buffers_override, bool sync_mode) {
  return run_input_stream_internal(nodes, guard, owner, last_pipeline, sample, sess_opt, opt, name_transform,
                                   insert_queue2, sync_num_buffers_override, sync_mode);
}

} // namespace simaai::neat
