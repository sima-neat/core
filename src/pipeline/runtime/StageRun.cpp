#include "pipeline/StageRun.h"

#include "builder/PreprocessMetaRequirement.h"
#include "model/internal/ModelInternal.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "pipeline/Graph.h"
#include "pipeline/internal/PipelineBuild.h"
#include "pipeline/internal/contract/ContractApply.h"
#include "pipeline/internal/contract/ContractCompiler.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/RenderedMlaContractQuery.h"
#include "pipeline/internal/sima/ContractRender.h"
#include "pipeline/internal/TensorBufferEnvelope.h"
#include "pipeline/internal/TensorTransfer.h"
#include "pipeline/internal/TensorUtil.h"
#include "pipeline/internal/TensorMath.h"
#include "pipeline/TessellatedTensor.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace simaai::neat::stages {
using simaai::neat::pipeline_internal::upper_copy;
namespace {
Sample run_single_sample(Run& runner, const Sample& input, int timeout_ms, const char* where) {
  Sample outputs = runner.run(Sample{input}, timeout_ms);
  if (outputs.size() != 1) {
    throw std::runtime_error(std::string(where ? where : "StageRun") +
                             ": expected exactly 1 sample output");
  }
  return std::move(outputs.front());
}

enum class StageKind {
  Preproc,
  Infer,
  MLA,
  Postprocess,
  BoxDecode,
};

struct StageInputKey {
  std::string media_type;
  std::string format;
  int width = -1;
  int height = -1;
  int depth = -1;
  TensorDType dtype = TensorDType::Int8;
  TensorLayout layout = TensorLayout::Unknown;
  std::vector<int64_t> shape;
};

bool stage_trace_enabled() {
  return pipeline_internal::env_bool("SIMA_DISPATCHER_TRACE", false);
}

bool stage_tensor_stats_enabled() {
  return pipeline_internal::env_bool("SIMA_STAGE_TENSOR_STATS", false);
}

void stage_trace(const char* label) {
  if (!stage_trace_enabled())
    return;
  std::fprintf(stderr, "[TRACE] %s\n", label);
}

void log_stage_tensor_stats(const char* label, const Sample& sample) {
  if (!stage_tensor_stats_enabled() || !sample_has_tensor_list(sample)) {
    return;
  }
  const TensorList& tensors =
      sample_tensor_list(const_cast<Sample&>(sample), label ? label : "StageRun::tensor_stats");
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const Tensor& tensor = tensors[i];
    std::vector<std::uint8_t> bytes;
    try {
      bytes = tensor.copy_payload_bytes();
    } catch (const std::exception& e) {
      std::fprintf(
          stderr,
          "[stage][tensor-stats] %s tensor=%zu logical=%d segment=%s bytes=<error> detail=%s\n",
          label ? label : "StageRun", i, tensor.route.logical_index,
          tensor.route.segment_name.empty() ? "<empty>" : tensor.route.segment_name.c_str(),
          e.what());
      continue;
    }

    std::size_t zero_count = 0U;
    std::uint64_t hash = 1469598103934665603ULL;
    for (const auto byte : bytes) {
      if (byte == 0U) {
        ++zero_count;
      }
      hash ^= static_cast<std::uint64_t>(byte);
      hash *= 1099511628211ULL;
    }
    std::ostringstream samples;
    samples << "[";
    const std::size_t sample_count = std::min<std::size_t>(8U, bytes.size());
    for (std::size_t bi = 0; bi < sample_count; ++bi) {
      if (bi > 0U) {
        samples << ",";
      }
      samples << static_cast<unsigned int>(bytes[bi]);
    }
    samples << "]";

    const double zero_ratio =
        bytes.empty() ? 1.0 : static_cast<double>(zero_count) / static_cast<double>(bytes.size());
    std::fprintf(stderr,
                 "[stage][tensor-stats] %s tensor=%zu logical=%d mem=%d segment=%s offset=%lld "
                 "bytes=%zu zero_ratio=%.6f hash=0x%016llx samples=%s\n",
                 label ? label : "StageRun", i, tensor.route.logical_index,
                 tensor.route.memory_index,
                 tensor.route.segment_name.empty() ? "<empty>" : tensor.route.segment_name.c_str(),
                 static_cast<long long>(tensor.byte_offset), bytes.size(), zero_ratio,
                 static_cast<unsigned long long>(hash), samples.str().c_str());
  }
}

struct StageKey {
  StageKind kind = StageKind::Preproc;
  std::string model_id;
  StageInputKey input;
  BoxDecodeOptions box_opt{BoxDecodeType::Unspecified};
};

struct WireCaps {
  std::string media_type;
  std::string format;
  TensorDims dims;
  std::string buffer_name;
  std::string caps_override;
};

struct WireInput {
  simaai::neat::Tensor tensor;
  InputOptions appsrc;
  WireCaps caps;
};

struct StagePreprocessMetaRequirement {
  std::string stage_name;
  std::string plugin_name;
  std::vector<std::string> required_fields;
  std::optional<bool> expect_resize;
  std::optional<bool> expect_normalize;
  std::optional<bool> expect_quantize;
  std::optional<bool> expect_tessellate;
};

InputOptions appsrc_for_tensor_wire(const simaai::neat::Tensor& input, const WireCaps& wire);
bool stage_debug_enabled();
const char* dtype_name(TensorDType dtype);
std::string shape_string(const std::vector<int64_t>& shape);
int64_t tensor_total_bytes(const simaai::neat::Tensor& tensor);
TensorDims contract_tensor_dims_projection_from_shape(std::vector<int64_t> shape,
                                                      TensorLayout layout);
namespace rendered_stage_query = simaai::neat::pipeline_internal::rendered_stage_query;

PreprocOutputInfo stage_preproc_output_info(const std::vector<std::shared_ptr<Node>>& group) {
  const PreprocOutputInfo info = rendered_stage_query::preproc_output_info_from_nodes(group);
  if (info.primary_output_name.empty()) {
    throw std::runtime_error("StageRun: missing preproc output contract in rendered manifest");
  }
  return info;
}

std::string stage_primary_input_buffer_name(const std::vector<std::shared_ptr<Node>>& group) {
  const std::string buffer_name = rendered_stage_query::primary_input_buffer_name(group);
  if (buffer_name.empty()) {
    throw std::runtime_error("StageRun: missing primary input buffer name in rendered manifest");
  }
  return buffer_name;
}

std::vector<MlaOutputTensorInfo>
stage_mla_output_tensors_info(const std::vector<std::shared_ptr<Node>>& group) {
  return rendered_stage_query::mla_output_tensors_from_nodes(group);
}

MlaInputTensorInfo stage_mla_input_tensor_info(const std::vector<std::shared_ptr<Node>>& group) {
  MlaInputTensorInfo info = rendered_stage_query::mla_input_tensor_info_from_nodes(group);
  if (info.logical_shape.empty()) {
    throw std::runtime_error("StageRun: missing MLA input contract in rendered manifest");
  }
  return info;
}

std::string stage_meta_error_message(const StagePreprocessMetaRequirement& req,
                                     const std::string& detail) {
  std::ostringstream oss;
  oss << "StageRun: stage='" << req.stage_name << "' plugin='" << req.plugin_name
      << "' preprocess metadata contract violation: " << detail << " (no fallback allowed)";
  return oss.str();
}

std::optional<StagePreprocessMetaRequirement>
collect_preprocess_meta_requirement(const std::vector<std::shared_ptr<Node>>& group,
                                    const std::string& default_stage_name) {
  StagePreprocessMetaRequirement out;
  bool found = false;
  for (const auto& node : group) {
    if (!node)
      continue;
    const auto* provider = dynamic_cast<const PreprocessMetaRequirementProvider*>(node.get());
    if (!provider)
      continue;
    const auto req = provider->preprocess_meta_requirement();
    if (!req.has_value() || req->required_fields.empty())
      continue;
    if (!found) {
      out.stage_name = req->stage_name.empty() ? default_stage_name : req->stage_name;
      out.plugin_name = req->plugin_name.empty() ? node->kind() : req->plugin_name;
      out.expect_resize = req->expect_resize;
      out.expect_normalize = req->expect_normalize;
      out.expect_quantize = req->expect_quantize;
      out.expect_tessellate = req->expect_tessellate;
      found = true;
    } else {
      if (!out.expect_resize.has_value() && req->expect_resize.has_value()) {
        out.expect_resize = req->expect_resize;
      }
      if (!out.expect_normalize.has_value() && req->expect_normalize.has_value()) {
        out.expect_normalize = req->expect_normalize;
      }
      if (!out.expect_quantize.has_value() && req->expect_quantize.has_value()) {
        out.expect_quantize = req->expect_quantize;
      }
      if (!out.expect_tessellate.has_value() && req->expect_tessellate.has_value()) {
        out.expect_tessellate = req->expect_tessellate;
      }
    }
    for (const auto& field : req->required_fields) {
      if (field.empty())
        continue;
      if (std::find(out.required_fields.begin(), out.required_fields.end(), field) ==
          out.required_fields.end()) {
        out.required_fields.push_back(field);
      }
    }
  }
  if (!found || out.required_fields.empty()) {
    return std::nullopt;
  }
  return out;
}

PreprocessRuntimeMeta enforce_required_preprocess_meta(const simaai::neat::Tensor& input,
                                                       const StagePreprocessMetaRequirement& req) {
  const std::shared_ptr<void> holder = pipeline_internal::holder_from_tensor(input);
  if (!holder) {
    throw std::runtime_error(
        stage_meta_error_message(req, "missing tensor holder for required preprocess metadata"));
  }
  GstBuffer* in_buf = pipeline_internal::buffer_from_tensor_holder(holder);
  if (!in_buf) {
    throw std::runtime_error(
        stage_meta_error_message(req, "missing GstBuffer for required preprocess metadata"));
  }
  PreprocessRuntimeMeta meta{};
  const auto validation_error =
      validate_simaai_preprocess_meta_required_fields(in_buf, req.required_fields, &meta);
  gst_buffer_unref(in_buf);
  if (validation_error.has_value()) {
    throw std::runtime_error(stage_meta_error_message(req, *validation_error));
  }
  auto fail_mismatch = [&](const char* field, const char* op) {
    std::ostringstream oss;
    oss << "invalid preprocess metadata field '" << field << "': expected op '" << op
        << "' enabled=true but observed false";
    throw std::runtime_error(stage_meta_error_message(req, oss.str()));
  };
  if (req.expect_resize.has_value() && *req.expect_resize) {
    const bool resize_applied = !(meta.resize_mode.empty() || meta.resize_mode == "none");
    if (!resize_applied) {
      fail_mismatch("preproc_resize_mode", "resize");
    }
  }
  if (req.expect_normalize.has_value() && *req.expect_normalize && !meta.normalize) {
    fail_mismatch("preproc_normalize", "normalize");
  }
  if (req.expect_quantize.has_value() && *req.expect_quantize && !meta.quantize) {
    fail_mismatch("preproc_quantize", "quantize");
  }
  if (req.expect_tessellate.has_value() && *req.expect_tessellate && !meta.tessellate) {
    fail_mismatch("preproc_tessellate", "tessellate");
  }
  return meta;
}

bool stage_debug_enabled() {
  return pipeline_internal::env_bool("SIMA_STAGE_DEBUG", false);
}

bool shadow_change_env_enabled() {
  static int enabled = -1;
  if (enabled >= 0) {
    return enabled != 0;
  }
  const char* v = std::getenv("SHADOW_CHANGE");
  if (!v || !*v) {
    enabled = 0;
    return false;
  }
  if (!std::strcmp(v, "1") || !std::strcmp(v, "true") || !std::strcmp(v, "TRUE") ||
      !std::strcmp(v, "yes") || !std::strcmp(v, "YES") || !std::strcmp(v, "on") ||
      !std::strcmp(v, "ON")) {
    enabled = 1;
    return true;
  }
  enabled = 0;
  return false;
}

GstBuffer* tensor_holder_buffer(const simaai::neat::Tensor& tensor) {
  const std::shared_ptr<void> holder = pipeline_internal::holder_from_tensor(tensor);
  if (!holder) {
    return nullptr;
  }
  return pipeline_internal::buffer_from_tensor_holder(holder);
}

void propagate_preprocess_meta_to_tensor_if_missing(const GstBuffer* source_buf,
                                                    const PreprocessRuntimeMeta* source_meta,
                                                    simaai::neat::Tensor* tensor) {
  if (!tensor) {
    return;
  }
  std::optional<PreprocessRuntimeMeta> owned_meta;
  if (!source_meta && source_buf) {
    owned_meta = read_simaai_preprocess_meta(const_cast<GstBuffer*>(source_buf));
    if (owned_meta.has_value()) {
      source_meta = &owned_meta.value();
    }
  }
  if (!source_meta) {
    return;
  }
  if (!tensor->semantic.preprocess.has_value()) {
    tensor->semantic.preprocess = *source_meta;
  }
  GstBuffer* dst_buf = tensor_holder_buffer(*tensor);
  if (!dst_buf) {
    return;
  }
  if (has_simaai_preprocess_meta(dst_buf)) {
    gst_buffer_unref(dst_buf);
    return;
  }
  if (source_buf && has_simaai_preprocess_meta(const_cast<GstBuffer*>(source_buf))) {
    std::string copy_err;
    if (!copy_simaai_preprocess_meta(dst_buf, const_cast<GstBuffer*>(source_buf), &copy_err) &&
        stage_debug_enabled()) {
      std::fprintf(stderr, "[stage][meta] failed to propagate preprocess meta: %s\n",
                   copy_err.empty() ? "<unknown>" : copy_err.c_str());
    }
  } else if (!write_simaai_preprocess_meta(dst_buf, *source_meta) && stage_debug_enabled()) {
    std::fprintf(stderr,
                 "[stage][meta] failed to write preprocess meta from tensor semantic state\n");
  }
  gst_buffer_unref(dst_buf);
}

namespace {

void propagate_preprocess_meta_to_sample_tree_if_missing(const GstBuffer* source_buf,
                                                         const PreprocessRuntimeMeta* source_meta,
                                                         Sample* sample) {
  if (!sample) {
    return;
  }
  if (sample->kind == SampleKind::Tensor && sample->tensor.has_value()) {
    propagate_preprocess_meta_to_tensor_if_missing(source_buf, source_meta,
                                                   &sample->tensor.value());
    return;
  }
  if (sample->kind == SampleKind::TensorSet) {
    for (auto& tensor : sample->tensors) {
      propagate_preprocess_meta_to_tensor_if_missing(source_buf, source_meta, &tensor);
    }
    return;
  }
  if (sample->kind == SampleKind::Bundle) {
    for (auto& field : sample->fields) {
      propagate_preprocess_meta_to_sample_tree_if_missing(source_buf, source_meta, &field);
    }
  }
}

} // namespace

void propagate_preprocess_meta_to_sample_if_missing(const simaai::neat::Tensor& source,
                                                    Sample* sample) {
  if (!sample) {
    return;
  }
  std::optional<PreprocessRuntimeMeta> source_meta_from_buffer;
  const PreprocessRuntimeMeta* source_meta =
      source.semantic.preprocess.has_value() ? &source.semantic.preprocess.value() : nullptr;
  GstBuffer* source_buf = tensor_holder_buffer(source);
  if (!source_buf && !source_meta) {
    return;
  }
  if (source_buf && !source_meta) {
    source_meta_from_buffer = read_simaai_preprocess_meta(source_buf);
    if (source_meta_from_buffer.has_value()) {
      source_meta = &source_meta_from_buffer.value();
    }
  }
  if (!source_meta) {
    if (source_buf) {
      gst_buffer_unref(source_buf);
    }
    return;
  }
  if (source_buf && !has_simaai_preprocess_meta(source_buf) &&
      !source.semantic.preprocess.has_value()) {
    gst_buffer_unref(source_buf);
    return;
  }
  propagate_preprocess_meta_to_sample_tree_if_missing(source_buf, source_meta, sample);
  if (source_buf) {
    gst_buffer_unref(source_buf);
  }
}

bool sample_requires_message_path(const Sample& sample) {
  if (sample.kind == SampleKind::Bundle) {
    return true;
  }
  if (!sample_has_tensor_list(sample)) {
    return sample.payload_type != PayloadType::Auto || !sample.media_type.empty() ||
           !sample.format.empty() || !sample.payload_tag.empty() || !sample.caps_string.empty() ||
           !sample.port_name.empty() || !sample.segment_name.empty() || sample.memory_index >= 0 ||
           sample.route_slot >= 0 || sample.logical_output_index >= 0 || sample.frame_id >= 0 ||
           sample.pts_ns >= 0 || sample.dts_ns >= 0 || sample.duration_ns >= 0;
  }
  if (sample.tensors.size() != 1U || !sample.fields.empty()) {
    return true;
  }
  return !sample.caps_string.empty() || !sample.port_name.empty() || sample.frame_id >= 0 ||
         !sample.stream_id.empty() || !sample.stream_label.empty() || sample.input_seq >= 0 ||
         sample.orig_input_seq >= 0 || sample.pts_ns >= 0 || sample.dts_ns >= 0 ||
         sample.duration_ns >= 0;
}

const char* sample_kind_name(SampleKind kind) {
  switch (kind) {
  case SampleKind::Tensor:
    return "Tensor";
  case SampleKind::TensorSet:
    return "TensorSet";
  case SampleKind::Bundle:
    return "Bundle";
  case SampleKind::Unknown:
    return "Unknown";
  }
  return "Unknown";
}

const char* dtype_name(TensorDType dtype) {
  switch (dtype) {
  case TensorDType::UInt8:
    return "UInt8";
  case TensorDType::Int8:
    return "Int8";
  case TensorDType::UInt16:
    return "UInt16";
  case TensorDType::Int16:
    return "Int16";
  case TensorDType::Int32:
    return "Int32";
  case TensorDType::BFloat16:
    return "BFloat16";
  case TensorDType::Float32:
    return "Float32";
  case TensorDType::Float64:
    return "Float64";
  }
  return "Unknown";
}

enum class DTypeFamily {
  Unknown = 0,
  Int8,
  Int16,
  Int32,
  BFloat16,
  Float32,
  Float64,
};

const char* dtype_family_name(DTypeFamily family) {
  switch (family) {
  case DTypeFamily::Int8:
    return "INT8";
  case DTypeFamily::Int16:
    return "INT16";
  case DTypeFamily::Int32:
    return "INT32";
  case DTypeFamily::BFloat16:
    return "BF16";
  case DTypeFamily::Float32:
    return "FP32";
  case DTypeFamily::Float64:
    return "FP64";
  case DTypeFamily::Unknown:
    break;
  }
  return "UNKNOWN";
}

DTypeFamily dtype_family_from_tensor_dtype(TensorDType dtype) {
  switch (dtype) {
  case TensorDType::UInt8:
  case TensorDType::Int8:
    return DTypeFamily::Int8;
  case TensorDType::UInt16:
  case TensorDType::Int16:
    return DTypeFamily::Int16;
  case TensorDType::Int32:
    return DTypeFamily::Int32;
  case TensorDType::BFloat16:
    return DTypeFamily::BFloat16;
  case TensorDType::Float32:
    return DTypeFamily::Float32;
  case TensorDType::Float64:
    return DTypeFamily::Float64;
  }
  return DTypeFamily::Unknown;
}

DTypeFamily dtype_family_from_token(std::string token) {
  token = upper_copy(std::move(token));
  if (token.empty()) {
    return DTypeFamily::Unknown;
  }
  if (token.find("BF16") != std::string::npos || token.find("BFLOAT16") != std::string::npos) {
    return DTypeFamily::BFloat16;
  }
  if (token.find("INT8") != std::string::npos || token.find("UINT8") != std::string::npos) {
    return DTypeFamily::Int8;
  }
  if (token.find("INT16") != std::string::npos || token.find("UINT16") != std::string::npos) {
    return DTypeFamily::Int16;
  }
  if (token.find("INT32") != std::string::npos) {
    return DTypeFamily::Int32;
  }
  if (token.find("FP64") != std::string::npos || token.find("FLOAT64") != std::string::npos) {
    return DTypeFamily::Float64;
  }
  if (token.find("FP32") != std::string::npos || token.find("FLOAT32") != std::string::npos) {
    return DTypeFamily::Float32;
  }
  return DTypeFamily::Unknown;
}

std::string shape_string(const std::vector<int64_t>& shape) {
  if (shape.empty())
    return "";
  std::string out;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0)
      out.push_back('x');
    out += std::to_string(shape[i]);
  }
  return out;
}

void log_stage_tensor_holder_state(const char* label, const simaai::neat::Tensor& tensor) {
  if (!stage_debug_enabled())
    return;
  const auto* storage = tensor.storage.get();
  const void* holder_ptr = (storage && storage->holder) ? storage->holder.get() : nullptr;
  std::fprintf(stderr,
               "[stage][holder] %s storage=%p kind=%d holder=%p device=%d:%d read_only=%d "
               "planes=%zu shape=%s\n",
               label ? label : "tensor", static_cast<const void*>(storage),
               storage ? static_cast<int>(storage->kind) : -1, holder_ptr,
               static_cast<int>(tensor.device.type), tensor.device.id, tensor.read_only ? 1 : 0,
               tensor.planes.size(), shape_string(tensor.shape).c_str());
}

void log_stage_holder_buffer_memories(const char* label, const simaai::neat::Tensor& tensor) {
  if (!stage_debug_enabled()) {
    return;
  }
  const std::shared_ptr<void> holder = pipeline_internal::holder_from_tensor(tensor);
  if (!holder) {
    std::fprintf(stderr, "[stage][holder] %s holder=<none>\n", label ? label : "tensor");
    return;
  }
  GstBuffer* buf = pipeline_internal::buffer_from_tensor_holder(holder);
  if (!buf) {
    std::fprintf(stderr, "[stage][holder] %s holder present but buffer_from_tensor_holder failed\n",
                 label ? label : "tensor");
    return;
  }

  const guint n_mems = gst_buffer_n_memory(buf);
  std::fprintf(stderr, "[stage][holder] %s holder_buffer=%p memories=%u\n",
               label ? label : "tensor", static_cast<void*>(buf), static_cast<unsigned>(n_mems));
  for (guint i = 0; i < n_mems; ++i) {
    GstMemory* mem = gst_buffer_peek_memory(buf, i);
    gsize offset = 0;
    gsize maxsize = 0;
    const gsize size = mem ? gst_memory_get_sizes(mem, &offset, &maxsize) : 0;
    const char* allocator_name =
        (mem && mem->allocator && mem->allocator->mem_type) ? mem->allocator->mem_type : "<null>";
    std::fprintf(stderr, "[stage][holder]   mem[%u] allocator=%s size=%zu offset=%zu max=%zu\n",
                 static_cast<unsigned>(i), allocator_name, static_cast<size_t>(size),
                 static_cast<size_t>(offset), static_cast<size_t>(maxsize));
  }
  gst_buffer_unref(buf);
}

void log_stage_group_nodes(const char* stage_name,
                           const std::vector<std::shared_ptr<Node>>& group) {
  if (!stage_debug_enabled())
    return;
  std::fprintf(stderr, "[stage][group] %s nodes=%zu\n", stage_name ? stage_name : "unknown",
               group.size());
  size_t index = 0;
  for (const auto& node : group) {
    const std::string kind = node ? node->kind() : std::string("<null>");
    std::fprintf(stderr, "[stage][group] %s[%zu]=%s\n", stage_name ? stage_name : "unknown", index,
                 kind.c_str());
    ++index;
  }
}

bool tensor_is_gst_sample_backed(const simaai::neat::Tensor& tensor) {
  return tensor.storage && tensor.storage->kind == simaai::neat::StorageKind::GstSample &&
         static_cast<bool>(tensor.storage->holder);
}

const simaai::neat::Tensor* find_gst_sample_backed_tensor_for_memory_view(const Sample& sample,
                                                                          int memory_index) {
  if (!sample_has_tensor_list(sample) || sample.tensors.empty()) {
    return nullptr;
  }

  const simaai::neat::Tensor* first_backed = nullptr;
  for (const auto& tensor : sample.tensors) {
    if (!tensor.planes.empty() || !tensor_is_gst_sample_backed(tensor)) {
      continue;
    }
    if (!first_backed) {
      first_backed = &tensor;
    }
    if (memory_index >= 0 && (tensor.route.memory_index == memory_index ||
                              tensor.route.physical_index == memory_index)) {
      return &tensor;
    }
  }
  return first_backed;
}

void log_stage_output_sample(const char* stage_name, const Sample& sample) {
  if (!stage_debug_enabled())
    return;
  std::fprintf(stderr,
               "[stage][sample] %s kind=%s payload_type=%d media=%s format=%s payload=%s caps=%s "
               "owned=%d tensors=%zu fields=%zu output_index=%d logical_output_index=%d "
               "memory_index=%d route_slot=%d segment=%s\n",
               stage_name ? stage_name : "unknown", sample_kind_name(sample.kind),
               static_cast<int>(sample.payload_type), sample.media_type.c_str(),
               sample.format.c_str(), sample.payload_tag.c_str(), sample.caps_string.c_str(),
               sample.owned ? 1 : 0, sample.tensors.size(), sample.fields.size(),
               sample.output_index, sample.logical_output_index, sample.memory_index,
               sample.route_slot,
               sample.segment_name.empty() ? "<empty>" : sample.segment_name.c_str());
  for (std::size_t i = 0; i < sample.tensors.size(); ++i) {
    log_stage_tensor_holder_state(("sample.tensors[" + std::to_string(i) + "]").c_str(),
                                  sample.tensors[i]);
  }
}

int dtype_bytes(TensorDType dtype);
int tensor_depth_from_shape(const std::vector<int64_t>& shape);

std::string image_format_name(simaai::neat::ImageSpec::PixelFormat fmt) {
  switch (fmt) {
  case simaai::neat::ImageSpec::PixelFormat::RGB:
    return "RGB";
  case simaai::neat::ImageSpec::PixelFormat::BGR:
    return "BGR";
  case simaai::neat::ImageSpec::PixelFormat::GRAY8:
    return "GRAY8";
  case simaai::neat::ImageSpec::PixelFormat::NV12:
    return "NV12";
  case simaai::neat::ImageSpec::PixelFormat::I420:
    return "I420";
  case simaai::neat::ImageSpec::PixelFormat::UNKNOWN:
    break;
  }
  return "";
}

simaai::neat::ImageSpec::PixelFormat image_format_from_string(const std::string& fmt) {
  const std::string up = upper_copy(fmt);
  if (up == "RGB")
    return simaai::neat::ImageSpec::PixelFormat::RGB;
  if (up == "BGR")
    return simaai::neat::ImageSpec::PixelFormat::BGR;
  if (up == "GRAY8" || up == "GRAY")
    return simaai::neat::ImageSpec::PixelFormat::GRAY8;
  if (up == "NV12")
    return simaai::neat::ImageSpec::PixelFormat::NV12;
  if (up == "I420" || up == "YUV420")
    return simaai::neat::ImageSpec::PixelFormat::I420;
  return simaai::neat::ImageSpec::PixelFormat::UNKNOWN;
}

simaai::neat::FormatSpec preprocess_color_format_to_format_spec(PreprocessColorFormat fmt) {
  switch (fmt) {
  case PreprocessColorFormat::RGB:
    return FormatSpec{FormatTag::RGB};
  case PreprocessColorFormat::BGR:
    return FormatSpec{FormatTag::BGR};
  case PreprocessColorFormat::GRAY8:
    return FormatSpec{FormatTag::GRAY8};
  case PreprocessColorFormat::NV12:
    return FormatSpec{FormatTag::NV12};
  case PreprocessColorFormat::I420:
    return FormatSpec{FormatTag::I420};
  case PreprocessColorFormat::Auto:
  default:
    return FormatSpec{};
  }
}

std::string format_from_tensor(const simaai::neat::Tensor& tensor) {
  if (tensor.semantic.byte_stream.has_value()) {
    return format_tag_to_string(FormatTag::ByteStream);
  }
  if (tensor.semantic.tess.has_value()) {
    return upper_copy(tensor.semantic.tess->format);
  }
  if (tensor.semantic.image.has_value()) {
    const std::string fmt = image_format_name(tensor.semantic.image->format);
    if (!fmt.empty())
      return fmt;
  }
  return "";
}

simaai::neat::Tensor require_supported_tessellated_dtype(simaai::neat::Tensor tensor,
                                                         const char* where) {
  const std::string prefix = (where && *where) ? (std::string(where) + ": ") : "";
  if (!tensor.storage) {
    throw std::runtime_error(prefix + "tessellated tensor: missing tensor storage");
  }

  const std::string fmt = format_from_tensor(tensor);
  const bool fmt_int8 = is_tessellated_int8_format(fmt);
  const bool fmt_bf16 = is_tessellated_bf16_format(fmt);
  const bool fmt_int16 = (!fmt.empty() && (fmt.find("INT16") != std::string::npos ||
                                           fmt.find("EVXX_INT16") != std::string::npos));
  const bool dtype_int8 = (tensor.dtype == TensorDType::Int8 || tensor.dtype == TensorDType::UInt8);
  const bool dtype_bf16 = (tensor.dtype == TensorDType::BFloat16);
  const bool dtype_int16 =
      (tensor.dtype == TensorDType::Int16 || tensor.dtype == TensorDType::UInt16);

  if (!(fmt_int8 || fmt_bf16 || fmt_int16 || dtype_int8 || dtype_bf16 || dtype_int16)) {
    throw std::runtime_error(prefix + "tessellated tensor: unsupported dtype/format: dtype=" +
                             std::string(dtype_name(tensor.dtype)) + " format=" + fmt);
  }

  if (fmt_int8 && tensor.dtype == TensorDType::UInt8) {
    tensor.dtype = TensorDType::Int8;
  }
  if (fmt_int16 && tensor.dtype == TensorDType::UInt16) {
    tensor.dtype = TensorDType::Int16;
  }

  if (!fmt.empty() && !tensor.semantic.tess.has_value()) {
    simaai::neat::TessSpec tess;
    tess.format = fmt;
    tensor.semantic.tess = tess;
  } else if (!fmt.empty()) {
    tensor.semantic.tess->format = fmt;
  }
  return tensor;
}

int64_t tensor_dense_bytes_tight(const simaai::neat::Tensor& tensor) {
  if (tensor.shape.empty())
    return 0;
  size_t total = dtype_bytes(tensor.dtype);
  for (const auto dim : tensor.shape) {
    if (dim <= 0)
      return 0;
    total *= static_cast<size_t>(dim);
  }
  return static_cast<int64_t>(total);
}

int64_t tensor_plane_bytes_tight(const simaai::neat::Plane& plane, TensorDType dtype) {
  if (plane.shape.size() < 2)
    return 0;
  const int64_t h = plane.shape[0];
  const int64_t w = plane.shape[1];
  if (h <= 0 || w <= 0)
    return 0;
  return static_cast<int64_t>(dtype_bytes(dtype)) * h * w;
}

int64_t tensor_total_bytes(const simaai::neat::Tensor& tensor) {
  if (tensor.is_composite()) {
    int64_t total = 0;
    for (const auto& plane : tensor.planes) {
      total += tensor_plane_bytes_tight(plane, tensor.dtype);
    }
    return total;
  }
  return tensor_dense_bytes_tight(tensor);
}

std::string tensor_primary_segment_name(const simaai::neat::Tensor& tensor) {
  if (!tensor.route.segment_name.empty()) {
    return tensor.route.segment_name;
  }
  if (tensor.storage && !tensor.storage->sima_segments.empty()) {
    int memory_index = tensor.route.memory_index;
    if (memory_index < 0) {
      memory_index = tensor.route.physical_index;
    }
    std::size_t segment_index = 0U;
    if (memory_index >= 0 &&
        static_cast<std::size_t>(memory_index) < tensor.storage->sima_segments.size()) {
      segment_index = static_cast<std::size_t>(memory_index);
    }
    if (!tensor.storage->sima_segments[segment_index].name.empty()) {
      return tensor.storage->sima_segments[segment_index].name;
    }
  }
  return {};
}

std::string sample_primary_segment_name(const Sample& sample) {
  if (sample_has_tensor_list(sample) && !sample.tensors.empty()) {
    const std::string tensor_name = tensor_primary_segment_name(sample.tensors.front());
    if (!tensor_name.empty()) {
      return tensor_name;
    }
  }
  if (!sample.segment_name.empty()) {
    return sample.segment_name;
  }
  if (sample.kind == SampleKind::Bundle && !sample.fields.empty()) {
    return sample_primary_segment_name(sample.fields.front());
  }
  return {};
}

void apply_stage_source_segment_name(WireCaps* wire, const Sample& sample) {
  if (!wire) {
    return;
  }
  const std::string segment_name = sample_primary_segment_name(sample);
  if (!segment_name.empty()) {
    wire->buffer_name = segment_name;
  }
}

int64_t tensor_primary_segment_size(const simaai::neat::Tensor& tensor) {
  if (!tensor.storage || tensor.storage->sima_segments.empty()) {
    return 0;
  }
  if (!tensor.route.segment_name.empty()) {
    for (const auto& segment : tensor.storage->sima_segments) {
      if (segment.name == tensor.route.segment_name) {
        return static_cast<int64_t>(segment.size_bytes);
      }
    }
  }
  int memory_index = tensor.route.memory_index;
  if (memory_index < 0) {
    memory_index = tensor.route.physical_index;
  }
  std::size_t segment_index = 0U;
  if (memory_index >= 0 &&
      static_cast<std::size_t>(memory_index) < tensor.storage->sima_segments.size()) {
    segment_index = static_cast<std::size_t>(memory_index);
  }
  return static_cast<int64_t>(tensor.storage->sima_segments[segment_index].size_bytes);
}

int64_t tensor_sample_memory_size(const simaai::neat::Tensor& tensor, int memory_index) {
  const std::shared_ptr<void> holder = pipeline_internal::holder_from_tensor(tensor);
  if (!holder) {
    return 0;
  }
  GstBuffer* buffer = pipeline_internal::buffer_from_tensor_holder(holder);
  if (!buffer) {
    return 0;
  }
  const guint n_mems = gst_buffer_n_memory(buffer);
  if (n_mems == 0U) {
    gst_buffer_unref(buffer);
    return 0;
  }
  guint index = 0U;
  if (memory_index >= 0 && static_cast<guint>(memory_index) < n_mems) {
    index = static_cast<guint>(memory_index);
  }
  GstMemory* mem = gst_buffer_peek_memory(buffer, index);
  gsize offset = 0;
  gsize maxsize = 0;
  const gsize size = mem ? gst_memory_get_sizes(mem, &offset, &maxsize) : 0;
  gst_buffer_unref(buffer);
  return static_cast<int64_t>(size);
}

TensorDims dims_from_tensor(const simaai::neat::Tensor& tensor) {
  TensorDims dims;
  int h = (tensor.shape.size() > 0) ? static_cast<int>(tensor.shape[0]) : -1;
  int w = (tensor.shape.size() > 1) ? static_cast<int>(tensor.shape[1]) : -1;
  int d = (tensor.shape.size() > 2) ? static_cast<int>(tensor.shape[2]) : -1;
  if (tensor.is_composite() && !tensor.planes.empty()) {
    const auto& y = tensor.planes.front();
    if (y.shape.size() >= 2) {
      h = (h > 0) ? h : static_cast<int>(y.shape[0]);
      w = (w > 0) ? w : static_cast<int>(y.shape[1]);
    }
  }
  dims.width = w;
  dims.height = h;
  if (d > 0)
    dims.depth = d;
  return dims;
}

int dtype_bytes(TensorDType dtype) {
  switch (dtype) {
  case TensorDType::Int8:
  case TensorDType::UInt8:
    return 1;
  case TensorDType::UInt16:
  case TensorDType::Int16:
  case TensorDType::BFloat16:
    return 2;
  case TensorDType::Int32:
  case TensorDType::Float32:
    return 4;
  case TensorDType::Float64:
    return 8;
  }
  return 1;
}

void apply_tensor_dims(simaai::neat::Tensor& tensor, const TensorDims& dims) {
  if (dims.width <= 0 || dims.height <= 0 || dims.depth <= 0)
    return;
  tensor.shape = {dims.height, dims.width, dims.depth};
  const int64_t elem = dtype_bytes(tensor.dtype);
  tensor.strides_bytes = {dims.width * dims.depth * elem, dims.depth * elem, elem};
}

void apply_tensor_hw(simaai::neat::Tensor& tensor, const TensorDims& dims) {
  if (dims.width > 0 && dims.height > 0 && dims.depth > 0) {
    tensor.shape = {dims.height, dims.width, dims.depth};
    const int64_t elem = dtype_bytes(tensor.dtype);
    tensor.strides_bytes = {dims.width * dims.depth * elem, dims.depth * elem, elem};
  } else if (dims.width > 0 && dims.height > 0) {
    tensor.shape = {dims.height, dims.width};
    const int64_t elem = dtype_bytes(tensor.dtype);
    tensor.strides_bytes = {dims.width * elem, elem};
  }
}

std::optional<TensorDType> tensor_dtype_from_declared_format(const std::string& fmt) {
  if (fmt.empty()) {
    return std::nullopt;
  }
  const std::string up = upper_copy(fmt);
  auto contains = [&](const char* token) {
    return token && *token && up.find(token) != std::string::npos;
  };

  if (contains("BFLOAT16") || contains("BF16")) {
    return TensorDType::BFloat16;
  }
  if (contains("UINT16")) {
    return TensorDType::UInt16;
  }
  if (contains("INT16")) {
    return TensorDType::Int16;
  }
  if (contains("UINT8")) {
    return TensorDType::UInt8;
  }
  if (contains("INT8")) {
    return TensorDType::Int8;
  }
  if (contains("FLOAT64") || contains("FP64")) {
    return TensorDType::Float64;
  }
  if (contains("FLOAT32") || contains("FP32")) {
    return TensorDType::Float32;
  }
  if (contains("INT32")) {
    return TensorDType::Int32;
  }
  return std::nullopt;
}

void apply_tensor_dtype_from_format(simaai::neat::Tensor& tensor, const std::string& fmt) {
  if (const auto parsed = tensor_dtype_from_declared_format(fmt); parsed.has_value()) {
    tensor.dtype = *parsed;
  }
  if (!fmt.empty()) {
    if (!tensor.semantic.tess.has_value()) {
      simaai::neat::TessSpec tess;
      tess.format = fmt;
      tensor.semantic.tess = tess;
    } else {
      tensor.semantic.tess->format = fmt;
    }
  }
}

void apply_preproc_output_override(simaai::neat::Tensor& tensor, const PreprocOutputInfo& info) {
  // JSON config overrides caps. Preproc caps can remain RGB even when tessellated.
  if (info.transport_kind == PreprocOutputTransportKind::Dense) {
    apply_tensor_dims(tensor, info.logical_dims);
    if (info.logical_layout != TensorLayout::Unknown) {
      tensor.layout = info.logical_layout;
    }
  }
  if (info.output_dtype.empty())
    return;
  const std::string fmt = upper_copy(info.output_dtype);
  apply_tensor_dtype_from_format(tensor, fmt);
}

int resolve_preproc_selected_memory_index(const Sample& sample, const PreprocOutputInfo& info) {
  if (!find_gst_sample_backed_tensor_for_memory_view(sample, -1)) {
    return -1;
  }
  pipeline_internal::TensorBufferView view;
  std::string view_err;
  if (!pipeline_internal::tensor_buffer_view_from_sample(sample, &view, &view_err) ||
      !view.buffer) {
    throw std::runtime_error("Preproc: tensor buffer descriptor unavailable: " + view_err);
  }
  const auto it = std::find_if(view.tensors.begin(), view.tensors.end(),
                               [&](const pipeline_internal::TensorBufferTensorDescriptor& tensor) {
                                 return tensor.route_slot == info.primary_route_slot &&
                                        tensor.logical_name == info.primary_output_name &&
                                        tensor.segment_name == info.primary_output_name;
                               });
  if (it == view.tensors.end()) {
    throw std::runtime_error("Preproc: failed to resolve selected output '" +
                             info.primary_output_name + "'");
  }
  const auto* tensor_view = &(*it);
  if (!tensor_view) {
    throw std::runtime_error("Preproc: missing logical tensor for primary output '" +
                             info.primary_output_name + "'");
  }
  if (tensor_view->memory_index >= 0) {
    return tensor_view->memory_index;
  }
  return tensor_view->physical_index;
}

TensorDims mla_output_dims_from_shape(const std::vector<int64_t>& shape) {
  TensorDims dims;
  if (shape.size() >= 3U) {
    dims.height = static_cast<int>(shape[shape.size() - 3U]);
    dims.width = static_cast<int>(shape[shape.size() - 2U]);
    dims.depth = static_cast<int>(shape[shape.size() - 1U]);
  } else if (shape.size() == 2U) {
    dims.height = static_cast<int>(shape[0]);
    dims.width = static_cast<int>(shape[1]);
    dims.depth = 1;
  } else if (shape.size() == 1U) {
    dims.width = static_cast<int>(shape[0]);
    dims.height = 1;
    dims.depth = 1;
  }
  return dims;
}

MlaOutputInfo mla_output_info_from_contract_tensor(const MlaOutputTensorInfo& contract) {
  MlaOutputInfo info;
  info.data_type = contract.data_type;
  info.logical_data_type = contract.data_type;
  info.logical_shape = contract.shape;
  info.output_format = contract.output_format;
  info.layout = contract.layout;
  info.size_bytes = contract.size_bytes;
  info.dims = mla_output_dims_from_shape(contract.shape);
  return info;
}

int64_t mla_logical_bytes_from_dims(const MlaOutputInfo& info) {
  if (info.dims.width <= 0 || info.dims.height <= 0 || info.dims.depth <= 0) {
    return 0;
  }
  const std::string dtype_token =
      !info.logical_data_type.empty() ? info.logical_data_type : info.data_type;
  const auto parsed = tensor_dtype_from_declared_format(upper_copy(dtype_token));
  if (!parsed.has_value()) {
    return 0;
  }
  const int elem = dtype_bytes(*parsed);
  if (elem <= 0) {
    return 0;
  }
  return static_cast<int64_t>(info.dims.width) * static_cast<int64_t>(info.dims.height) *
         static_cast<int64_t>(info.dims.depth) * static_cast<int64_t>(elem);
}

bool mla_info_indicates_packed_envelope(const MlaOutputInfo& info) {
  const int64_t logical_bytes = mla_logical_bytes_from_dims(info);
  return logical_bytes > 0 && info.size_bytes > logical_bytes;
}

void enforce_pre_mla_input_bytes_guard(const simaai::neat::Tensor& selected_input,
                                       const std::vector<std::shared_ptr<Node>>& infer_group,
                                       const simaai::neat::Model& model, const char* stage_name) {
  if (!shadow_change_env_enabled()) {
    return;
  }
  const MlaInputTensorInfo mla_input = stage_mla_input_tensor_info(infer_group);
  const int64_t contract_logical_bytes = mla_input.span_size_bytes;
  const std::string contract_input_dtype = mla_input.logical_dtype;
  if (contract_logical_bytes <= 0 || contract_input_dtype.empty()) {
    throw std::runtime_error("StageRun: missing strict MLA input contract from rendered manifest");
  }

  // Byte-contract semantics:
  // - handle_bytes: physical backing memory bytes (allocator segment / GstMemory holder).
  // - runtime_logical_bytes: bytes implied by selected tensor shape/dtype at runtime.
  // - contract_logical_bytes: strict MLA input bytes from MPK-derived contract.
  int64_t handle_bytes = tensor_primary_segment_size(selected_input);
  std::string selected_tensor = tensor_primary_segment_name(selected_input);
  if (handle_bytes <= 0) {
    handle_bytes = tensor_sample_memory_size(selected_input, 0);
  }
  const int64_t runtime_logical_bytes = tensor_total_bytes(selected_input);
  if (selected_tensor.empty()) {
    selected_tensor = "output_tensor";
  }

  const auto route_flags = simaai::neat::internal::ModelAccess::preprocess_contract_flags(model);
  const std::string model_id = simaai::neat::internal::ModelAccess::model_id(model);
  const DTypeFamily runtime_dtype = dtype_family_from_tensor_dtype(selected_input.dtype);
  const DTypeFamily contract_dtype = dtype_family_from_token(contract_input_dtype);
  const std::shared_ptr<void> holder = pipeline_internal::holder_from_tensor(selected_input);
  GstBuffer* buffer = holder ? pipeline_internal::buffer_from_tensor_holder(holder) : nullptr;
  const guint holder_memories = buffer ? gst_buffer_n_memory(buffer) : 0U;
  if (buffer) {
    gst_buffer_unref(buffer);
  }

  auto fail_guard = [&](const char* code, const char* detail) {
    std::fprintf(stderr,
                 "[stage][pre-mla-guard] code=%s stage=%s component=StageRun detail=%s "
                 "model_id=%s selected_tensor=%s handle_bytes=%lld runtime_logical_bytes=%lld "
                 "contract_logical_bytes=%lld runtime_dtype=%s contract_dtype=%s "
                 "contract_input_dtype=%s holder_memories=%u quant_needed=%d tess_needed=%d\n",
                 code ? code : "PRE_MLA_INPUT_GUARD_FAILED", stage_name ? stage_name : "Infer",
                 detail ? detail : "byte_contract_failure", model_id.c_str(),
                 selected_tensor.c_str(), static_cast<long long>(handle_bytes),
                 static_cast<long long>(runtime_logical_bytes),
                 static_cast<long long>(contract_logical_bytes), dtype_family_name(runtime_dtype),
                 dtype_family_name(contract_dtype), contract_input_dtype.c_str(),
                 static_cast<unsigned>(holder_memories), route_flags.quant_needed ? 1 : 0,
                 route_flags.tess_needed ? 1 : 0);
    std::ostringstream oss;
    oss << "StageRun pre-MLA guard failed: code=" << (code ? code : "PRE_MLA_INPUT_GUARD_FAILED")
        << " stage=" << (stage_name ? stage_name : "Infer") << " model_id=" << model_id
        << " selected_tensor=" << selected_tensor << " handle_bytes=" << handle_bytes
        << " runtime_logical_bytes=" << runtime_logical_bytes
        << " contract_logical_bytes=" << contract_logical_bytes
        << " runtime_dtype=" << dtype_family_name(runtime_dtype)
        << " contract_dtype=" << dtype_family_name(contract_dtype)
        << " contract_input_dtype=" << contract_input_dtype
        << " holder_memories=" << holder_memories
        << " detail=" << (detail ? detail : "byte_contract_failure")
        << " quant_needed=" << (route_flags.quant_needed ? 1 : 0)
        << " tess_needed=" << (route_flags.tess_needed ? 1 : 0);
    throw std::runtime_error(oss.str());
  };

  if (handle_bytes <= 0) {
    fail_guard("PRE_MLA_INPUT_HANDLE_UNKNOWN", "handle_bytes_unavailable");
  }
  if (runtime_logical_bytes <= 0) {
    fail_guard("PRE_MLA_INPUT_RUNTIME_UNKNOWN", "runtime_logical_bytes_unavailable");
  }
  if (contract_dtype != DTypeFamily::Unknown && runtime_dtype != DTypeFamily::Unknown &&
      runtime_dtype != contract_dtype) {
    fail_guard("PRE_MLA_INPUT_DTYPE_MISMATCH", "runtime_dtype_ne_contract_input_dtype");
  }
  if (handle_bytes < runtime_logical_bytes) {
    fail_guard("PRE_MLA_INPUT_HANDLE_TOO_SMALL", "handle_lt_runtime_logical");
  }
  if (runtime_logical_bytes != contract_logical_bytes) {
    fail_guard("PRE_MLA_INPUT_CONTRACT_MISMATCH", "runtime_logical_ne_contract_logical");
  }
  if (stage_debug_enabled()) {
    std::fprintf(stderr,
                 "[stage][pre-mla-guard] code=PRE_MLA_INPUT_SIZE_OK stage=%s component=StageRun "
                 "detail=byte_contract_match model_id=%s selected_tensor=%s handle_bytes=%lld "
                 "runtime_logical_bytes=%lld contract_logical_bytes=%lld runtime_dtype=%s "
                 "contract_dtype=%s contract_input_dtype=%s holder_memories=%u quant_needed=%d "
                 "tess_needed=%d\n",
                 stage_name ? stage_name : "Infer", model_id.c_str(), selected_tensor.c_str(),
                 static_cast<long long>(handle_bytes),
                 static_cast<long long>(runtime_logical_bytes),
                 static_cast<long long>(contract_logical_bytes), dtype_family_name(runtime_dtype),
                 dtype_family_name(contract_dtype), contract_input_dtype.c_str(),
                 static_cast<unsigned>(holder_memories), route_flags.quant_needed ? 1 : 0,
                 route_flags.tess_needed ? 1 : 0);
  }
}

void apply_mla_output_override(simaai::neat::Tensor& tensor, const MlaOutputInfo& info) {
  // JSON config overrides caps. MLA caps can be "MLA" even though dtype is INT8/BF16.
  const std::string dtype_for_tensor =
      !info.logical_data_type.empty() ? info.logical_data_type : info.data_type;
  if (!dtype_for_tensor.empty()) {
    const std::string fmt = upper_copy(dtype_for_tensor);
    apply_tensor_dtype_from_format(tensor, fmt);
  }
  if (!info.logical_shape.empty()) {
    tensor.shape = info.logical_shape;
    const int64_t elem = dtype_bytes(tensor.dtype);
    if (elem > 0) {
      tensor.strides_bytes =
          simaai::neat::pipeline_internal::contiguous_strides_bytes(tensor.shape, elem);
    }
  } else {
    apply_tensor_dims(tensor, info.dims);
  }
}

WireCaps build_wire_caps_from_tensor(const std::vector<std::shared_ptr<Node>>& group,
                                     const simaai::neat::Tensor& input,
                                     const TensorDims* dims_override, const char* media_type,
                                     const char* default_format) {
  WireCaps wire;
  wire.media_type = media_type ? media_type : "application/vnd.simaai.tensor";
  wire.format = default_format ? default_format : format_from_tensor(input);
  wire.dims = dims_override ? *dims_override : dims_from_tensor(input);
  wire.buffer_name = stage_primary_input_buffer_name(group);
  return wire;
}

std::string wire_caps_format_from_dtype_token(const std::string& dtype_token,
                                              const std::string& fallback_format) {
  const std::string fallback = !fallback_format.empty() ? fallback_format : std::string("INT8");
  const FormatSpec fallback_spec{fallback};
  if (fallback_spec.tag == FormatTag::ByteStream) {
    return format_tag_to_string(FormatTag::ByteStream);
  }
  if (dtype_token.empty()) {
    return fallback;
  }
  const std::string up = upper_copy(dtype_token);
  if (up == "INT8") {
    return "EVXX_INT8";
  }
  if (up == "BF16" || up == "BFLOAT16") {
    return "EVXX_BFLOAT16";
  }
  if (up == "FP32" || up == "FLOAT32") {
    return "FP32";
  }
  if (up == "FP64" || up == "FLOAT64") {
    return "FP64";
  }
  if (up == "INT16" || up == "UINT16" || up == "INT32" || up == "UINT8") {
    return up;
  }
  return up;
}

// Boundary/wire helper only. This projects legacy width/height/depth views from
// contract shape + layout and must not be treated as semantic tensor truth.
TensorDims contract_tensor_dims_projection_from_shape(std::vector<int64_t> shape,
                                                      TensorLayout layout) {
  if (shape.size() >= 4U && shape.front() == 1) {
    shape.erase(shape.begin());
  }
  TensorDims dims;
  if (shape.size() >= 3U && layout != TensorLayout::CHW && layout != TensorLayout::HWC) {
    return dims;
  }
  const bool chw_like = layout == TensorLayout::CHW;
  if (shape.size() >= 3U) {
    const int64_t a = shape[shape.size() - 3U];
    const int64_t b = shape[shape.size() - 2U];
    const int64_t c = shape[shape.size() - 1U];
    if (chw_like) {
      dims.depth = static_cast<int>(a);
      dims.height = static_cast<int>(b);
      dims.width = static_cast<int>(c);
    } else {
      dims.height = static_cast<int>(a);
      dims.width = static_cast<int>(b);
      dims.depth = static_cast<int>(c);
    }
  } else if (shape.size() == 2U) {
    dims.height = static_cast<int>(shape[0]);
    dims.width = static_cast<int>(shape[1]);
    dims.depth = 1;
  } else if (shape.size() == 1U) {
    dims.width = static_cast<int>(shape[0]);
    dims.height = 1;
    dims.depth = 1;
  }
  return dims;
}

WireCaps
build_mla_wire_caps_from_contract_or_tensor(const std::vector<std::shared_ptr<Node>>& group,
                                            const simaai::neat::Tensor& input) {
  const MlaInputTensorInfo mla_input = stage_mla_input_tensor_info(group);
  if (mla_input.span_size_bytes <= 0 || mla_input.logical_dtype.empty()) {
    throw std::runtime_error("StageRun: missing strict MLA wire contract in rendered manifest");
  }

  WireCaps wire;
  wire.media_type =
      !mla_input.media_type.empty() ? mla_input.media_type : "application/vnd.simaai.tensor";
  wire.format =
      wire_caps_format_from_dtype_token(mla_input.logical_dtype, format_from_tensor(input));
  wire.dims = mla_input.physical_shape.has_value()
                  ? contract_tensor_dims_projection_from_shape(*mla_input.physical_shape,
                                                               mla_input.logical_layout)
                  : contract_tensor_dims_projection_from_shape(mla_input.logical_shape,
                                                               mla_input.logical_layout);
  wire.buffer_name = stage_primary_input_buffer_name(group);
  return wire;
}

WireInput build_wire_input_from_tensor(const simaai::neat::Tensor& input, const WireCaps& wire) {
  WireInput out;
  out.caps = wire;
  // User tensor format is INT8/BF16; wire caps are plugin-expected (MLA/multi-tensor).
  out.tensor = input;
  if (!wire.format.empty()) {
    const std::string fmt = upper_copy(wire.format);
    if (wire.media_type == "video/x-raw") {
      out.tensor.dtype = TensorDType::UInt8;
      out.tensor.semantic.tess.reset();
      out.tensor.semantic.byte_stream.reset();
      if (!out.tensor.semantic.image.has_value()) {
        simaai::neat::ImageSpec image;
        image.format = image_format_from_string(fmt);
        out.tensor.semantic.image = image;
      } else {
        out.tensor.semantic.image->format = image_format_from_string(fmt);
      }
    } else {
      const FormatSpec fmt_spec{fmt};
      if (fmt_spec.tag == FormatTag::ByteStream) {
        out.tensor.semantic.tess.reset();
        out.tensor.semantic.byte_stream = simaai::neat::ByteStreamSpec{};
        out.tensor.layout = TensorLayout::Unknown;
      } else if (!out.tensor.semantic.tess.has_value()) {
        simaai::neat::TessSpec tess;
        tess.format = fmt;
        out.tensor.semantic.tess = tess;
      } else {
        out.tensor.semantic.tess->format = fmt;
      }
    }
  }
  apply_tensor_hw(out.tensor, wire.dims);
  out.appsrc = appsrc_for_tensor_wire(out.tensor, wire);
  if (!wire.caps_override.empty()) {
    out.appsrc.caps_override = wire.caps_override;
  }
  return out;
}

bool operator==(const StageInputKey& a, const StageInputKey& b) {
  if (a.media_type != b.media_type || a.format != b.format) {
    return false;
  }
  if (upper_copy(a.media_type) == "APPLICATION/VND.SIMAAI.TENSOR") {
    return a.dtype == b.dtype && a.layout == b.layout && a.shape == b.shape;
  }
  return a.width == b.width && a.height == b.height && a.depth == b.depth;
}

bool operator==(const BoxDecodeOptions& a, const BoxDecodeOptions& b) {
  return a.decode_type == b.decode_type && a.detection_threshold == b.detection_threshold &&
         a.nms_iou_threshold == b.nms_iou_threshold && a.top_k == b.top_k;
}

bool operator==(const StageKey& a, const StageKey& b) {
  return a.kind == b.kind && a.model_id == b.model_id && a.input == b.input &&
         a.box_opt == b.box_opt;
}

size_t hash_combine(size_t seed, size_t v) {
  return seed ^ (v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

struct StageKeyHash {
  size_t operator()(const StageKey& k) const {
    size_t h = std::hash<int>()(static_cast<int>(k.kind));
    h = hash_combine(h, std::hash<std::string>()(k.model_id));
    h = hash_combine(h, std::hash<std::string>()(k.input.media_type));
    h = hash_combine(h, std::hash<std::string>()(k.input.format));
    if (upper_copy(k.input.media_type) == "APPLICATION/VND.SIMAAI.TENSOR") {
      h = hash_combine(h, std::hash<int>()(static_cast<int>(k.input.dtype)));
      h = hash_combine(h, std::hash<int>()(static_cast<int>(k.input.layout)));
      for (const auto dim : k.input.shape) {
        h = hash_combine(h, std::hash<int64_t>()(dim));
      }
    } else {
      h = hash_combine(h, std::hash<int>()(k.input.width));
      h = hash_combine(h, std::hash<int>()(k.input.height));
      h = hash_combine(h, std::hash<int>()(k.input.depth));
    }
    h = hash_combine(h, std::hash<int>()(static_cast<int>(k.box_opt.decode_type)));
    h = hash_combine(h, std::hash<double>()(k.box_opt.detection_threshold));
    h = hash_combine(h, std::hash<double>()(k.box_opt.nms_iou_threshold));
    h = hash_combine(h, std::hash<int>()(k.box_opt.top_k));
    return h;
  }
};

std::mutex g_cache_mu;
std::unordered_map<StageKey, std::shared_ptr<Run>, StageKeyHash> g_cache;

InputOptions appsrc_for_mat(const cv::Mat& input, const std::vector<std::shared_ptr<Node>>& group) {
  InputOptions opt;
  opt.payload_type = PayloadType::Image;
  opt.width = input.cols;
  opt.height = input.rows;
  opt.depth = input.channels();
  if (opt.depth == 1) {
    opt.format = "GRAY8";
  } else {
    opt.format = "BGR";
  }
  opt.buffer_name = stage_primary_input_buffer_name(group);
  return opt;
}

int tensor_depth_from_shape(const std::vector<int64_t>& shape) {
  if (shape.size() >= 3)
    return static_cast<int>(shape[2]);
  return -1;
}

int shape_dim(const std::vector<int64_t>& shape, size_t index) {
  if (index >= shape.size())
    return -1;
  return static_cast<int>(shape[index]);
}

// Wire caps are for plugin negotiation only; user-facing tensor metadata stays INT8/BF16.
InputOptions appsrc_for_tensor_wire(const simaai::neat::Tensor& input, const WireCaps& wire) {
  InputOptions opt;
  opt.payload_type =
      wire.media_type.empty() ? PayloadType::Tensor : input_type_from_media_type(wire.media_type);
  const std::string input_fmt = format_from_tensor(input);
  opt.format = wire.format.empty() ? input_fmt : wire.format;
  if (opt.format.empty()) {
    switch (input.dtype) {
    case TensorDType::BFloat16:
      opt.format = "BF16";
      break;
    case TensorDType::Int16:
      opt.format = "INT16";
      break;
    case TensorDType::UInt16:
      opt.format = "UINT16";
      break;
    case TensorDType::Float32:
      opt.format = "FP32";
      break;
    case TensorDType::Int32:
      opt.format = "INT32";
      break;
    case TensorDType::Float64:
      opt.format = "FP64";
      break;
    case TensorDType::UInt8:
      opt.format = "UINT8";
      break;
    case TensorDType::Int8:
    default:
      opt.format = "INT8";
      break;
    }
    if (stage_debug_enabled()) {
      const std::string fmt = opt.format.str();
      std::fprintf(stderr, "[stage] appsrc_for_tensor_wire: inferred format=%s from dtype=%s\n",
                   fmt.c_str(), dtype_name(input.dtype));
    }
  }
  const int shape_h = shape_dim(input.shape, 0);
  const int shape_w = shape_dim(input.shape, 1);
  const int shape_d = shape_dim(input.shape, 2);
  opt.width = (wire.dims.width > 0) ? wire.dims.width : shape_w;
  opt.height = (wire.dims.height > 0) ? wire.dims.height : shape_h;
  opt.depth = (wire.dims.depth > 0) ? wire.dims.depth : shape_d;
  if (opt.width <= 0 || opt.height <= 0) {
    throw std::runtime_error("StageRun: tensor input missing width/height");
  }
  const std::string source_segment_name = tensor_primary_segment_name(input);
  if (!source_segment_name.empty()) {
    opt.buffer_name = source_segment_name;
  } else {
    opt.buffer_name = wire.buffer_name.empty() ? "decoder" : wire.buffer_name;
  }
  const std::string wire_format = upper_copy(opt.format);
  if (wire_format.rfind("EVXX_", 0) == 0 || wire_format.rfind("EV74_", 0) == 0) {
    // Staged handoff for EVXX/EV74-packed tensors should preserve the EV-side ingress target.
    // Falling back to the generic ModelFragment=>DMS0 heuristic breaks standalone MLA ingress.
    opt.memory_policy = InputMemoryPolicy::Ev74;
    opt.use_simaai_pool = true;
  }
  return opt;
}

StageInputKey make_input_key(const InputOptions& opt,
                             const simaai::neat::Tensor* tensor = nullptr) {
  StageInputKey key;
  key.media_type = resolve_input_media_type(opt);
  key.format = upper_copy(opt.format);
  if (tensor && upper_copy(resolve_input_media_type(opt)) == "APPLICATION/VND.SIMAAI.TENSOR") {
    key.dtype = tensor->dtype;
    key.layout = tensor->layout;
    key.shape = tensor->shape;
  } else {
    key.width = opt.width;
    key.height = opt.height;
    key.depth = opt.depth;
  }
  return key;
}

RunOptions stage_run_defaults() {
  RunOptions opt;
  opt.preset = RunPreset::Reliable;
  opt.queue_depth = 1;
  opt.overflow_policy = OverflowPolicy::Block;
  // Standalone stage chaining must preserve the original runtime tensor topology
  // (packed parent vs split OFMs) across stage boundaries.
  opt.output_memory = OutputMemory::ZeroCopy;
  opt.advanced.copy_input = false;
  opt.advanced.sync_num_buffers_override = 1;
  return opt;
}

int default_timeout_ms() {
  const char* env = std::getenv("SIMA_GST_RUN_INPUT_TIMEOUT_MS");
  if (!env || !*env)
    return 10000;
  return std::max(10, std::atoi(env));
}

simaai::neat::Tensor take_tensor(const Sample& out, const char* where) {
  return require_single_tensor(out, where);
}

Sample tensor_as_sample(const simaai::neat::Tensor& input) {
  Sample out = sample_from_tensors(TensorList{input});
  out.payload_type = PayloadType::Tensor;
  out.media_type = "application/vnd.simaai.tensor";
  out.format = format_from_tensor(input);
  if (input.semantic.tess.has_value() && !input.semantic.tess->format.empty()) {
    out.payload_tag = upper_copy(input.semantic.tess->format);
  }
  return out;
}

Sample make_stage_tensor_input_sample(const Sample& source, const simaai::neat::Tensor& tensor,
                                      const WireCaps& wire) {
  // Standalone stage chaining must preserve the exact tensor envelope produced by
  // the previous stage. Repacking a single tensor into a detached CPU-owned sample
  // breaks packed-parent MLA outputs and loses shared segment topology.
  const bool preserving_source_envelope =
      sample_has_tensor_list(source) && source.tensors.size() == 1U && source.fields.empty();
  Sample out = preserving_source_envelope ? source : tensor_as_sample(tensor);
  if (preserving_source_envelope) {
    if (out.kind == SampleKind::Tensor && out.tensor.has_value()) {
      *out.tensor = tensor;
    }
    if (!out.tensors.empty()) {
      out.tensors.front() = tensor;
    }
  }
  if (!wire.media_type.empty()) {
    out.payload_type = payload_type_from_media_type(wire.media_type);
    out.media_type = wire.media_type;
  }
  if (!wire.format.empty()) {
    out.format = wire.format;
    out.payload_tag = wire.format;
  }
  out.owned = source.owned;
  out.frame_id = source.frame_id;
  out.stream_id = source.stream_id;
  out.input_seq = source.input_seq;
  out.orig_input_seq = source.orig_input_seq;
  out.pts_ns = source.pts_ns;
  out.dts_ns = source.dts_ns;
  out.duration_ns = source.duration_ns;
  out.port_name.clear();
  if (!preserving_source_envelope) {
    out.caps_string.clear();
    if (out.segment_name.empty()) {
      out.segment_name = wire.buffer_name;
    }
    if (out.stream_label.empty()) {
      out.stream_label = tensor.route.name;
    }
  }
  return out;
}

Sample make_stage_multi_output_input_sample(const Sample& source, const WireCaps& wire) {
  Sample out = source;
  if (!wire.media_type.empty()) {
    out.payload_type = payload_type_from_media_type(wire.media_type);
    out.media_type = wire.media_type;
  }
  if (!wire.format.empty()) {
    out.format = wire.format;
    out.payload_tag = wire.format;
  }
  out.caps_string.clear();
  return out;
}

bool sample_is_bbox_tensor(const Sample& sample) {
  if (sample.kind != SampleKind::TensorSet || sample.tensors.size() != 1U) {
    return false;
  }
  const Tensor& tensor = sample.tensors.front();
  std::string fmt = sample.payload_tag;
  if (fmt.empty()) {
    fmt = sample.format;
  }
  if (fmt.empty() && tensor.semantic.tess.has_value()) {
    fmt = tensor.semantic.tess->format;
  }
  return upper_copy(fmt) == "BBOX";
}

std::optional<BoxDecodeResult> try_decode_bbox_sample_recursive(const Sample& sample, int img_w,
                                                                int img_h, int expected_topk) {
  if (sample_is_bbox_tensor(sample)) {
    BoxDecodeResult out;
    out.raw =
        require_single_tensor(sample, "try_decode_bbox_sample_recursive").copy_payload_bytes();
    out.boxes = parse_bbox_bytes(out.raw, img_w, img_h, expected_topk, true);
    return out;
  }
  if (sample.kind == SampleKind::TensorSet) {
    if (sample.tensors.size() <= 1U) {
      return std::nullopt;
    }
    for (const auto& tensor : sample.tensors) {
      Sample field = sample_from_tensors(TensorList{tensor});
      field.payload_type = PayloadType::Tensor;
      field.media_type = "application/vnd.simaai.tensor";
      field.output_index = tensor.route.logical_index;
      field.logical_output_index = tensor.route.logical_index;
      field.memory_index = tensor.route.memory_index;
      field.segment_name = tensor.route.segment_name;
      field.stream_label = tensor.route.name;
      if (auto decoded = try_decode_bbox_sample_recursive(field, img_w, img_h, expected_topk);
          decoded.has_value()) {
        return decoded;
      }
    }
    return std::nullopt;
  }
  if (sample.kind == SampleKind::Bundle) {
    for (const auto& field : sample.fields) {
      if (auto decoded = try_decode_bbox_sample_recursive(field, img_w, img_h, expected_topk);
          decoded.has_value()) {
        return decoded;
      }
    }
    return std::nullopt;
  }
  return std::nullopt;
}

std::optional<BoxDecodeResult> try_decode_bbox_payload_tensor_sample(const Sample& sample,
                                                                     int img_w, int img_h,
                                                                     int expected_topk) {
  if (sample.kind != SampleKind::TensorSet || sample.tensors.size() != 1U) {
    return std::nullopt;
  }
  const auto& tensor = sample.tensors.front();
  if (tensor.dtype != TensorDType::UInt8 || tensor.shape.size() != 1U) {
    return std::nullopt;
  }
  if (!sample.payload_tag.empty() || !sample.format.empty()) {
    return std::nullopt;
  }
  if (tensor.semantic.image.has_value() || tensor.semantic.tess.has_value() ||
      tensor.semantic.encoded.has_value()) {
    return std::nullopt;
  }
  BoxDecodeResult out;
  try {
    out.raw = tensor.copy_payload_bytes();
    out.boxes = parse_bbox_bytes(out.raw, img_w, img_h, expected_topk, true);
    return out;
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

TensorList collect_tensors_from_sample(const Sample& sample, const char* where) {
  try {
    return tensors_from_sample(sample, true);
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string(where) + ": " + e.what());
  }
}

struct SelectedTensorSample {
  const Sample* sample = nullptr;
  int logical_output_index = -1;
  int memory_index = -1;
  int route_slot = -1;
};

struct SelectedTensorSampleMutable {
  Sample* sample = nullptr;
  int logical_output_index = -1;
  int memory_index = -1;
  int route_slot = -1;
};

int logical_output_index_for_sample(const Sample& sample) {
  if (sample.logical_output_index >= 0) {
    return sample.logical_output_index;
  }
  return sample.output_index;
}

int memory_index_for_sample(const Sample& sample, int logical_output_index) {
  if (sample.memory_index >= 0) {
    return sample.memory_index;
  }
  if (logical_output_index >= 0) {
    return logical_output_index;
  }
  if (sample.output_index >= 0) {
    return sample.output_index;
  }
  return 0;
}

bool sample_matches_identity_token(const Sample& sample, const std::string& token) {
  if (token.empty()) {
    return false;
  }
  return sample.segment_name == token || sample.stream_label == token;
}

int bundle_field_match_score(const Sample& bundle, const Sample& field) {
  int score = 0;
  const int bundle_logical_index = logical_output_index_for_sample(bundle);
  const int field_logical_index = logical_output_index_for_sample(field);
  if (bundle_logical_index >= 0 &&
      (field_logical_index == bundle_logical_index || field.output_index == bundle_logical_index)) {
    score += 1000;
  }
  if (bundle.route_slot >= 0 && field.route_slot == bundle.route_slot) {
    score += 100;
  }
  if (!bundle.segment_name.empty() && sample_matches_identity_token(field, bundle.segment_name)) {
    score += 50;
  }
  if (!bundle.stream_label.empty() && sample_matches_identity_token(field, bundle.stream_label)) {
    score += 25;
  }
  if (field_logical_index == 0 || field.output_index == 0 || field.route_slot == 0) {
    score += 1;
  }
  return score;
}

SelectedTensorSample select_tensor_sample(const Sample& out, const char* where) {
  SelectedTensorSample selected;
  if (!sample_has_tensor_list(out) && !sample_is_multi_output(out)) {
    throw std::runtime_error(std::string(where) + ": expected tensor output");
  }
  if (out.kind == SampleKind::TensorSet) {
    if (out.tensors.empty()) {
      throw std::runtime_error(std::string(where) + ": TensorSet has no tensors");
    }
    selected.sample = &out;
    selected.logical_output_index = out.tensors.front().route.logical_index;
    selected.memory_index = (out.tensors.front().route.memory_index >= 0)
                                ? out.tensors.front().route.memory_index
                                : out.memory_index;
    selected.route_slot = out.route_slot;
    return selected;
  }
  if (out.fields.empty()) {
    throw std::runtime_error(std::string(where) + ": multi-output sample has no fields");
  }

  const Sample* first_tensor = nullptr;
  const Sample* best_match = nullptr;
  int best_score = -1;
  std::size_t tensor_field_count = 0U;
  for (const auto& field : out.fields) {
    if (!sample_has_tensor_list(field)) {
      continue;
    }
    ++tensor_field_count;
    if (!first_tensor) {
      first_tensor = &field;
    }
    const int score = bundle_field_match_score(out, field);
    if (score > best_score) {
      best_score = score;
      best_match = &field;
    }
  }
  if (best_match && best_score > 0) {
    selected.sample = best_match;
    selected.logical_output_index = logical_output_index_for_sample(*best_match);
    selected.memory_index = memory_index_for_sample(*best_match, selected.logical_output_index);
    selected.route_slot = best_match->route_slot;
    return selected;
  }
  if (first_tensor) {
    selected.sample = first_tensor;
    selected.logical_output_index = logical_output_index_for_sample(*first_tensor);
    selected.memory_index = memory_index_for_sample(*first_tensor, selected.logical_output_index);
    selected.route_slot = first_tensor->route_slot;
    if (stage_debug_enabled()) {
      std::fprintf(stderr,
                   "[stage][bundle] %s: no explicit bundle field identity matched; "
                   "tensor_fields=%zu fallback=first logical=%d route_slot=%d segment=%s\n",
                   where ? where : "StageRun", tensor_field_count, selected.logical_output_index,
                   selected.route_slot,
                   first_tensor->segment_name.empty() ? "<empty>"
                                                      : first_tensor->segment_name.c_str());
    }
    return selected;
  }
  throw std::runtime_error(std::string(where) + ": multi-output sample contains no tensor fields");
}

SelectedTensorSampleMutable select_tensor_sample_mutable(Sample& out, const char* where) {
  SelectedTensorSampleMutable selected;
  if (!sample_has_tensor_list(out) && !sample_is_multi_output(out)) {
    throw std::runtime_error(std::string(where) + ": expected tensor output");
  }
  if (out.kind == SampleKind::TensorSet) {
    if (out.tensors.empty()) {
      throw std::runtime_error(std::string(where) + ": TensorSet has no tensors");
    }
    selected.sample = &out;
    selected.logical_output_index = out.tensors.front().route.logical_index;
    selected.memory_index = (out.tensors.front().route.memory_index >= 0)
                                ? out.tensors.front().route.memory_index
                                : out.memory_index;
    selected.route_slot = out.route_slot;
    return selected;
  }
  if (out.fields.empty()) {
    throw std::runtime_error(std::string(where) + ": multi-output sample has no fields");
  }

  Sample* first_tensor = nullptr;
  Sample* best_match = nullptr;
  int best_score = -1;
  std::size_t tensor_field_count = 0U;
  for (auto& field : out.fields) {
    if (!sample_has_tensor_list(field)) {
      continue;
    }
    ++tensor_field_count;
    if (!first_tensor) {
      first_tensor = &field;
    }
    const int score = bundle_field_match_score(out, field);
    if (score > best_score) {
      best_score = score;
      best_match = &field;
    }
  }
  if (best_match && best_score > 0) {
    selected.sample = best_match;
    selected.logical_output_index = logical_output_index_for_sample(*best_match);
    selected.memory_index = memory_index_for_sample(*best_match, selected.logical_output_index);
    selected.route_slot = best_match->route_slot;
    return selected;
  }
  if (first_tensor) {
    selected.sample = first_tensor;
    selected.logical_output_index = logical_output_index_for_sample(*first_tensor);
    selected.memory_index = memory_index_for_sample(*first_tensor, selected.logical_output_index);
    selected.route_slot = first_tensor->route_slot;
    if (stage_debug_enabled()) {
      std::fprintf(stderr,
                   "[stage][bundle] %s: no explicit bundle field identity matched; "
                   "tensor_fields=%zu fallback=first logical=%d route_slot=%d segment=%s\n",
                   where ? where : "StageRun", tensor_field_count, selected.logical_output_index,
                   selected.route_slot,
                   first_tensor->segment_name.empty() ? "<empty>"
                                                      : first_tensor->segment_name.c_str());
    }
    return selected;
  }
  throw std::runtime_error(std::string(where) + ": multi-output sample contains no tensor fields");
}

simaai::neat::Tensor select_stage_output_tensor_view(const Sample& sample, int memory_index,
                                                     const char* where, const char* source_label,
                                                     const char* selected_label,
                                                     const char* direct_label) {
  if (const Tensor* sample_tensor =
          find_gst_sample_backed_tensor_for_memory_view(sample, memory_index)) {
    log_stage_tensor_holder_state(source_label, *sample_tensor);
    simaai::neat::Tensor tensor =
        pipeline_internal::tensor_view_from_sample_memory(*sample_tensor, memory_index);
    log_stage_tensor_holder_state(selected_label, tensor);
    return tensor;
  }
  if (sample_has_tensor_list(sample) && !sample.tensors.empty() &&
      sample.tensors.front().planes.empty() && stage_debug_enabled()) {
    std::fprintf(stderr, "[stage][holder] %s: skipping sample-memory copy (not GstSample-backed)\n",
                 where ? where : "StageRun");
  }
  simaai::neat::Tensor tensor = take_tensor(sample, where);
  log_stage_tensor_holder_state(direct_label, tensor);
  return tensor;
}

Sample push_and_pull_tensor_preferring_holder(Run& runner, const simaai::neat::Tensor& input,
                                              int timeout_ms, const char* stage_name) {
  const bool gst_sample_backed =
      input.storage && input.storage->kind == simaai::neat::StorageKind::GstSample;
  const std::shared_ptr<void> holder = pipeline_internal::holder_from_tensor(input);
  if (holder && gst_sample_backed) {
    GstBuffer* probe = pipeline_internal::buffer_from_tensor_holder(holder);
    if (probe) {
      gst_buffer_unref(probe);
      if (stage_debug_enabled()) {
        std::fprintf(stderr,
                     "[stage][holder] %s: using push_holder fast path to preserve GstSimaMeta\n",
                     stage_name ? stage_name : "StageRun");
      }
      if (!runner.push_holder(holder)) {
        throw std::runtime_error(std::string(stage_name ? stage_name : "StageRun") +
                                 ": push_holder failed");
      }
      auto out = runner.pull(timeout_ms);
      if (!out.has_value()) {
        throw std::runtime_error(std::string(stage_name ? stage_name : "StageRun") +
                                 ": timeout waiting for output after push_holder");
      }
      return std::move(*out);
    }
  }
  if (holder && stage_debug_enabled() && !gst_sample_backed) {
    std::fprintf(stderr,
                 "[stage][holder] %s: skipping push_and_pull_holder fast path (storage kind=%d, "
                 "using tensor "
                 "payload path)\n",
                 stage_name ? stage_name : "StageRun",
                 input.storage ? static_cast<int>(input.storage->kind) : -1);
  }
  return sample_from_tensors(runner.run(TensorList{input}, timeout_ms));
}

std::shared_ptr<Run> get_or_build(const StageKey& key, const std::function<Run()>& builder) {
  {
    std::lock_guard<std::mutex> lock(g_cache_mu);
    auto it = g_cache.find(key);
    if (it != g_cache.end())
      return it->second;
  }

  Run run = builder();
  auto handle = std::make_shared<Run>(std::move(run));

  std::lock_guard<std::mutex> lock(g_cache_mu);
  auto it = g_cache.find(key);
  if (it != g_cache.end())
    return it->second;
  g_cache.emplace(key, handle);
  return handle;
}

} // namespace

TensorList Tensors(const simaai::neat::Sample& sample) {
  return collect_tensors_from_sample(sample, "Tensors");
}

simaai::neat::Sample PreprocSample(const cv::Mat& input, const simaai::neat::Model& model) {
  auto group = simaai::neat::internal::ModelAccess::build_preprocess_nodes(model, true);
  InputOptions src_opt = appsrc_for_mat(input, group);
  log_stage_group_nodes("Preproc", group);

  const PreprocOutputInfo preproc_info = stage_preproc_output_info(group);
  const auto resolved_preproc = model.resolved_preprocess_plan();
  const auto contract_flags = simaai::neat::internal::ModelAccess::preprocess_contract_flags(model);
  if (resolved_preproc.enabled) {
    const auto in_fmt_spec = preprocess_color_format_to_format_spec(
        resolved_preproc.effective.color_convert.input_format);
    if (resolved_preproc.effective.color_convert.input_format != PreprocessColorFormat::Auto &&
        !in_fmt_spec.str().empty()) {
      src_opt.format = in_fmt_spec.str();
    }
    PreprocessMetaTemplate meta;
    meta.enabled = true;
    meta.quantize = contract_flags.quant_needed;
    meta.tessellate = contract_flags.tess_needed;
    meta.normalize = resolved_preproc.effective.normalize.enable == AutoFlag::On;
    if (resolved_preproc.effective.resize.enable == AutoFlag::On) {
      meta.target_width = resolved_preproc.effective.resize.width;
      meta.target_height = resolved_preproc.effective.resize.height;
      meta.scaled_width = (resolved_preproc.effective.resize.width > 0)
                              ? resolved_preproc.effective.resize.width
                              : preproc_info.logical_dims.width;
      meta.scaled_height = (resolved_preproc.effective.resize.height > 0)
                               ? resolved_preproc.effective.resize.height
                               : preproc_info.logical_dims.height;
      switch (resolved_preproc.effective.resize.mode) {
      case ResizeMode::Stretch:
        meta.resize_mode = "stretch";
        break;
      case ResizeMode::Letterbox:
        meta.resize_mode = "letterbox";
        break;
      case ResizeMode::Crop:
        meta.resize_mode = "crop";
        break;
      }
      meta.pad_value = resolved_preproc.effective.resize.pad_value;
    } else {
      meta.resize_mode = "none";
    }
    const auto out_fmt_spec = preprocess_color_format_to_format_spec(
        resolved_preproc.effective.color_convert.output_format);
    meta.color_in =
        (resolved_preproc.effective.color_convert.input_format == PreprocessColorFormat::Auto)
            ? src_opt.format.str()
            : in_fmt_spec.str();
    meta.color_out =
        (resolved_preproc.effective.color_convert.output_format == PreprocessColorFormat::Auto)
            ? std::string{}
            : out_fmt_spec.str();
    meta.axis_perm = resolved_preproc.effective.layout_convert.perm;
    if (stage_debug_enabled()) {
      std::fprintf(stderr,
                   "[stage][preproc-meta] source=route_contract quant=%d tess=%d "
                   "effective_quant=%d effective_tess=%d\n",
                   meta.quantize ? 1 : 0, meta.tessellate ? 1 : 0,
                   resolved_preproc.effective.quantize.enable == AutoFlag::On ? 1 : 0,
                   resolved_preproc.effective.tessellate.enable == AutoFlag::On ? 1 : 0);
    }
    src_opt.preprocess_meta = meta;
  }

  StageKey key;
  key.kind = StageKind::Preproc;
  key.model_id = simaai::neat::internal::ModelAccess::model_id(model);
  key.input = make_input_key(src_opt);

  auto runner = get_or_build(key, [&]() {
    RunOptions opt = stage_run_defaults();
    // Keep the GstSample alive so we can copy the tessellated memory.
    opt.output_memory = OutputMemory::ZeroCopy;
    auto input_node = simaai::neat::nodes::Input(src_opt);
    auto output_node = simaai::neat::nodes::Output();
    std::vector<std::shared_ptr<Node>> nodes;
    nodes.reserve(group.size() + 2);
    nodes.push_back(input_node);
    for (const auto& node : group) {
      nodes.push_back(node);
    }
    nodes.push_back(output_node);

    Graph p;
    p.add(input_node);
    for (const auto& node : group) {
      p.add(node);
    }
    p.add(output_node);
    return p.build(std::vector<cv::Mat>{input}, RunMode::Sync, opt);
  });

  const int timeout_ms = default_timeout_ms();
  Sample out = sample_from_tensors(runner->run(std::vector<cv::Mat>{input}, timeout_ms));
  log_stage_output_sample("Preproc: output sample", out);
  simaai::neat::Tensor tensor;
  const std::string selected_output_name = preproc_info.primary_output_name;
  if (sample_has_tensor_list(out) && out.tensors.size() == 1U &&
      tensor_is_gst_sample_backed(out.tensors.front())) {
    const int mem_index = resolve_preproc_selected_memory_index(out, preproc_info);
    if (stage_debug_enabled()) {
      std::fprintf(stderr, "[stage][preproc-select] primary_output=%s selected_index=%d\n",
                   selected_output_name.empty() ? "<empty>" : selected_output_name.c_str(),
                   mem_index);
    }
    tensor = select_stage_output_tensor_view(
        out, mem_index, "Preproc", "Preproc: source before tensor_view_from_sample_memory",
        "Preproc: selected tensor view", "Preproc: direct tensor");
  } else {
    tensor = take_tensor(out, "Preproc");
    log_stage_tensor_holder_state("Preproc: direct tensor", tensor);
  }
  const bool packed_tessellated_handoff =
      preproc_info.transport_kind == PreprocOutputTransportKind::Packed;
  apply_preproc_output_override(tensor, preproc_info);
  if (!packed_tessellated_handoff) {
  }
  const std::string pre_fmt = upper_copy(format_from_tensor(tensor));
  if (stage_debug_enabled()) {
    const std::shared_ptr<void> out_holder = pipeline_internal::holder_from_tensor(tensor);
    if (out_holder) {
      GstBuffer* out_buf = pipeline_internal::buffer_from_tensor_holder(out_holder);
      if (out_buf) {
        const auto meta = read_simaai_preprocess_meta(out_buf);
        if (meta.has_value()) {
          std::fprintf(
              stderr,
              "[stage][holder] Preproc output meta original=%dx%d resized=%dx%d scaled=%dx%d\n",
              meta->original_width, meta->original_height, meta->resized_width,
              meta->resized_height, meta->scaled_width, meta->scaled_height);
        } else {
          std::fprintf(stderr, "[stage][holder] Preproc output meta missing\n");
        }
        gst_buffer_unref(out_buf);
      }
    }
  }
  tensor = require_supported_tessellated_dtype(std::move(tensor), "Preproc");

  out = sample_from_tensors(TensorList{std::move(tensor)});
  out.payload_type = PayloadType::Tensor;
  out.media_type = "application/vnd.simaai.tensor";
  out.format = format_from_tensor(out.tensors.front());
  out.segment_name = selected_output_name;
  out.stream_label = selected_output_name;
  if (!out.tensors.front().semantic.tess.has_value() ||
      out.tensors.front().semantic.tess->format.empty()) {
    out.payload_tag = out.format;
  } else {
    out.payload_tag = upper_copy(out.tensors.front().semantic.tess->format);
  }
  log_stage_output_sample("Preproc: selected output sample", out);
  return out;
}

TensorList Preproc(const std::vector<cv::Mat>& inputs, const simaai::neat::Model& model) {
  TensorList out;
  out.reserve(inputs.size());
  for (const auto& input : inputs) {
    TensorList tensors = tensors_from_sample(PreprocSample(input, model), true);
    out.insert(out.end(), tensors.begin(), tensors.end());
  }
  return out;
}

simaai::neat::Sample InferSample(const simaai::neat::Sample& input,
                                 const simaai::neat::Model& model) {
  auto group = simaai::neat::internal::ModelAccess::build_infer_nodes(model, true);
  log_stage_group_nodes("Infer", group);
  const std::vector<MlaOutputTensorInfo> infer_outputs = stage_mla_output_tensors_info(group);
  if (infer_outputs.empty()) {
    throw std::runtime_error("Infer: MLA contract preflight failed: missing MLA output tensor "
                             "contract from strict MPK metadata");
  }
  MlaOutputInfo infer_info = mla_output_info_from_contract_tensor(infer_outputs.front());
  if (infer_info.data_type.empty() || infer_info.size_bytes <= 0) {
    throw std::runtime_error(
        "Infer: MLA contract preflight failed: missing physical output contract "
        "(dtype/size_bytes) from strict MPK metadata");
  }
  const SelectedTensorSample selected_input = select_tensor_sample(input, "Infer input");
  const simaai::neat::Tensor& selected_tensor =
      require_single_tensor(*selected_input.sample, "Infer input");
  enforce_pre_mla_input_bytes_guard(selected_tensor, group, model, "Infer");
  const WireCaps wire = build_mla_wire_caps_from_contract_or_tensor(group, selected_tensor);
  WireCaps stage_wire = wire;
  apply_stage_source_segment_name(&stage_wire, *selected_input.sample);
  const WireInput wire_input = build_wire_input_from_tensor(selected_tensor, stage_wire);
  const InputOptions src_opt = wire_input.appsrc;
  const Sample stage_input =
      make_stage_tensor_input_sample(*selected_input.sample, wire_input.tensor, stage_wire);

  StageKey key;
  key.kind = StageKind::Infer;
  key.model_id = simaai::neat::internal::ModelAccess::model_id(model);
  key.input = make_input_key(src_opt, &wire_input.tensor);

  stage_trace("StageRun::Infer: before get_or_build");
  auto runner = get_or_build(key, [&]() {
    RunOptions opt = stage_run_defaults();
    // Preserve the GstBuffer so post-processing can reuse plugin metadata.
    opt.output_memory = OutputMemory::ZeroCopy;
    auto input_node = simaai::neat::nodes::Input(src_opt);
    auto output_node = simaai::neat::nodes::Output();
    std::vector<std::shared_ptr<Node>> nodes;
    nodes.reserve(group.size() + 2);
    nodes.push_back(input_node);
    for (const auto& node : group) {
      nodes.push_back(node);
    }
    nodes.push_back(output_node);

    Graph p;
    p.add(input_node);
    for (const auto& node : group) {
      p.add(node);
    }
    p.add(output_node);
    return p.build(Sample{stage_input}, RunMode::Sync, opt);
  });
  stage_trace("StageRun::Infer: after get_or_build");

  const int timeout_ms = default_timeout_ms();
  stage_trace("StageRun::Infer: before push_and_pull");
  Sample out = run_single_sample(*runner, stage_input, timeout_ms, "Infer");
  propagate_preprocess_meta_to_sample_if_missing(selected_tensor, &out);
  log_stage_output_sample("Infer: output sample", out);
  log_stage_tensor_stats("Infer: output sample", out);
  stage_trace("StageRun::Infer: after push_and_pull");
  return out;
}

simaai::neat::Sample InferSample(const simaai::neat::Tensor& input,
                                 const simaai::neat::Model& model) {
  return InferSample(tensor_as_sample(input), model);
}

Sample Infer(const Sample& inputs, const simaai::neat::Model& model) {
  if (inputs.kind != SampleKind::Bundle) {
    return InferSample(inputs, model);
  }
  Sample out;
  out.reserve(inputs.size());
  for (const auto& input : inputs) {
    out.push_back(InferSample(input, model));
  }
  return out;
}

TensorList InferOutputs(const simaai::neat::Sample& input, const simaai::neat::Model& model) {
  return collect_tensors_from_sample(InferSample(input, model), "InferOutputs");
}

TensorList InferOutputs(const simaai::neat::Tensor& input, const simaai::neat::Model& model) {
  return InferOutputs(tensor_as_sample(input), model);
}

simaai::neat::Tensor Infer(const simaai::neat::Tensor& input, const simaai::neat::Model& model);

TensorList Infer(const TensorList& inputs, const simaai::neat::Model& model) {
  TensorList out;
  out.reserve(inputs.size());
  for (const auto& input : inputs) {
    out.push_back(Infer(input, model));
  }
  return out;
}

simaai::neat::Tensor Infer(const simaai::neat::Tensor& input, const simaai::neat::Model& model) {
  auto group = simaai::neat::internal::ModelAccess::build_infer_nodes(model, true);
  const std::vector<MlaOutputTensorInfo> infer_outputs = stage_mla_output_tensors_info(group);
  if (infer_outputs.empty()) {
    throw std::runtime_error("Infer: MLA contract preflight failed: missing MLA output tensor "
                             "contract from strict MPK metadata");
  }
  MlaOutputInfo infer_info = mla_output_info_from_contract_tensor(infer_outputs.front());
  if (infer_info.data_type.empty() || infer_info.size_bytes <= 0) {
    throw std::runtime_error(
        "Infer: MLA contract preflight failed: missing physical output contract "
        "(dtype/size_bytes) from strict MPK metadata");
  }
  Sample out = InferSample(tensor_as_sample(input), model);
  const SelectedTensorSample selected = select_tensor_sample(out, "Infer");
  const Sample& selected_sample = *selected.sample;
  const int logical_output_index =
      (selected.logical_output_index >= 0) ? selected.logical_output_index : 0;
  simaai::neat::Tensor tensor;
  if (static_cast<std::size_t>(logical_output_index) < infer_outputs.size()) {
    infer_info = mla_output_info_from_contract_tensor(
        infer_outputs[static_cast<std::size_t>(logical_output_index)]);
  }
  tensor = select_stage_output_tensor_view(selected_sample, selected.memory_index, "Infer",
                                           "Infer: source before tensor_view_from_sample_memory",
                                           "Infer: selected tensor view", "Infer: direct tensor");
  const std::string infer_fmt = upper_copy(format_from_tensor(tensor));
  if (stage_debug_enabled()) {
    const int64_t actual_bytes = tensor_total_bytes(tensor);
    const size_t plane0 =
        tensor.planes.empty()
            ? 0
            : static_cast<size_t>(tensor_plane_bytes_tight(tensor.planes[0], tensor.dtype));
    std::fprintf(stderr,
                 "[DBG] StageRun Infer out: format=%s dtype=%s w=%d h=%d shape=%s plane0=%zu "
                 "bytes=%lld infer_size=%lld\n",
                 infer_fmt.c_str(), dtype_name(tensor.dtype),
                 (tensor.shape.size() > 1) ? static_cast<int>(tensor.shape[1]) : -1,
                 (tensor.shape.size() > 0) ? static_cast<int>(tensor.shape[0]) : -1,
                 shape_string(tensor.shape).c_str(), plane0, static_cast<long long>(actual_bytes),
                 static_cast<long long>(infer_info.size_bytes));
  }
  return require_supported_tessellated_dtype(std::move(tensor), "Infer");
}

simaai::neat::Sample MLASample(const simaai::neat::Sample& input,
                               const simaai::neat::Model& model) {
  auto group = simaai::neat::internal::ModelAccess::build_infer_nodes(model, true);
  log_stage_group_nodes("MLA", group);
  const std::vector<MlaOutputTensorInfo> mla_outputs = stage_mla_output_tensors_info(group);
  if (mla_outputs.empty()) {
    throw std::runtime_error("MLA: contract preflight failed: missing MLA output tensor contract "
                             "from strict MPK metadata");
  }
  MlaOutputInfo mla_info = mla_output_info_from_contract_tensor(mla_outputs.front());
  if (mla_info.data_type.empty() || mla_info.size_bytes <= 0) {
    throw std::runtime_error("MLA: contract preflight failed: missing physical output contract "
                             "(dtype/size_bytes) from strict MPK metadata");
  }
  const SelectedTensorSample selected_input = select_tensor_sample(input, "MLA input");
  const simaai::neat::Tensor& selected_tensor =
      require_single_tensor(*selected_input.sample, "MLA input");
  enforce_pre_mla_input_bytes_guard(selected_tensor, group, model, "MLA");
  const WireCaps wire = build_mla_wire_caps_from_contract_or_tensor(group, selected_tensor);
  WireCaps stage_wire = wire;
  apply_stage_source_segment_name(&stage_wire, *selected_input.sample);
  const WireInput wire_input = build_wire_input_from_tensor(selected_tensor, stage_wire);
  const InputOptions src_opt = wire_input.appsrc;
  const Sample stage_input =
      make_stage_tensor_input_sample(*selected_input.sample, wire_input.tensor, stage_wire);

  StageKey key;
  key.kind = StageKind::MLA;
  key.model_id = simaai::neat::internal::ModelAccess::model_id(model);
  key.input = make_input_key(src_opt, &wire_input.tensor);

  stage_trace("StageRun::MLA: before get_or_build");
  auto runner = get_or_build(key, [&]() {
    RunOptions opt = stage_run_defaults();
    // Preserve the GstBuffer so BoxDecode can reuse plugin metadata.
    opt.output_memory = OutputMemory::ZeroCopy;
    auto input_node = simaai::neat::nodes::Input(src_opt);
    auto output_node = simaai::neat::nodes::Output();
    std::vector<std::shared_ptr<Node>> nodes;
    nodes.reserve(group.size() + 2);
    nodes.push_back(input_node);
    for (const auto& node : group) {
      nodes.push_back(node);
    }
    nodes.push_back(output_node);

    Graph p;
    p.add(input_node);
    for (const auto& node : group) {
      p.add(node);
    }
    p.add(output_node);
    return p.build(Sample{stage_input}, RunMode::Sync, opt);
  });
  stage_trace("StageRun::MLA: after get_or_build");

  const int timeout_ms = default_timeout_ms();
  stage_trace("StageRun::MLA: before push_and_pull");
  Sample out = run_single_sample(*runner, stage_input, timeout_ms, "MLA");
  propagate_preprocess_meta_to_sample_if_missing(selected_tensor, &out);
  log_stage_output_sample("MLA: output sample", out);
  stage_trace("StageRun::MLA: after push_and_pull");
  return out;
}

simaai::neat::Sample MLASample(const simaai::neat::Tensor& input,
                               const simaai::neat::Model& model) {
  return MLASample(tensor_as_sample(input), model);
}

Sample MLA(const Sample& inputs, const simaai::neat::Model& model) {
  if (inputs.kind != SampleKind::Bundle) {
    return MLASample(inputs, model);
  }
  Sample out;
  out.reserve(inputs.size());
  for (const auto& input : inputs) {
    out.push_back(MLASample(input, model));
  }
  return out;
}

TensorList MLAOutputs(const simaai::neat::Sample& input, const simaai::neat::Model& model) {
  return collect_tensors_from_sample(MLASample(input, model), "MLAOutputs");
}

TensorList MLAOutputs(const simaai::neat::Tensor& input, const simaai::neat::Model& model) {
  return MLAOutputs(tensor_as_sample(input), model);
}

TensorList MLA(const TensorList& inputs, const simaai::neat::Model& model) {
  return MLAOutputs(sample_from_tensors(inputs), model);
}

simaai::neat::Tensor MLA(const simaai::neat::Tensor& input, const simaai::neat::Model& model) {
  const std::vector<MlaOutputTensorInfo> mla_outputs = stage_mla_output_tensors_info(
      simaai::neat::internal::ModelAccess::build_infer_nodes(model, true));
  MlaOutputInfo mla_info = mla_output_info_from_contract_tensor(mla_outputs.front());
  Sample out = MLASample(tensor_as_sample(input), model);
  const SelectedTensorSample selected = select_tensor_sample(out, "MLA");
  const Sample& selected_sample = *selected.sample;
  const int logical_output_index =
      (selected.logical_output_index >= 0) ? selected.logical_output_index : 0;
  simaai::neat::Tensor tensor;
  if (static_cast<std::size_t>(logical_output_index) < mla_outputs.size()) {
    mla_info = mla_output_info_from_contract_tensor(
        mla_outputs[static_cast<std::size_t>(logical_output_index)]);
  }
  tensor = select_stage_output_tensor_view(selected_sample, selected.memory_index, "MLA",
                                           "MLA: source before tensor_view_from_sample_memory",
                                           "MLA: selected tensor view", "MLA: direct tensor");
  const std::string mla_fmt = upper_copy(format_from_tensor(tensor));
  if (stage_debug_enabled()) {
    const int64_t actual_bytes = tensor_total_bytes(tensor);
    const size_t plane0 =
        tensor.planes.empty()
            ? 0
            : static_cast<size_t>(tensor_plane_bytes_tight(tensor.planes[0], tensor.dtype));
    std::fprintf(stderr,
                 "[DBG] StageRun MLA out: format=%s dtype=%s w=%d h=%d shape=%s plane0=%zu "
                 "bytes=%lld mla_size=%lld\n",
                 mla_fmt.c_str(), dtype_name(tensor.dtype),
                 (tensor.shape.size() > 1) ? static_cast<int>(tensor.shape[1]) : -1,
                 (tensor.shape.size() > 0) ? static_cast<int>(tensor.shape[0]) : -1,
                 shape_string(tensor.shape).c_str(), plane0, static_cast<long long>(actual_bytes),
                 static_cast<long long>(mla_info.size_bytes));
  }
  return require_supported_tessellated_dtype(std::move(tensor), "MLA");
}

Sample Postprocess(const simaai::neat::Sample& input, const simaai::neat::Model& model) {
  if (input.kind == SampleKind::Bundle) {
    Sample out;
    out.reserve(input.size());
    for (const auto& field : input) {
      out.push_back(Postprocess(field, model));
    }
    return out;
  }
  auto group = simaai::neat::internal::ModelAccess::build_public_postprocess_nodes(model);
  if (group.empty()) {
    return input;
  }
  log_stage_group_nodes("Postprocess", group);

  Sample stage_input = input;
  {
    SelectedTensorSampleMutable selected_stage_input =
        select_tensor_sample_mutable(stage_input, "Postprocess staged input");
    if (!selected_stage_input.sample || !sample_has_tensor_list(*selected_stage_input.sample)) {
      throw std::runtime_error("Postprocess: staged input missing selected tensor field");
    }
    const auto& tensors =
        sample_tensor_list(*selected_stage_input.sample, "Postprocess staged input");
    const auto& tensor = tensors.front();
    if (selected_stage_input.sample->media_type.empty()) {
      selected_stage_input.sample->media_type = "application/vnd.simaai.tensor";
    }
    if (selected_stage_input.sample->format.empty()) {
      const std::string fmt = format_from_tensor(tensor);
      if (!fmt.empty()) {
        selected_stage_input.sample->format = fmt;
      }
    }
    if (selected_stage_input.sample->payload_tag.empty() &&
        !selected_stage_input.sample->format.empty()) {
      selected_stage_input.sample->payload_tag = selected_stage_input.sample->format;
    }
  }

  const SelectedTensorSample selected_input =
      select_tensor_sample(stage_input, "Postprocess input");
  const TensorList& selected_input_tensors =
      sample_tensor_list(const_cast<Sample&>(*selected_input.sample), "Postprocess input");
  const simaai::neat::Tensor& selected_tensor = selected_input_tensors.front();
  const WireCaps wire = build_wire_caps_from_tensor(group, selected_tensor, nullptr,
                                                    "application/vnd.simaai.tensor", nullptr);
  WireCaps stage_wire = wire;
  if (!sample_is_multi_output(stage_input)) {
    apply_stage_source_segment_name(&stage_wire, *selected_input.sample);
  }
  const WireInput wire_input = build_wire_input_from_tensor(selected_tensor, stage_wire);
  const InputOptions src_opt = wire_input.appsrc;
  if (sample_is_multi_output(stage_input)) {
    stage_input = make_stage_multi_output_input_sample(stage_input, stage_wire);
  } else {
    stage_input = make_stage_tensor_input_sample(stage_input, wire_input.tensor, stage_wire);
  }
  StageKey key;
  key.kind = StageKind::Postprocess;
  key.model_id = simaai::neat::internal::ModelAccess::model_id(model);
  key.input = make_input_key(src_opt, &wire_input.tensor);

  auto runner = get_or_build(key, [&]() {
    RunOptions run_opt = stage_run_defaults();
    run_opt.output_memory = OutputMemory::ZeroCopy;
    auto input_node = simaai::neat::nodes::Input(src_opt);
    auto output_node = simaai::neat::nodes::Output();
    std::vector<std::shared_ptr<Node>> nodes;
    nodes.reserve(group.size() + 2);
    nodes.push_back(input_node);
    for (const auto& node : group) {
      nodes.push_back(node);
    }
    nodes.push_back(output_node);

    Graph p;
    p.add(input_node);
    for (const auto& node : group) {
      p.add(node);
    }
    p.add(output_node);
    return p.build(Sample{stage_input}, RunMode::Sync, run_opt);
  });

  const int timeout_ms = default_timeout_ms();
  log_stage_tensor_stats("Postprocess: staged input", stage_input);
  Sample out = run_single_sample(*runner, stage_input, timeout_ms, "Postprocess");
  propagate_preprocess_meta_to_sample_if_missing(selected_tensor, &out);
  log_stage_output_sample("Postprocess: output sample", out);
  log_stage_tensor_stats("Postprocess: output sample", out);
  return out;
}

Sample Postprocess(const simaai::neat::Tensor& input, const simaai::neat::Model& model) {
  return Postprocess(tensor_as_sample(input), model);
}

TensorList PostprocessOutputs(const simaai::neat::Sample& input, const simaai::neat::Model& model) {
  return collect_tensors_from_sample(Postprocess(input, model), "PostprocessOutputs");
}

TensorList PostprocessOutputs(const simaai::neat::Tensor& input, const simaai::neat::Model& model) {
  return PostprocessOutputs(tensor_as_sample(input), model);
}

TensorList Postprocess(const TensorList& inputs, const simaai::neat::Model& model) {
  return PostprocessOutputs(sample_from_tensors(inputs), model);
}

Sample BoxDecodeSample(const simaai::neat::Sample& input, const simaai::neat::Model& model,
                       const BoxDecodeOptions& opt) {
  if (opt.decode_type == BoxDecodeType::Unspecified) {
    throw std::runtime_error(
        "BoxDecode: decode_type is required and cannot be BoxDecodeType::Unspecified");
  }
  const SelectedTensorSample selected_input = select_tensor_sample(input, "BoxDecode input");
  const Sample& selected_input_sample = *selected_input.sample;
  const TensorList& selected_input_tensors =
      sample_tensor_list(const_cast<Sample&>(selected_input_sample), "BoxDecode input");
  const simaai::neat::Tensor& selected_input_tensor = selected_input_tensors.front();
  const auto resolved_preproc = model.resolved_preprocess_plan();
  const bool require_explicit_original =
      resolved_preproc.graph_family != PreprocessGraphFamily::Preproc;
  std::optional<PreprocessRuntimeMeta> pre_meta;
  if (const std::shared_ptr<void> holder =
          pipeline_internal::holder_from_tensor(selected_input_tensor);
      holder) {
    GstBuffer* in_buf = pipeline_internal::buffer_from_tensor_holder(holder);
    if (in_buf) {
      pre_meta = read_simaai_preprocess_meta(in_buf);
      gst_buffer_unref(in_buf);
    }
  }
  std::optional<bool> route_tess_needed = std::nullopt;
  std::optional<bool> route_quant_needed = std::nullopt;
  int original_width = 0;
  int original_height = 0;
  if (pre_meta.has_value()) {
    route_tess_needed = pre_meta->tessellate;
    route_quant_needed = pre_meta->quantize;
    if (require_explicit_original) {
      original_width = pre_meta->original_width;
      original_height = pre_meta->original_height;
    }
  }

  if (!simaai::neat::internal::ModelAccess::has_model_managed_stage(
          model, simaai::neat::internal::StageNodeKind::BoxDecode)) {
    try {
      simaai::neat::internal::ModelAccess::require_model_managed_stage(
          model, simaai::neat::internal::StageNodeKind::BoxDecode,
          "stages::BoxDecode(..., model, ...)");
    } catch (const std::exception& e) {
      throw std::runtime_error(
          std::string(e.what()) +
          " When using stages::BoxDecode(..., model, ...), construct the Model with "
          "Model::Options::decode_type set first (for example BoxDecodeType::YoloV8). "
          "The BoxDecodeOptions argument to stages::BoxDecode configures that stage call, "
          "but it does not retarget an already-constructed Model route.");
    }
  }

  auto node = simaai::neat::nodes::SimaBoxDecode(
      model, opt.decode_type, opt.detection_threshold, opt.nms_iou_threshold, opt.top_k, "",
      route_tess_needed, route_quant_needed, original_width, original_height);
  (void)node->backend_fragment(0);
  std::vector<std::shared_ptr<Node>> group{node};
  log_stage_group_nodes("BoxDecode", group);
  if (const auto req = collect_preprocess_meta_requirement(group, "boxdecode")) {
    const PreprocessRuntimeMeta meta =
        enforce_required_preprocess_meta(selected_input_tensor, *req);
    original_width = meta.original_width;
    original_height = meta.original_height;
  }
  if (original_width <= 0 || original_height <= 0) {
    throw std::runtime_error(
        "BoxDecode: stage='boxdecode' plugin='neatboxdecode' preprocess metadata contract "
        "violation: preproc_original_width/preproc_original_height must be > 0 "
        "(no fallback allowed)");
  }

  auto box_model_opt = simaai::neat::internal::ModelAccess::options(model);
  box_model_opt.decode_type = opt.decode_type;
  if (opt.detection_threshold > 0.0) {
    box_model_opt.score_threshold = static_cast<float>(opt.detection_threshold);
  }
  if (opt.nms_iou_threshold > 0.0) {
    box_model_opt.nms_iou_threshold = static_cast<float>(opt.nms_iou_threshold);
  }
  if (opt.top_k > 0) {
    box_model_opt.top_k = opt.top_k;
  }
  simaai::neat::Model box_model =
      simaai::neat::internal::ModelAccess::clone_with_options(model, box_model_opt);
  return Postprocess(input, box_model);
}

BoxDecodeResult DecodeBoxDecodeResultSample(const simaai::neat::Sample& input,
                                            const simaai::neat::Model& model,
                                            const BoxDecodeOptions& opt) {
  const SelectedTensorSample selected_input = select_tensor_sample(input, "BoxDecode input");
  const TensorList& selected_input_tensors =
      sample_tensor_list(const_cast<Sample&>(*selected_input.sample), "BoxDecode input");
  const simaai::neat::Tensor& selected_input_tensor = selected_input_tensors.front();
  int original_width = 0;
  int original_height = 0;
  if (const std::shared_ptr<void> holder =
          pipeline_internal::holder_from_tensor(selected_input_tensor);
      holder) {
    GstBuffer* in_buf = pipeline_internal::buffer_from_tensor_holder(holder);
    if (in_buf) {
      if (const auto pre_meta = read_simaai_preprocess_meta(in_buf); pre_meta.has_value()) {
        original_width = pre_meta->original_width;
        original_height = pre_meta->original_height;
      }
      gst_buffer_unref(in_buf);
    }
  }
  Sample out = BoxDecodeSample(input, model, opt);
  if (auto decoded =
          try_decode_bbox_sample_recursive(out, original_width, original_height, opt.top_k);
      decoded.has_value()) {
    return *decoded;
  }
  const SelectedTensorSample selected_output = select_tensor_sample(out, "BoxDecode");
  if (auto decoded = try_decode_bbox_payload_tensor_sample(*selected_output.sample, original_width,
                                                           original_height, opt.top_k);
      decoded.has_value()) {
    return *decoded;
  }
  simaai::neat::Tensor tensor = take_tensor(*selected_output.sample, "BoxDecode");
  return decode_bbox_tensor(tensor, original_width, original_height, opt.top_k, true);
}

Sample BoxDecode(const Sample& inputs, const simaai::neat::Model& model,
                 const BoxDecodeOptions& opt) {
  Sample out;
  out.reserve(inputs.size());
  for (const auto& input : inputs) {
    out.push_back(BoxDecodeSample(input, model, opt));
  }
  return out;
}

BoxDecodeResultList BoxDecodeResults(const Sample& inputs, const simaai::neat::Model& model,
                                     const BoxDecodeOptions& opt) {
  BoxDecodeResultList out;
  out.reserve(inputs.size());
  for (const auto& input : inputs) {
    out.push_back(DecodeBoxDecodeResultSample(input, model, opt));
  }
  return out;
}

} // namespace simaai::neat::stages
