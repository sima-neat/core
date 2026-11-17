// The MLA-only model route on the Dataflow backend (DataflowMlaRoute.h; H1).
#if defined(SIMANEAT_WITH_DATAFLOW)

#include "pipeline/runtime/DataflowMlaRoute.h"

#include "pipeline/ErrorCodes.h"
#include "pipeline/internal/DataflowErrorMap.h"
#include "pipeline/internal/ErrorUtil.h"
#include "pipeline/internal/RenderedMlaContractQuery.h"
#include "pipeline/runtime/DataflowInputBinding.h"

#if defined(SIMANEAT_WITH_DATAFLOW_SOC)
#pragma GCC visibility push(hidden)
#include "dataflow/soc/mla_op.h"
#pragma GCC visibility pop
#endif

#include <algorithm>
#include <cctype>
#include <optional>
#include <utility>
#include <variant>

namespace simaai::neat::runtime {

namespace {

namespace df = pipeline_internal::df;
namespace si = pipeline_internal::sima;

std::string upper(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return value;
}

// A contract dtype token as the published tensor-set descriptor carries it and the pull reads it
// (StageRecipePreparation.cpp tensorbuffer_dtype_from_token_local, InputStreamPull.cpp
// tensor_dtype_from_tensor_set_descriptor). nullopt: a token this backend does not carry.
std::optional<TensorDType> dtype_of(const std::string& token) {
  const std::string up = upper(token);
  if (up == "INT8" || up == "EVXX_INT8")
    return TensorDType::Int8;
  if (up == "UINT8" || up == "U8" || up == "EVXX_UINT8")
    return TensorDType::UInt8;
  if (up == "BF16" || up == "BFLOAT16" || up == "EVXX_BFLOAT16")
    return TensorDType::BFloat16;
  if (up == "FP32" || up == "FLOAT32" || up == "EVXX_FP32" || up == "EVXX_FLOAT32")
    return TensorDType::Float32;
  if (up == "INT16" || up == "EVXX_INT16")
    return TensorDType::Int16;
  if (up == "UINT16" || up == "EVXX_UINT16")
    return TensorDType::UInt16;
  if (up == "INT32" || up == "EVXX_INT32")
    return TensorDType::Int32;
  return std::nullopt;
}

// The output caps' format token (StageRecipePreparation.cpp
// processmla_caps_format_from_dtype_local): what a GStreamer pull leaves in TessSpec::format and
// in Sample::payload_tag.
std::string caps_format_of(const std::string& token) {
  const std::string up = upper(token);
  if (up.empty() || up == "INT8")
    return "EVXX_INT8";
  if (up == "FP32" || up == "FLOAT32")
    return "FP32";
  if (up == "FP64" || up == "FLOAT64")
    return "FP64";
  if (up == "BF16" || up == "BFLOAT16")
    return "EVXX_BFLOAT16";
  return up;
}

std::int64_t dtype_size(TensorDType dtype) {
  switch (dtype) {
  case TensorDType::UInt16:
  case TensorDType::Int16:
  case TensorDType::BFloat16:
    return 2;
  case TensorDType::Int32:
  case TensorDType::Float32:
    return 4;
  case TensorDType::Float64:
    return 8;
  default:
    return 1;
  }
}

TensorLayout layout_of(const std::string& token) {
  const std::string normalized = si::tensorsemantics::normalize_layout_token(token);
  if (normalized == "HWC")
    return TensorLayout::HWC;
  if (normalized == "CHW")
    return TensorLayout::CHW;
  if (normalized == "HW")
    return TensorLayout::HW;
  return TensorLayout::Unknown;
}

std::string dims_text(const std::vector<std::int64_t>& shape) {
  std::string out = "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    out += (i == 0 ? "" : ",") + std::to_string(shape[i]);
  }
  return out + "]";
}

} // namespace

bool dataflow_has_mla_route() noexcept {
#if defined(SIMANEAT_WITH_DATAFLOW_SOC)
  return true;
#else
  return false;
#endif
}

std::shared_ptr<const DataflowMlaStage>
plan_dataflow_mla_stage(const std::vector<std::shared_ptr<Node>>& route, std::string* why) {
  const auto decline = [why](std::string reason) {
    *why = std::move(reason);
    return nullptr;
  };
  // The contracts of the route's nodes, compiled and rendered as GStreamer's build compiles them.
  si::ManifestBuildDiagnostics diagnostics;
  std::optional<si::SimaPluginStaticManifest> manifest;
  try {
    manifest = pipeline_internal::rendered_stage_query::rendered_manifest_from_nodes(
        std::span<const std::shared_ptr<Node>>(route), "DataflowMlaRoute", &diagnostics);
  } catch (const std::exception& error) {
    return decline(std::string("the model fragment's contracts do not compile: ") + error.what());
  }
  if (!manifest) {
    return decline(
        "the model fragment's contracts do not compile" +
        (diagnostics.errors.empty() ? std::string() : ": " + diagnostics.errors.front()));
  }
  if (manifest->stages.size() != 1U ||
      manifest->stages.front().payload_kind != si::StagePayloadKind::ProcessMla) {
    std::string kinds;
    for (const si::StageStaticSpec& stage : manifest->stages) {
      kinds += (kinds.empty() ? "" : ", ") +
               (stage.kernel_kind.empty() ? stage.plugin_kind : stage.kernel_kind);
    }
    return decline("a model fragment of " + std::to_string(manifest->stages.size()) + " stages (" +
                   kinds +
                   ") is not supported before the next H1 step: only one MLA "
                   "stage (Model::inference() of an MLA-only model) runs on Dataflow");
  }
  const si::StageStaticSpec& stage = manifest->stages.front();
  const si::ProcessMlaStagePayload& mla = stage.processmla;
  if (mla.model_path.empty()) {
    return decline("the MLA stage names no model file");
  }
  if (mla.batch_size > 1 || mla.batch_sz_model > 1) {
    return decline("a batched MLA model (batch " +
                   std::to_string(std::max(mla.batch_size, mla.batch_sz_model)) +
                   ") is not supported on Dataflow yet");
  }
  if (stage.physical_inputs.size() != 1U || stage.logical_inputs.size() != 1U) {
    return decline("an MLA stage of " + std::to_string(stage.physical_inputs.size()) +
                   " input buffers (" + std::to_string(stage.logical_inputs.size()) +
                   " tensors) is not supported on Dataflow yet: the runtime has no join");
  }
  if (stage.physical_outputs.size() != 1U || stage.logical_outputs.size() != 1U) {
    return decline("an MLA stage of " + std::to_string(stage.physical_outputs.size()) +
                   " output buffers (" + std::to_string(stage.logical_outputs.size()) +
                   " tensors) is not supported on Dataflow yet");
  }
  const si::LogicalInputStaticSpec& in = stage.logical_inputs.front();
  const si::LogicalTensorStaticSpec& out = stage.logical_outputs.front();
  const si::PhysicalBufferStaticSpec& in_buffer = stage.physical_inputs.front();
  const si::PhysicalBufferStaticSpec& out_buffer = stage.physical_outputs.front();
  const std::optional<TensorDType> in_dtype = dtype_of(in.dtype);
  const std::optional<TensorDType> out_dtype = dtype_of(out.dtype);
  if (!in_dtype || !out_dtype) {
    return decline("the MLA stage's dtype '" + (in_dtype ? out.dtype : in.dtype) +
                   "' has no Dataflow dtype");
  }
  if (in.materialization_kind != si::TensorMaterializationKind::Direct || in.byte_offset != 0 ||
      in_buffer.size_bytes == 0U || in_buffer.source_byte_offset != 0) {
    return decline("the MLA stage's input is not one whole buffer (a repacked or offset view), "
                   "which is not supported on Dataflow yet");
  }
  if (out.byte_offset < 0 || out.size_bytes == 0U || out_buffer.size_bytes == 0U ||
      static_cast<std::uint64_t>(out.byte_offset) > out_buffer.size_bytes ||
      out.size_bytes > out_buffer.size_bytes - static_cast<std::uint64_t>(out.byte_offset) ||
      out.shape.empty()) {
    return decline("the MLA stage's published output does not lie in its output buffer");
  }

  auto planned = std::make_shared<DataflowMlaStage>();
  planned->model_path = mla.model_path;
  planned->input_bytes = in_buffer.size_bytes;
  planned->input_dtype = *in_dtype;
  planned->input_shape = in.shape;
  planned->output_bytes = out_buffer.size_bytes;
  DataflowPulledTensor& pulled = planned->output;
  pulled.dtype = *out_dtype;
  pulled.layout = layout_of(out.layout);
  pulled.shape = out.shape;
  pulled.strides_bytes = out.stride_bytes;
  if (pulled.strides_bytes.empty()) {
    // StageRecipePreparation.cpp contiguous_stride_bytes_local.
    pulled.strides_bytes.assign(pulled.shape.size(), 0);
    std::int64_t stride = dtype_size(pulled.dtype);
    for (std::size_t i = pulled.shape.size(); i-- > 0U;) {
      pulled.strides_bytes[i] = stride;
      stride *= std::max<std::int64_t>(pulled.shape[i], 1);
    }
  }
  pulled.byte_offset = out.byte_offset;
  pulled.size_bytes = out.size_bytes;
  pulled.stage_key = stage.logical_stage_id.empty() ? stage.element_name : stage.logical_stage_id;
  pulled.format = caps_format_of(out.dtype);
  pulled.route.stage_key = pulled.stage_key;
  pulled.route.logical_index = out.logical_index;
  pulled.route.backend_output_index = out.backend_output_index;
  pulled.route.route_slot = out.output_slot;
  // One output buffer: the final layout's allocation 0 (build_publish_contract_from_manifest_
  // stage_local names a finalized stage's physical outputs "allocation_<index>").
  pulled.route.physical_index = 0;
  pulled.route.memory_index = 0;
  pulled.route.physical_byte_offset = out.byte_offset;
  pulled.route.name =
      out.logical_name.empty() ? "output" + std::to_string(out.logical_index) : out.logical_name;
  pulled.route.backend_name = out.backend_name;
  pulled.route.segment_name = "allocation_0";
  if (out.quant.has_value()) {
    QuantSpec quant;
    quant.axis = out.quant->axis;
    for (const double scale : out.quant->scales) {
      quant.scales.push_back(static_cast<float>(scale));
    }
    for (const std::int64_t zero_point : out.quant->zero_points) {
      quant.zero_points.push_back(static_cast<std::int32_t>(zero_point));
    }
    if (!quant.scales.empty()) {
      quant.scale = quant.scales.front();
    }
    if (!quant.zero_points.empty()) {
      quant.zero_point = quant.zero_points.front();
    }
    pulled.quant = std::move(quant);
  }
  planned->text = "MLA stage " + pulled.stage_key + " (depth " + std::to_string(planned->depth) +
                  "): input " + in.dtype + " " + dims_text(in.shape) + " in " +
                  std::to_string(planned->input_bytes) + " bytes, output " + out.dtype + " " +
                  dims_text(out.shape) + " (" + out.layout + ") " + std::to_string(out.size_bytes) +
                  " bytes at " + std::to_string(out.byte_offset) + " of " +
                  std::to_string(planned->output_bytes) + (out.quant ? ", quantized" : "");
  return planned;
}

std::string bind_dataflow_mla_input(const DataflowMlaStage& stage, DataflowInputBinding& binding) {
  const std::string expected = "the model's MLA input is " + dims_text(stage.input_shape) + " of " +
                               std::to_string(stage.input_bytes) + " bytes";
  if (binding.is_record() || binding.planes != DataflowInputBinding::Planes::None) {
    return "an item of several tensors or planes where one dense tensor is expected; " + expected;
  }
  if (binding.spec.memory == df::MemoryKind::Host) {
    return "an input in system memory (InputOptions::memory_policy SystemMemory, or "
           "use_simaai_pool = false) in front of a model: the MLA takes device memory";
  }
  if (binding.dtype != stage.input_dtype) {
    return "dtype differs from the model's MLA input; " + expected;
  }
  // A bounded axis (InputOptions::max_*) is fixed at its bound: the MLA's input has one size.
  if (const std::string why = fix_dataflow_binding_at_bound(binding); !why.empty()) {
    return why;
  }
  if (df::slot_bytes(binding.spec) != stage.input_bytes) {
    return "a tensor of " + std::to_string(df::slot_bytes(binding.spec)) + " bytes; " + expected;
  }
  binding.pulled = &stage.output;
  return {};
}

#if defined(SIMANEAT_WITH_DATAFLOW_SOC)

std::shared_ptr<void> load_dataflow_mla_model(const DataflowMlaStage& stage, const char* where) {
  df::Result<std::shared_ptr<df::soc::MlaModel>> loaded = df::soc::MlaModel::load(stage.model_path);
  if (const df::Error* error = df::failure(loaded)) {
    pipeline_internal::throw_dataflow_error(*error, where, "MLA model " + stage.model_path);
  }
  std::shared_ptr<df::soc::MlaModel> model =
      std::get<std::shared_ptr<df::soc::MlaModel>>(std::move(loaded));
  const std::span<const df::PortSpec> inputs = model->inputs();
  const std::span<const df::PortSpec> outputs = model->outputs();
  if (inputs.size() != 1U || outputs.size() != 1U ||
      df::slot_bytes(inputs.front()) != stage.input_bytes ||
      df::slot_bytes(outputs.front()) != stage.output_bytes) {
    pipeline_internal::error_util::throw_session_error(
        error_codes::kPipelineShape,
        std::string(where) + ": the MLA model " + stage.model_path + " has " +
            std::to_string(inputs.size()) + " input(s) and " + std::to_string(outputs.size()) +
            " output(s)" +
            (inputs.size() == 1U && outputs.size() == 1U
                 ? " of " + std::to_string(df::slot_bytes(inputs.front())) + " and " +
                       std::to_string(df::slot_bytes(outputs.front())) + " bytes"
                 : std::string()) +
            ", not the " + std::to_string(stage.input_bytes) + " and " +
            std::to_string(stage.output_bytes) +
            " bytes of the model package's contract (Dataflow backend)");
  }
  return model;
}

std::uint32_t add_dataflow_mla_op(df::Net& net, const DataflowMlaStage& stage,
                                  const std::shared_ptr<void>& model, const df::PortSpec& input,
                                  df::PortSpec& output) {
  auto mla = std::static_pointer_cast<df::soc::MlaModel>(model);
  // The output port carries the MLA's output buffer as it is (bytes; the pulled tensor is a view
  // of it, DataflowPulledTensor), in the MLA's own memory.
  output = mla->outputs().front();
  df::soc::MlaOp::Options options;
  options.depth = stage.depth;
  options.inputs = {input};
  options.outputs = {output};
  return net.add<df::soc::MlaOp>("mla", std::move(mla), std::move(options)).index;
}

#else

std::shared_ptr<void> load_dataflow_mla_model(const DataflowMlaStage&, const char* where) {
  pipeline_internal::error_util::throw_session_error(
      error_codes::kPipelineShape,
      std::string(where) + ": this build of libsima_neat has no MLA operator (dataflow::soc)");
}

std::uint32_t add_dataflow_mla_op(df::Net&, const DataflowMlaStage&, const std::shared_ptr<void>&,
                                  const df::PortSpec&, df::PortSpec&) {
  pipeline_internal::error_util::throw_session_error(
      error_codes::kPipelineShape,
      "Graph::build: this build of libsima_neat has no MLA operator (dataflow::soc)");
}

#endif // SIMANEAT_WITH_DATAFLOW_SOC

} // namespace simaai::neat::runtime

#endif // SIMANEAT_WITH_DATAFLOW
