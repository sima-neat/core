// A model route on the Dataflow backend (DataflowModelRoute.h; H1).
#if defined(SIMANEAT_WITH_DATAFLOW)

#include "pipeline/runtime/DataflowModelRoute.h"

#include "pipeline/ErrorCodes.h"
#include "pipeline/internal/DataflowErrorMap.h"
#include "pipeline/internal/ErrorUtil.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/internal/RenderedMlaContractQuery.h"
#include "pipeline/internal/sima/KernelStageRecipe.h"
#include "pipeline/internal/sima/PreprocFrameAdapter.h"
#include "pipeline/internal/sima/StageRecipePreparation.h"
#include "pipeline/runtime/DataflowInputBinding.h"

#include <simaai/scheduler/client/kernel.h>

#if defined(SIMANEAT_WITH_DATAFLOW_SOC)
#pragma GCC visibility push(hidden)
#include "dataflow/soc/mla_op.h"
#if defined(SIMANEAT_WITH_DATAFLOW_SOC_KERNELS)
#include "dataflow/soc/kernel_op.h"
#endif
#pragma GCC visibility pop
#endif

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <variant>

namespace simaai::neat::runtime {

namespace {

namespace df = pipeline_internal::df;
namespace si = pipeline_internal::sima;
namespace client = ::simaai::scheduler::client;

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

bool dataflow_has_model_route() noexcept {
#if defined(SIMANEAT_WITH_DATAFLOW_SOC)
  return true;
#else
  return false;
#endif
}

namespace {

// The dtype of a kernel request's tensor, as Core's tensors name it.
std::optional<TensorDType> dtype_of(client::DType dtype) {
  switch (dtype) {
  case client::DType::Float32:
    return TensorDType::Float32;
  case client::DType::BFloat16:
    return TensorDType::BFloat16;
  case client::DType::Int8:
    return TensorDType::Int8;
  case client::DType::UInt8:
    return TensorDType::UInt8;
  case client::DType::Int16:
    return TensorDType::Int16;
  case client::DType::Int32:
    return TensorDType::Int32;
  case client::DType::UInt16:
    return TensorDType::UInt16;
  default:
    return std::nullopt;
  }
}

std::string scalar_text(const client::ParameterScalar& scalar) {
  return std::visit(
      [](const auto& value) -> std::string {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, std::string>) {
          return value;
        } else if constexpr (std::is_same_v<T, bool>) {
          return value ? "true" : "false";
        } else if constexpr (std::is_same_v<T, double>) {
          char text[40];
          std::snprintf(text, sizeof text, "%.17g", value);
          return text;
        } else {
          return std::to_string(value);
        }
      },
      scalar.value);
}

// A kernel request's parameters as text, for the [NEAT] trace: "name=value, ...".
std::string parameters_text(const std::vector<client::Parameter>& parameters) {
  std::string out;
  for (const client::Parameter& parameter : parameters) {
    out += (out.empty() ? "" : ", ") + parameter.name + "=";
    if (const auto* scalar = std::get_if<client::ParameterScalar>(&parameter.value.value)) {
      out += scalar_text(*scalar);
    } else if (const auto* array = std::get_if<client::ParameterArray>(&parameter.value.value)) {
      out += "[";
      for (std::size_t i = 0; i < array->size(); ++i) {
        out += (i == 0 ? "" : ",") + scalar_text((*array)[i]);
      }
      out += "]";
    } else {
      out += "(...)";
    }
  }
  return out;
}

const char* target_text(client::Hardware target) {
  return target == client::Hardware::A65 ? "A65" : "EV74";
}

// The caps format token of a kernel stage's output (StageRecipePreparation.cpp
// processcvu_caps_format_from_dtype_local): what a GStreamer pull leaves in TessSpec::format and
// in Sample::payload_tag behind a kernel stage.
std::string kernel_caps_format_of(const std::string& token, const std::string& fallback) {
  const std::string up = upper(token);
  if (up.empty()) {
    return fallback.empty() ? std::string("EVXX_INT8") : fallback;
  }
  if (up.rfind("EVXX_", 0) == 0) {
    return up;
  }
  if (up == "INT8" || up == "INT16" || up == "INT32") {
    return "EVXX_" + up;
  }
  if (up == "FP16" || up == "FLOAT16" || up == "HALF") {
    return "EVXX_FLOAT16";
  }
  if (up == "FP32" || up == "FLOAT32") {
    return "EVXX_FLOAT32";
  }
  if (up == "BF16" || up == "BFLOAT16") {
    return "EVXX_BFLOAT16";
  }
  return up;
}

// The stage's name, as GStreamer names its element before the per-pipeline numbering.
std::string stage_key_of(const si::StageStaticSpec& stage) {
  return stage.logical_stage_id.empty() ? stage.element_name : stage.logical_stage_id;
}

std::string stage_kind_text(const si::StageStaticSpec& stage) {
  if (stage.native_kernel && !stage.native_kernel->requests.empty()) {
    return stage.native_kernel->requests.front().operation + "@" +
           target_text(stage.native_kernel->target);
  }
  if (stage.native_kernel) {
    return (stage.processcvu.graph_name.empty() ? stage.kernel_kind : stage.processcvu.graph_name) +
           "@" + target_text(stage.native_kernel->target);
  }
  return stage.kernel_kind.empty() ? stage.plugin_kind : stage.kernel_kind;
}

// The pulled tensor of the route: the last stage's one published output `out`, described as
// GStreamer's publication and pull describe it.
void describe_pulled(const si::StageStaticSpec& stage, const si::LogicalTensorStaticSpec& out,
                     TensorDType dtype, DataflowPulledTensor& pulled) {
  pulled.dtype = dtype;
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
  pulled.stage_key = stage_key_of(stage);
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
}

// One MLA stage of the manifest as a route stage; "" or why this backend cannot carry it.
std::string plan_mla_stage(const si::StageStaticSpec& stage, bool first, bool last,
                           DataflowModelRoute& route) {
  const si::ProcessMlaStagePayload& mla = stage.processmla;
  if (mla.model_path.empty()) {
    return "the MLA stage names no model file";
  }
  if (mla.batch_size > 1 || mla.batch_sz_model > 1) {
    return "a batched MLA model (batch " +
           std::to_string(std::max(mla.batch_size, mla.batch_sz_model)) +
           ") is not supported on Dataflow yet";
  }
  if (stage.physical_inputs.size() != 1U || stage.logical_inputs.size() != 1U) {
    return "an MLA stage of " + std::to_string(stage.physical_inputs.size()) + " input buffers (" +
           std::to_string(stage.logical_inputs.size()) +
           " tensors) is not supported on Dataflow yet: the runtime has no join";
  }
  if (stage.physical_outputs.size() != 1U || stage.logical_outputs.size() != 1U) {
    return "an MLA stage of " + std::to_string(stage.physical_outputs.size()) +
           " output buffers (" + std::to_string(stage.logical_outputs.size()) +
           " tensors) is not supported on Dataflow yet";
  }
  const si::LogicalInputStaticSpec& in = stage.logical_inputs.front();
  const si::LogicalTensorStaticSpec& out = stage.logical_outputs.front();
  const si::PhysicalBufferStaticSpec& in_buffer = stage.physical_inputs.front();
  const si::PhysicalBufferStaticSpec& out_buffer = stage.physical_outputs.front();
  const std::optional<TensorDType> in_dtype = dtype_of(in.dtype);
  const std::optional<TensorDType> out_dtype = dtype_of(out.dtype);
  if (!in_dtype || !out_dtype) {
    return "the MLA stage's dtype '" + (in_dtype ? out.dtype : in.dtype) +
           "' has no Dataflow dtype";
  }
  if (in.materialization_kind != si::TensorMaterializationKind::Direct || in.byte_offset != 0 ||
      in_buffer.size_bytes == 0U || in_buffer.source_byte_offset != 0) {
    return "the MLA stage's input is not one whole buffer (a repacked or offset view), "
           "which is not supported on Dataflow yet";
  }
  if (out.byte_offset < 0 || out.size_bytes == 0U || out_buffer.size_bytes == 0U ||
      static_cast<std::uint64_t>(out.byte_offset) > out_buffer.size_bytes ||
      out.size_bytes > out_buffer.size_bytes - static_cast<std::uint64_t>(out.byte_offset) ||
      out.shape.empty()) {
    return "the MLA stage's published output does not lie in its output buffer";
  }
  DataflowModelStage planned;
  planned.kind = DataflowModelStage::Kind::Mla;
  planned.key = stage_key_of(stage);
  planned.model_path = mla.model_path;
  planned.input_bytes = in_buffer.size_bytes;
  planned.output_bytes = out_buffer.size_bytes;
  if (first) {
    route.input_bytes = planned.input_bytes;
    route.input_dtype = *in_dtype;
    route.input_shape = in.shape;
  }
  if (last) {
    describe_pulled(stage, out, *out_dtype, route.output);
    route.output.format = caps_format_of(out.dtype);
    route.output.sample_label = route.output.stage_key;
  }
  route.text += std::string(route.text.empty() ? "" : " -> ") + "MLA stage " + planned.key +
                " (depth " + std::to_string(planned.depth) + "): input " + in.dtype + " " +
                dims_text(in.shape) + " in " + std::to_string(planned.input_bytes) +
                " bytes, output " + out.dtype + " " + dims_text(out.shape) + " (" + out.layout +
                ") " + std::to_string(out.size_bytes) + " bytes at " +
                std::to_string(out.byte_offset) + " of " + std::to_string(planned.output_bytes) +
                (out.quant ? ", quantized" : "");
  route.stages.push_back(std::move(planned));
  return {};
}

#if defined(SIMANEAT_WITH_DATAFLOW_SOC_KERNELS)

// The frame an image stage is planned for: what the build says of the input's size.
struct PlannedFrame {
  std::int64_t width = 0, height = 0, channels = 0;
  std::string from; ///< "the seed" or "the Input's declaration", for messages
};

// The size of the frames the input will carry, from the build's seed or, without one, from the
// Input's declaration (InputOptions::shape with every axis fixed, or width and height). "" and
// `frame` set, or why the build does not say it.
std::string planned_frame(const InputOptions& declared, std::span<const Tensor> seed,
                          PlannedFrame& frame) {
  const auto of_shape = [&frame](const std::vector<std::int64_t>& shape) {
    const std::size_t lead = shape.size() == 4U && shape[0] == 1 ? 1U : 0U;
    if (shape.size() != 3U + lead) {
      return false;
    }
    frame.height = shape[lead];
    frame.width = shape[lead + 1U];
    frame.channels = shape[lead + 2U];
    return frame.height > 0 && frame.width > 0 && frame.channels > 0;
  };
  if (!seed.empty()) {
    frame.from = "the seed";
    if (seed.size() != 1U || seed.front().is_composite()) {
      return "the seed is a planar image or several tensors (NV12, I420): a planar input is not "
             "supported on Dataflow yet";
    }
    const Tensor& tensor = seed.front();
    if (tensor.dtype != TensorDType::UInt8 || !tensor.is_dense() ||
        tensor.layout != TensorLayout::HWC || !of_shape(tensor.shape)) {
      return "the seed is not a packed UInt8 image of shape [height, width, channels] (or with a "
             "batch axis of 1 in front)";
    }
    if (!tensor.is_contiguous()) {
      return "the seed's rows are padded (its row stride is not width x channels): a per-frame "
             "row stride is not supported on Dataflow yet";
    }
    return {};
  }
  frame.from = "the Input's declaration";
  if (!declared.shape.empty()) {
    if (declared.dtype != TensorDType::UInt8 || !of_shape(declared.shape)) {
      return "the Input's declared shape does not fix a UInt8 frame [height, width, channels]: "
             "the frame's size is not known at the build";
    }
    return {};
  }
  if (declared.width > 0 && declared.height > 0) {
    frame.width = declared.width;
    frame.height = declared.height;
    return {};
  }
  return "the frame's size is not known at the build (no seed, and the Input declares neither a "
         "fixed shape nor width and height): the Dataflow backend runs this stage at one frame "
         "size, and a size that may change with each push is not supported on it yet";
}

// The image preproc stage (the Kernels package's "preproc", which GStreamer runs through a frame
// adapter) as one kernel at the frame size the build says; "" or why this backend cannot carry
// it. The request, the published tensor's description and the preprocess facts all come from the
// stage's own checked policy (PreprocFrameAdapter), by the functions the adapter runs for a frame
// of that geometry.
std::string plan_frame_stage(const si::StageStaticSpec& stage, const std::string& name, bool first,
                             bool last, const InputOptions& declared, std::span<const Tensor> seed,
                             DataflowModelRoute& route) {
  using Adapter = si::kernel_lowering::PreprocFrameAdapter;
  const si::KernelStageRecipe& recipe = *stage.native_kernel;
  const si::ProcessCvuStagePayload& payload = stage.processcvu;
  if (!recipe.frame_adapter_factory || payload.graph_name != "preproc" || !first) {
    return name + " is an image stage with per-frame geometry, which is not supported on "
                  "Dataflow yet";
  }
  if (payload.image_transform || recipe.image_publication || stage.image_publication) {
    return name + " is an image frontend (it publishes image planes), which is not supported on "
                  "Dataflow yet";
  }
  const std::string format = upper(payload.input_img_type);
  if (format != "RGB" && format != "BGR") {
    return name + " reads " + (format.empty() ? std::string("an unnamed format") : format) +
           " frames: only packed RGB and BGR frames are supported on Dataflow yet (NV12, I420 "
           "and NV16 are several planes; the kernel operator takes one input)";
  }
  if (payload.batch_size != 1) {
    return name + " has a batch of " + std::to_string(payload.batch_size) +
           " (an ROI list or a batched model), which is not supported on Dataflow yet";
  }
  if (recipe.members.size() != 1U || recipe.members.front().outputs.size() != 1U ||
      !recipe.resolver_factories.empty() || !recipe.constant_data.empty() ||
      !recipe.private_buffers.empty() || !recipe.private_outputs.empty() ||
      stage.logical_outputs.size() != 1U) {
    return name + " has several outputs, variants, constants or private buffers, which is not "
                  "supported on Dataflow yet";
  }
  PlannedFrame frame;
  if (const std::string why = planned_frame(declared, seed, frame); !why.empty()) {
    return name + ": " + why;
  }
  auto checked = Adapter::check_policy(payload);
  if (const auto* error = std::get_if<client::Error>(&checked)) {
    return name + ": the stage's image policy does not check: " + error->field + ": " +
           error->message;
  }
  // Row pitch: the packed frame's own (a slot of the input's pool holds exactly the frame).
  const std::int64_t channels = frame.channels > 0 ? frame.channels : 3;
  auto fixed = Adapter::checked_fixed_frame(
      std::get<std::shared_ptr<const Adapter::CheckedPolicy>>(std::move(checked)),
      static_cast<std::size_t>(frame.width), static_cast<std::size_t>(frame.height),
      static_cast<std::size_t>(frame.width * channels));
  if (const auto* error = std::get_if<client::Error>(&fixed)) {
    return name + ": a " + std::to_string(frame.width) + "x" + std::to_string(frame.height) +
           " frame (" + frame.from + ") is not one the stage takes: " + error->field + ": " +
           error->message;
  }
  Adapter::FixedFrame& request = std::get<Adapter::FixedFrame>(fixed);
  if (request.inputs.size() != 1U || request.outputs.size() != 1U ||
      request.inputs.front().shape.size() != 4U) {
    return name + ": the frame's kernel request has several tensors";
  }
  if (frame.channels > 0 && request.inputs.front().shape[3] != frame.channels) {
    return name + ": " + frame.from + " has " + std::to_string(frame.channels) +
           " channels where the model's " + format + " input has " +
           std::to_string(request.inputs.front().shape[3]);
  }
  auto inspected = client::inspect_kernel(request.executable);
  if (const auto* error = std::get_if<client::Error>(&inspected)) {
    return name + ": the Kernels package gave no contract: " + error->message;
  }
  const client::KernelInspection& contract = std::get<client::KernelInspection>(inspected);
  if (contract.input_requirements().size() != 1U || contract.output_requirements().size() != 1U ||
      contract.workspace().storage_bytes != 0U || contract.requires_host_tasks() ||
      contract.input_extent(0).mode != client::InputExtentMode::Fixed) {
    return name + ": the frame's kernel needs a workspace, host tasks or a varying input extent, "
                  "which is not supported on Dataflow yet";
  }
  const std::optional<TensorDType> in_dtype = dtype_of(request.inputs.front().dtype);
  const std::optional<TensorDType> out_dtype = dtype_of(request.outputs.front().dtype);
  const simaai::gst::TensorBufferPreprocessFacts* const facts =
      simaai::gst::tensor_buffer_preprocess_facts(request.facts);
  if (!in_dtype || !out_dtype || !facts) {
    return name + ": the frame's kernel request has no Core dtype or no preprocess facts";
  }
  DataflowModelStage planned;
  planned.kind = DataflowModelStage::Kind::Kernel;
  planned.key = stage_key_of(stage);
  planned.ev74 = true;
  planned.operation = "preproc";
  planned.input = request.inputs.front();
  planned.output = request.outputs.front();
  planned.parameters = std::move(request.parameters);
  planned.input_bytes = contract.input_requirements().front().storage_bytes;
  planned.output_bytes = contract.output_requirements().front().storage_bytes;
  route.input_bytes = planned.input_bytes;
  route.input_dtype = *in_dtype;
  route.input_shape = planned.input.shape;
  route.kernel_input = true;
  route.ev74_input = true;
  route.frame_input = true;
  try {
    // What GStreamer's pull leaves on every tensor behind this stage, for this frame size.
    route.output.preprocess = preprocess_from_compact_facts(*facts);
    route.output.frame_semantics = true;
  } catch (const std::exception& error) {
    return name + ": " + error.what();
  }
  const si::LogicalTensorStaticSpec& out = stage.logical_outputs.front();
  if (last) {
    const std::optional<TensorDType> published = dtype_of(out.dtype);
    if (!published || *published != *out_dtype) {
      return name + ": the published tensor's dtype '" + out.dtype + "' is not the kernel's";
    }
    if (out.byte_offset != 0 || out.shape.empty() ||
        (out.size_bytes != 0U && out.size_bytes != planned.output_bytes)) {
      return name + ": the published output (" + std::to_string(out.size_bytes) + " bytes at " +
             std::to_string(out.byte_offset) + ") is not the kernel's whole output buffer of " +
             std::to_string(planned.output_bytes) + " bytes";
    }
    describe_pulled(stage, out, *out_dtype, route.output);
    route.output.sample_label =
        payload.primary_output_name.empty() ? route.output.stage_key : payload.primary_output_name;
    if (const auto* tiled = std::get_if<client::MlaTiled>(&planned.output.layout)) {
      // The publication of a tiled tensor as its packed storage (StageRecipePreparation.cpp
      // apply_kernel_publication_views): no caps format, the checked tile description, and no
      // strides on the pulled tensor whatever the manifest's logical output states (a tiled
      // tensor is read through its storage).
      const char* const dtype = planned.output.dtype == client::DType::BFloat16 ? "BF16"
                                : planned.output.dtype == client::DType::Int8   ? "INT8"
                                : planned.output.dtype == client::DType::UInt8  ? "UINT8"
                                                                                : nullptr;
      if (!dtype || out.publication_relation != si::PublicationViewRelation::PackedStorage) {
        return name +
               " publishes its tiled output as another view than its packed storage "
               "(publication relation " +
               std::to_string(static_cast<int>(out.publication_relation)) +
               "), which is not supported on Dataflow yet";
      }
      TessSpec tess;
      tess.format =
          std::string("MLA_TILED_") + dtype +
          (tiled->channels == client::MlaTiled::Channels::Compact ? "_COMPACT" : "_PAD16");
      tess.slice_shape.assign(tiled->tile_shape.begin(), tiled->tile_shape.end());
      tess.axes.assign(tiled->axes.begin(), tiled->axes.end());
      tess.storage_bytes = planned.output_bytes;
      route.output.tess = std::move(tess);
      route.output.format.clear();
      route.output.strides_bytes.clear();
    } else {
      route.output.format = kernel_caps_format_of(out.dtype, payload.output_dtype);
    }
  }
  route.text +=
      "kernel stage " + planned.key + " (preproc on the EV74 for " + std::to_string(frame.width) +
      "x" + std::to_string(frame.height) + " " + format + " frames, " + frame.from + "; " +
      parameters_text(planned.parameters) + "): " + std::to_string(planned.input_bytes) + " -> " +
      std::to_string(planned.output_bytes) + " bytes" +
      (last ? ", output " + out.dtype + " " + dims_text(out.shape) + " " +
                  std::to_string(out.size_bytes) + " bytes at " + std::to_string(out.byte_offset)
            : std::string());
  route.stages.push_back(std::move(planned));
  return {};
}

#endif // SIMANEAT_WITH_DATAFLOW_SOC_KERNELS

// One kernel stage of the manifest (its checked recipe, as prepare_kernel_stage_recipes left it)
// as a route stage; "" or why this backend cannot carry it on this step.
std::string plan_kernel_stage(const si::StageStaticSpec& stage, bool first, bool last,
                              const InputOptions& declared, std::span<const Tensor> seed,
                              DataflowModelRoute& route) {
  const si::KernelStageRecipe& recipe = *stage.native_kernel;
  const std::string key = stage_key_of(stage);
  const std::string name = "kernel stage " + key + " (" + stage_kind_text(stage) + ")";
#if !defined(SIMANEAT_WITH_DATAFLOW_SOC_KERNELS)
  (void)recipe;
  (void)first;
  (void)last;
  (void)declared;
  (void)seed;
  (void)route;
  return name + ": this build of libsima_neat has no kernel operator (dataflow::soc was built "
                "without the Runner's Executor and the Kernels package)";
#else
  const bool ev74 = recipe.target != client::Hardware::A65;
  if (ev74) {
    // A board whose EV74 runs the stock firmware keeps the route on GStreamer, as before this
    // backend ran a kernel on the EV74 at all. Never the A65 kernel in the EV74's place: the
    // placement is the Graph's (ProcessCvuRunTargetPolicy.cpp), not this backend's.
    // The runtime's own condition (soc::KernelOp refuses the EV74 target on it at instantiate),
    // asked here because an input may be bound, and its Session instantiated, only by its first
    // push: by then the Run is on this backend and cannot go back to GStreamer. Cold: sysfs is
    // read; nothing is opened on the device.
    if (std::string obstacle; !df::soc::ev74_runs_kernels_firmware(&obstacle)) {
      return name + " is placed on the EV74 (processcvu run target requested '" +
             stage.processcvu.requested_run_target + "', resolved '" +
             stage.processcvu.resolved_exec_backend +
             "'): the Dataflow backend runs an EV74 kernel stage on the Kernels firmware "
             "(sima-neat-native-cvu-fw) only, and " +
             obstacle;
    }
  }
  if (recipe.frame_adapter_factory || recipe.image_publication || stage.image_publication) {
    return plan_frame_stage(stage, name, first, last, declared, seed, route);
  }
  if (recipe.members.size() != 1U) {
    return name + " runs " + std::to_string(recipe.members.size()) +
           " kernel invocations per frame (a batched model or several tensors), which is not "
           "supported on Dataflow yet";
  }
  if (recipe.requests.size() != recipe.members.size()) {
    return name + " has no recorded kernel request (not a canonical transform), which is not "
                  "supported on Dataflow yet";
  }
  if (!recipe.resolver_factories.empty() || !recipe.constant_data.empty() ||
      !recipe.private_buffers.empty() || !recipe.private_outputs.empty()) {
    return name + " has per-frame variants, constants or private buffers, which is not supported "
                  "on Dataflow yet";
  }
  const client::TensorStageMember& member = recipe.members.front();
  const si::KernelStageRecipe::Request& request = recipe.requests.front();
  if (member.inputs.size() != 1U || member.outputs.size() != 1U || request.inputs.size() != 1U ||
      request.outputs.size() != 1U) {
    return name + " has " + std::to_string(member.inputs.size()) + " inputs and " +
           std::to_string(member.outputs.size()) +
           " outputs: a kernel of several inputs or outputs is not supported on Dataflow yet (the "
           "runtime has no join)";
  }
  if (member.inputs.front().byte_offset != 0U || member.outputs.front().byte_offset != 0U ||
      member.inputs.front().source != client::TensorStagePortSource::Input ||
      member.outputs.front().source != client::TensorStagePortSource::Output ||
      !member.dependencies.empty()) {
    return name + " reads or writes inside a shared buffer (an offset view), which is not "
                  "supported on Dataflow yet";
  }
  if (stage.logical_inputs.size() > 1U || stage.physical_inputs.size() > 1U ||
      stage.logical_outputs.size() != 1U || stage.input_bindings.size() > 1U) {
    return name + " has " + std::to_string(stage.logical_inputs.size()) + " input tensors and " +
           std::to_string(stage.logical_outputs.size()) +
           " published tensors, which is not supported on Dataflow yet";
  }
  auto inspected = client::inspect_kernel(member.executable);
  if (const auto* error = std::get_if<client::Error>(&inspected)) {
    return name + ": the Kernels package gave no contract: " + error->message;
  }
  const client::KernelInspection& contract = std::get<client::KernelInspection>(inspected);
  if (contract.input_requirements().size() != 1U || contract.output_requirements().size() != 1U) {
    return name + ": the kernel's contract has several ports";
  }
  if (contract.workspace().storage_bytes != 0U) {
    return name + " needs a workspace of " + std::to_string(contract.workspace().storage_bytes) +
           " bytes, which is not supported on Dataflow yet";
  }
  if (contract.requires_host_tasks()) {
    return name + " needs host tasks, which is not supported on Dataflow yet";
  }
  if (contract.input_extent(0).mode != client::InputExtentMode::Fixed) {
    return name + " reads an input of varying extent, which is not supported on Dataflow yet";
  }
  const std::optional<TensorDType> in_dtype = dtype_of(request.inputs.front().dtype);
  const std::optional<TensorDType> out_dtype = dtype_of(request.outputs.front().dtype);
  if (!in_dtype || !out_dtype) {
    return name + ": a tensor's dtype has no Core dtype";
  }
  DataflowModelStage planned;
  planned.kind = DataflowModelStage::Kind::Kernel;
  planned.key = key;
  planned.ev74 = ev74;
  planned.operation = request.operation;
  planned.input = request.inputs.front();
  planned.output = request.outputs.front();
  planned.parameters = request.parameters;
  planned.input_bytes = contract.input_requirements().front().storage_bytes;
  planned.output_bytes = contract.output_requirements().front().storage_bytes;
  if (first) {
    if (!std::holds_alternative<client::Contiguous>(planned.input.layout)) {
      return name + " reads a strided or tiled tensor from the application, which is not "
                    "supported on Dataflow yet";
    }
    route.input_bytes = planned.input_bytes;
    route.input_dtype = *in_dtype;
    route.input_shape = planned.input.shape;
    route.kernel_input = true;
    route.ev74_input = ev74;
  }
  const si::LogicalTensorStaticSpec& out = stage.logical_outputs.front();
  if (last) {
    const std::optional<TensorDType> published = dtype_of(out.dtype);
    if (!published || *published != *out_dtype) {
      return name + ": the published tensor's dtype '" + out.dtype + "' is not the kernel's";
    }
    if (out.byte_offset < 0 || out.size_bytes == 0U || out.shape.empty() ||
        static_cast<std::uint64_t>(out.byte_offset) > planned.output_bytes ||
        out.size_bytes > planned.output_bytes - static_cast<std::uint64_t>(out.byte_offset)) {
      return name + ": the published output does not lie in the kernel's output buffer";
    }
    describe_pulled(stage, out, *out_dtype, route.output);
    route.output.format = kernel_caps_format_of(out.dtype, stage.processcvu.output_dtype);
    // TensorStagePublication::output_name (finalize_processcvu_publication).
    route.output.sample_label = stage.processcvu.primary_output_name.empty()
                                    ? route.output.stage_key
                                    : stage.processcvu.primary_output_name;
  }
  route.text +=
      std::string(route.text.empty() ? "" : " -> ") + "kernel stage " + key + " (" +
      planned.operation + " on the " + target_text(recipe.target) +
      (planned.parameters.empty() ? std::string() : "; " + parameters_text(planned.parameters)) +
      "): " + std::to_string(planned.input_bytes) + " -> " + std::to_string(planned.output_bytes) +
      " bytes" +
      (last ? ", output " + out.dtype + " " + dims_text(out.shape) + " " +
                  std::to_string(out.size_bytes) + " bytes at " + std::to_string(out.byte_offset)
            : std::string());
  route.stages.push_back(std::move(planned));
  return {};
#endif
}

} // namespace

std::shared_ptr<const DataflowModelRoute>
plan_dataflow_model_route(const std::vector<std::shared_ptr<Node>>& route,
                          const GraphOptions& options, const InputOptions& declared,
                          std::span<const Tensor> seed, std::string* why) {
  const auto decline = [why](std::string reason) {
    *why = std::move(reason);
    return nullptr;
  };
  // The contracts of the route's nodes, compiled and rendered as GStreamer's build compiles them
  // (GraphBuildInput.cpp maybe_compile_build_result_contracts): the Graph's processcvu options
  // decide each kernel stage's target (ProcessCvuRunTargetPolicy.cpp).
  ContractCompileInput input;
  input.pipeline_label = "DataflowModelRoute";
  input.processcvu_requested_run_target = options.processcvu_requested_run_target;
  input.processcvu = options.processcvu;
  si::ManifestBuildDiagnostics diagnostics;
  std::optional<si::SimaPluginStaticManifest> manifest;
  try {
    manifest = pipeline_internal::rendered_stage_query::rendered_manifest_from_nodes(
        std::span<const std::shared_ptr<Node>>(route), input, &diagnostics);
  } catch (const std::exception& error) {
    return decline(std::string("the model's contracts do not compile: ") + error.what());
  }
  if (!manifest) {
    return decline(
        "the model's contracts do not compile" +
        (diagnostics.errors.empty() ? std::string() : ": " + diagnostics.errors.front()));
  }
  if (manifest->stages.empty()) {
    return decline("the model's nodes render no stage");
  }
  // The kernel stages' checked recipes, by the function GStreamer's build lowers them with
  // (GraphBuildPipeline.cpp): the canonical operation, its tensors and parameters, checked by the
  // Kernels package for the stage's target. Cold; no device is touched.
  const bool kernels = std::any_of(manifest->stages.begin(), manifest->stages.end(),
                                   [](const si::StageStaticSpec& stage) {
                                     return stage.payload_kind != si::StagePayloadKind::ProcessMla;
                                   });
  if (kernels) {
    try {
      const std::optional<si::SimaPluginStaticManifest> original = manifest;
      if (auto failure =
              si::prepare_kernel_stage_recipes(&*manifest, original, {}, {}, NameTransform{})) {
        return decline("the model's kernel stages do not lower: " + failure->field + ": " +
                       failure->message);
      }
    } catch (const std::exception& error) {
      return decline(std::string("the model's kernel stages do not lower: ") + error.what());
    }
  }
  std::string kinds;
  for (const si::StageStaticSpec& stage : manifest->stages) {
    kinds += (kinds.empty() ? "" : ", ") + stage_kind_text(stage);
  }
  auto planned = std::make_shared<DataflowModelRoute>();
  std::size_t mla_stages = 0;
  for (std::size_t i = 0; i < manifest->stages.size(); ++i) {
    const si::StageStaticSpec& stage = manifest->stages[i];
    const bool first = i == 0U;
    const bool last = i + 1U == manifest->stages.size();
    std::string reason;
    if (stage.payload_kind == si::StagePayloadKind::ProcessMla) {
      ++mla_stages;
      reason = plan_mla_stage(stage, first, last, *planned);
    } else if (stage.native_kernel) {
      if (!first && planned->stages.back().kind == DataflowModelStage::Kind::Kernel) {
        reason = "two kernel stages in a row (" + kinds + ") are not supported on Dataflow yet";
      } else {
        reason = plan_kernel_stage(stage, first, last, declared, seed, *planned);
      }
    } else {
      reason = "stage " + stage_key_of(stage) + " (" + stage_kind_text(stage) +
               ") is neither an MLA stage nor a kernel stage";
    }
    if (!reason.empty()) {
      return decline("a model route of " + std::to_string(manifest->stages.size()) + " stages (" +
                     kinds + "): " + reason);
    }
  }
  // Exactly one MLA stage, or the image preproc alone (Model::preprocess() between the Input and
  // the Output).
  if (mla_stages != 1U &&
      !(mla_stages == 0U && planned->frame_input && planned->stages.size() == 1U)) {
    return decline("a model route of " + std::to_string(manifest->stages.size()) + " stages (" +
                   kinds + ") with " + std::to_string(mla_stages) +
                   " MLA stages is not supported on Dataflow yet: exactly one runs");
  }
  for (std::size_t i = 0; i + 1U < planned->stages.size(); ++i) {
    if (planned->stages[i].output_bytes != planned->stages[i + 1U].input_bytes) {
      return decline("stage " + planned->stages[i].key + " writes " +
                     std::to_string(planned->stages[i].output_bytes) + " bytes and stage " +
                     planned->stages[i + 1U].key + " reads " +
                     std::to_string(planned->stages[i + 1U].input_bytes) +
                     ": not one whole buffer handed on, which is not supported on Dataflow yet");
    }
  }
  return planned;
}

std::string bind_dataflow_model_input(const DataflowModelRoute& route,
                                      DataflowInputBinding& binding) {
  // The first stage's input: the MLA's (an MLA-only route), or the model's public tensor.
  const std::string input = route.kernel_input ? "the model's input" : "the model's MLA input";
  const std::string expected = input + " is " + dims_text(route.input_shape) + " of " +
                               std::to_string(route.input_bytes) + " bytes";
  if (binding.is_record() || binding.planes != DataflowInputBinding::Planes::None) {
    return "an item of several tensors or planes where one dense tensor is expected; " + expected;
  }
  if (binding.spec.memory == df::MemoryKind::Host) {
    return "an input in system memory (InputOptions::memory_policy SystemMemory, or "
           "use_simaai_pool = false) in front of a model: the MLA takes device memory";
  }
  if (binding.dtype != route.input_dtype) {
    return "dtype differs from " + input + "; " + expected;
  }
  if (route.frame_input) {
    // In front of the image preproc the port is the frame the route was built for (the kernel's
    // tensor), wherever in the declaration's bounds that is: the item's shape must admit it,
    // with or without the kernel's leading batch axis.
    auto* spec = std::get_if<df::TensorSpec>(&binding.spec.payload);
    const std::vector<std::int64_t>& want = route.input_shape;
    bool fits = spec && std::holds_alternative<df::Dense>(spec->layout) && spec->shape.rank != 0 &&
                spec->shape.rank <= want.size();
    const std::size_t lead = fits ? want.size() - spec->shape.rank : 0U;
    for (std::size_t i = 0; fits && i < want.size(); ++i) {
      fits = i < lead ? want[i] == 1 : spec->shape.dims[i - lead].contains(want[i]);
    }
    if (!fits) {
      return "a tensor that is not a frame of the size the Run was built for; " + expected;
    }
    spec->shape.rank = static_cast<std::uint8_t>(want.size());
    for (std::size_t i = 0; i < want.size(); ++i) {
      spec->shape.dims[i] = df::Dim(want[i]);
    }
    if (const df::Status status = df::validate(binding.spec)) {
      return std::string("the input's spec is not valid: ") + status->message;
    }
    binding.rank = want.size();
    std::array<char, df::kDescribeMax> text{};
    df::Writer writer{std::span<char>(text)};
    df::describe(binding.spec, writer);
    binding.text = std::string(writer.view());
  } else if (const std::string why = fix_dataflow_binding_at_bound(binding); !why.empty()) {
    // A bounded axis (InputOptions::max_*) is fixed at its bound: the model's input has one size.
    return why;
  }
  if (df::slot_bytes(binding.spec) != route.input_bytes) {
    return "a tensor of " + std::to_string(df::slot_bytes(binding.spec)) + " bytes; " + expected;
  }
  if (route.ev74_input && binding.spec.memory != df::MemoryKind::Ev74) {
    // The compiler gives such an input an Ev74 pool whatever was declared
    // (DataflowGraphCompiler.cpp).
    return "an input outside EV74 memory in front of a kernel stage on the EV74, which reaches "
           "linux,cma memory only";
  }
  if (route.kernel_input) {
    // The kernel's port is the kernel's tensor: the Feed's spec must be exactly it (the push
    // brings a tensor without the leading unit axes to it, dataflow_model_input_view).
    const auto* spec = std::get_if<df::TensorSpec>(&binding.spec.payload);
    bool same = spec && spec->shape.rank == route.input_shape.size() &&
                std::holds_alternative<df::Dense>(spec->layout);
    for (std::size_t i = 0; same && i < route.input_shape.size(); ++i) {
      same = spec->shape.dims[i].max == route.input_shape[i];
    }
    if (!same) {
      return "a tensor whose shape or layout is not the model's first stage's; " + expected;
    }
  }
  binding.pulled = &route.output;
  return {};
}

void apply_dataflow_frame_semantics(const DataflowPulledTensor& pulled, Tensor& tensor) {
  if (pulled.tess) {
    tensor.semantic.tess = pulled.tess;
  }
  if (pulled.preprocess) {
    tensor.semantic.preprocess = pulled.preprocess;
  }
}

void refuse_dataflow_frame(const DataflowModelRoute& route, const Tensor& tensor,
                           const char* where) {
  if (tensor.semantic.preprocess && tensor.semantic.preprocess->has_roi_list()) {
    pipeline_internal::error_util::throw_session_error(
        error_codes::kInputShape,
        std::string(where) +
            ": a frame with ROI metadata (Semantic::preprocess with an ROI or an ROI list) in "
            "front of the image preproc stage is not supported on the Dataflow backend yet; build "
            "this Run with RunBackend::GStreamer");
  }
  const std::vector<std::int64_t>& want = route.input_shape;
  const std::size_t rank = tensor.shape.size();
  if (tensor.is_composite() || want.size() != 4U || (rank != 3U && rank != 4U) ||
      (rank == 4U && tensor.shape[0] != 1)) {
    return;
  }
  const std::size_t lead = rank - 3U;
  if (tensor.shape[lead] != want[1] || tensor.shape[lead + 1U] != want[2]) {
    pipeline_internal::error_util::throw_session_error(
        error_codes::kInputShape,
        std::string(where) + ": a frame of " + std::to_string(tensor.shape[lead + 1U]) + "x" +
            std::to_string(tensor.shape[lead]) + " where the Run was built for " +
            std::to_string(want[2]) + "x" + std::to_string(want[1]) +
            ": on the Dataflow backend the image preproc stage runs at the frame size the input "
            "had at the build (its seed or its declaration); a size that changes between pushes "
            "needs RunBackend::GStreamer");
  }
}

const Tensor& dataflow_model_input_view(const DataflowModelRoute& route, const Tensor& tensor,
                                        Tensor& view) {
  const std::vector<std::int64_t>& want = route.input_shape;
  if (!route.kernel_input || tensor.shape.size() >= want.size() || tensor.shape.empty() ||
      !tensor.is_dense() || !tensor.is_contiguous()) {
    return tensor;
  }
  const std::size_t lead = want.size() - tensor.shape.size();
  for (std::size_t i = 0; i < want.size(); ++i) {
    if (i < lead ? want[i] != 1 : want[i] != tensor.shape[i - lead]) {
      return tensor;
    }
  }
  view = tensor;
  view.shape = want;
  std::int64_t stride = dtype_size(tensor.dtype);
  view.strides_bytes.assign(want.size(), 0);
  for (std::size_t i = want.size(); i-- > 0U;) {
    view.strides_bytes[i] = stride;
    stride *= std::max<std::int64_t>(want[i], 1);
  }
  if (!view.axis_semantics.empty()) {
    view.axis_semantics.clear();
  }
  return view;
}

#if defined(SIMANEAT_WITH_DATAFLOW_SOC)

namespace {

// One Run's model and kernels, in stage order (one of the two per stage).
struct LoadedRoute {
  std::vector<std::shared_ptr<df::soc::MlaModel>> models;
#if defined(SIMANEAT_WITH_DATAFLOW_SOC_KERNELS)
  std::vector<std::shared_ptr<df::soc::Kernel>> kernels;
#endif
};

#if defined(SIMANEAT_WITH_DATAFLOW_SOC_KERNELS)

static_assert(static_cast<int>(df::DType::Float32) == static_cast<int>(client::DType::Float32) &&
              static_cast<int>(df::DType::BFloat16) == static_cast<int>(client::DType::BFloat16) &&
              static_cast<int>(df::DType::Int8) == static_cast<int>(client::DType::Int8) &&
              static_cast<int>(df::DType::UInt8) == static_cast<int>(client::DType::UInt8) &&
              static_cast<int>(df::DType::Int16) == static_cast<int>(client::DType::Int16) &&
              static_cast<int>(df::DType::Int32) == static_cast<int>(client::DType::Int32) &&
              static_cast<int>(df::DType::Float16) == static_cast<int>(client::DType::Float16) &&
              static_cast<int>(df::DType::UInt16) == static_cast<int>(client::DType::UInt16));

// A kernel request's tensor (the Scheduler client's vocabulary) in the runtime's, which mirrors
// it. An MLA layout's bytes and alignment are left for the kernel to give (soc::Kernel::make); its
// format token, which only names the layout in messages, is the dtype's.
std::optional<df::TensorSpec> runtime_tensor(const client::TensorSpec& tensor) {
  if (tensor.shape.empty() || tensor.shape.size() > df::kMaxRank) {
    return std::nullopt;
  }
  df::TensorSpec out;
  out.dtype = static_cast<df::DType>(static_cast<int>(tensor.dtype));
  out.shape.rank = static_cast<std::uint8_t>(tensor.shape.size());
  for (std::size_t i = 0; i < tensor.shape.size(); ++i) {
    if (tensor.shape[i] <= 0) {
      return std::nullopt;
    }
    out.shape.dims[i] = df::Dim(tensor.shape[i]);
  }
  const std::optional<df::FormatToken> token = df::FormatToken::of(df::dtype_name(out.dtype));
  if (std::holds_alternative<client::Contiguous>(tensor.layout)) {
    out.layout = df::Dense{};
  } else if (const auto* strided = std::get_if<client::Strided>(&tensor.layout)) {
    df::Strided layout;
    for (std::size_t i = 0; i < tensor.shape.size(); ++i) {
      layout.stride_bytes[i] = strided->strides_bytes[i];
    }
    out.layout = layout;
  } else if (const auto* tiled = std::get_if<client::MlaTiled>(&tensor.layout)) {
    df::MlaTiled layout;
    layout.tile_shape = tiled->tile_shape;
    layout.axes = tiled->axes;
    layout.channels = tiled->channels == client::MlaTiled::Channels::PadTo16
                          ? df::MlaTiled::Channels::PadTo16
                          : df::MlaTiled::Channels::Compact;
    layout.format = *token;
    out.layout = layout;
  } else if (const auto* blocked = std::get_if<client::MlaCBlocked>(&tensor.layout)) {
    df::MlaCBlocked layout;
    layout.tile_shape = blocked->tile_shape;
    layout.format = *token;
    out.layout = layout;
  } else {
    return std::nullopt;
  }
  return out;
}

df::soc::KernelParameter::Scalar runtime_scalar(const client::ParameterScalar& scalar) {
  return std::visit([](const auto& value) { return df::soc::KernelParameter::Scalar(value); },
                    scalar.value);
}

df::soc::KernelParameter runtime_parameter(const client::Parameter& parameter) {
  df::soc::KernelParameter out;
  out.name = parameter.name;
  if (const auto* scalar = std::get_if<client::ParameterScalar>(&parameter.value.value)) {
    out.value = runtime_scalar(*scalar);
  } else if (const auto* array = std::get_if<client::ParameterArray>(&parameter.value.value)) {
    std::vector<df::soc::KernelParameter::Scalar> values;
    for (const client::ParameterScalar& value : *array) {
      values.push_back(runtime_scalar(value));
    }
    out.value = std::move(values);
  } else {
    std::vector<df::soc::KernelParameter::Record> records;
    for (const client::ParameterRecord& record :
         std::get<client::ParameterRecords>(parameter.value.value)) {
      df::soc::KernelParameter::Record fields;
      for (const client::ScalarParameter& field : record) {
        fields.emplace_back(field.name, runtime_scalar(field.value));
      }
      records.push_back(std::move(fields));
    }
    out.value = std::move(records);
  }
  return out;
}

std::shared_ptr<df::soc::Kernel> make_kernel(const DataflowModelStage& stage, const char* where) {
  const std::optional<df::TensorSpec> input = runtime_tensor(stage.input);
  const std::optional<df::TensorSpec> output = runtime_tensor(stage.output);
  if (!input || !output) {
    pipeline_internal::error_util::throw_session_error(
        error_codes::kPipelineShape,
        std::string(where) + ": kernel stage " + stage.key + " (" + stage.operation +
            ") has a tensor the Dataflow runtime does not describe (Dataflow backend)");
  }
  std::vector<df::soc::KernelParameter> parameters;
  for (const client::Parameter& parameter : stage.parameters) {
    parameters.push_back(runtime_parameter(parameter));
  }
  df::Result<std::shared_ptr<df::soc::Kernel>> made = df::soc::Kernel::make(
      stage.operation, stage.ev74 ? df::soc::Kernel::Target::Ev74 : df::soc::Kernel::Target::A65,
      {&*input, 1}, {&*output, 1}, parameters);
  if (const df::Error* error = df::failure(made)) {
    pipeline_internal::throw_dataflow_error(
        *error, where, "kernel stage " + stage.key + " (" + stage.operation + ")");
  }
  std::shared_ptr<df::soc::Kernel> kernel =
      std::get<std::shared_ptr<df::soc::Kernel>>(std::move(made));
  if (kernel->inputs().size() != 1U || kernel->outputs().size() != 1U ||
      df::slot_bytes(kernel->inputs().front()) != stage.input_bytes ||
      df::slot_bytes(kernel->outputs().front()) != stage.output_bytes) {
    pipeline_internal::error_util::throw_session_error(
        error_codes::kPipelineShape,
        std::string(where) + ": kernel stage " + stage.key + " (" + stage.operation +
            ") does not have the " + std::to_string(stage.input_bytes) + " and " +
            std::to_string(stage.output_bytes) +
            " bytes of the stage Core lowered for GStreamer (Dataflow backend)");
  }
  return kernel;
}

#endif // SIMANEAT_WITH_DATAFLOW_SOC_KERNELS

std::shared_ptr<df::soc::MlaModel> load_model(const DataflowModelStage& stage, const char* where) {
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

} // namespace

std::shared_ptr<void> load_dataflow_model_route(const DataflowModelRoute& route,
                                                const char* where) {
  auto loaded = std::make_shared<LoadedRoute>();
  loaded->models.resize(route.stages.size());
#if defined(SIMANEAT_WITH_DATAFLOW_SOC_KERNELS)
  loaded->kernels.resize(route.stages.size());
#endif
  for (std::size_t i = 0; i < route.stages.size(); ++i) {
    const DataflowModelStage& stage = route.stages[i];
    if (stage.kind == DataflowModelStage::Kind::Mla) {
      loaded->models[i] = load_model(stage, where);
      continue;
    }
#if defined(SIMANEAT_WITH_DATAFLOW_SOC_KERNELS)
    loaded->kernels[i] = make_kernel(stage, where);
#else
    pipeline_internal::error_util::throw_session_error(
        error_codes::kPipelineShape,
        std::string(where) + ": this build of libsima_neat has no kernel operator (dataflow::soc)");
#endif
  }
  return loaded;
}

std::vector<std::uint32_t> add_dataflow_model_ops(df::Net& net, const DataflowModelRoute& route,
                                                  const std::shared_ptr<void>& loaded_route,
                                                  const df::PortSpec& input, df::PortSpec& output) {
  const auto loaded = std::static_pointer_cast<LoadedRoute>(loaded_route);
  std::vector<std::uint32_t> operators;
  // What the previous operator's output port carries, which the next one's input port must be.
  df::PortSpec carried = input;
  for (std::size_t i = 0; i < route.stages.size(); ++i) {
    const DataflowModelStage& stage = route.stages[i];
    const bool last = i + 1U == route.stages.size();
    if (stage.kind == DataflowModelStage::Kind::Mla) {
      const std::shared_ptr<df::soc::MlaModel>& model = loaded->models[i];
      df::soc::MlaOp::Options options;
      options.depth = stage.depth;
      options.inputs = {carried};
      // The output port: the next kernel's tensor (the MLA's bytes, typed as the kernel reads
      // them), or, at the route's end, the MLA's output buffer as it is (bytes; the pulled tensor
      // is a view of it, DataflowPulledTensor). In the MLA's own memory, except in front of a
      // kernel on the EV74, which reaches linux,cma only: there the MLA writes Ev74 memory (the
      // kernel's own port spec) and the EV74 reads it by descriptor.
      carried = model->outputs().front();
#if defined(SIMANEAT_WITH_DATAFLOW_SOC_KERNELS)
      if (!last) {
        carried = loaded->kernels[i + 1U]->inputs().front();
        if (!route.stages[i + 1U].ev74) {
          carried.memory = df::MemoryKind::Dms0;
        }
      }
#endif
      options.outputs = {carried};
      operators.push_back(net.add<df::soc::MlaOp>("mla", model, std::move(options)).index);
      continue;
    }
#if defined(SIMANEAT_WITH_DATAFLOW_SOC_KERNELS)
    // An A65 kernel reads its input through the CPU's address, in whatever memory the previous
    // operator (or the Feed) publishes, and writes its output into device memory: the MLA takes
    // it by descriptor, and at the route's end Core's pull hands a DMA-BUF out, as GStreamer's
    // appsink does. The write epoch of that output is the runtime's (a job-written port).
    // A kernel on the EV74 has its ports in Ev74 memory by itself (Options{}): it reads and
    // writes by descriptor, and the Plan settles the Feed's or the MLA's port in front of it and
    // the MLA's or the Fetch's behind it to that memory.
    const std::shared_ptr<df::soc::Kernel>& kernel = loaded->kernels[i];
    df::soc::KernelOp::Options options;
    if (!stage.ev74) {
      options.inputs = {carried.memory};
      options.outputs = {df::MemoryKind::Dms0};
    }
    carried = kernel->outputs().front();
    if (!stage.ev74) {
      carried.memory = df::MemoryKind::Dms0;
    }
    operators.push_back(net.add<df::soc::KernelOp>(stage.key, kernel, std::move(options)).index);
#endif
    (void)last;
  }
  output = carried;
  return operators;
}

#else

std::shared_ptr<void> load_dataflow_model_route(const DataflowModelRoute&, const char* where) {
  pipeline_internal::error_util::throw_session_error(
      error_codes::kPipelineShape,
      std::string(where) + ": this build of libsima_neat has no MLA operator (dataflow::soc)");
}

std::vector<std::uint32_t> add_dataflow_model_ops(df::Net&, const DataflowModelRoute&,
                                                  const std::shared_ptr<void>&, const df::PortSpec&,
                                                  df::PortSpec&) {
  pipeline_internal::error_util::throw_session_error(
      error_codes::kPipelineShape,
      "Graph::build: this build of libsima_neat has no MLA operator (dataflow::soc)");
}

#endif // SIMANEAT_WITH_DATAFLOW_SOC

} // namespace simaai::neat::runtime

#endif // SIMANEAT_WITH_DATAFLOW
