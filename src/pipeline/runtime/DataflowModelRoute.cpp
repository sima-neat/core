// A model route on the Dataflow backend (DataflowModelRoute.h).
#if defined(SIMANEAT_WITH_DATAFLOW)

#include "pipeline/runtime/DataflowModelRoute.h"

#include "pipeline/DetectionTypes.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/internal/CapsText.h"
#include "pipeline/internal/DataflowErrorMap.h"
#include "pipeline/internal/DataflowNotCarried.h"
#include "pipeline/internal/ErrorUtil.h"
#include "pipeline/internal/InputPolicy.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/internal/PipelineBuild.h"
#include "pipeline/internal/RenderedMlaContractQuery.h"
#include "pipeline/internal/SampleUtil.h"
#include "pipeline/internal/UxLogging.h"
#include "pipeline/internal/sima/BoxDecodeGeometry.h"
#include "pipeline/internal/sima/KernelStageRecipe.h"
#include "pipeline/internal/sima/PreprocFrameAdapter.h"
#include "pipeline/internal/sima/StageRecipePreparation.h"
#include "pipeline/runtime/DataflowDeviceMemory.h"
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
#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <optional>
#include <span>
#include <string_view>
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

// A slot's alignment in the runtime's pools: every alignment a kernel may ask of a tensor that is a
// view of a port's buffer (soc::KernelOp::Options::Packing) divides it, and BoxDecode's outputs
// lie at multiples of it in their one buffer (the layout GStreamer's stage arena gives them).
constexpr std::size_t kSlotAlignment = 64;

std::size_t aligned(std::size_t bytes) {
  return (bytes + kSlotAlignment - 1U) & ~(kSlotAlignment - 1U);
}

// The MLA's buffers: the model compiler makes each input and output buffer of a model a multiple
// of this many bytes, so a buffer is up to 15 bytes longer than the tensor in it
// (DataflowModelRoute.h, "The MLA's padding").
constexpr std::uint64_t kMlaBufferGranule = 16;

// Whether an MLA input buffer of `buffer` bytes is a tensor of `tensor` bytes with the MLA's
// padding behind it: longer than the tensor, and the tensor's bytes rounded up to the granule.
bool mla_pads(std::uint64_t tensor, std::uint64_t buffer) {
  return tensor != 0U && tensor < buffer &&
         (tensor + kMlaBufferGranule - 1U) / kMlaBufferGranule * kMlaBufferGranule == buffer;
}

// The axes of an image stage's frame tensor, [1, height, width, channels].
constexpr std::size_t kHeightAxis = 1, kWidthAxis = 2;

// The channels of a BGR or RGB frame when nothing the build says gives them.
constexpr std::int64_t kFrameChannels = 3;

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

#if defined(SIMANEAT_WITH_DATAFLOW_SOC_KERNELS)

// The frames of a route whose image preproc takes a size that changes between pushes
// (DataflowModelRoute.h).
//
// The stamp's user word of a frame states its geometry: bit 63, the width in bits 32..62, the
// height in bits 0..31 (frame_word). Bit 63 keeps Core's reading of the word as "pushed sequence
// number plus one" (DataflowInputPort::orig_input_seq) below zero, which is "none": a model
// route is a Run of one input, whose pushes leave the word to its backend. The runtime carries
// the word through every operator untouched (Stamp::user, derive()), so the BoxDecode operator's
// key function and the pull read from the message itself which frame it came from: there is no
// entry per frame, nothing to write before a push and nothing to retire when a frame is
// dropped, evicted or still in flight at a close.
//
// The table holds, per frame geometry, what Core derives from it: the preprocess facts and the
// two kernels. It is bounded (kKept geometries, the one used longest ago replaced), guarded by
// one mutex, and every entry is immutable and shared: a reader (the Worker inside a variant's
// make, a pull, a push) keeps what it read whatever is replaced meanwhile. An entry that was
// replaced while a frame of its geometry was in flight is made again from the word (cold). The
// Worker's key function reads the stamp alone: it takes no lock and allocates nothing.
//
// The two kernels of a geometry: the preproc's is made for the frame's size (its tensors are
// the frame's). The BoxDecode kernel is the Run's loaded one (`decode_kernel`, the port's
// bound's) with the frame's geometry in the variant the stage binds for it (Kernel::with): the
// original image size in "image_size", or, for oriented boxes, the size and the frame's
// model-to-original transform in "image_geometry". A variant keeps the kernel's tensors: the
// operator prepares it from one it keeps and runs it on the buffers it has admitted. A frame
// size whose letterbox has two different scales is one an oriented-box decode does not take: it
// is refused at its push (FrameNotCarried) and the Run goes on. The table starts with the bound's
// geometry, whose kernels the Run loaded (seed_dataflow_frames).
struct DataflowFrames {
  using Adapter = si::kernel_lowering::PreprocFrameAdapter;
  // What the plan fixed.
  std::shared_ptr<const Adapter::CheckedPolicy> policy; ///< the image stage's checked policy
  /// The BoxDecode stage's recipe when the route has one behind its MLA stage; else null.
  std::shared_ptr<const si::KernelStageRecipe> decode;
  /// The image stage as planned at the port's bound: a geometry's is this with its request.
  DataflowModelStage preproc_stage;
  /// What the image stage publishes with a frame of the bound's size.
  PreprocessRuntimeMeta bound_preprocess;
  std::int64_t max_width = 0, max_height = 0; ///< the port's bound
  /// The stage's capacity envelope. The bound is below it where the Input's capacity is (its
  /// declared bound, or the seed's size: GStreamer's rule, resolve_shape_limits).
  std::int64_t envelope_width = 0, envelope_height = 0;
  std::int64_t channels = 0; ///< of an RGB or BGR frame (0: NV12)
  bool nv12 = false;
  std::string format; ///< the frames' format, for messages

  struct Geometry {
    std::uint64_t word = 0;
    PreprocessRuntimeMeta preprocess; ///< what the pull publishes with a frame of it
    std::shared_ptr<df::soc::Kernel> preproc, decode;
  };
  /// As many geometries as a soc::KernelOp keeps variants: a rotation of frame sizes that the
  /// operators keep finds every size in this table too.
  static constexpr std::size_t kKept = 16;
  static constexpr std::uint64_t kFrameWord = std::uint64_t{1} << 63;
  static std::uint64_t frame_word(std::int64_t width, std::int64_t height) noexcept {
    return kFrameWord | (static_cast<std::uint64_t>(width) << 32) |
           static_cast<std::uint64_t>(height);
  }
  static std::int64_t width_of(std::uint64_t word) noexcept {
    return static_cast<std::int64_t>((word & ~kFrameWord) >> 32);
  }
  static std::int64_t height_of(std::uint64_t word) noexcept {
    return static_cast<std::int64_t>(word & 0xFFFFFFFFU);
  }

  mutable std::mutex mu;
  mutable std::array<std::shared_ptr<const Geometry>, kKept> kept;
  mutable std::array<std::uint64_t, kKept> used{};
  mutable std::uint64_t clock = 0;
  /// The Run's BoxDecode kernel, at the port's bound (the operator's own); null without a
  /// BoxDecode stage. Written once when the route is loaded (seed_dataflow_frames), before its
  /// Session exists and before any push.
  std::shared_ptr<df::soc::Kernel> decode_kernel;
  /// The word of the frame that passed dataflow_frame_word last: a push of the same geometry
  /// compares with it and asks nothing else.
  mutable std::atomic<std::uint64_t> passed{0};
};

#endif // SIMANEAT_WITH_DATAFLOW_SOC_KERNELS

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

// The EV74 firmware a kernel stage on the EV74 runs on (the Kernels package's).
constexpr std::string_view kKernelsFirmwareName = "sima-neat-native-cvu-fw";

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

// The stage's name: its name in the model package, with the prefix and suffix the Graph's creator
// set (plan_dataflow_model_route), as GStreamer's publication keys the stage
// (stage_key_from_stage_spec_local).
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

// A stage as an error names it: its place in the route, its name, its kind and target, its
// model: "stage 2 'detess_dequant_0' (kernel detessdequant on the EV74) of model yolo11n".
std::string stage_subject(std::size_t index, const si::StageStaticSpec& stage,
                          const std::string& model) {
  std::string kind;
  if (stage.payload_kind == si::StagePayloadKind::ProcessMla) {
    kind = "model on the MLA";
  } else if (stage.native_kernel) {
    const si::KernelStageRecipe& recipe = *stage.native_kernel;
    kind = "kernel " +
           (!recipe.requests.empty()               ? recipe.requests.front().operation
            : !stage.processcvu.graph_name.empty() ? stage.processcvu.graph_name
                                                   : stage.kernel_kind) +
           " on the " + target_text(recipe.target);
  } else {
    kind = stage.kernel_kind.empty() ? stage.plugin_kind : stage.kernel_kind;
  }
  return "stage " + std::to_string(index) + " '" + stage_key_of(stage) + "' (" + kind + ")" +
         (model.empty() ? std::string() : " of model " + model);
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

// One MLA stage of the manifest as a route stage; "" or the fact about it that this backend does
// not carry (DataflowNotCarried::fact, the stage its subject). `next`
// is the stage behind it, null when the MLA stage is the route's last. The model writes one output
// buffer; the tensors it publishes in that buffer may be several, each at its byte offset (a
// detection model's heads). Then the route ends here and the pull returns them as a list, or the
// stage behind is BoxDecode, whose kernel reads them as views of the one buffer.
std::string plan_mla_stage(const si::StageStaticSpec& stage, bool first,
                           const si::StageStaticSpec* next, DataflowModelRoute& route) {
  const bool last = next == nullptr;
  const si::ProcessMlaStagePayload& mla = stage.processmla;
  if (mla.model_path.empty()) {
    return "it names no model file";
  }
  if (mla.batch_size > 1 || mla.batch_sz_model > 1) {
    return "it is a batched MLA model (batch " +
           std::to_string(std::max(mla.batch_size, mla.batch_sz_model)) + ")";
  }
  if (stage.physical_inputs.size() != 1U || stage.logical_inputs.size() != 1U) {
    return "it reads " + std::to_string(stage.physical_inputs.size()) + " input buffers (" +
           std::to_string(stage.logical_inputs.size()) +
           " tensors): an MLA stage reads one input buffer of one tensor";
  }
  const std::size_t tensors = stage.logical_outputs.size();
  if (stage.physical_outputs.size() != 1U || tensors == 0U) {
    return "it writes " + std::to_string(stage.physical_outputs.size()) + " output buffers (" +
           std::to_string(tensors) +
           " tensors): an MLA stage writes one output buffer (its tensors may be several, each at "
           "its offset of that buffer)";
  }
  if (tensors > 1U && !last && !(next->native_kernel && next->native_kernel->boxdecode_program)) {
    return "it publishes " + std::to_string(tensors) +
           " tensors in its one output buffer and stage '" + stage_key_of(*next) + "' (" +
           stage_kind_text(*next) +
           ") reads them: behind an MLA stage of several tensors only BoxDecode runs (such a "
           "model also runs when Model::inference() is the route's last stage: the pull returns "
           "the tensors)";
  }
  const si::LogicalInputStaticSpec& in = stage.logical_inputs.front();
  const si::LogicalTensorStaticSpec& out = stage.logical_outputs.front();
  const si::PhysicalBufferStaticSpec& in_buffer = stage.physical_inputs.front();
  const si::PhysicalBufferStaticSpec& out_buffer = stage.physical_outputs.front();
  const std::optional<TensorDType> in_dtype = dtype_of(in.dtype);
  if (!in_dtype) {
    return "the dtype '" + in.dtype + "' of its input is not a tensor dtype of Core";
  }
  if (in.materialization_kind != si::TensorMaterializationKind::Direct || in.byte_offset != 0 ||
      in_buffer.size_bytes == 0U || in_buffer.source_byte_offset != 0) {
    return "the MLA stage's input is not one whole buffer (a repacked or offset view)";
  }
  for (std::size_t i = 0; i < tensors; ++i) {
    // Each published tensor: a view of the one output buffer, in the stage's logical order (the
    // order GStreamer's tensor-set descriptor lists them in and its pull returns them in).
    const si::LogicalTensorStaticSpec& tensor = stage.logical_outputs[i];
    const std::string which =
        tensors == 1U ? std::string("the MLA stage's published output")
                      : "the MLA stage's published tensor " + std::to_string(i) + " of " +
                            std::to_string(tensors) + " (" + tensor.logical_name + ")";
    if (!dtype_of(tensor.dtype)) {
      return "the dtype '" + tensor.dtype + "' of " + which + " is not a tensor dtype of Core";
    }
    if (tensor.byte_offset < 0 || tensor.size_bytes == 0U ||
        static_cast<std::uint64_t>(tensor.byte_offset) > out_buffer.size_bytes ||
        tensor.size_bytes >
            out_buffer.size_bytes - static_cast<std::uint64_t>(tensor.byte_offset) ||
        tensor.shape.empty()) {
      return which + " does not lie in its output buffer";
    }
    if (tensors > 1U && (tensor.physical_index > 0 || tensor.logical_index != static_cast<int>(i) ||
                         (i < stage.output_order.size() &&
                          (stage.output_order[i].logical_output_index != static_cast<int>(i) ||
                           stage.output_order[i].output_slot != static_cast<int>(i))))) {
      return which +
             " is not published in the stage's logical order from its one output buffer "
             "(logical index " +
             std::to_string(tensor.logical_index) + ", buffer " +
             std::to_string(tensor.physical_index) + ")";
    }
  }
  DataflowModelStage planned;
  planned.key = stage_key_of(stage);
  planned.input_bytes = in_buffer.size_bytes;
  planned.output_bytes = out_buffer.size_bytes;
  const DataflowModelStage::Mla& model =
      planned.kind.emplace<DataflowModelStage::Mla>(DataflowModelStage::Mla{mla.model_path});
  if (first) {
    route.input_bytes = planned.input_bytes;
    route.input_dtype = *in_dtype;
    route.input_shape = in.shape;
    // The application pushes the MLA's input buffer. Where that is the model's tensor and the
    // MLA's padding, the binding's refusal of the model's tensor says so.
    std::uint64_t tensor_bytes = static_cast<std::uint64_t>(dtype_size(*in_dtype));
    for (const std::int64_t dim : in.shape) {
      tensor_bytes = dim > 0 ? tensor_bytes * static_cast<std::uint64_t>(dim) : 0U;
    }
    if (!in.shape.empty() && mla_pads(tensor_bytes, planned.input_bytes)) {
      route.input_padding = planned.input_bytes - tensor_bytes;
    }
  }
  if (last) {
    describe_pulled(stage, out, *dtype_of(out.dtype), route.output);
    route.output.sample_label = route.output.stage_key;
    if (tensors == 1U) {
      route.output.format = caps_format_of(out.dtype);
    } else {
      // Several tensors: the stage's caps are the generic tensor-set's, without a format
      // (StageRecipePreparation.cpp build_kernel_stage_publication), so a GStreamer pull leaves
      // no TessSpec on the tensors and no tag on the Sample. Each further tensor is described as
      // the first is, and carries what an image preproc in front of the MLA publishes with every
      // frame, as the first does.
      route.output.more.reserve(tensors - 1U);
      for (std::size_t i = 1; i < tensors; ++i) {
        const si::LogicalTensorStaticSpec& tensor = stage.logical_outputs[i];
        DataflowPulledTensor& next = route.output.more.emplace_back();
        describe_pulled(stage, tensor, *dtype_of(tensor.dtype), next);
        next.preprocess = route.output.preprocess;
        next.frame_semantics = route.output.frame_semantics;
        next.frames = route.output.frames;
      }
    }
  }
  route.text += std::string(route.text.empty() ? "" : " -> ") + "MLA stage " + planned.key +
                " (depth " + std::to_string(model.depth) + "): input " + in.dtype + " " +
                dims_text(in.shape) + " in " + std::to_string(planned.input_bytes) + " bytes" +
                (first && route.input_padding != 0U
                     ? " (the last " + std::to_string(route.input_padding) +
                           " the MLA's padding: the route's input is the whole buffer)"
                     : std::string()) +
                ", ";
  for (std::size_t i = 0; i < tensors; ++i) {
    const si::LogicalTensorStaticSpec& tensor = stage.logical_outputs[i];
    route.text += (i == 0U ? (tensors == 1U ? std::string("output ")
                                            : std::to_string(tensors) + " outputs in one buffer: ")
                           : std::string("; ")) +
                  tensor.dtype + " " + dims_text(tensor.shape) + " (" + tensor.layout + ") " +
                  std::to_string(tensor.size_bytes) + " bytes at " +
                  std::to_string(tensor.byte_offset) + (tensor.quant ? ", quantized" : "");
  }
  route.text += " of " + std::to_string(planned.output_bytes);
  route.stages.push_back(std::move(planned));
  return {};
}

#if defined(SIMANEAT_WITH_DATAFLOW_SOC_KERNELS)

// The contract of a checked kernel request (`inspected`, the Kernels package's inspection of its
// executable) when it is one this backend runs as one operator: `inputs` and `outputs` ports, no
// workspace, no host tasks, every input of a fixed extent. Null with `why`, the fact about the
// kernel, otherwise.
const client::KernelInspection*
plain_contract(const client::Result<client::KernelInspection>& inspected, std::size_t inputs,
               std::size_t outputs, std::string& why) {
  if (const auto* error = std::get_if<client::Error>(&inspected)) {
    why = "the Kernels package gave no contract: " + error->message;
    return nullptr;
  }
  const client::KernelInspection& contract = std::get<client::KernelInspection>(inspected);
  if (contract.input_requirements().size() != inputs ||
      contract.output_requirements().size() != outputs) {
    why = "the kernel's contract has other ports than its request";
  } else if (contract.workspace().storage_bytes != 0U) {
    why = "the kernel needs a workspace of " + std::to_string(contract.workspace().storage_bytes) +
          " bytes";
  } else if (contract.requires_host_tasks()) {
    why = "the kernel needs host tasks";
  } else {
    for (std::size_t i = 0; i < inputs; ++i) {
      if (contract.input_extent(i).mode != client::InputExtentMode::Fixed) {
        why = "the kernel reads its input " + std::to_string(i) + " at a varying extent";
        return nullptr;
      }
    }
    return &contract;
  }
  return nullptr;
}

// A kernel stage's one output tensor (`planned`, as its own port; `needs`: what the kernel asks
// of it) in front of the MLA stage `next`, when the MLA's input buffer is that tensor and the
// MLA's padding: the stage's output port becomes the MLA's buffer as bytes and the tensor a view
// at its front (soc::KernelOp::Options::Packing), where the kernel writes its tensor's bytes and
// no others. The padding behind them is written by nobody after the port's pool wrote the slot
// with zeros when it made it (DataflowModelRoute.h, "The MLA's padding"). "" when the port stays
// the tensor's own (no MLA stage behind, or buffers that agree, or that differ otherwise: the
// route's check says so) and when it became the MLA's; otherwise the fact about the stage that
// this backend does not carry.
std::string pad_to_mla_input(const si::StageStaticSpec* next,
                             const client::TensorRequirements& needs, DataflowModelStage& planned) {
  if (!next || next->payload_kind != si::StagePayloadKind::ProcessMla ||
      next->physical_inputs.size() != 1U ||
      !mla_pads(planned.output_bytes, next->physical_inputs.front().size_bytes)) {
    return {};
  }
  if (needs.alignment_bytes == 0U || kSlotAlignment % needs.alignment_bytes != 0U) {
    return "its output tensor asks for an alignment of " + std::to_string(needs.alignment_bytes) +
           " bytes, which the front of the MLA's input buffer (a buffer of bytes) does not have";
  }
  DataflowModelStage::Kernel& kernel = std::get<DataflowModelStage::Kernel>(planned.kind);
  kernel.packed_outputs.push_back(
      {kernel.output, 0U, static_cast<std::size_t>(planned.output_bytes)});
  planned.output_bytes = next->physical_inputs.front().size_bytes;
  return {};
}

// A kernel stage's one input tensor (`planned`, as its own port; `needs`: what the kernel asks of
// it) behind the MLA stage `mla`: the bytes of the MLA's one output buffer at the offset the
// manifest gives the tensor (where the stage's input buffer begins in the MLA's, the tensor in
// that, the kernel's port in the tensor). The whole buffer from its first byte: the port stays
// the tensor's own. Otherwise (a buffer longer than the tensor: the MLA's padding, or a strided
// view that ends before the buffer does) the stage's input port becomes the MLA's buffer as bytes
// and the tensor a view of it (soc::KernelOp::Options::Packing): the kernel reads its tensor's
// bytes and no others, so what the MLA leaves in its padding reaches nobody. "" or the fact about
// the stage that this backend does not carry.
std::string view_of_mla_output(const si::StageStaticSpec& stage,
                               const client::TensorStageMember& member,
                               const client::TensorRequirements& needs,
                               const DataflowModelStage& mla, DataflowModelStage& planned) {
  const std::uint64_t buffer = mla.output_bytes;
  const std::uint64_t bytes = planned.input_bytes;
  std::int64_t at = 0;
  if (!stage.physical_inputs.empty()) {
    const si::PhysicalBufferStaticSpec& carrier = stage.physical_inputs.front();
    if (carrier.source_physical_index > 0) {
      return "it reads output buffer " + std::to_string(carrier.source_physical_index) +
             " of the MLA stage '" + mla.key + "', which writes one output buffer";
    }
    at = carrier.source_byte_offset;
  }
  if (!stage.logical_inputs.empty() && at >= 0) {
    const std::int64_t in_carrier = stage.logical_inputs.front().byte_offset;
    at = in_carrier < 0 ? -1 : at + in_carrier;
  }
  const std::uint64_t offset =
      at < 0 ? buffer + 1U : static_cast<std::uint64_t>(at) + member.inputs.front().byte_offset;
  if (offset == 0U && bytes == buffer) {
    return {};
  }
  if (bytes == 0U || offset > buffer || bytes > buffer - offset) {
    return "its input tensor of " + std::to_string(bytes) + " bytes at offset " +
           (at < 0 ? std::string("(negative)") : std::to_string(offset)) +
           " does not lie in the output buffer of " + std::to_string(buffer) +
           " bytes that the MLA stage '" + mla.key + "' writes";
  }
  if (needs.alignment_bytes == 0U || kSlotAlignment % needs.alignment_bytes != 0U ||
      offset % needs.alignment_bytes != 0U) {
    return "its input tensor asks for an alignment of " + std::to_string(needs.alignment_bytes) +
           " bytes, which a view at offset " + std::to_string(offset) +
           " of the MLA's output buffer (a buffer of bytes) does not have";
  }
  DataflowModelStage::Kernel& kernel = std::get<DataflowModelStage::Kernel>(planned.kind);
  kernel.packed_inputs.push_back(
      {kernel.input, static_cast<std::size_t>(offset), static_cast<std::size_t>(bytes)});
  planned.input_bytes = buffer;
  return {};
}

// The frames an image stage is planned for: what the build says of the input's size.
struct PlannedFrame {
  /// The Input declares the frame's size: every frame has it. Otherwise width and height are a
  /// bound (0: none declared) and each frame brings its own size.
  bool fixed = false;
  std::int64_t width = 0, height = 0, channels = 0;
  /// The seed's size, when the build has a seed (0: none).
  std::int64_t seed_width = 0, seed_height = 0;
  std::string from; ///< who says the size, for messages
};

// What the build says of the frames the input will carry. The Input's declaration fixes their
// size (InputOptions::shape with every axis fixed, or width and height); a declaration that
// only bounds it (max_width and max_height, or a shape with dynamic axes and max_shape) and a
// build that says nothing leave it to each frame. A seed is an example of a frame, not a
// declaration: it must be a frame the stage takes, and fixes nothing. "" and `frame` set, or
// why the stage cannot be planned for such an input. `nv12`: the model's frames are NV12.
std::string planned_frame(const InputOptions& declared, std::span<const Tensor> seed, bool nv12,
                          PlannedFrame& frame) {
  // [height, width, channels], or with a batch axis of 1 in front; -1: a dynamic axis.
  const auto of_shape = [](const std::vector<std::int64_t>& shape, std::int64_t& height,
                           std::int64_t& width, std::int64_t& channels) {
    const std::size_t lead = shape.size() == 4U && shape[0] == 1 ? 1U : 0U;
    if (shape.size() != 3U + lead) {
      return false;
    }
    height = shape[lead];
    width = shape[lead + 1U];
    channels = shape[lead + 2U];
    return true;
  };
  if (!seed.empty()) {
    if (seed.size() != 1U) {
      return "the seed is several tensors: the stage reads one frame";
    }
    const Tensor& tensor = seed.front();
    if (nv12) {
      const Plane* const luma = tensor.try_plane(PlaneRole::Y);
      if (tensor.dtype != TensorDType::UInt8 || !tensor.is_nv12() || !luma ||
          luma->shape.size() < 2U || luma->shape[0] <= 0 || luma->shape[1] <= 0) {
        return "the seed is not an NV12 frame (a UInt8 tensor with a Y and a UV plane) where the "
               "model reads NV12 frames";
      }
      frame.seed_height = luma->shape[0];
      frame.seed_width = luma->shape[1];
    } else {
      if (tensor.is_composite()) {
        return "the seed is a planar image (NV12, I420) where the model reads packed frames";
      }
      std::int64_t channels = 0;
      // Packed: its layout says HWC, or it says none and its axes say height, width, channels
      // (a Model's own dummy frame).
      using Axis = TensorAxisSemantic;
      const bool packed =
          tensor.layout == TensorLayout::HWC ||
          (tensor.layout == TensorLayout::Unknown &&
           (tensor.axis_semantics == std::vector<Axis>{Axis::H, Axis::W, Axis::C} ||
            tensor.axis_semantics == std::vector<Axis>{Axis::N, Axis::H, Axis::W, Axis::C}));
      if (tensor.dtype != TensorDType::UInt8 || !tensor.is_dense() || !packed ||
          !of_shape(tensor.shape, frame.seed_height, frame.seed_width, channels) ||
          frame.seed_height <= 0 || frame.seed_width <= 0 || channels <= 0) {
        return "the seed is not a packed UInt8 image of shape [height, width, channels] (or with "
               "a batch axis of 1 in front)";
      }
      if (!tensor.is_contiguous()) {
        return "the seed's rows are padded (its row stride is not width x channels) where the "
               "stage reads frames whose rows are tight";
      }
      frame.channels = channels;
    }
  }
  if (!declared.shape.empty()) {
    std::int64_t height = 0, width = 0, channels = 0;
    if (nv12 || declared.dtype != TensorDType::UInt8 ||
        !of_shape(declared.shape, height, width, channels) || channels <= 0) {
      return nv12 ? "the Input declares a tensor shape where the model reads NV12 frames (two "
                    "planes): declare the image's format and size instead"
                  : "the Input's declared shape is not a UInt8 frame [height, width, channels]";
    }
    frame.channels = channels;
    if (height > 0 && width > 0) {
      frame.fixed = true;
      frame.height = height;
      frame.width = width;
      frame.from = "the Input's declaration";
      return {};
    }
    // Dynamic axes: each frame's own size, up to the declared bound of the axis.
    std::int64_t max_height = 0, max_width = 0, max_channels = 0;
    if (!declared.max_shape.empty() &&
        !of_shape(declared.max_shape, max_height, max_width, max_channels)) {
      return "the Input's declared max_shape is not the bound of a frame [height, width, "
             "channels]";
    }
    frame.height = height > 0 ? height : std::max<std::int64_t>(max_height, 0);
    frame.width = width > 0 ? width : std::max<std::int64_t>(max_width, 0);
    frame.from = "each frame's own size";
    return {};
  }
  if (declared.width > 0 && declared.height > 0 && !nv12) {
    frame.fixed = true;
    frame.width = declared.width;
    frame.height = declared.height;
    frame.from = "the Input's declaration";
    return {};
  }
  // NV12: one byte port whatever is declared, so a declared size is the port's frame.
  frame.width = declared.width > 0 ? declared.width : std::max(declared.max_width, 0);
  frame.height = declared.height > 0 ? declared.height : std::max(declared.max_height, 0);
  frame.from = "each frame's own size";
  return {};
}

// The image preproc stage (the Kernels package's "preproc", which GStreamer runs through a frame
// adapter); "" or the fact about it that this backend does not carry. With a declared frame
// size it is one kernel
// at that size. Without one the stage's request is the one at the port's bound and
// `route.frames` gives each frame its own (DataflowFrames). The requests, the published
// tensor's description and the preprocess facts all come from the stage's own checked policy
// (PreprocFrameAdapter), by the functions the adapter runs for a frame of that geometry. `next`
// is the stage behind it (null at the route's end): in front of an MLA stage whose input buffer
// is the kernel's tensor and the MLA's padding, the kernel writes at the buffer's front
// (pad_to_mla_input), whatever the frame's size (its output is the model's input).
std::string plan_frame_stage(const si::StageStaticSpec& stage, bool first, bool last,
                             const si::StageStaticSpec* next, const InputOptions& declared,
                             std::span<const Tensor> seed, DataflowModelRoute& route) {
  using Adapter = si::kernel_lowering::PreprocFrameAdapter;
  const si::KernelStageRecipe& recipe = *stage.native_kernel;
  const si::ProcessCvuStagePayload& payload = stage.processcvu;
  if (!recipe.frame_adapter_factory || payload.graph_name != "preproc" || !first) {
    return "it is an image stage with per-frame geometry that is not the image preproc at the "
           "route's front";
  }
  if (payload.image_transform || recipe.image_publication || stage.image_publication) {
    return "it is an image frontend (it publishes image planes)";
  }
  const std::string format = upper(payload.input_img_type);
  const bool nv12 = format == "NV12";
  if (format != "RGB" && format != "BGR" && !nv12) {
    return "it reads " + (format.empty() ? std::string("an unnamed format") : format) +
           " frames: only packed RGB and BGR frames and NV12 frames run (I420 and NV16 are other "
           "plane layouts, GRAY another packing)";
  }
  if (payload.batch_size != 1) {
    return "it has a batch of " + std::to_string(payload.batch_size) +
           " (an ROI list or a batched model)";
  }
  if (recipe.members.size() != 1U || recipe.members.front().outputs.size() != 1U ||
      !recipe.resolver_factories.empty() || !recipe.constant_data.empty() ||
      !recipe.private_buffers.empty() || !recipe.private_outputs.empty() ||
      stage.logical_outputs.size() != 1U) {
    return "it has several outputs, variants, constants or private buffers";
  }
  PlannedFrame frame;
  if (const std::string why = planned_frame(declared, seed, nv12, frame); !why.empty()) {
    return why;
  }
  auto checked = Adapter::check_policy(payload);
  if (const auto* error = std::get_if<client::Error>(&checked)) {
    return "the stage's image policy does not check: " + error->field + ": " + error->message;
  }
  const std::shared_ptr<const Adapter::CheckedPolicy> policy =
      std::get<std::shared_ptr<const Adapter::CheckedPolicy>>(std::move(checked));
  // The frame's request at one size, rows not padded: a slot of the input's pool holds exactly
  // the frame, and a wrapped frame is one whose rows are tight.
  const auto request_at = [&](std::int64_t width, std::int64_t height) {
    return Adapter::checked_fixed_frame(
        policy, static_cast<std::size_t>(width), static_cast<std::size_t>(height),
        static_cast<std::size_t>(
            nv12 ? width : width * (frame.channels > 0 ? frame.channels : kFrameChannels)));
  };
  const auto [envelope_width, envelope_height] = Adapter::checked_envelope(*policy);
  if (!frame.fixed) {
    // The port's bound: the Input's capacity as GStreamer's build resolves it (InputPolicy.cpp
    // resolve_shape_limits: the declared bound, else the declared or the seed's size), inside
    // the stage's capacity envelope; the envelope when the build says nothing.
    if (frame.width <= 0) {
      frame.width = frame.seed_width;
    }
    if (frame.height <= 0) {
      frame.height = frame.seed_height;
    }
    const auto bound = [nv12](std::int64_t declared_bound, std::size_t envelope) {
      std::int64_t value = static_cast<std::int64_t>(envelope);
      if (declared_bound > 0 && declared_bound < value) {
        value = declared_bound;
      }
      return nv12 ? value & ~std::int64_t{1} : value; // an NV12 frame's sizes are even
    };
    frame.width = bound(frame.width, envelope_width);
    frame.height = bound(frame.height, envelope_height);
  }
  // The seed is a frame the Run will be pushed: it must be one the stage takes. A seed of the
  // bound's size is checked as the bound, below.
  const bool seed_is_bound = frame.seed_width == frame.width && frame.seed_height == frame.height;
  if (!frame.fixed && frame.seed_width > 0 && !seed_is_bound) {
    if (frame.seed_width > frame.width || frame.seed_height > frame.height) {
      return "a " + std::to_string(frame.seed_width) + "x" + std::to_string(frame.seed_height) +
             " frame (the seed) is above the " + std::to_string(frame.width) + "x" +
             std::to_string(frame.height) +
             " the Input declares as its largest (InputOptions::max_width, max_height)";
    }
    auto seeded = request_at(frame.seed_width, frame.seed_height);
    if (const auto* error = std::get_if<client::Error>(&seeded)) {
      return "a " + std::to_string(frame.seed_width) + "x" + std::to_string(frame.seed_height) +
             " frame (the seed) is not one the stage takes: " + error->field + ": " +
             error->message;
    }
  }
  auto fixed = request_at(frame.width, frame.height);
  if (const auto* error = std::get_if<client::Error>(&fixed)) {
    return "a " + std::to_string(frame.width) + "x" + std::to_string(frame.height) + " frame (" +
           (frame.fixed     ? frame.from
            : seed_is_bound ? std::string("the seed")
                            : std::string("the input's bound")) +
           ") is not one the stage takes: " + error->field + ": " + error->message;
  }
  Adapter::FixedFrame& request = std::get<Adapter::FixedFrame>(fixed);
  const std::size_t planes = nv12 ? 2U : 1U;
  if (request.inputs.size() != planes || request.outputs.size() != 1U ||
      request.inputs.front().shape.size() != 4U) {
    return "the frame's kernel request has other tensors than the frame's planes";
  }
  if (!nv12 && frame.channels > 0 && request.inputs.front().shape[3] != frame.channels) {
    return std::string(frame.seed_width > 0 ? "the seed" : "the Input's declaration") + " has " +
           std::to_string(frame.channels) + " channels where the model's " + format +
           " input has " + std::to_string(request.inputs.front().shape[3]);
  }
  const auto inspected = client::inspect_kernel(request.executable);
  std::string unsupported;
  const client::KernelInspection* const plain = plain_contract(inspected, planes, 1U, unsupported);
  if (!plain) {
    return unsupported;
  }
  const client::KernelInspection& contract = *plain;
  const std::optional<TensorDType> in_dtype = dtype_of(request.inputs.front().dtype);
  const std::optional<TensorDType> out_dtype = dtype_of(request.outputs.front().dtype);
  const simaai::gst::TensorBufferPreprocessFacts* const facts =
      simaai::gst::tensor_buffer_preprocess_facts(request.facts);
  if (!in_dtype || !out_dtype || !facts) {
    return "the frame's kernel request has no Core dtype or no preprocess facts";
  }
  DataflowModelStage planned;
  planned.key = stage_key_of(stage);
  planned.output_bytes = contract.output_requirements().front().storage_bytes;
  DataflowModelStage::Kernel& kernel = planned.kind.emplace<DataflowModelStage::Kernel>();
  kernel.ev74 = true;
  kernel.operation = "preproc";
  kernel.output = request.outputs.front();
  kernel.parameters = std::move(request.parameters);
  const std::uint64_t tensor_bytes = planned.output_bytes;
  if (const std::string why =
          pad_to_mla_input(next, contract.output_requirements().front(), planned);
      !why.empty()) {
    return why;
  }
  if (nv12) {
    // One port of bytes, the frame at the bound as it is: the luma plane at 0, the chroma
    // plane behind its rows. A smaller frame's planes are written to the same two offsets.
    const std::size_t luma = static_cast<std::size_t>(frame.width * frame.height);
    kernel.packed_inputs.push_back(
        {request.inputs[0], 0U, contract.input_requirements()[0].storage_bytes});
    kernel.packed_inputs.push_back(
        {request.inputs[1], luma, contract.input_requirements()[1].storage_bytes});
    planned.input_bytes = luma + luma / 2U;
    if (kernel.packed_inputs[0].bytes > luma || kernel.packed_inputs[1].bytes > luma / 2U) {
      return "the NV12 frame's planes do not lie in the frame's bytes";
    }
    const std::size_t alignment = contract.input_requirements()[1].alignment_bytes;
    if (alignment == 0U || kSlotAlignment % alignment != 0U || luma % alignment != 0U) {
      return "the chroma plane of a " + std::to_string(frame.width) + "x" +
             std::to_string(frame.height) + " NV12 frame begins at byte " + std::to_string(luma) +
             " of the frame, which is not a multiple of the " + std::to_string(alignment) +
             " bytes the kernel asks of a plane: such a frame's planes are not two views of one "
             "buffer";
    }
    route.frame_shape = {static_cast<std::int64_t>(planned.input_bytes)};
    route.planar_input = true;
  } else {
    kernel.input = request.inputs.front();
    planned.input_bytes = contract.input_requirements().front().storage_bytes;
    route.frame_shape = kernel.input.shape;
  }
  route.input_bytes = planned.input_bytes;
  route.input_dtype = *in_dtype;
  route.kernel_input = true;
  route.ev74_input = true;
  route.frame_input = true;
  if (frame.fixed) {
    route.input_shape = route.frame_shape;
  }
  try {
    // What GStreamer's pull leaves on every tensor behind this stage, for this frame size (a
    // route whose frames change size publishes each frame's own in its place).
    route.output.preprocess = preprocess_from_compact_facts(*facts);
    route.output.frame_semantics = true;
  } catch (const std::exception& error) {
    return error.what();
  }
  if (!frame.fixed) {
    // Each frame is run by its geometry's kernel: an RGB or BGR frame's extent on the bounded port
    // says its geometry; an NV12 frame's byte port does not, its stamp's user word does.
    kernel.variants = nv12 ? DataflowModelStage::Kernel::Variants::ImageByWord
                           : DataflowModelStage::Kernel::Variants::ImageByExtent;
    auto frames = std::make_shared<DataflowFrames>();
    frames->policy = policy;
    frames->preproc_stage = planned;
    frames->bound_preprocess = *route.output.preprocess;
    frames->max_width = frame.width;
    frames->max_height = frame.height;
    frames->envelope_width = static_cast<std::int64_t>(envelope_width);
    frames->envelope_height = static_cast<std::int64_t>(envelope_height);
    frames->channels = nv12 ? 0 : kernel.input.shape[3];
    frames->nv12 = nv12;
    frames->format = format;
    route.output.frames = frames.get();
    route.frames = std::move(frames);
  }
  const si::LogicalTensorStaticSpec& out = stage.logical_outputs.front();
  if (last) {
    const std::optional<TensorDType> published = dtype_of(out.dtype);
    if (!published || *published != *out_dtype) {
      return "the published tensor's dtype '" + out.dtype + "' is not the kernel's";
    }
    if (out.byte_offset != 0 || out.shape.empty() ||
        (out.size_bytes != 0U && out.size_bytes != planned.output_bytes)) {
      return "the published output (" + std::to_string(out.size_bytes) + " bytes at " +
             std::to_string(out.byte_offset) + ") is not the kernel's whole output buffer of " +
             std::to_string(planned.output_bytes) + " bytes";
    }
    describe_pulled(stage, out, *out_dtype, route.output);
    route.output.sample_label =
        payload.primary_output_name.empty() ? route.output.stage_key : payload.primary_output_name;
    if (const auto* tiled = std::get_if<client::MlaTiled>(&kernel.output.layout)) {
      // The publication of a tiled tensor as its packed storage (StageRecipePreparation.cpp
      // apply_kernel_publication_views): no caps format, the checked tile description, and no
      // strides on the pulled tensor whatever the manifest's logical output states (a tiled
      // tensor is read through its storage).
      const char* const dtype = kernel.output.dtype == client::DType::BFloat16 ? "BF16"
                                : kernel.output.dtype == client::DType::Int8   ? "INT8"
                                : kernel.output.dtype == client::DType::UInt8  ? "UINT8"
                                                                               : nullptr;
      if (!dtype || out.publication_relation != si::PublicationViewRelation::PackedStorage) {
        return "it publishes its tiled output as another view than its packed storage "
               "(publication relation " +
               std::to_string(static_cast<int>(out.publication_relation)) + ")";
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
      "kernel stage " + planned.key + " (preproc on the EV74 for " +
      (frame.fixed ? std::to_string(frame.width) + "x" + std::to_string(frame.height) + " " +
                         format + " frames, " + frame.from
                   : format + " frames of a size that may change with each push, up to " +
                         std::to_string(frame.width) + "x" + std::to_string(frame.height) +
                         (nv12 ? " (the port's frame: its planes are two views of the port's "
                                 "bytes)"
                               : "") +
                         ", a kernel per frame size; at the bound") +
      "; " + parameters_text(kernel.parameters) + "): " + std::to_string(planned.input_bytes) +
      " -> " + std::to_string(tensor_bytes) + " bytes" +
      (kernel.packed_outputs.empty() ? std::string()
                                     : " at the front of the MLA's input buffer of " +
                                           std::to_string(planned.output_bytes)) +
      (last ? ", output " + out.dtype + " " + dims_text(out.shape) + " " +
                  std::to_string(out.size_bytes) + " bytes at " + std::to_string(out.byte_offset)
            : std::string());
  route.stages.push_back(std::move(planned));
  return {};
}

// The decode families the BoxDecode stage runs: the (profile, task) pairs of the Kernels package's
// "boxdecode" whose wire a device row has shown byte-equal to GStreamer's. The stage has one shape
// for every family (one decode kernel on the heads in the MLA's buffer, the stage's serializer over
// its outputs); what a family's row shows is its own: the kernel's one-phase entry gives the bytes
// of the staged entry GStreamer's host tasks give it, and its per-frame geometry variant gives the
// bytes of the request. A family is added with its row; one without a row is not carried.
struct CarriedDecode {
  std::string_view profile, task;
};
constexpr CarriedDecode kCarriedDecodes[] = {
    {"yolo_dfl_v1", "detection"},  {"yolo_dfl_obb_v1", "detection"}, {"yolo26_obb_v1", "detection"},
    {"yolo26_v1", "detection"},    {"yolox_v1", "detection"},        {"yolo26_v1", "pose"},
    {"yolo26_v1", "segmentation"},
};

bool carried_decode(std::string_view profile, std::string_view task) {
  for (const CarriedDecode& carried : kCarriedDecodes) {
    if (carried.profile == profile && carried.task == task) {
      return true;
    }
  }
  return false;
}

// "yolo_dfl_v1/detection, yolo26_v1/detection, ...": the families, for the text of a refusal.
std::string carried_decodes_text() {
  std::string out;
  for (const CarriedDecode& carried : kCarriedDecodes) {
    out +=
        (out.empty() ? "" : ", ") + std::string(carried.profile) + "/" + std::string(carried.task);
  }
  return out;
}

// A kernel request's tensor as the runtime states it (defined with the kernels' making, below).
std::optional<df::TensorSpec> runtime_tensor(const client::TensorSpec& tensor);

// The two byte counts the Run's kernel object holds a dense or strided tensor of a request in
// (soc::Kernel::make checks the Kernels package's count against them): `own`, the tensor's own
// extent, to the end of its last element; and for a strided tensor `whole`, its rows whole,
// every row of its outermost dimension to the end of its pitch (a head whose pixels are pitched
// for more channels than it has, read to the end of the last pixel's padding). Both 0: an MLA
// layout, whose bytes are the Kernels package's, or a tensor the runtime does not state.
struct KernelObjectBytes {
  std::size_t own = 0, whole = 0;
};
KernelObjectBytes kernel_object_bytes(const client::TensorSpec& tensor) {
  const std::optional<df::TensorSpec> spec = runtime_tensor(tensor);
  if (!spec || std::holds_alternative<df::MlaTiled>(spec->layout) ||
      std::holds_alternative<df::MlaCBlocked>(spec->layout)) {
    return {};
  }
  const df::PortSpec port{.payload = *spec, .memory = df::MemoryKind::Host};
  if (df::validate(port)) {
    return {};
  }
  KernelObjectBytes held{df::slot_bytes(port), 0};
  std::int64_t whole = 0;
  if (const auto* strided = std::get_if<df::Strided>(&spec->layout);
      strided && spec->shape.rank > 0U &&
      !__builtin_mul_overflow(strided->stride_bytes[0], spec->shape.dims[0].max, &whole) &&
      whole > 0) {
    held.whole = static_cast<std::size_t>(whole);
  }
  return held;
}

// A frame's preprocess facts as the BoxDecode stage's geometry reads them (null: the route has no
// image preproc, its frames carry none). The result views `meta`.
si::BoxDecodeFrameFacts boxdecode_facts(const PreprocessRuntimeMeta* meta) {
  si::BoxDecodeFrameFacts facts;
  if (meta) {
    facts.original_width = meta->original_width;
    facts.original_height = meta->original_height;
    facts.resized_width = meta->resized_width;
    facts.resized_height = meta->resized_height;
    facts.resize_mode = meta->resize_mode;
    facts.roi_list = meta->roi_list_enabled;
    facts.affine = {meta->affine_m00, meta->affine_m01, meta->affine_m02,
                    meta->affine_m10, meta->affine_m11, meta->affine_m12};
  }
  return facts;
}

// The BoxDecode stage behind the MLA stage, at the route's end, as two route stages: the Kernels
// package's "boxdecode" with its heads as views of the MLA's one output buffer and its outputs as
// views of one small buffer, and Core's wire serializer behind it; "" or the fact about it that
// this backend does not carry. The request, the heads' places, the serializer and the published
// tensor's description are the stage's own (KernelStageRecipe::BoxDecodeProgram, the manifest); the
// original image size and the resize mode are those of the frames the route's image preproc was
// planned for. The stage in front is the MLA stage when there is one (the planner declines two
// kernel stages in a row, and the wire serializer is a route's last stage).
std::string plan_boxdecode_stage(const si::StageStaticSpec& stage, bool first, bool last,
                                 DataflowModelRoute& route) {
  const si::KernelStageRecipe& recipe = *stage.native_kernel;
  const si::KernelStageRecipe::BoxDecodeProgram& program = *recipe.boxdecode_program;
  if (first || !last) {
    return "it is not the route's last stage behind the model's MLA stage: BoxDecode runs "
           "there only (its kernel reads the heads in the MLA's output buffer), not on heads the "
           "application pushes";
  }
  if (!carried_decode(program.profile, program.task)) {
    return "it decodes profile '" + program.profile + "', task '" + program.task +
           "': the BoxDecode stage runs the decode families " + carried_decodes_text() + " only";
  }
  if (!program.write || (recipe.output_format != kDetectionFormatBbox &&
                         recipe.output_format != kDetectionFormatBboxObbV1)) {
    return "it publishes another wire than BBOX or BBOX_OBB_V1 ('" + recipe.output_format + "')";
  }
  // The decode is the recipe's first member (BoxDecodeProgram::member) when no detess member is
  // in front of it, and the stage's host task (the serializer) the second.
  if (recipe.members.size() != 2U) {
    return "it runs " + std::to_string(recipe.members.size()) +
           " invocations where it runs the decode and its serializer (detess invocations in front "
           "of the decode: heads in tiled, not C-blocked, storage)";
  }
  const client::TensorStageMember& member = recipe.members.front();
  if (!recipe.constant_data.empty() || !member.dependencies.empty() ||
      member.inputs.size() != program.request.inputs.size() ||
      member.outputs.size() != program.request.outputs.size() || member.inputs.empty() ||
      stage.logical_outputs.size() != 1U) {
    return "it is not one decode on the A65 from the stage's inputs into one published wire";
  }
  if (stage.physical_inputs.size() != 1U) {
    return "it reads its heads from " + std::to_string(stage.physical_inputs.size()) +
           " buffers: the heads are views of one buffer (a model that writes one output buffer)";
  }
  // The frames' geometry: what the route's image preproc publishes with every frame, a constant
  // of the frame size it was planned for. The request is the one GStreamer's resolver selects
  // for such a frame.
  const PreprocessRuntimeMeta* const facts =
      route.output.preprocess ? &*route.output.preprocess : nullptr;
  // A decode that takes each frame's affine as well as its size runs for the oriented-box
  // families, which always take both (their rows show the "image_geometry" variant). The other
  // families take it only for a crop, an off-centre letterbox or an explicit transform, which
  // no row has shown.
  auto fixed = program.complete_geometry && !program.uniform_transform
                   ? client::Result<si::BoxDecodeFrameGeometry>(client::Error{
                         client::ErrorCode::InvalidRequest, "boxdecode.geometry",
                         "a complete-geometry profile takes each frame's preprocess affine"})
                   : si::boxdecode_geometry_for_frames(recipe, boxdecode_facts(facts));
  if (const auto* error = std::get_if<client::Error>(&fixed)) {
    return (facts ? "the frames of the route's image preproc (" +
                        std::to_string(facts->original_width) + "x" +
                        std::to_string(facts->original_height) + ", resize mode '" +
                        facts->resize_mode + "') are not ones the decode takes: "
                  : std::string("the route has no image preproc stage in front, whose frame size "
                                "at the build fixes the decode's image geometry (here the "
                                "geometry comes with each frame): ")) +
           error->message;
  }
  si::KernelStageRecipe::Request& request = std::get<si::BoxDecodeFrameGeometry>(fixed).request;
  const auto inspected = client::inspect_kernel(member.executable);
  std::string unsupported;
  const client::KernelInspection* const plain =
      plain_contract(inspected, request.inputs.size(), request.outputs.size(), unsupported);
  if (!plain) {
    return unsupported;
  }
  const client::KernelInspection& contract = *plain;
  DataflowModelStage decode;
  decode.key = stage_key_of(stage);
  DataflowModelStage::Kernel& kernel = decode.kind.emplace<DataflowModelStage::Kernel>();
  kernel.operation = request.operation;
  // The heads: each the kernel's tensor at the byte offset the manifest gives its logical input
  // in the one buffer the stage binds (the MLA's output buffer).
  const si::PhysicalBufferStaticSpec& carrier = stage.physical_inputs.front();
  decode.input_bytes = carrier.size_bytes;
  for (std::size_t i = 0; i < member.inputs.size(); ++i) {
    const client::TensorStagePort& port = member.inputs[i];
    if (port.source != client::TensorStagePortSource::Input ||
        port.tensor >= stage.logical_inputs.size()) {
      return "it reads a private buffer";
    }
    const si::LogicalInputStaticSpec& logical = stage.logical_inputs[port.tensor];
    const std::string head = "head " + std::to_string(i) + " (" + logical.logical_name + ")";
    const std::uint64_t bytes = contract.input_requirements()[i].storage_bytes;
    const std::uint64_t offset =
        logical.byte_offset < 0
            ? carrier.size_bytes + 1U
            : static_cast<std::uint64_t>(logical.byte_offset) + port.byte_offset;
    if (logical.physical_index != carrier.physical_index || bytes == 0U ||
        offset > carrier.size_bytes || bytes > carrier.size_bytes - offset) {
      return head + " of " + std::to_string(bytes) + " bytes at " +
             std::to_string(logical.byte_offset) +
             " does not lie in the stage's one input "
             "buffer of " +
             std::to_string(carrier.size_bytes) + " bytes";
    }
    if (const KernelObjectBytes held = kernel_object_bytes(request.inputs[i]);
        held.own != 0U && held.own != bytes && held.whole != bytes) {
      return head + " is a tensor of " + dims_text(request.inputs[i].shape) +
             " whose kernel reads " + std::to_string(bytes) +
             " bytes of it, which is neither the tensor's own extent (" + std::to_string(held.own) +
             " bytes)" +
             (held.whole != 0U ? " nor its rows whole to the end of their pitch (" +
                                     std::to_string(held.whole) + " bytes)"
                               : std::string());
    }
    for (const DataflowModelStage::View& other : kernel.packed_inputs) {
      if (offset < other.offset + other.bytes && other.offset < offset + bytes) {
        return head + " shares bytes with another head in the stage's input buffer";
      }
    }
    kernel.packed_inputs.push_back(
        {request.inputs[i], static_cast<std::size_t>(offset), static_cast<std::size_t>(bytes)});
  }
  // The kernel's outputs: one buffer, each at a slot's alignment.
  std::size_t end = 0;
  for (std::size_t i = 0; i < request.outputs.size(); ++i) {
    const client::TensorRequirements& needs = contract.output_requirements()[i];
    if (needs.storage_bytes == 0U || needs.alignment_bytes == 0U ||
        kSlotAlignment % needs.alignment_bytes != 0U) {
      return "the kernel's output " + std::to_string(i) + " asks for an alignment of " +
             std::to_string(needs.alignment_bytes) + " bytes, which a packed port does not have";
    }
    const std::size_t offset = aligned(end);
    kernel.packed_outputs.push_back({request.outputs[i], offset, needs.storage_bytes});
    end = offset + needs.storage_bytes;
  }
  decode.output_bytes = aligned(end);
  kernel.parameters = std::move(request.parameters);
  if (route.frames) {
    // Frames of a changing size: the frame's geometry is its stamp's user word (the MLA's
    // buffer does not tell it); each geometry's kernel is a variant of this one (DataflowFrames).
    kernel.variants = DataflowModelStage::Kernel::Variants::DecodeByWord;
  }
  // The wire, as the stage publishes it: one UINT8 tensor of the serializer's bytes.
  const si::LogicalTensorStaticSpec& out = stage.logical_outputs.front();
  const std::optional<TensorDType> published = dtype_of(out.dtype);
  if (!published || *published != TensorDType::UInt8 || out.byte_offset != 0 ||
      out.size_bytes != program.wire_bytes || out.shape.empty()) {
    return "the published output (" + out.dtype + ", " + std::to_string(out.size_bytes) +
           " bytes at " + std::to_string(out.byte_offset) + ") is not the wire of " +
           std::to_string(program.wire_bytes) + " bytes";
  }
  DataflowModelStage wire;
  wire.key = decode.key + "_wire";
  wire.input_bytes = decode.output_bytes;
  wire.output_bytes = program.wire_bytes;
  wire.kind = DataflowModelStage::Wire{kernel.packed_outputs, program.wire, program.write};
  describe_pulled(stage, out, *published, route.output);
  // The stage's wire identity (finalize_stage_publications: the caps format), on the Sample and,
  // as GStreamer's pull leaves it, in both the legacy TessSpec and Semantic::detection.
  route.output.format = recipe.output_format;
  route.output.detection = DetectionSpec{recipe.output_format};
  route.output.frame_semantics = true;
  route.output.sample_label = stage.processcvu.primary_output_name.empty()
                                  ? route.output.stage_key
                                  : stage.processcvu.primary_output_name;
  std::string heads;
  for (const DataflowModelStage::View& view : kernel.packed_inputs) {
    heads += (heads.empty() ? "" : ", ") + dims_text(view.tensor.shape) + " at " +
             std::to_string(view.offset);
  }
  route.text +=
      " -> kernel stage " + decode.key + " (boxdecode on the A65, " + program.profile + " " +
      program.task +
      (facts ? " for " + std::to_string(facts->original_width) + "x" +
                   std::to_string(facts->original_height) + " " + facts->resize_mode + " frames" +
                   (route.frames ? " at the bound, a kernel per frame size by the stamp's user "
                                   "word"
                                 : "")
             : std::string()) +
      "; " + std::to_string(kernel.packed_inputs.size()) + " heads in the " +
      std::to_string(decode.input_bytes) + " bytes: " + heads + "; " +
      std::to_string(kernel.packed_outputs.size()) + " outputs in " +
      std::to_string(decode.output_bytes) + " bytes; " + parameters_text(kernel.parameters) +
      ") -> wire " + recipe.output_format + " (Core's serializer on the Worker): output " +
      out.dtype + " " + dims_text(out.shape) + " " + std::to_string(out.size_bytes) + " bytes";
  if (route.frames) {
    // Frames of a changing size: the kernel planned here is the operator's own, at the port's
    // bound; each frame size is checked by the same function (DataflowFrames).
    route.frames->decode = stage.native_kernel;
  }
  route.stages.push_back(std::move(decode));
  route.stages.push_back(std::move(wire));
  return {};
}

#endif // SIMANEAT_WITH_DATAFLOW_SOC_KERNELS

// What the planner reads besides the manifest (plan_dataflow_route_stages).
struct PlanInput {
  const InputOptions& declared;
  std::span<const Tensor> seed;
  const std::string& ev74_obstacle;
};

// One kernel stage of the manifest (its checked recipe, as prepare_kernel_stage_recipes left it)
// as a route stage; "" or the fact about it that this backend does not carry. `board` is set
// when that fact is the board's: the stage is placed on the EV74 and the EV74 does not run the
// Kernels firmware (`in.ev74_obstacle` says what it runs). `next` is the stage behind it (null
// at the route's end). A canonical transform beside the MLA stage has its one tensor of that side
// in the MLA's buffer: the whole buffer, or, where the MLA's buffer is longer, a view of it
// (view_of_mla_output behind the MLA, pad_to_mla_input in front of it).
std::string plan_kernel_stage(const si::StageStaticSpec& stage, bool first, bool last,
                              const si::StageStaticSpec* next, const PlanInput& in,
                              DataflowModelRoute& route, bool& board) {
  const si::KernelStageRecipe& recipe = *stage.native_kernel;
  const std::string key = stage_key_of(stage);
  board = false;
#if !defined(SIMANEAT_WITH_DATAFLOW_SOC_KERNELS)
  (void)recipe;
  (void)first;
  (void)last;
  (void)next;
  (void)in;
  (void)route;
  return "this build of libsima_neat runs no kernel stage on the Dataflow backend (it was built "
         "without the Runner's Executor and the Kernels package)";
#else
  const bool ev74 = recipe.target != client::Hardware::A65;
  if (ev74 && !in.ev74_obstacle.empty()) {
    // Never the A65 kernel in the EV74's place: the placement is the Graph's (the processcvu run
    // target policy), not this backend's. It is the runtime's own condition (soc::KernelOp
    // refuses the EV74 target on it at instantiate), asked at the plan because an input may be
    // bound, and its Session instantiated, only by its first push.
    board = true;
    return in.ev74_obstacle + ", and the Dataflow backend runs EV74 kernels on the firmware '" +
           std::string(kKernelsFirmwareName) + "' only";
  }
  if (recipe.frame_adapter_factory || recipe.image_publication || stage.image_publication) {
    return plan_frame_stage(stage, first, last, next, in.declared, in.seed, route);
  }
  if (recipe.boxdecode_program) {
    return plan_boxdecode_stage(stage, first, last, route);
  }
  if (recipe.members.size() != 1U) {
    return "it runs " + std::to_string(recipe.members.size()) +
           " kernel invocations per frame (a batched model or several tensors)";
  }
  if (recipe.requests.size() != recipe.members.size()) {
    return "it has no recorded kernel request (not a canonical transform)";
  }
  if (!recipe.resolver_factories.empty() || !recipe.constant_data.empty() ||
      !recipe.private_buffers.empty() || !recipe.private_outputs.empty()) {
    return "it has per-frame variants, constants or private buffers";
  }
  const client::TensorStageMember& member = recipe.members.front();
  const si::KernelStageRecipe::Request& request = recipe.requests.front();
  if (member.inputs.size() != 1U || member.outputs.size() != 1U || request.inputs.size() != 1U ||
      request.outputs.size() != 1U) {
    return "it has " + std::to_string(member.inputs.size()) + " inputs and " +
           std::to_string(member.outputs.size()) +
           " outputs: a kernel of several inputs or outputs";
  }
  // Behind the MLA stage the input may begin inside the MLA's buffer (view_of_mla_output); the
  // application's tensor and the stage's output begin at their buffers' first byte.
  if ((first && member.inputs.front().byte_offset != 0U) ||
      member.outputs.front().byte_offset != 0U ||
      member.inputs.front().source != client::TensorStagePortSource::Input ||
      member.outputs.front().source != client::TensorStagePortSource::Output ||
      !member.dependencies.empty()) {
    return "it reads or writes inside a shared buffer (an offset view)";
  }
  if (stage.logical_inputs.size() > 1U || stage.physical_inputs.size() > 1U ||
      stage.logical_outputs.size() != 1U || stage.input_bindings.size() > 1U) {
    return "it has " + std::to_string(stage.logical_inputs.size()) + " input tensors and " +
           std::to_string(stage.logical_outputs.size()) + " published tensors";
  }
  const auto inspected = client::inspect_kernel(member.executable);
  std::string unsupported;
  const client::KernelInspection* const contract = plain_contract(inspected, 1U, 1U, unsupported);
  if (!contract) {
    return unsupported;
  }
  const std::optional<TensorDType> in_dtype = dtype_of(request.inputs.front().dtype);
  const std::optional<TensorDType> out_dtype = dtype_of(request.outputs.front().dtype);
  if (!in_dtype || !out_dtype) {
    return "a tensor's dtype has no Core dtype";
  }
  DataflowModelStage planned;
  planned.key = key;
  planned.input_bytes = contract->input_requirements().front().storage_bytes;
  planned.output_bytes = contract->output_requirements().front().storage_bytes;
  DataflowModelStage::Kernel& kernel = planned.kind.emplace<DataflowModelStage::Kernel>();
  kernel.ev74 = ev74;
  kernel.operation = request.operation;
  kernel.input = request.inputs.front();
  kernel.output = request.outputs.front();
  kernel.parameters = request.parameters;
  const std::uint64_t in_tensor_bytes = planned.input_bytes;
  const std::uint64_t out_tensor_bytes = planned.output_bytes;
  if (!first && std::holds_alternative<DataflowModelStage::Mla>(route.stages.back().kind)) {
    if (const std::string why =
            view_of_mla_output(stage, member, contract->input_requirements().front(),
                               route.stages.back(), planned);
        !why.empty()) {
      return why;
    }
  }
  if (const std::string why =
          pad_to_mla_input(next, contract->output_requirements().front(), planned);
      !why.empty()) {
    return why;
  }
  if (first) {
    if (!std::holds_alternative<client::Contiguous>(kernel.input.layout)) {
      return "it reads a strided or tiled tensor from the application";
    }
    route.input_bytes = planned.input_bytes;
    route.input_dtype = *in_dtype;
    route.input_shape = kernel.input.shape;
    route.kernel_input = true;
    route.ev74_input = ev74;
  }
  const si::LogicalTensorStaticSpec& out = stage.logical_outputs.front();
  if (last) {
    const std::optional<TensorDType> published = dtype_of(out.dtype);
    if (!published || *published != *out_dtype) {
      return "the published tensor's dtype '" + out.dtype + "' is not the kernel's";
    }
    if (out.byte_offset < 0 || out.size_bytes == 0U || out.shape.empty() ||
        static_cast<std::uint64_t>(out.byte_offset) > planned.output_bytes ||
        out.size_bytes > planned.output_bytes - static_cast<std::uint64_t>(out.byte_offset)) {
      return "the published output does not lie in the kernel's output buffer";
    }
    if (first && !std::holds_alternative<client::Contiguous>(kernel.output.layout)) {
      // Alone on the route (Model::preprocess() between the Input and the Output), the stage
      // publishes the tensor it would hand the MLA. A dense one is described as any kernel
      // stage's published tensor; a tiled one is published as its packed storage, which this
      // stage's plan does not describe.
      return "it publishes a tiled or strided tensor as the route's output (a transform into the "
             "MLA's layout alone on the route)";
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
      kernel.operation + " on the " + target_text(recipe.target) +
      (kernel.parameters.empty() ? std::string() : "; " + parameters_text(kernel.parameters)) +
      "): " + std::to_string(in_tensor_bytes) +
      (kernel.packed_inputs.empty() ? std::string()
                                    : " at " + std::to_string(kernel.packed_inputs.front().offset) +
                                          " of the MLA's output buffer of " +
                                          std::to_string(planned.input_bytes)) +
      " -> " + std::to_string(out_tensor_bytes) + " bytes" +
      (kernel.packed_outputs.empty() ? std::string()
                                     : " at the front of the MLA's input buffer of " +
                                           std::to_string(planned.output_bytes)) +
      (last ? ", output " + out.dtype + " " + dims_text(out.shape) + " " +
                  std::to_string(out.size_bytes) + " bytes at " + std::to_string(out.byte_offset)
            : std::string());
  route.stages.push_back(std::move(planned));
  return {};
#endif
}

} // namespace

namespace {

// A seed that carries a frame's preprocess metadata (Semantic::preprocess) is read by the contracts
// GStreamer's build compiles with the seed as their ingress sample: the BoxDecode recipe checks
// the decode's geometry against it. A seed they refuse fails that build, with this error; one they
// take changes nothing on either backend (each frame's geometry is the image stage's own).
void check_seed_preprocess(const std::vector<std::shared_ptr<Node>>& route,
                           const GraphOptions& options, std::span<const Tensor> seed) {
  if (std::none_of(seed.begin(), seed.end(),
                   [](const Tensor& t) { return t.semantic.preprocess.has_value(); })) {
    return;
  }
  ContractCompileInput input;
  input.pipeline_label = "Graph::build(input)";
  input.processcvu_requested_run_target = options.processcvu_requested_run_target;
  input.processcvu = options.processcvu;
  input.ingress.ingress_sample = sample_from_tensors(TensorList(seed.begin(), seed.end()));
  si::ManifestBuildDiagnostics diagnostics;
  std::vector<std::string> errors;
  try {
    (void)pipeline_internal::rendered_stage_query::rendered_manifest_from_nodes(
        std::span<const std::shared_ptr<Node>>(route), input, &diagnostics);
    errors = diagnostics.errors;
  } catch (const std::exception& error) {
    errors.push_back(error.what());
  }
  if (errors.empty()) {
    return;
  }
  std::string message = "Graph::build(input): contract compilation failed";
  for (const std::string& error : errors) {
    message += "\n  - " + error;
  }
  pipeline_internal::error_util::throw_session_error(
      error_codes::kPipelineShape, message, {},
      "Supply the required input context and correct the reported stage contract.");
}

} // namespace

std::optional<si::SimaPluginStaticManifest>
dataflow_model_manifest(const std::vector<std::shared_ptr<Node>>& route,
                        const GraphOptions& options, pipeline_internal::DataflowNotCarried* why) {
  const auto decline = [why](std::string fact) {
    why->subject = "this model";
    why->fact = std::move(fact);
    return std::nullopt;
  };
  // The contracts of the route's nodes, compiled and rendered as GStreamer's build compiles them
  // (maybe_compile_build_result_contracts): the Graph's processcvu options decide each kernel
  // stage's target (the processcvu run target policy).
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
  // The kernel stages' checked recipes, by the function GStreamer's build lowers them with: the
  // canonical operation, its tensors and parameters, checked by the Kernels package for the
  // stage's target. Cold; no device is touched.
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
  return manifest;
}

namespace {

// Sample::caps_string of a route's output: the caps GStreamer's publication of the route's last
// stage gives its buffers (StageRecipePreparation.cpp), in GStreamer's text (CapsText). One tensor
// with a caps format: that format as the format and the dtype, the rank, each dim and the shape
// (processmla_make_tensor_caps_from_tensor). Several tensors, a tiled one, or the BoxDecode wire:
// the generic tensor-set's, the wire with its format (build_kernel_stage_publication).
std::string dataflow_output_caps(const DataflowPulledTensor& out) {
  pipeline_internal::CapsText caps("application/vnd.simaai.tensor");
  if (out.more.empty() && !out.format.empty() && !out.detection && !out.shape.empty()) {
    caps.string("format", out.format)
        .string("dtype", out.format)
        .integer("rank", static_cast<std::int32_t>(out.shape.size()));
    std::string csv;
    for (std::size_t i = 0; i < out.shape.size(); ++i) {
      caps.integer("dim" + std::to_string(i), static_cast<std::int32_t>(out.shape[i]));
      csv += (i == 0U ? "" : ",") + std::to_string(out.shape[i]);
    }
    caps.string("shape", csv);
  } else {
    caps.string("representation", "tensor-set").string("storage", "tensorbuffer");
    if (out.detection && !out.format.empty()) {
      caps.string("format", out.format);
    }
  }
  return caps.text();
}

} // namespace

std::shared_ptr<const DataflowModelRoute>
plan_dataflow_route_stages(const si::SimaPluginStaticManifest& manifest,
                           const InputOptions& declared, std::span<const Tensor> seed,
                           const std::string& ev74_obstacle,
                           pipeline_internal::DataflowNotCarried* why) {
  const auto decline = [why](std::string subject, std::string fact) {
    why->subject = std::move(subject);
    why->fact = std::move(fact);
    return nullptr;
  };
  const std::string of_model =
      manifest.model_id.empty() ? std::string() : " of model " + manifest.model_id;
  if (manifest.stages.empty()) {
    return decline("the model route" + of_model, "the model's nodes render no stage");
  }
  std::string kinds;
  for (const si::StageStaticSpec& stage : manifest.stages) {
    kinds += (kinds.empty() ? "" : ", ") + stage_kind_text(stage);
  }
  // The whole route as a subject, for a fact that is no one stage's.
  const std::string route_subject = "the model route" + of_model + " (" +
                                    std::to_string(manifest.stages.size()) + " stages: " + kinds +
                                    ")";
  const PlanInput in{declared, seed, ev74_obstacle};
  auto planned = std::make_shared<DataflowModelRoute>();
  std::size_t mla_stages = 0;
  for (std::size_t i = 0; i < manifest.stages.size(); ++i) {
    const si::StageStaticSpec& stage = manifest.stages[i];
    const bool first = i == 0U;
    const bool last = i + 1U == manifest.stages.size();
    std::string fact;
    bool board = false;
    if (stage.payload_kind == si::StagePayloadKind::ProcessMla) {
      ++mla_stages;
      fact = plan_mla_stage(stage, first, last ? nullptr : &manifest.stages[i + 1U], *planned);
    } else if (stage.native_kernel) {
      if (!first &&
          std::holds_alternative<DataflowModelStage::Kernel>(planned->stages.back().kind)) {
        fact = "it is the second of two kernel stages in a row (the route's stages: " + kinds + ")";
      } else {
        fact = plan_kernel_stage(stage, first, last, last ? nullptr : &manifest.stages[i + 1U],
                                 in, *planned, board);
      }
    } else {
      fact = "it is neither an MLA stage nor a kernel stage";
    }
    if (!fact.empty()) {
      if (board) {
        why->code = error_codes::kDispatcherUnavailable;
        why->remedy = pipeline_internal::DataflowRemedy::BackendOrFirmware;
      }
      return decline(stage_subject(i, stage, manifest.model_id), std::move(fact));
    }
  }
  // Exactly one MLA stage, or one kernel stage alone (Model::preprocess() between the Input and
  // the Output: the image preproc, or a canonical transform of the application's tensor).
  if (mla_stages != 1U &&
      !(mla_stages == 0U && planned->kernel_input && planned->stages.size() == 1U)) {
    return decline(route_subject, "it is a route with " + std::to_string(mla_stages) +
                                      " MLA stages: a route has exactly one MLA stage, or is "
                                      "one kernel stage alone");
  }
  // Each stage reads the buffer the stage in front writes. Beside the MLA stage a kernel stage's
  // port was brought to the MLA's buffer where that is the kernel's tensor and the MLA's padding
  // (pad_to_mla_input, view_of_mla_output), so what differs here is neither.
  for (std::size_t i = 0; i + 1U < planned->stages.size(); ++i) {
    if (planned->stages[i].output_bytes != planned->stages[i + 1U].input_bytes) {
      return decline(route_subject,
                     "stage '" + planned->stages[i].key + "' writes " +
                         std::to_string(planned->stages[i].output_bytes) + " bytes and stage '" +
                         planned->stages[i + 1U].key + "' reads " +
                         std::to_string(planned->stages[i + 1U].input_bytes) +
                         ": not one whole buffer handed on, nor a tensor at the front of an MLA "
                         "input buffer that is the tensor's bytes rounded up to a multiple of " +
                         std::to_string(kMlaBufferGranule));
    }
  }
  planned->output.beyond_leaf = planned->output.frame_semantics || !planned->output.more.empty();
  planned->output.caps = dataflow_output_caps(planned->output);
  // A model route's process keeps the output a GStreamer Run of it would give: the drivers' lines
  // off stdout and stderr (Core's filter, which GStreamer's initialisation installs elsewhere).
  pipeline_internal::ux::install_device_log_filters();
  return planned;
}

std::shared_ptr<const DataflowModelRoute>
plan_dataflow_model_route(const std::vector<std::shared_ptr<Node>>& route,
                          const GraphOptions& options, const InputOptions& declared,
                          std::span<const Tensor> seed,
                          pipeline_internal::DataflowNotCarried* why) {
  std::optional<si::SimaPluginStaticManifest> manifest =
      dataflow_model_manifest(route, options, why);
  if (!manifest) {
    return nullptr;
  }
  check_seed_preprocess(route, options, seed);
  // A prefix or suffix in `options` names the stages as GStreamer's build names its elements
  // (apply_name_transform over the manifest's stages): the route's stage key, and the Sample label
  // drawn from it, carry it. Without one a stage keeps its name in the model package.
  const NameTransform names = make_name_transform(options);
  if (name_transform_enabled(names)) {
    for (si::StageStaticSpec& stage : manifest->stages) {
      stage.element_name = apply_name_transform(names, stage.element_name);
      stage.logical_stage_id = apply_name_transform(names, stage.logical_stage_id);
      for (si::InputBindingStaticSpec& binding : stage.input_bindings) {
        binding.src_stage_id = apply_name_transform(names, binding.src_stage_id);
      }
    }
  }
  std::string ev74_obstacle;
#if defined(SIMANEAT_WITH_DATAFLOW_SOC_KERNELS)
  // Asked only of a route with a kernel stage on the EV74. Cold: sysfs is read; nothing is
  // opened on the device.
  if (std::any_of(
          manifest->stages.begin(), manifest->stages.end(), [](const si::StageStaticSpec& stage) {
            return stage.native_kernel && stage.native_kernel->target != client::Hardware::A65;
          })) {
    (void)df::soc::ev74_runs_kernels_firmware(&ev74_obstacle);
  }
#endif
  return plan_dataflow_route_stages(*manifest, declared, seed, ev74_obstacle, why);
}

InputOptions dataflow_model_input_contract(const DataflowModelRoute& route,
                                           const InputOptions& declared) {
  InputOptions contract;
  contract.memory_policy = declared.memory_policy;
  contract.pool_max_buffers = declared.pool_max_buffers;
#if defined(SIMANEAT_WITH_DATAFLOW_SOC_KERNELS)
  if (route.frames) {
    const DataflowFrames& frames = *route.frames;
    contract.payload_type = PayloadType::Image;
    contract.format = frames.format;
    contract.max_width = static_cast<int>(frames.max_width);
    contract.max_height = static_cast<int>(frames.max_height);
    return contract;
  }
#endif
  if (route.frame_input || !route.kernel_input || route.input_shape.empty()) {
    return declared;
  }
  contract.payload_type = PayloadType::Tensor;
  contract.shape = route.input_shape;
  contract.dtype = route.input_dtype;
  return contract;
}

std::string bind_dataflow_model_input(const DataflowModelRoute& route,
                                      DataflowInputBinding& binding) {
  // The first stage's input: the MLA's (an MLA-only route), or the model's public tensor. In
  // front of an image preproc whose frames change size, the port's bound.
  const std::vector<std::int64_t>& want = route.frames ? route.frame_shape : route.input_shape;
  const std::string input = route.kernel_input ? "the model's input" : "the model's MLA input";
  // An MLA stage first, with a buffer longer than the model's tensor: the input is the buffer, and
  // the refusal of the model's tensor says what the buffer is (GStreamer's inference-only ingress
  // refuses that tensor by the same count of bytes).
  const std::string expected =
      input + " is " +
      (route.planar_input ? std::string("an NV12 frame in ")
       : route.frames     ? "a frame of up to " + dims_text(want) + " in "
       : route.input_padding != 0U
           ? dims_text(want) + " in a buffer of "
           : dims_text(want) + " of ") +
      std::to_string(route.input_bytes) + " bytes" +
      (route.input_padding != 0U
           ? " (the tensor's " + std::to_string(route.input_bytes - route.input_padding) +
                 " and " + std::to_string(route.input_padding) +
                 " of the MLA's padding)"
           : std::string());
  const bool planes = binding.planes != DataflowInputBinding::Planes::None;
  if (binding.is_record() ||
      (planes && !(route.planar_input && binding.planes == DataflowInputBinding::Planes::Nv12))) {
    return "an item of several tensors or planes where one dense tensor is expected; " + expected;
  }
  if (route.planar_input && !planes) {
    return "a dense tensor where the model reads NV12 frames (a tensor with a Y and a UV "
           "plane); " +
           expected;
  }
  if (binding.spec.memory == df::MemoryKind::Host) {
    return "an input in system memory (InputOptions::memory_policy SystemMemory, or "
           "use_simaai_pool = false) in front of a model: the MLA takes device memory";
  }
  if (binding.dtype != route.input_dtype) {
    return "dtype differs from " + input + "; " + expected;
  }
  const auto describe = [&binding] {
    std::array<char, df::kDescribeMax> text{};
    df::Writer writer{std::span<char>(text)};
    df::describe(binding.spec, writer);
    binding.text = std::string(writer.view());
  };
  if (route.planar_input) {
    // NV12 frames: the port is one buffer of bytes, the frame at the bound as it is; the planes
    // are the kernel's two views of it (soc::KernelOp::Options::Packing). The binding is that
    // byte tensor from here on: a pushed frame is brought to it by the frames' own push.
    df::TensorSpec bytes;
    bytes.dtype = df::DType::UInt8;
    bytes.shape.rank = 1;
    bytes.shape.dims[0] = df::Dim(want.front());
    bytes.layout = df::Dense{};
    binding.spec.payload = bytes;
    if (const df::Status status = df::validate(binding.spec)) {
      return std::string("the input's spec is not valid: ") + status->message;
    }
    binding.planes = DataflowInputBinding::Planes::None;
    binding.rank = 1;
    describe();
  } else if (route.frame_input) {
    // In front of the image preproc the port is the kernel's tensor: the frame the route was
    // built for, wherever in the declaration's bounds that is, or (frames of a changing size)
    // every frame up to the route's bound. The item's shape must admit it, with or without the
    // kernel's leading batch axis.
    auto* spec = std::get_if<df::TensorSpec>(&binding.spec.payload);
    bool fits = spec && std::holds_alternative<df::Dense>(spec->layout) && spec->shape.rank != 0 &&
                spec->shape.rank <= want.size();
    const std::size_t lead = fits ? want.size() - spec->shape.rank : 0U;
    for (std::size_t i = 0; fits && i < want.size(); ++i) {
      // A changing size: the item's own height and width (a seed's, a first push's) bind
      // nothing; its other axes must be the frame's.
      const bool sized = route.frames && (i == kHeightAxis || i == kWidthAxis);
      fits = i < lead ? want[i] == 1 : sized || spec->shape.dims[i - lead].contains(want[i]);
    }
    if (!fits) {
      return std::string(route.frames ? "a tensor that is not a frame the model's image stage "
                                        "takes; "
                                      : "a tensor that is not a frame of the size the Run was "
                                        "built for; ") +
             expected;
    }
    spec->shape.rank = static_cast<std::uint8_t>(want.size());
    for (std::size_t i = 0; i < want.size(); ++i) {
      spec->shape.dims[i] = route.frames && (i == kHeightAxis || i == kWidthAxis)
                                ? df::Dim(1, want[i])
                                : df::Dim(want[i]);
    }
    if (const df::Status status = df::validate(binding.spec)) {
      return std::string("the input's spec is not valid: ") + status->message;
    }
    binding.rank = want.size();
    describe();
  } else if (const std::string why = fix_dataflow_binding_at_bound(binding); !why.empty()) {
    // A bounded axis (InputOptions::max_*) is fixed at its bound: the model's input has one size.
    return why;
  }
  if (df::slot_bytes(binding.spec) != route.input_bytes) {
    return "a tensor of " + std::to_string(df::slot_bytes(binding.spec)) + " bytes; " + expected;
  }
  if (route.ev74_input && binding.spec.memory != df::MemoryKind::Ev74) {
    // The compiler gives such an input an Ev74 pool whatever was declared.
    return "an input outside EV74 memory in front of a kernel stage on the EV74, which reaches "
           "linux,cma memory only";
  }
  if (route.kernel_input && !route.frame_input) {
    // The kernel's port is the kernel's tensor: the Feed's spec must be exactly it (the push
    // brings a tensor without the leading unit axes to it, dataflow_model_input_view). A frame
    // route's spec was set to it above.
    const auto* spec = std::get_if<df::TensorSpec>(&binding.spec.payload);
    bool same =
        spec && spec->shape.rank == want.size() && std::holds_alternative<df::Dense>(spec->layout);
    for (std::size_t i = 0; same && i < want.size(); ++i) {
      same = spec->shape.dims[i].max == want[i];
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
  if (pulled.detection) {
    tensor.semantic.detection = pulled.detection;
  }
}

namespace {
// The preprocess facts of the frame whose stamp's user word is `word`, on a route whose frames
// change size (`frames`, defined with the table's functions below); null when `frames` is.
std::shared_ptr<const PreprocessRuntimeMeta> frame_facts(const DataflowFrames* frames,
                                                         std::uint64_t word);
} // namespace

[[gnu::cold]] void fill_dataflow_stage_tensors(
    const DataflowPulledTensor& pulled, const std::shared_ptr<TensorBuffer>& storage,
    const df::Message& message, const std::shared_ptr<const TensorEnvelope>& envelope,
    TensorList& out) {
  // The frame's own preprocess facts, on a route whose frames change size: by the geometry the
  // push stated in the message's stamp.
  const std::shared_ptr<const PreprocessRuntimeMeta> own =
      pulled.frames ? frame_facts(pulled.frames, message.stamp().user) : nullptr;
  const auto fill = [&](const DataflowPulledTensor& one) {
    Tensor& tensor = out.emplace_back();
    // The stage's output buffer is a slot of its own pool (or its Owned copy): from byte 0.
    dataflow_device::fill_stage_leaf(tensor, one, storage, 0);
    if (one.format.empty()) {
      // No format in the stage's caps (several tensors, or a tiled one whose description follows):
      // GStreamer's pull leaves no TessSpec for it.
      tensor.semantic.tess.reset();
    }
    if (one.frame_semantics) {
      apply_dataflow_frame_semantics(one, tensor);
      if (own && one.preprocess) {
        tensor.semantic.preprocess = *own;
      }
    }
    tensor.envelope = envelope;
  };
  out.reserve(out.size() + 1U + pulled.more.size());
  fill(pulled);
  for (const DataflowPulledTensor& next : pulled.more) {
    fill(next);
  }
}

namespace {

// What a pushed frame carries that the route's image preproc stage does not take.
constexpr const char* kFrameWithRois =
    "it carries ROI metadata (Semantic::preprocess with an ROI or an ROI list) in front of the "
    "image preproc stage";

// A frame's image format as the route's frames name theirs (DataflowFrames::format).
[[maybe_unused]] std::string_view pixel_format_text(ImageSpec::PixelFormat format) {
  switch (format) {
  case ImageSpec::PixelFormat::RGB:
    return "RGB";
  case ImageSpec::PixelFormat::BGR:
    return "BGR";
  case ImageSpec::PixelFormat::GRAY8:
    return "GRAY8";
  case ImageSpec::PixelFormat::NV12:
    return "NV12";
  case ImageSpec::PixelFormat::NV16:
    return "NV16";
  case ImageSpec::PixelFormat::I420:
    return "I420";
  case ImageSpec::PixelFormat::UNKNOWN:
    break;
  }
  return "";
}

[[noreturn]] void throw_frame_not_carried(const char* where, const char* fact) {
  pipeline_internal::throw_dataflow_not_carried(where,
                                                {error_codes::kInputShape, "this frame", fact});
}

} // namespace

void refuse_dataflow_frame(const DataflowModelRoute& route, const Tensor& tensor,
                           const char* where) {
  if (tensor.semantic.preprocess && tensor.semantic.preprocess->has_roi_list()) {
    throw_frame_not_carried(where, kFrameWithRois);
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
            std::to_string(tensor.shape[lead]) + " where the Input declares " +
            std::to_string(want[2]) + "x" + std::to_string(want[1]) +
            ": on the Dataflow backend a declared frame size (InputOptions width and height, or "
            "a shape with every axis fixed) is the size of every frame; for frames whose size "
            "changes, declare InputOptions::max_width and max_height instead, or no size");
  }
}

#if !defined(SIMANEAT_WITH_DATAFLOW_SOC_KERNELS)

// A build without the kernel operator plans no image stage: no route has frames.
namespace {
std::shared_ptr<const PreprocessRuntimeMeta> frame_facts(const DataflowFrames*, std::uint64_t) {
  return nullptr;
}
} // namespace

std::uint64_t dataflow_frame_word(const DataflowModelRoute&, const Tensor&, const char*) {
  return 0;
}

bool dataflow_planar_frame_wraps(const DataflowModelRoute&, const Tensor&) {
  return false;
}

bool copy_dataflow_planar_frame(const DataflowModelRoute&, const Tensor&, std::span<std::byte>) {
  return false;
}

#endif

const Tensor& dataflow_model_input_view(const DataflowModelRoute& route, const Tensor& tensor,
                                        Tensor& view) {
  const std::vector<std::int64_t>& want = route.frames ? route.frame_shape : route.input_shape;
  if (!route.kernel_input || route.planar_input || tensor.shape.size() >= want.size() ||
      tensor.shape.empty() || !tensor.is_dense() || !tensor.is_contiguous()) {
    return tensor;
  }
  const std::size_t lead = want.size() - tensor.shape.size();
  for (std::size_t i = 0; i < want.size(); ++i) {
    // A changing size: the frame's own height and width (dataflow_frame_word holds them
    // against the bound).
    const bool sized = route.frames && (i == kHeightAxis || i == kWidthAxis);
    if (i < lead ? want[i] != 1 : !sized && want[i] != tensor.shape[i - lead]) {
      return tensor;
    }
  }
  view = tensor;
  view.shape.assign(lead, 1);
  view.shape.insert(view.shape.end(), tensor.shape.begin(), tensor.shape.end());
  std::int64_t stride = dtype_size(tensor.dtype);
  view.strides_bytes.assign(view.shape.size(), 0);
  for (std::size_t i = view.shape.size(); i-- > 0U;) {
    view.strides_bytes[i] = stride;
    stride *= std::max<std::int64_t>(view.shape[i], 1);
  }
  // The axes' names, where the tensor names them: a frame's height, width and channels behind the
  // stage's one batch axis (a Model's own dummy frame, which says no layout). Otherwise none.
  using Axis = TensorAxisSemantic;
  if (lead == 1U && tensor.axis_semantics == std::vector<Axis>{Axis::H, Axis::W, Axis::C}) {
    view.axis_semantics = {Axis::N, Axis::H, Axis::W, Axis::C};
  } else if (!view.axis_semantics.empty()) {
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

// The kernel's tensors of one side, in the runtime's vocabulary: the side's one tensor, or, for a
// side whose tensors are views of one buffer, every one of them. False when one has no
// description in the runtime.
bool runtime_side(const client::TensorSpec& one, const std::vector<DataflowModelStage::View>& views,
                  std::vector<df::TensorSpec>& out) {
  const auto add = [&out](const client::TensorSpec& tensor) {
    const std::optional<df::TensorSpec> spec = runtime_tensor(tensor);
    if (spec) {
      out.push_back(*spec);
    }
    return spec.has_value();
  };
  if (views.empty()) {
    return add(one);
  }
  return std::all_of(views.begin(), views.end(),
                     [&add](const DataflowModelStage::View& view) { return add(view.tensor); });
}

// Whether the kernel's tensors of one side have the bytes the plan read from the stage Core
// lowered for GStreamer.
bool same_bytes(std::span<const df::PortSpec> made, std::uint64_t bytes,
                const std::vector<DataflowModelStage::View>& views) {
  if (views.empty()) {
    return made.size() == 1U && df::slot_bytes(made.front()) == bytes;
  }
  bool same = made.size() == views.size();
  for (std::size_t i = 0; same && i < views.size(); ++i) {
    same = df::slot_bytes(made[i]) == views[i].bytes;
  }
  return same;
}

// The soc::Kernel of a kernel stage. Throws NeatError when the Kernels package refuses the
// request or the kernel's bytes are not the stage's.
std::shared_ptr<df::soc::Kernel> make_kernel(const DataflowModelStage& stage, const char* where,
                                             pipeline_internal::DataflowErrorSite site) {
  const DataflowModelStage::Kernel& request = std::get<DataflowModelStage::Kernel>(stage.kind);
  const std::string name = "kernel stage " + stage.key + " (" + request.operation + ")";
  std::vector<df::TensorSpec> inputs, outputs;
  if (!runtime_side(request.input, request.packed_inputs, inputs) ||
      !runtime_side(request.output, request.packed_outputs, outputs)) {
    pipeline_internal::error_util::throw_session_error(
        error_codes::kPipelineShape,
        std::string(where) + ": " + name +
            " has a tensor the Dataflow runtime does not describe (Dataflow backend)");
  }
  std::vector<df::soc::KernelParameter> parameters;
  for (const client::Parameter& parameter : request.parameters) {
    parameters.push_back(runtime_parameter(parameter));
  }
  df::Result<std::shared_ptr<df::soc::Kernel>> made = df::soc::Kernel::make(
      request.operation,
      request.ev74 ? df::soc::Kernel::Target::Ev74 : df::soc::Kernel::Target::A65, inputs, outputs,
      parameters);
  if (const df::Error* error = df::failure(made)) {
    pipeline_internal::throw_dataflow_error(*error, where, name, site);
  }
  std::shared_ptr<df::soc::Kernel> kernel =
      std::get<std::shared_ptr<df::soc::Kernel>>(std::move(made));
  if (!same_bytes(kernel->inputs(), stage.input_bytes, request.packed_inputs) ||
      !same_bytes(kernel->outputs(), stage.output_bytes, request.packed_outputs)) {
    pipeline_internal::error_util::throw_session_error(
        error_codes::kPipelineShape,
        std::string(where) + ": " + name + " does not have the " +
            std::to_string(stage.input_bytes) + " and " + std::to_string(stage.output_bytes) +
            " bytes of the stage Core lowered for GStreamer (Dataflow backend)");
  }
  return kernel;
}

// ---- the frames' table (DataflowFrames) ---------------------------------------------------

using FrameGeometry = std::shared_ptr<const DataflowFrames::Geometry>;

// A frame geometry this backend does not carry: the fact about "this frame" (the one error form),
// where a std::string in its place is a mistake of the frame.
struct FrameNotCarried {
  std::string fact;
};
using FrameGeometryOrWhy = std::variant<FrameGeometry, std::string, FrameNotCarried>;

// Why `found` is no geometry, as one text.
std::string no_geometry_text(const FrameGeometryOrWhy& found) {
  if (const std::string* why = std::get_if<std::string>(&found)) {
    return *why;
  }
  const FrameNotCarried* const not_carried = std::get_if<FrameNotCarried>(&found);
  return not_carried ? not_carried->fact : std::string();
}

// What Core derives from one frame geometry, by the functions the plan used for the port's
// bound: the image stage's request for it and the facts it publishes
// (PreprocFrameAdapter::checked_fixed_frame), what the BoxDecode stage runs for frames of it
// (boxdecode_geometry_for_frames), and the two kernels made of them. Cold: the Kernels package
// checks the preproc's request and the decode's variant. The geometry, or why it has none.
FrameGeometryOrWhy make_frame_geometry(const DataflowFrames& frames, std::uint64_t word,
                                       const char* where) {
  using Adapter = DataflowFrames::Adapter;
  const std::int64_t width = DataflowFrames::width_of(word);
  const std::int64_t height = DataflowFrames::height_of(word);
  if ((word & DataflowFrames::kFrameWord) == 0U || width <= 0 || height <= 0) {
    return std::string("the message's stamp states no frame geometry");
  }
  auto fixed = Adapter::checked_fixed_frame(
      frames.policy, static_cast<std::size_t>(width), static_cast<std::size_t>(height),
      static_cast<std::size_t>(frames.nv12 ? width : width * frames.channels));
  if (const auto* error = std::get_if<client::Error>(&fixed)) {
    return error->field + ": " + error->message;
  }
  Adapter::FixedFrame& request = std::get<Adapter::FixedFrame>(fixed);
  auto inspected = client::inspect_kernel(request.executable);
  if (const auto* error = std::get_if<client::Error>(&inspected)) {
    return "the Kernels package gave no contract: " + error->message;
  }
  const client::KernelInspection& contract = std::get<client::KernelInspection>(inspected);
  const simaai::gst::TensorBufferPreprocessFacts* const facts =
      simaai::gst::tensor_buffer_preprocess_facts(request.facts);
  const std::size_t planes = frames.nv12 ? 2U : 1U;
  if (!facts || request.inputs.size() != planes || contract.input_requirements().size() != planes) {
    return std::string("the frame's kernel request has other tensors than the frame's planes");
  }
  auto geometry = std::make_shared<DataflowFrames::Geometry>();
  geometry->word = word;
  try {
    geometry->preprocess = preprocess_from_compact_facts(*facts);
    // An oriented-box decode maps a box back with one scale for both axes (the Kernels package's
    // rule and the resolver's). The letterbox has two wherever the image's shorter side does
    // not land on a whole number of canvas pixels.
    if (const PreprocessRuntimeMeta& own = geometry->preprocess;
        frames.decode && frames.decode->boxdecode_program->uniform_transform &&
        own.affine_m00 != own.affine_m11) {
      char scales[96];
      std::snprintf(scales, sizeof scales, "%.9g across, %.9g down", own.affine_m00,
                    own.affine_m11);
      return FrameNotCarried{"it is resized onto the model's " + std::to_string(own.resized_width) +
                             "x" + std::to_string(own.resized_height) +
                             " canvas with two different scales (" + scales +
                             "), and the oriented-box decode maps boxes back with one scale"};
    }
    DataflowModelStage stage = frames.preproc_stage;
    DataflowModelStage::Kernel& kernel = std::get<DataflowModelStage::Kernel>(stage.kind);
    kernel.parameters = std::move(request.parameters);
    if (frames.nv12) {
      // The planes at the port's two offsets, each as long as this frame's.
      for (std::size_t i = 0; i < planes; ++i) {
        kernel.packed_inputs[i].tensor = request.inputs[i];
        kernel.packed_inputs[i].bytes = contract.input_requirements()[i].storage_bytes;
      }
    } else {
      kernel.input = request.inputs.front();
      stage.input_bytes = contract.input_requirements().front().storage_bytes;
    }
    geometry->preproc = make_kernel(stage, where, pipeline_internal::DataflowErrorSite::Running);
    if (frames.decode) {
      // The resolver's checks for such a frame (its resize mode, its size, its affine); then the
      // Run's kernel, or the variant the stage binds for a frame's geometry with the frame's
      // values: "image_size", or "image_geometry" with the model-to-original transform as well.
      auto decode =
          si::boxdecode_geometry_for_frames(*frames.decode, boxdecode_facts(&geometry->preprocess));
      if (const auto* error = std::get_if<client::Error>(&decode)) {
        return "the decode does not take such frames: " + error->message;
      }
      const si::BoxDecodeFrameGeometry& selected = std::get<si::BoxDecodeFrameGeometry>(decode);
      const std::shared_ptr<df::soc::Kernel>& base = frames.decode_kernel;
      geometry->decode = base;
      if (!selected.variant.empty()) {
        using Scalar = df::soc::KernelParameter::Scalar;
        std::vector<df::soc::KernelParameter> values;
        values.push_back(
            {"image_size", std::vector<Scalar>{Scalar{std::int64_t{selected.image_size[0]}},
                                               Scalar{std::int64_t{selected.image_size[1]}}}});
        if (selected.model_to_original) {
          // The FP32 values the resolver hands the variant, each widened once.
          std::vector<Scalar> transform;
          for (const float value : *selected.model_to_original) {
            transform.push_back(Scalar{static_cast<double>(value)});
          }
          values.push_back({"model_to_original", std::move(transform)});
        }
        df::Result<std::shared_ptr<df::soc::Kernel>> varied = base->with(selected.variant, values);
        if (const df::Error* error = df::failure(varied)) {
          return std::string("the decode does not take such frames: ") + error->message;
        }
        geometry->decode = std::get<std::shared_ptr<df::soc::Kernel>>(std::move(varied));
      }
    }
  } catch (const std::exception& error) {
    return std::string(error.what());
  }
  return FrameGeometry(std::move(geometry));
}

// The table's first entry: the port's bound's geometry, whose kernels the Run loaded (the
// operators' own, made from the plan's requests for that geometry), so that a frame of the
// bound's size (the seed's, typically) makes no kernel at its first push; and the BoxDecode
// kernel each geometry's decode is a variant of.
void seed_dataflow_frames(DataflowFrames& frames, std::shared_ptr<df::soc::Kernel> preproc,
                          std::shared_ptr<df::soc::Kernel> decode) {
  frames.decode_kernel = decode;
  auto bound = std::make_shared<DataflowFrames::Geometry>();
  bound->word = DataflowFrames::frame_word(frames.max_width, frames.max_height);
  bound->preprocess = frames.bound_preprocess;
  bound->preproc = std::move(preproc);
  bound->decode = std::move(decode);
  std::lock_guard<std::mutex> lock(frames.mu);
  frames.kept[0] = std::move(bound);
  frames.used[0] = ++frames.clock;
}

// The table's entry for the geometry `word` states, made and kept when the table has none (the
// entry used longest ago leaves; whoever still reads it keeps it).
FrameGeometryOrWhy frame_geometry(const DataflowFrames& frames, std::uint64_t word,
                                  const char* where) {
  const auto kept = [&frames, word]() -> FrameGeometry {
    for (std::size_t i = 0; i < DataflowFrames::kKept; ++i) {
      if (frames.kept[i] && frames.kept[i]->word == word) {
        frames.used[i] = ++frames.clock;
        return frames.kept[i];
      }
    }
    return nullptr;
  };
  {
    std::lock_guard<std::mutex> lock(frames.mu);
    if (FrameGeometry found = kept()) {
      return found;
    }
  }
  FrameGeometryOrWhy made = make_frame_geometry(frames, word, where);
  if (!std::holds_alternative<FrameGeometry>(made)) {
    return made;
  }
  FrameGeometry replaced; // released outside the lock
  std::lock_guard<std::mutex> lock(frames.mu);
  if (FrameGeometry found = kept()) {
    return found; // another thread made it meanwhile
  }
  std::size_t slot = 0;
  for (std::size_t i = 0; i < DataflowFrames::kKept; ++i) {
    if (!frames.kept[i]) {
      slot = i;
      break;
    }
    if (frames.used[i] < frames.used[slot]) {
      slot = i;
    }
  }
  replaced = std::move(frames.kept[slot]);
  frames.kept[slot] = std::get<FrameGeometry>(made);
  frames.used[slot] = ++frames.clock;
  return made;
}

std::shared_ptr<const PreprocessRuntimeMeta> frame_facts(const DataflowFrames* frames,
                                                         std::uint64_t word) {
  if (!frames) {
    return nullptr;
  }
  FrameGeometryOrWhy found = frame_geometry(*frames, word, "Run::pull");
  if (!std::holds_alternative<FrameGeometry>(found)) {
    pipeline_internal::error_util::throw_session_error(
        error_codes::kRuntimePull,
        "Run::pull: no preprocess facts for the pulled frame's geometry: " +
            no_geometry_text(found) + " (Dataflow backend)");
  }
  const FrameGeometry& geometry = std::get<FrameGeometry>(found);
  return std::shared_ptr<const PreprocessRuntimeMeta>(geometry, &geometry->preprocess);
}

// A variant's kernel for the Worker (soc::KernelOp::Options::Variants::make): the table's, which
// the frame's push made; made again when the table let it go meanwhile (a geometry its push
// passed, so a failure here is only out of memory: its text is the operator's, below).
df::Result<std::shared_ptr<df::soc::Kernel>>
frame_kernel(const DataflowFrames& frames, std::uint64_t word, bool decode) noexcept {
  try {
    FrameGeometryOrWhy found = frame_geometry(frames, word, "the Dataflow route's operator");
    if (const FrameGeometry* geometry = std::get_if<FrameGeometry>(&found)) {
      return decode ? (*geometry)->decode : (*geometry)->preproc;
    }
  } catch (...) {
  }
  df::Error error = df::make_error(df::ErrorCode::InvalidArgument, df::Phase::Process,
                                   "no kernel for a frame's geometry, which its push had made");
  error.origin = df::Origin::Core;
  return error;
}

#endif // SIMANEAT_WITH_DATAFLOW_SOC_KERNELS

std::shared_ptr<df::soc::MlaModel> load_model(const DataflowModelStage& stage, const char* where) {
  const std::string& path = std::get<DataflowModelStage::Mla>(stage.kind).model_path;
  df::Result<std::shared_ptr<df::soc::MlaModel>> loaded = df::soc::MlaModel::load(path);
  if (const df::Error* error = df::failure(loaded)) {
    pipeline_internal::throw_dataflow_error(*error, where, "MLA model " + path,
                                            pipeline_internal::DataflowErrorSite::Build);
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
        std::string(where) + ": the MLA model " + path + " has " + std::to_string(inputs.size()) +
            " input(s) and " + std::to_string(outputs.size()) + " output(s)" +
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

// A port that is one buffer of bytes: UInt8 [bytes] dense (what soc::MlaModel::outputs() gives
// for a model's output buffer, and what a packed side of soc::KernelOp is).
df::PortSpec byte_port(std::uint64_t bytes, df::MemoryKind memory) {
  df::TensorSpec tensor;
  tensor.dtype = df::DType::UInt8;
  tensor.shape.rank = 1;
  tensor.shape.dims[0] = df::Dim(static_cast<std::int64_t>(bytes));
  tensor.layout = df::Dense{};
  df::PortSpec port;
  port.payload = tensor;
  port.memory = memory;
  return port;
}

// Core's BoxDecode wire serializer as an operator behind the BoxDecode kernel (GStreamer runs it
// as the stage's host task): one message of the kernel's outputs in, one wire out, written on
// the Worker by the stage's own serializer. Nothing is allocated per message. The wire leaves in
// device memory (Dms0), as every model route's last stage's output does (Core's pull hands a
// DMA-BUF out), written here by the CPU: the port is job-written, so the write epoch on the loan
// is begun here and ended by the Worker when the step, submitted and completed at once, retires.
class BoxDecodeWireOp final : public df::Operator {
public:
  BoxDecodeWireOp(const df::PortSpec& input, const df::PortSpec& output,
                  const DataflowModelStage::Wire& wire)
      : in_spec_(input), out_spec_(output), program_(wire.program), write_(wire.write),
        inputs_(wire.inputs.size()) {
    offsets_.reserve(wire.inputs.size());
    for (std::size_t i = 0; i < wire.inputs.size(); ++i) {
      offsets_.push_back(wire.inputs[i].offset);
      inputs_[i] = {nullptr, wire.inputs[i].bytes};
    }
  }

  [[nodiscard]] df::OperatorSpec describe() override {
    df::OperatorSpec spec;
    spec.inputs.push_back({.name = "in", .port = &in_, .spec = &in_spec_});
    spec.outputs.push_back({.name = "out", .port = &out_, .spec = &out_spec_, .job_written = true});
    return spec;
  }

  [[nodiscard]] df::Result<df::InFlight> process(df::Step& step) noexcept override {
    df::Message message = in_.receive();
    if (const df::Status begun = step.begin_write(0)) [[unlikely]] {
      return *begun;
    }
    const std::byte* const from = message.bytes().data();
    const std::span<std::byte> to = step.loan(0).bytes();
    for (std::size_t i = 0; i < inputs_.size(); ++i) {
      inputs_[i].data = from + offsets_[i];
    }
    if (write_(program_.get(), inputs_, {to.data(), to.size()})) [[unlikely]] {
      df::Error error = df::make_error(df::ErrorCode::Internal, df::Phase::Process,
                                       "the BoxDecode wire serializer refused the kernel's "
                                       "outputs");
      error.origin = df::Origin::Core;
      return error;
    }
    const df::InFlight job = step.submit({&message, 1}, df::derive(message.stamp()));
    job.complete();
    return job;
  }

private:
  df::In<df::Tensor> in_;
  df::Out<df::Tensor> out_;
  df::PortSpec in_spec_, out_spec_;
  std::shared_ptr<const void> program_;
  decltype(DataflowModelStage::Wire::write) write_;
  std::vector<std::size_t> offsets_; ///< each kernel output's offset in the message
  std::vector<client::ConstBytes>
      inputs_; ///< the serializer's inputs: bytes fixed, data per message
};

} // namespace

#if defined(SIMANEAT_WITH_DATAFLOW_SOC_KERNELS)

std::uint64_t dataflow_frame_word(const DataflowModelRoute& route, const Tensor& tensor,
                                  const char* where) {
  const DataflowFrames& frames = *route.frames;
  const auto refuse = [where](const std::string& why) {
    pipeline_internal::error_util::throw_session_error(error_codes::kInputShape,
                                                       std::string(where) + ": " + why);
  };
  // A frame that says it is of another image format than the route's frames: GStreamer's push
  // check (validate_public_input) and its text, before anything else is read of it.
  if (tensor.semantic.image && pixel_format_text(tensor.semantic.image->format) != frames.format) {
    refuse("simaai::neat::Tensor image format mismatch");
  }
  std::int64_t width = 0, height = 0;
  if (frames.nv12) {
    const Plane* const luma = tensor.try_plane(PlaneRole::Y);
    if (tensor.dtype != TensorDType::UInt8 || !tensor.is_nv12() || !luma ||
        luma->shape.size() < 2U) {
      refuse("a tensor that is not an NV12 frame (UInt8, a Y and a UV plane) where the model "
             "reads NV12 frames (Dataflow backend)");
    }
    height = luma->shape[0];
    width = luma->shape[1];
  } else {
    const std::size_t rank = tensor.shape.size();
    const std::size_t lead = rank == 4U ? 1U : 0U;
    if (tensor.is_composite() || tensor.dtype != TensorDType::UInt8 || !tensor.is_dense() ||
        (rank != 3U && rank != 4U) || (rank == 4U && tensor.shape[0] != 1) ||
        tensor.shape[lead + 2U] != frames.channels) {
      refuse("a tensor that is not a packed UInt8 " + frames.format +
             " frame of shape [height, width, " + std::to_string(frames.channels) +
             "] (or with a batch axis of 1 in front), which is what the model's image preproc "
             "stage reads (Dataflow backend)");
    }
    height = tensor.shape[lead];
    width = tensor.shape[lead + 1U];
  }
  // An upstream stage's preprocess metadata without an ROI is not read, as GStreamer's image
  // stage does not read it on a frame that is not the build's seed: the frame is the stage's
  // input as it is, and the pull publishes the stage's own facts.
  if (tensor.semantic.preprocess && tensor.semantic.preprocess->has_roi_list()) {
    throw_frame_not_carried(where, kFrameWithRois);
  }
  if (width <= 0 || height <= 0 || width > frames.envelope_width ||
      height > frames.envelope_height) {
    refuse("a frame of " + std::to_string(width) + "x" + std::to_string(height) +
           " where the model's image preproc stage takes frames up to " +
           std::to_string(frames.envelope_width) + "x" + std::to_string(frames.envelope_height) +
           " (the capacity it is prepared for): push frames within that size (Dataflow "
           "backend)");
  }
  // Above the Input's capacity (its declared bound, or its seed's size): GStreamer's refusal.
  const auto above = [where](const char* axis, std::int64_t actual, std::int64_t capacity) {
    pipeline_internal::error_util::throw_session_error(
        error_codes::kInputCapacity,
        pipeline_internal::shape_limit_exceeded_message(where, axis, static_cast<int>(actual),
                                                        static_cast<int>(capacity)),
        {}, pipeline_internal::shape_limit_fix_hint(axis, static_cast<int>(actual)));
  };
  if (width > frames.max_width) {
    above("width", width, frames.max_width);
  }
  if (height > frames.max_height) {
    above("height", height, frames.max_height);
  }
  const std::uint64_t word = DataflowFrames::frame_word(width, height);
  if (word == frames.passed.load(std::memory_order_relaxed)) {
    return word;
  }
  // The geometry's kernels are made here when the table has none, so that the Worker is never
  // asked for one the Kernels package refuses.
  const FrameGeometryOrWhy found = frame_geometry(frames, word, where);
  if (const FrameNotCarried* not_carried = std::get_if<FrameNotCarried>(&found)) {
    throw_frame_not_carried(where, not_carried->fact.c_str());
  }
  if (const std::string* why = std::get_if<std::string>(&found)) {
    refuse("a " + std::to_string(width) + "x" + std::to_string(height) + " " + frames.format +
           " frame is not one the model's image preproc stage takes: " + *why +
           " (Dataflow backend)");
  }
  frames.passed.store(word, std::memory_order_relaxed);
  return word;
}

bool dataflow_planar_frame_wraps(const DataflowModelRoute& route, const Tensor& tensor) {
  const DataflowFrames& frames = *route.frames;
  const Plane* const luma = tensor.try_plane(PlaneRole::Y);
  const Plane* const chroma = tensor.try_plane(PlaneRole::UV);
  const auto tight = [](const Plane& plane, std::int64_t rows, std::int64_t row_bytes) {
    return plane.shape.size() >= 2U && plane.shape[0] == rows && plane.shape[1] == row_bytes &&
           (plane.strides_bytes.empty() || plane.strides_bytes[0] == row_bytes) &&
           (plane.strides_bytes.size() < 2U || plane.strides_bytes[1] == 1);
  };
  const std::int64_t width = frames.max_width, height = frames.max_height;
  return luma && chroma && tensor.byte_offset == 0 && luma->byte_offset == 0 &&
         tight(*luma, height, width) && tight(*chroma, height / 2, width) &&
         chroma->byte_offset == width * height;
}

bool copy_dataflow_planar_frame(const DataflowModelRoute& route, const Tensor& tensor,
                                std::span<std::byte> slot) {
  const DataflowFrames& frames = *route.frames;
  std::optional<Nv12Mapped> mapped;
  try {
    mapped = tensor.map_nv12_read();
  } catch (const std::exception&) {
    return false;
  }
  if (!mapped) {
    return false;
  }
  const Nv12View& view = mapped->view;
  const auto width = static_cast<std::size_t>(view.width);
  const auto height = static_cast<std::size_t>(view.height);
  // The port's two offsets (the frame at the bound as it is); this frame's rows tight at each.
  const std::size_t chroma_at = static_cast<std::size_t>(frames.max_width * frames.max_height);
  if (!view.y || !view.uv || width == 0U || height == 0U || width * height > chroma_at ||
      slot.size() < chroma_at || slot.size() - chroma_at < width * (height / 2U)) {
    return false;
  }
  auto* const to = reinterpret_cast<std::uint8_t*>(slot.data());
  for (std::size_t row = 0; row < height; ++row) {
    std::memcpy(to + row * width, view.y + row * static_cast<std::size_t>(view.y_stride), width);
  }
  for (std::size_t row = 0; row < height / 2U; ++row) {
    std::memcpy(to + chroma_at + row * width,
                view.uv + row * static_cast<std::size_t>(view.uv_stride), width);
  }
  return true;
}

#endif // SIMANEAT_WITH_DATAFLOW_SOC_KERNELS

std::shared_ptr<void> load_dataflow_model_route(const DataflowModelRoute& route,
                                                const char* where) {
  auto loaded = std::make_shared<LoadedRoute>();
  loaded->models.resize(route.stages.size());
#if defined(SIMANEAT_WITH_DATAFLOW_SOC_KERNELS)
  loaded->kernels.resize(route.stages.size());
  std::shared_ptr<df::soc::Kernel> decode; // BoxDecode's: the kernel stage in front of the wire
#endif
  for (std::size_t i = 0; i < route.stages.size(); ++i) {
    const DataflowModelStage& stage = route.stages[i];
    if (std::holds_alternative<DataflowModelStage::Mla>(stage.kind)) {
      loaded->models[i] = load_model(stage, where);
    } else if (std::holds_alternative<DataflowModelStage::Kernel>(stage.kind)) {
#if defined(SIMANEAT_WITH_DATAFLOW_SOC_KERNELS)
      loaded->kernels[i] = make_kernel(stage, where, pipeline_internal::DataflowErrorSite::Build);
#else
      pipeline_internal::error_util::throw_session_error(
          error_codes::kPipelineShape, std::string(where) +
                                           ": this build of libsima_neat has no kernel operator "
                                           "(dataflow::soc)");
#endif
    } else {
#if defined(SIMANEAT_WITH_DATAFLOW_SOC_KERNELS)
      decode = loaded->kernels[i - 1U]; // Core's serializer itself loads nothing
#endif
    }
  }
#if defined(SIMANEAT_WITH_DATAFLOW_SOC_KERNELS)
  if (route.frames) {
    seed_dataflow_frames(*route.frames, loaded->kernels.front(), std::move(decode));
  }
#endif
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
    (void)last; // read with the kernel operator only
    if (std::holds_alternative<DataflowModelStage::Mla>(stage.kind)) {
      const std::shared_ptr<df::soc::MlaModel>& model = loaded->models[i];
      df::soc::MlaOp::Options options;
      options.depth = std::get<DataflowModelStage::Mla>(stage.kind).depth;
      options.inputs = {carried};
      // The output port: at the route's end, or in front of a kernel whose tensors are views of
      // this buffer (BoxDecode's heads; one tensor shorter than the buffer, which ends in the
      // MLA's padding), the MLA's output buffer as it is (bytes; a pulled tensor is a view of it,
      // DataflowPulledTensor). In front of a kernel whose one input is the whole buffer, that
      // kernel's tensor (the MLA's bytes, typed as the kernel reads them). Either way in the
      // MLA's memory (Dms0) for an A65 kernel, which reads it through the CPU's address, and in
      // Ev74 memory for a kernel on the EV74, which reaches linux,cma only and reads it by
      // descriptor.
      carried = model->outputs().front();
#if defined(SIMANEAT_WITH_DATAFLOW_SOC_KERNELS)
      if (!last) {
        const auto& next = std::get<DataflowModelStage::Kernel>(route.stages[i + 1U].kind);
        if (next.packed_inputs.empty()) {
          carried = loaded->kernels[i + 1U]->inputs().front();
        }
        carried.memory = next.ev74 ? df::MemoryKind::Ev74 : df::MemoryKind::Dms0;
      }
#endif
      options.outputs = {carried};
      operators.push_back(net.add<df::soc::MlaOp>("mla", model, std::move(options)).index);
      continue;
    }
#if defined(SIMANEAT_WITH_DATAFLOW_SOC_KERNELS)
    if (const auto* wire = std::get_if<DataflowModelStage::Wire>(&stage.kind)) {
      const df::PortSpec produced = byte_port(stage.output_bytes, df::MemoryKind::Dms0);
      operators.push_back(net.add<BoxDecodeWireOp>(stage.key, carried, produced, *wire).index);
      carried = produced;
      continue;
    }
    const DataflowModelStage::Kernel& request = std::get<DataflowModelStage::Kernel>(stage.kind);
    const std::shared_ptr<df::soc::Kernel>& kernel = loaded->kernels[i];
    // Where the ports' bytes live. A kernel on the EV74 has its ports in Ev74 memory by itself
    // (Options{}): it reads and writes by descriptor, and the Plan settles the Feed's or the
    // MLA's port in front of it and the MLA's or the Fetch's behind it to that memory. An A65
    // kernel reads its input through the CPU's address, in whatever memory the previous operator
    // (or the Feed) publishes, and writes its output into device memory (Dms0): the MLA takes it
    // by descriptor, and at the route's end Core's pull hands a DMA-BUF out, as GStreamer's
    // appsink does; the write epoch of that output is the runtime's (a job-written port). BoxDecode
    // writes its outputs into one host buffer, which the wire operator behind it reads on the
    // Worker. A kernel whose one output is the front of the MLA's longer input buffer writes into
    // a port of that buffer's bytes, in the memory it would write its own tensor in.
    df::soc::KernelOp::Options options;
    df::PortSpec produced = kernel->outputs().front();
    if (!request.packed_outputs.empty()) {
      produced = byte_port(stage.output_bytes, produced.memory);
    }
    if (!request.ev74) {
      const bool host_out =
          !last && std::holds_alternative<DataflowModelStage::Wire>(route.stages[i + 1U].kind);
      options.inputs = {carried.memory};
      options.outputs = {host_out ? df::MemoryKind::Host : df::MemoryKind::Dms0};
      produced.memory = options.outputs.front();
    }
    // A packed side: one port of the side's bytes, each kernel tensor a view of it at its offset
    // (the planes of an NV12 frame in the Feed's buffer; the heads in the MLA's buffer, read
    // through the CPU's address behind the port's invalidate; BoxDecode's outputs).
    if (!request.packed_inputs.empty()) {
      options.pack_inputs.bytes = stage.input_bytes;
      for (const DataflowModelStage::View& view : request.packed_inputs) {
        options.pack_inputs.offsets.push_back(view.offset);
      }
    }
    if (!request.packed_outputs.empty()) {
      options.pack_outputs.bytes = stage.output_bytes;
      for (const DataflowModelStage::View& view : request.packed_outputs) {
        options.pack_outputs.offsets.push_back(view.offset);
      }
    }
    // Frames of a changing size: the operator keeps a kernel per frame geometry and runs each
    // message with its own, the frames' table's (DataflowFrames).
    const std::shared_ptr<DataflowFrames>& frames = route.frames;
    using Variants = DataflowModelStage::Kernel::Variants;
    if (request.variants == Variants::ImageByExtent) {
      // RGB or BGR frames on the Feed's bounded tensor: the message's extent is the frame's size.
      options.variants.input = std::get<df::TensorSpec>(carried.payload);
      options.variants.make = [frames](std::uint64_t, const df::Extent& extent) {
        return frame_kernel(
            *frames, DataflowFrames::frame_word(extent.dims[kWidthAxis], extent.dims[kHeightAxis]),
            false);
      };
    } else if (request.variants != Variants::None) {
      const bool decode = request.variants == Variants::DecodeByWord;
      options.variants.keyed = true; // the key is the stamp's user word (dataflow_frame_word)
      options.variants.make = [frames, decode](std::uint64_t word, const df::Extent&) {
        return frame_kernel(*frames, word, decode);
      };
    }
    carried = produced;
    operators.push_back(net.add<df::soc::KernelOp>(stage.key, kernel, std::move(options)).index);
#endif
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
