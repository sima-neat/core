#include "pipeline/internal/sima/CanonicalKernelLowering.h"
#include "pipeline/internal/sima/CanonicalStageIdentity.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace simaai::neat::pipeline_internal::sima::kernel_lowering {
using namespace ::simaai::scheduler::client;
namespace kernels = ::simaai::scheduler::client;
using ::simaai::scheduler::client::check_kernel;
using ::simaai::scheduler::client::inspect_kernel;
namespace {
Error invalid(const char* field, const char* message) {
  return {ErrorCode::InvalidRequest, field, message};
}
struct Family {
  std::string_view canonical;
  enum Scalar { None, Quant, Dequant, Tess } scalar;
};
constexpr std::array families{Family{"cast", Family::None},
                              Family{"quant", Family::Quant},
                              Family{"dequant", Family::Dequant},
                              Family{"casttess", Family::None},
                              Family{"detesscast", Family::None},
                              Family{"quanttess", Family::Quant},
                              Family{"detessdequant", Family::Dequant},
                              Family{"tess", Family::Tess},
                              Family{"detess", Family::None}};

Result<DType> dtype(std::string_view meaning, std::uint32_t wire) {
  if (const auto scalar = scalar_from_model_descriptor(meaning, wire))
    return *scalar;
  return Error{ErrorCode::InvalidRequest, "dtype",
               "missing, ambiguous or contradictory scalar type: name='" + std::string(meaning) +
                   "', wire=" + std::to_string(wire)};
}

Result<TensorSpec> tensor(const sima_ev_tensor_desc& desc, DType scalar,
                          bool blocked_bf16 = false) {
  if (desc.shape.rank == 0 || desc.shape.rank > std::size(desc.shape.sizes))
    return invalid("shape.rank", "legacy tensor rank must be in [1,6]");
  TensorSpec result{std::vector<std::int64_t>(desc.shape.rank), scalar};
  // Packed wire fields may be unaligned. Copy values rather than binding spans
  // or references into their storage.
  for (std::size_t i = 0; i < desc.shape.rank; ++i)
    result.shape[i] = desc.shape.sizes[i];
  if (desc.layout_kind == SIMA_EV_LAYOUT_STRIDED) {
    kernels::Strided strides;
    std::int64_t dense =
        (result.dtype == DType::Int8 || result.dtype == DType::UInt8)
            ? 1
            : (result.dtype == DType::Float32 || result.dtype == DType::Int32 ? 4 : 2);
    bool contiguous = true;
    for (std::size_t i = desc.shape.rank; i-- > 0;) {
      strides.strides_bytes[i] = desc.layout.strided.strides_bytes[i];
      if (result.shape[i] <= 0 || (result.shape[i] > 1 && strides.strides_bytes[i] != dense))
        contiguous = false;
      if (result.shape[i] > 0 &&
          dense <= std::numeric_limits<std::int64_t>::max() / result.shape[i])
        dense *= result.shape[i];
      else
        contiguous = false;
    }
    result.layout = contiguous ? Layout{kernels::Contiguous{}} : Layout{strides};
    return result;
  }
  if (desc.layout_kind != SIMA_EV_LAYOUT_TILED)
    return invalid("layout", "opaque legacy storage has no canonical tensor mapping");
  if (desc.layout.tiled.tile_align_bytes != kernels::MlaTiled::tile_alignment_bytes)
    return invalid("tile_align_bytes", "native MLA tiles require 16-byte alignment");
  constexpr auto compact = SIMA_EV_TILED_FLAG_COMPACT_CHANNELS;
  constexpr auto padded = SIMA_EV_TILED_FLAG_PADDED_HWC_C16;
  constexpr auto blocked = SIMA_EV_TILED_FLAG_CBLOCK16;
  const auto authored_flags = desc.layout.tiled.flags;
  const auto flags = blocked_bf16 ? static_cast<std::uint32_t>(blocked) : authored_flags;
  if (blocked_bf16 && (authored_flags & ~(compact | padded | blocked)))
    return invalid("layout.flags", "unknown authored blocked BF16 encoding");
  if ((flags & ~(compact | padded | blocked)) != 0 ||
      (flags != 0 && flags != compact && flags != padded && flags != blocked))
    return invalid("layout.flags", "unsupported or conflicting legacy tiled encoding");
  // NONE is operation-dependent in the legacy ABI (DetessDequant uses
  // CBlock16), not an alias for pixel-major PadTo16. Fail closed until a
  // family-specific conversion/materialization is explicitly implemented.
  if (flags == SIMA_EV_TILED_FLAG_NONE)
    return invalid("layout.flags", "untagged legacy tiled encoding requires explicit migration");
  kernels::MlaTiled tiled;
  tiled.channels =
      flags & compact ? kernels::MlaTiled::Channels::Compact : kernels::MlaTiled::Channels::PadTo16;
  for (std::size_t i = 0; i < desc.shape.rank; ++i) {
    const auto semantic = desc.shape.axis_semantics[i];
    int component = -1;
    switch (semantic) {
    case SIMA_EV_AXIS_D:
      component = 0;
      break;
    case SIMA_EV_AXIS_H:
      component = 1;
      break;
    case SIMA_EV_AXIS_W:
      component = 2;
      break;
    case SIMA_EV_AXIS_C:
      component = 3;
      break;
    case SIMA_EV_AXIS_N:
    case SIMA_EV_AXIS_UNKNOWN:
      if (desc.layout.tiled.tile_sizes[i] != 1)
        return invalid("layout.tile", "outer-dimension tiling requires explicit migration");
      break;
    default:
      return invalid("shape.axis", "unknown legacy axis semantic");
    }
    if (component >= 0) {
      if (tiled.axes[component] != -1)
        return invalid("shape.axis", "duplicate tiled axis semantic");
      tiled.axes[component] = static_cast<std::int8_t>(i);
      tiled.tile_shape[component] = desc.layout.tiled.tile_sizes[i];
    }
  }
  if (tiled.axes[3] < 0)
    return invalid("shape.axis", "tiled channel axis must be explicit");
  if (flags == blocked) {
    // Compiler CBlock16 is a distinct physical reader, not a padded-HWC alias.
    // Preserve the declared shape; the native blocked contract uses trailing HWC.
    const auto rank = static_cast<int>(desc.shape.rank);
    if (rank < 3 || tiled.axes[1] != rank - 3 || tiled.axes[2] != rank - 2 ||
        tiled.axes[3] != rank - 1 ||
        (tiled.axes[0] >= 0 && (result.shape[tiled.axes[0]] != 1 || tiled.tile_shape[0] != 1)))
      return invalid("layout.axes",
                     "blocked transport requires trailing HWC and untiled outer axes");
    const auto extent = [&](int component) {
      const auto axis = tiled.axes[component];
      return axis < 0 ? std::int64_t(1) : std::min(result.shape[axis], tiled.tile_shape[component]);
    };
    // Retain the established fast route only where byte identity is proved.
    // BF16 block-local byte planes are never covered by this INT8 proof.
    const bool identical = result.dtype == DType::Int8 && extent(1) > 0 && extent(2) > 0 &&
                           extent(3) > 0 && (extent(3) <= 16 || (extent(1) == 1 && extent(2) == 1));
    result.layout = identical
                        ? Layout{tiled}
                        : Layout{kernels::MlaCBlocked{
                              {tiled.tile_shape[1], tiled.tile_shape[2], tiled.tile_shape[3]}}};
    return result;
  }
  result.layout = tiled;
  return result;
}

} // namespace

Result<TensorSpec> tensor_spec_from_model_descriptor(const sima_ev_tensor_desc& descriptor,
                                                     std::string_view meaning) {
  auto scalar = dtype(meaning, descriptor.dtype);
  if (const auto* error = std::get_if<Error>(&scalar))
    return *error;
  return tensor(descriptor, std::get<DType>(scalar));
}

Result<TensorSpec> tensor_spec_from_model_descriptor(const AuthoredTensorDescriptor& authored) {
  return tensor(authored.descriptor(), authored.scalar());
}

Result<std::shared_ptr<KernelStageRecipe>> check_transform_stage(const StageStaticSpec& stage) {
  const auto& cfg = stage.processcvu;
  const auto family = std::find_if(families.begin(), families.end(), [&](const Family& f) {
    return f.canonical == canonical_transform_operation(cfg.graph_name);
  });
  if (family == families.end())
    return invalid("graph", "unknown canonical transform operation");
  const auto count = cfg.input_tensors.size();
  if (!count || count > std::size_t(std::numeric_limits<std::int32_t>::max()) ||
      cfg.output_tensors.size() != count || cfg.batch_size <= 0 ||
      cfg.num_in_tensor != static_cast<std::int32_t>(count))
    return invalid("tensors", "inconsistent transform arity or batch count");
  if (count > std::numeric_limits<std::size_t>::max() / std::size_t(cfg.batch_size))
    return invalid("batch_size", "transform member count overflow");
  const auto target = cfg.resolved_exec_backend == "A65" ? Hardware::A65 : Hardware::EV74;
  if (cfg.resolved_exec_backend != "A65" && cfg.resolved_exec_backend != "EV74" &&
      cfg.resolved_exec_backend != "EVXX")
    return invalid("target", "Core must resolve the backend before preparation");
  const auto exact = [count](const auto& array) { return array.empty() || array.size() == count; };
  if (!exact(cfg.q_scale_list) || !exact(cfg.dq_scale_list) || !exact(cfg.q_zp_list) ||
      !exact(cfg.dq_zp_list))
    return invalid("parameters", "per-member arrays must be empty or match tensor count");
  const auto same_float_values = [](const auto& left, const auto& right) {
    return left.size() == right.size() &&
           std::equal(left.begin(), left.end(), right.begin(), [](double a, double b) {
             return static_cast<float>(a) == static_cast<float>(b);
           });
  };
  if (family->scalar == Family::Dequant &&
      ((!cfg.q_scale_list.empty() && !cfg.dq_scale_list.empty() &&
        !same_float_values(cfg.q_scale_list, cfg.dq_scale_list)) ||
       (!cfg.q_zp_list.empty() && !cfg.dq_zp_list.empty() && cfg.q_zp_list != cfg.dq_zp_list)))
    return invalid("quantization", "contradictory quant/dequant arrays");
  if (family->canonical == "detesscast" &&
      (stage.logical_inputs.size() != count ||
       std::any_of(
           stage.logical_inputs.begin(), stage.logical_inputs.end(), [](const auto& logical) {
             return logical.materialization_kind != TensorMaterializationKind::Direct &&
                    logical.materialization_kind != TensorMaterializationKind::OffsetView &&
                    logical.materialization_kind != TensorMaterializationKind::Bf16LaneSplitRepack;
           })))
    return invalid("materialization", "DetessCast requires exact input materialization policy");
  std::vector<TensorSpec> inputs, outputs;
  std::vector<float> scalar_values(count, 1.0f);
  std::vector<std::int32_t> zero_points(count, 0);
  std::vector<std::string_view> rounding(count, "nearest_even");
  inputs.reserve(count);
  outputs.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    const auto& legacy_input = cfg.input_tensors[i].descriptor();
    const auto& legacy_output = cfg.output_tensors[i].descriptor();
    if (legacy_input.shape.rank != legacy_output.shape.rank ||
        legacy_input.shape.rank > std::size(legacy_input.shape.sizes))
      return invalid("shape", "transform input/output ranks differ");
    for (std::size_t axis = 0; axis < legacy_input.shape.rank; ++axis)
      if (legacy_input.shape.sizes[axis] != legacy_output.shape.sizes[axis] ||
          legacy_input.shape.axis_semantics[axis] != legacy_output.shape.axis_semantics[axis])
        return invalid("shape", "transform input/output logical axes differ");
    auto in = tensor(cfg.input_tensors[i], cfg.input_tensors[i].scalar(),
                     family->canonical == "detesscast" && (cfg.opt_flags & (1U << 6)) != 0);
    auto out = tensor(cfg.output_tensors[i], cfg.output_tensors[i].scalar());
    const auto at_port = [&](Error error, const char* port) {
      error.field = std::string(port) + "[" + std::to_string(i) + "]." + error.field;
      error.message = cfg.graph_name + ": " + error.message;
      return error;
    };
    if (auto* error = std::get_if<Error>(&in))
      return at_port(*error, "inputs");
    if (auto* error = std::get_if<Error>(&out))
      return at_port(*error, "outputs");
    inputs.push_back(std::get<TensorSpec>(std::move(in)));
    outputs.push_back(std::get<TensorSpec>(std::move(out)));
    if (i < stage.logical_inputs.size() && stage.logical_inputs[i].materialization_kind ==
                                               TensorMaterializationKind::Bf16LaneSplitRepack)
      return invalid("materialization", "lane-split repack must be explicitly "
                                        "materialized before native execution");
    if (family->scalar == Family::Quant || family->scalar == Family::Dequant) {
      const bool dequant = family->scalar == Family::Dequant;
      const auto& scales =
          dequant && !cfg.dq_scale_list.empty() ? cfg.dq_scale_list : cfg.q_scale_list;
      const auto& zeros = dequant && !cfg.dq_zp_list.empty() ? cfg.dq_zp_list : cfg.q_zp_list;
      if ((scales.empty() && !cfg.has_q_scale) || (zeros.empty() && !cfg.has_q_zp))
        return invalid("quantization", "each transform member requires scale and zero point");
      const float authored = scales.empty() ? cfg.q_scale : scales[i];
      if (!(authored > 0) || !std::isfinite(authored))
        return invalid("scale", "legacy public scale must be finite and positive");
      // Old prepared fields retain the public multiplier; the old descriptor
      // builder applied this reciprocal exactly once for both dequant families.
      scalar_values[i] = dequant ? 1.0f / authored : authored;
      zero_points[i] = zeros.empty() ? cfg.q_zp : zeros[i];
      if (!dequant) {
        constexpr std::array<std::string_view, 4> modes{"toward_zero", "nearest_even",
                                                        "toward_positive", "toward_negative"};
        const int round = cfg.round_off;
        if (round < 0 || round >= static_cast<int>(modes.size()))
          return invalid("round_off", "unsupported legacy rounding mode");
        rounding[i] = modes[round];
      }
    }
    if (const auto* tile = std::get_if<kernels::MlaTiled>(&outputs.back().layout);
        tile && family->scalar == Family::Tess && outputs.back().dtype == DType::Int8 &&
        tile->channels == kernels::MlaTiled::Channels::PadTo16)
      return invalid("channel_padding", "legacy Tess has no explicit native INT8 padding value");
  }
  std::vector<KernelExecutable> checked;
  checked.reserve(count);
  auto recipe = std::make_shared<KernelStageRecipe>();
  recipe->target = target;
  auto& members = recipe->members;
  members.reserve(count * std::size_t(cfg.batch_size));
  for (std::size_t i = 0; i < count; ++i) {
    // Core emits the named canonical request directly. Numerical validation and
    // operation resolution remain in the operation-owned Kernels provider.
    const std::span<const TensorSpec> in{&inputs[i], 1}, out{&outputs[i], 1};
    std::vector<Parameter> parameters;
    if (family->scalar == Family::Quant) {
      parameters = {{"multiplier", scalar_values[i]},
                    {"zero_point", zero_points[i]},
                    {"rounding", rounding[i]}};
    } else if (family->scalar == Family::Dequant) {
      parameters = {{"scale", scalar_values[i]}, {"zero_point", zero_points[i]}};
    }
    auto compiled = check_kernel({family->canonical, target, in, out, parameters});
    if (const auto* error = std::get_if<Error>(&compiled))
      return *error;
    checked.push_back(std::get<KernelExecutable>(std::move(compiled)));
    auto inspected = inspect_kernel(checked.back());
    if (const auto* error = std::get_if<Error>(&inspected))
      return *error;
    const auto& contract = std::get<KernelInspection>(inspected);
    const auto in_bytes = contract.input_requirements()[0].storage_bytes;
    const auto out_bytes = contract.output_requirements()[0].storage_bytes;
    const auto batches = std::size_t(cfg.batch_size);
    if (in_bytes > std::numeric_limits<std::size_t>::max() / batches ||
        out_bytes > std::numeric_limits<std::size_t>::max() / batches)
      return invalid("storage", "batched transform span overflow");
    // finalizeTensorStorageSpan inferred a missing per-member span, rejected
    // contradictory declarations, then applied batch_size separately. A total
    // batched allocation size is not a valid per-member descriptor span.
    const auto matches = [](std::uint64_t declared, std::size_t span) {
      return declared == 0 || declared == span;
    };
    if (!matches(cfg.input_tensors[i].descriptor().storage.nbytes, in_bytes) ||
        !matches(cfg.output_tensors[i].descriptor().storage.nbytes, out_bytes))
      return invalid("storage", "legacy span differs from canonical physical storage");
    for (std::size_t batch = 0; batch < batches; ++batch) {
      gs::TensorStageMember member;
      member.executable = checked.back();
      member.inputs.push_back({i, batch * in_bytes, gs::TensorStagePortSource::Input});
      member.outputs.push_back({i, batch * out_bytes, gs::TensorStagePortSource::Output});
      members.push_back(std::move(member));
    }
  }
  return recipe;
}
} // namespace simaai::neat::pipeline_internal::sima::kernel_lowering
