#include "pipeline/internal/sima/CanonicalKernelLowering.h"
#include <array>
#include <cmath>
#include <iostream>
#include <simaai/kernels/detail/executable.h>
#include <simaai/kernels/ops/cast.h>
#include <simaai/kernels/ops/quant.h>
#include <simaai/kernels/ops/dequant.h>
#include <simaai/kernels/ops/casttess.h>
#include <simaai/kernels/ops/quanttess.h>
#include <simaai/kernels/ops/detesscast.h>
#include <simaai/kernels/ops/detessdequant.h>
#include <simaai/kernels/ops/tess.h>
#include <simaai/kernels/ops/detess.h>

#include <stdexcept>
namespace rt = simaai::runtime;
namespace k = simaai::kernels;
namespace core = simaai::neat::pipeline_internal::sima;
namespace lower = core::kernel_lowering;
namespace {
void check(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}
template <class T> T take(rt::Result<T> r) {
  if (auto* e = std::get_if<rt::Error>(&r))
    throw std::runtime_error(e->field + ": " + e->message);
  return std::get<T>(std::move(r));
}
struct Transport : rt::Ev74Transport {
  mutable unsigned queries = 0;
  bool supports(std::string_view, std::uint32_t, std::size_t) const noexcept override {
    ++queries;
    return true;
  }
  rt::Status submit(const rt::Ev74Packet&, Retired) noexcept override {
    return rt::Error{rt::ErrorCode::ExecutionFailed, "test",
                     "no device execution in compatibility test"};
  }
};
sima_ev_tensor_desc descriptor(std::uint32_t dtype, bool tiled) {
  sima_ev_tensor_desc d{};
  d.dtype = dtype;
  d.shape.rank = 3;
  d.shape.sizes[0] = 2;
  d.shape.sizes[1] = 3;
  d.shape.sizes[2] = 4;
  d.shape.axis_semantics[0] = SIMA_EV_AXIS_H;
  d.shape.axis_semantics[1] = SIMA_EV_AXIS_W;
  d.shape.axis_semantics[2] = SIMA_EV_AXIS_C;
  d.layout_kind = tiled ? SIMA_EV_LAYOUT_TILED : SIMA_EV_LAYOUT_STRIDED;
  if (tiled) {
    d.layout.tiled.flags = SIMA_EV_TILED_FLAG_COMPACT_CHANNELS;
    d.layout.tiled.tile_align_bytes = 16;
    d.layout.tiled.tile_sizes[0] = 2;
    d.layout.tiled.tile_sizes[1] = 3;
    d.layout.tiled.tile_sizes[2] = 4;
  } else {
    const int bytes = dtype == SIMA_EV_DTYPE_INT8 ? 1 : dtype == SIMA_EV_DTYPE_FP32 ? 4 : 2;
    d.layout.strided.strides_bytes[0] = 12 * bytes;
    d.layout.strided.strides_bytes[1] = 4 * bytes;
    d.layout.strided.strides_bytes[2] = bytes;
  }
  return d;
}
struct Recipe {
  const char* legacy;
  const char* canonical;
  int id;
  std::uint32_t in, out;
  bool tiled_in, tiled_out;
  const char* input_meaning;
  const char* output_meaning;
};
constexpr std::array recipes{Recipe{"cast", "cast", 221, SIMA_EV_DTYPE_FP32, SIMA_EV_DTYPE_BF16,
                                    false, false, "FP32", "BF16"},
                             Recipe{"quantize", "quant", 222, SIMA_EV_DTYPE_FP32,
                                    SIMA_EV_DTYPE_INT8, false, false, "FP32", "INT8"},
                             Recipe{"dequantize", "dequant", 223, SIMA_EV_DTYPE_INT8,
                                    SIMA_EV_DTYPE_FP32, false, false, "INT8", "FP32"},
                             Recipe{"casttess", "casttess", 224, SIMA_EV_DTYPE_FP32,
                                    SIMA_EV_DTYPE_BF16, false, true, "FP32", "BF16"},
                             Recipe{"detesscast", "detesscast", 225, SIMA_EV_DTYPE_BF16,
                                    SIMA_EV_DTYPE_FP32, true, false, "BF16", "FP32"},
                             Recipe{"quantizetessellate", "quanttess", 226, SIMA_EV_DTYPE_FP32,
                                    SIMA_EV_DTYPE_INT8, false, true, "FP32", "INT8"},
                             Recipe{"detessdequant", "detessdequant", 227, SIMA_EV_DTYPE_INT8,
                                    SIMA_EV_DTYPE_FP32, true, false, "INT8", "FP32"},
                             Recipe{"tessellate", "tess", 2, SIMA_EV_DTYPE_BF16, SIMA_EV_DTYPE_BF16,
                                    false, true, "BF16", "BF16"},
                             Recipe{"detessellate", "detess", 3, SIMA_EV_DTYPE_BF16,
                                    SIMA_EV_DTYPE_BF16, true, false, "BF16", "BF16"}};
core::StageStaticSpec config(const Recipe& r) {
  core::StageStaticSpec c;
  c.processcvu.graph_name = r.legacy;
  c.processcvu.graph_id = r.id;
  c.processcvu.resolved_exec_backend = "EV74";
  c.processcvu.batch_size = 1;
  c.processcvu.num_in_tensor = 1;
  c.processcvu.round_off = 3;
  c.processcvu.has_q_scale = c.processcvu.has_q_zp = true;
  c.processcvu.q_scale = .25F;
  c.processcvu.q_zp = -3;
  c.processcvu.input_tensors = {descriptor(r.in, r.tiled_in)};
  c.processcvu.output_tensors = {descriptor(r.out, r.tiled_out)};
  c.processcvu.input_dtype = r.input_meaning;
  c.processcvu.output_dtype = r.output_meaning;
  if (r.id == 225)
    c.logical_inputs = {{.dtype = r.input_meaning,
                         .materialization_kind = core::TensorMaterializationKind::Direct}};
  return c;
}
k::TensorSpec spec(std::uint32_t type, bool tiled) {
  const auto dtype = type == SIMA_EV_DTYPE_FP32   ? k::DType::Float32
                     : type == SIMA_EV_DTYPE_BF16 ? k::DType::BFloat16
                                                  : k::DType::Int8;
  k::TensorSpec s{{2, 3, 4}, dtype};
  if (tiled)
    s.layout = k::MlaTiled{{-1, 0, 1, 2}, {1, 2, 3, 4}, k::MlaTiled::Channels::Compact};
  return s;
}
k::Result<k::Executable>
reference(const Recipe& recipe, std::span<const k::TensorSpec> in,
          std::span<const k::TensorSpec> out,
          k::ops::quant::Rounding rounding = k::ops::quant::Rounding::TowardNegative) {
  namespace op = k::ops;
  const std::string_view name = recipe.canonical;
  if (name == "cast")
    return op::cast::make_executable({in, out, {}}, k::Target::EV74);
  if (name == "quant")
    return op::quant::make_executable({in, out, op::quant::Params{.25F, -3, rounding}},
                                      k::Target::EV74);
  if (name == "dequant")
    return op::dequant::make_executable({in, out, op::dequant::Params{4.F, -3}}, k::Target::EV74);
  if (name == "casttess")
    return op::casttess::make_executable({in, out, {}}, k::Target::EV74);
  if (name == "detesscast")
    return op::detesscast::make_executable({in, out, {}}, k::Target::EV74);
  if (name == "quanttess")
    return op::quanttess::make_executable({in, out, op::quanttess::Params{.25F, -3, rounding}},
                                          k::Target::EV74);
  if (name == "detessdequant")
    return op::detessdequant::make_executable({in, out, op::detessdequant::Params{4.F, -3}},
                                              k::Target::EV74);
  if (name == "tess")
    return op::tess::make_executable({in, out, {}}, k::Target::EV74);
  return op::detess::make_executable({in, out, {}}, k::Target::EV74);
}
struct Prepared {
  rt::KernelSession kernels;
  std::vector<rt::gstreamer::TensorStageMember> members;
};
rt::Result<Prepared> prepare(const core::StageStaticSpec& stage, rt::KernelSessionOptions options) {
  auto checked = lower::check_transform_stage(stage);
  if (const auto* error = std::get_if<rt::Error>(&checked))
    return *error;
  auto recipe = std::get<std::shared_ptr<core::KernelStageRecipe>>(std::move(checked));
  std::vector<rt::KernelExecutable> executables;
  for (const auto& member : recipe->members) {
    if (std::none_of(executables.begin(), executables.end(), [&](const auto& executable) {
          return &executable.state() == &member.executable.state();
        }))
      executables.push_back(member.executable);
  }
  auto prepared = rt::prepare_kernels(executables, std::move(options));
  if (const auto* error = std::get_if<rt::Error>(&prepared))
    return *error;
  return Prepared{std::get<rt::KernelSession>(std::move(prepared)), std::move(recipe->members)};
}
void contract() {
  auto transport = std::make_shared<Transport>();
  rt::KernelSessionOptions options;
  options.runtime.ev74 = transport;
  for (const auto& r : recipes) {
    auto c = config(r);
    auto translated = take(prepare(c, options));
    check(translated.kernels.size() == 1 && translated.members.size() == 1,
          "transform arity changed");
    const std::array in{spec(r.in, r.tiled_in)}, out{spec(r.out, r.tiled_out)};
    auto reference = ::reference(r, in, out);
    check(std::holds_alternative<k::Executable>(reference), "reference contract failed");
    const auto& expected = std::get<k::Executable>(reference).state();
    const auto& actual = translated.kernels.operation(0)->executable().state();
    check(actual.packet.entry == expected.packet.entry &&
              actual.packet.payload == expected.packet.payload,
          "legacy lowering changed scale/rounding/dtype/layout packet");
  }
  // Legacy scalar and per-member rounding must select the exact same packet
  // as the typed kernel API, for Quant and its fused tessellation variant.
  using Round = k::ops::quant::Rounding;
  constexpr std::array modes{Round::TowardZero, Round::NearestEven, Round::TowardPositive,
                             Round::TowardNegative};
  for (const auto index : {1U, 5U})
    for (std::size_t mode = 0; mode < modes.size(); ++mode) {
      const auto& recipe = recipes[index];
      for (const bool per_member : {false, true}) {
        auto c = config(recipe);
        c.processcvu.round_off = static_cast<int>(mode);
        auto actual = take(lower::check_transform_stage(c));
        const std::array in{spec(recipe.in, recipe.tiled_in)},
            out{spec(recipe.out, recipe.tiled_out)};
        if (per_member) {
          // The retired adapter's per-member override had no Core model field.
          // Preserve its exact numerical oracle through the canonical named API.
          constexpr std::array<const char*, 4> names{"toward_zero", "nearest_even",
                                                     "toward_positive", "toward_negative"};
          const std::array<rt::Parameter, 3> parameters{
              {{"multiplier", .25F}, {"zero_point", -3}, {"rounding", names[mode]}}};
          actual->members.front().executable = take(::simaai::scheduler::check_kernel(
              {recipe.canonical, rt::Hardware::EV74, in, out, parameters}));
        }
        auto expected = std::get<k::Executable>(::reference(recipe, in, out, modes[mode]));
        check(actual->members.front().executable.state().packet.payload ==
                  expected.state().packet.payload,
              "rounding scalar/per-member precedence changed the lowered packet");
      }
    }
  auto reject = [&](const core::StageStaticSpec& c) {
    const auto before = transport->queries;
    auto result = prepare(c, options);
    check(std::holds_alternative<rt::Error>(result) && before == transport->queries,
          "invalid legacy metadata reached transport or was accepted");
  };
  auto canonical_quanttess = config(recipes[5]);
  const auto historical_quanttess = take(prepare(canonical_quanttess, options));
  canonical_quanttess.processcvu.graph_name = "quanttess";
  const auto current_quanttess = take(prepare(canonical_quanttess, options));
  check(current_quanttess.kernels.operation(0)->executable().state().packet.payload ==
            historical_quanttess.kernels.operation(0)->executable().state().packet.payload,
        "Core canonical QuantTess name changed the native packet");
  auto c = config(recipes[0]);
  c.processcvu.batch_size = 2;
  c.processcvu.input_tensors[0].storage.nbytes = 96;
  c.processcvu.output_tensors[0].storage.nbytes = 48;
  auto batched = take(prepare(c, options));
  check(batched.members.size() == 2 && batched.members[1].inputs[0].byte_offset == 96 &&
            batched.members[1].outputs[0].byte_offset == 48,
        "batch offsets changed");
  c.processcvu.input_tensors[0].storage.nbytes = 192;
  reject(c); // Pre-batched span was never per-member metadata.
  c = config(recipes[0]);
  c.processcvu.output_tensors[0].shape.axis_semantics[0] = SIMA_EV_AXIS_W;
  reject(c);
  c = config(recipes[6]);
  c.processcvu.input_tensors[0].layout.tiled.flags = SIMA_EV_TILED_FLAG_NONE;
  reject(c);
  // CBlock16 is not generally pixel-major. Only the proven INT8 degenerate
  // cases may reuse the existing native packet without a conversion or copy.
  for (const auto dimensions : {std::array{1, 1, 1000}, std::array{2, 3, 4}}) {
    c = config(recipes[6]);
    auto& input = c.processcvu.input_tensors[0];
    auto& output = c.processcvu.output_tensors[0];
    input.layout.tiled.flags = SIMA_EV_TILED_FLAG_CBLOCK16;
    for (unsigned axis = 0; axis < 3; ++axis) {
      input.shape.sizes[axis] = output.shape.sizes[axis] = dimensions[axis];
      input.layout.tiled.tile_sizes[axis] = dimensions[axis];
    }
    output.layout.strided.strides_bytes[2] = 4;
    output.layout.strided.strides_bytes[1] = 4 * dimensions[2];
    output.layout.strided.strides_bytes[0] = 4 * dimensions[1] * dimensions[2];
    auto equivalent = take(prepare(c, options));
    input.layout.tiled.flags = SIMA_EV_TILED_FLAG_PADDED_HWC_C16;
    auto canonical = take(prepare(c, options));
    check(equivalent.kernels.operation(0)->executable().state().packet.payload ==
              canonical.kernels.operation(0)->executable().state().packet.payload,
          "equivalent INT8 CBlock16 translation changed native packet");
    const auto pixels = dimensions[0] * dimensions[1];
    const auto stored_channels = (dimensions[2] + 15) / 16 * 16;
    for (int pixel = 0; pixel < pixels; ++pixel)
      for (int channel = 0; channel < dimensions[2]; ++channel)
        check((channel / 16 * pixels + pixel) * 16 + channel % 16 ==
                  pixel * stored_channels + channel,
              "admitted CBlock16 storage differs from pixel-major storage");
  }
  // Exact compiler contract: 2x3x33 logical INT8, 48 stored lanes per pixel.
  // Unlike the proven degeneracies above, this requires the native block reader.
  for (const auto recipe : {recipes[6], recipes[4]}) {
    c = config(recipe);
    auto& input = c.processcvu.input_tensors[0];
    auto& output = c.processcvu.output_tensors[0];
    input.layout.tiled.flags = SIMA_EV_TILED_FLAG_CBLOCK16;
    input.shape.sizes[2] = output.shape.sizes[2] = 33;
    input.layout.tiled.tile_sizes[2] = 33;
    input.storage.nbytes = 2 * 3 * 48 * (recipe.in == SIMA_EV_DTYPE_INT8 ? 1 : 2);
    output.layout.strided.strides_bytes[1] = 33 * 4;
    output.layout.strided.strides_bytes[0] = 3 * 33 * 4;
    output.storage.nbytes = 2 * 3 * 33 * 4;
    // Preserve the compiler's explicit unit N axis, not an inferred reshape.
    for (unsigned axis = 3; axis > 0; --axis) {
      input.shape.sizes[axis] = input.shape.sizes[axis - 1];
      output.shape.sizes[axis] = output.shape.sizes[axis - 1];
      input.shape.axis_semantics[axis] = input.shape.axis_semantics[axis - 1];
      output.shape.axis_semantics[axis] = output.shape.axis_semantics[axis - 1];
      input.layout.tiled.tile_sizes[axis] = input.layout.tiled.tile_sizes[axis - 1];
      output.layout.strided.strides_bytes[axis] = output.layout.strided.strides_bytes[axis - 1];
    }
    input.shape.rank = output.shape.rank = 4;
    input.shape.sizes[0] = output.shape.sizes[0] = 1;
    input.shape.axis_semantics[0] = output.shape.axis_semantics[0] = SIMA_EV_AXIS_N;
    input.layout.tiled.tile_sizes[0] = 1;
    output.layout.strided.strides_bytes[0] = output.storage.nbytes;
    const auto checked = take(lower::check_transform_stage(c));
    const auto& packet = checked->members.at(0).executable.state().packet.payload;
    check((std::to_integer<unsigned>(packet[16]) & 1U) != 0,
          "compiler blocked contract did not select the fused native reader");
    if (recipe.in == SIMA_EV_DTYPE_BF16) {
      // Older compiled DetessCast records author the same storage through bit6,
      // not the explicit CBlock marker. No lane-repack permission is implied.
      c.processcvu.opt_flags |= 1U << 6;
      input.layout.tiled.flags = SIMA_EV_TILED_FLAG_NONE;
      const auto authored = take(lower::check_transform_stage(c));
      check(authored->members.at(0).executable.state().packet.payload == packet,
            "Core bit6 blocked BF16 contract changed the native layout");
    }
  }
  c = config(recipes[4]);
  c.logical_inputs = {
      {.materialization_kind = core::TensorMaterializationKind::Bf16LaneSplitRepack}};
  reject(c);
  c = config(recipes[4]);
  c.logical_inputs.clear();
  reject(c);
  c = config(recipes[0]);
  c.processcvu.output_dtype = "FP16";
  reject(c);
  c = config(recipes[0]);
  c.processcvu.output_dtype.clear();
  reject(c); // INT16/BF16 ambiguity.
  c = config(recipes[1]);
  c.processcvu.q_scale_list = {1.F, 2.F};
  reject(c);
  c = config(recipes[2]);
  c.processcvu.q_scale_list = {1.F};
  c.processcvu.dq_scale_list = {2.F};
  reject(c);
  c = config(recipes[2]);
  c.processcvu.q_scale_list = {.25};
  c.processcvu.dq_scale_list = {std::nextafter(.25, 1.0)};
  take(lower::check_transform_stage(c)); // Both authored scales round to the same legacy float.

  c = config(recipes[5]);
  c.processcvu.graph_id = 202;
  take(lower::check_transform_stage(c)); // Legacy numeric storage is not dispatch authority.
  c.processcvu.graph_name = "unknown_transform";
  reject(c);
  c = config(recipes[1]);
  c.processcvu.round_off = 9;
  reject(c);
  c = config(recipes[1]);
  c.processcvu.has_q_scale = false;
  reject(c);
  c = config(recipes[1]);
  c.processcvu.batch_size = 3;
  c.processcvu.num_in_tensor = 2;
  c.processcvu.input_tensors.push_back(c.processcvu.input_tensors[0]);
  c.processcvu.output_tensors.push_back(c.processcvu.output_tensors[0]);
  c.processcvu.q_scale_list = {.25F, .5F};
  c.processcvu.q_zp_list = {0, -3};
  c.processcvu.round_off = 0;
  auto group = take(prepare(c, options));
  check(group.kernels.size() == 2 && group.members.size() == 6 &&
            &group.members[3].executable.state() ==
                &group.kernels.operation(1)->executable().state(),
        "multi-member transform grouping changed");
}
} // namespace
int main() {
  try {
    contract();
    std::cout << "Core canonical transform lowering: PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
