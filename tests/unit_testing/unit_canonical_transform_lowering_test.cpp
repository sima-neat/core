#include "pipeline/internal/sima/CanonicalKernelLowering.h"
#include <array>
#include <cmath>
#include <iostream>

#include <stdexcept>
#include <simaai/scheduler/testing/reference.h>
#include <simaai/scheduler/testing/probe.h>
namespace test = simaai::scheduler::testing;
namespace rt = simaai::scheduler::client;
namespace k = simaai::scheduler::client;
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
template <class Edit> void change_geometry(core::AuthoredTensorDescriptor& tensor, Edit edit) {
  auto descriptor = tensor.descriptor();
  edit(descriptor);
  tensor = tensor.with_geometry(descriptor);
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
  c.processcvu.input_tensors = {{descriptor(r.in, r.tiled_in), r.input_meaning}};
  c.processcvu.output_tensors = {{descriptor(r.out, r.tiled_out), r.output_meaning}};
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
k::Result<k::KernelExecutable> reference(const Recipe& recipe, std::span<const k::TensorSpec> in,
                                         std::span<const k::TensorSpec> out,
                                         test::Rounding rounding = test::Rounding::TowardNegative) {
  return test::reference_transform(recipe.canonical, in, out, rounding);
}
struct Sessions {
  std::vector<std::unique_ptr<test::OperationProbe>> values;
  std::size_t size() const {
    return values.size();
  }
  const rt::PreparedSelection* operation(std::size_t i) const {
    return &values.at(i)->operation();
  }
};
struct Prepared {
  Sessions kernels;
  std::vector<rt::TensorStageMember> members;
};
rt::Result<Prepared> prepare(const core::StageStaticSpec& stage, test::Options options) {
  auto checked = lower::check_transform_stage(stage);
  if (const auto* error = std::get_if<rt::Error>(&checked))
    return *error;
  auto recipe = std::get<std::shared_ptr<core::KernelStageRecipe>>(std::move(checked));
  Prepared result;
  for (const auto& member : recipe->members) {
    if (std::any_of(result.kernels.values.begin(), result.kernels.values.end(), [&](const auto& p) {
          return p->operation().executable() == member.executable;
        }))
      continue;
    auto probe = std::make_unique<test::OperationProbe>();
    if (auto error = probe->prepare(member.executable, options))
      return *error;
    result.kernels.values.push_back(std::move(probe));
  }
  result.members = std::move(recipe->members);
  return result;
}
void contract() {
  auto transport = std::make_shared<test::Capture>();
  test::Options options{transport, std::make_shared<test::Domain>(transport)};
  for (const auto& r : recipes) {
    auto c = config(r);
    auto translated = take(prepare(c, options));
    check(translated.kernels.size() == 1 && translated.members.size() == 1,
          "transform arity changed");
    const std::array in{spec(r.in, r.tiled_in)}, out{spec(r.out, r.tiled_out)};
    auto reference = ::reference(r, in, out);
    check(std::holds_alternative<k::KernelExecutable>(reference), "reference contract failed");
    const auto expected = test::packet_snapshot(std::get<k::KernelExecutable>(reference));
    const auto actual = test::packet_snapshot(translated.kernels.operation(0)->executable());
    check(actual.entry == expected.entry && actual.payload == expected.payload,
          "legacy lowering changed scale/rounding/dtype/layout packet");
  }
  // Core's scalar rounding must match the typed API for Quant and QuantTess.
  using Round = test::Rounding;
  constexpr std::array modes{Round::TowardZero, Round::NearestEven, Round::TowardPositive,
                             Round::TowardNegative};
  for (const auto index : {1U, 5U})
    for (std::size_t mode = 0; mode < modes.size(); ++mode) {
      const auto& recipe = recipes[index];
      auto c = config(recipe);
      c.processcvu.round_off = static_cast<int>(mode);
      const auto actual = take(lower::check_transform_stage(c));
      const std::array in{spec(recipe.in, recipe.tiled_in)},
          out{spec(recipe.out, recipe.tiled_out)};
      const auto expected =
          test::packet_snapshot(take(::reference(recipe, in, out, modes[mode]))).payload;
      check(test::packet_snapshot(actual->members.front().executable).payload == expected,
            "Core scalar rounding changed the lowered packet");

      // Keep these distinct provider assertions until Scheduler owns their coverage.
      constexpr std::array<const char*, 4> names{"toward_zero", "nearest_even", "toward_positive",
                                                 "toward_negative"};
      const std::array<rt::Parameter, 3> parameters{
          {{"multiplier", .25F}, {"zero_point", -3}, {"rounding", names[mode]}}};
      const auto named =
          take(rt::check_kernel({recipe.canonical, rt::Target::EV74, in, out, parameters}));
      check(test::packet_snapshot(named).payload == expected,
            "Scheduler named rounding differs from the typed reference");
    }
  auto reject = [&](const core::StageStaticSpec& c) {
    const auto before = transport->admissions;
    auto result = prepare(c, options);
    check(std::holds_alternative<rt::Error>(result) && before == transport->admissions,
          "invalid legacy metadata reached transport or was accepted");
  };
  auto canonical_quanttess = config(recipes[5]);
  const auto historical_quanttess = take(prepare(canonical_quanttess, options));
  canonical_quanttess.processcvu.graph_name = "quanttess";
  const auto current_quanttess = take(prepare(canonical_quanttess, options));
  check(test::packet_snapshot(current_quanttess.kernels.operation(0)->executable()).payload ==
            test::packet_snapshot(historical_quanttess.kernels.operation(0)->executable()).payload,
        "Core canonical QuantTess name changed the native packet");
  auto c = config(recipes[0]);
  c.processcvu.batch_size = 2;
  change_geometry(c.processcvu.input_tensors[0], [](auto& desc) { desc.storage.nbytes = 96; });
  change_geometry(c.processcvu.output_tensors[0], [](auto& desc) { desc.storage.nbytes = 48; });
  auto batched = take(prepare(c, options));
  check(batched.members.size() == 2 && batched.members[1].inputs[0].byte_offset == 96 &&
            batched.members[1].outputs[0].byte_offset == 48,
        "batch offsets changed");
  change_geometry(c.processcvu.input_tensors[0], [](auto& desc) { desc.storage.nbytes = 192; });
  reject(c); // Pre-batched span was never per-member metadata.
  c = config(recipes[0]);
  change_geometry(c.processcvu.output_tensors[0],
                  [](auto& desc) { desc.shape.axis_semantics[0] = SIMA_EV_AXIS_W; });
  reject(c);
  c = config(recipes[6]);
  change_geometry(c.processcvu.input_tensors[0],
                  [](auto& desc) { desc.layout.tiled.flags = SIMA_EV_TILED_FLAG_NONE; });
  reject(c);
  // CBlock16 is not generally pixel-major. Only the proven INT8 degenerate
  // cases may reuse the existing native packet without a conversion or copy.
  for (const auto dimensions : {std::array{1, 1, 1000}, std::array{2, 3, 4}}) {
    c = config(recipes[6]);
    auto input = c.processcvu.input_tensors[0].descriptor();
    auto output = c.processcvu.output_tensors[0].descriptor();
    input.layout.tiled.flags = SIMA_EV_TILED_FLAG_CBLOCK16;
    for (unsigned axis = 0; axis < 3; ++axis) {
      input.shape.sizes[axis] = output.shape.sizes[axis] = dimensions[axis];
      input.layout.tiled.tile_sizes[axis] = dimensions[axis];
    }
    output.layout.strided.strides_bytes[2] = 4;
    output.layout.strided.strides_bytes[1] = 4 * dimensions[2];
    output.layout.strided.strides_bytes[0] = 4 * dimensions[1] * dimensions[2];
    c.processcvu.input_tensors[0] = c.processcvu.input_tensors[0].with_geometry(input);
    c.processcvu.output_tensors[0] = c.processcvu.output_tensors[0].with_geometry(output);
    auto equivalent = take(prepare(c, options));
    input.layout.tiled.flags = SIMA_EV_TILED_FLAG_PADDED_HWC_C16;
    c.processcvu.input_tensors[0] = c.processcvu.input_tensors[0].with_geometry(input);
    c.processcvu.output_tensors[0] = c.processcvu.output_tensors[0].with_geometry(output);
    auto canonical = take(prepare(c, options));
    check(test::packet_snapshot(equivalent.kernels.operation(0)->executable()).payload ==
              test::packet_snapshot(canonical.kernels.operation(0)->executable()).payload,
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
    auto input = c.processcvu.input_tensors[0].descriptor();
    auto output = c.processcvu.output_tensors[0].descriptor();
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
    c.processcvu.input_tensors[0] = c.processcvu.input_tensors[0].with_geometry(input);
    c.processcvu.output_tensors[0] = c.processcvu.output_tensors[0].with_geometry(output);
    const auto checked = take(lower::check_transform_stage(c));
    const auto packet = test::packet_snapshot(checked->members.at(0).executable).payload;
    check((std::to_integer<unsigned>(packet[16]) & 1U) != 0,
          "compiler blocked contract did not select the fused native reader");
    if (recipe.in == SIMA_EV_DTYPE_BF16) {
      // Older compiled DetessCast records author the same storage through bit6,
      // not the explicit CBlock marker. No lane-repack permission is implied.
      c.processcvu.opt_flags |= 1U << 6;
      input.layout.tiled.flags = SIMA_EV_TILED_FLAG_NONE;
      c.processcvu.input_tensors[0] = c.processcvu.input_tensors[0].with_geometry(input);
      c.processcvu.output_tensors[0] = c.processcvu.output_tensors[0].with_geometry(output);
      const auto authored = take(lower::check_transform_stage(c));
      check(test::packet_snapshot(authored->members.at(0).executable).payload == packet,
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
  // Publication and carrier metadata are not numerical authority.
  c.processcvu.output_dtype = "FP16";
  c.processcvu.runtime_output_dtype_list = {"INT8", "INT8"};
  c.logical_inputs = {{.dtype = "INT8"}};
  take(lower::check_transform_stage(c));
  c.processcvu.output_dtype.clear();
  take(lower::check_transform_stage(c));
  for (const auto annotation : {"FP16", ""}) {
    bool rejected = false;
    try {
      core::AuthoredTensorDescriptor invalid(descriptor(SIMA_EV_DTYPE_BF16, false), annotation);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    check(rejected, "ambiguous/contradictory authored scalar was accepted");
  }
  // Equal legacy wire codes retain distinct numerical meanings through copies
  // and reordering. Public slot dtypes are deliberately unrelated.
  std::vector<core::AuthoredTensorDescriptor> authored{
      {descriptor(SIMA_EV_DTYPE_BF16, false), "BF16"},
      {descriptor(SIMA_EV_DTYPE_INT16, false), "INT16"},
      {descriptor(SIMA_EV_DTYPE_FP32, false), "FP32"}};
  auto reordered = authored;
  std::swap(reordered[0], reordered[1]);
  check(take(lower::tensor_spec_from_model_descriptor(reordered[0])).dtype == rt::DType::Int16 &&
            take(lower::tensor_spec_from_model_descriptor(reordered[1])).dtype ==
                rt::DType::BFloat16 &&
            take(lower::tensor_spec_from_model_descriptor(reordered[2])).dtype ==
                rt::DType::Float32 &&
            authored[0].scalar() == rt::DType::BFloat16,
        "copy/reorder lost the scalar attached to an aliased descriptor wire");
  auto moved_geometry = reordered[1].descriptor();
  moved_geometry.storage.addr = 960;
  moved_geometry.storage.nbytes = 48;
  const auto relocated = reordered[1].with_geometry(moved_geometry);
  check(relocated.scalar() == rt::DType::BFloat16 && relocated.descriptor().storage.addr == 960,
        "storage relocation must preserve numerical meaning");
  moved_geometry.dtype = SIMA_EV_DTYPE_FP32;
  bool refused_type_rewrite = false;
  try {
    (void)relocated.with_geometry(moved_geometry);
  } catch (const std::invalid_argument&) {
    refused_type_rewrite = true;
  }
  check(refused_type_rewrite, "geometry-only rewrite changed numerical scalar");
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
            group.members[3].executable == group.kernels.operation(1)->executable(),
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
