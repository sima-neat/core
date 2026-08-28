#include "pipeline/internal/sima/static_contract/DmabufPlanContractProjection.h"
#include "pipeline/internal/TerminalOutputContractQuery.h"
#include "pipeline/internal/sima/stagesemantics/ProcessCvuStageSemantics.h"
#include "pipeline/internal/sima/stagesemantics/ProcessMlaStageSemantics.h"
#include "nodes/sima/Preproc.h"

#include "test_main.h"
#include "test_utils.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace sc = simaai::neat::pipeline_internal::sima::static_contract;
namespace sima = simaai::neat::pipeline_internal::sima;

namespace {

sc::ModelExecutionPlan make_plan() {
  sc::ModelExecutionPlanData data;
  data.contract_version = "2.0.0";
  for (std::size_t index = 0; index < 30U; ++index) {
    const bool input = index < 2U;
    const std::size_t port = input ? index : index - 2U;
    data.values.push_back(sc::ValueSpec{static_cast<sc::ValueId>(index),
                                        std::string(input ? "ifm" : "ofm") + std::to_string(port),
                                        4096U + port * 4096U});
  }
  for (std::size_t index = 0; index < 28U; ++index) {
    data.values.push_back(sc::ValueSpec{static_cast<sc::ValueId>(30U + index),
                                        "post_" + std::to_string(index), 4096U});
  }
  for (std::size_t index = 2U; index < 30U; ++index) {
    data.values[index].logical_dtype = "uint8";
    data.values[index].logical_shape = sc::TensorShape{
        static_cast<std::int64_t>(data.values[index].required_bytes)};
  }
  data.model_inputs = {0U, 1U};
  sc::OpSpec mla;
  mla.id = 0U;
  mla.sequence = 1U;
  mla.name = "MLA_0";
  mla.kind = sc::OpKind::Mla;
  mla.processor = "MLA";
  mla.inputs = {0U, 1U};
  for (std::size_t index = 0; index < 28U; ++index) {
    mla.outputs.push_back(static_cast<sc::ValueId>(index + 2U));
  }
  mla.config = sc::MlaOpConfig{"model.elf", 4};
  data.ops.push_back(std::move(mla));

  sc::OpSpec post;
  post.id = 1U;
  post.sequence = 2U;
  post.name = "Dequant_0";
  post.kind = sc::OpKind::Dequantize;
  post.processor = "EV74";
  post.kernel = "dequantize";
  for (std::size_t index = 0; index < 28U; ++index) {
    post.inputs.push_back(static_cast<sc::ValueId>(index + 2U));
    post.outputs.push_back(static_cast<sc::ValueId>(index + 30U));
  }
  post.config = sc::DequantizeOpConfig{};
  data.ops.push_back(std::move(post));

  for (std::size_t index = 0; index < 2U; ++index) {
    data.backend_ports.push_back(sc::BackendPortSpec{
        0U, sc::BackendPortDirection::Input, index,
        "data.ifm.persistent.input_" + std::to_string(index), static_cast<sc::ValueId>(index),
        data.values[index].required_bytes, sc::kLegacyEvoCmaRegionAlignmentBytes,
        sc::BackendPortAlignmentAuthority::LegacyPolicy, sc::BackendPortAccess::ReadOnly});
  }
  for (std::size_t index = 0; index < 28U; ++index) {
    const auto value_id = static_cast<sc::ValueId>(index + 2U);
    const auto alignment = index == 7U ? 8192U : sc::kLegacyEvoCmaRegionAlignmentBytes;
    const auto alignment_authority = index == 7U ? sc::BackendPortAlignmentAuthority::Contract
                                                 : sc::BackendPortAlignmentAuthority::LegacyPolicy;
    data.backend_ports.push_back(
        sc::BackendPortSpec{0U, sc::BackendPortDirection::Output, index,
                            "data.ofm.persistent.output_" + std::to_string(index), value_id,
                            data.values[value_id].required_bytes, alignment, alignment_authority,
                            sc::BackendPortAccess::WriteOnly});
    const auto public_value_id = static_cast<sc::ValueId>(index + 30U);
    data.model_outputs.push_back(
        sc::ModelOutputSpec{index, data.values[public_value_id].name, public_value_id});
  }
  std::string error;
  auto plan = sc::ModelExecutionPlan::create(std::move(data), &error);
  require(plan.has_value(), "test execution plan must be valid: " + error);
  return std::move(*plan);
}

sc::ModelExecutionPlan make_frontend_plan() {
  sc::ModelExecutionPlanData data;
  data.contract_version = "2.0.0";
  data.values = {
      sc::ValueSpec{0U, "image_0", 100U},
      sc::ValueSpec{1U, "image_1", 200U},
      sc::ValueSpec{2U, "quantize_0", 638976U},
      sc::ValueSpec{3U, "quantize_1", 319488U},
      sc::ValueSpec{4U, "ofm_0", 192U, "uint8", sc::TensorShape{192}},
  };
  data.carriers = {
      {0U, 100U, 64U, sc::ValueRepresentation::Dense},
      {1U, 200U, 64U, sc::ValueRepresentation::Dense},
      {2U, 638976U, 64U, sc::ValueRepresentation::Dense},
      {3U, 319488U, 64U, sc::ValueRepresentation::Dense},
      {4U, 192U, 64U, sc::ValueRepresentation::BackendNative},
  };
  data.values[0].storage_binding =
      sc::StorageBinding{sc::StorageBindingKind::External, 0U,          0U, 100U, {},
                         sc::StorageAccess::ReadOnly,      std::nullopt};
  data.values[1].storage_binding =
      sc::StorageBinding{sc::StorageBindingKind::External, 1U,          0U, 200U, {},
                         sc::StorageAccess::ReadOnly,      std::nullopt};
  data.values[2].storage_binding =
      sc::StorageBinding{sc::StorageBindingKind::Root, 2U,          0U, 638976U, {},
                         sc::StorageAccess::ReadWrite, std::nullopt};
  data.values[3].storage_binding =
      sc::StorageBinding{sc::StorageBindingKind::Root, 3U,          0U, 319488U, {},
                         sc::StorageAccess::ReadWrite, std::nullopt};
  data.values[4].storage_binding = sc::StorageBinding{
      sc::StorageBindingKind::Root, 4U, 0U, 192U, {}, sc::StorageAccess::ReadWrite, std::nullopt};
  data.model_inputs = {0U, 1U};

  sc::OpSpec quantize_0;
  quantize_0.id = 0U;
  quantize_0.sequence = 1U;
  quantize_0.name = "Quantize_0";
  quantize_0.kind = sc::OpKind::Quantize;
  quantize_0.processor = "EV74";
  quantize_0.inputs = {0U};
  quantize_0.outputs = {2U};
  quantize_0.config = sc::QuantizeOpConfig{"int8", 8, "nearest", {}};
  data.ops.push_back(std::move(quantize_0));

  sc::OpSpec quantize_1;
  quantize_1.id = 1U;
  quantize_1.sequence = 2U;
  quantize_1.name = "Quantize_1";
  quantize_1.kind = sc::OpKind::Quantize;
  quantize_1.processor = "EV74";
  quantize_1.inputs = {1U};
  quantize_1.outputs = {3U};
  quantize_1.config = sc::QuantizeOpConfig{"int8", 8, "nearest", {}};
  data.ops.push_back(std::move(quantize_1));

  sc::OpSpec mla;
  mla.id = 2U;
  mla.sequence = 3U;
  mla.name = "MLA_0";
  mla.kind = sc::OpKind::Mla;
  mla.processor = "MLA";
  mla.inputs = {2U, 3U};
  mla.outputs = {4U};
  mla.config = sc::MlaOpConfig{"model.elf", 4};
  data.ops.push_back(std::move(mla));

  data.backend_ports = {
      sc::BackendPortSpec{0U, sc::BackendPortDirection::Input, 0U, "data.ifm.persistent.input_0",
                          2U, 638976U, sc::kLegacyEvoCmaRegionAlignmentBytes,
                          sc::BackendPortAlignmentAuthority::LegacyPolicy,
                          sc::BackendPortAccess::ReadOnly},
      sc::BackendPortSpec{0U, sc::BackendPortDirection::Input, 1U, "data.ifm.persistent.input_1",
                          3U, 319488U, sc::kLegacyEvoCmaRegionAlignmentBytes,
                          sc::BackendPortAlignmentAuthority::LegacyPolicy,
                          sc::BackendPortAccess::ReadOnly},
      sc::BackendPortSpec{0U, sc::BackendPortDirection::Output, 0U, "data.ofm.persistent.output_0",
                          4U, 192U, sc::kLegacyEvoCmaRegionAlignmentBytes,
                          sc::BackendPortAlignmentAuthority::LegacyPolicy,
                          sc::BackendPortAccess::WriteOnly},
  };
  data.model_outputs = {sc::ModelOutputSpec{0U, "ofm_0", 4U}};

  std::string error;
  auto plan = sc::ModelExecutionPlan::create(std::move(data), &error);
  require(plan.has_value(), "frontend execution plan must be valid: " + error);
  return std::move(*plan);
}

sc::ModelExecutionPlan make_nonzero_view_to_terminal_mla_plan() {
  sc::ModelExecutionPlanData data;
  data.contract_version = "2.1.0";
  data.carriers = {
      {0U, 32768U, 4096U, sc::ValueRepresentation::Dense},
      {1U, 8192U, 4096U, sc::ValueRepresentation::Dense},
      {3U, 64U, 4096U, sc::ValueRepresentation::BackendNative},
  };
  data.values = {
      sc::ValueSpec{0U, "input", 32768U, "float32", sc::TensorShape{8192}},
      sc::ValueSpec{1U, "cvu_parent", 8192U, "int8", sc::TensorShape{8192}},
      sc::ValueSpec{2U, "mla_ifm_view", 4096U, "int8", sc::TensorShape{4096},
                    std::nullopt, {}, sc::ValueRepresentation::Dense,
                    sc::ReadExpression{1U, 4096U, {1}}},
      sc::ValueSpec{3U, "mla_ofm", 64U, "uint8", sc::TensorShape{64}},
  };
  data.values[0].storage_binding = sc::StorageBinding{
      sc::StorageBindingKind::External, 0U, 0U, 32768U, {},
      sc::StorageAccess::ReadOnly, std::nullopt};
  data.values[1].storage_binding = sc::StorageBinding{
      sc::StorageBindingKind::Root, 1U, 0U, 8192U, {},
      sc::StorageAccess::ReadWrite, std::nullopt};
  data.values[3].storage_binding = sc::StorageBinding{
      sc::StorageBindingKind::Root, 3U, 0U, 64U, {},
      sc::StorageAccess::ReadWrite, std::nullopt};
  data.model_inputs = {0U};

  sc::OpSpec quantize;
  quantize.id = 0U;
  quantize.sequence = 1U;
  quantize.name = "cvu_parent";
  quantize.kind = sc::OpKind::Quantize;
  quantize.processor = "EV74";
  quantize.inputs = {0U};
  quantize.outputs = {1U};
  quantize.config = sc::QuantizeOpConfig{"int8", 8, "nearest", {}};
  data.ops.push_back(std::move(quantize));

  sc::OpSpec mla;
  mla.id = 1U;
  mla.sequence = 2U;
  mla.name = "MLA_0";
  mla.kind = sc::OpKind::Mla;
  mla.processor = "MLA";
  mla.inputs = {2U};
  mla.outputs = {3U};
  mla.config = sc::MlaOpConfig{"model.elf", 4};
  data.ops.push_back(std::move(mla));
  data.backend_ports = {
      {0U, sc::BackendPortDirection::Input, 0U, "data.ifm.b0", 2U, 4096U, 4096U,
       sc::BackendPortAlignmentAuthority::Contract, sc::BackendPortAccess::ReadOnly},
      {0U, sc::BackendPortDirection::Output, 0U, "data.ofm.b0", 3U, 64U, 4096U,
       sc::BackendPortAlignmentAuthority::Contract, sc::BackendPortAccess::WriteOnly},
  };
  data.model_outputs = {{0U, "mla_ofm", 3U}};

  std::string error;
  auto plan = sc::ModelExecutionPlan::create(std::move(data), &error);
  require(plan.has_value(), "nonzero view-to-MLA plan must be valid: " + error);
  return std::move(*plan);
}

sc::ModelExecutionPlan make_packed_plan(const std::uint64_t packed_ifm_bytes = 960U) {
  sc::ModelExecutionPlanData data;
  data.contract_version = "2.0.0";
  data.values = {
      sc::ValueSpec{0U, "image_0", 100U},
      sc::ValueSpec{1U, "image_1", 200U},
      sc::ValueSpec{2U, "quantize_0", 640U},
      sc::ValueSpec{3U, "quantize_1", 320U},
      sc::ValueSpec{4U, "packed_ifm", packed_ifm_bytes},
      sc::ValueSpec{5U, "packed_ofm", 400U},
      sc::ValueSpec{6U,
                    "unpack_0",
                    200U,
                    "int8",
                    sc::TensorShape{1, 10, 1, 20},
                    std::nullopt,
                    {},
                    sc::ValueRepresentation::BackendNative,
                    sc::ReadExpression{5U, 0U, {200, 20, 20, 1}}},
      sc::ValueSpec{7U,
                    "slice_0",
                    10U,
                    "int8",
                    sc::TensorShape{1, 10, 1, 1},
                    std::nullopt,
                    {},
                    sc::ValueRepresentation::Dense,
                    sc::ReadExpression{5U, 0U, {200, 20, 20, 1}}},
      sc::ValueSpec{8U,
                    "unpack_1",
                    200U,
                    "int8",
                    sc::TensorShape{1, 10, 1, 20},
                    std::nullopt,
                    {},
                    sc::ValueRepresentation::BackendNative,
                    sc::ReadExpression{5U, 200U, {200, 20, 20, 1}}},
      sc::ValueSpec{9U, "dequant_0", 40U},
      sc::ValueSpec{10U, "dequant_1", 800U},
  };
  data.model_inputs = {0U, 1U};
  data.values[0].storage_binding = sc::StorageBinding{
      sc::StorageBindingKind::External, 0U, 0U, 100U, {},
      sc::StorageAccess::ReadOnly, std::nullopt};
  data.values[1].storage_binding = sc::StorageBinding{
      sc::StorageBindingKind::External, 1U, 0U, 200U, {},
      sc::StorageAccess::ReadOnly, std::nullopt};
  // The frozen batch-one Pack is an address relation.  Its two producers write
  // directly into disjoint spans of the single MLA parent carrier.
  data.values[2].storage_binding = sc::StorageBinding{
      sc::StorageBindingKind::Root, 4U, 0U, 640U, {},
      sc::StorageAccess::ReadWrite, std::nullopt};
  data.values[3].storage_binding = sc::StorageBinding{
      sc::StorageBindingKind::Root, 4U, 640U, 320U, {},
      sc::StorageAccess::ReadWrite, std::nullopt};
  data.values[4].storage_binding = sc::StorageBinding{
      sc::StorageBindingKind::Root, 4U, 0U, packed_ifm_bytes, {},
      sc::StorageAccess::ReadWrite, std::nullopt};

  const auto add_op = [&](sc::OpKind kind, std::string name, std::vector<sc::ValueId> inputs,
                          std::vector<sc::ValueId> outputs, sc::OpConfig config) {
    sc::OpSpec op;
    op.id = static_cast<sc::OpId>(data.ops.size());
    op.sequence = data.ops.size() + 1U;
    op.name = std::move(name);
    op.kind = kind;
    op.processor = kind == sc::OpKind::Mla ? "MLA" : "EV74";
    op.inputs = std::move(inputs);
    op.outputs = std::move(outputs);
    op.config = std::move(config);
    data.ops.push_back(std::move(op));
  };
  add_op(sc::OpKind::Quantize, "quantize_0", {0U}, {2U},
         sc::QuantizeOpConfig{"int8", 8, "TONEAREST", {}});
  add_op(sc::OpKind::Quantize, "quantize_1", {1U}, {3U},
         sc::QuantizeOpConfig{"int8", 8, "TONEAREST", {}});
  add_op(sc::OpKind::Pack, "pack", {2U, 3U}, {4U},
         sc::PackOpConfig{{sc::PackComponentPlacement{2U, 0U, 640U},
                           sc::PackComponentPlacement{3U, 640U, 320U}}});
  add_op(sc::OpKind::Mla, "MLA_0", {4U}, {5U}, sc::MlaOpConfig{"model.elf", 4});
  add_op(sc::OpKind::Unpack, "unpack", {5U}, {6U, 8U},
         sc::UnpackOpConfig{{"int8", "int8"}, {{1, 10, 1, 20}, {1, 10, 1, 20}}});
  add_op(sc::OpKind::Slice, "slice", {6U}, {7U},
         sc::SliceOpConfig{{0, 0, 0, 0}, {1, 10, 1, 1}, {1, 10, 1, 20}, {1, 10, 1, 1}});
  add_op(sc::OpKind::Dequantize, "dequantize_0", {7U}, {9U}, sc::DequantizeOpConfig{"int8", {}});
  add_op(sc::OpKind::Dequantize, "dequantize_1", {8U}, {10U}, sc::DequantizeOpConfig{"int8", {}});

  data.backend_ports = {
      sc::BackendPortSpec{0U, sc::BackendPortDirection::Input, 0U, "data.ifm.b0", 4U,
                          packed_ifm_bytes, sc::kLegacyEvoCmaRegionAlignmentBytes,
                          sc::BackendPortAlignmentAuthority::LegacyPolicy,
                          sc::BackendPortAccess::ReadOnly},
      sc::BackendPortSpec{0U, sc::BackendPortDirection::Output, 0U, "data.ofm.b0", 5U, 400U,
                          sc::kLegacyEvoCmaRegionAlignmentBytes,
                          sc::BackendPortAlignmentAuthority::LegacyPolicy,
                          sc::BackendPortAccess::WriteOnly},
  };
  data.model_outputs = {{0U, "dequant_0", 9U}, {1U, "dequant_1", 10U}};
  std::string error;
  auto plan = sc::ModelExecutionPlan::create(std::move(data), &error);
  require(plan.has_value(), "packed execution plan must be valid: " + error);
  return std::move(*plan);
}

sc::ModelExecutionPlan
make_bf16_consumer_over_int8_unpack_plan(const std::size_t members = 2U) {
  require(members > 0U, "BF16-over-INT8 Unpack plan needs at least one member");
  sc::ModelExecutionPlanData data;
  data.contract_version = "2.1.0";
  data.values = {sc::ValueSpec{0U, "ifm", 64U, "int8", sc::TensorShape{1, 64}},
                 sc::ValueSpec{1U, "packed_ofm", members * 200U}};
  data.model_inputs = {0U};
  const auto add_op = [&](const sc::OpKind kind, std::string name,
                          std::vector<sc::ValueId> inputs,
                          std::vector<sc::ValueId> outputs, sc::OpConfig config,
                          std::vector<sc::TensorShape> input_shapes = {},
                          std::vector<sc::TensorShape> output_shapes = {}) {
    sc::OpSpec op;
    op.id = static_cast<sc::OpId>(data.ops.size());
    op.sequence = data.ops.size() + 1U;
    op.name = std::move(name);
    op.kind = kind;
    op.processor = kind == sc::OpKind::Mla ? "MLA" : "EV74";
    op.inputs = std::move(inputs);
    op.outputs = std::move(outputs);
    op.input_shapes = std::move(input_shapes);
    op.output_shapes = std::move(output_shapes);
    op.config = std::move(config);
    data.ops.push_back(std::move(op));
  };
  add_op(sc::OpKind::Mla, "MLA_0", {0U}, {1U}, sc::MlaOpConfig{"model.elf", 4});
  const sc::TensorShape frame{1, 1, 1, 100};
  const sc::TensorShape slice{1, 1, 1, 16};
  std::vector<sc::ValueId> unpack_outputs;
  sc::UnpackOpConfig unpack;
  for (std::size_t member = 0; member < members; ++member) {
    const auto unpack_id = static_cast<sc::ValueId>(2U + member * 3U);
    const auto detess_id = static_cast<sc::ValueId>(unpack_id + 1U);
    const auto cast_id = static_cast<sc::ValueId>(unpack_id + 2U);
    const auto suffix = std::to_string(member);
    data.values.push_back(
        sc::ValueSpec{unpack_id, "unpack_" + suffix, 200U, "bfloat16",
                      sc::TensorShape{1, 200}, std::nullopt, {},
                      sc::ValueRepresentation::BackendNative,
                      sc::ReadExpression{1U, member * 200U, {200, 1}}});
    data.values.push_back(sc::ValueSpec{detess_id, "detess_" + suffix, 200U, "bfloat16", frame});
    data.values.push_back(sc::ValueSpec{cast_id, "cast_" + suffix, 400U, "float32", frame});
    unpack_outputs.push_back(unpack_id);
    unpack.tensor_types.push_back("int8");
    unpack.tensor_shapes.push_back({1, 200});
    data.model_outputs.push_back({member, "cast_" + suffix, cast_id});
  }
  add_op(sc::OpKind::Unpack, "unpack", {1U}, unpack_outputs, std::move(unpack));
  for (std::size_t member = 0; member < members; ++member) {
    const auto unpack_id = static_cast<sc::ValueId>(2U + member * 3U);
    const auto detess_id = static_cast<sc::ValueId>(unpack_id + 1U);
    const auto cast_id = static_cast<sc::ValueId>(unpack_id + 2U);
    const auto suffix = std::to_string(member);
    add_op(sc::OpKind::Detessellate, "detess_" + suffix, {unpack_id}, {detess_id},
           sc::DetessellateOpConfig{slice, frame, true, true, "bfloat16"},
           {{1, 200}}, {frame});
    add_op(sc::OpKind::Cast, "cast_" + suffix, {detess_id}, {cast_id},
           sc::CastOpConfig{"float32"}, {frame}, {frame});
  }
  data.backend_ports = {
      {0U, sc::BackendPortDirection::Input, 0U, "data.ifm.0", 0U, 64U, 64U,
       sc::BackendPortAlignmentAuthority::Contract, sc::BackendPortAccess::ReadOnly},
      {0U, sc::BackendPortDirection::Output, 0U, "data.ofm.0", 1U, members * 200U, 64U,
       sc::BackendPortAlignmentAuthority::Contract, sc::BackendPortAccess::WriteOnly},
  };
  std::string error;
  auto plan = sc::ModelExecutionPlan::create(std::move(data), &error);
  require(plan.has_value(), "BF16-over-INT8 Unpack plan must be valid: " + error);
  return std::move(*plan);
}

sc::ModelExecutionPlan make_terminal_bf16_over_int8_unpack_plan(
    const bool storage_span_mismatch = false,
    const bool storage_stride_mismatch = false,
    const bool overrun_backend_port = false,
    const bool composed_slice = false) {
  constexpr std::array<std::uint64_t, 6> kSpans = {
      819200U, 204800U, 51200U, 1024000U, 256000U, 64000U};
  constexpr std::uint64_t kParentBytes = 2419200U;
  sc::ModelExecutionPlanData data;
  data.contract_version = "2.1.0";
  data.values = {
      sc::ValueSpec{0U, "input", 256U, "float32", sc::TensorShape{1, 64}},
      sc::ValueSpec{1U, "quantized_ifm", 64U, "int8", sc::TensorShape{1, 64}},
      sc::ValueSpec{2U, "packed_ofm", kParentBytes, "int8",
                    sc::TensorShape{1, static_cast<std::int64_t>(kParentBytes)}},
  };
  data.model_inputs = {0U};

  sc::OpSpec quantize;
  quantize.id = 0U;
  quantize.sequence = 1U;
  quantize.name = "quantized_ifm";
  quantize.kind = sc::OpKind::Quantize;
  quantize.processor = "EV74";
  quantize.inputs = {0U};
  quantize.outputs = {1U};
  quantize.config = sc::QuantizeOpConfig{"int8", 8, "TONEAREST", {}};
  data.ops.push_back(std::move(quantize));

  sc::OpSpec mla;
  mla.id = 1U;
  mla.sequence = 2U;
  mla.name = "MLA_0";
  mla.kind = sc::OpKind::Mla;
  mla.processor = "MLA";
  mla.inputs = {1U};
  mla.outputs = {2U};
  mla.config = sc::MlaOpConfig{"model.elf", 4};
  data.ops.push_back(std::move(mla));

  sc::OpSpec unpack;
  unpack.id = 2U;
  unpack.sequence = 3U;
  unpack.name = "MLA_0_ofm_unpack_transform";
  unpack.kind = sc::OpKind::Unpack;
  unpack.processor = "EV74";
  unpack.kernel = "unpack_transform";
  unpack.inputs = {2U};
  sc::UnpackOpConfig unpack_config;
  std::uint64_t offset = 0U;
  for (std::size_t index = 0U; index < kSpans.size(); ++index) {
    const auto value_id = static_cast<sc::ValueId>(3U + index);
    const auto span = kSpans[index];
    const sc::TensorShape shape{1, static_cast<std::int64_t>(span)};
    sc::ValueSpec value{value_id,
                        "raw_head_" + std::to_string(index),
                        span,
                        "bfloat16",
                        shape,
                        "HWC",
                        {},
                        sc::ValueRepresentation::BackendNative,
                        sc::ReadExpression{2U, offset, {static_cast<std::int64_t>(span), 1}}};
    if (storage_span_mismatch && index + 1U == kSpans.size()) {
      value.storage_binding = sc::StorageBinding{
          sc::StorageBindingKind::View, 2U, offset, span + 1U,
          {static_cast<std::int64_t>(span), 1}, sc::StorageAccess::ReadOnly, 2U};
    } else if (storage_stride_mismatch && index + 1U == kSpans.size()) {
      value.storage_binding = sc::StorageBinding{
          sc::StorageBindingKind::View, 2U, offset, span,
          {static_cast<std::int64_t>(span + 1U), 1}, sc::StorageAccess::ReadOnly, 2U};
    }
    data.values.push_back(std::move(value));
    unpack.outputs.push_back(value_id);
    unpack.output_shapes.push_back(shape);
    unpack_config.tensor_types.push_back("int8");
    unpack_config.tensor_shapes.push_back(shape);
    data.model_outputs.push_back(
        {index, "raw_head_" + std::to_string(index), value_id});
    offset += span;
  }
  if (overrun_backend_port) {
    auto& last = data.values.back();
    last.read_expression->byte_offset += 1U;
    // Keep the normalized storage exact and enlarge only the carrier catalogue;
    // the MLA backend port remains the smaller immutable physical authority.
    last.storage_binding = sc::StorageBinding{
        sc::StorageBindingKind::View, 2U, last.read_expression->byte_offset,
        kSpans.back(), {static_cast<std::int64_t>(kSpans.back()), 1},
        sc::StorageAccess::ReadOnly, 2U};
  }
  unpack.config = std::move(unpack_config);
  data.ops.push_back(std::move(unpack));
  if (composed_slice) {
    const auto slice_id = static_cast<sc::ValueId>(data.values.size());
    data.values.push_back(sc::ValueSpec{
        slice_id, "raw_head_5_slice", 32000U, "bfloat16",
        sc::TensorShape{1, 32000}, "HWC", {},
        sc::ValueRepresentation::BackendNative,
        sc::ReadExpression{2U, 2355200U, {64000, 1}}});
    sc::OpSpec slice;
    slice.id = 3U;
    slice.sequence = 4U;
    slice.name = "raw_head_5_slice";
    slice.kind = sc::OpKind::Slice;
    slice.processor = "EV74";
    slice.inputs = {8U};
    slice.outputs = {slice_id};
    slice.input_shapes = {{1, 64000}};
    slice.output_shapes = {{1, 32000}};
    slice.config = sc::SliceOpConfig{{0, 0}, {1, 32000}, {1, 64000},
                                     {1, 32000}};
    data.ops.push_back(std::move(slice));
    data.model_outputs.back() = {5U, "raw_head_5_slice", slice_id};
  }

  data.backend_ports = {
      {0U, sc::BackendPortDirection::Input, 0U, "data.ifm.b0", 1U, 64U, 4096U,
       sc::BackendPortAlignmentAuthority::Contract,
       sc::BackendPortAccess::ReadOnly},
      {0U, sc::BackendPortDirection::Output, 0U, "data.ofm.b0", 2U,
       kParentBytes, 4096U, sc::BackendPortAlignmentAuthority::Contract,
       sc::BackendPortAccess::WriteOnly},
  };
  if (storage_span_mismatch || overrun_backend_port) {
    data.carriers = {
        {0U, 256U, 4096U, sc::ValueRepresentation::Dense},
        {1U, 64U, 4096U, sc::ValueRepresentation::Dense},
        {2U, kParentBytes + 1U, 4096U, sc::ValueRepresentation::BackendNative},
    };
  }
  std::string error;
  auto plan = sc::ModelExecutionPlan::create(std::move(data), &error);
  require(plan.has_value(),
          "terminal BF16-over-INT8 Unpack plan must be valid: " + error);
  return std::move(*plan);
}

sc::ModelExecutionPlan make_terminal_dense_bf16_plan() {
  sc::ModelExecutionPlanData data;
  data.contract_version = "2.1.0";
  data.values = {
      sc::ValueSpec{0U, "input", 256U, "float32", sc::TensorShape{1, 64}},
      sc::ValueSpec{1U, "quantized_ifm", 64U, "int8", sc::TensorShape{1, 64}},
      sc::ValueSpec{2U, "dense_bf16_ofm", 128U, "bfloat16",
                    sc::TensorShape{1, 64}, "HWC"},
  };
  data.model_inputs = {0U};
  sc::OpSpec quantize;
  quantize.id = 0U;
  quantize.sequence = 1U;
  quantize.name = "quantized_ifm";
  quantize.kind = sc::OpKind::Quantize;
  quantize.processor = "EV74";
  quantize.inputs = {0U};
  quantize.outputs = {1U};
  quantize.config = sc::QuantizeOpConfig{"int8", 8, "TONEAREST", {}};
  data.ops.push_back(std::move(quantize));
  sc::OpSpec mla;
  mla.id = 1U;
  mla.sequence = 2U;
  mla.name = "MLA_0";
  mla.kind = sc::OpKind::Mla;
  mla.processor = "MLA";
  mla.inputs = {1U};
  mla.outputs = {2U};
  mla.config = sc::MlaOpConfig{"model.elf", 4};
  data.ops.push_back(std::move(mla));
  data.backend_ports = {
      {0U, sc::BackendPortDirection::Input, 0U, "data.ifm.b0", 1U, 64U,
       4096U, sc::BackendPortAlignmentAuthority::Contract,
       sc::BackendPortAccess::ReadOnly},
      {0U, sc::BackendPortDirection::Output, 0U, "data.ofm.b0", 2U, 128U,
       4096U, sc::BackendPortAlignmentAuthority::Contract,
       sc::BackendPortAccess::WriteOnly},
  };
  data.model_outputs = {{0U, "dense_bf16_ofm", 2U}};
  std::string error;
  auto plan = sc::ModelExecutionPlan::create(std::move(data), &error);
  require(plan.has_value(), "terminal dense BF16 plan must be valid: " + error);
  return std::move(*plan);
}

sc::ModelExecutionPlan make_pitched_batch_pack_plan(const std::uint32_t batch_count = 2U) {
  require(batch_count > 1U, "pitched Pack fixture needs a multi-sample batch");
  const std::uint64_t child_bytes = static_cast<std::uint64_t>(batch_count) * 4U;
  const std::uint64_t child_span = static_cast<std::uint64_t>(batch_count - 1U) * 8U + 4U;
  const std::uint64_t parent_bytes = static_cast<std::uint64_t>(batch_count) * 8U;
  sc::ModelExecutionPlanData data;
  data.contract_version = "2.1.0";
  const auto binding = [](const sc::StorageBindingKind kind, const sc::CarrierId carrier,
                          const std::uint64_t offset, const std::uint64_t span,
                          std::vector<std::int64_t> strides,
                          const sc::StorageAccess access) {
    return sc::StorageBinding{kind, carrier, offset, span, std::move(strides), access,
                              std::nullopt};
  };
  data.values = {
      sc::ValueSpec{0U, "left_in", child_bytes, "float32",
                    sc::TensorShape{batch_count, 1}, "dense", {},
                    sc::ValueRepresentation::Dense, std::nullopt,
                    binding(sc::StorageBindingKind::External, 0U, 0U, child_bytes, {4, 4},
                            sc::StorageAccess::ReadOnly)},
      sc::ValueSpec{1U, "right_in", child_bytes, "float32",
                    sc::TensorShape{batch_count, 1}, "dense", {},
                    sc::ValueRepresentation::Dense, std::nullopt,
                    binding(sc::StorageBindingKind::External, 1U, 0U, child_bytes, {4, 4},
                            sc::StorageAccess::ReadOnly)},
      sc::ValueSpec{2U, "left", child_bytes, "float32",
                    sc::TensorShape{batch_count, 1}, "dense", {},
                    sc::ValueRepresentation::Dense, std::nullopt,
                    binding(sc::StorageBindingKind::Root, 4U, 0U, child_span, {8, 4},
                            sc::StorageAccess::ReadWrite)},
      sc::ValueSpec{3U, "right", child_bytes, "float32",
                    sc::TensorShape{batch_count, 1}, "dense", {},
                    sc::ValueRepresentation::Dense, std::nullopt,
                    binding(sc::StorageBindingKind::Root, 4U, 4U, child_span, {8, 4},
                            sc::StorageAccess::ReadWrite)},
      sc::ValueSpec{4U, "packed", parent_bytes, "float32",
                    sc::TensorShape{batch_count, 2}, "dense", {},
                    sc::ValueRepresentation::Packed, std::nullopt,
                    binding(sc::StorageBindingKind::Root, 4U, 0U, parent_bytes, {},
                            sc::StorageAccess::ReadWrite)},
      sc::ValueSpec{5U, "ofm", 4U, "float32", sc::TensorShape{1, 1}, "dense", {},
                    sc::ValueRepresentation::BackendNative, std::nullopt,
                    binding(sc::StorageBindingKind::Root, 5U, 0U, 4U, {},
                            sc::StorageAccess::ReadWrite)},
  };
  data.carriers = {
      {0U, child_bytes, 64U, sc::ValueRepresentation::Dense},
      {1U, child_bytes, 64U, sc::ValueRepresentation::Dense},
      {4U, parent_bytes, 64U, sc::ValueRepresentation::Packed},
      {5U, 4U, 64U, sc::ValueRepresentation::BackendNative},
  };
  data.model_inputs = {0U, 1U};
  const auto add_op = [&](const sc::OpKind kind, std::string name,
                          std::vector<sc::ValueId> inputs,
                          std::vector<sc::ValueId> outputs, sc::OpConfig config,
                          std::vector<sc::OpId> dependencies = {}) {
    sc::OpSpec op;
    op.id = static_cast<sc::OpId>(data.ops.size());
    op.sequence = data.ops.size() + 1U;
    op.name = std::move(name);
    op.kind = kind;
    op.processor = kind == sc::OpKind::Mla ? "MLA" : "EV74";
    op.inputs = std::move(inputs);
    op.outputs = std::move(outputs);
    op.config = std::move(config);
    op.dependencies = std::move(dependencies);
    data.ops.push_back(std::move(op));
  };
  add_op(sc::OpKind::Cast, "left_cast", {0U}, {2U}, sc::CastOpConfig{"float32"});
  add_op(sc::OpKind::Cast, "right_cast", {1U}, {3U}, sc::CastOpConfig{"float32"});
  sc::PackOpConfig pack;
  pack.batch_count = batch_count;
  pack.parent_required_bytes = parent_bytes;
  pack.materializes = false;
  for (std::uint32_t batch = 0U; batch < batch_count; ++batch) {
    pack.spans.push_back({2U, batch, batch * 4U, batch * 8U, 4U, 4U, "none"});
    pack.spans.push_back({3U, batch, batch * 4U, batch * 8U + 4U, 4U, 4U, "none"});
  }
  add_op(sc::OpKind::Pack, "pack", {2U, 3U}, {4U}, std::move(pack), {0U, 1U});
  add_op(sc::OpKind::Mla, "mla", {4U}, {5U}, sc::MlaOpConfig{"model.elf", 4}, {2U});
  data.backend_ports = {
      {0U, sc::BackendPortDirection::Input, 0U, "data.ifm.b0", 4U, parent_bytes, 64U,
       sc::BackendPortAlignmentAuthority::Contract, sc::BackendPortAccess::ReadOnly},
      {0U, sc::BackendPortDirection::Output, 0U, "data.ofm.b0", 5U, 4U, 64U,
       sc::BackendPortAlignmentAuthority::Contract, sc::BackendPortAccess::WriteOnly},
  };
  data.model_outputs = {{0U, "ofm", 5U}};
  std::string error;
  auto plan = sc::ModelExecutionPlan::create(std::move(data), &error);
  require(plan.has_value(), "pitched Pack execution plan must be valid: " + error);
  return std::move(*plan);
}

sc::ModelExecutionPlan make_single_read_plan() {
  sc::ModelExecutionPlanData data;
  data.contract_version = "2.0.0";
  data.values = {
      sc::ValueSpec{0U, "ifm", 64U, "int8", sc::TensorShape{1, 1, 1, 64}},
      sc::ValueSpec{1U, "physical_ofm", 192U, "int8", sc::TensorShape{1, 1, 1, 192}},
      sc::ValueSpec{2U,
                    "slice_view",
                    12U,
                    "int8",
                    sc::TensorShape{1, 1, 1, 12},
                    std::nullopt,
                    {},
                    sc::ValueRepresentation::Dense,
                    sc::ReadExpression{1U, 8U, {192, 192, 16, 1}}},
      sc::ValueSpec{3U, "dequantized", 48U, "fp32", sc::TensorShape{1, 1, 1, 12}},
  };
  data.model_inputs = {0U};

  const auto add_op = [&](sc::OpKind kind, std::string name, std::vector<sc::ValueId> inputs,
                          std::vector<sc::ValueId> outputs, sc::OpConfig config) {
    sc::OpSpec op;
    op.id = static_cast<sc::OpId>(data.ops.size());
    op.sequence = data.ops.size() + 1U;
    op.name = std::move(name);
    op.kind = kind;
    op.processor = kind == sc::OpKind::Mla ? "MLA" : "EV74";
    op.inputs = std::move(inputs);
    op.outputs = std::move(outputs);
    op.config = std::move(config);
    data.ops.push_back(std::move(op));
  };
  add_op(sc::OpKind::Mla, "MLA_0", {0U}, {1U}, sc::MlaOpConfig{"model.elf", 4});
  add_op(sc::OpKind::Slice, "slice", {1U}, {2U},
         sc::SliceOpConfig{{0, 0, 0, 0}, {1, 1, 1, 12}, {1, 1, 1, 192}, {1, 1, 1, 12}});
  add_op(sc::OpKind::Dequantize, "dequantize", {2U}, {3U}, sc::DequantizeOpConfig{"int8", {}});

  data.backend_ports = {
      sc::BackendPortSpec{0U, sc::BackendPortDirection::Input, 0U, "data.ifm.b0", 0U, 64U,
                          sc::kLegacyEvoCmaRegionAlignmentBytes,
                          sc::BackendPortAlignmentAuthority::LegacyPolicy,
                          sc::BackendPortAccess::ReadOnly},
      sc::BackendPortSpec{0U, sc::BackendPortDirection::Output, 0U, "data.ofm.b0", 1U, 192U,
                          sc::kLegacyEvoCmaRegionAlignmentBytes,
                          sc::BackendPortAlignmentAuthority::LegacyPolicy,
                          sc::BackendPortAccess::WriteOnly},
  };
  data.model_outputs = {{0U, "dequantized", 3U}};
  std::string error;
  auto plan = sc::ModelExecutionPlan::create(std::move(data), &error);
  require(plan.has_value(), "single-read execution plan must be valid: " + error);
  return std::move(*plan);
}

sc::ModelExecutionPlan make_terminal_two_slice_plan() {
  sc::ModelExecutionPlanData data;
  data.contract_version = "2.1.0";
  data.values = {
      sc::ValueSpec{0U, "ifm", 64U, "int8", sc::TensorShape{64}},
      sc::ValueSpec{1U, "preprocessed", 64U, "int8", sc::TensorShape{64}},
      sc::ValueSpec{2U, "packed_ofm", 64U, "int8", sc::TensorShape{64}},
      sc::ValueSpec{3U, "left_head", 16U, "int8", sc::TensorShape{1, 2, 2, 4},
                    std::nullopt, {}, sc::ValueRepresentation::Dense,
                    sc::ReadExpression{2U, 8U, {32, 16, 4, 1}}},
      sc::ValueSpec{4U, "right_head", 16U, "int8", sc::TensorShape{1, 2, 2, 4},
                    std::nullopt, {}, sc::ValueRepresentation::Dense,
                    sc::ReadExpression{2U, 32U, {32, 16, 4, 1}}},
  };
  data.model_inputs = {0U};

  sc::OpSpec preproc;
  preproc.id = 0U;
  preproc.sequence = 1U;
  preproc.name = "Cast_0";
  preproc.kind = sc::OpKind::Cast;
  preproc.processor = "EV74";
  preproc.inputs = {0U};
  preproc.outputs = {1U};
  preproc.config = sc::CastOpConfig{"int8"};
  data.ops.push_back(std::move(preproc));

  sc::OpSpec mla;
  mla.id = 1U;
  mla.sequence = 2U;
  mla.name = "MLA_0";
  mla.kind = sc::OpKind::Mla;
  mla.processor = "MLA";
  mla.inputs = {1U};
  mla.outputs = {2U};
  mla.config = sc::MlaOpConfig{"model.elf", 4};
  data.ops.push_back(std::move(mla));

  const auto append_slice = [&](const sc::OpId id, const sc::ValueId output,
                                const std::int64_t begin) {
    sc::OpSpec slice;
    slice.id = id;
    slice.sequence = id + 1U;
    slice.name = "slice_" + std::to_string(id);
    slice.kind = sc::OpKind::Slice;
    slice.processor = "HOST";
    slice.inputs = {2U};
    slice.outputs = {output};
    slice.config = sc::SliceOpConfig{{0, 0, 0, begin}, {1, 2, 2, begin + 4},
                                     {1, 2, 2, 16}, {1, 2, 2, 4}};
    data.ops.push_back(std::move(slice));
  };
  append_slice(2U, 3U, 2);
  append_slice(3U, 4U, 8);
  data.backend_ports = {
      {0U, sc::BackendPortDirection::Input, 0U, "data.ifm.b0", 1U, 64U, 64U,
       sc::BackendPortAlignmentAuthority::Contract, sc::BackendPortAccess::ReadOnly},
      {0U, sc::BackendPortDirection::Output, 0U, "data.ofm.b0", 2U, 64U, 64U,
       sc::BackendPortAlignmentAuthority::Contract, sc::BackendPortAccess::WriteOnly},
  };
  data.model_outputs = {{0U, "left_head", 3U}, {1U, "right_head", 4U}};
  std::string error;
  auto plan = sc::ModelExecutionPlan::create(std::move(data), &error);
  require(plan.has_value(), "terminal two-Slice plan must be valid: " + error);
  return std::move(*plan);
}

sc::ModelExecutionPlan make_mla_a65_successor_plan() {
  sc::ModelExecutionPlanData data;
  data.contract_version = "2.1.0";
  data.values = {
      sc::ValueSpec{0U, "ifm", 64U, "uint8", sc::TensorShape{64}},
      sc::ValueSpec{1U, "mla_ofm", 64U, "uint8", sc::TensorShape{64}},
      sc::ValueSpec{2U, "host_ofm", 64U, "uint8", sc::TensorShape{64}},
  };
  data.model_inputs = {0U};

  sc::OpSpec mla;
  mla.id = 0U;
  mla.sequence = 1U;
  mla.name = "MLA_0";
  mla.kind = sc::OpKind::Mla;
  mla.processor = "MLA";
  mla.inputs = {0U};
  mla.outputs = {1U};
  mla.config = sc::MlaOpConfig{"model.elf", 4};
  data.ops.push_back(std::move(mla));

  sc::OpSpec host;
  host.id = 1U;
  host.sequence = 2U;
  host.name = "APU_0";
  host.kind = sc::OpKind::HostTvm;
  host.processor = "A65";
  host.inputs = {1U};
  host.outputs = {2U};
  host.config = sc::HostTvmOpConfig{
      "post.so", {"mla_ofm"}, {{"uint8", {64}}}, {{"uint8", {64}}}, {-1}, {}};
  data.ops.push_back(std::move(host));

  data.backend_ports = {
      {0U, sc::BackendPortDirection::Input, 0U, "data.ifm.b0", 0U, 64U, 64U,
       sc::BackendPortAlignmentAuthority::Contract, sc::BackendPortAccess::ReadOnly},
      {0U, sc::BackendPortDirection::Output, 0U, "data.ofm.b0", 1U, 64U, 64U,
       sc::BackendPortAlignmentAuthority::Contract, sc::BackendPortAccess::WriteOnly},
  };
  data.model_outputs = {{0U, "host_ofm", 2U}};
  std::string error;
  auto plan = sc::ModelExecutionPlan::create(std::move(data), &error);
  require(plan.has_value(), "MLA-to-A65 plan must be valid: " + error);
  return std::move(*plan);
}

sc::ModelExecutionPlan make_two_mla_plan() {
  sc::ModelExecutionPlanData data;
  data.contract_version = "2.0.0";
  data.values = {
      sc::ValueSpec{0U, "input", 64U},
      sc::ValueSpec{1U, "encoded", 64U, "uint8", sc::TensorShape{64}},
      sc::ValueSpec{2U, "decoded", 32U, "uint8", sc::TensorShape{32}},
  };
  data.model_inputs = {0U};
  const auto add_mla = [&](const char* name, const char* executable, const sc::ValueId input,
                           const sc::ValueId output) {
    sc::OpSpec op;
    op.id = static_cast<sc::OpId>(data.ops.size());
    op.sequence = data.ops.size() + 1U;
    op.name = name;
    op.kind = sc::OpKind::Mla;
    op.processor = "MLA";
    op.inputs = {input};
    op.outputs = {output};
    op.config = sc::MlaOpConfig{executable, 4};
    data.ops.push_back(std::move(op));
  };
  add_mla("MLA_encoder", "encoder.so", 0U, 1U);
  add_mla("MLA_decoder", "decoder.elf", 1U, 2U);
  for (std::size_t stage = 0; stage < 2U; ++stage) {
    const sc::ValueId input = stage == 0U ? 0U : 1U;
    const sc::ValueId output = stage == 0U ? 1U : 2U;
    data.backend_ports.push_back(sc::BackendPortSpec{
        stage, sc::BackendPortDirection::Input, 0U, "data.ifm.b0", input,
        data.values[input].required_bytes, sc::kLegacyEvoCmaRegionAlignmentBytes,
        sc::BackendPortAlignmentAuthority::LegacyPolicy, sc::BackendPortAccess::ReadOnly});
    data.backend_ports.push_back(sc::BackendPortSpec{
        stage, sc::BackendPortDirection::Output, 0U, "data.ofm.b0", output,
        data.values[output].required_bytes, sc::kLegacyEvoCmaRegionAlignmentBytes,
        sc::BackendPortAlignmentAuthority::LegacyPolicy, sc::BackendPortAccess::WriteOnly});
  }
  data.model_outputs = {{0U, "decoded", 2U}};
  std::string error;
  auto plan = sc::ModelExecutionPlan::create(std::move(data), &error);
  require(plan.has_value(), "two-MLA plan must be valid: " + error);
  return std::move(*plan);
}

sc::ModelExecutionPlan make_terminal_multi_ofm_plan(const bool exact_outputs = true) {
  sc::ModelExecutionPlanData data;
  data.contract_version = "2.1.0";
  data.values = {
      sc::ValueSpec{0U, "image", 64U, "uint8", sc::TensorShape{64}},
      sc::ValueSpec{1U, "MLA_171_0/pred_boxes", 4800U, "FP32",
                    sc::TensorShape{1, 300, 4}},
      sc::ValueSpec{2U, "MLA_171_1/pred_logits", 109200U, "FP32",
                    sc::TensorShape{1, 300, 91}},
  };
  data.values[2].logical_layout = "normal";
  data.values[2].representation = sc::ValueRepresentation::Dense;
  data.values[2].storage_binding = sc::StorageBinding{
      sc::StorageBindingKind::Root, 2U, 0U, 110396U, {110400, 368, 4},
      sc::StorageAccess::ReadWrite, std::nullopt};
  if (!exact_outputs) {
    data.values[1].logical_dtype.reset();
    data.values[1].logical_shape.reset();
  }
  data.model_inputs = {0U};
  data.model_outputs = {
      {0U, data.values[1].name, 1U},
      {1U, data.values[2].name, 2U},
  };
  sc::OpSpec mla;
  mla.id = 0U;
  mla.sequence = 1U;
  mla.name = "MLA_171";
  mla.kind = sc::OpKind::Mla;
  mla.processor = "MLA";
  mla.inputs = {0U};
  mla.outputs = {1U, 2U};
  mla.input_shapes = {{64}};
  mla.output_shapes = {{1, 300, 4}, {1, 300, 91}};
  mla.config = sc::MlaOpConfig{"MLA_171.elf", 4};
  data.ops.push_back(std::move(mla));
  data.backend_ports = {
      {0U, sc::BackendPortDirection::Input, 0U, "data.ifm.b0", 0U, 64U, 64U,
       sc::BackendPortAlignmentAuthority::Contract, sc::BackendPortAccess::ReadOnly},
      {0U, sc::BackendPortDirection::Output, 0U, "data.ofm.b0", 1U, 4800U, 64U,
       sc::BackendPortAlignmentAuthority::Contract, sc::BackendPortAccess::WriteOnly},
      {0U, sc::BackendPortDirection::Output, 1U,
       "data.ofm.persistent.afe_mla_output_1.b0", 2U, 110400U, 4096U,
       sc::BackendPortAlignmentAuthority::Contract, sc::BackendPortAccess::WriteOnly},
  };
  std::string error;
  auto plan = sc::ModelExecutionPlan::create(std::move(data), &error);
  require(plan.has_value(), "terminal multi-OFM plan must be valid: " + error);
  return std::move(*plan);
}

sc::ModelExecutionPlan make_split_terminal_multi_ofm_plan() {
  sc::ModelExecutionPlanData data;
  data.contract_version = "2.1.0";
  data.values = {
      sc::ValueSpec{0U, "image", 64U, "uint8", sc::TensorShape{64}},
      sc::ValueSpec{1U, "prepared", 64U, "uint8", sc::TensorShape{64}},
      sc::ValueSpec{2U, "MLA_171_0/pred_boxes", 4800U, "FP32",
                    sc::TensorShape{1, 300, 4}},
      sc::ValueSpec{3U, "MLA_171_1/pred_logits", 109200U, "FP32",
                    sc::TensorShape{1, 300, 91}},
  };
  data.values[3].logical_layout = "normal";
  data.values[3].representation = sc::ValueRepresentation::Dense;
  data.values[3].storage_binding = sc::StorageBinding{
      sc::StorageBindingKind::Root, 3U, 0U, 110396U, {110400, 368, 4},
      sc::StorageAccess::ReadWrite, std::nullopt};
  data.model_inputs = {0U};
  data.model_outputs = {
      {0U, data.values[2].name, 2U},
      {1U, data.values[3].name, 3U},
  };
  const auto add_mla = [&](const sc::OpId id, const char* name,
                           const char* executable,
                           std::vector<sc::ValueId> inputs,
                           std::vector<sc::ValueId> outputs) {
    sc::OpSpec op;
    op.id = id;
    op.sequence = id + 1U;
    op.name = name;
    op.kind = sc::OpKind::Mla;
    op.processor = "MLA";
    op.inputs = std::move(inputs);
    op.outputs = std::move(outputs);
    op.config = sc::MlaOpConfig{executable, 4};
    data.ops.push_back(std::move(op));
  };
  add_mla(0U, "MLA_encoder", "encoder.elf", {0U}, {1U});
  add_mla(1U, "MLA_171", "MLA_171.elf", {1U}, {2U, 3U});
  data.backend_ports = {
      {0U, sc::BackendPortDirection::Input, 0U, "data.ifm.b0", 0U, 64U, 64U,
       sc::BackendPortAlignmentAuthority::Contract, sc::BackendPortAccess::ReadOnly},
      {0U, sc::BackendPortDirection::Output, 0U, "data.ofm.b0", 1U, 64U, 64U,
       sc::BackendPortAlignmentAuthority::Contract, sc::BackendPortAccess::WriteOnly},
      {1U, sc::BackendPortDirection::Input, 0U, "data.ifm.b0", 1U, 64U, 64U,
       sc::BackendPortAlignmentAuthority::Contract, sc::BackendPortAccess::ReadOnly},
      {1U, sc::BackendPortDirection::Output, 0U, "data.ofm.b0", 2U, 4800U, 64U,
       sc::BackendPortAlignmentAuthority::Contract, sc::BackendPortAccess::WriteOnly},
      {1U, sc::BackendPortDirection::Output, 1U,
       "data.ofm.persistent.afe_mla_output_1.b0", 3U, 110400U, 4096U,
       sc::BackendPortAlignmentAuthority::Contract, sc::BackendPortAccess::WriteOnly},
  };
  std::string error;
  auto plan = sc::ModelExecutionPlan::create(std::move(data), &error);
  require(plan.has_value(), "split terminal multi-OFM plan must be valid: " + error);
  return std::move(*plan);
}

sima::MlaStaticContract make_projection(const sc::ModelExecutionPlan& plan) {
  sima::MlaStaticContract contract;
  for (const auto& port : plan.backend_ports()) {
    const auto* value = plan.value(port.value_id);
    sima::PhysicalBufferStaticSpec physical;
    physical.physical_index = static_cast<int>(port.port_index);
    physical.size_bytes = value ? value->required_bytes : 0U;
    physical.segment_name = value ? value->name : std::string{};
    if (port.direction == sc::BackendPortDirection::Input) {
      contract.physical_inputs.push_back(std::move(physical));
      sima::TensorStaticSpec logical;
      logical.tensor_index = static_cast<int>(port.port_index);
      contract.logical_inputs.push_back(std::move(logical));
    } else {
      contract.dispatcher_physical_outputs.push_back(std::move(physical));
    }
  }
  return contract;
}

sima::MlaStaticContract make_projection(const sc::ModelExecutionPlan& plan,
                                        const std::size_t stage_index) {
  sima::MlaStaticContract contract;
  for (const auto direction : {sc::BackendPortDirection::Input, sc::BackendPortDirection::Output}) {
    for (const auto& port : plan.backend_ports(stage_index, direction)) {
      const auto* value = plan.value(port.value_id);
      sima::PhysicalBufferStaticSpec physical;
      physical.physical_index = static_cast<int>(port.port_index);
      physical.size_bytes = value ? value->required_bytes : 0U;
      physical.segment_name = value ? value->name : std::string{};
      if (direction == sc::BackendPortDirection::Input) {
        contract.physical_inputs.push_back(std::move(physical));
        sima::TensorStaticSpec logical;
        logical.tensor_index = static_cast<int>(port.port_index);
        contract.logical_inputs.push_back(std::move(logical));
      } else {
        contract.dispatcher_physical_outputs.push_back(std::move(physical));
      }
    }
  }
  return contract;
}

sc::ModelExecutionPlan make_grouped_padded_cast_plan() {
  sc::ModelExecutionPlanData data;
  data.contract_version = "2.1.0";
  const auto storage = [](const sc::StorageBindingKind kind, const sc::CarrierId carrier,
                          const std::uint64_t physical_span,
                          std::vector<std::int64_t> strides,
                          const sc::StorageAccess access) {
    return sc::StorageBinding{kind, carrier, 0U, physical_span, std::move(strides), access,
                              std::nullopt};
  };
  data.carriers = {
      {0U, 16U, 64U, sc::ValueRepresentation::Dense},
      {1U, 9808U, 16U, sc::ValueRepresentation::BackendNative},
      {2U, 2464U, 16U, sc::ValueRepresentation::BackendNative},
      {3U, 19600U, 64U, sc::ValueRepresentation::Dense},
      {4U, 4900U, 64U, sc::ValueRepresentation::Dense},
      {5U, 4U, 64U, sc::ValueRepresentation::BackendNative},
  };
  data.values = {
      sc::ValueSpec{0U, "input", 16U, "BF16", sc::TensorShape{1, 8}, "", {},
                    sc::ValueRepresentation::Dense, std::nullopt,
                    storage(sc::StorageBindingKind::External, 0U, 16U, {16, 2},
                            sc::StorageAccess::ReadOnly)},
      sc::ValueSpec{1U, "MLA_0_0", 9800U, "BF16", sc::TensorShape{1, 1225, 4}, "", {},
                    sc::ValueRepresentation::BackendNative, std::nullopt,
                    storage(sc::StorageBindingKind::Root, 1U, 9808U, {9808, 8, 2},
                            sc::StorageAccess::ReadWrite)},
      sc::ValueSpec{2U, "MLA_0_1", 2450U, "BF16", sc::TensorShape{1, 1225}, "", {},
                    sc::ValueRepresentation::BackendNative, std::nullopt,
                    storage(sc::StorageBindingKind::Root, 2U, 2464U, {2464, 2},
                            sc::StorageAccess::ReadWrite)},
      sc::ValueSpec{3U, "cast_0", 19600U, "FP32", sc::TensorShape{1, 1225, 4}, "", {},
                    sc::ValueRepresentation::Dense, std::nullopt,
                    storage(sc::StorageBindingKind::Root, 3U, 19600U, {19600, 16, 4},
                            sc::StorageAccess::ReadWrite)},
      sc::ValueSpec{4U, "cast_1", 4900U, "FP32", sc::TensorShape{1, 1225}, "", {},
                    sc::ValueRepresentation::Dense, std::nullopt,
                    storage(sc::StorageBindingKind::Root, 4U, 4900U, {4900, 4},
                            sc::StorageAccess::ReadWrite)},
      sc::ValueSpec{5U, "output", 4U, "FP32", sc::TensorShape{1}, "", {},
                    sc::ValueRepresentation::BackendNative, std::nullopt,
                    storage(sc::StorageBindingKind::Root, 5U, 4U, {4},
                            sc::StorageAccess::ReadWrite)},
  };
  data.model_inputs = {0U};
  const auto append_op = [&](const sc::OpKind kind, std::string name,
                             std::vector<sc::ValueId> inputs,
                             std::vector<sc::ValueId> outputs, sc::OpConfig config) {
    sc::OpSpec op;
    op.id = static_cast<sc::OpId>(data.ops.size());
    op.sequence = data.ops.size() + 1U;
    op.name = std::move(name);
    op.kind = kind;
    op.processor = kind == sc::OpKind::Mla ? "MLA" : "EV74";
    op.inputs = std::move(inputs);
    op.outputs = std::move(outputs);
    op.config = std::move(config);
    data.ops.push_back(std::move(op));
  };
  append_op(sc::OpKind::Mla, "MLA_0", {0U}, {1U, 2U}, sc::MlaOpConfig{"first.elf", 4});
  append_op(sc::OpKind::Cast, "cast_0", {1U}, {3U}, sc::CastOpConfig{"FP32"});
  append_op(sc::OpKind::Cast, "cast_1", {2U}, {4U}, sc::CastOpConfig{"FP32"});
  append_op(sc::OpKind::Mla, "MLA_1", {3U, 4U}, {5U}, sc::MlaOpConfig{"last.elf", 4});
  data.backend_ports = {
      {0U, sc::BackendPortDirection::Input, 0U, "data.ifm.input_0", 0U, 16U, 64U,
       sc::BackendPortAlignmentAuthority::Contract, sc::BackendPortAccess::ReadOnly},
      {0U, sc::BackendPortDirection::Output, 0U, "data.ofm.output_0", 1U, 9808U, 16U,
       sc::BackendPortAlignmentAuthority::Contract, sc::BackendPortAccess::WriteOnly},
      {0U, sc::BackendPortDirection::Output, 1U, "data.ofm.output_1", 2U, 2464U, 16U,
       sc::BackendPortAlignmentAuthority::Contract, sc::BackendPortAccess::WriteOnly},
      {1U, sc::BackendPortDirection::Input, 0U, "data.ifm.input_0", 3U, 19600U, 64U,
       sc::BackendPortAlignmentAuthority::Contract, sc::BackendPortAccess::ReadOnly},
      {1U, sc::BackendPortDirection::Input, 1U, "data.ifm.input_1", 4U, 4900U, 64U,
       sc::BackendPortAlignmentAuthority::Contract, sc::BackendPortAccess::ReadOnly},
      {1U, sc::BackendPortDirection::Output, 0U, "data.ofm.output_0", 5U, 4U, 64U,
       sc::BackendPortAlignmentAuthority::Contract, sc::BackendPortAccess::WriteOnly},
  };
  data.model_outputs = {{0U, "output", 5U}};
  std::string error;
  auto plan = sc::ModelExecutionPlan::create(std::move(data), &error);
  require(plan.has_value(), "grouped padded-Cast execution plan must be valid: " + error);
  return std::move(*plan);
}

sc::ModelExecutionPlan make_fused_transform_plan(const sc::OpKind first_kind,
                                                 const sc::OpKind second_kind,
                                                 const std::size_t members,
                                                 const std::uint64_t input_offset = 0U) {
  sc::ModelExecutionPlanData data;
  data.contract_version = "2.1.0";
  const sc::TensorShape shape{1, 2, 2, 16};
  const sc::TensorShape tile{1, 1, 1, 16};
  const auto dtype_bytes = [](const std::string& dtype) {
    return dtype == "FP32" ? 4U : dtype == "BF16" ? 2U : 1U;
  };
  const std::string input_dtype = first_kind == sc::OpKind::Detessellate
                                      ? (second_kind == sc::OpKind::Cast ? "BF16" : "INT8")
                                      : "FP32";
  const std::string middle_dtype = first_kind == sc::OpKind::Quantize ? "INT8"
                                   : first_kind == sc::OpKind::Cast   ? "BF16"
                                                                      : input_dtype;
  const std::string output_dtype = second_kind == sc::OpKind::Dequantize ||
                                           second_kind == sc::OpKind::Cast
                                       ? "FP32"
                                       : middle_dtype;
  std::vector<sc::ValueId> fused_outputs;
  for (std::size_t index = 0; index < members; ++index) {
    const auto input_id = static_cast<sc::ValueId>(data.values.size());
    const auto middle_id = input_id + 1U;
    const auto output_id = input_id + 2U;
    const auto input_bytes = 64U * dtype_bytes(input_dtype);
    const auto middle_bytes = 64U * dtype_bytes(middle_dtype);
    const auto output_bytes = 64U * dtype_bytes(output_dtype);
    const auto input_carrier = static_cast<sc::CarrierId>(data.carriers.size());
    data.carriers.push_back(
        {input_carrier, input_offset + input_bytes, 64U, sc::ValueRepresentation::Dense});
    data.carriers.push_back(
        {input_carrier + 1U, middle_bytes, 64U, sc::ValueRepresentation::Dense});
    const auto output_representation = second_kind == sc::OpKind::Tessellate
                                           ? sc::ValueRepresentation::Tessellated
                                           : sc::ValueRepresentation::Dense;
    data.carriers.push_back(
        {input_carrier + 2U, output_bytes + 32U, 32U, output_representation});
    data.values.push_back(sc::ValueSpec{
        input_id, "input_" + std::to_string(index), input_bytes, input_dtype, shape, "HWC", {},
        sc::ValueRepresentation::Dense, std::nullopt,
        sc::StorageBinding{sc::StorageBindingKind::External, input_carrier, input_offset,
                           input_bytes, {}, sc::StorageAccess::ReadOnly, std::nullopt}});
    data.values.push_back(sc::ValueSpec{
        middle_id, "middle_" + std::to_string(index), middle_bytes, middle_dtype, shape, "HWC",
        {}, sc::ValueRepresentation::Dense, std::nullopt,
        sc::StorageBinding{sc::StorageBindingKind::Root, input_carrier + 1U, 0U, middle_bytes, {},
                           sc::StorageAccess::ReadWrite, std::nullopt}});
    data.values.push_back(sc::ValueSpec{
        output_id, "output_" + std::to_string(index), output_bytes, output_dtype, shape, "HWC",
        {}, output_representation, std::nullopt,
        sc::StorageBinding{sc::StorageBindingKind::Root, input_carrier + 2U, 32U, output_bytes, {},
                           sc::StorageAccess::ReadWrite, std::nullopt}});
    data.model_inputs.push_back(input_id);
    data.model_outputs.push_back({index, data.values.back().name, output_id});
    fused_outputs.push_back(output_id);

    const auto append_op = [&](const sc::OpKind kind, const sc::ValueId input,
                               const sc::ValueId output, sc::OpConfig config) {
      sc::OpSpec op;
      op.id = static_cast<sc::OpId>(data.ops.size());
      op.sequence = data.ops.size() + 1U;
      op.name = "transform_" + std::to_string(op.id);
      op.kind = kind;
      op.processor = "EV74";
      op.inputs = {input};
      op.outputs = {output};
      op.input_shapes = {shape};
      op.output_shapes = {shape};
      op.config = std::move(config);
      data.ops.push_back(std::move(op));
    };
    sc::OpConfig first_config = first_kind == sc::OpKind::Quantize
                                    ? sc::OpConfig{sc::QuantizeOpConfig{
                                          "INT8", 8, "TONEAREST", {{0.25, 0}}}}
                                    : first_kind == sc::OpKind::Cast
                                          ? sc::OpConfig{sc::CastOpConfig{"BF16"}}
                                          : sc::OpConfig{sc::DetessellateOpConfig{
                                                tile, shape, true, true, input_dtype}};
    sc::OpConfig second_config = second_kind == sc::OpKind::Tessellate
                                     ? sc::OpConfig{sc::TessellateOpConfig{
                                           tile, true, true, middle_dtype}}
                                 : second_kind == sc::OpKind::Dequantize
                                     ? sc::OpConfig{sc::DequantizeOpConfig{
                                           "INT8", {{0.25, 0}}}}
                                     : sc::OpConfig{sc::CastOpConfig{"FP32"}};
    append_op(first_kind, input_id, middle_id, std::move(first_config));
    append_op(second_kind, middle_id, output_id, std::move(second_config));
  }
  if (members > 1U && second_kind == sc::OpKind::Tessellate) {
    // Give the grouped ingress cohort a real MLA boundary.  Without a boundary,
    // the physical lowerer intentionally emits independent residual singleton
    // cohorts, which must never be merged by projection.
    data.model_outputs.clear();
    const auto output_id = static_cast<sc::ValueId>(data.values.size());
    const auto carrier_id = static_cast<sc::CarrierId>(data.carriers.size());
    data.carriers.push_back(
        {carrier_id, 64U, 64U, sc::ValueRepresentation::BackendNative});
    data.values.push_back(sc::ValueSpec{
        output_id, "mla_output", 64U, "INT8", shape, "HWC", {},
        sc::ValueRepresentation::BackendNative, std::nullopt,
        sc::StorageBinding{sc::StorageBindingKind::Root, carrier_id, 0U, 64U, {},
                           sc::StorageAccess::ReadWrite, std::nullopt}});
    sc::OpSpec mla;
    mla.id = static_cast<sc::OpId>(data.ops.size());
    mla.sequence = data.ops.size() + 1U;
    mla.name = "MLA_0";
    mla.kind = sc::OpKind::Mla;
    mla.processor = "MLA";
    mla.inputs = fused_outputs;
    mla.outputs = {output_id};
    mla.input_shapes.assign(fused_outputs.size(), shape);
    mla.output_shapes = {shape};
    mla.config = sc::MlaOpConfig{"model.elf", 4};
    data.ops.push_back(std::move(mla));
    for (std::size_t index = 0; index < fused_outputs.size(); ++index) {
      const auto* value = &data.values[fused_outputs[index]];
      data.backend_ports.push_back(
          {0U, sc::BackendPortDirection::Input, index,
           "data.ifm." + std::to_string(index), fused_outputs[index], value->required_bytes,
           32U, sc::BackendPortAlignmentAuthority::Contract,
           sc::BackendPortAccess::ReadOnly});
    }
    data.backend_ports.push_back(
        {0U, sc::BackendPortDirection::Output, 0U, "data.ofm.0", output_id, 64U, 64U,
         sc::BackendPortAlignmentAuthority::Contract,
         sc::BackendPortAccess::WriteOnly});
    data.model_outputs.push_back({0U, "mla_output", output_id});
  }
  std::string error;
  auto plan = sc::ModelExecutionPlan::create(std::move(data), &error);
  require(plan.has_value(), "fused transform plan must be valid: " + error);
  return std::move(*plan);
}

sc::ModelExecutionPlan make_model_managed_preproc_absorption_plan() {
  sc::ModelExecutionPlanData data;
  data.contract_version = "2.1.0";
  const sc::TensorShape shape{1, 2, 2, 3};
  data.carriers = {
      {0U, 48U, 64U, sc::ValueRepresentation::Dense},
      {1U, 12U, 64U, sc::ValueRepresentation::Dense},
      {2U, 4U, 64U, sc::ValueRepresentation::BackendNative},
  };
  data.values = {
      sc::ValueSpec{0U, "images", 48U, "FP32", shape, "HWC", {},
                    sc::ValueRepresentation::Dense, std::nullopt,
                    sc::StorageBinding{sc::StorageBindingKind::External, 0U, 0U, 48U, {},
                                       sc::StorageAccess::ReadOnly, std::nullopt}},
      sc::ValueSpec{1U, "quantize_0", 12U, "INT8", shape, "HWC", {{0.25, -128}},
                    sc::ValueRepresentation::Dense, std::nullopt,
                    sc::StorageBinding{sc::StorageBindingKind::Root, 1U, 0U, 12U, {},
                                       sc::StorageAccess::ReadWrite, std::nullopt}},
      sc::ValueSpec{2U, "ofm", 4U, "INT8", sc::TensorShape{4}, std::nullopt, {},
                    sc::ValueRepresentation::BackendNative, std::nullopt,
                    sc::StorageBinding{sc::StorageBindingKind::Root, 2U, 0U, 4U, {},
                                       sc::StorageAccess::ReadWrite, std::nullopt}},
  };
  data.model_inputs = {0U};
  sc::OpSpec quant;
  quant.id = 0U;
  quant.sequence = 1U;
  quant.name = "quantization_transform";
  quant.kind = sc::OpKind::Quantize;
  quant.processor = "EV74";
  quant.inputs = {0U};
  quant.outputs = {1U};
  quant.input_shapes = {shape};
  quant.output_shapes = {shape};
  quant.config = sc::QuantizeOpConfig{"INT8", 8, "TONEAREST", {{0.25, -128}}};
  data.ops.push_back(std::move(quant));
  sc::OpSpec mla;
  mla.id = 1U;
  mla.sequence = 2U;
  mla.name = "MLA_0";
  mla.kind = sc::OpKind::Mla;
  mla.processor = "MLA";
  mla.inputs = {1U};
  mla.outputs = {2U};
  mla.input_shapes = {shape};
  mla.output_shapes = {sc::TensorShape{4}};
  mla.config = sc::MlaOpConfig{"model.elf", 4};
  data.ops.push_back(std::move(mla));
  data.backend_ports = {
      {0U, sc::BackendPortDirection::Input, 0U, "data.ifm.input_0", 1U, 12U, 64U,
       sc::BackendPortAlignmentAuthority::Contract, sc::BackendPortAccess::ReadOnly},
      {0U, sc::BackendPortDirection::Output, 0U, "data.ofm.output_0", 2U, 4U, 64U,
       sc::BackendPortAlignmentAuthority::Contract, sc::BackendPortAccess::WriteOnly},
  };
  data.model_outputs = {{0U, "ofm", 2U}};
  std::string error;
  auto plan = sc::ModelExecutionPlan::create(std::move(data), &error);
  require(plan.has_value(), "graph-200 absorption plan must be valid: " + error);
  return std::move(*plan);
}

sc::ModelExecutionPlan make_model_managed_tess_preproc_absorption_plan() {
  sc::ModelExecutionPlanData data;
  data.contract_version = "2.1.0";
  const sc::TensorShape shape{1, 2, 2, 3};
  const sc::TensorShape tile{2, 2, 3};
  data.carriers = {
      {0U, 48U, 64U, sc::ValueRepresentation::Dense},
      {1U, 12U, 64U, sc::ValueRepresentation::Dense},
      {2U, 12U, 64U, sc::ValueRepresentation::Tessellated},
      {3U, 4U, 64U, sc::ValueRepresentation::BackendNative},
  };
  data.values = {
      sc::ValueSpec{0U, "images", 48U, "FP32", shape, "HWC", {},
                    sc::ValueRepresentation::Dense, std::nullopt,
                    sc::StorageBinding{sc::StorageBindingKind::External, 0U, 0U, 48U, {},
                                       sc::StorageAccess::ReadOnly, std::nullopt}},
      sc::ValueSpec{1U, "quantize_0", 12U, "INT8", shape, "HWC", {{0.25, -128}},
                    sc::ValueRepresentation::Dense, std::nullopt,
                    sc::StorageBinding{sc::StorageBindingKind::Root, 1U, 0U, 12U, {},
                                       sc::StorageAccess::ReadWrite, std::nullopt}},
      sc::ValueSpec{2U, "tessellated_quantize_0", 12U, "INT8", shape, "HWC", {},
                    sc::ValueRepresentation::Tessellated, std::nullopt,
                    sc::StorageBinding{sc::StorageBindingKind::Root, 2U, 0U, 12U, {},
                                       sc::StorageAccess::ReadWrite, std::nullopt}},
      sc::ValueSpec{3U, "ofm", 4U, "INT8", sc::TensorShape{4}, std::nullopt, {},
                    sc::ValueRepresentation::BackendNative, std::nullopt,
                    sc::StorageBinding{sc::StorageBindingKind::Root, 3U, 0U, 4U, {},
                                       sc::StorageAccess::ReadWrite, std::nullopt}},
  };
  data.model_inputs = {0U};
  sc::OpSpec quant;
  quant.id = 0U;
  quant.sequence = 1U;
  quant.name = "quantization_transform";
  quant.kind = sc::OpKind::Quantize;
  quant.processor = "EV74";
  quant.inputs = {0U};
  quant.outputs = {1U};
  quant.input_shapes = {shape};
  quant.output_shapes = {shape};
  quant.config = sc::QuantizeOpConfig{"INT8", 8, "TONEAREST", {{0.25, -128}}};
  data.ops.push_back(std::move(quant));
  sc::OpSpec tess;
  tess.id = 1U;
  tess.sequence = 2U;
  tess.name = "tessellation_transform";
  tess.kind = sc::OpKind::Tessellate;
  tess.processor = "EV74";
  tess.inputs = {1U};
  tess.outputs = {2U};
  tess.input_shapes = {shape};
  tess.output_shapes = {shape};
  tess.config = sc::TessellateOpConfig{tile, false, false, "INT8"};
  data.ops.push_back(std::move(tess));
  sc::OpSpec mla;
  mla.id = 2U;
  mla.sequence = 3U;
  mla.name = "MLA_0";
  mla.kind = sc::OpKind::Mla;
  mla.processor = "MLA";
  mla.inputs = {2U};
  mla.outputs = {3U};
  mla.input_shapes = {shape};
  mla.output_shapes = {sc::TensorShape{4}};
  mla.config = sc::MlaOpConfig{"model.elf", 4};
  data.ops.push_back(std::move(mla));
  data.backend_ports = {
      {0U, sc::BackendPortDirection::Input, 0U, "data.ifm.input_0", 2U, 12U, 64U,
       sc::BackendPortAlignmentAuthority::Contract, sc::BackendPortAccess::ReadOnly},
      {0U, sc::BackendPortDirection::Output, 0U, "data.ofm.output_0", 3U, 4U, 64U,
       sc::BackendPortAlignmentAuthority::Contract, sc::BackendPortAccess::WriteOnly},
  };
  data.model_outputs = {{0U, "ofm", 3U}};
  std::string error;
  auto plan = sc::ModelExecutionPlan::create(std::move(data), &error);
  require(plan.has_value(), "tessellated graph-200 absorption plan must be valid: " + error);
  return std::move(*plan);
}

} // namespace

RUN_TEST(
    "unit_dmabuf_plan_contract_projection_test", ([] {
      const auto plan = make_plan();
      auto contract = make_projection(plan);
      std::string error;
      auto input_sources = sc::resolve_mla_input_physical_sources(plan, {}, &error);
      require(input_sources.has_value(), "model-input carriers must resolve: " + error);
      require(sc::apply_dmabuf_plan_contract_projection(plan, &contract, *input_sources, &error),
              "exact 2/28 projection must pass: " + error);
      require(contract.consumer_keeps_distinct_physical_inputs, "two IFMs must remain distinct");
      require(contract.elf_ifm_symbol_names.size() == 2U,
              "all exact IFM symbols must be projected");
      require(contract.elf_ofm_symbol_names.size() == 28U,
              "all exact OFM symbols must be projected");
      require(contract.input_bindings.size() == 2U &&
                  contract.input_bindings[0].src_logical_output_index == 0 &&
                  contract.input_bindings[1].src_logical_output_index == 1,
              "compatibility bindings must select ordered physical carriers");
      require(contract.physical_inputs[0].source_physical_index == 0 &&
                  contract.physical_inputs[1].source_physical_index == 1 &&
                  contract.physical_inputs[0].source_byte_offset == 0 &&
                  contract.physical_inputs[1].source_byte_offset == 0,
              "IFM ValueIds must lower to canonical physical carriers");
      require(contract.physical_outputs.size() == 28U && contract.logical_outputs.size() == 28U,
              "ProcessMLA must publish all raw backend OFMs");
      require(contract.physical_inputs[0].required_alignment_bytes ==
                      plan.backend_ports()[0].required_alignment_bytes &&
                  contract.physical_inputs[1].required_alignment_bytes ==
                      plan.backend_ports()[1].required_alignment_bytes,
              "Core must project the exact IFM alignment contract");
      require(contract.physical_outputs[7].required_alignment_bytes ==
                      plan.backend_ports()[2U + 7U].required_alignment_bytes &&
                  contract.dispatcher_physical_outputs[7].required_alignment_bytes ==
                      plan.backend_ports()[2U + 7U].required_alignment_bytes,
              "Core must project the exact OFM alignment contract");
      require(contract.logical_outputs[7].logical_index == 7 &&
                  contract.logical_outputs[7].backend_output_index == 7 &&
                  contract.logical_outputs[7].size_bytes ==
                      plan.value(plan.backend_ports()[2U + 7U].value_id)->required_bytes,
              "raw OFM publication must preserve backend order and extent");
      require(contract.frame_arena_size_bytes > 0U &&
                  contract.frame_arena_role == sima::FrameArenaRole::Allocate,
              "direct MLA must allocate the first common frame arena");

      const auto terminal_multi_ofm_plan = make_terminal_multi_ofm_plan();
      auto terminal_multi_ofm_contract = make_projection(terminal_multi_ofm_plan);
      auto terminal_multi_ofm_sources =
          sc::resolve_mla_input_physical_sources(terminal_multi_ofm_plan, {}, &error);
      require(terminal_multi_ofm_sources.has_value(),
              "terminal multi-OFM input carrier must resolve: " + error);
      require(sc::apply_dmabuf_plan_contract_projection(
                  terminal_multi_ofm_plan, &terminal_multi_ofm_contract,
                  *terminal_multi_ofm_sources, &error),
              "terminal multi-OFM projection must pass: " + error);
      require(terminal_multi_ofm_contract.logical_outputs.size() == 2U &&
                  terminal_multi_ofm_contract.logical_outputs[0].dtype == "FP32" &&
                  terminal_multi_ofm_contract.logical_outputs[0].dtype_source ==
                      sima::DTypeSource::InternalContract &&
                  terminal_multi_ofm_contract.logical_outputs[0].shape ==
                      std::vector<std::int64_t>({1, 300, 4}) &&
                  terminal_multi_ofm_contract.logical_outputs[1].dtype == "FP32" &&
                  terminal_multi_ofm_contract.logical_outputs[1].dtype_source ==
                      sima::DTypeSource::InternalContract &&
                  terminal_multi_ofm_contract.logical_outputs[1].shape ==
                      std::vector<std::int64_t>({1, 300, 91}) &&
                  terminal_multi_ofm_contract.logical_outputs[1].size_bytes == 109200U &&
                  terminal_multi_ofm_contract.logical_outputs[1].stride_bytes ==
                      std::vector<std::int64_t>({110400, 368, 4}) &&
                  terminal_multi_ofm_contract.physical_outputs[1].size_bytes == 110400U &&
                  terminal_multi_ofm_contract.dispatcher_physical_outputs[1].size_bytes ==
                      110400U,
              "strict MLA projection must retain authoritative terminal FP32 metadata");

      auto terminal_multi_physical =
          sc::PhysicalExecutionLowerer::lower(terminal_multi_ofm_plan, &error);
      require(terminal_multi_physical.has_value(),
              "terminal multi-OFM physical plan must lower: " + error);
      const auto terminal_multi_policy = sc::select_mla_output_carrier_policy(
          terminal_multi_ofm_plan, *terminal_multi_physical, 0U);
      const auto terminal_multi_detached = sc::detached_mla_output_roots(
          terminal_multi_ofm_plan, *terminal_multi_physical);
      auto terminal_multi_shared_arena = sc::FrameSlotArenaPlan::compile(
          terminal_multi_ofm_plan, *terminal_multi_physical,
          sc::FrameSlotArenaReuse::DisjointLifetimes,
          sc::kLegacyEvoCmaRegionAlignmentBytes, &error,
          sc::kModalixProductionArenaDmsPolicy, terminal_multi_detached);
      require(terminal_multi_policy ==
                  sc::MlaOutputCarrierPolicy::SharedFrameArena &&
                  terminal_multi_detached.empty() &&
                  terminal_multi_shared_arena.has_value() &&
                  terminal_multi_shared_arena->region(1U) != nullptr &&
                  terminal_multi_shared_arena->region(2U) != nullptr &&
                  terminal_multi_shared_arena->placement().domain ==
                      sc::ArenaStorageDomain::Dms,
              "public-input terminal MLA must keep its already-compact EV-free DMS output "
              "arena rather than creating an empty shared plan");
      auto contradictory_terminal = make_projection(terminal_multi_ofm_plan);
      error.clear();
      require(!sc::apply_dmabuf_plan_contract_projection(
                  terminal_multi_ofm_plan, 0U, *terminal_multi_shared_arena,
                  sc::MlaOutputCarrierPolicy::SeparateCpuVisible,
                  &contradictory_terminal, *terminal_multi_ofm_sources, &error) &&
                  error.find("contradicts the frame-arena authority") !=
                      std::string::npos,
              "MLA projection must reject a caller-authored policy which contradicts the arena");
      const auto split_terminal_multi_plan = make_split_terminal_multi_ofm_plan();
      auto split_terminal_physical =
          sc::PhysicalExecutionLowerer::lower(split_terminal_multi_plan, &error);
      require(split_terminal_physical.has_value(),
              "split terminal multi-OFM physical plan must lower: " + error);
      const auto split_terminal_detached = sc::detached_mla_output_roots(
          split_terminal_multi_plan, *split_terminal_physical);
      auto split_terminal_arena = sc::FrameSlotArenaPlan::compile(
          split_terminal_multi_plan, *split_terminal_physical,
          sc::FrameSlotArenaReuse::DisjointLifetimes,
          sc::kLegacyEvoCmaRegionAlignmentBytes, &error,
          sc::kModalixProductionArenaDmsPolicy, split_terminal_detached);
      require(split_terminal_arena.has_value() &&
                  split_terminal_detached == std::vector<sc::ValueId>({2U, 3U}) &&
                  sc::mla_output_carrier_policy_from_arena(
                      split_terminal_multi_plan, *split_terminal_arena, 1U) ==
                      sc::MlaOutputCarrierPolicy::SeparateCpuVisible,
              "connected terminal multi-OFM route must author the split policy in its arena");
      auto split_terminal_sources = sc::resolve_mla_input_physical_sources(
          split_terminal_multi_plan, 1U, *split_terminal_arena, {}, &error);
      require(split_terminal_sources.has_value(),
              "split terminal multi-OFM input must resolve from the retained arena: " + error);
      auto split_terminal_multi = make_projection(split_terminal_multi_plan, 1U);
      require(sc::apply_dmabuf_plan_contract_projection(
                  split_terminal_multi_plan, 1U, *split_terminal_arena,
                  &split_terminal_multi, *split_terminal_sources, &error),
              "terminal multi-OFM split projection must pass: " + error);
      require(split_terminal_multi.physical_outputs.size() == 2U &&
                  split_terminal_multi.physical_outputs[0].source_byte_offset == 0 &&
                  split_terminal_multi.physical_outputs[1].source_byte_offset == 8192 &&
                  split_terminal_multi.frame_arena_size_bytes == 118784U &&
                  split_terminal_multi.logical_outputs.size() == 2U &&
                  split_terminal_multi.logical_outputs[1].physical_index == 1 &&
                  split_terminal_multi.logical_outputs[1].size_bytes == 109200U &&
                  split_terminal_multi.logical_outputs[1].stride_bytes ==
                      std::vector<std::int64_t>({110400, 368, 4}),
              "split multi-OFM layout must preserve unequal alignments, the required gap, "
              "final allocation rounding, and padded logical strides");

      sima::SimaPluginStaticManifest terminal_manifest;
      sima::StageStaticSpec terminal_stage;
      terminal_stage.logical_stage_id = "MLA_171";
      terminal_stage.element_name = "simaaiprocessmla_171";
      terminal_stage.payload_kind = sima::StagePayloadKind::ProcessMla;
      terminal_stage.processmla.dispatcher_output_names = {
          "MLA_171_0/pred_boxes", "MLA_171_1/pred_logits"};
      terminal_stage.processmla.dispatcher_output_sizes = {4800U, 109200U};
      terminal_stage.physical_outputs = terminal_multi_ofm_contract.physical_outputs;
      terminal_stage.logical_outputs = terminal_multi_ofm_contract.logical_outputs;
      terminal_manifest.stages.push_back(std::move(terminal_stage));
      auto terminal_override =
          simaai::neat::pipeline_internal::terminal_output_contract::
              build_output_override_from_manifest(terminal_manifest, {}, &error);
      require(terminal_override.has_value(),
              "projected terminal multi-OFM contract must publish: " + error);
      require(terminal_override->outputs.size() == 2U &&
                  terminal_override->outputs[0].dtype == simaai::neat::TensorDType::Float32 &&
                  terminal_override->outputs[0].shape ==
                      std::vector<std::int64_t>({1, 300, 4}) &&
                  terminal_override->outputs[1].dtype == simaai::neat::TensorDType::Float32 &&
                  terminal_override->outputs[1].shape ==
                      std::vector<std::int64_t>({1, 300, 91}) &&
                  terminal_override->outputs[1].strides_bytes ==
                      std::vector<std::int64_t>({110400, 368, 4}),
              "terminal override must not demote exact MLA outputs to UINT8 carriers");

      const auto incomplete_terminal_plan = make_terminal_multi_ofm_plan(false);
      auto incomplete_terminal_contract = make_projection(incomplete_terminal_plan);
      auto incomplete_terminal_sources =
          sc::resolve_mla_input_physical_sources(incomplete_terminal_plan, {}, &error);
      require(incomplete_terminal_sources.has_value(),
              "incomplete terminal fixture input carrier must still resolve: " + error);
      error.clear();
      require(!sc::apply_dmabuf_plan_contract_projection(
                  incomplete_terminal_plan, &incomplete_terminal_contract,
                  *incomplete_terminal_sources, &error) &&
                  error.find("no exact typed logical output contract") != std::string::npos,
              "strict MLA projection must reject missing public dtype/shape evidence");

      const auto padded_cast_plan = make_grouped_padded_cast_plan();
      auto padded_cast_physical = sc::PhysicalExecutionLowerer::lower(padded_cast_plan, &error);
      require(padded_cast_physical.has_value(),
              "grouped padded-Cast plan must lower: " + error);
      std::vector<sc::PhysicalCommandId> padded_cast_commands;
      for (const auto& command : padded_cast_physical->commands) {
        if (command.engine == sc::PhysicalEngine::Cvu && command.graph_id == 221U) {
          padded_cast_commands.push_back(command.id);
        }
      }
      require(padded_cast_commands.size() == 1U,
              "the aligned padded-Cast lanes must form one graph221 cohort");
      auto padded_cast_arena = sc::FrameSlotArenaPlan::compile(
          padded_cast_plan, *padded_cast_physical,
          sc::FrameSlotArenaReuse::DisjointLifetimes,
          sc::kLegacyEvoCmaRegionAlignmentBytes, &error);
      require(padded_cast_arena.has_value(),
              "grouped padded-Cast arena must compile: " + error);
      auto padded_cast_contract = sc::build_dmabuf_plan_processcvu_command_contract(
          padded_cast_plan, *padded_cast_physical, padded_cast_commands,
          *padded_cast_arena, &error);
      require(padded_cast_contract.has_value(),
              "grouped padded-Cast contract must build: " + error);
      require(padded_cast_contract->payload.input_tensors.size() == 2U &&
                  padded_cast_contract->runtime_contract.physical_inputs.size() == 2U &&
                  padded_cast_contract->payload.input_tensors[0].storage.nbytes == 9800U &&
                  padded_cast_contract->payload.input_tensors[1].storage.nbytes == 2450U &&
                  padded_cast_contract->payload.input_tensors[0]
                          .layout.strided.strides_bytes[0] == 9808 &&
                  padded_cast_contract->payload.input_tensors[1]
                          .layout.strided.strides_bytes[0] == 2464 &&
                  padded_cast_contract->runtime_contract.physical_inputs[0].size_bytes ==
                      9808U &&
                  padded_cast_contract->runtime_contract.physical_inputs[1].size_bytes ==
                      2464U,
              "graph221 must keep QMLA carrier padding in physical buffers while its tensor "
              "descriptors retain exact addressed spans and strides");

      struct FusedCase {
        sc::OpKind first;
        sc::OpKind second;
        std::uint32_t graph_id;
        std::size_t members;
        std::uint64_t input_offset;
      };
      const std::array<FusedCase, 4U> fused_cases{{
          {sc::OpKind::Cast, sc::OpKind::Tessellate, 224U, 1U, 0U},
          {sc::OpKind::Detessellate, sc::OpKind::Cast, 225U, 1U, 64U},
          {sc::OpKind::Quantize, sc::OpKind::Tessellate, 226U, 6U, 0U},
          {sc::OpKind::Detessellate, sc::OpKind::Dequantize, 227U, 1U, 0U},
      }};
      for (const auto& fused_case : fused_cases) {
        const auto fused_plan = make_fused_transform_plan(
            fused_case.first, fused_case.second, fused_case.members, fused_case.input_offset);
        auto fused_physical = sc::PhysicalExecutionLowerer::lower(fused_plan, &error);
        require(fused_physical.has_value(), "fused transform must lower: " + error);
        std::vector<sc::PhysicalCommandId> fused_commands;
        for (const auto& command : fused_physical->commands) {
          if (command.engine == sc::PhysicalEngine::Cvu &&
              command.graph_id == fused_case.graph_id) {
            fused_commands.push_back(command.id);
          }
        }
        require(!fused_commands.empty(), "fused graph id was not selected");
        auto fused_arena = sc::FrameSlotArenaPlan::compile(
            fused_plan, *fused_physical, sc::FrameSlotArenaReuse::DisjointLifetimes,
            sc::kLegacyEvoCmaRegionAlignmentBytes, &error);
        require(fused_arena.has_value(), "fused frame arena must compile: " + error);
        for (std::size_t member = 0; member < fused_case.members; ++member) {
          require(fused_arena->region(static_cast<sc::ValueId>(member * 3U + 1U)) == nullptr,
                  "fused semantic intermediate must not own an arena region");
        }
        auto fused_contract = sc::build_dmabuf_plan_processcvu_command_contract(
            fused_plan, *fused_physical, fused_commands, *fused_arena, &error);
        require(fused_contract.has_value(),
                "fused graph " + std::to_string(fused_case.graph_id) +
                    " command contract must build: " + error);
        require(static_cast<std::uint32_t>(fused_contract->payload.graph_id) ==
                        fused_case.graph_id &&
                    fused_contract->payload.input_tensors.size() == fused_case.members &&
                    fused_contract->payload.output_tensors.size() == fused_case.members &&
                    fused_contract->runtime_contract.physical_inputs.size() ==
                        fused_case.members &&
                    fused_contract->runtime_contract.physical_outputs.size() ==
                        fused_case.members,
                "fused command must publish exactly one outer input/output per member");
        if (fused_case.members == 1U) {
          require(fused_contract->payload.default_output_names.size() == 1U &&
                      fused_contract->payload.default_output_names.front() == "output_0" &&
                      fused_contract->payload.primary_output_name == "output_0",
                  "single-member fused command must use its outer value as the canonical "
                  "runtime and published output identity");
        } else {
          require(fused_contract->payload.default_output_names.size() ==
                          fused_case.members &&
                      fused_contract->payload.primary_output_name == "output_0",
                  "grouped fused command must preserve every outer runtime output identity");
          for (std::size_t member = 0; member < fused_case.members; ++member) {
            require(fused_contract->payload.default_output_names[member] ==
                        "output_" + std::to_string(member),
                    "grouped fused output order must match the physical member order");
          }
        }
        for (std::size_t member = 0; member < fused_case.members; ++member) {
          const auto output_id = static_cast<sc::ValueId>(member * 3U + 2U);
          const auto* output_region = fused_arena->region(output_id);
          require(output_region &&
                      fused_contract->runtime_contract.physical_outputs[member]
                              .source_byte_offset ==
                          static_cast<std::int64_t>(output_region->byte_offset + 32U),
                  "fused output must preserve its exact nonzero parent offset");
        }
        if (fused_case.graph_id == 225U) {
          require(fused_contract->payload.opt_flags != 0U &&
                      fused_contract->runtime_contract.logical_inputs.front()
                              .materialization_kind == sima::TensorMaterializationKind::OffsetView,
                  "graph225 C16 lane layout must retain opt flags and its offset-view binding");
        }
      }

      auto wrong_size = make_projection(plan);
      ++wrong_size.dispatcher_physical_outputs[7].size_bytes;
      require(!sc::apply_dmabuf_plan_contract_projection(plan, &wrong_size, *input_sources, &error),
              "an OFM byte mismatch must fail closed");

      auto wrong_name = make_projection(plan);
      wrong_name.physical_inputs[1].segment_name = "guessed_ifm";
      require(!sc::apply_dmabuf_plan_contract_projection(plan, &wrong_name, *input_sources, &error),
              "a guessed IFM name must fail closed");

      auto aliased_sources = *input_sources;
      aliased_sources[1].source_physical_index = aliased_sources[0].source_physical_index;
      auto aliased_contract = make_projection(plan);
      require(!sc::apply_dmabuf_plan_contract_projection(plan, &aliased_contract, aliased_sources,
                                                         &error),
              "two IFMs must not alias one physical carrier");

      // Real AFE plans commonly create the MLA IFMs in frontend CVU ops.
      // The setup lowering must preserve those graph values while mapping
      // them to the exact (possibly non-identity) upstream logical slots.
      const auto frontend_plan = make_frontend_plan();
      std::vector<sima::LogicalTensorStaticSpec> upstream(2U);
      upstream[0].logical_index = 9;
      upstream[0].backend_name = "quantize_1";
      upstream[0].physical_index = 1;
      upstream[0].size_bytes = 319488U;
      upstream[1].logical_index = 4;
      upstream[1].segment_name = "quantize_0";
      upstream[1].physical_index = 0;
      upstream[1].size_bytes = 638976U;
      auto frontend_sources =
          sc::resolve_mla_input_physical_sources(frontend_plan, upstream, &error);
      require(frontend_sources.has_value(),
              "frontend-produced IFM carriers must resolve: " + error);
      require(frontend_sources->size() == 2U && (*frontend_sources)[0].value_id == 2U &&
                  (*frontend_sources)[0].source_physical_index == 0 &&
                  (*frontend_sources)[1].value_id == 3U &&
                  (*frontend_sources)[1].source_physical_index == 1,
              "frontend-produced IFMs must retain backend order and exact carriers");
      auto frontend_mla = make_projection(frontend_plan);
      require(sc::apply_dmabuf_plan_contract_projection(frontend_plan, &frontend_mla,
                                                        *frontend_sources, &error),
              "frontend MLA frame-arena projection must pass: " + error);
      require(frontend_mla.frame_arena_role == sima::FrameArenaRole::Allocate &&
                  frontend_mla.frame_arena_storage_domain == sc::ArenaStorageDomain::Dms &&
                  frontend_mla.frame_arena_size_bytes > 0U &&
                  frontend_mla.physical_inputs[0].source_physical_index == 0 &&
                  frontend_mla.physical_inputs[1].source_physical_index == 0 &&
                  frontend_mla.physical_inputs[0].source_byte_offset !=
                      frontend_mla.physical_inputs[1].source_byte_offset,
              "terminal MLA must import distinct pre-CVU regions and own its output pool");

      sima::ProcessCvuStagePayload frontend_cvu;
      frontend_cvu.graph_family_enum = sima::ProcessCvuGraphFamily::Quant;
      simaai::neat::CompiledRuntimeContract frontend_runtime;
      for (std::size_t index = 0; index < 2U; ++index) {
        sima::PhysicalBufferStaticSpec output;
        output.physical_index = static_cast<int>(index);
        output.size_bytes = frontend_plan.values()[2U + index].required_bytes;
        output.segment_name = frontend_plan.values()[2U + index].name;
        frontend_runtime.physical_outputs.push_back(std::move(output));
        sima::LogicalTensorStaticSpec logical;
        logical.logical_index = static_cast<int>(index);
        logical.backend_output_index = static_cast<int>(index);
        logical.physical_index = static_cast<int>(index);
        logical.output_slot = static_cast<int>(index);
        logical.tensor_index = static_cast<int>(index);
        logical.size_bytes = frontend_plan.values()[2U + index].required_bytes;
        logical.logical_name = frontend_plan.values()[2U + index].name;
        logical.backend_name = frontend_plan.values()[2U + index].name;
        logical.segment_name = frontend_plan.values()[2U + index].name;
        frontend_runtime.logical_outputs.push_back(std::move(logical));
      }
      simaai::neat::CompiledExposedView frontend_exposed;
      frontend_exposed.exposed_logical_outputs = frontend_runtime.logical_outputs;
      require(sc::apply_dmabuf_plan_processcvu_contract_projection(
                  frontend_plan, sc::ProcessCvuMlaBoundary::Inputs, &frontend_cvu,
                  &frontend_runtime, &frontend_exposed, &error),
              "pre-CVU frame-arena projection must pass: " + error);
      require(frontend_runtime.frame_arena_role == sima::FrameArenaRole::Allocate &&
                  frontend_runtime.frame_arena_storage_domain == sc::ArenaStorageDomain::Cma &&
                  frontend_runtime.physical_outputs[0].source_byte_offset ==
                      frontend_mla.physical_inputs[0].source_byte_offset &&
                  frontend_runtime.physical_outputs[1].source_byte_offset ==
                      frontend_mla.physical_inputs[1].source_byte_offset,
              "pre-CVU output regions must be the terminal MLA's exact imported IFM offsets");

      auto frontend_physical = sc::PhysicalExecutionLowerer::lower(frontend_plan, &error);
      require(frontend_physical.has_value(),
              "grouped command fixture must lower a physical plan: " + error);
      auto unfiltered_frontend_arena = sc::FrameSlotArenaPlan::compile(
          frontend_plan, *frontend_physical, sc::FrameSlotArenaReuse::DisjointLifetimes,
          sc::kLegacyEvoCmaRegionAlignmentBytes, &error, sc::kModalixProductionArenaDmsPolicy);
      require(unfiltered_frontend_arena.has_value(),
              "grouped command fixture must compile its unfiltered frame arena: " + error);
      const auto frontend_detached =
          sc::detached_mla_output_roots(frontend_plan, *frontend_physical);
      auto frontend_arena = sc::FrameSlotArenaPlan::compile(
          frontend_plan, *frontend_physical, sc::FrameSlotArenaReuse::DisjointLifetimes,
          sc::kLegacyEvoCmaRegionAlignmentBytes, &error, sc::kModalixProductionArenaDmsPolicy,
          frontend_detached);
      require(frontend_arena.has_value(),
              "grouped command fixture must compile one frame arena: " + error);
      const auto frontend_output_policy =
          sc::select_mla_output_carrier_policy(frontend_plan, *frontend_physical, 0U);
      require(frontend_output_policy == sc::MlaOutputCarrierPolicy::SeparateCpuVisible,
              "terminal MLA public output must select a distinct CPU-visible carrier");
      auto split_frontend_sources = sc::resolve_mla_input_physical_sources(
          frontend_plan, 0U, *frontend_arena, upstream, &error);
      require(split_frontend_sources.has_value(),
              "split terminal MLA arena IFMs must resolve: " + error);
      auto terminal_frontend_mla = make_projection(frontend_plan);
      require(sc::apply_dmabuf_plan_contract_projection(
                  frontend_plan, 0U, *frontend_arena, frontend_output_policy,
                  &terminal_frontend_mla, *split_frontend_sources, &error),
              "terminal MLA split-carrier projection must pass: " + error);
      require(frontend_arena->placement().domain == sc::ArenaStorageDomain::Cma &&
                  frontend_arena->placement().requires_access(sc::ArenaDeviceAccess::Ev74) &&
                  terminal_frontend_mla.frame_arena_role == sima::FrameArenaRole::Allocate &&
                  terminal_frontend_mla.frame_arena_storage_domain == sc::ArenaStorageDomain::Dms &&
                  terminal_frontend_mla.frame_arena_size_bytes == 4096U &&
                  terminal_frontend_mla.frame_arena_size_bytes <
                      frontend_arena->allocation_bytes() &&
                  terminal_frontend_mla.frame_arena_escape_policy ==
                      sc::ArenaEscapePolicy::CpuMappablePublic &&
                  (terminal_frontend_mla.frame_arena_required_device_access &
                   static_cast<std::uint32_t>(sc::ArenaDeviceAccess::Mla)) != 0U &&
                  (terminal_frontend_mla.frame_arena_required_device_access &
                   static_cast<std::uint32_t>(sc::ArenaDeviceAccess::CpuA65)) != 0U &&
                  (terminal_frontend_mla.frame_arena_required_device_access &
                   static_cast<std::uint32_t>(sc::ArenaDeviceAccess::Ev74)) == 0U,
              "EV ingress must keep its CMA arena while terminal MLA owns one right-sized "
              "DMS output carrier");
      require(terminal_frontend_mla.physical_inputs.size() == 2U &&
                  terminal_frontend_mla.physical_inputs[0].source_physical_index == 0 &&
                  terminal_frontend_mla.physical_inputs[1].source_physical_index == 0 &&
                  terminal_frontend_mla.physical_inputs[0].source_byte_offset !=
                      terminal_frontend_mla.physical_inputs[1].source_byte_offset &&
                  terminal_frontend_mla.physical_outputs.size() == 1U &&
                  terminal_frontend_mla.physical_outputs[0].source_byte_offset == 0,
              "split output ownership must keep exact upstream arena IFM offsets without "
              "copying or treating them as regions of the output pool");
      auto compact_frontend_cvu = frontend_cvu;
      auto compact_frontend_runtime = frontend_runtime;
      auto compact_frontend_exposed = frontend_exposed;
      require(sc::apply_dmabuf_plan_processcvu_contract_projection(
                  frontend_plan, 0U, *frontend_arena, sc::ProcessCvuMlaBoundary::Inputs,
                  &compact_frontend_cvu, &compact_frontend_runtime, &compact_frontend_exposed,
                  &error),
              "compact frontend CMA projection must pass: " + error);
      require(frontend_detached == std::vector<sc::ValueId>({4U}) &&
                  frontend_arena->region(4U) == nullptr &&
                  frontend_arena->allocation_bytes() <
                      unfiltered_frontend_arena->allocation_bytes() &&
                  compact_frontend_runtime.frame_arena_size_bytes ==
                      frontend_arena->allocation_bytes(),
              "YOLO-like EV ingress arena must exclude the detached terminal MLA OFM root");
      require(compact_frontend_runtime.physical_outputs.size() == 2U &&
                  compact_frontend_runtime.physical_outputs[0].source_byte_offset ==
                      terminal_frontend_mla.physical_inputs[0].source_byte_offset &&
                  compact_frontend_runtime.physical_outputs[1].source_byte_offset ==
                      terminal_frontend_mla.physical_inputs[1].source_byte_offset &&
                  compact_frontend_runtime.physical_outputs[0].source_byte_offset !=
                      compact_frontend_runtime.physical_outputs[1].source_byte_offset &&
                  (compact_frontend_runtime.physical_outputs[0].source_byte_offset != 0 ||
                   compact_frontend_runtime.physical_outputs[1].source_byte_offset != 0),
              "each zero-offset imported IFM selector must name an upstream physical binding "
              "that retains its exact nonzero absolute offset in the shared CMA parent");

      const auto view_ifm_plan = make_nonzero_view_to_terminal_mla_plan();
      auto view_ifm_physical = sc::PhysicalExecutionLowerer::lower(view_ifm_plan, &error);
      require(view_ifm_physical.has_value(), "nonzero view-to-MLA fixture must lower: " + error);
      const auto view_ifm_detached =
          sc::detached_mla_output_roots(view_ifm_plan, *view_ifm_physical);
      auto view_ifm_arena = sc::FrameSlotArenaPlan::compile(
          view_ifm_plan, *view_ifm_physical, sc::FrameSlotArenaReuse::DisjointLifetimes,
          sc::kLegacyEvoCmaRegionAlignmentBytes, &error, sc::kModalixProductionArenaDmsPolicy,
          view_ifm_detached);
      require(view_ifm_arena.has_value(), "nonzero view-to-MLA arena must compile: " + error);
      auto view_ifm_sources =
          sc::resolve_mla_input_physical_sources(view_ifm_plan, 0U, *view_ifm_arena, {}, &error);
      require(view_ifm_sources.has_value(), "nonzero view-to-MLA source must resolve: " + error);
      auto view_ifm_contract = make_projection(view_ifm_plan);
      require(sc::apply_dmabuf_plan_contract_projection(
                  view_ifm_plan, 0U, *view_ifm_arena,
                  sc::mla_output_carrier_policy_from_arena(view_ifm_plan, *view_ifm_arena, 0U),
                  &view_ifm_contract, *view_ifm_sources, &error),
              "nonzero view-to-MLA projection must pass: " + error);
      const auto* view_ifm_parent = view_ifm_arena->region(1U);
      const auto expected_view_ifm_offset =
          view_ifm_parent ? static_cast<std::int64_t>(view_ifm_parent->byte_offset + 4096U) : -1;
      view_ifm_contract.stage_id = "MLA_view_ifm";
      view_ifm_contract.node_name = "MLA_view_ifm";
      view_ifm_contract.model_path = "view-ifm.elf";
      view_ifm_contract.batch_size = 1;
      view_ifm_contract.batch_sz_model = 1;
      const auto compiled_view_ifm =
          sima::stagesemantics::build_mla_compiled_contract(view_ifm_contract);
      require(view_ifm_detached == std::vector<sc::ValueId>({3U}) && view_ifm_parent != nullptr &&
                  expected_view_ifm_offset > 0 && view_ifm_contract.physical_inputs.size() == 1U &&
                  view_ifm_contract.physical_inputs[0].source_physical_index == 0 &&
                  view_ifm_contract.physical_inputs[0].source_byte_offset ==
                      expected_view_ifm_offset &&
                  view_ifm_contract.input_bindings.size() == 1U &&
                  view_ifm_contract.input_bindings[0].src_physical_byte_offset ==
                      expected_view_ifm_offset &&
                  compiled_view_ifm.runtime_contract.logical_inputs.size() == 1U &&
                  compiled_view_ifm.runtime_contract.logical_inputs[0].byte_offset == 0,
              "an affine MLA IFM view must add its carrier-relative offset exactly once "
              "while its logical port remains local");

      sima::ProcessCvuStagePayload grouped_quant;
      grouped_quant.graph_family_enum = sima::ProcessCvuGraphFamily::Quant;
      grouped_quant.graph_family = "quantize";
      grouped_quant.input_tensors.resize(2U);
      grouped_quant.output_tensors.resize(2U);
      grouped_quant.runtime_output_physical_index_list.resize(2U, -1);
      simaai::neat::CompiledRuntimeContract grouped_runtime;
      grouped_runtime.logical_inputs.resize(2U);
      grouped_runtime.input_bindings.resize(2U);
      grouped_runtime.logical_outputs.resize(2U);
      grouped_runtime.physical_outputs.resize(1U);
      grouped_runtime.physical_outputs.front().device_kind = sima::DeviceKind::Evxx;
      simaai::neat::CompiledExposedView grouped_exposed;
      std::vector<sc::PhysicalCommandId> grouped_commands;
      std::vector<sc::PhysicalCommandId> non_cvu_commands;
      for (const auto& command : frontend_physical->commands) {
        if (command.engine == sc::PhysicalEngine::Cvu && command.graph_id == 222U) {
          grouped_commands.push_back(command.id);
        } else {
          non_cvu_commands.push_back(command.id);
        }
      }
      require(!grouped_commands.empty() && !non_cvu_commands.empty(),
              "grouped fixture must contain quantize and MLA physical commands");
      require(sc::apply_dmabuf_plan_processcvu_command_projection(
                  frontend_plan, *frontend_physical, grouped_commands, *frontend_arena,
                  &grouped_quant,
                  &grouped_runtime, &grouped_exposed, &error),
              "two sibling quantize operations must render as one exact cohort: " + error);
      const auto* grouped_region_0 = frontend_arena->region(2U);
      const auto* grouped_region_1 = frontend_arena->region(3U);
      require(grouped_quant.graph_id == 222 && grouped_quant.maximum_members == 32U &&
                  grouped_runtime.physical_inputs.size() == 2U &&
                  grouped_runtime.physical_outputs.size() == 2U && grouped_region_0 &&
                  grouped_region_1 &&
                  grouped_runtime.physical_inputs[0].physical_index == 0 &&
                  grouped_runtime.physical_inputs[1].physical_index == 1 &&
                  grouped_runtime.physical_inputs[0].source_physical_index == 0 &&
                  grouped_runtime.physical_inputs[1].source_physical_index == 1 &&
                  grouped_runtime.physical_inputs[0].size_bytes == 100U &&
                  grouped_runtime.physical_inputs[1].size_bytes == 200U &&
                  grouped_runtime.logical_inputs[0].physical_index == 0 &&
                  grouped_runtime.logical_inputs[1].physical_index == 1 &&
                  grouped_runtime.input_bindings[0].sink_pad_index == 0 &&
                  grouped_runtime.input_bindings[1].sink_pad_index == 1 &&
                  grouped_runtime.input_bindings[0].src_physical_output_index == 0 &&
                  grouped_runtime.input_bindings[1].src_physical_output_index == 1 &&
                  grouped_runtime.physical_outputs[0].source_byte_offset ==
                      static_cast<std::int64_t>(grouped_region_0->byte_offset) &&
                  grouped_runtime.physical_outputs[1].source_byte_offset ==
                      static_cast<std::int64_t>(grouped_region_1->byte_offset) &&
                  grouped_runtime.logical_outputs[0].byte_offset == 0 &&
                  grouped_runtime.logical_outputs[1].byte_offset == 0 &&
                  grouped_runtime.frame_arena_role == sima::FrameArenaRole::Allocate &&
                  grouped_runtime.consumer_keeps_distinct_physical_inputs,
              "grouped quantize must retain two public carrier pads and write two absolute arena "
              "regions");

      auto heterogeneous_payload = grouped_quant;
      auto heterogeneous_runtime = grouped_runtime;
      auto heterogeneous_exposed = grouped_exposed;
      const std::array<sc::PhysicalCommandId, 2U> heterogeneous_commands{
          grouped_commands.front(), non_cvu_commands.front()};
      error.clear();
      require(!sc::apply_dmabuf_plan_processcvu_command_projection(
                  frontend_plan, *frontend_physical, heterogeneous_commands, *frontend_arena,
                  &heterogeneous_payload, &heterogeneous_runtime,
                  &heterogeneous_exposed, &error),
              "a heterogeneous semantic frontier must fail closed");

      auto ambiguous = upstream;
      ambiguous.push_back(upstream[1]);
      ambiguous.back().logical_index = 10;
      require(!sc::resolve_mla_input_physical_sources(frontend_plan, ambiguous, &error),
              "ambiguous upstream IFM names must fail closed");

      auto missing = upstream;
      missing[0].backend_name.clear();
      require(!sc::resolve_mla_input_physical_sources(frontend_plan, missing, &error),
              "missing upstream IFM values must fail closed");

      const auto packed_plan = make_packed_plan();
      std::vector<sima::LogicalTensorStaticSpec> packed_upstream(2U);
      packed_upstream[0].logical_index = 11;
      packed_upstream[0].backend_name = "quantize_0";
      packed_upstream[0].physical_index = 0;
      packed_upstream[0].byte_offset = 0;
      packed_upstream[0].size_bytes = 640U;
      packed_upstream[1].logical_index = 12;
      packed_upstream[1].backend_name = "quantize_1";
      packed_upstream[1].physical_index = 0;
      packed_upstream[1].byte_offset = 640;
      packed_upstream[1].size_bytes = 320U;
      auto packed_sources =
          sc::resolve_mla_input_physical_sources(packed_plan, packed_upstream, &error);
      require(packed_sources.has_value(), "packed parent IFM must resolve: " + error);
      require(packed_sources->size() == 1U && (*packed_sources)[0].source_physical_index == 0,
              "packed IFM must resolve to its one physical parent");
      auto packed_contract = make_projection(packed_plan);
      require(sc::apply_dmabuf_plan_contract_projection(packed_plan, &packed_contract,
                                                        *packed_sources, &error),
              "packed 1/1 projection must pass: " + error);
      require(packed_contract.physical_outputs.size() == 1U &&
                  packed_contract.logical_outputs.size() == 2U,
              "packed OFM must stay one physical port with two logical read views");
      require(!packed_contract.logical_inputs[0].parent_carrier &&
                  packed_contract.physical_inputs[0].source_physical_index == 0,
              "packed IFM must not retain a logical parent-carrier anchor");
      require(packed_contract.logical_outputs[0].byte_offset == 0 &&
                  packed_contract.logical_outputs[0].stride_bytes ==
                      std::vector<std::int64_t>({200, 20, 20, 1}) &&
                  packed_contract.logical_outputs[1].byte_offset == 200,
              "unpack/slice must project only root-relative address expressions");

      const auto terminal_bf16_unpack_plan =
          make_terminal_bf16_over_int8_unpack_plan();
      auto terminal_bf16_unpack_physical =
          sc::PhysicalExecutionLowerer::lower(terminal_bf16_unpack_plan, &error);
      require(terminal_bf16_unpack_physical.has_value(),
              "terminal BF16-over-INT8 Unpack plan must lower: " + error);
      const auto terminal_bf16_unpack_detached = sc::detached_mla_output_roots(
          terminal_bf16_unpack_plan, *terminal_bf16_unpack_physical);
      auto terminal_bf16_unpack_arena = sc::FrameSlotArenaPlan::compile(
          terminal_bf16_unpack_plan, *terminal_bf16_unpack_physical,
          sc::FrameSlotArenaReuse::DisjointLifetimes,
          sc::kLegacyEvoCmaRegionAlignmentBytes, &error,
          sc::kModalixProductionArenaDmsPolicy,
          terminal_bf16_unpack_detached);
      require(terminal_bf16_unpack_arena.has_value() &&
                  sc::mla_output_carrier_policy_from_arena(
                      terminal_bf16_unpack_plan, *terminal_bf16_unpack_arena,
                      0U) == sc::MlaOutputCarrierPolicy::SeparateCpuVisible,
              "terminal packed MLA output must own one detached DMS carrier: " +
                  error);
      auto terminal_bf16_unpack_sources = sc::resolve_mla_input_physical_sources(
          terminal_bf16_unpack_plan, 0U, *terminal_bf16_unpack_arena, {},
          &error);
      require(terminal_bf16_unpack_sources.has_value(),
              "terminal packed MLA input must resolve from its retained CMA arena: " +
                  error);
      auto terminal_bf16_unpack_contract =
          make_projection(terminal_bf16_unpack_plan);
      require(sc::apply_dmabuf_plan_contract_projection(
                  terminal_bf16_unpack_plan, 0U,
                  *terminal_bf16_unpack_arena,
                  &terminal_bf16_unpack_contract,
                  *terminal_bf16_unpack_sources, &error),
              "terminal BF16-over-INT8 Unpack projection must pass: " + error);
      constexpr std::array<std::uint64_t, 6> kRow07Offsets = {
          0U, 819200U, 1024000U, 1075200U, 2099200U, 2355200U};
      constexpr std::array<std::uint64_t, 6> kRow07Spans = {
          819200U, 204800U, 51200U, 1024000U, 256000U, 64000U};
      require(terminal_bf16_unpack_contract.physical_outputs.size() == 1U &&
                  terminal_bf16_unpack_contract.physical_outputs[0].size_bytes ==
                      2419200U &&
                  terminal_bf16_unpack_contract.logical_outputs.size() ==
                      kRow07Spans.size() &&
                  terminal_bf16_unpack_contract.frame_arena_storage_domain ==
                      sc::ArenaStorageDomain::Dms,
              "Row07 route cut must retain one exact 2,419,200-byte DMS parent");
      for (std::size_t index = 0U; index < kRow07Spans.size(); ++index) {
        const auto& logical =
            terminal_bf16_unpack_contract.logical_outputs[index];
        require(logical.logical_index == static_cast<int>(index) &&
                    logical.physical_index == 0 &&
                    logical.byte_offset ==
                        static_cast<std::int64_t>(kRow07Offsets[index]) &&
                    logical.size_bytes == kRow07Spans[index] &&
                    logical.shape == std::vector<std::int64_t>(
                                         {1, static_cast<std::int64_t>(
                                                 kRow07Spans[index])}) &&
                    logical.stride_bytes ==
                        std::vector<std::int64_t>(
                            {static_cast<std::int64_t>(kRow07Spans[index]),
                             1}) &&
                    logical.dtype == "int8" && logical.layout.empty(),
                "Row07 producer catalogue must publish each exact raw Unpack "
                "transport view, not downstream BF16/HWC semantics");
      }
      require(kRow07Offsets.back() + kRow07Spans.back() == 2419200U,
              "Row07 ordered views must end exactly at the physical parent "
              "extent");

      const auto terminal_dense_bf16_plan = make_terminal_dense_bf16_plan();
      auto terminal_dense_bf16_physical =
          sc::PhysicalExecutionLowerer::lower(terminal_dense_bf16_plan, &error);
      require(terminal_dense_bf16_physical.has_value(),
              "terminal dense BF16 plan must lower: " + error);
      const auto terminal_dense_bf16_detached =
          sc::detached_mla_output_roots(terminal_dense_bf16_plan,
                                        *terminal_dense_bf16_physical);
      auto terminal_dense_bf16_arena = sc::FrameSlotArenaPlan::compile(
          terminal_dense_bf16_plan, *terminal_dense_bf16_physical,
          sc::FrameSlotArenaReuse::DisjointLifetimes,
          sc::kLegacyEvoCmaRegionAlignmentBytes, &error,
          sc::kModalixProductionArenaDmsPolicy,
          terminal_dense_bf16_detached);
      require(terminal_dense_bf16_arena.has_value(),
              "terminal dense BF16 arena must compile: " + error);
      auto terminal_dense_bf16_contract = make_projection(terminal_dense_bf16_plan);
      auto terminal_dense_bf16_sources = sc::resolve_mla_input_physical_sources(
          terminal_dense_bf16_plan, 0U, *terminal_dense_bf16_arena, {}, &error);
      require(terminal_dense_bf16_sources.has_value() &&
                  sc::apply_dmabuf_plan_contract_projection(
                      terminal_dense_bf16_plan, 0U,
                      *terminal_dense_bf16_arena,
                      &terminal_dense_bf16_contract,
                      *terminal_dense_bf16_sources, &error) &&
                  terminal_dense_bf16_contract.logical_outputs.size() == 1U &&
                  terminal_dense_bf16_contract.logical_outputs[0].dtype ==
                      "bfloat16" &&
                  terminal_dense_bf16_contract.logical_outputs[0].layout ==
                      "HWC",
              "a true dense BF16 MLA output must retain its own semantic "
              "dtype/layout when no Unpack carrier overrides it: " + error);

      const auto terminal_composed_slice =
          make_terminal_bf16_over_int8_unpack_plan(false, false, false, true);
      auto terminal_composed_slice_physical =
          sc::PhysicalExecutionLowerer::lower(terminal_composed_slice, &error);
      require(terminal_composed_slice_physical.has_value(),
              "terminal composed-Slice plan must lower: " + error);
      const auto terminal_composed_slice_detached =
          sc::detached_mla_output_roots(terminal_composed_slice,
                                        *terminal_composed_slice_physical);
      auto terminal_composed_slice_arena = sc::FrameSlotArenaPlan::compile(
          terminal_composed_slice, *terminal_composed_slice_physical,
          sc::FrameSlotArenaReuse::DisjointLifetimes,
          sc::kLegacyEvoCmaRegionAlignmentBytes, &error,
          sc::kModalixProductionArenaDmsPolicy,
          terminal_composed_slice_detached);
      require(terminal_composed_slice_arena.has_value(),
              "terminal composed-Slice arena must compile: " + error);
      auto terminal_composed_slice_contract =
          make_projection(terminal_composed_slice);
      auto terminal_composed_slice_sources =
          sc::resolve_mla_input_physical_sources(
              terminal_composed_slice, 0U, *terminal_composed_slice_arena,
              {}, &error);
      require(terminal_composed_slice_sources.has_value() &&
                  sc::apply_dmabuf_plan_contract_projection(
                      terminal_composed_slice, 0U,
                      *terminal_composed_slice_arena,
                      &terminal_composed_slice_contract,
                      *terminal_composed_slice_sources, &error) &&
                  terminal_composed_slice_contract.logical_outputs.size() ==
                      6U &&
                  terminal_composed_slice_contract.logical_outputs.back()
                          .byte_offset == 2355200 &&
                  terminal_composed_slice_contract.logical_outputs.back()
                          .size_bytes == 32000U &&
                  terminal_composed_slice_contract.logical_outputs.back()
                          .stride_bytes ==
                      std::vector<std::int64_t>({64000, 1}) &&
                  terminal_composed_slice_contract.logical_outputs.back().dtype ==
                      "int8" &&
                  terminal_composed_slice_contract.logical_outputs.back().layout
                      .empty(),
              "Unpack-to-Slice publication must trace the exact raw carrier "
              "dtype without importing removed Detess semantics: " + error);

      for (std::size_t malformed_index = 0U; malformed_index < 3U;
           ++malformed_index) {
        const auto malformed = make_terminal_bf16_over_int8_unpack_plan(
            malformed_index == 0U, malformed_index == 1U,
            malformed_index == 2U);
        auto malformed_physical =
            sc::PhysicalExecutionLowerer::lower(malformed, &error);
        require(malformed_physical.has_value(),
                "malformed publication fixture must remain physically "
                "lowerable: " + error);
        const auto malformed_detached =
            sc::detached_mla_output_roots(malformed, *malformed_physical);
        auto malformed_arena = sc::FrameSlotArenaPlan::compile(
            malformed, *malformed_physical,
            sc::FrameSlotArenaReuse::DisjointLifetimes,
            sc::kLegacyEvoCmaRegionAlignmentBytes, &error,
            sc::kModalixProductionArenaDmsPolicy, malformed_detached);
        require(malformed_arena.has_value(),
                "malformed publication fixture must reach projection: " +
                    error);
        auto malformed_sources = sc::resolve_mla_input_physical_sources(
            malformed, 0U, *malformed_arena, {}, &error);
        require(malformed_sources.has_value(),
                "malformed publication fixture input must resolve: " + error);
        auto malformed_contract = make_projection(malformed);
        error.clear();
        require(!sc::apply_dmabuf_plan_contract_projection(
                    malformed, 0U, *malformed_arena, &malformed_contract,
                    *malformed_sources, &error) &&
                    (error.find("normalized storage") != std::string::npos ||
                     error.find("exceeds its physical output port") !=
                         std::string::npos),
                "malformed carrier span/stride/overrun must fail closed at "
                "projection: " + error);
      }

      constexpr std::size_t kGroupedUnpackMembers = 6U;
      const auto bf16_unpack_plan =
          make_bf16_consumer_over_int8_unpack_plan(kGroupedUnpackMembers);
      auto bf16_unpack_contract = make_projection(bf16_unpack_plan);
      auto bf16_unpack_sources =
          sc::resolve_mla_input_physical_sources(bf16_unpack_plan, {}, &error);
      require(bf16_unpack_sources.has_value(),
              "BF16-over-INT8 Unpack MLA input must resolve: " + error);
      require(sc::apply_dmabuf_plan_contract_projection(
                  bf16_unpack_plan, &bf16_unpack_contract, *bf16_unpack_sources, &error),
              "BF16-over-INT8 Unpack MLA projection must pass: " + error);
      require(bf16_unpack_contract.logical_outputs.size() == kGroupedUnpackMembers &&
                  std::all_of(bf16_unpack_contract.logical_outputs.begin(),
                              bf16_unpack_contract.logical_outputs.end(),
                              [](const auto& output) {
                                return output.dtype == "int8" &&
                                       output.stride_bytes ==
                                           std::vector<std::int64_t>({200, 1});
                              }),
              "MLA must publish exact Unpack carrier units, not downstream BF16 semantics");

      auto bf16_unpack_physical =
          sc::PhysicalExecutionLowerer::lower(bf16_unpack_plan, &error);
      require(bf16_unpack_physical.has_value(),
              "BF16-over-INT8 Unpack physical plan must lower: " + error);
      std::vector<sc::PhysicalCommandId> detesscast_commands;
      for (const auto& command : bf16_unpack_physical->commands) {
        if (command.engine == sc::PhysicalEngine::Cvu && command.graph_id == 225U) {
          detesscast_commands.push_back(command.id);
        }
      }
      require(!detesscast_commands.empty(),
              "BF16 consumers must lower to the registered graph225 cohort");
      auto bf16_unpack_arena = sc::FrameSlotArenaPlan::compile(
          bf16_unpack_plan, *bf16_unpack_physical,
          sc::FrameSlotArenaReuse::DisjointLifetimes,
          sc::kLegacyEvoCmaRegionAlignmentBytes, &error);
      require(bf16_unpack_arena.has_value(),
              "BF16-over-INT8 Unpack frame arena must compile: " + error);
      auto detesscast_contract = sc::build_dmabuf_plan_processcvu_command_contract(
          bf16_unpack_plan, *bf16_unpack_physical, detesscast_commands,
          *bf16_unpack_arena, &error);
      require(detesscast_contract.has_value(),
              "BF16 graph225 command contract must build: " + error);
      require(detesscast_contract->payload.input_tensors.size() == kGroupedUnpackMembers &&
                  std::all_of(detesscast_contract->payload.input_tensors.begin(),
                              detesscast_contract->payload.input_tensors.end(),
                              [](const auto& input) {
                                return input.dtype == SIMA_EV_DTYPE_BF16;
                              }),
              "graph225 must retain BF16 consumer descriptors independently of MLA carrier "
              "publication dtype");
      require(detesscast_contract->runtime_contract.input_bindings.size() ==
                  kGroupedUnpackMembers,
              "six graph225 members must publish six exact input selectors");
      for (std::size_t member = 0; member < kGroupedUnpackMembers; ++member) {
        const auto& route = detesscast_contract->runtime_contract.input_bindings[member];
        require(route.sink_pad_index == 0 &&
                    route.local_logical_input_index == static_cast<int>(member) &&
                    route.src_logical_output_index == static_cast<int>(member) &&
                    route.src_output_slot == static_cast<int>(member) &&
                    route.src_physical_output_index == static_cast<int>(member),
                "one internal MLA arena pad must retain six distinct source selectors");
      }

      sima::ProcessCvuStagePayload packed_pre;
      packed_pre.graph_family_enum = sima::ProcessCvuGraphFamily::Quant;
      simaai::neat::CompiledRuntimeContract packed_pre_runtime;
      sima::PhysicalBufferStaticSpec packed_pre_parent;
      packed_pre_parent.physical_index = 0;
      packed_pre_parent.size_bytes = 960U;
      packed_pre_parent.segment_name = "output_tensor";
      packed_pre_runtime.physical_outputs.push_back(std::move(packed_pre_parent));
      for (std::size_t index = 0; index < 2U; ++index) {
        sima::LogicalTensorStaticSpec logical;
        logical.logical_index = static_cast<int>(index);
        logical.backend_output_index = static_cast<int>(index);
        logical.physical_index = 0;
        logical.output_slot = static_cast<int>(index);
        logical.tensor_index = static_cast<int>(index);
        logical.byte_offset = index == 0U ? 0 : 640;
        logical.size_bytes = index == 0U ? 640U : 320U;
        logical.logical_name = "quantize_" + std::to_string(index);
        logical.backend_name = logical.logical_name;
        logical.segment_name = "output_tensor";
        packed_pre_runtime.logical_outputs.push_back(std::move(logical));
      }
      simaai::neat::CompiledExposedView packed_pre_exposed;
      packed_pre_exposed.exposed_logical_outputs = packed_pre_runtime.logical_outputs;
      require(sc::apply_dmabuf_plan_processcvu_contract_projection(
                  packed_plan, sc::ProcessCvuMlaBoundary::Inputs, &packed_pre, &packed_pre_runtime,
                  &packed_pre_exposed, &error),
              "packed pre-CVU placement must project: " + error);
      require(packed_pre_runtime.physical_outputs.size() == 2U &&
                  packed_pre_runtime.physical_outputs[1].source_byte_offset -
                          packed_pre_runtime.physical_outputs[0].source_byte_offset ==
                      640 &&
                  packed_pre_runtime.physical_outputs[0].required_alignment_bytes == 16U &&
                  !packed_pre_runtime.consumer_keeps_distinct_physical_inputs,
              "Pack children must be direct writes into one ordered parent carrier");

      const auto single_read_plan = make_single_read_plan();
      auto single_read_contract = make_projection(single_read_plan);
      auto single_read_sources =
          sc::resolve_mla_input_physical_sources(single_read_plan, {}, &error);
      require(single_read_sources.has_value(),
              "single-read model-input carrier must resolve: " + error);
      require(sc::apply_dmabuf_plan_contract_projection(single_read_plan, &single_read_contract,
                                                        *single_read_sources, &error),
              "single read-view projection must pass: " + error);
      require(single_read_contract.logical_outputs.size() == 1U &&
                  single_read_contract.logical_outputs[0].byte_offset == 8 &&
                  single_read_contract.logical_outputs[0].size_bytes == 12U &&
                  single_read_contract.logical_outputs[0].stride_bytes ==
                      std::vector<std::int64_t>({192, 192, 16, 1}),
              "one Slice-derived OFM must retain its exact affine read expression");
      auto single_read_physical =
          sc::PhysicalExecutionLowerer::lower(single_read_plan, &error);
      require(single_read_physical.has_value(),
              "MLA-to-EV negative fixture must lower: " + error);
      const auto single_read_detached =
          sc::detached_mla_output_roots(single_read_plan, *single_read_physical);
      auto single_read_arena = sc::FrameSlotArenaPlan::compile(
          single_read_plan, *single_read_physical,
          sc::FrameSlotArenaReuse::DisjointLifetimes,
          sc::kLegacyEvoCmaRegionAlignmentBytes, &error,
          sc::kModalixProductionArenaDmsPolicy, single_read_detached);
      require(sc::select_mla_output_carrier_policy(
                  single_read_plan, *single_read_physical, 0U) ==
                  sc::MlaOutputCarrierPolicy::SharedFrameArena &&
                  single_read_detached.empty() && single_read_arena.has_value() &&
                  single_read_arena->region(1U) != nullptr &&
                  single_read_arena->region(3U) != nullptr &&
                  single_read_arena->placement().domain == sc::ArenaStorageDomain::Cma,
              "an EV command after MLA must keep its producer and consumer in CMA rather than "
              "detaching a CPU-only output carrier");

      const auto terminal_slice_plan = make_terminal_two_slice_plan();
      auto terminal_slice_physical =
          sc::PhysicalExecutionLowerer::lower(terminal_slice_plan, &error);
      require(terminal_slice_physical.has_value(),
              "pure terminal two-Slice fixture must lower: " + error);
      const auto terminal_slice_policy = sc::select_mla_output_carrier_policy(
          terminal_slice_plan, *terminal_slice_physical, 0U);
      const auto terminal_slice_detached = sc::detached_mla_output_roots(
          terminal_slice_plan, *terminal_slice_physical);
      auto terminal_slice_arena = sc::FrameSlotArenaPlan::compile(
          terminal_slice_plan, *terminal_slice_physical,
          sc::FrameSlotArenaReuse::DisjointLifetimes,
          sc::kLegacyEvoCmaRegionAlignmentBytes, &error,
          sc::kModalixProductionArenaDmsPolicy, terminal_slice_detached);
      require(terminal_slice_policy ==
                  sc::MlaOutputCarrierPolicy::SeparateCpuVisible &&
                  terminal_slice_detached == std::vector<sc::ValueId>({2U}) &&
                  terminal_slice_arena.has_value() &&
                  terminal_slice_arena->region(1U) != nullptr &&
                  terminal_slice_arena->region(2U) == nullptr,
              "pure terminal read views must detach their one physical MLA root");
      auto terminal_slice_sources = sc::resolve_mla_input_physical_sources(
          terminal_slice_plan, 0U, *terminal_slice_arena, {}, &error);
      require(terminal_slice_sources.has_value(),
              "terminal two-Slice arena IFM must resolve: " + error);
      auto terminal_slice_contract = make_projection(terminal_slice_plan);
      require(sc::apply_dmabuf_plan_contract_projection(
                  terminal_slice_plan, 0U, *terminal_slice_arena,
                  terminal_slice_policy, &terminal_slice_contract,
                  *terminal_slice_sources, &error),
              "pure terminal read-view projection must pass: " + error);
      require(terminal_slice_contract.physical_outputs.size() == 1U &&
                  terminal_slice_contract.logical_outputs.size() == 2U &&
                  terminal_slice_contract.logical_outputs[0].logical_name == "left_head" &&
                  terminal_slice_contract.logical_outputs[0].physical_index == 0 &&
                  terminal_slice_contract.logical_outputs[0].byte_offset == 8 &&
                  terminal_slice_contract.logical_outputs[0].shape ==
                      sc::TensorShape({1, 2, 2, 4}) &&
                  terminal_slice_contract.logical_outputs[0].stride_bytes ==
                      std::vector<std::int64_t>({32, 16, 4, 1}) &&
                  terminal_slice_contract.logical_outputs[1].logical_name == "right_head" &&
                  terminal_slice_contract.logical_outputs[1].physical_index == 0 &&
                  terminal_slice_contract.logical_outputs[1].byte_offset == 32 &&
                  terminal_slice_contract.logical_outputs[1].stride_bytes ==
                      std::vector<std::int64_t>({32, 16, 4, 1}),
              "separate terminal publication must retain exact public arity, order, names, "
              "root-relative offsets, shapes, and strides");

      const auto mla_a65_plan = make_mla_a65_successor_plan();
      auto mla_a65_physical = sc::PhysicalExecutionLowerer::lower(mla_a65_plan, &error);
      require(mla_a65_physical.has_value(),
              "MLA-to-A65 fixture must lower: " + error);
      const auto mla_a65_detached =
          sc::detached_mla_output_roots(mla_a65_plan, *mla_a65_physical);
      auto mla_a65_arena = sc::FrameSlotArenaPlan::compile(
          mla_a65_plan, *mla_a65_physical,
          sc::FrameSlotArenaReuse::DisjointLifetimes,
          sc::kLegacyEvoCmaRegionAlignmentBytes, &error,
          sc::kModalixProductionArenaDmsPolicy, mla_a65_detached);
      require(sc::select_mla_output_carrier_policy(
                  mla_a65_plan, *mla_a65_physical, 0U) ==
                  sc::MlaOutputCarrierPolicy::SharedFrameArena &&
                  mla_a65_detached.empty() && mla_a65_arena.has_value() &&
                  mla_a65_arena->region(1U) != nullptr &&
                  mla_a65_arena->region(2U) != nullptr &&
                  mla_a65_arena->placement().domain == sc::ArenaStorageDomain::Dms &&
                  mla_a65_arena->placement().requires_access(sc::ArenaDeviceAccess::Mla) &&
                  mla_a65_arena->placement().requires_access(sc::ArenaDeviceAccess::CpuA65) &&
                  !mla_a65_arena->placement().requires_access(sc::ArenaDeviceAccess::Ev74),
              "an exact A65 successor must keep the MLA output in the shared DMS arena; only "
              "a public terminal boundary may detach it");

      auto gapped = packed_upstream;
      ++gapped[1].byte_offset;
      require(!sc::resolve_mla_input_physical_sources(packed_plan, gapped, &error),
              "packed IFM with a gap must fail closed");

      const auto incomplete_parent_plan = make_packed_plan(1920U);
      require(
          !sc::resolve_mla_input_physical_sources(incomplete_parent_plan, packed_upstream, &error),
          "legacy doubled-size BF16 parent without exact producer coverage must fail closed");

      const auto pitched_plan = make_pitched_batch_pack_plan();
      std::vector<sima::LogicalTensorStaticSpec> pitched_upstream(2U);
      pitched_upstream[0].logical_index = 0;
      pitched_upstream[0].backend_name = "left";
      pitched_upstream[0].physical_index = 0;
      pitched_upstream[0].byte_offset = 0;
      pitched_upstream[0].size_bytes = 8U;
      pitched_upstream[0].stride_bytes = {8, 4};
      pitched_upstream[1].logical_index = 1;
      pitched_upstream[1].backend_name = "right";
      pitched_upstream[1].physical_index = 0;
      pitched_upstream[1].byte_offset = 4;
      pitched_upstream[1].size_bytes = 8U;
      pitched_upstream[1].stride_bytes = {8, 4};
      auto pitched_sources =
          sc::resolve_mla_input_physical_sources(pitched_plan, pitched_upstream, &error);
      require(pitched_sources && pitched_sources->size() == 1U &&
                  pitched_sources->front().source_physical_index == 0,
              "batch-two pitched Pack must resolve to its shared parent: " + error);
      const auto arbitrary_batch_plan = make_pitched_batch_pack_plan(3U);
      auto arbitrary_batch_upstream = pitched_upstream;
      arbitrary_batch_upstream[0].size_bytes = 12U;
      arbitrary_batch_upstream[1].size_bytes = 12U;
      auto arbitrary_batch_sources = sc::resolve_mla_input_physical_sources(
          arbitrary_batch_plan, arbitrary_batch_upstream, &error);
      require(arbitrary_batch_sources && arbitrary_batch_sources->size() == 1U &&
                  arbitrary_batch_sources->front().source_physical_index == 0,
              "non-power-of-two static-batch Pack must resolve generically: " + error);
      auto wrong_pitch = pitched_upstream;
      wrong_pitch[1].stride_bytes = {4, 4};
      require(!sc::resolve_mla_input_physical_sources(pitched_plan, wrong_pitch, &error),
              "a published Pack child with the wrong pitch must fail closed");

      sima::ProcessCvuStagePayload dequant;
      dequant.graph_family_enum = sima::ProcessCvuGraphFamily::Dequant;
      dequant.graph_id = 6;
      simaai::neat::CompiledRuntimeContract post_runtime;
      sima::PhysicalBufferStaticSpec parent_output;
      parent_output.physical_index = 0;
      parent_output.allocator_index = 0;
      parent_output.size_bytes = 28U * 4096U;
      parent_output.device_kind = sima::DeviceKind::Evxx;
      parent_output.segment_name = "output_tensor";
      post_runtime.physical_outputs.push_back(std::move(parent_output));
      for (std::size_t index = 0; index < 28U; ++index) {
        sima::LogicalTensorStaticSpec logical;
        logical.logical_index = static_cast<int>(index);
        logical.backend_output_index = static_cast<int>(index);
        logical.physical_index = 0;
        logical.output_slot = static_cast<int>(index);
        logical.tensor_index = static_cast<int>(index);
        logical.size_bytes = 4096U;
        logical.logical_name = "post_" + std::to_string(index);
        logical.backend_name = logical.logical_name;
        logical.segment_name = "output_tensor";
        post_runtime.logical_outputs.push_back(std::move(logical));
      }
      simaai::neat::CompiledExposedView post_exposed;
      post_exposed.exposed_logical_outputs = post_runtime.logical_outputs;
      require(sc::apply_dmabuf_plan_processcvu_contract_projection(
                  plan, sc::ProcessCvuMlaBoundary::Outputs, &dequant, &post_runtime, &post_exposed,
                  &error),
              "dequant must project onto the production driver graph: " + error);
      require(dequant.dmabuf_plan_contract && dequant.graph_id == 223,
              "dequant must use canonical graph 223 in dmabuf-plan mode");
      require(post_runtime.physical_outputs.size() == 28U &&
                  post_runtime.logical_outputs[7].physical_index == 7 &&
                  post_runtime.physical_outputs[7].required_alignment_bytes == 8192U,
              "post-CVU output region must inherit the exact ordered MLA OFM alignment");
      require(post_runtime.frame_arena_role == sima::FrameArenaRole::ReuseInput &&
                  post_runtime.frame_arena_size_bytes == contract.frame_arena_size_bytes,
              "post-CVU must reuse the same parent frame arena");
      require(post_runtime.frame_arena_storage_domain == sc::ArenaStorageDomain::Cma &&
                  post_runtime.frame_arena_provenance ==
                      sc::ArenaAllocationProvenance::CoreAllocated &&
                  (post_runtime.frame_arena_required_device_access &
                   static_cast<std::uint32_t>(sc::ArenaDeviceAccess::Ev74)) != 0U,
              "post-CVU must retain the Core-authored CMA/domain/provenance access join");

      const std::vector<std::pair<sima::ProcessCvuGraphFamily, int>> driver_families = {
          {sima::ProcessCvuGraphFamily::Tess, 2},
          {sima::ProcessCvuGraphFamily::Detess, 3},
      };
      const std::vector<std::string> canonical_tokens = {
          "tessellate", "detessellate",
      };
      std::size_t family_index = 0U;
      for (const auto& [family, graph_id] : driver_families) {
        sima::ProcessCvuStagePayload projected;
        projected.graph_family_enum = family;
        auto family_runtime = post_runtime;
        auto family_exposed = post_exposed;
        require(sc::apply_dmabuf_plan_processcvu_contract_projection(
                    plan, sc::ProcessCvuMlaBoundary::Outputs, &projected, &family_runtime,
                    &family_exposed, &error),
                "tess/detess family must project onto a production driver graph: " + error);
        require(projected.dmabuf_plan_contract && projected.graph_id == graph_id &&
                    projected.graph_name == canonical_tokens[family_index++] &&
                    projected.descriptor_abi_id ==
                        SIMA_PLUGIN_CVU_DESCRIPTOR_ABI_TENSOR_TRANSFORM_PAIR_V1 &&
                    projected.descriptor_contract_version == 1U &&
                    projected.binding_schema_version == 1U &&
                    projected.supported_placement_mask ==
                        (SIMA_PLUGIN_CVU_PLACEMENT_EV74 | SIMA_PLUGIN_CVU_PLACEMENT_A65) &&
                    projected.allowed_frame_patch_mask == SIMA_PLUGIN_CVU_FRAME_PATCH_METADATA,
                "tess/detess family must publish its exact /dev/cvu registry handshake");
      }

      for (const auto retired : {sima::ProcessCvuGraphFamily::QuantTess,
                                 sima::ProcessCvuGraphFamily::CastTess,
                                 sima::ProcessCvuGraphFamily::DetessCast,
                                 sima::ProcessCvuGraphFamily::DetessDequant}) {
        sima::ProcessCvuStagePayload projected;
        projected.graph_family_enum = retired;
        auto retired_runtime = post_runtime;
        auto retired_exposed = post_exposed;
        error.clear();
        require(!sc::apply_dmabuf_plan_processcvu_contract_projection(
                    plan, sc::ProcessCvuMlaBoundary::Outputs, &projected, &retired_runtime,
                    &retired_exposed, &error) &&
                    error.find("does not support") != std::string::npos,
                "retired fused ProcessCVU families must fail closed");
      }

      sima::ProcessCvuStagePayload preproc;
      preproc.graph_family_enum = sima::ProcessCvuGraphFamily::Preproc;
      auto preproc_runtime = packed_pre_runtime;
      auto preproc_exposed = packed_pre_exposed;
      require(sc::apply_dmabuf_plan_processcvu_contract_projection(
                  packed_plan, sc::ProcessCvuMlaBoundary::Inputs, &preproc, &preproc_runtime,
                  &preproc_exposed, &error),
              "preproc must project through its strict graph-200 descriptor ABI: " + error);
      require(preproc.dmabuf_plan_contract && preproc.graph_id == 200 &&
                  preproc.graph_name == "preproc" &&
                  preproc.descriptor_abi_id == SIMA_PLUGIN_CVU_DESCRIPTOR_ABI_PREPROC_V1 &&
                  preproc.descriptor_contract_version == 1U &&
                  preproc.binding_schema_version == 1U && preproc.maximum_members == 1U &&
                  preproc.supported_placement_mask == SIMA_PLUGIN_CVU_PLACEMENT_EV74 &&
                  preproc.allowed_frame_patch_mask ==
                      (SIMA_PLUGIN_CVU_FRAME_PATCH_METADATA |
                       SIMA_PLUGIN_CVU_FRAME_PATCH_PREPROC_GEOMETRY |
                       SIMA_PLUGIN_CVU_FRAME_PATCH_PREPROC_SCALAR_ROI |
                       SIMA_PLUGIN_CVU_FRAME_PATCH_PREPROC_ROI_LIST |
                       SIMA_PLUGIN_CVU_FRAME_PATCH_PREPROC_PLANE_LAYOUT),
              "preproc must publish the exact bounded graph-200 registry handshake");

      {
        const auto absorption_plan = make_model_managed_preproc_absorption_plan();
        auto absorption_physical = sc::PhysicalExecutionLowerer::lower(absorption_plan, &error);
        require(absorption_physical.has_value(),
                "graph-200 absorption physical lowering must pass: " + error);
        const auto absorption_detached =
            sc::detached_mla_output_roots(absorption_plan, *absorption_physical);
        auto absorption_arena = sc::FrameSlotArenaPlan::compile(
            absorption_plan, *absorption_physical, sc::FrameSlotArenaReuse::DisjointLifetimes,
            sc::kLegacyEvoCmaRegionAlignmentBytes, &error, sc::kModalixProductionArenaDmsPolicy,
            absorption_detached);
        require(absorption_arena.has_value(), "graph-200 absorption arena must compile: " + error);
        require(absorption_detached == std::vector<sc::ValueId>({2U}) &&
                    absorption_arena->region(1U) != nullptr &&
                    absorption_arena->region(2U) == nullptr &&
                    absorption_arena->placement().domain == sc::ArenaStorageDomain::Cma &&
                    absorption_arena->placement().requires_access(sc::ArenaDeviceAccess::Ev74) &&
                    absorption_arena->placement().requires_access(sc::ArenaDeviceAccess::Mla) &&
                    absorption_arena->placement().escape == sc::ArenaEscapePolicy::InternalOnly,
                "graph-200 must retain only its EV74/MLA CMA ingress carrier");

        simaai::neat::PreprocOptions graph200;
        graph200.model_managed_contract = true;
        graph200.input_shape = {2, 2, 3};
        graph200.output_shape = {2, 2, 3};
        graph200.scaled_height = 2;
        graph200.scaled_width = 2;
        graph200.input_img_type = "NV12";
        graph200.output_img_type = "RGB";
        graph200.output_dtype = "INT8";
        graph200.tessellate = false;
        graph200.q_scale = 0.25;
        graph200.q_zp = -128;
        auto graph200_contract =
            sima::stagesemantics::build_processcvu_compiled_contract_from_options(graph200);
        std::vector<sc::PhysicalCommandId> absorbed_commands;
        require(sc::project_model_managed_preproc_contract(absorption_plan, *absorption_physical,
                                                           *absorption_arena, &graph200_contract,
                                                           &absorbed_commands, &error),
                "graph-200 must replace the exact leading graph222 command: " + error);
        require(
            absorbed_commands.size() == 1U &&
                absorption_physical->commands[absorbed_commands.front()].graph_id == 222U &&
                graph200_contract.payload.graph_id == 200 &&
                graph200_contract.payload.dmabuf_plan_contract &&
                graph200_contract.payload.input_tensors.size() == 1U &&
                graph200_contract.payload.output_tensors.size() == 1U &&
                graph200_contract.payload.input_tensors.front().shape.rank == 3U &&
                graph200_contract.payload.input_tensors.front().shape.sizes[0] == 2 &&
                graph200_contract.payload.input_tensors.front().shape.sizes[1] == 2 &&
                graph200_contract.payload.input_tensors.front().shape.sizes[2] == 3 &&
                graph200_contract.payload.input_tensors.front().layout_kind ==
                    SIMA_EV_LAYOUT_STRIDED &&
                graph200_contract.payload.input_tensors.front().layout.strided.strides_bytes[0] ==
                    6 &&
                graph200_contract.payload.input_tensors.front().layout.strided.strides_bytes[1] ==
                    3 &&
                graph200_contract.payload.input_tensors.front().layout.strided.strides_bytes[2] ==
                    1 &&
                graph200_contract.payload.default_output_names ==
                    std::vector<std::string>{"output_rgb_image"} &&
                graph200_contract.payload.runtime_output_logical_index_list ==
                    std::vector<int>{0} &&
                graph200_contract.runtime_contract.frame_arena_role ==
                    sima::FrameArenaRole::Allocate &&
                graph200_contract.runtime_contract.input_bindings.front()
                        .src_logical_output_index == 0 &&
                graph200_contract.runtime_contract.input_bindings.front().src_output_slot == 0 &&
                graph200_contract.runtime_contract.input_bindings.front()
                        .src_physical_output_index == 0 &&
                graph200_contract.runtime_contract.logical_inputs.front().dtype == "UINT8" &&
                graph200_contract.runtime_contract.logical_inputs.front().layout == "HW" &&
                graph200_contract.runtime_contract.logical_inputs.front().shape ==
                    std::vector<std::int64_t>({3, 2}) &&
                graph200_contract.runtime_contract.logical_inputs.front().size_bytes == 6U &&
                graph200_contract.runtime_contract.logical_outputs.front().shape ==
                    sc::TensorShape({1, 2, 2, 3}) &&
                graph200_contract.runtime_contract.logical_outputs.front().byte_offset == 0 &&
                graph200_contract.runtime_contract.logical_outputs.front().segment_name ==
                    "quantize_0" &&
                graph200_contract.runtime_contract.physical_outputs.front().source_byte_offset ==
                    static_cast<std::int64_t>(absorption_arena->region(1U)->byte_offset),
            "graph-200 must write the original quant result directly into the one arena");

        auto graph200_mla_sources = sc::resolve_mla_input_physical_sources(
            absorption_plan, 0U, *absorption_arena, {}, &error);
        require(graph200_mla_sources.has_value(),
                "graph-200 first-MLA source must resolve from the compact arena: " + error);
        auto graph200_mla = make_projection(absorption_plan);
        require(
            sc::apply_dmabuf_plan_contract_projection(
                absorption_plan, 0U, *absorption_arena,
                sc::mla_output_carrier_policy_from_arena(absorption_plan, *absorption_arena, 0U),
                &graph200_mla, *graph200_mla_sources, &error),
            "graph-200 first-MLA projection must pass: " + error);
        require(graph200_mla.physical_inputs.size() == 1U &&
                    graph200_mla.physical_inputs[0].source_physical_index == 0 &&
                    graph200_mla.physical_inputs[0].source_byte_offset ==
                        graph200_contract.runtime_contract.physical_outputs[0].source_byte_offset &&
                    graph200_mla.input_bindings.size() == 1U &&
                    graph200_mla.input_bindings[0].src_physical_output_index == 0 &&
                    graph200_mla.input_bindings[0].src_physical_byte_offset ==
                        graph200_contract.runtime_contract.physical_outputs[0].source_byte_offset,
                "graph-200 publication and first-MLA IFM must name one exact CMA region");

        const auto tess_absorption_plan = make_model_managed_tess_preproc_absorption_plan();
        auto tess_absorption_physical =
            sc::PhysicalExecutionLowerer::lower(tess_absorption_plan, &error);
        require(tess_absorption_physical.has_value(),
                "tessellated graph-200 absorption physical lowering must pass: " + error);
        const auto tess_absorption_detached =
            sc::detached_mla_output_roots(tess_absorption_plan, *tess_absorption_physical);
        auto tess_absorption_arena = sc::FrameSlotArenaPlan::compile(
            tess_absorption_plan, *tess_absorption_physical,
            sc::FrameSlotArenaReuse::DisjointLifetimes, sc::kLegacyEvoCmaRegionAlignmentBytes,
            &error, sc::kModalixProductionArenaDmsPolicy, tess_absorption_detached);
        require(tess_absorption_arena.has_value(),
                "tessellated graph-200 absorption arena must compile: " + error);
        require(
            tess_absorption_detached == std::vector<sc::ValueId>({3U}) &&
                tess_absorption_arena->region(2U) != nullptr &&
                tess_absorption_arena->region(3U) == nullptr &&
                tess_absorption_arena->placement().domain == sc::ArenaStorageDomain::Cma &&
                tess_absorption_arena->placement().requires_access(sc::ArenaDeviceAccess::Ev74) &&
                tess_absorption_arena->placement().requires_access(sc::ArenaDeviceAccess::Mla),
            "tessellated graph-200 must retain only its EV74/MLA CMA ingress carrier");
        auto tess_graph200 = graph200;
        tess_graph200.tessellate = true;
        tess_graph200.slice_shape = {2, 2, 3};
        auto tess_graph200_contract = sima::stagesemantics::
            build_processcvu_compiled_contract_from_options(tess_graph200);
        std::vector<sc::PhysicalCommandId> tess_absorbed_commands;
        error.clear();
        require(sc::project_model_managed_preproc_contract(
                    tess_absorption_plan, *tess_absorption_physical,
                    *tess_absorption_arena, &tess_graph200_contract,
                    &tess_absorbed_commands, &error),
                "tessellated graph-200 must project its selected output onto the exact "
                "first-MLA value: " + error);
        require(tess_absorbed_commands.size() == 1U &&
                    tess_absorption_physical
                            ->commands[tess_absorbed_commands.front()]
                            .graph_id == 226U &&
                    tess_graph200_contract.payload.default_output_names ==
                        std::vector<std::string>{"output_tessellated_image"} &&
                    tess_graph200_contract.runtime_contract.logical_outputs.size() == 1U &&
                    tess_graph200_contract.runtime_contract.physical_outputs.size() == 1U &&
                    tess_graph200_contract.runtime_contract.logical_outputs.front()
                            .logical_index == 0 &&
                    tess_graph200_contract.runtime_contract.logical_outputs.front()
                            .output_slot == 0 &&
                    tess_graph200_contract.runtime_contract.logical_outputs.front()
                            .physical_index == 0 &&
                    tess_graph200_contract.runtime_contract.logical_outputs.front()
                            .logical_name == "output_tessellated_image" &&
                    tess_graph200_contract.runtime_contract.logical_outputs.front()
                            .segment_name == "tessellated_quantize_0" &&
                    tess_graph200_contract.runtime_contract.physical_outputs.front()
                            .source_byte_offset ==
                        static_cast<std::int64_t>(
                            tess_absorption_arena->region(2U)->byte_offset),
                "tessellated graph-200 single handoff must discard the unselected RGB "
                "firmware pointer and retain the compiler-authored MLA ingress");

        auto wrong_q = graph200_contract;
        wrong_q.payload.q_scale = 0.5;
        absorbed_commands.clear();
        error.clear();
        require(!sc::project_model_managed_preproc_contract(
                    absorption_plan, *absorption_physical, *absorption_arena, &wrong_q,
                    &absorbed_commands, &error) &&
                    error.find("quantization contradicts") != std::string::npos,
                "graph-200 absorption must reject mismatched compiler quantization");
      }

      const auto two_mla = make_two_mla_plan();
      std::string arena_error;
      const auto two_mla_physical =
          sc::PhysicalExecutionLowerer::lower(two_mla, &arena_error);
      require(two_mla_physical.has_value(),
              "two MLA stages need an exact physical plan: " + arena_error);
      const auto two_mla_detached =
          sc::detached_mla_output_roots(two_mla, *two_mla_physical);
      const auto shared_arena =
          sc::FrameSlotArenaPlan::compile(
              two_mla, *two_mla_physical,
              sc::FrameSlotArenaReuse::DisjointLifetimes,
              sc::kLegacyEvoCmaRegionAlignmentBytes, &arena_error,
              sc::kModalixProductionArenaDmsPolicy, two_mla_detached);
      require(shared_arena.has_value(), "two MLA stages need one graph arena: " + arena_error);

      const auto encoder_policy =
          sc::select_mla_output_carrier_policy(two_mla, *two_mla_physical, 0U);
      const auto decoder_policy =
          sc::select_mla_output_carrier_policy(two_mla, *two_mla_physical, 1U);
      require(encoder_policy == sc::MlaOutputCarrierPolicy::SharedFrameArena &&
                  decoder_policy == sc::MlaOutputCarrierPolicy::SeparateCpuVisible &&
                  two_mla_detached == std::vector<sc::ValueId>({2U}) &&
                  shared_arena->region(1U) != nullptr &&
                  shared_arena->region(2U) == nullptr,
              "only the exact terminal public MLA in a multi-MLA graph may detach its output");

      auto encoder_sources = sc::resolve_mla_input_physical_sources(two_mla, 0U, {}, &error);
      require(encoder_sources.has_value(), "encoder public IFM must resolve: " + error);
      auto encoder = make_projection(two_mla, 0U);
      require(sc::apply_dmabuf_plan_contract_projection(two_mla, 0U, *shared_arena, &encoder,
                                                        *encoder_sources, &error),
              "encoder stage-local projection must pass: " + error);

      std::vector<sima::LogicalTensorStaticSpec> encoder_outputs = encoder.logical_outputs;
      auto decoder_sources =
          sc::resolve_mla_input_physical_sources(two_mla, 1U, encoder_outputs, &error);
      require(decoder_sources.has_value(), "decoder internal IFM must resolve: " + error);
      auto decoder = make_projection(two_mla, 1U);
      require(sc::apply_dmabuf_plan_contract_projection(
                  two_mla, 1U, *shared_arena, decoder_policy, &decoder,
                  *decoder_sources, &error),
              "decoder stage-local projection must pass: " + error);
      require(encoder.frame_arena_role == sima::FrameArenaRole::Allocate &&
                  decoder.frame_arena_role == sima::FrameArenaRole::Allocate &&
                  decoder.frame_arena_storage_domain == sc::ArenaStorageDomain::Dms &&
                  decoder.frame_arena_size_bytes == 4096U &&
                  decoder.physical_inputs[0].source_physical_index == 0 &&
                  decoder.physical_inputs[0].source_byte_offset == 0 &&
                  decoder.physical_outputs[0].source_byte_offset == 0,
              "terminal MLA must import the prior carrier and own a distinct right-sized "
              "CPU-visible output pool");

      auto ambiguous_contract = make_projection(two_mla, 0U);
      require(!sc::apply_dmabuf_plan_contract_projection(two_mla, &ambiguous_contract,
                                                         *encoder_sources, &error),
              "stage-less projection must reject a multi-MLA plan");
    }));
