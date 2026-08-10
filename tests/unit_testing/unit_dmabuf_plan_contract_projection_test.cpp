#include "pipeline/internal/sima/static_contract/DmabufPlanContractProjection.h"

#include "test_main.h"
#include "test_utils.h"

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
    data.values.push_back(sc::ValueSpec{
        static_cast<sc::ValueId>(30U + index),
        "post_" + std::to_string(index), 4096U});
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
    const auto alignment =
        index == 7U ? 8192U : sc::kLegacyEvoCmaRegionAlignmentBytes;
    const auto alignment_authority =
        index == 7U ? sc::BackendPortAlignmentAuthority::Contract
                    : sc::BackendPortAlignmentAuthority::LegacyPolicy;
    data.backend_ports.push_back(sc::BackendPortSpec{
        0U, sc::BackendPortDirection::Output, index,
        "data.ofm.persistent.output_" + std::to_string(index), value_id,
        data.values[value_id].required_bytes, alignment, alignment_authority,
        sc::BackendPortAccess::WriteOnly});
    const auto public_value_id = static_cast<sc::ValueId>(index + 30U);
    data.model_outputs.push_back(
        sc::ModelOutputSpec{index, data.values[public_value_id].name,
                            public_value_id});
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
      sc::ValueSpec{0U, "image_0", 100U},       sc::ValueSpec{1U, "image_1", 200U},
      sc::ValueSpec{2U, "quantize_0", 638976U}, sc::ValueSpec{3U, "quantize_1", 319488U},
      sc::ValueSpec{4U, "ofm_0", 192U},
  };
  data.model_inputs = {0U, 1U};

  sc::OpSpec quantize_0;
  quantize_0.id = 0U;
  quantize_0.sequence = 1U;
  quantize_0.name = "Quantize_0";
  quantize_0.kind = sc::OpKind::Quantize;
  quantize_0.processor = "CVU";
  quantize_0.inputs = {0U};
  quantize_0.outputs = {2U};
  quantize_0.config = sc::QuantizeOpConfig{"int8", 8, "nearest", {}};
  data.ops.push_back(std::move(quantize_0));

  sc::OpSpec quantize_1;
  quantize_1.id = 1U;
  quantize_1.sequence = 2U;
  quantize_1.name = "Quantize_1";
  quantize_1.kind = sc::OpKind::Quantize;
  quantize_1.processor = "CVU";
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
      sc::ValueSpec{6U, "unpack_0", 200U, "int8", sc::TensorShape{1, 10, 1, 20},
                    std::nullopt, {}, sc::ValueRepresentation::BackendNative,
                    sc::ReadExpression{5U, 0U, {200, 20, 20, 1}}},
      sc::ValueSpec{7U, "slice_0", 10U, "int8", sc::TensorShape{1, 10, 1, 1},
                    std::nullopt, {}, sc::ValueRepresentation::Dense,
                    sc::ReadExpression{5U, 0U, {200, 20, 20, 1}}},
      sc::ValueSpec{8U, "unpack_1", 200U, "int8", sc::TensorShape{1, 10, 1, 20},
                    std::nullopt, {}, sc::ValueRepresentation::BackendNative,
                    sc::ReadExpression{5U, 200U, {200, 20, 20, 1}}},
      sc::ValueSpec{9U, "dequant_0", 40U},
      sc::ValueSpec{10U, "dequant_1", 800U},
      sc::ValueSpec{11U, "published_0", 40U},
      sc::ValueSpec{12U, "published_1", 800U},
  };
  data.model_inputs = {0U, 1U};

  const auto add_op = [&](sc::OpKind kind, std::string name,
                          std::vector<sc::ValueId> inputs,
                          std::vector<sc::ValueId> outputs,
                          sc::OpConfig config) {
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
         sc::UnpackOpConfig{{"int8", "int8"},
                            {{1, 10, 1, 20}, {1, 10, 1, 20}}});
  add_op(sc::OpKind::Slice, "slice", {6U}, {7U},
         sc::SliceOpConfig{{0, 0, 0, 0}, {1, 10, 1, 1},
                           {1, 10, 1, 20}, {1, 10, 1, 1}});
  add_op(sc::OpKind::Dequantize, "dequantize_0", {7U}, {9U},
         sc::DequantizeOpConfig{"int8", {}});
  add_op(sc::OpKind::Dequantize, "dequantize_1", {8U}, {10U},
         sc::DequantizeOpConfig{"int8", {}});
  add_op(sc::OpKind::PassThrough, "publish", {9U, 10U}, {11U, 12U},
         sc::PassThroughOpConfig{});

  data.backend_ports = {
      sc::BackendPortSpec{0U, sc::BackendPortDirection::Input, 0U, "data.ifm.b0", 4U,
                          packed_ifm_bytes,
                          sc::kLegacyEvoCmaRegionAlignmentBytes,
                          sc::BackendPortAlignmentAuthority::LegacyPolicy,
                          sc::BackendPortAccess::ReadOnly},
      sc::BackendPortSpec{0U, sc::BackendPortDirection::Output, 0U, "data.ofm.b0", 5U, 400U,
                          sc::kLegacyEvoCmaRegionAlignmentBytes,
                          sc::BackendPortAlignmentAuthority::LegacyPolicy,
                          sc::BackendPortAccess::WriteOnly},
  };
  data.model_outputs = {{0U, "published_0", 11U}, {1U, "published_1", 12U}};
  std::string error;
  auto plan = sc::ModelExecutionPlan::create(std::move(data), &error);
  require(plan.has_value(), "packed execution plan must be valid: " + error);
  return std::move(*plan);
}

sc::ModelExecutionPlan make_single_read_plan() {
  sc::ModelExecutionPlanData data;
  data.contract_version = "2.0.0";
  data.values = {
      sc::ValueSpec{0U, "ifm", 64U, "int8", sc::TensorShape{1, 1, 1, 64}},
      sc::ValueSpec{1U, "physical_ofm", 192U, "int8", sc::TensorShape{1, 1, 1, 192}},
      sc::ValueSpec{2U, "slice_view", 12U, "int8", sc::TensorShape{1, 1, 1, 12},
                    std::nullopt, {}, sc::ValueRepresentation::Dense,
                    sc::ReadExpression{1U, 8U, {192, 192, 16, 1}}},
      sc::ValueSpec{3U, "dequantized", 48U, "fp32", sc::TensorShape{1, 1, 1, 12}},
      sc::ValueSpec{4U, "published", 48U, "fp32", sc::TensorShape{1, 1, 1, 12}},
  };
  data.model_inputs = {0U};

  const auto add_op = [&](sc::OpKind kind, std::string name,
                          std::vector<sc::ValueId> inputs,
                          std::vector<sc::ValueId> outputs,
                          sc::OpConfig config) {
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
  add_op(sc::OpKind::Mla, "MLA_0", {0U}, {1U},
         sc::MlaOpConfig{"model.elf", 4});
  add_op(sc::OpKind::Slice, "slice", {1U}, {2U},
         sc::SliceOpConfig{{0, 0, 0, 0}, {1, 1, 1, 12},
                           {1, 1, 1, 192}, {1, 1, 1, 12}});
  add_op(sc::OpKind::Dequantize, "dequantize", {2U}, {3U},
         sc::DequantizeOpConfig{"int8", {}});
  add_op(sc::OpKind::PassThrough, "publish", {3U}, {4U},
         sc::PassThroughOpConfig{});

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
  data.model_outputs = {{0U, "published", 4U}};
  std::string error;
  auto plan = sc::ModelExecutionPlan::create(std::move(data), &error);
  require(plan.has_value(), "single-read execution plan must be valid: " + error);
  return std::move(*plan);
}

sima::MlaStaticContract make_projection(const sc::ModelExecutionPlan& plan) {
  sima::MlaStaticContract contract;
  for (const auto& port : plan.backend_ports()) {
    const auto* value = plan.value(port.value_id);
    sima::PhysicalBufferStaticSpec physical;
    physical.physical_index = static_cast<int>(port.port_index);
    physical.size_bytes = port.required_bytes;
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
                      plan.backend_ports()[2U + 7U].required_bytes,
              "raw OFM publication must preserve backend order and extent");
      require(contract.frame_arena_size_bytes > 0U &&
                  contract.frame_arena_role == sima::FrameArenaRole::Allocate,
              "direct MLA must allocate the first common frame arena");

      auto wrong_size = make_projection(plan);
      ++wrong_size.dispatcher_physical_outputs[7].size_bytes;
      require(!sc::apply_dmabuf_plan_contract_projection(plan, &wrong_size, *input_sources, &error),
              "an OFM byte mismatch must fail closed");

      auto wrong_name = make_projection(plan);
      wrong_name.physical_inputs[1].segment_name = "guessed_ifm";
      require(!sc::apply_dmabuf_plan_contract_projection(plan, &wrong_name, *input_sources, &error),
              "a guessed IFM name must fail closed");

      auto aliased_sources = *input_sources;
      aliased_sources[1].source_physical_index =
          aliased_sources[0].source_physical_index;
      auto aliased_contract = make_projection(plan);
      require(!sc::apply_dmabuf_plan_contract_projection(plan, &aliased_contract,
                                                         aliased_sources,
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
      require(frontend_sources->size() == 2U &&
                  (*frontend_sources)[0].value_id == 2U &&
                  (*frontend_sources)[0].source_physical_index == 0 &&
                  (*frontend_sources)[1].value_id == 3U &&
                  (*frontend_sources)[1].source_physical_index == 1,
              "frontend-produced IFMs must retain backend order and exact carriers");
      auto frontend_mla = make_projection(frontend_plan);
      require(sc::apply_dmabuf_plan_contract_projection(
                  frontend_plan, &frontend_mla, *frontend_sources, &error),
              "frontend MLA frame-arena projection must pass: " + error);
      require(frontend_mla.frame_arena_role == sima::FrameArenaRole::ReuseInput &&
                  frontend_mla.frame_arena_size_bytes > 0U &&
                  frontend_mla.physical_inputs[0].source_byte_offset !=
                      frontend_mla.physical_inputs[1].source_byte_offset,
              "MLA must reuse distinct pre-CVU regions in one parent arena");

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
                  frontend_plan, sc::ProcessCvuMlaBoundary::Inputs,
                  &frontend_cvu, &frontend_runtime, &frontend_exposed, &error),
              "pre-CVU frame-arena projection must pass: " + error);
      require(frontend_runtime.frame_arena_role ==
                      sima::FrameArenaRole::Allocate &&
                  frontend_runtime.frame_arena_size_bytes ==
                      frontend_mla.frame_arena_size_bytes &&
                  frontend_runtime.physical_outputs[0].source_byte_offset ==
                      frontend_mla.physical_inputs[0].source_byte_offset &&
                  frontend_runtime.physical_outputs[1].source_byte_offset ==
                      frontend_mla.physical_inputs[1].source_byte_offset,
              "pre-CVU and MLA must share one exact absolute region plan");

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
      require(packed_sources->size() == 1U &&
                  (*packed_sources)[0].source_physical_index == 0,
              "packed IFM must resolve to its one physical parent");
      auto packed_contract = make_projection(packed_plan);
      require(sc::apply_dmabuf_plan_contract_projection(
                  packed_plan, &packed_contract, *packed_sources, &error),
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
      packed_pre_exposed.exposed_logical_outputs =
          packed_pre_runtime.logical_outputs;
      require(sc::apply_dmabuf_plan_processcvu_contract_projection(
                  packed_plan, sc::ProcessCvuMlaBoundary::Inputs, &packed_pre,
                  &packed_pre_runtime, &packed_pre_exposed, &error),
              "packed pre-CVU placement must project: " + error);
      require(packed_pre_runtime.physical_outputs.size() == 2U &&
                  packed_pre_runtime.physical_outputs[1].source_byte_offset -
                          packed_pre_runtime.physical_outputs[0].source_byte_offset ==
                      640 &&
                  packed_pre_runtime.physical_outputs[0]
                          .required_alignment_bytes == 16U &&
                  !packed_pre_runtime.consumer_keeps_distinct_physical_inputs,
              "Pack children must be direct writes into one ordered parent carrier");

      const auto single_read_plan = make_single_read_plan();
      auto single_read_contract = make_projection(single_read_plan);
      auto single_read_sources =
          sc::resolve_mla_input_physical_sources(single_read_plan, {}, &error);
      require(single_read_sources.has_value(),
              "single-read model-input carrier must resolve: " + error);
      require(sc::apply_dmabuf_plan_contract_projection(
                  single_read_plan, &single_read_contract, *single_read_sources, &error),
              "single read-view projection must pass: " + error);
      require(single_read_contract.logical_outputs.size() == 1U &&
                  single_read_contract.logical_outputs[0].byte_offset == 8 &&
                  single_read_contract.logical_outputs[0].size_bytes == 12U &&
                  single_read_contract.logical_outputs[0].stride_bytes ==
                      std::vector<std::int64_t>({192, 192, 16, 1}),
              "one Slice-derived OFM must retain its exact affine read expression");

      auto gapped = packed_upstream;
      ++gapped[1].byte_offset;
      require(!sc::resolve_mla_input_physical_sources(packed_plan, gapped, &error),
              "packed IFM with a gap must fail closed");

      const auto incomplete_parent_plan = make_packed_plan(1920U);
      require(!sc::resolve_mla_input_physical_sources(
                  incomplete_parent_plan, packed_upstream, &error),
              "legacy doubled-size BF16 parent without exact producer coverage must fail closed");

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
                  plan, sc::ProcessCvuMlaBoundary::Outputs, &dequant,
                  &post_runtime, &post_exposed, &error),
              "dequant must project onto the production driver graph: " + error);
      require(dequant.dmabuf_plan_contract && dequant.graph_id == 223,
              "dequant must use canonical graph 223 in dmabuf-plan mode");
      require(post_runtime.physical_outputs.size() == 28U &&
                  post_runtime.logical_outputs[7].physical_index == 7 &&
                  post_runtime.physical_outputs[7].required_alignment_bytes == 8192U,
              "post-CVU output region must inherit the exact ordered MLA OFM alignment");
      require(post_runtime.frame_arena_role ==
                      sima::FrameArenaRole::ReuseInput &&
                  post_runtime.frame_arena_size_bytes ==
                      contract.frame_arena_size_bytes,
              "post-CVU must reuse the same parent frame arena");

      const std::vector<std::pair<sima::ProcessCvuGraphFamily, int>> driver_families = {
          {sima::ProcessCvuGraphFamily::Tess, 2},
          {sima::ProcessCvuGraphFamily::QuantTess, 226},
          {sima::ProcessCvuGraphFamily::CastTess, 224},
          {sima::ProcessCvuGraphFamily::Detess, 3},
          {sima::ProcessCvuGraphFamily::DetessCast, 225},
          {sima::ProcessCvuGraphFamily::DetessDequant, 227},
      };
      const std::vector<std::string> canonical_tokens = {
          "tessellate", "quantizetessellate", "casttess", "detessellate",
          "detesscast", "detessdequant",
      };
      std::size_t family_index = 0U;
      for (const auto& [family, graph_id] : driver_families) {
        sima::ProcessCvuStagePayload projected;
        projected.graph_family_enum = family;
        auto family_runtime = post_runtime;
        auto family_exposed = post_exposed;
        require(sc::apply_dmabuf_plan_processcvu_contract_projection(
                    plan, sc::ProcessCvuMlaBoundary::Outputs, &projected,
                    &family_runtime, &family_exposed, &error),
                "tess/detess family must project onto a production driver graph: " + error);
        require(projected.dmabuf_plan_contract && projected.graph_id == graph_id &&
                    projected.graph_name == canonical_tokens[family_index++] &&
                    projected.descriptor_abi_id ==
                        SIMA_PLUGIN_CVU_DESCRIPTOR_ABI_TENSOR_TRANSFORM_PAIR_V1 &&
                    projected.descriptor_contract_version == 1U &&
                    projected.binding_schema_version == 1U &&
                    projected.supported_placement_mask ==
                        (SIMA_PLUGIN_CVU_PLACEMENT_EV74 |
                         SIMA_PLUGIN_CVU_PLACEMENT_A65) &&
                    projected.allowed_frame_patch_mask ==
                        SIMA_PLUGIN_CVU_FRAME_PATCH_METADATA,
                "tess/detess family must publish its exact /dev/cvu registry handshake");
      }

      sima::ProcessCvuStagePayload preproc;
      preproc.graph_family_enum = sima::ProcessCvuGraphFamily::Preproc;
      auto preproc_runtime = packed_pre_runtime;
      auto preproc_exposed = packed_pre_exposed;
      require(sc::apply_dmabuf_plan_processcvu_contract_projection(
                  packed_plan, sc::ProcessCvuMlaBoundary::Inputs, &preproc,
                  &preproc_runtime, &preproc_exposed, &error),
              "preproc must project through its strict graph-200 descriptor ABI: " +
                  error);
      require(preproc.dmabuf_plan_contract && preproc.graph_id == 200 &&
                  preproc.graph_name == "preproc" &&
                  preproc.descriptor_abi_id ==
                      SIMA_PLUGIN_CVU_DESCRIPTOR_ABI_PREPROC_V1 &&
                  preproc.descriptor_contract_version == 1U &&
                  preproc.binding_schema_version == 1U &&
                  preproc.supported_placement_mask ==
                      SIMA_PLUGIN_CVU_PLACEMENT_EV74 &&
                  preproc.allowed_frame_patch_mask ==
                      (SIMA_PLUGIN_CVU_FRAME_PATCH_METADATA |
                       SIMA_PLUGIN_CVU_FRAME_PATCH_PREPROC_GEOMETRY |
                       SIMA_PLUGIN_CVU_FRAME_PATCH_PREPROC_SCALAR_ROI |
                       SIMA_PLUGIN_CVU_FRAME_PATCH_PREPROC_ROI_LIST |
                       SIMA_PLUGIN_CVU_FRAME_PATCH_PREPROC_PLANE_LAYOUT),
              "preproc must publish the exact bounded graph-200 registry handshake");
    }));
