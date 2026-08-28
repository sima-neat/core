#define SIMA_NEAT_INTERNAL 1
#include "pipeline/internal/sima/ProcessCvuRunTargetPolicy.h"
#include "pipeline/internal/sima/static_contract/PhysicalExecutionPlan.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace sc = simaai::neat::pipeline_internal::sima::static_contract;

namespace {

void require(bool condition, const std::string& detail) {
  if (!condition) {
    std::cerr << "FAIL: " << detail << '\n';
    std::exit(1);
  }
}

enum class FusedFamily { QuantTess, CastTess, DetessCast, DetessDequant };

std::uint32_t graph_id(FusedFamily family) {
  switch (family) {
  case FusedFamily::QuantTess: return 226U;
  case FusedFamily::CastTess: return 224U;
  case FusedFamily::DetessCast: return 225U;
  case FusedFamily::DetessDequant: return 227U;
  }
  return 0U;
}

std::string family_name(FusedFamily family) {
  switch (family) {
  case FusedFamily::QuantTess: return "quant+tess";
  case FusedFamily::CastTess: return "cast+tess";
  case FusedFamily::DetessCast: return "detess+cast";
  case FusedFamily::DetessDequant: return "detess+dequant";
  }
  return {};
}

sc::ValueSpec value(sc::ValueId id, std::string name, std::string dtype,
                    sc::ValueRepresentation representation = sc::ValueRepresentation::Dense) {
  sc::ValueSpec result;
  result.id = id;
  result.name = std::move(name);
  result.required_bytes = dtype == "FP32" ? 64U : 16U;
  result.logical_dtype = std::move(dtype);
  result.logical_shape = sc::TensorShape{1, 1, 1, 16};
  result.logical_layout = "HWC";
  result.representation = representation;
  return result;
}

void add_unary(sc::ModelExecutionPlanData& data, sc::OpKind kind, std::string name,
               sc::ValueId input, sc::ValueId output, sc::OpConfig config) {
  sc::OpSpec op;
  op.id = static_cast<sc::OpId>(data.ops.size());
  op.sequence = data.ops.size() + 1U;
  op.name = std::move(name);
  op.kind = kind;
  op.processor = "EV74";
  op.inputs = {input};
  op.outputs = {output};
  op.input_shapes = {{1, 1, 1, 16}};
  op.output_shapes = {{1, 1, 1, 16}};
  op.config = std::move(config);
  data.ops.push_back(std::move(op));
}

sc::ModelExecutionPlan make_fused_plan(FusedFamily family, std::size_t lanes,
                                       bool publish_first_intermediate = false,
                                       bool branch_first_intermediate = false,
                                       bool malformed_first = false) {
  sc::ModelExecutionPlanData data;
  data.contract_version = "2.1.0";
  const bool ingress = family == FusedFamily::QuantTess || family == FusedFamily::CastTess;
  std::vector<sc::ValueId> boundary_values;
  std::vector<sc::ValueId> intermediates;
  std::vector<sc::ValueId> outer_values;

  if (ingress) {
    for (std::size_t lane = 0; lane < lanes; ++lane) {
      const auto id = static_cast<sc::ValueId>(data.values.size());
      data.model_inputs.push_back(id);
      data.values.push_back(value(id, "input_" + std::to_string(lane), "FP32"));
      boundary_values.push_back(id);
    }
    const auto middle_dtype = family == FusedFamily::QuantTess ? "INT8" : "BF16";
    for (std::size_t lane = 0; lane < lanes; ++lane) {
      const auto id = static_cast<sc::ValueId>(data.values.size());
      data.values.push_back(value(id, "middle_" + std::to_string(lane), middle_dtype));
      intermediates.push_back(id);
      if (family == FusedFamily::QuantTess) {
        add_unary(data, sc::OpKind::Quantize, "quant_" + std::to_string(lane),
                  boundary_values[lane], id,
                  sc::QuantizeOpConfig{malformed_first && lane == 0 ? "BF16" : "INT8", 8,
                                       "TONEAREST", {{0.25, -7}}});
      } else {
        add_unary(data, sc::OpKind::Cast, "cast_" + std::to_string(lane),
                  boundary_values[lane], id,
                  sc::CastOpConfig{malformed_first && lane == 0 ? "FP32" : "BF16"});
      }
    }
    for (std::size_t lane = 0; lane < lanes; ++lane) {
      const auto id = static_cast<sc::ValueId>(data.values.size());
      data.values.push_back(value(id, "tess_" + std::to_string(lane), middle_dtype,
                                  sc::ValueRepresentation::Tessellated));
      outer_values.push_back(id);
      add_unary(data, sc::OpKind::Tessellate, "tess_" + std::to_string(lane),
                intermediates[lane], id,
                sc::TessellateOpConfig{{1, 1, 1, 16}, false, false, middle_dtype});
    }
    const auto mla_output = static_cast<sc::ValueId>(data.values.size());
    data.values.push_back(value(mla_output, "mla_output", "INT8",
                                sc::ValueRepresentation::BackendNative));
    sc::OpSpec mla;
    mla.id = static_cast<sc::OpId>(data.ops.size());
    mla.sequence = data.ops.size() + 1U;
    mla.name = "MLA_0";
    mla.kind = sc::OpKind::Mla;
    mla.processor = "MLA";
    mla.inputs = outer_values;
    mla.outputs = {mla_output};
    mla.config = sc::MlaOpConfig{"model.elf", 4};
    data.ops.push_back(std::move(mla));
    for (std::size_t port = 0; port < outer_values.size(); ++port) {
      data.backend_ports.push_back(
          {0U, sc::BackendPortDirection::Input, port, "data.ifm." + std::to_string(port),
           outer_values[port], 16U, 4096U, sc::BackendPortAlignmentAuthority::LegacyPolicy,
           sc::BackendPortAccess::ReadOnly});
    }
    data.backend_ports.push_back(
        {0U, sc::BackendPortDirection::Output, 0U, "data.ofm.b0", mla_output, 16U, 4096U,
         sc::BackendPortAlignmentAuthority::LegacyPolicy, sc::BackendPortAccess::WriteOnly});
    data.model_outputs.push_back({0U, "mla_output", mla_output});
  } else {
    const auto mla_input = static_cast<sc::ValueId>(data.values.size());
    data.model_inputs.push_back(mla_input);
    data.values.push_back(value(mla_input, "mla_input", "INT8"));
    const auto middle_dtype = family == FusedFamily::DetessCast ? "BF16" : "INT8";
    for (std::size_t lane = 0; lane < lanes; ++lane) {
      const auto id = static_cast<sc::ValueId>(data.values.size());
      data.values.push_back(value(id, "mla_ofm_" + std::to_string(lane), middle_dtype,
                                  sc::ValueRepresentation::BackendNative));
      boundary_values.push_back(id);
    }
    sc::OpSpec mla;
    mla.id = static_cast<sc::OpId>(data.ops.size());
    mla.sequence = data.ops.size() + 1U;
    mla.name = "MLA_0";
    mla.kind = sc::OpKind::Mla;
    mla.processor = "MLA";
    mla.inputs = {mla_input};
    mla.outputs = boundary_values;
    mla.config = sc::MlaOpConfig{"model.elf", 4};
    data.ops.push_back(std::move(mla));
    data.backend_ports.push_back(
        {0U, sc::BackendPortDirection::Input, 0U, "data.ifm.b0", mla_input, 16U, 4096U,
         sc::BackendPortAlignmentAuthority::LegacyPolicy, sc::BackendPortAccess::ReadOnly});
    for (std::size_t lane = 0; lane < lanes; ++lane) {
      data.backend_ports.push_back(
          {0U, sc::BackendPortDirection::Output, lane, "data.ofm." + std::to_string(lane),
           boundary_values[lane], 16U, 4096U, sc::BackendPortAlignmentAuthority::LegacyPolicy,
           sc::BackendPortAccess::WriteOnly});
    }
    for (std::size_t lane = 0; lane < lanes; ++lane) {
      const auto id = static_cast<sc::ValueId>(data.values.size());
      data.values.push_back(value(id, "middle_" + std::to_string(lane), middle_dtype));
      intermediates.push_back(id);
      add_unary(data, sc::OpKind::Detessellate, "detess_" + std::to_string(lane),
                boundary_values[lane], id,
                sc::DetessellateOpConfig{{1, 1, 1, 16}, {1, 1, 1, 16}, false, false,
                                         malformed_first && lane == 0 ? "FP32" : middle_dtype});
    }
    for (std::size_t lane = 0; lane < lanes; ++lane) {
      const auto id = static_cast<sc::ValueId>(data.values.size());
      data.values.push_back(value(id, "output_" + std::to_string(lane), "FP32"));
      outer_values.push_back(id);
      if (family == FusedFamily::DetessCast) {
        add_unary(data, sc::OpKind::Cast, "cast_" + std::to_string(lane),
                  intermediates[lane], id, sc::CastOpConfig{"FP32"});
      } else {
        add_unary(data, sc::OpKind::Dequantize, "dequant_" + std::to_string(lane),
                  intermediates[lane], id, sc::DequantizeOpConfig{"INT8", {{0.25, -7}}});
      }
      data.model_outputs.push_back({lane, data.values[id].name, id});
    }
  }

  if (publish_first_intermediate) {
    data.model_outputs.push_back(
        {data.model_outputs.size(), data.values[intermediates.front()].name, intermediates.front()});
  }
  if (branch_first_intermediate) {
    const auto id = static_cast<sc::ValueId>(data.values.size());
    data.values.push_back(value(id, "branch_output", data.values[intermediates.front()].logical_dtype.value()));
    add_unary(data, sc::OpKind::PassThrough, "branch", intermediates.front(), id,
              sc::PassThroughOpConfig{});
    data.model_outputs.push_back({data.model_outputs.size(), "branch_output", id});
  }

  std::string error;
  auto plan = sc::ModelExecutionPlan::create(std::move(data), &error);
  require(plan.has_value(), "semantic fixture must be valid: " + error);
  return std::move(*plan);
}

sc::ModelExecutionPlan make_overlapping_pattern_plan() {
  sc::ModelExecutionPlanData data;
  data.contract_version = "2.1.0";
  data.model_inputs = {0U};
  data.values = {
      value(0U, "input", "BF16", sc::ValueRepresentation::BackendNative),
      value(1U, "detess", "BF16"), value(2U, "cast", "BF16"),
      value(3U, "tess", "BF16", sc::ValueRepresentation::Tessellated),
      value(4U, "mla_output", "INT8", sc::ValueRepresentation::BackendNative),
  };
  add_unary(data, sc::OpKind::Detessellate, "detess", 0U, 1U,
            sc::DetessellateOpConfig{{1, 1, 1, 16}, {1, 1, 1, 16}, false, false, "BF16"});
  add_unary(data, sc::OpKind::Cast, "cast", 1U, 2U, sc::CastOpConfig{"BF16"});
  add_unary(data, sc::OpKind::Tessellate, "tess", 2U, 3U,
            sc::TessellateOpConfig{{1, 1, 1, 16}, false, false, "BF16"});
  sc::OpSpec mla;
  mla.id = 3U;
  mla.sequence = 4U;
  mla.name = "mla";
  mla.kind = sc::OpKind::Mla;
  mla.processor = "MLA";
  mla.inputs = {3U};
  mla.outputs = {4U};
  mla.config = sc::MlaOpConfig{"model.elf", 4};
  data.ops.push_back(std::move(mla));
  data.backend_ports = {
      {0U, sc::BackendPortDirection::Input, 0U, "ifm", 3U, 16U, 4096U,
       sc::BackendPortAlignmentAuthority::LegacyPolicy, sc::BackendPortAccess::ReadOnly},
      {0U, sc::BackendPortDirection::Output, 0U, "ofm", 4U, 16U, 4096U,
       sc::BackendPortAlignmentAuthority::LegacyPolicy, sc::BackendPortAccess::WriteOnly},
  };
  data.model_outputs = {{0U, "mla_output", 4U}};
  std::string error;
  auto plan = sc::ModelExecutionPlan::create(std::move(data), &error);
  require(plan.has_value(), "overlap fixture must be semantically valid: " + error);
  return std::move(*plan);
}

sc::ModelExecutionPlan make_nonzero_view_barrier_plan() {
  sc::ModelExecutionPlanData data;
  data.contract_version = "2.1.0";
  data.model_inputs = {0U};
  data.values = {
      value(0U, "mla_input", "INT8"),
      value(1U, "mla_ofm", "INT8", sc::ValueRepresentation::BackendNative),
      value(2U, "detess", "INT8"), value(3U, "offset_view", "INT8"),
      value(4U, "output", "FP32"),
  };
  data.values[3].logical_shape = sc::TensorShape{1, 16};
  data.values[4].logical_shape = sc::TensorShape{1, 16};
  sc::StorageBinding detess_storage;
  detess_storage.kind = sc::StorageBindingKind::Root;
  detess_storage.carrier_id = 2U;
  detess_storage.physical_span = 17U;
  detess_storage.access = sc::StorageAccess::ReadWrite;
  data.values[2].storage_binding = detess_storage;
  data.values[3].read_expression = sc::ReadExpression{2U, 1U, {16, 1}};

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
  add_unary(data, sc::OpKind::Detessellate, "detess", 1U, 2U,
            sc::DetessellateOpConfig{{1, 1, 1, 16}, {1, 1, 1, 16}, false, false, "INT8"});
  add_unary(data, sc::OpKind::Reshape, "offset_view", 2U, 3U,
            sc::ReshapeOpConfig{{1, 16}});
  data.ops.back().output_shapes = {{1, 16}};
  add_unary(data, sc::OpKind::Dequantize, "dequant", 3U, 4U,
            sc::DequantizeOpConfig{"INT8", {{0.25, -7}}});
  data.ops.back().input_shapes = {{1, 16}};
  data.ops.back().output_shapes = {{1, 16}};
  data.backend_ports = {
      {0U, sc::BackendPortDirection::Input, 0U, "ifm", 0U, 16U, 4096U,
       sc::BackendPortAlignmentAuthority::LegacyPolicy, sc::BackendPortAccess::ReadOnly},
      {0U, sc::BackendPortDirection::Output, 0U, "ofm", 1U, 16U, 4096U,
       sc::BackendPortAlignmentAuthority::LegacyPolicy, sc::BackendPortAccess::WriteOnly},
  };
  data.model_outputs = {{0U, "output", 4U}};
  std::string error;
  auto plan = sc::ModelExecutionPlan::create(std::move(data), &error);
  require(plan.has_value(), "nonzero-view fixture must be semantically valid: " + error);
  return std::move(*plan);
}

sc::ModelExecutionPlan make_materializing_pack_barrier_plan() {
  sc::ModelExecutionPlanData data;
  data.contract_version = "2.1.0";
  data.model_inputs = {0U, 3U};
  data.values = {
      value(0U, "mla_input", "INT8"),
      value(1U, "mla_ofm", "INT8", sc::ValueRepresentation::BackendNative),
      value(2U, "detess", "INT8"), value(3U, "pack_peer", "INT8"),
      value(4U, "packed", "INT8"), value(5U, "output", "FP32"),
  };
  data.values[4].required_bytes = 32U;
  data.values[4].logical_shape = sc::TensorShape{1, 32};
  data.values[5].required_bytes = 128U;
  data.values[5].logical_shape = sc::TensorShape{1, 32};

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
  add_unary(data, sc::OpKind::Detessellate, "detess", 1U, 2U,
            sc::DetessellateOpConfig{{1, 1, 1, 16}, {1, 1, 1, 16}, false, false, "INT8"});
  sc::PackOpConfig pack_config;
  pack_config.components = {{2U, 0U, 16U}, {3U, 16U, 16U}};
  pack_config.materializes = true;
  sc::OpSpec pack;
  pack.id = 2U;
  pack.sequence = 3U;
  pack.name = "materializing_pack";
  pack.kind = sc::OpKind::Pack;
  pack.processor = "EV74";
  pack.inputs = {2U, 3U};
  pack.outputs = {4U};
  pack.input_shapes = {{1, 1, 1, 16}, {1, 1, 1, 16}};
  pack.output_shapes = {{1, 32}};
  pack.config = std::move(pack_config);
  data.ops.push_back(std::move(pack));
  add_unary(data, sc::OpKind::Dequantize, "dequant", 4U, 5U,
            sc::DequantizeOpConfig{"INT8", {{0.25, -7}}});
  data.ops.back().input_shapes = {{1, 32}};
  data.ops.back().output_shapes = {{1, 32}};
  data.backend_ports = {
      {0U, sc::BackendPortDirection::Input, 0U, "ifm", 0U, 16U, 4096U,
       sc::BackendPortAlignmentAuthority::LegacyPolicy, sc::BackendPortAccess::ReadOnly},
      {0U, sc::BackendPortDirection::Output, 0U, "ofm", 1U, 16U, 4096U,
       sc::BackendPortAlignmentAuthority::LegacyPolicy, sc::BackendPortAccess::WriteOnly},
  };
  data.model_outputs = {{0U, "output", 5U}};
  std::string error;
  auto plan = sc::ModelExecutionPlan::create(std::move(data), &error);
  require(plan.has_value(), "materializing-Pack fixture must be semantically valid: " + error);
  return std::move(*plan);
}

std::vector<const sc::PhysicalCommand*> commands(const sc::PhysicalExecutionPlan& plan,
                                                  std::uint32_t graph) {
  std::vector<const sc::PhysicalCommand*> result;
  for (const auto& command : plan.commands) {
    if (command.graph_id == graph) result.push_back(&command);
  }
  return result;
}

void verify_family_and_capacity(FusedFamily family, std::size_t lanes) {
  std::string error;
  const auto semantic = make_fused_plan(family, lanes);
  const auto physical = sc::PhysicalExecutionLowerer::lower(semantic, &error);
  require(physical.has_value(), family_name(family) + " lowers: " + error);
  auto fused = commands(*physical, graph_id(family));
  std::sort(fused.begin(), fused.end(), [](const auto* left, const auto* right) {
    return left->members.front().ordinal < right->members.front().ordinal;
  });
  const auto expected_chunks = (lanes + 31U) / 32U;
  require(fused.size() == expected_chunks,
          family_name(family) + " has exactly ceil(N/32) submissions");
  std::size_t seen = 0U;
  for (std::size_t chunk = 0; chunk < fused.size(); ++chunk) {
    const auto expected = std::min<std::size_t>(32U, lanes - seen);
    require(fused[chunk]->members.size() == expected,
            family_name(family) + " chunk has exact capacity split");
    const auto expected_role =
        family == FusedFamily::QuantTess || family == FusedFamily::CastTess
            ? sc::PhysicalCommandRole::Ingress
            : sc::PhysicalCommandRole::Egress;
    require(fused[chunk]->role == expected_role,
            family_name(family) + " retains its exact MLA-boundary placement role");
    require(fused[chunk]->inputs.size() == expected && fused[chunk]->outputs.size() == expected,
            family_name(family) + " binds one outer input/output per member");
    for (const auto& member : fused[chunk]->members) {
      require(member.semantic_chain.size() == 2U && member.ordinal == seen,
              family_name(family) + " retains ordered two-op provenance");
      require(member.outer_inputs.size() == 1U && member.outer_outputs.size() == 1U,
              family_name(family) + " exposes only outer values");
      require(physical->command_for_semantic_op[member.semantic_chain[0]] == fused[chunk]->id &&
                  physical->command_for_semantic_op[member.semantic_chain[1]] == fused[chunk]->id,
              family_name(family) + " maps both semantic ops to one command");
      ++seen;
    }
  }
  require(seen == lanes, family_name(family) + " preserves every semantic lane");
  const auto mla = std::find_if(physical->commands.begin(), physical->commands.end(),
                                [](const auto& command) {
                                  return command.engine == sc::PhysicalEngine::Mla;
                                });
  require(mla != physical->commands.end(), "fixture retains MLA command");
  if (family == FusedFamily::QuantTess || family == FusedFamily::CastTess) {
    std::vector<sc::PhysicalCommandId> fused_ids;
    for (const auto* command : fused) fused_ids.push_back(command->id);
    std::sort(fused_ids.begin(), fused_ids.end());
    require(mla->predecessors == fused_ids,
            family_name(family) + " makes MLA wait for every capacity chunk");
  } else {
    for (const auto* command : fused) {
      require(command->predecessors == std::vector<sc::PhysicalCommandId>{mla->id},
              family_name(family) + " makes every egress chunk wait for MLA");
    }
  }
}

void test_all_families_at_capacity_boundaries() {
  for (const auto family : {FusedFamily::QuantTess, FusedFamily::CastTess,
                            FusedFamily::DetessCast, FusedFamily::DetessDequant}) {
    for (const auto lanes : {1U, 2U, 3U, 31U, 32U, 33U, 65U}) {
      verify_family_and_capacity(family, lanes);
    }
  }
}

void test_public_and_branch_barriers_do_not_fuse_observed_edge() {
  for (const auto branch : {false, true}) {
    std::string error;
    auto semantic = make_fused_plan(FusedFamily::QuantTess, 3U, !branch, branch);
    const auto physical = sc::PhysicalExecutionLowerer::lower(semantic, &error);
    require(physical.has_value(), "barrier fixture lowers with standalone commands: " + error);
    const auto fused = commands(*physical, 226U);
    const auto quant = commands(*physical, 222U);
    const auto tess = commands(*physical, 2U);
    require(fused.size() == 1U && fused.front()->members.size() == 2U,
            "barrier terminates only the observed lane");
    require(quant.size() == 1U && tess.size() == 1U &&
                quant.front()->members.size() == 1U && tess.front()->members.size() == 1U,
            "observed lane remains true standalone commands rather than partial fusion");
  }
}

void test_relation_transparency_rejects_offset_and_materialization() {
  std::string error;
  const auto offset = make_nonzero_view_barrier_plan();
  require(!sc::resolve_exact_private_ordered_relation_path(offset, 1U, 3U),
          "nonzero-offset view is never transparent to a fused kernel");
  const auto offset_physical = sc::PhysicalExecutionLowerer::lower(offset, &error);
  require(offset_physical && commands(*offset_physical, 227U).empty(),
          "nonzero-offset view splits instead of selecting graph227");

  const auto materializing = make_materializing_pack_barrier_plan();
  require(!sc::resolve_exact_private_ordered_relation_path(materializing, 1U, 3U),
          "materializing Pack is never an address-only fusion relation");
  error.clear();
  const auto materializing_physical =
      sc::PhysicalExecutionLowerer::lower(materializing, &error);
  require(!materializing_physical && error.find("no implementation") != std::string::npos,
          "materializing relation fails closed without a registered implementation");
}

void test_invalid_mandatory_pair_fails_without_standalone_fallback() {
  for (const auto family : {FusedFamily::QuantTess, FusedFamily::CastTess,
                            FusedFamily::DetessCast, FusedFamily::DetessDequant}) {
    std::string error;
    const auto physical = sc::PhysicalExecutionLowerer::lower(
        make_fused_plan(family, 1U, false, false, true), &error);
    require(!physical && error.find("mandatory fused CVU chain") != std::string::npos,
            family_name(family) + " rejects an invalid eligible pair instead of splitting it");
  }
}

void test_overlapping_mandatory_pairs_reject_without_scan_order_choice() {
  std::string error;
  const auto physical =
      sc::PhysicalExecutionLowerer::lower(make_overlapping_pattern_plan(), &error);
  require(!physical && error.find("overlapping mandatory CVU patterns") != std::string::npos,
          "overlapping 225/224 patterns require a registered longer implementation");
}

void test_deterministic_repeat() {
  const auto semantic = make_fused_plan(FusedFamily::QuantTess, 33U);
  const auto first = sc::PhysicalExecutionLowerer::lower(semantic);
  const auto second = sc::PhysicalExecutionLowerer::lower(semantic);
  require(first && second && first->deterministic_digest_material == second->deterministic_digest_material,
          "repeated fused lowering produces identical digest material");
}

void test_dependency_tracker_chunk_failure() {
  std::string error;
  const auto physical = sc::PhysicalExecutionLowerer::lower(
      make_fused_plan(FusedFamily::QuantTess, 33U), &error);
  require(physical.has_value(), "tracker fixture lowers: " + error);
  auto tracker = sc::PhysicalExecutionTracker::create(*physical, &error);
  require(tracker.has_value(), "tracker accepts fused plan: " + error);
  auto fused = commands(*physical, 226U);
  std::sort(fused.begin(), fused.end(), [](const auto* a, const auto* b) {
    return a->members.front().ordinal < b->members.front().ordinal;
  });
  require(tracker->claim(fused[0]->id, &error) && tracker->claim(fused[1]->id, &error),
          "both independent fused chunks can be in flight");
  require(tracker->complete(fused[0]->id, &error), "first fused chunk completes");
  require(tracker->fail(fused[1]->id, &error), "second fused chunk fails");
  require(tracker->terminal() && !tracker->succeeded(),
          "partial capacity completion cannot publish frame success");
}

void test_dependency_tracker_cross_engine_parallelism() {
  sc::PhysicalExecutionPlan plan;
  plan.commands = {
      sc::PhysicalCommand{.id = 0U, .topological_rank = 0U,
                          .engine = sc::PhysicalEngine::Cvu,
                          .role = sc::PhysicalCommandRole::Interstitial,
                          .implementation_id = "cvu.graph221.cast.v1", .graph_id = 221U,
                          .maximum_members = 32U,
                          .members = {{0U, {0U}, {0U}, {1U}}},
                          .inputs = {0U}, .outputs = {1U}},
      sc::PhysicalCommand{.id = 1U, .topological_rank = 1U,
                          .engine = sc::PhysicalEngine::A65,
                          .implementation_id = "a65.available",
                          .members = {{0U, {1U}, {2U}, {3U}}},
                          .inputs = {2U}, .outputs = {3U}},
  };
  std::string error;
  auto tracker = sc::PhysicalExecutionTracker::create(plan, &error);
  require(tracker && tracker->ready(0U) && tracker->ready(1U),
          "independent engines are both ready: " + error);
  require(tracker->claim(1U, &error) && tracker->claim(0U, &error) &&
              tracker->complete(0U, &error) && tracker->complete(1U, &error) &&
              tracker->succeeded(),
          "independent cross-engine work has no head-of-line barrier");
}

void test_exact_physical_role_placement_precedes_coarse_target() {
  namespace sima = simaai::neat::pipeline_internal::sima;
  simaai::neat::ContractCompileInput input;
  input.processcvu_requested_run_target = "EV74";
  input.processcvu.pre_run_target = "EV74";
  input.processcvu.post_run_target = "A65";

  sima::ProcessCvuStagePayload cast;
  cast.graph_family_enum = sima::ProcessCvuGraphFamily::Cast;
  cast.graph_id = 221;
  cast.graph_name = "cast";
  cast.graph_family = "cast";
  const auto post = sima::resolve_processcvu_backend_decision(
      cast, input, "physical_cvu_cohort_1",
      sc::PhysicalCommandRole::Egress);
  require(post.effective_run_target == "A65" &&
              post.reason.find("processcvu_post") != std::string::npos,
          "post graph221 must honor explicit post=A65 before coarse EV74");
  const auto pre = sima::resolve_processcvu_backend_decision(
      cast, input, "physical_cvu_cohort_1",
      sc::PhysicalCommandRole::Ingress);
  require(pre.effective_run_target == "EV74" &&
              pre.reason.find("processcvu_pre") != std::string::npos,
          "pre graph221 must retain explicit pre=EV74");

  struct FusedPlacement {
    sima::ProcessCvuGraphFamily family;
    int graph_id;
    sc::PhysicalCommandRole role;
    const char* expected;
  };
  const std::vector<FusedPlacement> fused{
      {sima::ProcessCvuGraphFamily::CastTess, 224, sc::PhysicalCommandRole::Ingress, "EV74"},
      {sima::ProcessCvuGraphFamily::DetessCast, 225, sc::PhysicalCommandRole::Egress, "A65"},
      {sima::ProcessCvuGraphFamily::QuantTess, 226, sc::PhysicalCommandRole::Ingress, "EV74"},
      {sima::ProcessCvuGraphFamily::DetessDequant, 227, sc::PhysicalCommandRole::Egress, "A65"},
  };
  for (const auto& item : fused) {
    sima::ProcessCvuStagePayload payload;
    payload.graph_family_enum = item.family;
    payload.graph_id = item.graph_id;
    const auto decision = sima::resolve_processcvu_backend_decision(
        payload, input, "physical_cvu_cohort_1", item.role);
    require(decision.effective_run_target == item.expected,
            "mandatory fused graph must retain its exact physical placement role");
  }

  sima::ProcessCvuStagePayload interstitial;
  interstitial.graph_family_enum = sima::ProcessCvuGraphFamily::DetessCast;
  interstitial.graph_id = 225;
  const char* original_env = std::getenv("SIMA_PROCESSCVU_RUN_TARGET");
  const std::optional<std::string> saved_env =
      original_env ? std::optional<std::string>(original_env) : std::nullopt;
  require(::setenv("SIMA_PROCESSCVU_RUN_TARGET", "A65", 1) == 0,
          "test must install the legacy environment override");
  const auto middle = sima::resolve_processcvu_backend_decision(
      interstitial, input, "physical_cvu_cohort_1",
      sc::PhysicalCommandRole::Interstitial);
  if (saved_env.has_value()) {
    require(::setenv("SIMA_PROCESSCVU_RUN_TARGET", saved_env->c_str(), 1) == 0,
            "test must restore the processcvu environment override");
  } else {
    require(::unsetenv("SIMA_PROCESSCVU_RUN_TARGET") == 0,
            "test must clear the processcvu environment override");
  }
  require(middle.effective_run_target == "EV74" &&
              middle.reason.find("coarse_request") != std::string::npos,
          "interstitial role must use the coarse request without family/name/env inference");
}

} // namespace

int main() {
  test_all_families_at_capacity_boundaries();
  test_public_and_branch_barriers_do_not_fuse_observed_edge();
  test_relation_transparency_rejects_offset_and_materialization();
  test_invalid_mandatory_pair_fails_without_standalone_fallback();
  test_overlapping_mandatory_pairs_reject_without_scan_order_choice();
  test_deterministic_repeat();
  test_dependency_tracker_chunk_failure();
  test_dependency_tracker_cross_engine_parallelism();
  test_exact_physical_role_placement_precedes_coarse_target();
  std::cout << "unit_physical_execution_plan_test: PASS\n";
  return 0;
}
