#define SIMA_NEAT_INTERNAL 1
#include "pipeline/internal/sima/static_contract/FrameSlotArenaPlan.h"
#include "pipeline/internal/sima/static_contract/PhysicalExecutionPlan.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace sc = simaai::neat::pipeline_internal::sima::static_contract;

namespace {

int failures = 0;

#define CHECK(expr)                                                            \
  do {                                                                         \
    if (!(expr)) {                                                             \
      std::cerr << __FILE__ << ':' << __LINE__ << ": CHECK failed: " #expr    \
                << '\n';                                                       \
      ++failures;                                                              \
    }                                                                          \
  } while (false)

sc::ValueSpec value(sc::ValueId id, std::string name, std::uint64_t bytes) {
  sc::ValueSpec result;
  result.id = id;
  result.name = std::move(name);
  result.required_bytes = bytes;
  return result;
}

sc::ModelExecutionPlan make_plan() {
  sc::ModelExecutionPlanData data;
  data.contract_version = "arena-test-v1";
  data.values = {
      value(0, "input0", 64),  value(1, "input1", 64),
      value(2, "ifm0", 1024), value(3, "ifm1", 2048),
      value(4, "ofm0", 1536), value(5, "ofm1", 512),
      value(6, "out0", 768),  value(7, "out1", 2304),
  };
  data.model_inputs = {0, 1};

  sc::OpSpec pre;
  pre.id = 0;
  pre.sequence = 1;
  pre.name = "pre";
  pre.kind = sc::OpKind::Quantize;
  pre.processor = "EV74";
  pre.kernel = "quantize";
  pre.inputs = {0, 1};
  pre.outputs = {2, 3};
  pre.config = sc::QuantizeOpConfig{};

  sc::OpSpec mla;
  mla.id = 1;
  mla.sequence = 2;
  mla.name = "mla";
  mla.kind = sc::OpKind::Mla;
  mla.processor = "MLA";
  mla.inputs = {2, 3};
  mla.outputs = {4, 5};
  mla.config = sc::MlaOpConfig{"model.elf", 4};

  sc::OpSpec post;
  post.id = 2;
  post.sequence = 3;
  post.name = "post";
  post.kind = sc::OpKind::Dequantize;
  post.processor = "EV74";
  post.kernel = "dequantize";
  post.inputs = {4, 5};
  post.outputs = {6, 7};
  post.config = sc::DequantizeOpConfig{};
  data.ops = {std::move(pre), std::move(mla), std::move(post)};

  data.backend_ports = {
      {0, sc::BackendPortDirection::Input, 0, "ifm0", 2, 1024, 4096,
       sc::BackendPortAlignmentAuthority::LegacyPolicy,
       sc::BackendPortAccess::ReadOnly},
      {0, sc::BackendPortDirection::Input, 1, "ifm1", 3, 2048, 4096,
       sc::BackendPortAlignmentAuthority::LegacyPolicy,
       sc::BackendPortAccess::ReadOnly},
      {0, sc::BackendPortDirection::Output, 0, "ofm0", 4, 1536, 8192,
       sc::BackendPortAlignmentAuthority::Contract,
       sc::BackendPortAccess::WriteOnly},
      {0, sc::BackendPortDirection::Output, 1, "ofm1", 5, 512, 4096,
       sc::BackendPortAlignmentAuthority::LegacyPolicy,
       sc::BackendPortAccess::WriteOnly},
  };
  data.model_outputs = {{0, "out0", 6}, {1, "out1", 7}};

  std::string error;
  auto plan = sc::ModelExecutionPlan::create(std::move(data), &error);
  if (!plan) {
    std::cerr << "unable to build fixture: " << error << '\n';
    std::abort();
  }
  return std::move(*plan);
}

sc::ModelExecutionPlan make_staggered_read_plan() {
  sc::ModelExecutionPlanData data;
  data.contract_version = "arena-read-test-v1";
  data.values = {
      value(0, "input", 64), value(1, "ofm0", 4096),
      value(2, "ofm1", 4096), value(3, "view0", 4096),
      value(4, "post0", 4096), value(5, "view1", 4096),
      value(6, "post1", 4096),
  };
  data.values[3].logical_shape = sc::TensorShape{4096};
  data.values[3].read_expression = sc::ReadExpression{1, 0, {1}};
  data.values[5].logical_shape = sc::TensorShape{4096};
  data.values[5].read_expression = sc::ReadExpression{2, 0, {1}};
  data.model_inputs = {0};

  const auto add = [&](sc::OpKind kind, std::string name,
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
  add(sc::OpKind::Mla, "mla", {0}, {1, 2},
      sc::MlaOpConfig{"model.elf", 4});
  add(sc::OpKind::Slice, "slice0", {1}, {3}, sc::SliceOpConfig{});
  add(sc::OpKind::Dequantize, "post0", {3}, {4},
      sc::DequantizeOpConfig{});
  add(sc::OpKind::Slice, "slice1", {2}, {5}, sc::SliceOpConfig{});
  add(sc::OpKind::Dequantize, "post1", {5}, {6},
      sc::DequantizeOpConfig{});
  data.backend_ports = {
      {0, sc::BackendPortDirection::Input, 0, "ifm0", 0, 64, 4096,
       sc::BackendPortAlignmentAuthority::LegacyPolicy,
       sc::BackendPortAccess::ReadOnly},
      {0, sc::BackendPortDirection::Output, 0, "ofm0", 1, 4096, 4096,
       sc::BackendPortAlignmentAuthority::LegacyPolicy,
       sc::BackendPortAccess::WriteOnly},
      {0, sc::BackendPortDirection::Output, 1, "ofm1", 2, 4096, 4096,
       sc::BackendPortAlignmentAuthority::LegacyPolicy,
       sc::BackendPortAccess::WriteOnly},
  };
  data.model_outputs = {{0, "post0", 4}, {1, "post1", 6}};
  std::string error;
  auto plan = sc::ModelExecutionPlan::create(std::move(data), &error);
  if (!plan) {
    std::cerr << "unable to build staggered read fixture: " << error << '\n';
    std::abort();
  }
  return std::move(*plan);
}

sc::ModelExecutionPlan make_fused_quanttess_arena_plan(std::size_t lanes,
                                                       std::vector<sc::ValueId>* intermediates,
                                                       std::vector<sc::ValueId>* final_tess) {
  sc::ModelExecutionPlanData data;
  data.contract_version = "arena-fused-v1";
  std::vector<sc::ValueId> inputs;
  for (std::size_t lane = 0; lane < lanes; ++lane) {
    const auto id = static_cast<sc::ValueId>(data.values.size());
    auto item = value(id, "input_" + std::to_string(lane), 64U);
    item.logical_dtype = "FP32";
    item.logical_shape = sc::TensorShape{1, 1, 1, 16};
    data.values.push_back(std::move(item));
    data.model_inputs.push_back(id);
    inputs.push_back(id);
  }
  for (std::size_t lane = 0; lane < lanes; ++lane) {
    const auto id = static_cast<sc::ValueId>(data.values.size());
    auto item = value(id, "quant_" + std::to_string(lane), 16U);
    item.logical_dtype = "INT8";
    item.logical_shape = sc::TensorShape{1, 1, 1, 16};
    data.values.push_back(std::move(item));
    intermediates->push_back(id);

    sc::OpSpec op;
    op.id = static_cast<sc::OpId>(data.ops.size());
    op.sequence = data.ops.size() + 1U;
    op.name = "quant_" + std::to_string(lane);
    op.kind = sc::OpKind::Quantize;
    op.processor = "EV74";
    op.inputs = {inputs[lane]};
    op.outputs = {id};
    op.input_shapes = {{1, 1, 1, 16}};
    op.output_shapes = {{1, 1, 1, 16}};
    op.config = sc::QuantizeOpConfig{"INT8", 8, "TONEAREST", {{0.25, -7}}};
    data.ops.push_back(std::move(op));
  }
  for (std::size_t lane = 0; lane < lanes; ++lane) {
    const auto id = static_cast<sc::ValueId>(data.values.size());
    auto item = value(id, "tess_" + std::to_string(lane), 16U);
    item.logical_dtype = "INT8";
    item.logical_shape = sc::TensorShape{1, 1, 1, 16};
    item.representation = sc::ValueRepresentation::Tessellated;
    data.values.push_back(std::move(item));
    final_tess->push_back(id);

    sc::OpSpec op;
    op.id = static_cast<sc::OpId>(data.ops.size());
    op.sequence = data.ops.size() + 1U;
    op.name = "tess_" + std::to_string(lane);
    op.kind = sc::OpKind::Tessellate;
    op.processor = "EV74";
    op.inputs = {(*intermediates)[lane]};
    op.outputs = {id};
    op.input_shapes = {{1, 1, 1, 16}};
    op.output_shapes = {{1, 1, 1, 16}};
    op.config = sc::TessellateOpConfig{{1, 1, 1, 16}, false, false, "INT8"};
    data.ops.push_back(std::move(op));
  }
  const auto mla_output = static_cast<sc::ValueId>(data.values.size());
  auto output = value(mla_output, "mla_output", 16U);
  output.logical_dtype = "INT8";
  output.logical_shape = sc::TensorShape{1, 1, 1, 16};
  data.values.push_back(std::move(output));
  sc::OpSpec mla;
  mla.id = static_cast<sc::OpId>(data.ops.size());
  mla.sequence = data.ops.size() + 1U;
  mla.name = "mla";
  mla.kind = sc::OpKind::Mla;
  mla.processor = "MLA";
  mla.inputs = *final_tess;
  mla.outputs = {mla_output};
  mla.config = sc::MlaOpConfig{"model.elf", 4};
  data.ops.push_back(std::move(mla));
  for (std::size_t lane = 0; lane < lanes; ++lane) {
    data.backend_ports.push_back(
        {0U, sc::BackendPortDirection::Input, lane, "ifm_" + std::to_string(lane),
         (*final_tess)[lane], 16U, 4096U,
         sc::BackendPortAlignmentAuthority::LegacyPolicy,
         sc::BackendPortAccess::ReadOnly});
  }
  data.backend_ports.push_back(
      {0U, sc::BackendPortDirection::Output, 0U, "ofm", mla_output, 16U, 4096U,
       sc::BackendPortAlignmentAuthority::LegacyPolicy,
       sc::BackendPortAccess::WriteOnly});
  data.model_outputs = {{0U, "mla_output", mla_output}};
  std::string error;
  auto plan = sc::ModelExecutionPlan::create(std::move(data), &error);
  if (!plan) {
    std::cerr << "unable to build fused arena fixture: " << error << '\n';
    std::abort();
  }
  return std::move(*plan);
}

sc::ModelExecutionPlan make_producer_direct_pack_plan(
    std::size_t lanes, std::vector<sc::ValueId>* intermediates,
    std::vector<sc::ValueId>* producer_outputs, sc::ValueId* parent_value) {
  sc::ModelExecutionPlanData data;
  data.contract_version = "arena-direct-pack-v1";
  std::vector<sc::ValueId> inputs;
  for (std::size_t lane = 0; lane < lanes; ++lane) {
    const auto id = static_cast<sc::ValueId>(data.values.size());
    auto item = value(id, "input_" + std::to_string(lane), 64U);
    item.logical_dtype = "FP32";
    item.logical_shape = sc::TensorShape{1, 1, 1, 16};
    data.values.push_back(std::move(item));
    data.model_inputs.push_back(id);
    inputs.push_back(id);
  }
  for (std::size_t lane = 0; lane < lanes; ++lane) {
    const auto id = static_cast<sc::ValueId>(data.values.size());
    auto item = value(id, "quant_" + std::to_string(lane), 16U);
    item.logical_dtype = "INT8";
    item.logical_shape = sc::TensorShape{1, 1, 1, 16};
    data.values.push_back(std::move(item));
    intermediates->push_back(id);
    sc::OpSpec quant;
    quant.id = static_cast<sc::OpId>(data.ops.size());
    quant.sequence = data.ops.size() + 1U;
    quant.name = "quant_" + std::to_string(lane);
    quant.kind = sc::OpKind::Quantize;
    quant.processor = "EV74";
    quant.inputs = {inputs[lane]};
    quant.outputs = {id};
    quant.input_shapes = {{1, 1, 1, 16}};
    quant.output_shapes = {{1, 1, 1, 16}};
    quant.config = sc::QuantizeOpConfig{"INT8", 8, "TONEAREST", {{0.25, -7}}};
    data.ops.push_back(std::move(quant));
  }
  constexpr sc::CarrierId shared_parent = 0x40000000U;
  for (std::size_t lane = 0; lane < lanes; ++lane) {
    const auto id = static_cast<sc::ValueId>(data.values.size());
    auto item = value(id, "tess_" + std::to_string(lane), 16U);
    item.logical_dtype = "INT8";
    item.logical_shape = sc::TensorShape{1, 1, 1, 16};
    item.representation = sc::ValueRepresentation::Tessellated;
    item.storage_binding = sc::StorageBinding{
        sc::StorageBindingKind::Root, shared_parent, lane * 16U, 16U, {},
        sc::StorageAccess::WriteOnly, std::nullopt};
    data.values.push_back(std::move(item));
    producer_outputs->push_back(id);
    sc::OpSpec tess;
    tess.id = static_cast<sc::OpId>(data.ops.size());
    tess.sequence = data.ops.size() + 1U;
    tess.name = "tess_" + std::to_string(lane);
    tess.kind = sc::OpKind::Tessellate;
    tess.processor = "EV74";
    tess.inputs = {(*intermediates)[lane]};
    tess.outputs = {id};
    tess.input_shapes = {{1, 1, 1, 16}};
    tess.output_shapes = {{1, 1, 1, 16}};
    tess.config = sc::TessellateOpConfig{{1, 1, 1, 16}, false, false, "INT8"};
    data.ops.push_back(std::move(tess));
  }
  *parent_value = static_cast<sc::ValueId>(data.values.size());
  auto parent = value(*parent_value, "packed_parent", lanes * 16U);
  parent.logical_dtype = "INT8";
  parent.logical_shape = sc::TensorShape{1, 1, 1,
                                         static_cast<std::int64_t>(lanes * 16U)};
  parent.representation = sc::ValueRepresentation::Packed;
  parent.storage_binding = sc::StorageBinding{
      sc::StorageBindingKind::Root, shared_parent, 0U, lanes * 16U, {},
      sc::StorageAccess::ReadWrite, std::nullopt};
  data.values.push_back(std::move(parent));
  sc::PackOpConfig pack_config;
  pack_config.batch_count = 1U;
  pack_config.parent_required_bytes = lanes * 16U;
  pack_config.materializes = false;
  for (std::size_t lane = 0; lane < lanes; ++lane) {
    pack_config.spans.push_back({(*producer_outputs)[lane], 0U, 0U, lane * 16U,
                                 16U, 16U, "none"});
  }
  sc::OpSpec pack;
  pack.id = static_cast<sc::OpId>(data.ops.size());
  pack.sequence = data.ops.size() + 1U;
  pack.name = "pack";
  pack.kind = sc::OpKind::Pack;
  pack.processor = "EV74";
  pack.inputs = *producer_outputs;
  pack.outputs = {*parent_value};
  pack.config = std::move(pack_config);
  data.ops.push_back(std::move(pack));

  const auto mla_output = static_cast<sc::ValueId>(data.values.size());
  auto output = value(mla_output, "mla_output", 16U);
  output.logical_dtype = "INT8";
  output.logical_shape = sc::TensorShape{1, 1, 1, 16};
  data.values.push_back(std::move(output));
  sc::OpSpec mla;
  mla.id = static_cast<sc::OpId>(data.ops.size());
  mla.sequence = data.ops.size() + 1U;
  mla.name = "mla";
  mla.kind = sc::OpKind::Mla;
  mla.processor = "MLA";
  mla.inputs = {*parent_value};
  mla.outputs = {mla_output};
  mla.config = sc::MlaOpConfig{"model.elf", 4};
  data.ops.push_back(std::move(mla));
  data.backend_ports = {
      {0U, sc::BackendPortDirection::Input, 0U, "ifm", *parent_value,
       lanes * 16U, 4096U, sc::BackendPortAlignmentAuthority::LegacyPolicy,
       sc::BackendPortAccess::ReadOnly},
      {0U, sc::BackendPortDirection::Output, 0U, "ofm", mla_output, 16U, 4096U,
       sc::BackendPortAlignmentAuthority::LegacyPolicy,
       sc::BackendPortAccess::WriteOnly},
  };
  data.model_outputs = {{0U, "mla_output", mla_output}};
  std::string error;
  auto plan = sc::ModelExecutionPlan::create(std::move(data), &error);
  if (!plan) {
    std::cerr << "unable to build direct-Pack fixture: " << error << '\n';
    std::abort();
  }
  return std::move(*plan);
}

bool overlaps(const sc::FrameSlotArenaRegion& lhs,
              const sc::FrameSlotArenaRegion& rhs) {
  return lhs.byte_offset < rhs.byte_offset + rhs.size_bytes &&
         rhs.byte_offset < lhs.byte_offset + lhs.size_bytes;
}

void test_no_reuse_oracle() {
  const auto execution = make_plan();
  std::string error;
  const auto plan = sc::FrameSlotArenaPlan::compile(
      execution, sc::FrameSlotArenaReuse::Disabled, 4096, &error);
  CHECK(plan.has_value());
  CHECK(error.empty());
  if (!plan) {
    return;
  }
  CHECK(plan->regions().size() == 6U);
  CHECK(plan->region(0) == nullptr);
  CHECK(plan->region(2) != nullptr);
  CHECK(plan->region(2)->byte_offset == 0U);
  CHECK(plan->region(3)->byte_offset == 4096U);
  CHECK(plan->region(4)->byte_offset == 8192U);
  CHECK(plan->region(5)->byte_offset == 12288U);
  CHECK(plan->region(6)->byte_offset == 16384U);
  CHECK(plan->region(7)->byte_offset == 24576U);
  CHECK(plan->allocation_bytes() == 32768U);
  CHECK(plan->allocation_alignment_bytes() == 8192U);
  CHECK(plan->placement().domain == sc::ArenaStorageDomain::Cma);
  CHECK(plan->placement().provenance ==
        sc::ArenaAllocationProvenance::CoreAllocated);
  CHECK(plan->placement().requires_access(sc::ArenaDeviceAccess::Ev74));
  CHECK(plan->placement().requires_access(sc::ArenaDeviceAccess::Mla));
  CHECK(plan->placement().requires_access(sc::ArenaDeviceAccess::CpuA65));
  CHECK(plan->placement().escape == sc::ArenaEscapePolicy::CpuMappablePublic);
}

void test_disjoint_lifetime_reuse() {
  const auto execution = make_plan();
  std::string error;
  const auto reused = sc::FrameSlotArenaPlan::compile(
      execution, sc::FrameSlotArenaReuse::DisjointLifetimes, 4096, &error);
  const auto oracle = sc::FrameSlotArenaPlan::compile(
      execution, sc::FrameSlotArenaReuse::Disabled, 4096, &error);
  CHECK(reused.has_value());
  CHECK(oracle.has_value());
  if (!reused || !oracle) {
    return;
  }
  CHECK(reused->allocation_bytes() < oracle->allocation_bytes());
  CHECK(reused->region(2)->lifetime.first_sequence == 1U);
  CHECK(reused->region(2)->lifetime.last_sequence == 2U);
  CHECK(reused->region(4)->lifetime.first_sequence == 2U);
  CHECK(reused->region(4)->lifetime.last_sequence == 3U);
  CHECK(reused->region(6)->lifetime.first_sequence == 3U);
  CHECK(reused->region(6)->lifetime.last_sequence == 4U);

  // Input and output of MLA overlap at sequence 2, so they may not alias.
  CHECK(!overlaps(*reused->region(2), *reused->region(4)));
  CHECK(!overlaps(*reused->region(3), *reused->region(5)));
  // The final outputs begin after the pre-MLA carriers' last use and therefore
  // deterministically reuse that dead bank.
  CHECK(overlaps(*reused->region(2), *reused->region(7)) ||
        overlaps(*reused->region(3), *reused->region(7)));

  const auto repeated = sc::FrameSlotArenaPlan::compile(
      execution, sc::FrameSlotArenaReuse::DisjointLifetimes, 4096, &error);
  CHECK(repeated.has_value());
  if (repeated) {
    CHECK(repeated->regions().size() == reused->regions().size());
    for (std::size_t i = 0; i < reused->regions().size(); ++i) {
      CHECK(repeated->regions()[i].value_id == reused->regions()[i].value_id);
      CHECK(repeated->regions()[i].byte_offset == reused->regions()[i].byte_offset);
    }
  }
}

void test_invalid_alignment_fails_closed() {
  const auto execution = make_plan();
  std::string error;
  CHECK(!sc::FrameSlotArenaPlan::compile(
      execution, sc::FrameSlotArenaReuse::DisjointLifetimes, 24, &error));
  CHECK(!error.empty());
}

void test_read_expressions_do_not_replace_real_producers() {
  const auto execution = make_staggered_read_plan();
  std::string error;
  const auto plan = sc::FrameSlotArenaPlan::compile(
      execution, sc::FrameSlotArenaReuse::DisjointLifetimes, 4096, &error);
  CHECK(plan.has_value());
  if (!plan) {
    return;
  }
  CHECK(plan->region(3) == plan->region(1));
  CHECK(plan->region(5) == plan->region(2));
  CHECK(plan->region(1)->lifetime.first_sequence == 1U);
  CHECK(plan->region(2)->lifetime.first_sequence == 1U);
  CHECK(!overlaps(*plan->region(1), *plan->region(2)));
}

void test_grouped_physical_command_keeps_every_input_and_output_disjoint() {
  const auto execution = make_staggered_read_plan();

  // The two Slice operations are relation-only.  Both Dequantize operations
  // are members of one physical CVU submission, exactly like a grouped
  // multi-OFM MLA egress.  Semantic sequence 3 would otherwise make post1 look
  // later than—and reusable with—ofm0, although the backend consumes ofm0 and
  // writes post1 in the same call.
  sc::PhysicalExecutionPlan physical;
  physical.command_for_semantic_op.resize(execution.ops().size());
  sc::PhysicalCommand mla;
  mla.id = 0;
  mla.cohort_id = 0;
  mla.topological_rank = 0;
  mla.engine = sc::PhysicalEngine::Mla;
  mla.implementation_id = "mla";
  mla.members = {{0U, {0U}, {0U}, {1U, 2U}}};
  mla.inputs = {0};
  mla.outputs = {1, 2};
  mla.successors = {1};
  physical.command_for_semantic_op[0] = 0;

  sc::PhysicalCommand post;
  post.id = 1;
  post.cohort_id = 1;
  post.topological_rank = 1;
  post.engine = sc::PhysicalEngine::Cvu;
  post.implementation_id = "cvu.graph223.dequantize.v1";
  post.graph_id = 223U;
  post.maximum_members = 32U;
  post.members = {{0U, {2U}, {3U}, {4U}}, {1U, {4U}, {5U}, {6U}}};
  post.inputs = {3, 5};
  post.outputs = {4, 6};
  post.predecessors = {0};
  physical.command_for_semantic_op[2] = 1;
  physical.command_for_semantic_op[4] = 1;
  physical.commands = {std::move(mla), std::move(post)};

  std::string error;
  const auto semantic = sc::FrameSlotArenaPlan::compile(
      execution, sc::FrameSlotArenaReuse::DisjointLifetimes, 4096, &error);
  CHECK(semantic.has_value());
  CHECK(semantic && overlaps(*semantic->region(1), *semantic->region(6)));

  const auto arena = sc::FrameSlotArenaPlan::compile(
      execution, physical, sc::FrameSlotArenaReuse::DisjointLifetimes,
      4096, &error);
  CHECK(arena.has_value());
  CHECK(error.empty());
  if (!arena) {
    return;
  }
  for (const auto input : {1U, 2U}) {
    for (const auto output : {4U, 6U}) {
      CHECK(!overlaps(*arena->region(input), *arena->region(output)));
    }
  }
  CHECK(!overlaps(*arena->region(4), *arena->region(6)));

  const auto cma_with_ev = sc::FrameSlotArenaPlan::compile(
      execution, physical, sc::FrameSlotArenaReuse::DisjointLifetimes,
      4096, &error, sc::ArenaDmsPolicy::PreferDmsForEligible);
  CHECK(cma_with_ev.has_value());
  CHECK(cma_with_ev &&
        cma_with_ev->placement().domain == sc::ArenaStorageDomain::Cma);

  auto a65_physical = physical;
  a65_physical.commands[1].engine = sc::PhysicalEngine::A65;
  const auto dms = sc::FrameSlotArenaPlan::compile(
      execution, a65_physical, sc::FrameSlotArenaReuse::DisjointLifetimes,
      4096, &error, sc::ArenaDmsPolicy::PreferDmsForEligible);
  CHECK(dms.has_value());
  if (dms) {
    CHECK(dms->placement().domain == sc::ArenaStorageDomain::Dms);
    CHECK(dms->placement().requires_access(sc::ArenaDeviceAccess::Mla));
    CHECK(dms->placement().requires_access(sc::ArenaDeviceAccess::CpuA65));
    CHECK(!dms->placement().requires_access(sc::ArenaDeviceAccess::Ev74));
  }
}

void test_resolved_a65_mla_a65_route_selects_one_dms_arena() {
  sc::ModelExecutionPlanData data;
  data.contract_version = "arena-a65-mla-a65-dms-v1";
  data.values = {
      value(0, "public_input", 64),
      value(1, "a65_pre_output", 64),
      value(2, "mla_output", 64),
      value(3, "a65_post_output", 64),
  };
  data.model_inputs = {0};

  sc::OpSpec pre;
  pre.id = 0;
  pre.sequence = 1;
  pre.name = "a65_pre";
  pre.kind = sc::OpKind::Cast;
  pre.processor = "A65";
  pre.inputs = {0};
  pre.outputs = {1};
  pre.config = sc::CastOpConfig{"INT8"};

  sc::OpSpec mla;
  mla.id = 1;
  mla.sequence = 2;
  mla.name = "mla";
  mla.kind = sc::OpKind::Mla;
  mla.processor = "MLA";
  mla.inputs = {1};
  mla.outputs = {2};
  mla.config = sc::MlaOpConfig{"model.elf", 4};

  sc::OpSpec post;
  post.id = 2;
  post.sequence = 3;
  post.name = "a65_post";
  post.kind = sc::OpKind::Cast;
  post.processor = "A65";
  post.inputs = {2};
  post.outputs = {3};
  post.config = sc::CastOpConfig{"FP32"};
  data.ops = {std::move(pre), std::move(mla), std::move(post)};
  data.backend_ports = {
      {0, sc::BackendPortDirection::Input, 0, "data.ifm.b0", 1, 64, 4096,
       sc::BackendPortAlignmentAuthority::Contract,
       sc::BackendPortAccess::ReadOnly},
      {0, sc::BackendPortDirection::Output, 0, "data.ofm.b0", 2, 64, 4096,
       sc::BackendPortAlignmentAuthority::Contract,
       sc::BackendPortAccess::WriteOnly},
  };
  data.model_outputs = {{0, "public_output", 3}};

  std::string error;
  auto execution = sc::ModelExecutionPlan::create(std::move(data), &error);
  CHECK(execution.has_value());
  if (!execution) {
    return;
  }

  sc::PhysicalExecutionPlan physical;
  physical.command_for_semantic_op = {0U, 1U, 2U};
  physical.commands = {
      sc::PhysicalCommand{
          .id = 0U,
          .topological_rank = 0U,
          .engine = sc::PhysicalEngine::A65,
          .implementation_id = "a65.prepared.pre",
          .members = {{0U, {0U}, {0U}, {1U}}},
          .inputs = {0U},
          .outputs = {1U},
          .successors = {1U},
      },
      sc::PhysicalCommand{
          .id = 1U,
          .topological_rank = 1U,
          .engine = sc::PhysicalEngine::Mla,
          .implementation_id = "mla.model.elf",
          .members = {{0U, {1U}, {1U}, {2U}}},
          .inputs = {1U},
          .outputs = {2U},
          .predecessors = {0U},
          .successors = {2U},
      },
      sc::PhysicalCommand{
          .id = 2U,
          .topological_rank = 2U,
          .engine = sc::PhysicalEngine::A65,
          .implementation_id = "a65.prepared.post",
          .members = {{0U, {2U}, {2U}, {3U}}},
          .inputs = {2U},
          .outputs = {3U},
          .predecessors = {1U},
      },
  };

  const auto arena = sc::FrameSlotArenaPlan::compile(
      *execution, physical, sc::FrameSlotArenaReuse::Disabled, 4096U, &error,
      sc::ArenaDmsPolicy::PreferDmsForEligible);
  CHECK(arena.has_value());
  CHECK(error.empty());
  if (!arena) {
    return;
  }
  CHECK(arena->placement().domain == sc::ArenaStorageDomain::Dms);
  CHECK(arena->placement().requires_access(sc::ArenaDeviceAccess::CpuA65));
  CHECK(arena->placement().requires_access(sc::ArenaDeviceAccess::Mla));
  CHECK(!arena->placement().requires_access(sc::ArenaDeviceAccess::Ev74));
  CHECK(arena->placement().provenance ==
        sc::ArenaAllocationProvenance::CoreAllocated);
  CHECK(arena->region(1) != nullptr);
  CHECK(arena->region(2) != nullptr);
  CHECK(arena->region(3) != nullptr);
}

void test_multiple_values_share_one_authored_carrier() {
  sc::ModelExecutionPlanData data;
  data.contract_version = "carrier-test-v1";
  data.values = {
      value(0, "input", 64),
      value(1, "left", 64),
      value(2, "right", 64),
      value(3, "output", 64),
  };
  data.model_inputs = {0};
  data.carriers = {
      {0, 64, 16, sc::ValueRepresentation::Dense},
      {10, 128, 16, sc::ValueRepresentation::Packed},
      {11, 64, 16, sc::ValueRepresentation::Dense},
  };
  data.values[0].storage_binding = sc::StorageBinding{
      sc::StorageBindingKind::External, 0, 0, 64, {}, sc::StorageAccess::ReadOnly, std::nullopt};
  data.values[1].storage_binding = sc::StorageBinding{
      sc::StorageBindingKind::Root, 10, 0, 64, {}, sc::StorageAccess::WriteOnly, std::nullopt};
  data.values[2].storage_binding = sc::StorageBinding{
      sc::StorageBindingKind::Root, 10, 64, 64, {}, sc::StorageAccess::WriteOnly, std::nullopt};
  data.values[3].storage_binding = sc::StorageBinding{
      sc::StorageBindingKind::Root, 11, 0, 64, {}, sc::StorageAccess::ReadWrite, std::nullopt};

  sc::OpSpec producer;
  producer.id = 0;
  producer.sequence = 1;
  producer.name = "direct-producers";
  producer.kind = sc::OpKind::Quantize;
  producer.processor = "EV74";
  producer.inputs = {0};
  producer.outputs = {1, 2};
  producer.config = sc::QuantizeOpConfig{};

  sc::OpSpec consumer;
  consumer.id = 1;
  consumer.sequence = 2;
  consumer.name = "consumer";
  consumer.kind = sc::OpKind::Dequantize;
  consumer.processor = "EV74";
  consumer.inputs = {1, 2};
  consumer.outputs = {3};
  consumer.config = sc::DequantizeOpConfig{};
  data.ops = {std::move(producer), std::move(consumer)};
  data.model_outputs = {{0, "output", 3}};

  std::string error;
  auto execution = sc::ModelExecutionPlan::create(std::move(data), &error);
  CHECK(execution.has_value());
  if (!execution) {
    return;
  }
  const auto arena = sc::FrameSlotArenaPlan::compile(
      *execution, sc::FrameSlotArenaReuse::Disabled, 16, &error);
  CHECK(arena.has_value());
  if (!arena) {
    return;
  }
  CHECK(arena->regions().size() == 2U);
  CHECK(arena->region(1) == arena->region(2));
  CHECK(arena->region(1)->carrier_id == 10U);
  CHECK(arena->region(1)->size_bytes == 128U);
}

void test_fused_intermediates_are_not_materialized_at_capacity_boundaries() {
  for (const auto lanes : {1U, 32U, 33U}) {
    std::vector<sc::ValueId> intermediates;
    std::vector<sc::ValueId> final_tess;
    const auto execution =
        make_fused_quanttess_arena_plan(lanes, &intermediates, &final_tess);
    std::string error;
    const auto physical = sc::PhysicalExecutionLowerer::lower(execution, &error);
    CHECK(physical.has_value());
    if (!physical) {
      continue;
    }
    const auto arena = sc::FrameSlotArenaPlan::compile(
        execution, *physical, sc::FrameSlotArenaReuse::Disabled, 4096U, &error);
    CHECK(arena.has_value());
    CHECK(error.empty());
    if (!arena) {
      continue;
    }
    for (const auto value_id : intermediates) {
      CHECK(arena->region(value_id) == nullptr);
    }
    for (const auto value_id : final_tess) {
      CHECK(arena->region(value_id) != nullptr);
    }
    // N final tessellation carriers plus one MLA output. Capacity chunking
    // changes submissions, never storage or final placement.
    CHECK(arena->regions().size() == lanes + 1U);
  }
}

void test_producer_direct_pack_preserves_every_lane_and_parent_offset() {
  for (const auto lanes : {3U, 33U}) {
    std::vector<sc::ValueId> intermediates;
    std::vector<sc::ValueId> producer_outputs;
    sc::ValueId parent = 0U;
    const auto execution = make_producer_direct_pack_plan(
        lanes, &intermediates, &producer_outputs, &parent);
    std::string error;
    const auto physical = sc::PhysicalExecutionLowerer::lower(execution, &error);
    CHECK(physical.has_value());
    if (!physical) {
      continue;
    }
    std::vector<const sc::PhysicalCommand*> fused;
    for (const auto& command : physical->commands) {
      if (command.graph_id == 226U) {
        fused.push_back(&command);
      }
    }
    CHECK(fused.size() == (lanes + 31U) / 32U);
    std::size_t member_count = 0U;
    for (const auto* command : fused) {
      member_count += command->members.size();
    }
    CHECK(member_count == lanes);
    CHECK(execution.value(producer_outputs.back())->storage_binding->byte_offset ==
          (lanes - 1U) * 16U);

    const auto arena = sc::FrameSlotArenaPlan::compile(
        execution, *physical, sc::FrameSlotArenaReuse::Disabled, 4096U, &error);
    CHECK(arena.has_value());
    if (!arena) {
      continue;
    }
    for (const auto intermediate : intermediates) {
      CHECK(arena->region(intermediate) == nullptr);
    }
    const auto* parent_region = arena->region(parent);
    CHECK(parent_region != nullptr);
    for (const auto output : producer_outputs) {
      CHECK(arena->region(output) == parent_region);
    }
    // One packed parent plus one MLA output; children are disjoint spans, not
    // N independently allocated carriers.
    CHECK(arena->regions().size() == 2U);
  }
}

} // namespace

int main() {
  test_no_reuse_oracle();
  test_disjoint_lifetime_reuse();
  test_invalid_alignment_fails_closed();
  test_read_expressions_do_not_replace_real_producers();
  test_grouped_physical_command_keeps_every_input_and_output_disjoint();
  test_resolved_a65_mla_a65_route_selects_one_dms_arena();
  test_multiple_values_share_one_authored_carrier();
  test_fused_intermediates_are_not_materialized_at_capacity_boundaries();
  test_producer_direct_pack_preserves_every_lane_and_parent_offset();
  return failures == 0 ? 0 : 1;
}
