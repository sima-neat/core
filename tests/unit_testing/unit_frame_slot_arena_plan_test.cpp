#define SIMA_NEAT_INTERNAL 1
#include "pipeline/internal/sima/static_contract/FrameSlotArenaPlan.h"

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
  CHECK(plan->region(3) == nullptr);
  CHECK(plan->region(5) == nullptr);
  CHECK(plan->region(1)->lifetime.first_sequence == 1U);
  CHECK(plan->region(2)->lifetime.first_sequence == 1U);
  CHECK(!overlaps(*plan->region(1), *plan->region(2)));
}

} // namespace

int main() {
  test_no_reuse_oracle();
  test_disjoint_lifetime_reuse();
  test_invalid_alignment_fails_closed();
  test_read_expressions_do_not_replace_real_producers();
  return failures == 0 ? 0 : 1;
}
