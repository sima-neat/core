#include "pipeline/internal/sima/InternalEdgeContractResolver.h"
#include "pipeline/internal/sima/PreparedRuntimeBuild.h"
#include "pipeline/internal/sima/StaticSpecBuilders.h"
#include "pipeline/internal/sima/TensorSemanticsUtil.h"

#include <stdexcept>
#include <cstdint>
#include <gst/gst.h>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace simaai::neat::pipeline_internal::sima;
using namespace simaai::neat::pipeline_internal::sima::edgecontract;
using namespace simaai::neat::pipeline_internal::sima::specbuilders;

namespace {
void check(bool condition, const char* expression) {
  if (!condition)
    throw std::runtime_error(expression);
}
#define CHECK(...) check((__VA_ARGS__), #__VA_ARGS__)

PhysicalBufferStaticSpec physical(int index, std::uint64_t size, const std::string& segment) {
  return build_physical_buffer_static_spec(index, index, size, DeviceKind::Evxx, segment);
}

LogicalTensorStaticSpec logical_output(int logical_index, int physical_index, int slot,
                                       std::vector<std::int64_t> shape, const std::string& dtype,
                                       const std::string& segment, std::uint64_t size_bytes = 0U) {
  return build_logical_output_static_spec(logical_index, logical_index, physical_index, slot,
                                          logical_index, shape, dtype, "HWC", segment, segment,
                                          segment, 0, size_bytes);
}

LogicalInputStaticSpec
logical_input(int logical_index, int physical_index, std::vector<std::int64_t> shape,
              const std::string& dtype, const std::string& segment, std::int64_t byte_offset = 0,
              std::uint64_t size_bytes = 0U,
              TensorMaterializationKind materialization = TensorMaterializationKind::Direct) {
  return build_logical_input_static_spec(logical_index, logical_index, physical_index, shape, dtype,
                                         "HWC", segment, segment, segment, byte_offset, size_bytes,
                                         materialization);
}

InputBindingStaticSpec
binding_to_producer(int producer_index, const std::string& producer_id, int logical_output_index,
                    int output_slot, int physical_output_index, const std::string& source_segment,
                    std::uint64_t source_size, std::int64_t source_byte_offset = 0) {
  auto binding = build_input_binding_static_spec(
      0, 0, "consumer_input", source_segment, logical_output_index, output_slot,
      physical_output_index, source_size, source_byte_offset, true);
  binding.src_stage_index = producer_index;
  binding.src_stage_id = producer_id;
  return binding;
}

sima_ev_tensor_desc dense_desc(std::vector<int> shape, const std::string& dtype,
                               const std::string& layout) {
  sima_ev_tensor_desc desc{};
  std::string error;
  CHECK(tensorsemantics::build_dense_tensor_desc(shape, dtype, layout, &desc, &error,
                                                 "missing output", "bad rank", "bad dim",
                                                 "bad dtype", "bad stride"));
  return desc;
}

void ensure_gst_ready() {
  static bool ready = false;
  if (ready) {
    return;
  }
  int argc = 0;
  char** argv = nullptr;
  gst_init(&argc, &argv);
  ready = true;
}

SimaPluginStaticManifest direct_manifest() {
  SimaPluginStaticManifest manifest;

  StageStaticSpec producer;
  producer.logical_stage_id = "producer";
  producer.payload_kind = StagePayloadKind::ProcessMla;
  producer.physical_outputs.push_back(physical(0, 16U, "producer_out"));
  producer.logical_outputs.push_back(logical_output(0, 0, 0, {16}, "UINT8", "producer_out", 16U));
  manifest.stages.push_back(std::move(producer));

  StageStaticSpec consumer;
  consumer.logical_stage_id = "consumer";
  consumer.payload_kind = StagePayloadKind::ProcessCvu;
  consumer.physical_inputs.push_back(physical(0, 16U, "consumer_input"));
  consumer.logical_inputs.push_back(logical_input(0, 0, {16}, "UINT8", "consumer_input", 0, 16U));
  consumer.input_bindings.push_back(
      binding_to_producer(0, "producer", 0, 0, 0, "producer_out", 16U));
  manifest.stages.push_back(std::move(consumer));

  return manifest;
}

void direct_edge_resolves_without_view_contract() {
  const auto manifest = direct_manifest();
  std::string error;
  const auto resolved = resolve_edge_contract_for_binding(manifest, 1U, 0U, &error);

  CHECK(resolved.has_value());
  CHECK(error.empty());
  CHECK(resolved->producer_stage_index == 0U);
  CHECK(resolved->consumer_stage_index == 1U);
  CHECK(resolved->binding_index == 0U);
  CHECK(resolved->producer_stage->logical_stage_id == "producer");
  CHECK(resolved->consumer_stage->logical_stage_id == "consumer");
  CHECK(resolved->producer_logical_output->logical_index == 0);
  CHECK(resolved->consumer_logical_input->logical_index == 0);
  CHECK(resolved->producer_physical_output->physical_index == 0);
  CHECK(!resolved->consumer_requires_view_contract);
}

void slice_view_edge_resolves_as_consumer_side_view_contract() {
  auto manifest = direct_manifest();
  auto& consumer = manifest.stages[1];
  consumer.physical_inputs.clear();
  consumer.logical_inputs.clear();
  consumer.input_bindings.clear();
  consumer.physical_inputs.push_back(physical(0, 4U, "slice_view"));
  consumer.logical_inputs.push_back(logical_input(0, 0, {4}, "UINT8", "slice_view", 4, 4U,
                                                  TensorMaterializationKind::OffsetView));
  consumer.input_bindings.push_back(
      binding_to_producer(0, "producer", 0, 0, 0, "producer_out", 16U, 4));

  std::string error;
  const auto resolved = resolve_edge_contract_for_binding(manifest, 1U, 0U, &error);

  CHECK(resolved.has_value());
  CHECK(error.empty());
  CHECK(resolved->producer_logical_output->shape == std::vector<std::int64_t>{16});
  CHECK(resolved->consumer_logical_input->shape == std::vector<std::int64_t>{4});
  CHECK(resolved->binding->src_physical_byte_offset == 4);
  CHECK(resolved->consumer_requires_view_contract);
}

void branched_multi_output_edge_selects_requested_output_slot() {
  SimaPluginStaticManifest manifest;

  StageStaticSpec producer;
  producer.logical_stage_id = "branching_producer";
  producer.payload_kind = StagePayloadKind::ProcessCvu;
  producer.physical_outputs.push_back(physical(0, 4U, "head0"));
  producer.physical_outputs.push_back(physical(1, 8U, "head1"));
  producer.logical_outputs.push_back(logical_output(0, 0, 0, {4}, "UINT8", "head0", 4U));
  producer.logical_outputs.push_back(logical_output(1, 1, 1, {2}, "INT32", "head1", 8U));
  producer.output_order.push_back(build_output_route_static_spec(0, 0, 0, "head0", "head0"));
  producer.output_order.push_back(build_output_route_static_spec(1, 1, 1, "head1", "head1"));
  manifest.stages.push_back(std::move(producer));

  StageStaticSpec consumer;
  consumer.logical_stage_id = "consumer";
  consumer.payload_kind = StagePayloadKind::ProcessCvu;
  consumer.physical_inputs.push_back(physical(0, 8U, "consumer_head1"));
  consumer.logical_inputs.push_back(logical_input(0, 0, {2}, "INT32", "consumer_head1", 0, 8U));
  consumer.input_bindings.push_back(
      binding_to_producer(0, "branching_producer", 1, 1, 1, "head1", 8U));
  manifest.stages.push_back(std::move(consumer));

  std::string error;
  const auto resolved = resolve_edge_contract_for_binding(manifest, 1U, 0U, &error);

  CHECK(resolved.has_value());
  CHECK(error.empty());
  CHECK(resolved->producer_logical_output->logical_index == 1);
  CHECK(resolved->producer_logical_output->output_slot == 1);
  CHECK(resolved->producer_physical_output->physical_index == 1);
  CHECK(resolved->producer_physical_output->segment_name == "head1");
  CHECK(!resolved->consumer_requires_view_contract);
}

void missing_producer_fails_closed_with_error() {
  auto manifest = direct_manifest();
  auto& binding = manifest.stages[1].input_bindings[0];
  binding.src_stage_index = 42;
  binding.src_stage_id = "missing_producer";
  binding.source_segment_name = "missing_output";
  binding.src_logical_output_index = 7;
  binding.src_output_slot = 7;
  binding.src_physical_output_index = 7;

  std::string error;
  const auto resolved = resolve_edge_contract_for_binding(manifest, 1U, 0U, &error);

  CHECK(!resolved.has_value());
  CHECK(!error.empty());
}

void all_consumer_edges_resolve_together() {
  const auto manifest = direct_manifest();
  std::string error;
  const auto resolved = resolve_consumer_edge_contracts(manifest, 1U, &error);

  CHECK(error.empty());
  CHECK(resolved.size() == 1U);
  CHECK(resolved[0].producer_stage_index == 0U);
  CHECK(resolved[0].binding_index == 0U);
}

SimaPluginStaticManifest processcvu_routing_manifest(InputBindingStaticSpec binding,
                                                     bool view_input) {
  SimaPluginStaticManifest manifest;

  StageStaticSpec producer;
  producer.logical_stage_id = "branching_producer";
  producer.physical_outputs.push_back(physical(0, 4U, "head0"));
  producer.physical_outputs.push_back(physical(1, 16U, "head1"));
  producer.logical_outputs.push_back(logical_output(0, 0, 0, {1, 1, 1}, "UINT8", "head0", 1U));
  producer.logical_outputs.push_back(logical_output(1, 1, 1, {1, 4, 1}, "INT32", "head1", 16U));
  producer.output_order.push_back(build_output_route_static_spec(0, 0, 0, "head0", "head0"));
  producer.output_order.push_back(build_output_route_static_spec(1, 1, 1, "head1", "head1"));
  manifest.stages.push_back(std::move(producer));

  StageStaticSpec consumer;
  consumer.logical_stage_id = "consumer_cvu";
  consumer.payload_kind = StagePayloadKind::ProcessCvu;
  consumer.processcvu.canonical_contract = true;
  consumer.processcvu.graph_family = "cast";
  consumer.processcvu.graph_name = "cast";
  consumer.processcvu.default_input_name = "consumer_input";
  consumer.processcvu.default_output_names = {"consumer_out"};
  consumer.processcvu.primary_output_name = "consumer_out";
  consumer.processcvu.input_dtype = "INT32";
  consumer.processcvu.output_dtype = "INT32";
  consumer.processcvu.out_dtype = "INT32";
  const auto input_shape = view_input ? std::vector<int>{1, 1, 1} : std::vector<int>{1, 4, 1};
  consumer.processcvu.input_tensors = {dense_desc(input_shape, "INT32", "HWC")};
  consumer.processcvu.output_tensors = {dense_desc(input_shape, "INT32", "HWC")};
  const std::uint64_t input_size = view_input ? 4U : 16U;
  const std::int64_t input_offset = view_input ? 4 : 0;
  consumer.physical_inputs.push_back(physical(0, input_size, "consumer_head1"));
  consumer.logical_inputs.push_back(logical_input(
      0, 0, {input_shape[0], input_shape[1], input_shape[2]}, "INT32", "consumer_head1",
      input_offset, input_size,
      view_input ? TensorMaterializationKind::OffsetView : TensorMaterializationKind::Direct));
  consumer.input_bindings.push_back(std::move(binding));
  consumer.physical_outputs.push_back(physical(0, input_size, "consumer_out"));
  consumer.logical_outputs.push_back(
      logical_output(0, 0, 0, {input_shape[0], input_shape[1], input_shape[2]}, "INT32",
                     "consumer_out", input_size));
  consumer.output_order.push_back(
      build_output_route_static_spec(0, 0, 0, "consumer_out", "consumer_out"));
  manifest.stages.push_back(std::move(consumer));

  return manifest;
}

struct CompletedInput {
  InputBindingStaticSpec binding;
  std::vector<std::int64_t> shape;
  std::int64_t physical_base = 0;
};
CompletedInput single_completed_kernel_input_binding(SimaPluginStaticManifest manifest) {
  ensure_gst_ready();
  const auto failure = complete_kernel_input_routes(&manifest, 1);
  CHECK(!failure);
  const auto& stage = manifest.stages[1];
  CHECK(stage.input_bindings.size() == 1U);
  CHECK(stage.logical_inputs.size() == 1U);
  // Route normalization is deliberately separate from numerical admission:
  // explicit contradictory overrides below are metadata precedence fixtures.
  // They must remain explicit, not be silently rewritten into executable views.
  return {stage.input_bindings.front(), stage.logical_inputs.front().shape,
          stage.physical_inputs.front().source_byte_offset};
}

void canonical_route_completion_processcvu_routing_uses_manifest_edge_resolution() {
  const auto input_binding = single_completed_kernel_input_binding(processcvu_routing_manifest(
      binding_to_producer(0, "branching_producer", 1, 1, 1, "head1", 16U), false));

  CHECK(input_binding.binding.src_logical_output_index == 1);
  CHECK(input_binding.binding.src_output_slot == 1);
  CHECK(input_binding.binding.src_physical_output_index == 1);
  CHECK(input_binding.binding.src_physical_size_bytes == 16U);
  CHECK(input_binding.binding.src_physical_byte_offset == 0);
  CHECK(input_binding.binding.source_segment_name == "head1");
  CHECK(input_binding.shape == std::vector<std::int64_t>({1, 4, 1}));
}

void canonical_route_completion_fills_missing_source_fields_from_resolved_edge() {
  const auto input_binding = single_completed_kernel_input_binding(processcvu_routing_manifest(
      binding_to_producer(0, "branching_producer", -1, 1, -1, "", 0U), false));

  CHECK(input_binding.binding.src_logical_output_index == 1);
  CHECK(input_binding.binding.src_output_slot == 1);
  CHECK(input_binding.binding.src_physical_output_index == 1);
  CHECK(input_binding.binding.src_physical_size_bytes == 16U);
  CHECK(input_binding.binding.src_physical_byte_offset == 0);
  CHECK(input_binding.binding.source_segment_name == "head1");
}

void canonical_route_completion_keeps_explicit_source_fields_when_edge_resolves() {
  const auto input_binding = single_completed_kernel_input_binding(processcvu_routing_manifest(
      binding_to_producer(0, "branching_producer", 7, 1, 9, "explicit_parent", 123U, 5), false));

  CHECK(input_binding.binding.src_logical_output_index == 7);
  CHECK(input_binding.binding.src_output_slot == 1);
  CHECK(input_binding.binding.src_physical_output_index == 9);
  CHECK(input_binding.binding.src_physical_size_bytes == 123U);
  CHECK(input_binding.binding.src_physical_byte_offset == 5);
  CHECK(input_binding.binding.source_segment_name == "explicit_parent");
}

void canonical_route_completion_keeps_explicit_zero_offset_when_source_fields_are_complete() {
  const auto input_binding = single_completed_kernel_input_binding(processcvu_routing_manifest(
      binding_to_producer(0, "branching_producer", 1, 1, 1, "explicit_parent", 16U, 0), true));

  CHECK(input_binding.binding.src_logical_output_index == 1);
  CHECK(input_binding.binding.src_output_slot == 1);
  CHECK(input_binding.binding.src_physical_output_index == 1);
  CHECK(input_binding.binding.src_physical_size_bytes == 16U);
  CHECK(input_binding.binding.src_physical_byte_offset == 0);
  CHECK(input_binding.binding.source_segment_name == "explicit_parent");
}

void canonical_route_completion_view_edge_fills_source_offset_from_consumer_view() {
  const auto input_binding = single_completed_kernel_input_binding(processcvu_routing_manifest(
      binding_to_producer(0, "branching_producer", -1, 1, -1, "", 0U), true));

  CHECK(input_binding.binding.src_logical_output_index == 1);
  CHECK(input_binding.binding.src_output_slot == 1);
  CHECK(input_binding.binding.src_physical_output_index == 1);
  CHECK(input_binding.binding.src_physical_size_bytes == 16U);
  CHECK(input_binding.binding.src_physical_byte_offset == 4);
  CHECK(input_binding.binding.source_segment_name == "head1");
  CHECK(input_binding.shape == std::vector<std::int64_t>({1, 1, 1}));
}

void canonical_route_completion_keeps_physical_base_separate() {
  auto manifest = processcvu_routing_manifest(
      binding_to_producer(0, "branching_producer", -1, 1, -1, "", 0U), false);
  manifest.stages[1].physical_inputs.front().source_byte_offset = 64;
  const auto input_binding = single_completed_kernel_input_binding(std::move(manifest));
  CHECK(input_binding.binding.src_physical_output_index == 1);
  CHECK(input_binding.binding.src_physical_size_bytes == 16);
  CHECK(input_binding.binding.src_physical_byte_offset == 0);
  CHECK(input_binding.physical_base == 64);
  CHECK(input_binding.physical_base + input_binding.binding.src_physical_byte_offset == 64);
}

void canonical_route_completion_fixed_transform_uses_complete_manifest_without_graph_overlay() {
  auto manifest = processcvu_routing_manifest(
      binding_to_producer(0, "branching_producer", 1, 1, 1, "head1", 16U), false);
  auto& transform = manifest.stages[1];
  transform.logical_stage_id = "tessellate_stage";
  transform.processcvu.graph_family = "tessellate";
  transform.processcvu.graph_name = "tessellate";

  // A physical execution stage is already a complete, Core-authored contract.
  // It must not be coupled back to an MPK graph-node name (which can differ
  // after cohorting/name sanitization) merely to prepare the fixed CVU op.
  const auto input_binding = single_completed_kernel_input_binding(std::move(manifest));
  CHECK(input_binding.binding.src_logical_output_index == 1);
  CHECK(input_binding.binding.src_physical_output_index == 1);
  CHECK(input_binding.binding.source_segment_name == "head1");
}

} // namespace

int main() {
  direct_edge_resolves_without_view_contract();
  slice_view_edge_resolves_as_consumer_side_view_contract();
  branched_multi_output_edge_selects_requested_output_slot();
  missing_producer_fails_closed_with_error();
  all_consumer_edges_resolve_together();
  canonical_route_completion_processcvu_routing_uses_manifest_edge_resolution();
  canonical_route_completion_fills_missing_source_fields_from_resolved_edge();
  canonical_route_completion_keeps_explicit_source_fields_when_edge_resolves();
  canonical_route_completion_keeps_explicit_zero_offset_when_source_fields_are_complete();
  canonical_route_completion_view_edge_fills_source_offset_from_consumer_view();
  canonical_route_completion_keeps_physical_base_separate();
  canonical_route_completion_fixed_transform_uses_complete_manifest_without_graph_overlay();
  return 0;
}
