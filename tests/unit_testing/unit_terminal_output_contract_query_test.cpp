#include "pipeline/internal/TerminalOutputContractQuery.h"

#include <cassert>
#include <cstdint>
#include <vector>

using namespace simaai::neat;
using namespace simaai::neat::pipeline_internal;
using namespace simaai::neat::pipeline_internal::terminal_output_contract;

namespace {

sima::PhysicalBufferStaticSpec physical(int index, std::uint64_t size, const char* name) {
  sima::PhysicalBufferStaticSpec out;
  out.physical_index = index;
  out.allocator_index = index;
  out.source_physical_index = index;
  out.size_bytes = size;
  out.segment_name = name ? name : "";
  return out;
}

sima::LogicalTensorStaticSpec logical(int index, int physical_index, std::vector<std::int64_t> shape,
                                      std::vector<std::int64_t> strides, const char* dtype,
                                      const char* name, const char* segment,
                                      std::uint64_t size_bytes = 0,
                                      sima::DTypeSource dtype_source =
                                          sima::DTypeSource::InternalContract) {
  sima::LogicalTensorStaticSpec out;
  out.logical_index = index;
  out.backend_output_index = index;
  out.physical_index = physical_index;
  out.output_slot = index;
  out.tensor_index = index;
  out.byte_offset = 0;
  out.size_bytes = size_bytes;
  out.shape = std::move(shape);
  out.stride_bytes = std::move(strides);
  out.dtype = dtype ? dtype : "";
  out.dtype_source = dtype_source;
  out.layout = "HWC";
  out.logical_name = name ? name : "";
  out.backend_name = out.logical_name;
  out.segment_name = segment ? segment : "";
  return out;
}

void normal_terminal_logical_contract_is_preserved() {
  sima::SimaPluginStaticManifest manifest;
  sima::StageStaticSpec stage;
  stage.logical_stage_id = "terminal_ev74";
  stage.payload_kind = sima::StagePayloadKind::ProcessCvu;
  stage.processcvu.graph_family_enum = sima::ProcessCvuGraphFamily::Cast;
  stage.physical_outputs.push_back(physical(0, 48, "scores"));
  stage.logical_outputs.push_back(logical(0, 0, {3, 4}, {16, 4}, "FP32", "scores", "scores", 48));
  manifest.stages.push_back(stage);

  std::string err;
  auto override = build_output_override_from_manifest(manifest, {}, &err);
  assert(override.has_value() && err.empty());
  assert(override->outputs.size() == 1U);
  assert(override->outputs[0].dtype == TensorDType::Float32);
  assert((override->outputs[0].shape == std::vector<std::int64_t>{3, 4}));
  assert((override->outputs[0].strides_bytes == std::vector<std::int64_t>{16, 4}));
  assert(override->outputs[0].memory_index == 0);
  assert(override->outputs[0].name == "scores");
}

void terminal_mla_with_downstream_slice_metadata_falls_back_to_raw_physical() {
  sima::SimaPluginStaticManifest manifest;
  sima::StageStaticSpec stage;
  stage.logical_stage_id = "MLA_0";
  stage.element_name = "simaaiprocessmla_1";
  stage.payload_kind = sima::StagePayloadKind::ProcessMla;
  stage.processmla.dispatcher_output_sizes = {12582912U};
  stage.physical_outputs.push_back(physical(0, 12582912U, "MLA_0"));
  stage.logical_outputs.push_back(logical(0, 0, {768, 1024, 1}, {4096, 4, 4}, "FP32",
                                          "slice_MLA_0_slice_transform", "MLA_0", 3145728U));
  manifest.stages.push_back(stage);

  std::string err;
  auto override = build_output_override_from_manifest(manifest, {}, &err);
  assert(override.has_value() && err.empty());
  assert(override->outputs.size() == 1U);
  assert(override->outputs[0].dtype == TensorDType::UInt8);
  assert((override->outputs[0].shape == std::vector<std::int64_t>{12582912}));
  assert((override->outputs[0].strides_bytes == std::vector<std::int64_t>{1}));
  assert(override->outputs[0].memory_index == 0);
  assert(override->outputs[0].name == "MLA_0");
}

void terminal_raw_fallback_drops_boundary_view_physical_name() {
  sima::SimaPluginStaticManifest manifest;
  sima::StageStaticSpec stage;
  stage.logical_stage_id = "MLA_0";
  stage.element_name = "simaaiprocessmla_1";
  stage.payload_kind = sima::StagePayloadKind::ProcessMla;
  stage.processmla.dispatcher_output_sizes = {12582912U};
  stage.physical_outputs.push_back(physical(0, 12582912U, "slice_MLA_0_slice_transform"));
  stage.logical_outputs.push_back(logical(0, 0, {768, 1024, 1}, {4096, 4, 4}, "FP32",
                                          "slice_MLA_0_slice_transform",
                                          "slice_MLA_0_slice_transform", 3145728U));
  manifest.stages.push_back(stage);

  std::string err;
  auto override = build_output_override_from_manifest(manifest, {}, &err);
  assert(override.has_value() && err.empty());
  assert(override->outputs.size() == 1U);
  assert(override->outputs[0].dtype == TensorDType::UInt8);
  assert((override->outputs[0].shape == std::vector<std::int64_t>{12582912}));
  assert(override->outputs[0].name == "MLA_0");
  assert(override->outputs[0].segment_name == "MLA_0");
}

void terminal_raw_fallback_keeps_non_view_dispatcher_name() {
  sima::SimaPluginStaticManifest manifest;
  sima::StageStaticSpec stage;
  stage.logical_stage_id = "MLA_0";
  stage.payload_kind = sima::StagePayloadKind::ProcessMla;
  stage.processmla.dispatcher_output_sizes = {64U};
  stage.processmla.dispatcher_output_names = {"raw_dispatcher_ofm"};
  stage.physical_outputs.push_back(physical(0, 64U, "slice_MLA_0_slice_transform"));
  stage.logical_outputs.push_back(logical(0, 0, {4, 4}, {16, 4}, "FP32",
                                          "slice_MLA_0_slice_transform",
                                          "slice_MLA_0_slice_transform", 64U));
  manifest.stages.push_back(stage);

  std::string err;
  auto override = build_output_override_from_manifest(manifest, {}, &err);
  assert(override.has_value() && err.empty());
  assert(override->outputs.size() == 1U);
  assert(override->outputs[0].name == "raw_dispatcher_ofm");
  assert(override->outputs[0].segment_name == "raw_dispatcher_ofm");
}

void terminal_selector_skips_boundary_view_transit() {
  sima::SimaPluginStaticManifest manifest;

  sima::StageStaticSpec producer;
  producer.logical_stage_id = "producer";
  producer.payload_kind = sima::StagePayloadKind::ProcessMla;
  producer.processmla.dispatcher_output_sizes = {16U};
  producer.physical_outputs.push_back(physical(0, 16U, "producer_raw"));
  producer.logical_outputs.push_back(logical(0, 0, {16}, {1}, "UINT8", "producer_raw",
                                           "producer_raw", 16U));
  manifest.stages.push_back(producer);

  sima::StageStaticSpec view;
  view.logical_stage_id = "slice_view";
  view.kernel_kind = "slice_transform";
  view.payload_kind = sima::StagePayloadKind::ProcessCvu;
  view.physical_outputs.push_back(physical(0, 4U, "slice_view"));
  view.logical_outputs.push_back(logical(0, 0, {4}, {1}, "UINT8", "slice_view", "slice_view", 4U));
  manifest.stages.push_back(view);

  const sima::StageStaticSpec* terminal = find_terminal_real_producer_for_endpoint(manifest, {});
  assert(terminal != nullptr);
  assert(terminal->logical_stage_id == "producer");

  auto override = build_output_override_from_manifest(manifest);
  assert(override.has_value());
  assert(override->outputs.size() == 1U);
  assert((override->outputs[0].shape == std::vector<std::int64_t>{16}));
  assert(override->outputs[0].name == "producer_raw");
}

void materialized_processcvu_terminal_is_not_demoted_by_detess_name() {
  sima::SimaPluginStaticManifest manifest;
  sima::StageStaticSpec stage;
  stage.logical_stage_id = "detess_terminal";
  stage.kernel_kind = "detessellate";
  stage.payload_kind = sima::StagePayloadKind::ProcessCvu;
  stage.processcvu.graph_family_enum = sima::ProcessCvuGraphFamily::Detess;
  stage.physical_outputs.push_back(physical(0, 32U, "detess_out"));
  stage.logical_outputs.push_back(logical(0, 0, {8}, {4}, "INT32", "detess_out", "detess_out", 32U));
  manifest.stages.push_back(stage);

  assert(classify_stage_for_publication(stage) == StagePublicationRole::MaterializedTransform);
  auto override = build_output_override_from_manifest(manifest);
  assert(override.has_value());
  assert(override->outputs[0].dtype == TensorDType::Int32);
  assert((override->outputs[0].shape == std::vector<std::int64_t>{8}));
}

void endpoint_selector_filters_by_route_slot() {
  sima::SimaPluginStaticManifest manifest;
  sima::StageStaticSpec stage;
  stage.logical_stage_id = "terminal_multi";
  stage.payload_kind = sima::StagePayloadKind::ProcessCvu;
  stage.processcvu.graph_family_enum = sima::ProcessCvuGraphFamily::Cast;
  stage.physical_outputs.push_back(physical(0, 4U, "head0"));
  stage.physical_outputs.push_back(physical(1, 8U, "head1"));
  stage.logical_outputs.push_back(logical(0, 0, {4}, {1}, "UINT8", "head0", "head0", 4U));
  stage.logical_outputs.push_back(logical(1, 1, {2}, {4}, "INT32", "head1", "head1", 8U));
  sima::StageOutputRoute route0;
  route0.output_slot = 0;
  route0.logical_output_index = 0;
  route0.tensor_index = 0;
  route0.cm_output_name = "head0";
  route0.segment_name = "head0";
  stage.output_order.push_back(route0);
  sima::StageOutputRoute route1;
  route1.output_slot = 1;
  route1.logical_output_index = 1;
  route1.tensor_index = 1;
  route1.cm_output_name = "head1";
  route1.segment_name = "head1";
  stage.output_order.push_back(route1);
  manifest.stages.push_back(stage);

  PublicOutputEndpointSelector endpoint;
  endpoint.route_slot = 1;
  std::string err;
  auto override = build_output_override_from_manifest(manifest, endpoint, &err);
  assert(override.has_value() && err.empty());
  assert(override->outputs.size() == 1U);
  assert(override->outputs[0].dtype == TensorDType::Int32);
  assert((override->outputs[0].shape == std::vector<std::int64_t>{2}));
  assert(override->outputs[0].memory_index == 1);
  assert(override->outputs[0].route_slot == 1);
  assert(override->outputs[0].name == "head1");
}

void endpoint_selector_filters_by_public_output_name() {
  sima::SimaPluginStaticManifest manifest;
  sima::StageStaticSpec stage;
  stage.logical_stage_id = "terminal_named";
  stage.payload_kind = sima::StagePayloadKind::ProcessCvu;
  stage.processcvu.graph_family_enum = sima::ProcessCvuGraphFamily::Cast;
  stage.physical_outputs.push_back(physical(0, 4U, "scores"));
  stage.physical_outputs.push_back(physical(1, 4U, "classes"));
  stage.logical_outputs.push_back(logical(0, 0, {4}, {1}, "UINT8", "scores", "scores", 4U));
  stage.logical_outputs.push_back(logical(1, 1, {1}, {4}, "INT32", "classes", "classes", 4U));
  manifest.stages.push_back(stage);

  PublicOutputEndpointSelector endpoint;
  endpoint.output_segment_name = "classes";
  std::string err;
  auto override = build_output_override_from_manifest(manifest, endpoint, &err);
  assert(override.has_value() && err.empty());
  assert(override->outputs.size() == 1U);
  assert(override->outputs[0].dtype == TensorDType::Int32);
  assert(override->outputs[0].name == "classes");
}

void terminal_stage_key_selects_non_last_branch() {
  sima::SimaPluginStaticManifest manifest;
  sima::StageStaticSpec branch_a;
  branch_a.logical_stage_id = "branch_a_terminal";
  branch_a.payload_kind = sima::StagePayloadKind::ProcessCvu;
  branch_a.processcvu.graph_family_enum = sima::ProcessCvuGraphFamily::Cast;
  branch_a.physical_outputs.push_back(physical(0, 4U, "branch_a_out"));
  branch_a.logical_outputs.push_back(
      logical(0, 0, {1}, {4}, "INT32", "branch_a_out", "branch_a_out", 4U));
  manifest.stages.push_back(branch_a);

  sima::StageStaticSpec branch_b;
  branch_b.logical_stage_id = "branch_b_terminal";
  branch_b.payload_kind = sima::StagePayloadKind::ProcessCvu;
  branch_b.processcvu.graph_family_enum = sima::ProcessCvuGraphFamily::Cast;
  branch_b.physical_outputs.push_back(physical(0, 8U, "branch_b_out"));
  branch_b.logical_outputs.push_back(
      logical(0, 0, {2}, {4}, "INT32", "branch_b_out", "branch_b_out", 8U));
  manifest.stages.push_back(branch_b);

  PublicOutputEndpointSelector endpoint;
  endpoint.terminal_stage_key = "branch_a_terminal";
  std::string err;
  auto override = build_output_override_from_manifest(manifest, endpoint, &err);
  assert(override.has_value() && err.empty());
  assert(override->outputs.size() == 1U);
  assert(override->outputs[0].name == "branch_a_out");
  assert((override->outputs[0].shape == std::vector<std::int64_t>{1}));
}

void inferred_dtype_terminal_falls_back_to_raw_bytes() {
  sima::SimaPluginStaticManifest manifest;
  sima::StageStaticSpec stage;
  stage.logical_stage_id = "terminal_inferred";
  stage.payload_kind = sima::StagePayloadKind::ProcessCvu;
  stage.processcvu.graph_family_enum = sima::ProcessCvuGraphFamily::Unknown;
  stage.physical_outputs.push_back(physical(0, 16U, "raw_parent"));
  auto inferred = logical(0, 0, {4}, {4}, "FP32", "semantic_guess", "raw_parent", 16U);
  inferred.dtype_source = sima::DTypeSource::InferredFromSize;
  stage.logical_outputs.push_back(std::move(inferred));
  manifest.stages.push_back(stage);

  std::string err;
  auto override = build_output_override_from_manifest(manifest, {}, &err);
  assert(override.has_value() && err.empty());
  assert(override->outputs.size() == 1U);
  assert(override->outputs[0].dtype == TensorDType::UInt8);
  assert((override->outputs[0].shape == std::vector<std::int64_t>{16}));
  assert(override->outputs[0].name == "raw_parent");
}

void unknown_dtype_source_terminal_falls_back_to_raw_bytes() {
  sima::SimaPluginStaticManifest manifest;
  sima::StageStaticSpec stage;
  stage.logical_stage_id = "terminal_unknown";
  stage.payload_kind = sima::StagePayloadKind::ProcessCvu;
  stage.processcvu.graph_family_enum = sima::ProcessCvuGraphFamily::Unknown;
  stage.physical_outputs.push_back(physical(0, 16U, "raw_parent"));
  stage.logical_outputs.push_back(logical(0, 0, {4}, {4}, "FP32", "semantic_unknown",
                                          "raw_parent", 16U, sima::DTypeSource::Unknown));
  manifest.stages.push_back(stage);

  std::string err;
  auto override = build_output_override_from_manifest(manifest, {}, &err);
  assert(override.has_value() && err.empty());
  assert(override->outputs.size() == 1U);
  assert(override->outputs[0].dtype == TensorDType::UInt8);
  assert((override->outputs[0].shape == std::vector<std::int64_t>{16}));
  assert(override->outputs[0].name == "raw_parent");
}

void output_override_route_slot_is_authoritative() {
  Tensor base;
  base.storage = make_cpu_owned_storage(16U);
  base.dtype = TensorDType::UInt8;
  base.shape = {16};
  base.strides_bytes = {1};
  base.route.logical_index = 99;
  base.route.route_slot = 99;
  base.route.memory_index = 0;
  base.route.name = "stale";
  base.route.segment_name = "stale_segment";

  Sample sample;
  sample.kind = SampleKind::Tensor;
  sample.tensor = base;

  OutputTensorOverride override;
  OutputTensorOverrideEntry entry;
  entry.shape = {16};
  entry.strides_bytes = {1};
  entry.dtype = TensorDType::UInt8;
  entry.logical_output_index = 0;
  entry.route_slot = 0;
  entry.memory_index = 0;
  entry.name = "public_output";
  entry.segment_name = "public_segment";
  override.outputs.push_back(entry);

  Sample out = apply_output_tensor_override(sample, override, /*materialize_output=*/false);
  assert(out.kind == SampleKind::TensorSet);
  assert(out.tensors.size() == 1U);
  assert(out.tensors[0].route.logical_index == 0);
  assert(out.tensors[0].route.route_slot == 0);
  assert(out.tensors[0].route.memory_index == 0);
  assert(out.tensors[0].route.name == "public_output");
  assert(out.tensors[0].route.segment_name == "public_segment");
}

} // namespace

int main() {
  normal_terminal_logical_contract_is_preserved();
  terminal_mla_with_downstream_slice_metadata_falls_back_to_raw_physical();
  terminal_raw_fallback_drops_boundary_view_physical_name();
  terminal_raw_fallback_keeps_non_view_dispatcher_name();
  terminal_selector_skips_boundary_view_transit();
  materialized_processcvu_terminal_is_not_demoted_by_detess_name();
  endpoint_selector_filters_by_route_slot();
  endpoint_selector_filters_by_public_output_name();
  terminal_stage_key_selects_non_last_branch();
  inferred_dtype_terminal_falls_back_to_raw_bytes();
  unknown_dtype_source_terminal_falls_back_to_raw_bytes();
  output_override_route_slot_is_authoritative();
  return 0;
}
