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
                                      std::uint64_t size_bytes = 0) {
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

} // namespace

int main() {
  normal_terminal_logical_contract_is_preserved();
  terminal_mla_with_downstream_slice_metadata_falls_back_to_raw_physical();
  terminal_selector_skips_boundary_view_transit();
  materialized_processcvu_terminal_is_not_demoted_by_detess_name();
  return 0;
}
