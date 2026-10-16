#include "builder/Node.h"
#include "builder/NodeContractConfigurable.h"
#include "builder/NodeContractProvider.h"
#include "model/internal/ModelPack.h"
#include "pipeline/internal/RenderedMlaContractQuery.h"
#include "pipeline/internal/SelectedStageBoundary.h"
#include "test_main.h"

#include <memory>
#include <string>
#include <vector>

namespace {

using namespace simaai::neat;
namespace query = pipeline_internal::rendered_stage_query;
namespace contract = pipeline_internal::sima;

CompiledMlaContract make_mla_contract() {
  CompiledMlaContract compiled;
  contract::LogicalInputStaticSpec logical;
  logical.logical_index = 0;
  logical.physical_index = 0;
  logical.shape = {1, 2, 3, 4};
  logical.dtype = "INT8";
  logical.layout = "NHWC";
  logical.segment_name = "logical_input";
  logical.size_bytes = 24U;
  compiled.runtime_contract.logical_inputs = {logical};
  contract::PhysicalBufferStaticSpec physical;
  physical.physical_index = 0;
  physical.segment_name = "producer_carrier";
  physical.source_byte_offset = 128;
  physical.size_bytes = 96U;
  compiled.runtime_contract.physical_inputs = {physical};
  contract::InputBindingStaticSpec binding;
  binding.local_logical_input_index = 0;
  binding.cm_input_name = "consumer_input";
  binding.source_segment_name = "authored_source";
  compiled.runtime_contract.input_bindings = {binding};
  compiled.runtime_contract.frame_arena_size_bytes = 4096U;
  return compiled;
}

class ContextProbe final : public Node,
                           public NodeContractProvider,
                           public NodeContractConfigurable {
public:
  std::string kind() const override {
    return "ContextProbe";
  }
  std::string backend_fragment(int) const override {
    return "identity name=context_probe";
  }
  std::vector<std::string> element_names(int) const override {
    return {"context_probe"};
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Static;
  }
  NodeContractDefinition contract_definition() const override {
    NodeContractDefinition definition;
    definition.node_kind = kind();
    definition.plugin_kind = "processmla";
    return definition;
  }
  bool compile_node_contract(const ContractCompileInput& input, CompiledNodeContract* out,
                             std::string* error) const override {
    ++compile_count;
    if (!input.ingress.ingress_sample.has_value()) {
      *error = "ContextProbe: requires actual ingress Sample authority";
      return false;
    }
    out->node_kind = kind();
    out->plugin_kind = "processmla";
    out->element_name = "context_probe";
    out->processmla = make_mla_contract();
    out->renderable = true;
    return true;
  }
  void apply_compiled_contract(const CompiledNodeContract&, std::string*) override {
    ++apply_count;
  }
  mutable int compile_count = 0;
  int apply_count = 0;
};

CompiledProcessCvuContract make_preproc_contract() {
  CompiledProcessCvuContract compiled;
  compiled.payload.graph_family = "preproc";
  compiled.payload.batch_size = 3;
  compiled.payload.primary_output_name = "selected";
  compiled.payload.default_output_names = {"selected"};
  compiled.payload.output_dtype = "INT8";
  compiled.payload.output_shapes = {{3, 4, 5, 3}};
  compiled.payload.runtime_output_logical_shapes = {{3, 4, 5, 3}};
  compiled.payload.runtime_output_logical_layout_list = {"NHWC"};
  compiled.payload.runtime_output_transport_kind_list = {
      contract::ProcessCvuOutputTransportKind::Dense};
  compiled.payload.runtime_output_semantic_kind_list = {
      contract::ProcessCvuOutputSemanticKind::QuantizedTensor};
  contract::LogicalTensorStaticSpec logical;
  logical.logical_index = 5;
  logical.output_slot = 9;
  logical.physical_index = 7;
  logical.logical_name = "selected";
  logical.segment_name = "packed_carrier";
  logical.dtype = "INT8";
  logical.layout = "NHWC";
  logical.shape = {3, 4, 5, 3};
  logical.stride_bytes = {96, 15, 3, 1};
  logical.size_bytes = 288U;
  compiled.runtime_contract.logical_outputs = {logical};
  contract::PhysicalBufferStaticSpec physical;
  physical.physical_index = 7;
  physical.segment_name = "packed_carrier";
  physical.source_byte_offset = 4096;
  physical.size_bytes = 288U;
  compiled.runtime_contract.physical_outputs = {physical};
  compiled.runtime_contract.frame_arena_size_bytes = 8192U;
  contract::StageOutputRoute route;
  route.logical_output_index = 5;
  route.output_slot = 9;
  route.cm_output_name = "selected";
  route.segment_name = "packed_carrier";
  compiled.runtime_contract.output_order = {route};
  compiled.exposed_view.primary_output_name = "selected";
  compiled.exposed_view.exposed_logical_outputs = {logical};
  compiled.exposed_view.exposed_output_order = {route};
  return compiled;
}

} // namespace

RUN_TEST("unit_stage_metadata_query_test", ([] {
           const auto mla = make_mla_contract();
           const auto mla_info = query::mla_input_tensor_info_from_compiled_contract(mla);
           require(mla_info.logical_shape == std::vector<std::int64_t>({1, 2, 3, 4}) &&
                       mla_info.logical_dtype == "INT8" &&
                       mla_info.logical_layout == TensorLayout::HWC &&
                       mla_info.segment_name == "authored_source" &&
                       mla_info.span_byte_offset == 128 && mla_info.span_size_bytes == 96 &&
                       !mla_info.physical_shape.has_value(),
                   "MLA projection changed exact ingress facts or invented physical shape");
           require(query::primary_input_buffer_name_from_runtime_contract(mla.runtime_contract) ==
                       "authored_source",
                   "compiled input binding must supply the authored wire label");

           auto preproc = make_preproc_contract();
           const auto dense = query::preproc_output_info_from_compiled_contract(preproc);
           require(
               dense.primary_output_name == "selected" && dense.primary_route_slot == 9 &&
                   dense.output_memory_order == std::vector<std::string>({"selected"}) &&
                   dense.output_dtype == "INT8" && dense.logical_dims.width == 5 &&
                   dense.logical_dims.height == 4 && dense.logical_dims.depth == 3 &&
                   dense.logical_layout == TensorLayout::HWC && dense.roi_slot_bytes == 96U &&
                   dense.roi_member_shape == std::vector<std::int64_t>({1, 4, 5, 3}) &&
                   dense.roi_member_strides_bytes == std::vector<std::int64_t>({96, 15, 3, 1}) &&
                   dense.transport_kind == stages::PreprocOutputTransportKind::Dense &&
                   dense.semantic_kind == stages::PreprocOutputSemanticKind::QuantizedTensor,
               "Preproc projection lost selected route, member pitch or exact ROI carrier span");
           preproc.payload.runtime_output_transport_kind_list = {
               contract::ProcessCvuOutputTransportKind::Packed};
           const auto packed = query::preproc_output_info_from_compiled_contract(preproc);
           require(packed.transport_kind == stages::PreprocOutputTransportKind::Packed &&
                       packed.roi_slot_bytes == dense.roi_slot_bytes &&
                       packed.roi_member_strides_bytes == dense.roi_member_strides_bytes &&
                       preproc.runtime_contract.frame_arena_size_bytes == 8192U &&
                       preproc.runtime_contract.physical_outputs.front().source_byte_offset == 4096,
                   "read-only projection must neither substitute aligned arena bytes nor reauthor "
                   "placement");

           auto probe = std::make_shared<ContextProbe>();
           const std::vector<std::shared_ptr<Node>> nodes{probe};
           std::string error = "stale error";
           const auto facts = internal::selected_model_fragment_stage_facts(nodes, &error);
           require(facts.has_value() && facts->stage_facts.size() == 1U &&
                       facts->stage_facts.front().empty() && error.empty() &&
                       probe->compile_count == 0 && probe->apply_count == 0,
                   "readonly model metadata query must not prepare an unrelated native ingress");

           contract::ManifestBuildDiagnostics diagnostics;
           require(!query::rendered_manifest_from_nodes(nodes, "unseeded-probe", &diagnostics) &&
                       diagnostics.errors.size() == 1U &&
                       diagnostics.errors.front().find(
                           "requires actual ingress Sample authority") != std::string::npos,
                   "intentional query must preserve the provider's real compilation diagnostic");
           ContractCompileInput input;
           input.pipeline_label = "seeded-probe";
           input.ingress.ingress_sample = Sample{};
           const auto manifest = query::rendered_manifest_from_nodes(nodes, input, &diagnostics);
           require(manifest.has_value() && diagnostics.errors.empty() &&
                       probe->compile_count == 2 && probe->apply_count == 0,
                   "explicit-context query must render without applying contracts to source nodes");
           const std::vector<std::shared_ptr<Node>> empty;
           require(!query::rendered_manifest_from_nodes(empty, "absent-probe", &diagnostics) &&
                       diagnostics.errors.empty(),
                   "valid absence must stay distinguishable from a compilation failure");
         }));
