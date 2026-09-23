#include "nodes/sima/Preproc.h"
#include "pipeline/internal/contract/CompiledNodeContract.h"
#include "pipeline/internal/contract/ContractCompiler.h"
#include "pipeline/internal/sima/ContractRender.h"
#include "test_main.h"

#include <array>
#include <cstdint>
#include <memory>
#include <sstream>
#include <vector>

namespace {

simaai::neat::PreprocOptions make_render_preproc_options() {
  simaai::neat::PreprocOptions opt;
  opt.model_managed_contract = true;
  opt.set_input_shape({1080, 1920, 3});
  opt.input_img_type = "RGB";
  opt.set_output_shape({640, 640, 3});
  opt.scaled_width = 640;
  opt.scaled_height = 640;
  opt.output_img_type = "RGB";
  opt.output_dtype = "EVXX_INT8";
  opt.tessellate = true;
  opt.single_output_handoff = true;
  opt.set_slice_shape({32, 128, 3});
  opt.q_scale = 0.25;
  opt.q_zp = 7;
  return opt;
}

simaai::neat::PreprocOptions make_non_tess_render_preproc_options() {
  auto opt = make_render_preproc_options();
  opt.tessellate = false;
  return opt;
}

const simaai::neat::pipeline_internal::sima::StageStaticSpec&
only_stage(const simaai::neat::pipeline_internal::sima::SimaPluginStaticManifest& manifest) {
  require(manifest.stages.size() == 1U, "manifest should contain exactly one stage");
  return manifest.stages.front();
}

std::string
join_errors(const simaai::neat::pipeline_internal::sima::ManifestBuildDiagnostics& diagnostics) {
  std::ostringstream oss;
  for (std::size_t i = 0; i < diagnostics.errors.size(); ++i) {
    if (i > 0) {
      oss << " | ";
    }
    oss << diagnostics.errors[i];
  }
  return oss.str();
}

void verify_render_manifest_equivalence() {
  using namespace simaai::neat;

  std::vector<std::shared_ptr<Node>> nodes_to_compile = {
      nodes::Preproc(make_render_preproc_options())};
  pipeline_internal::sima::ManifestBuildDiagnostics diagnostics;
  const auto compiled =
      compile_node_contracts(nodes_to_compile, ContractCompileInput{}, &diagnostics);
  require(diagnostics.errors.empty(), "compiled preproc contract should not emit errors");

  const auto manifest_opt =
      render_manifest_from_compiled_contracts(compiled, ContractCompileInput{}, &diagnostics);
  require(manifest_opt.has_value(),
          "compiled preproc contract should render a manifest: " + join_errors(diagnostics));
  const auto& stage = only_stage(*manifest_opt);

  require(stage.payload_kind == pipeline_internal::sima::StagePayloadKind::ProcessCvu,
          "rendered stage should be processcvu");
  // Runtime publishes both rgb and tessellated outputs; the exposed view
  // narrows to the selected tessellated handoff.
  require(stage.logical_outputs.size() == 1U,
          "rendered exposed view should only expose the selected logical output");
  require(stage.output_order.size() == 1U,
          "rendered exposed view should only expose one output route");
  require(stage.logical_outputs.front().logical_name == "output_tessellated_image",
          "rendered exposed logical output should preserve selected runtime identity");
  require(stage.processcvu.default_output_names.size() == 2U &&
              stage.processcvu.default_output_names[0] == "output_rgb_image" &&
              stage.processcvu.default_output_names[1] == "output_tessellated_image",
          "rendered stage should publish both runtime output names");
  require(stage.processcvu.preproc_single_output_handoff,
          "rendered stage should preserve explicit single-output handoff");
  require(stage.processcvu.primary_output_transport_kind ==
              pipeline_internal::sima::ProcessCvuOutputTransportKind::Packed,
          "rendered stage should mark the primary output transport as packed");
  require(stage.processcvu.primary_output_semantic_kind ==
              pipeline_internal::sima::ProcessCvuOutputSemanticKind::TessellatedImage,
          "rendered stage should mark the primary output semantic as tessellated");
  require(stage.processcvu.runtime_output_logical_shapes.size() == 2U,
          "rendered stage should preserve runtime logical shapes for both outputs");
}

void verify_runtime_output_order_rendering() {
  using namespace simaai::neat;
  using namespace simaai::neat::pipeline_internal::sima;

  auto make_logical = [](int logical_index, int physical_index, int output_slot, int tensor_index,
                         std::string name, int width, int height, int depth) {
    LogicalTensorStaticSpec logical;
    logical.logical_index = logical_index;
    logical.backend_output_index = logical_index;
    logical.physical_index = physical_index;
    logical.output_slot = output_slot;
    logical.tensor_index = tensor_index;
    logical.logical_name = name;
    logical.backend_name = name;
    logical.segment_name = name;
    logical.dtype = "INT8";
    logical.layout = "HWC";
    logical.shape = {height, width, depth};
    logical.size_bytes = static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height) *
                         static_cast<std::uint64_t>(depth);
    return logical;
  };

  auto make_physical = [](int physical_index, std::string name, std::uint64_t size_bytes) {
    PhysicalBufferStaticSpec physical;
    physical.physical_index = physical_index;
    physical.allocator_index = physical_index;
    physical.device_kind = DeviceKind::Evxx;
    physical.segment_name = std::move(name);
    physical.size_bytes = size_bytes;
    return physical;
  };

  CompiledNodeContract stage_contract;
  stage_contract.node_kind = "Preproc";
  stage_contract.plugin_kind = "processcvu";
  stage_contract.element_name = "n0_preproc";
  stage_contract.logical_stage_id = "n0_preproc";
  stage_contract.renderable = true;

  CompiledProcessCvuContract processcvu;
  processcvu.payload.graph_family = "preproc";
  processcvu.payload.graph_name = "preproc";
  processcvu.payload.input_dtype = "UINT8";
  processcvu.payload.output_dtype = "INT8";
  processcvu.payload.out_dtype = "INT8";
  processcvu.payload.runtime_output_logical_layout_list = {"HWC", "HWC"};
  processcvu.payload.default_input_name = "input_image";
  processcvu.payload.default_output_names = {"output_rgb_image", "output_tessellated_image"};
  processcvu.preproc_single_output_handoff = false;

  // This intentionally exercises a generic multi-output runtime contract, not the canonical
  // single-output preproc handoff path.
  processcvu.runtime_contract.logical_outputs.push_back(
      make_logical(1, 1, 1, 11, "output_tessellated_image", 128, 32, 3));
  processcvu.runtime_contract.logical_outputs.push_back(
      make_logical(0, 0, 0, 10, "output_rgb_image", 640, 640, 3));
  processcvu.runtime_contract.physical_outputs.push_back(
      make_physical(0, "output_rgb_image", 640U * 640U * 3U));
  processcvu.runtime_contract.physical_outputs.push_back(
      make_physical(1, "output_tessellated_image", 128U * 32U * 3U));

  StageOutputRoute rgb_route;
  rgb_route.output_slot = 0;
  rgb_route.logical_output_index = 0;
  rgb_route.tensor_index = 10;
  rgb_route.cm_output_name = "output_rgb_image";
  rgb_route.segment_name = "output_rgb_image";
  processcvu.runtime_contract.output_order.push_back(rgb_route);

  StageOutputRoute tess_route;
  tess_route.output_slot = 1;
  tess_route.logical_output_index = 1;
  tess_route.tensor_index = 11;
  tess_route.cm_output_name = "output_tessellated_image";
  tess_route.segment_name = "output_tessellated_image";
  processcvu.runtime_contract.output_order.push_back(tess_route);

  processcvu.exposed_view.primary_output_name = "output_tessellated_image";
  auto exposed_tess = processcvu.runtime_contract.logical_outputs.front();
  exposed_tess.output_slot = 0;
  processcvu.exposed_view.exposed_logical_outputs.push_back(exposed_tess);
  StageOutputRoute exposed_route = tess_route;
  exposed_route.output_slot = 0;
  processcvu.exposed_view.exposed_output_order.push_back(exposed_route);

  stage_contract.processcvu = std::move(processcvu);

  CompiledPipelineContracts compiled;
  compiled.fully_renderable = true;
  compiled.stages.push_back(std::move(stage_contract));

  ManifestBuildDiagnostics diagnostics;
  const auto manifest_opt =
      render_manifest_from_compiled_contracts(compiled, ContractCompileInput{}, &diagnostics);
  require(manifest_opt.has_value(),
          "manual processcvu contract should render a manifest: " + join_errors(diagnostics));
  const auto& stage = only_stage(*manifest_opt);

  require(stage.processcvu.default_output_names.size() == 2U,
          "manual multi-output runtime contract should preserve runtime output order");
  require(stage.processcvu.default_output_names[0] == "output_rgb_image" &&
              stage.processcvu.default_output_names[1] == "output_tessellated_image",
          "manual multi-output runtime names should follow runtime output_order, not logical "
          "storage order");
  require(stage.processcvu.runtime_output_logical_index_list.size() == 2U,
          "runtime output route metadata should be rendered for each runtime output");
  require(stage.processcvu.runtime_output_logical_index_list[0] == 0 &&
              stage.processcvu.runtime_output_logical_index_list[1] == 1,
          "runtime output logical indices should align with runtime output_order");
  require(stage.processcvu.runtime_output_output_slot_list[0] == 0 &&
              stage.processcvu.runtime_output_output_slot_list[1] == 1,
          "runtime output slots should align with runtime output_order");
  require(stage.processcvu.runtime_output_physical_index_list[0] == 0 &&
              stage.processcvu.runtime_output_physical_index_list[1] == 1,
          "runtime output physical indices should align with runtime output_order");
  require(stage.processcvu.runtime_output_transport_kind_list.size() == 2U &&
              stage.processcvu.runtime_output_transport_kind_list[0] ==
                  ProcessCvuOutputTransportKind::Dense &&
              stage.processcvu.runtime_output_transport_kind_list[1] ==
                  ProcessCvuOutputTransportKind::Dense,
          "manual multi-output runtime contract should default runtime transports to dense");
  require(stage.processcvu.runtime_output_logical_shapes.size() == 2U,
          "manual multi-output runtime contract should default logical shapes from runtime dims");
  // Each shape is {height, width, depth} in HWC layout.
  require(stage.processcvu.runtime_output_logical_shapes[0].size() >= 3U &&
              stage.processcvu.runtime_output_logical_shapes[0][0] == 640 &&
              stage.processcvu.runtime_output_logical_shapes[0][1] == 640 &&
              stage.processcvu.runtime_output_logical_shapes[0][2] == 3,
          "manual multi-output runtime contract should default logical shape for output 0");
  require(stage.processcvu.runtime_output_logical_shapes[1].size() >= 3U &&
              stage.processcvu.runtime_output_logical_shapes[1][0] == 32 &&
              stage.processcvu.runtime_output_logical_shapes[1][1] == 128 &&
              stage.processcvu.runtime_output_logical_shapes[1][2] == 3,
          "manual multi-output runtime contract should default logical shape for output 1");
}

void verify_projected_cast_mla_boundary_preserves_storage() {
  using namespace simaai::neat;
  using namespace simaai::neat::pipeline_internal::sima;
  namespace sc = simaai::neat::pipeline_internal::sima::static_contract;

  CompiledNodeContract cast_stage;
  cast_stage.node_kind = "ModelFragmentStage";
  cast_stage.plugin_kind = "processcvu";
  cast_stage.element_name = "cast_1";
  cast_stage.logical_stage_id = "cast_1";
  cast_stage.renderable = true;
  cast_stage.processcvu.emplace();
  auto& cast = *cast_stage.processcvu;
  cast.payload.graph_family = "cast";
  cast.payload.graph_name = "cast";
  cast.payload.graph_id = 221;
  cast.payload.dmabuf_plan_contract = true;
  cast.physical_command_role = sc::PhysicalCommandRole::Ingress;
  cast.runtime_contract.frame_arena_size_bytes = 16384U;
  cast.runtime_contract.frame_arena_role = FrameArenaRole::Allocate;
  cast.runtime_contract.consumer_keeps_distinct_physical_inputs = true;

  CompiledNodeContract mla_stage;
  mla_stage.node_kind = "ModelFragmentStage";
  mla_stage.plugin_kind = "processmla";
  mla_stage.element_name = "MLA_0";
  mla_stage.logical_stage_id = "MLA_0";
  mla_stage.renderable = true;
  mla_stage.processmla.emplace();
  auto& mla = *mla_stage.processmla;
  mla.payload.dmabuf_plan_contract = true;
  mla.runtime_contract.consumer_keeps_distinct_physical_inputs = true;
  mla.runtime_contract.frame_arena_size_bytes = 16384U;
  mla.runtime_contract.frame_arena_role = FrameArenaRole::ReuseInput;

  const std::array<std::uint64_t, 2> sizes{60U, 120U};
  for (int index = 0; index < 2; ++index) {
    const auto name = "cast_" + std::to_string(index);
    PhysicalBufferStaticSpec physical;
    physical.physical_index = index;
    physical.allocator_index = index;
    physical.source_physical_index = index;
    physical.size_bytes = sizes[index];
    physical.source_byte_offset = 4096 * (index + 1);
    physical.required_alignment_bytes = 128U << index;
    physical.memory_flags = 0x10U << index;
    physical.device_kind = DeviceKind::Evxx;
    physical.address_source = PhysicalAddressSource::FrameArenaSpan;
    physical.segment_name = name;
    cast.runtime_contract.physical_outputs.push_back(physical);
    mla.runtime_contract.physical_inputs.push_back(physical);

    LogicalTensorStaticSpec logical;
    logical.logical_index = index;
    logical.backend_output_index = index;
    logical.physical_index = index;
    logical.output_slot = index;
    logical.tensor_index = index;
    logical.size_bytes = sizes[index];
    logical.shape = {1, 3, 5, 2 * (index + 1)};
    logical.dtype = "BF16";
    logical.layout = "HWC";
    logical.logical_name = name;
    logical.backend_name = name;
    logical.segment_name = name;
    cast.runtime_contract.logical_outputs.push_back(logical);
    cast.exposed_view.exposed_logical_outputs.push_back(logical);

    StageOutputRoute output;
    output.output_slot = index;
    output.logical_output_index = index;
    output.tensor_index = index;
    output.cm_output_name = name;
    output.segment_name = name;
    cast.runtime_contract.output_order.push_back(output);
    cast.exposed_view.exposed_output_order.push_back(output);
    cast.payload.default_output_names.push_back(name);

    InputBindingStaticSpec input;
    input.sink_pad_index = 0;
    input.local_logical_input_index = index;
    input.src_logical_output_index = index;
    input.src_output_slot = index;
    input.src_physical_output_index = index;
    input.src_physical_size_bytes = sizes[index];
    input.cm_input_name = name;
    input.source_segment_name = name;
    input.required = true;
    mla.runtime_contract.input_bindings.push_back(input);
  }
  cast.exposed_view.primary_output_name = "cast_0";
  const auto expected_outputs = cast.runtime_contract.physical_outputs;
  CompiledPipelineContracts compiled;
  compiled.fully_renderable = true;
  compiled.stages.push_back(std::move(cast_stage));
  compiled.stages.push_back(std::move(mla_stage));
  ManifestBuildDiagnostics diagnostics;
  const auto manifest =
      render_manifest_from_compiled_contracts(compiled, ContractCompileInput{}, &diagnostics);
  require(manifest.has_value() && diagnostics.errors.empty(),
          "projected grouped cast/MLA boundary should render: " + join_errors(diagnostics));
  require(manifest->stages.size() == 2U, "both projected stages must survive rendering");
  const auto& producer = manifest->stages[0];
  const auto& consumer = manifest->stages[1];
  require(producer.frame_arena_role == FrameArenaRole::Allocate &&
              consumer.frame_arena_role == FrameArenaRole::ReuseInput &&
              producer.frame_arena_size_bytes == 16384U &&
              consumer.frame_arena_size_bytes == 16384U,
          "rendering must preserve projected arena ownership");
  require(producer.physical_outputs.size() == 2U && producer.logical_outputs.size() == 2U &&
              producer.output_order.size() == 2U && consumer.input_bindings.size() == 2U &&
              consumer.physical_inputs.size() == 2U,
          "rendering must preserve both distinct MLA input routes");
  for (int index = 0; index < 2; ++index) {
    const auto& expected = expected_outputs[index];
    for (const auto* actual :
         {&producer.physical_outputs[index], &consumer.physical_inputs[index]}) {
      require(actual->physical_index == expected.physical_index &&
                  actual->allocator_index == expected.allocator_index &&
                  actual->source_physical_index == expected.source_physical_index &&
                  actual->size_bytes == expected.size_bytes &&
                  actual->source_byte_offset == expected.source_byte_offset &&
                  actual->required_alignment_bytes == expected.required_alignment_bytes &&
                  actual->memory_flags == expected.memory_flags &&
                  actual->device_kind == expected.device_kind &&
                  actual->address_source == expected.address_source &&
                  actual->segment_name == expected.segment_name,
              "rendering must not reconstruct projected physical buffers with missing facts");
    }
    const auto& logical = producer.logical_outputs[index];
    const auto& output = producer.output_order[index];
    const auto& input = consumer.input_bindings[index];
    require(logical.physical_index == index && logical.logical_name == expected.segment_name &&
                logical.segment_name == expected.segment_name && logical.byte_offset == 0 &&
                output.output_slot == index && output.logical_output_index == index &&
                output.tensor_index == index && output.cm_output_name == expected.segment_name &&
                input.local_logical_input_index == index &&
                input.src_logical_output_index == index && input.src_output_slot == index &&
                input.src_physical_output_index == index &&
                input.src_physical_size_bytes == sizes[index] &&
                input.cm_input_name == expected.segment_name &&
                input.source_segment_name == expected.segment_name,
            "rendering must retain each cast output and MLA inbound binding identity");
  }
}

void verify_non_tess_preproc_semantic_rendering() {
  using namespace simaai::neat;

  std::vector<std::shared_ptr<Node>> nodes_to_compile = {
      nodes::Preproc(make_non_tess_render_preproc_options())};
  pipeline_internal::sima::ManifestBuildDiagnostics diagnostics;
  const auto compiled =
      compile_node_contracts(nodes_to_compile, ContractCompileInput{}, &diagnostics);
  require(diagnostics.errors.empty(), "non-tess preproc compile should not emit errors");

  const auto manifest_opt =
      render_manifest_from_compiled_contracts(compiled, ContractCompileInput{}, &diagnostics);
  require(manifest_opt.has_value(),
          "non-tess preproc contract should render a manifest: " + join_errors(diagnostics));
  const auto& stage = only_stage(*manifest_opt);

  require(stage.logical_outputs.size() == 1U &&
              stage.logical_outputs.front().logical_name == "output_rgb_image",
          "non-tess preproc should expose the RGB handoff output");
  require(!stage.processcvu.default_output_names.empty() &&
              std::find(stage.processcvu.default_output_names.begin(),
                        stage.processcvu.default_output_names.end(),
                        std::string("output_rgb_image")) !=
                  stage.processcvu.default_output_names.end(),
          "non-tess preproc runtime should publish output_rgb_image");
  require(!stage.processcvu.runtime_output_transport_kind_list.empty(),
          "non-tess preproc should publish runtime transport kinds");
  require(stage.processcvu.runtime_output_transport_kind_list.front() ==
              pipeline_internal::sima::ProcessCvuOutputTransportKind::Dense,
          "non-tess preproc primary runtime output should be dense");
  require(!stage.processcvu.runtime_output_semantic_kind_list.empty() &&
              stage.processcvu.runtime_output_semantic_kind_list.front() ==
                  pipeline_internal::sima::ProcessCvuOutputSemanticKind::Image,
          "non-tess preproc primary runtime output should be image-semantic");
}

void verify_nested_processcvu_role_placement_rendering() {
  using namespace simaai::neat;
  using namespace simaai::neat::pipeline_internal::sima;
  namespace sc = simaai::neat::pipeline_internal::sima::static_contract;

  struct Case {
    ProcessCvuGraphFamily family;
    int graph_id;
    sc::PhysicalCommandRole role;
    const char* expected_target;
    const char* expected_source;
  };
  const std::array cases{
      Case{ProcessCvuGraphFamily::Cast, 221, sc::PhysicalCommandRole::Egress, "A65",
           "processcvu_post"},
      Case{ProcessCvuGraphFamily::Dequant, 223, sc::PhysicalCommandRole::Egress, "A65",
           "processcvu_post"},
      Case{ProcessCvuGraphFamily::Quant, 222, sc::PhysicalCommandRole::Ingress, "EV74",
           "processcvu_pre"},
  };

  ContractCompileInput input;
  input.processcvu_requested_run_target = "EV74";
  input.processcvu.pre_run_target = "EV74";
  input.processcvu.post_run_target = "A65";

  CompiledPipelineContracts compiled;
  compiled.fully_renderable = true;
  CompiledNodeContract parent;
  parent.node_kind = "ModelFragment";
  parent.plugin_kind = "ModelFragment";
  parent.renderable = true;
  for (const auto& item : cases) {
    CompiledNodeContract child;
    child.node_kind = "ModelFragmentStage";
    child.plugin_kind = "processcvu";
    child.element_name = "physical_cvu_graph_" + std::to_string(item.graph_id);
    child.logical_stage_id = child.element_name;
    child.renderable = true;
    child.processcvu.emplace();
    child.processcvu->payload.graph_family_enum = item.family;
    child.processcvu->payload.graph_id = item.graph_id;
    child.processcvu->payload.graph_name = child.element_name;
    child.processcvu->payload.graph_family = child.element_name;
    child.processcvu->physical_command_role = item.role;

    const std::string output_name = child.element_name + "_output";
    LogicalTensorStaticSpec logical;
    logical.logical_index = 0;
    logical.backend_output_index = 0;
    logical.physical_index = 0;
    logical.output_slot = 0;
    logical.tensor_index = 0;
    logical.logical_name = output_name;
    logical.backend_name = output_name;
    logical.segment_name = output_name;
    logical.dtype = "FP32";
    logical.layout = "HWC";
    logical.shape = {1, 1, 1};
    logical.size_bytes = sizeof(float);
    child.processcvu->runtime_contract.logical_outputs.push_back(logical);

    PhysicalBufferStaticSpec physical;
    physical.physical_index = 0;
    physical.allocator_index = 0;
    physical.source_physical_index = 0;
    physical.size_bytes = logical.size_bytes;
    physical.segment_name = output_name;
    child.processcvu->runtime_contract.physical_outputs.push_back(physical);

    StageOutputRoute route;
    route.output_slot = 0;
    route.logical_output_index = 0;
    route.tensor_index = 0;
    route.cm_output_name = output_name;
    route.segment_name = output_name;
    child.processcvu->runtime_contract.output_order.push_back(route);
    child.processcvu->exposed_view.exposed_logical_outputs.push_back(std::move(logical));
    child.processcvu->exposed_view.exposed_output_order.push_back(std::move(route));
    child.processcvu->exposed_view.primary_output_name = output_name;
    parent.child_stages.push_back(std::move(child));
  }
  compiled.stages.push_back(std::move(parent));

  ManifestBuildDiagnostics diagnostics;
  const auto manifest = render_manifest_from_compiled_contracts(compiled, input, &diagnostics);
  require(manifest.has_value() && diagnostics.errors.empty(),
          "nested ProcessCVU children should render with their exact physical roles: " +
              join_errors(diagnostics));
  require(manifest->stages.size() == cases.size(),
          "nested ProcessCVU render should retain every child stage");
  for (std::size_t index = 0; index < cases.size(); ++index) {
    const auto& expected = cases[index];
    const auto& payload = manifest->stages[index].processcvu;
    require(payload.graph_id == expected.graph_id &&
                payload.requested_run_target == expected.expected_target &&
                payload.run_target == expected.expected_target &&
                payload.resolved_exec_backend ==
                    (std::string(expected.expected_target) == "A65" ? "A65" : "EVXX"),
            "nested ProcessCVU child must resolve the target from its exact physical role");
    require(payload.run_target_resolution_reason.find(expected.expected_source) !=
                std::string::npos,
            "nested ProcessCVU child must record the role-specific resolution source");
  }
}

} // namespace

RUN_TEST("unit_contract_render_manifest_equivalence_test", ([] {
           verify_render_manifest_equivalence();
           verify_runtime_output_order_rendering();
           verify_projected_cast_mla_boundary_preserves_storage();
           verify_non_tess_preproc_semantic_rendering();
           verify_nested_processcvu_role_placement_rendering();
         }));
