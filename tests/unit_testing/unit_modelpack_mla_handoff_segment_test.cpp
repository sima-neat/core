#include "support/yolov8_variant_fixture.h"
#include "support/cold_contract_gst_init.h"
#include "model/Model.h"
#include "model/internal/ModelInternal.h"
#include "model/internal/ModelPack.h"
#include "nodes/groups/ModelGroups.h"
#include "nodes/sima/CastTess.h"
#include "nodes/common/Output.h"
#include "nodes/sima/DetessDequant.h"
#include "pipeline/graph/internal/GraphBuildInternal.h"
#include "pipeline/internal/RenderedMlaContractQuery.h"
#include "pipeline/internal/sima/InternalEdgeContractResolver.h"
#include "pipeline/internal/contract/ContractCompiler.h"
#include "pipeline/internal/sima/CompiledProcessCvuContractQuery.h"
#include "test_main.h"
#include "pipeline/internal/sima/TensorSemanticsUtil.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <filesystem>
#include <memory>
#include <vector>
#include <string>
#include <sstream>

namespace {

namespace sima = simaai::neat::pipeline_internal::sima;

// Independent oracle: MPK publication geometry, not the projected output's
// own dtype/shape/span. Allocation padding may exceed this logical byte span.
std::string dequant_publication_error(const simaai::neat::CompiledProcessCvuContract& post,
                                      const std::vector<const sima::MpkTensorContract*>& authored) {
  if (post.payload.graph_name != "detessdequant")
    return "fusion: expected DetessDequant";
  const auto& outputs = post.exposed_view.exposed_logical_outputs;
  if (authored.empty() || outputs.size() != authored.size() ||
      post.runtime_contract.logical_outputs.size() != authored.size())
    return "outputs: missing authored dequant publication";
  for (const auto* expected : authored) {
    const auto match = std::find_if(outputs.begin(), outputs.end(), [&](const auto& output) {
      return output.logical_name == expected->name;
    });
    if (match == outputs.end() ||
        std::count_if(outputs.begin(), outputs.end(), [&](const auto& output) {
          return output.logical_name == expected->name;
        }) != 1)
      return "name: missing or duplicate publication " + expected->name;
    const auto& output = *match;
    const auto detail = " output=" + output.logical_name + " dtype=" + output.dtype +
                        " bytes=" + std::to_string(output.size_bytes);
    std::uint32_t dtype = 0;
    if (!sima::tensorsemantics::dtype_token_to_ev(output.dtype, &dtype) ||
        dtype != SIMA_EV_DTYPE_FP32)
      return "dtype: expected semantic Float32;" + detail;
    // The MPK authors the publication rank, including its batch axis. The
    // legacy logical_shape helper removes a leading singleton for image-geometry
    // queries; it is not the public tensor-shape authority. Never normalize the
    // expected or actual shape merely because their dense byte counts agree.
    const auto& shape = expected->mpk_shape;
    if (shape.empty())
      return "oracle: MPK dequant shape is missing for " + expected->name;
    std::uint64_t expected_bytes = sizeof(float);
    for (const auto extent : shape) {
      if (extent <= 0 || static_cast<std::uint64_t>(extent) >
                             std::numeric_limits<std::uint64_t>::max() / expected_bytes)
        return "oracle: invalid or overflowing MPK shape for " + expected->name;
      expected_bytes *= static_cast<std::uint64_t>(extent);
    }
    if (output.shape != shape) {
      const auto shape_text = [](const auto& dimensions) {
        std::ostringstream text;
        text << '[';
        for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
          if (axis)
            text << ',';
          text << dimensions[axis];
        }
        return text.str() + ']';
      };
      return "shape: publication=" + shape_text(output.shape) + " expected=" + shape_text(shape) +
             " inferred=" + shape_text(expected->logical_shape) + ";" + detail;
    }
    if (output.size_bytes != expected_bytes)
      return "bytes: expected=" + std::to_string(expected_bytes) + ";" + detail;
    const auto& physical = post.runtime_contract.physical_outputs;
    const auto carrier = std::find_if(physical.begin(), physical.end(), [&](const auto& value) {
      return value.physical_index == output.physical_index;
    });
    if (carrier == physical.end() || output.byte_offset < 0 ||
        static_cast<std::uint64_t>(output.byte_offset) > carrier->size_bytes ||
        output.size_bytes > carrier->size_bytes - static_cast<std::uint64_t>(output.byte_offset))
      return "carrier: publication exceeds its physical storage;" + detail;
  }
  return {};
}

void check_dequant_publication_oracle(const simaai::neat::CompiledProcessCvuContract& post,
                                      const std::vector<const sima::MpkTensorContract*>& authored) {
  const auto error = dequant_publication_error(post, authored);
  require(error.empty(), "YOLO fused publication: " + error);
  const auto rejected = [&](const auto& mutation, const std::string& category) {
    auto changed = post;
    mutation(changed);
    const auto failure = dequant_publication_error(changed, authored);
    require(failure.rfind(category + ":", 0) == 0,
            "publication oracle missed " + category + " mutation: " + failure);
  };
  for (const auto* alias : {"FP32", "float32"}) {
    auto changed = post;
    changed.exposed_view.exposed_logical_outputs.front().dtype = alias;
    require(dequant_publication_error(changed, authored).empty(),
            "Float32 spelling must not change publication semantics");
  }
  for (const auto* invalid : {"INT8", "BF16", "not_float32"})
    rejected(
        [&](auto& value) { value.exposed_view.exposed_logical_outputs.front().dtype = invalid; },
        "dtype");
  for (const auto delta : {-1, 1})
    rejected(
        [&](auto& value) {
          auto& bytes = value.exposed_view.exposed_logical_outputs.front().size_bytes;
          bytes = delta < 0 ? bytes - 1U : bytes + 1U;
        },
        "bytes");
  rejected([](auto& value) { value.exposed_view.exposed_logical_outputs.front().size_bytes = 0U; },
           "bytes");
  rejected([](auto& value) { ++value.exposed_view.exposed_logical_outputs.front().shape.back(); },
           "shape");
  rejected(
      [](auto& value) {
        auto& shape = value.exposed_view.exposed_logical_outputs.front().shape;
        shape.insert(shape.begin(), 1);
      },
      "shape");
  rejected(
      [](auto& value) { value.exposed_view.exposed_logical_outputs.front().physical_index = -1; },
      "carrier");
  rejected(
      [](auto& value) {
        const auto index = value.exposed_view.exposed_logical_outputs.front().physical_index;
        for (auto& carrier : value.runtime_contract.physical_outputs)
          if (carrier.physical_index == index)
            carrier.size_bytes = 0U;
      },
      "carrier");
  rejected(
      [](auto& value) {
        value.exposed_view.exposed_logical_outputs.front().logical_name = "not_an_authored_output";
      },
      "name");
  rejected([](auto& value) { value.exposed_view.exposed_logical_outputs.pop_back(); }, "outputs");
  rejected([](auto& value) { value.payload.graph_name = "detess"; }, "fusion");
}

// Logical MPK value names and stage-local carrier names are different namespaces.
// Join by the independently authored MLA input, then prove that producer and
// consumer address the same exact range of the retained frame arena. Backend
// names retain the authored identity even when Ingress publishes a public alias.
// Do not prescribe carrier/public labels or use equal byte counts as identity.
std::string bf16_handoff_error(const simaai::neat::CompiledProcessCvuContract& tess,
                               const simaai::neat::CompiledMlaContract& mla,
                               const sima::MpkTensorContract& authored) {
  const auto& output = tess.runtime_contract;
  const auto& input = mla.runtime_contract;
  if (output.logical_outputs.size() != 1U || output.physical_outputs.size() != 1U ||
      input.physical_inputs.size() != 1U || input.logical_inputs.size() != 1U ||
      input.input_bindings.size() != 1U || tess.exposed_view.exposed_logical_outputs.size() != 1U ||
      tess.exposed_view.exposed_output_order.size() != 1U)
    return "arity: expected one exact CastTess/MLA boundary";
  const auto& logical = output.logical_outputs.front();
  const auto& carrier = output.physical_outputs.front();
  const auto& exposed = tess.exposed_view.exposed_logical_outputs.front();
  const auto& route = tess.exposed_view.exposed_output_order.front();
  const auto& physical = input.physical_inputs.front();
  const auto& binding = input.input_bindings.front();
  const auto detail = " authored=" + authored.name + " producer=" + logical.logical_name +
                      " carrier=" + carrier.segment_name + " mla=" + physical.segment_name +
                      " logical_index=" + std::to_string(logical.logical_index) +
                      " logical_physical=" + std::to_string(logical.physical_index) +
                      " carrier_index=" + std::to_string(carrier.physical_index) +
                      " carrier_source=" + std::to_string(carrier.source_physical_index) +
                      " binding_logical=" + std::to_string(binding.src_logical_output_index) +
                      " binding_physical=" + std::to_string(binding.src_physical_output_index);
  if (authored.name.empty() || logical.logical_name.empty() ||
      logical.backend_name != authored.name || exposed.backend_name != authored.name ||
      exposed.logical_name != logical.logical_name ||
      route.cm_output_name != logical.logical_name ||
      tess.exposed_view.primary_output_name != logical.logical_name ||
      physical.segment_name != authored.name || binding.cm_input_name != authored.name ||
      binding.source_segment_name != authored.name)
    return "identity:" + detail;
  if (carrier.segment_name.empty() || carrier.physical_index < 0 ||
      carrier.source_physical_index < 0 || logical.logical_index < 0 || logical.output_slot < 0 ||
      logical.physical_index != carrier.physical_index ||
      exposed.logical_index != logical.logical_index ||
      exposed.output_slot != logical.output_slot || logical.segment_name != carrier.segment_name ||
      exposed.physical_index != carrier.physical_index ||
      exposed.segment_name != carrier.segment_name || route.segment_name != carrier.segment_name ||
      route.logical_output_index != logical.logical_index ||
      route.output_slot != logical.output_slot ||
      binding.src_logical_output_index != logical.logical_index ||
      binding.src_output_slot != logical.output_slot ||
      binding.src_physical_output_index != carrier.source_physical_index ||
      physical.source_physical_index != carrier.source_physical_index)
    return "mapping:" + detail;
  const auto contains = [](std::uint64_t capacity, std::int64_t offset, std::uint64_t bytes) {
    return offset >= 0 && static_cast<std::uint64_t>(offset) <= capacity &&
           bytes <= capacity - static_cast<std::uint64_t>(offset);
  };
  if (authored.size_bytes == 0U || logical.size_bytes != authored.size_bytes ||
      exposed.size_bytes != authored.size_bytes || carrier.size_bytes != authored.size_bytes ||
      physical.size_bytes != authored.size_bytes ||
      binding.src_physical_size_bytes != authored.size_bytes || logical.byte_offset != 0 ||
      exposed.byte_offset != 0 ||
      !contains(carrier.size_bytes, logical.byte_offset, logical.size_bytes))
    return "span: expected=" + std::to_string(authored.size_bytes) +
           " producer=" + std::to_string(carrier.size_bytes) +
           " mla=" + std::to_string(physical.size_bytes) + detail;
  if (output.frame_arena_size_bytes == 0U ||
      input.frame_arena_size_bytes != output.frame_arena_size_bytes ||
      output.frame_arena_role != sima::FrameArenaRole::Allocate ||
      input.frame_arena_role != sima::FrameArenaRole::ReuseInput ||
      physical.address_source != sima::PhysicalAddressSource::FrameArenaSpan ||
      physical.source_byte_offset != carrier.source_byte_offset ||
      binding.src_physical_byte_offset != carrier.source_byte_offset ||
      !contains(output.frame_arena_size_bytes, carrier.source_byte_offset, carrier.size_bytes))
    return "placement: producer_offset=" + std::to_string(carrier.source_byte_offset) +
           " mla_offset=" + std::to_string(physical.source_byte_offset) + detail;
  return {};
}

void check_bf16_handoff_oracle(const simaai::neat::CompiledProcessCvuContract& tess,
                               const simaai::neat::CompiledMlaContract& mla,
                               const sima::MpkTensorContract& authored) {
  const auto error = bf16_handoff_error(tess, mla, authored);
  require(error.empty(), "BF16 handoff: " + error);
  auto renamed = tess;
  const std::string replacement_name = "independently_renamed_producer_carrier";
  renamed.runtime_contract.physical_outputs.front().segment_name = replacement_name;
  renamed.runtime_contract.logical_outputs.front().segment_name = replacement_name;
  renamed.exposed_view.exposed_logical_outputs.front().segment_name = replacement_name;
  renamed.exposed_view.exposed_output_order.front().segment_name = replacement_name;
  require(bf16_handoff_error(renamed, mla, authored).empty(),
          "consistent producer carrier renaming must not change authored MLA edge identity");
  const auto rejected = [&](const auto& mutate, const std::string& category) {
    auto changed_tess = tess;
    auto changed_mla = mla;
    mutate(changed_tess, changed_mla);
    const auto failure = bf16_handoff_error(changed_tess, changed_mla, authored);
    require(failure.starts_with(category + ":"),
            "BF16 handoff oracle missed " + category + " mutation: " + failure);
  };
  rejected(
      [](auto& producer, auto&) {
        producer.runtime_contract.logical_outputs.front().logical_name += "_wrong";
      },
      "identity");
  rejected([](auto& producer,
              auto&) { ++producer.runtime_contract.logical_outputs.front().physical_index; },
           "mapping");
  rejected(
      [](auto&, auto& consumer) {
        ++consumer.runtime_contract.input_bindings.front().src_output_slot;
      },
      "mapping");
  rejected([](auto& producer,
              auto&) { --producer.runtime_contract.physical_outputs.front().size_bytes; },
           "span");
  rejected(
      [](auto&, auto& consumer) {
        ++consumer.runtime_contract.physical_inputs.front().source_byte_offset;
      },
      "placement");
  rejected(
      [](auto&, auto& consumer) {
        ++consumer.runtime_contract.input_bindings.front().src_physical_byte_offset;
      },
      "placement");
}

void require_default_sync_retention_contract(const simaai::neat::Model& model) {
  using namespace simaai::neat;
  namespace contract = pipeline_internal::sima;
  namespace query = pipeline_internal::rendered_stage_query;

  // The same public fragments as sync_yolov8_test_detessdequant, with a concrete
  // image ingress but no execution-depth override or injected stage selectors.
  auto input = model.input_appsrc_options(false);
  input.width = 1920;
  input.height = 1080;
  std::vector<std::shared_ptr<Node>> nodes{nodes::Input(input)};
  const auto preproc = internal::ModelAccess::build_public_preprocess_nodes(model);
  const auto infer = internal::ModelAccess::build_public_inference_nodes(model);
  nodes.insert(nodes.end(), preproc.begin(), preproc.end());
  nodes.insert(nodes.end(), infer.begin(), infer.end());
  nodes.push_back(std::make_shared<DetessDequant>(DetessDequantOptions(model)));
  nodes.push_back(nodes::Output());
  nodes = session_build_materialize_model_bound_nodes(nodes, true);
  session_build_apply_derived_input_contracts(&nodes);
  auto build = build_pipeline_full(nodes, false, "mysink", false, {});
  session_build_compile_contracts(&build, nodes, ContractCompileInput{},
                                  "unit_modelpack_mla_handoff_segment_test", &nodes);
  require(build.manifest_diagnostics.errors.empty() && build.rendered_manifest.has_value(),
          "real YOLO sync route must compile its MPK-derived typed manifest");
  const auto& manifest = *build.rendered_manifest;
  const auto* owner = query::find_preproc_stage(manifest);
  const auto* mla = query::find_mla_stage(manifest);
  const auto* terminal = query::find_processcvu_stage(manifest, "detessdequant");
  require(owner && mla && terminal &&
              owner->frame_arena_role == contract::FrameArenaRole::Allocate &&
              mla->frame_arena_role == contract::FrameArenaRole::ReuseInput &&
              terminal->frame_arena_role == contract::FrameArenaRole::ReuseInput,
          "real YOLO serial route must reuse the preproc-allocated frame arena");
  require(mla->physical_outputs.size() == 1U && mla->logical_outputs.size() == 6U &&
              terminal->input_bindings.size() == 6U,
          "retention regression must cover the real six-head packed MLA handoff");
  for (const auto* consumer : {mla, terminal}) {
    std::string error;
    const auto index = static_cast<std::size_t>(consumer - manifest.stages.data());
    const auto edges =
        contract::edgecontract::resolve_consumer_edge_contracts_exact(manifest, index, &error);
    require(error.empty() && edges.size() == consumer->input_bindings.size(),
            "real compiler-authored value bindings must resolve without synthesized selectors");
    for (const auto& edge : edges) {
      require(edge.producer_stage == (consumer == mla ? owner : mla),
              "all real YOLO views must resolve through MLA to the actual preproc owner");
    }
  }
  const auto pipeline = session_build_clamp_sync_build_result(build, -1);
  std::size_t numerical_stages = 0U;
  for (const auto& element : contract::parse_pipeline_elements(pipeline)) {
    if (element.plugin == "neatkernelrunner") {
      ++numerical_stages;
      require(element.fragment.find("num-buffers=") == std::string::npos &&
                  element.fragment.find("output-pool-min-buffers=") == std::string::npos,
              "Runtime storage capacities must not become legacy runner properties");
    } else if (element.plugin == "neatprocessmla") {
      require(element.fragment.find("num-buffers=1") != std::string::npos,
              "default sync MLA execution retains its independent serial window");
    }
  }
  require(numerical_stages >= 2U,
          "preproc and terminal transforms must both use the numerical runner");
  require(build.kernel_runner_output_pool_capacity >= 2U,
          "the typed runner pool must retain a carrier while the next serial frame executes");
}

} // namespace

RUN_TEST(
    "unit_modelpack_mla_handoff_segment_test", ([] {
      sima_test::initialize_cold_contract_gstreamer();
      using namespace simaai::neat;

      const std::filesystem::path core_root = sima_test::test_source_root();
      const std::string tar_path = sima_test::resolve_yolov8s_strict_mpk_tar(core_root);
      require(!tar_path.empty(), "expected modelzoo-backed yolo_v8s .tar.gz MPK with *_mpk.json");

      Model::Options model_opt;
      model_opt.preprocess.kind = InputKind::Image;
      model_opt.preprocess.enable = AutoFlag::On;
      model_opt.preprocess.color_convert.input_format = PreprocessColorFormat::BGR;
      model_opt.upstream_name = "decoder";
      Model model(tar_path, model_opt);
      const auto& pack = internal::ModelAccess::pack(model);
      pack.prepare_for_execution();
      const auto infer_nodes = internal::ModelAccess::build_public_inference_nodes(model);
      require(!infer_nodes.empty(), "inference fragment should compile from the YOLOv8 asset");

      pipeline_internal::sima::ManifestBuildDiagnostics diagnostics;
      const auto compiled =
          compile_node_contracts(infer_nodes, ContractCompileInput{}, &diagnostics);
      require(diagnostics.errors.empty(), "inference fragment contract compile failed");
      require(!compiled.stages.empty(),
              "compiled inference fragment should emit a container stage");
      const CompiledNodeContract* mla_stage = nullptr;
      const auto visit_stage = [&](const auto& self, const CompiledNodeContract& stage) -> void {
        if (!mla_stage && stage.processmla.has_value()) {
          mla_stage = &stage;
        }
        for (const auto& child : stage.child_stages) {
          self(self, child);
        }
      };
      for (const auto& stage : compiled.stages) {
        visit_stage(visit_stage, stage);
      }

      require(mla_stage != nullptr, "compiled full fragment should include an MLA stage");
      const auto infer_stage_facts =
          pack.stage_facts_for_model_stage(internal::ModelStage::MlaOnly);
      const auto pre_stage_facts =
          pack.stage_facts_for_model_stage(internal::ModelStage::Preprocess);
      const internal::ModelFragment::StageFacts* preproc_stage_fact = nullptr;
      for (const auto& fact : pre_stage_facts) {
        if (fact.processcvu_contract.has_value()) {
          preproc_stage_fact = &fact;
        }
      }
      require(preproc_stage_fact != nullptr,
              "YOLOv8 should expose a canonical preproc processcvu stage fact");
      require(preproc_stage_fact->processcvu_contract.has_value(),
              "YOLOv8 preproc stage fact should cache a processcvu contract");
      const auto preproc_handoff =
          pipeline_internal::sima::resolve_processcvu_single_handoff_output(
              *preproc_stage_fact->processcvu_contract);
      require(preproc_handoff.has_value(),
              "YOLOv8 preproc contract should resolve one canonical MLA handoff output");
      require(mla_stage->processmla->runtime_contract.input_bindings.size() == 1U,
              "YOLOv8 MLA stage should expose one input binding");
      // Binding source_segment is now the MPK transform name; the canonical
      // handoff alias is used to validate logical-output-slot identity.
      require(mla_stage->processmla->runtime_contract.logical_inputs.size() == 1U,
              "YOLOv8 MLA stage should expose one logical input contract");
      require(mla_stage->processmla->dispatcher_physical_outputs.size() == 1U,
              "YOLOv8 MLA stage should expose one dispatcher output for the packed MLA parent");
      require(mla_stage->processmla->runtime_contract.physical_outputs.size() == 1U,
              "YOLOv8 MLA stage should keep one physical output for the packed MLA parent");
      require(mla_stage->processmla->runtime_contract.logical_outputs.size() == 6U,
              "YOLOv8 MLA stage should expose six logical MLA views over the packed parent");
      for (const auto& logical : mla_stage->processmla->runtime_contract.logical_outputs) {
        require(logical.physical_index == 0,
                "YOLOv8 MLA logical outputs should all map to the single packed parent");
      }
      const internal::ModelFragment::StageFacts* mla_stage_fact = nullptr;
      for (const auto& fact : infer_stage_facts) {
        if (fact.mla_compiled.has_value()) {
          mla_stage_fact = &fact;
          break;
        }
      }
      require(mla_stage_fact != nullptr, "YOLOv8 should expose a canonical MLA stage fact");
      require(mla_stage_fact->mla_compiled->runtime_contract.input_bindings.size() == 1U,
              "YOLOv8 MLA stage fact should expose one input binding");
      require(mla_stage_fact->mla_compiled->runtime_contract.input_bindings.front()
                          .src_logical_output_index == preproc_handoff->logical_output_index &&
                  mla_stage_fact->mla_compiled->runtime_contract.input_bindings.front()
                          .src_output_slot == preproc_handoff->output_slot,
              "YOLOv8 MLA stage fact should preserve the exact compiler-authored handoff");

      const auto post_plan = pack.execution_plan().post;
      const auto post_stage_facts =
          pack.stage_facts_for_model_stage(internal::ModelStage::Postprocess);
      require(post_plan.size() == post_stage_facts.size(),
              "YOLOv8 post stage facts should align with the execution plan");
      bool checked_fused_publication = false;
      for (std::size_t i = 0; i < post_plan.size(); ++i) {
        if (!post_stage_facts[i].processcvu_contract.has_value()) {
          continue;
        }
        const auto& post = *post_stage_facts[i].processcvu_contract;
        require(post_plan[i].kind == internal::ExecutionStageKind::DetessDequant &&
                    post.payload.graph_name == "detessdequant",
                "YOLO INT8 postprocessing must retain mandatory DetessDequant fusion");
        std::vector<const pipeline_internal::sima::MpkTensorContract*> authored_outputs;
        for (const auto& plugin : pack.mpk_contract()->plugins) {
          if (plugin.kernel == "dequantization_transform" || plugin.kernel == "dequantize" ||
              plugin.kernel == "dequant") {
            for (const auto& tensor : plugin.output_tensors)
              authored_outputs.push_back(&tensor);
          }
        }
        require(!authored_outputs.empty() &&
                    post.runtime_contract.logical_outputs.size() == authored_outputs.size() &&
                    post.exposed_view.exposed_logical_outputs.size() == authored_outputs.size(),
                "production fused post-stage must retain every manifest-authored dequant output");
        check_dequant_publication_oracle(post, authored_outputs);
        checked_fused_publication = true;
        require(!post.exposed_view.primary_output_name.empty() &&
                    !post.exposed_view.exposed_output_order.empty() &&
                    post.exposed_view.primary_output_name ==
                        post.exposed_view.exposed_output_order.front().cm_output_name,
                "post-stage primary output must name its selected first publication");
        break;
      }

      require(checked_fused_publication, "YOLO must expose a fused post-stage to validate");
      require_default_sync_retention_contract(model);

      const std::vector<std::filesystem::path> bf16_candidates = {
          sima_test::resolve_yolov8_variant_fixture("yolov8n_A_W_BF16_mpk").tar_path,
          core_root / "tmp" / "yolov8n_drive" / "yolov8n_A_BF16_W_INT8_mpk.tar.gz",
      };
      bool tested_bf16 = false;
      for (const auto& bf16_path : bf16_candidates) {
        if (!std::filesystem::exists(bf16_path)) {
          continue;
        }

        Model::Options bf16_opt;
        bf16_opt.preprocess.kind = InputKind::Tensor;
        bf16_opt.preprocess.enable = AutoFlag::On;
        Model bf16_model(bf16_path.string(), bf16_opt);
        tested_bf16 = true;

        const CastTessOptions tess_opt(bf16_model);
        require(tess_opt.compiled_contract != nullptr,
                "BF16 tensor route should expose a canonical tess compiled contract");

        const auto& bf16_pack = internal::ModelAccess::pack(bf16_model);
        const auto execution_plan = bf16_pack.execution_plan();
        const auto pre_stage_facts =
            bf16_pack.stage_facts_for_model_stage(internal::ModelStage::Preprocess);
        require(execution_plan.pre.size() == pre_stage_facts.size(),
                "BF16 tensor route pre stage facts should align with the execution plan");

        const internal::ModelFragment::StageFacts* tess_stage_fact = nullptr;
        for (std::size_t i = 0; i < execution_plan.pre.size(); ++i) {
          if (execution_plan.pre[i].kind == internal::ExecutionStageKind::CastTess) {
            tess_stage_fact = &pre_stage_facts[i];
            break;
          }
        }
        require(tess_stage_fact != nullptr,
                "BF16 tensor route should include a tess stage in the canonical pre stage facts");
        require(tess_stage_fact->processcvu_contract.has_value(),
                "BF16 tensor route tess stage fact should cache a processcvu contract");

        const auto& canonical_tess = *tess_stage_fact->processcvu_contract;
        const auto* authored_mla =
            pipeline_internal::sima::get_mla_stage_io_contract(*bf16_pack.mpk_contract());
        require(authored_mla && authored_mla->input_tensors.size() == 1U,
                "BF16 manifest must author one exact MLA input carrier");
        const auto& expected_segment = authored_mla->input_tensors.front().name;
        require(canonical_tess.payload.graph_name == "casttess" &&
                    canonical_tess.payload.input_dtype == "FP32" &&
                    canonical_tess.payload.output_dtype == "BF16" &&
                    canonical_tess.payload.input_tensors.size() == 1U &&
                    canonical_tess.payload.output_tensors.size() == 1U &&
                    canonical_tess.payload.output_tensors.front().descriptor().storage.nbytes ==
                        authored_mla->input_tensors.front().size_bytes,
                "BF16 handoff must retain one fused FP32-to-BF16 CastTess outer boundary");
        require(!canonical_tess.exposed_view.primary_output_name.empty() &&
                    !canonical_tess.exposed_view.exposed_output_order.empty() &&
                    canonical_tess.exposed_view.primary_output_name ==
                        canonical_tess.exposed_view.exposed_output_order.front().cm_output_name,
                "BF16 CastTess primary must name the admitted selected publication");
        require(canonical_tess.runtime_contract.physical_outputs.size() == 1U,
                "BF16 tensor route canonical tess contract should expose one physical output");
        // The fused producer publishes the MPK value through its own carrier
        // namespace. Cross-stage identity is proved below by value + placement,
        // not by requiring its generated label to equal an MLA port label.
        const auto handoff =
            pipeline_internal::sima::resolve_processcvu_single_handoff_output(canonical_tess);
        require(handoff.has_value(),
                "BF16 tensor route canonical tess contract should resolve a single handoff output");
        require(handoff->segment_name ==
                    canonical_tess.runtime_contract.physical_outputs.front().segment_name,
                "shared processcvu handoff query should preserve its selected carrier segment");
        require(handoff->logical_output_index >= 0,
                "shared processcvu handoff query should preserve the tess logical output index");
        require(handoff->output_slot >= 0,
                "shared processcvu handoff query should preserve the tess output slot");
        require(handoff->physical_index ==
                    canonical_tess.runtime_contract.physical_outputs.front().physical_index,
                "shared processcvu handoff query should preserve physical index");
        require(tess_opt.compiled_contract->exposed_view.primary_output_name ==
                    canonical_tess.exposed_view.primary_output_name,
                "CastTessOptions(Model) should reuse the canonical stage-fact primary output");
        require(
            tess_opt.compiled_contract->runtime_contract.physical_outputs.front().segment_name ==
                canonical_tess.runtime_contract.physical_outputs.front().segment_name,
            "CastTessOptions(Model) should reuse the canonical tess physical output segment");

        const auto infer_stage_facts =
            bf16_pack.stage_facts_for_model_stage(internal::ModelStage::MlaOnly);
        const internal::ModelFragment::StageFacts* mla_stage_fact = nullptr;
        for (const auto& fact : infer_stage_facts) {
          if (fact.mla_compiled.has_value()) {
            mla_stage_fact = &fact;
            break;
          }
        }
        require(mla_stage_fact != nullptr,
                "BF16 tensor route should expose an MLA stage fact for infer");
        check_bf16_handoff_oracle(canonical_tess, *mla_stage_fact->mla_compiled,
                                  authored_mla->input_tensors.front());
        require(mla_stage_fact->mla_compiled->runtime_contract.physical_inputs.size() == 1U,
                "BF16 tensor route MLA stage fact should preserve one physical input");
        require(
            mla_stage_fact->mla_compiled->runtime_contract.physical_inputs.front().segment_name ==
                expected_segment,
            "BF16 tensor route MLA input should match the canonical tess handoff segment");
        require(
            mla_stage_fact->mla_compiled->runtime_contract.logical_inputs.front().size_bytes ==
                mla_stage_fact->mla_compiled->runtime_contract.physical_inputs.front().size_bytes,
            "BF16 tensor route MLA logical input should preserve the packed tess handoff byte "
            "size");
        // MLA preserves the tess producer's model semantics over the packed
        // byte envelope; HW/rank-2 were presentation choices of the old bridge.
        // Derive expected semantic facts from the MPK producer, not from either
        // compiled side of this handoff or the physical byte-shaped descriptor.
        const pipeline_internal::sima::MpkTensorContract* authored_tess = nullptr;
        for (const auto& plugin : bf16_pack.mpk_contract()->plugins) {
          if (plugin.processor == "MLA")
            continue;
          for (const auto& tensor : plugin.output_tensors) {
            if (tensor.name != expected_segment)
              continue;
            require(authored_tess == nullptr, "BF16 MLA input has ambiguous MPK producer");
            authored_tess = &tensor;
          }
        }
        require(authored_tess && !authored_tess->logical_shape.empty() &&
                    !authored_tess->logical_dtype.empty() &&
                    authored_tess->size_bytes == authored_mla->input_tensors.front().size_bytes,
                "BF16 MPK producer must author exact semantic geometry and packed bytes");
        const auto& mla_logical =
            mla_stage_fact->mla_compiled->runtime_contract.logical_inputs.front();
        std::uint32_t authored_dtype = 0U;
        std::uint32_t mla_dtype = 0U;
        require(pipeline_internal::sima::tensorsemantics::dtype_token_to_ev(
                    authored_tess->logical_dtype, &authored_dtype) &&
                    pipeline_internal::sima::tensorsemantics::dtype_token_to_ev(mla_logical.dtype,
                                                                                &mla_dtype) &&
                    authored_dtype == SIMA_EV_DTYPE_BF16 && mla_dtype == authored_dtype &&
                    mla_logical.shape == authored_tess->logical_shape,
                "BF16 MLA semantic input must preserve MPK tess producer dtype/shape, not a "
                "legacy byte-shaped presentation: actual_dtype=" +
                    mla_logical.dtype + " authored_dtype=" + authored_tess->logical_dtype);
        require(
            mla_stage_fact->mla_compiled->runtime_contract.logical_inputs.front().segment_name ==
                expected_segment,
            "BF16 tensor route MLA logical input should preserve the canonical tess handoff "
            "segment");
        require(
            mla_stage_fact->mla_compiled->runtime_contract.physical_inputs.front()
                    .source_physical_index ==
                canonical_tess.runtime_contract.physical_outputs.front().physical_index,
            "BF16 tensor route MLA input should follow the canonical tess physical output index");
        require(mla_stage_fact->mla_compiled->runtime_contract.input_bindings.front()
                        .src_logical_output_index == handoff->logical_output_index,
                "BF16 tensor route MLA input binding should preserve the canonical tess logical "
                "output index");
        require(
            mla_stage_fact->mla_compiled->runtime_contract.input_bindings.front().src_output_slot ==
                handoff->output_slot,
            "BF16 tensor route MLA input binding should preserve the canonical tess output slot");

        const auto post_stage_facts =
            bf16_pack.stage_facts_for_model_stage(internal::ModelStage::Postprocess);
        const internal::ModelFragment::StageFacts* detess_stage_fact = nullptr;
        for (const auto& fact : post_stage_facts) {
          if (fact.processcvu_contract.has_value() &&
              fact.processcvu_contract->runtime_contract.input_bindings.size() > 1U) {
            detess_stage_fact = &fact;
            break;
          }
        }
        require(detess_stage_fact != nullptr,
                "BF16 tensor route should expose a packed detess processcvu stage fact");
        require(detess_stage_fact->processcvu_contract->runtime_contract.input_bindings.size() ==
                    mla_stage_fact->mla_compiled->runtime_contract.logical_outputs.size(),
                "BF16 detess stage should bind every published MLA logical output");
        const auto& detess = *detess_stage_fact->processcvu_contract;
        const auto& mla_outputs = mla_stage_fact->mla_compiled->runtime_contract;
        require(detess.runtime_contract.logical_inputs.size() ==
                    detess.runtime_contract.input_bindings.size(),
                "BF16 DetessCast must provide one logical input per exact binding");
        require(detess.payload.graph_name == "detesscast" && detess.payload.input_dtype == "BF16" &&
                    detess.payload.output_dtype == "FP32",
                "BF16 postprocessing must retain mandatory DetessCast fusion");
        require(mla_outputs.physical_outputs.size() == authored_mla->output_tensors.size() &&
                    mla_outputs.physical_outputs.size() == 1U &&
                    detess.runtime_contract.frame_arena_size_bytes ==
                        mla_outputs.frame_arena_size_bytes,
                "BF16 packed heads must share the manifest-authored MLA output arena");
        const auto contains = [](std::uint64_t capacity, std::int64_t offset, std::uint64_t bytes) {
          return offset >= 0 && static_cast<std::uint64_t>(offset) <= capacity &&
                 bytes <= capacity - static_cast<std::uint64_t>(offset);
        };
        for (std::size_t i = 0; i < detess.runtime_contract.input_bindings.size(); ++i) {
          const auto& binding = detess.runtime_contract.input_bindings[i];
          const auto& input = detess.runtime_contract.logical_inputs[i];
          const auto& mla_output = mla_outputs.logical_outputs[i];
          const auto producer =
              std::find_if(mla_outputs.physical_outputs.begin(), mla_outputs.physical_outputs.end(),
                           [&](const auto& physical) {
                             return physical.physical_index == mla_output.physical_index;
                           });
          const auto consumer = std::find_if(
              detess.runtime_contract.physical_inputs.begin(),
              detess.runtime_contract.physical_inputs.end(), [&](const auto& physical) {
                return physical.physical_index == input.physical_index;
              });
          require(producer != mla_outputs.physical_outputs.end() &&
                      consumer != detess.runtime_contract.physical_inputs.end() &&
                      std::count_if(detess.runtime_contract.physical_inputs.begin(),
                                    detess.runtime_contract.physical_inputs.end(),
                                    [&](const auto& physical) {
                                      return physical.physical_index == input.physical_index;
                                    }) == 1,
                  "BF16 DetessCast head must resolve both exact physical carriers");
          require(binding.src_logical_output_index == mla_output.logical_index &&
                      binding.src_output_slot == mla_output.output_slot &&
                      input.logical_name == mla_output.logical_name &&
                      input.size_bytes == mla_output.size_bytes,
                  "BF16 DetessCast must retain each authored MLA logical head identity and size");
          require(
              binding.src_physical_output_index == producer->source_physical_index &&
                  consumer->source_physical_index == producer->source_physical_index &&
                  contains(producer->size_bytes, mla_output.byte_offset, mla_output.size_bytes) &&
                  contains(consumer->size_bytes, input.byte_offset, input.size_bytes) &&
                  contains(mla_outputs.frame_arena_size_bytes, producer->source_byte_offset,
                           producer->size_bytes) &&
                  contains(mla_outputs.frame_arena_size_bytes, consumer->source_byte_offset,
                           consumer->size_bytes),
              "BF16 DetessCast packed head must remain within its source carrier and arena");
          // Command projection may rebase the head into a selected physical
          // input. Compare absolute addresses, not offsets in different views.
          const auto producer_address = static_cast<std::uint64_t>(producer->source_byte_offset) +
                                        static_cast<std::uint64_t>(mla_output.byte_offset);
          const auto consumer_address = static_cast<std::uint64_t>(consumer->source_byte_offset) +
                                        static_cast<std::uint64_t>(input.byte_offset);
          require(producer_address == consumer_address &&
                      binding.src_physical_byte_offset == input.byte_offset &&
                      binding.src_physical_size_bytes == consumer->size_bytes,
                  "BF16 DetessCast must preserve absolute head address across physical-view "
                  "rebasing: head=" +
                      std::to_string(i) + " producer=" + std::to_string(producer_address) +
                      " consumer=" + std::to_string(consumer_address));
        }
        break;
      }
      require(tested_bf16, "BF16 CastTess handoff coverage requires an authentic model fixture");
    }));
