#include <neat.h>
#include "nodes/sima/Preproc.h"
#include "pipeline/internal/contract/ContractFacts.h"
#include "pipeline/internal/contract/ContractCompiler.h"
#include "pipeline/internal/sima/ContractRender.h"
#include "pipeline/internal/sima/stagesemantics/TransportStageSemantics.h"
#include "pipeline/internal/sima/StageRecipePreparation.h"
#include "pipeline/internal/sima/KernelStageRecipe.h"
#include "pipeline/internal/sima/StageMemoryPolicy.h"
#include "pipeline/internal/TerminalOutputContractQuery.h"
#include "test_main.h"
#include <gstsimaaitensorbuffer_meta.h>

namespace {
using namespace simaai::neat;
namespace ps = pipeline_internal::sima;
auto prepare_stages(ps::SimaPluginStaticManifest* manifest) {
  auto status = ps::prepare_kernel_stage_recipes(manifest, std::nullopt, {}, {}, {});
  if (status)
    return status;
  ps::finalize_pipeline_storage(*manifest);
  return ps::finalize_stage_publications(manifest);
}
void temporal_caps() {
  for (const auto rate : {std::pair{30, 1}, std::pair{30000, 1001}, std::pair{0, 1}}) {
    for (const int boundary : {0, 1, 2, 3}) {
      for (const bool seeded_rate : {false, true}) {
        InputOptions input;
        input.payload_type = PayloadType::Image;
        input.format = FormatTag::RGB;
        input.width = 36;
        input.height = 20;
        input.fps_n = seeded_rate ? 0 : rate.first;
        input.fps_d = seeded_rate ? 1 : rate.second;
        ResizeOptions resize;
        resize.width = 18;
        resize.height = 10;
        ColorConvertOptions color;
        color.output_format = ImageFormat::NV12;
        std::vector<std::shared_ptr<Node>> nodes{nodes::Input("image", input),
                                                 nodes::Resize(resize)};
        if (boundary == 1 || boundary == 2)
          nodes.push_back(nodes::VideoRate());
        if (boundary == 3)
          nodes.push_back(nodes::Custom("identity"));
        if (boundary == 2)
          nodes.push_back(nodes::CapsRaw("RGB", 18, 10, 24));
        nodes.push_back(nodes::ColorConvert(color));
        // Exercise canonical upstream boundary projection, not a test-authored
        // ProcessCVU payload or fabricated GstCaps. The two stages can have
        // different authored rates when a rate filter intervenes.
        OutputSpec seed;
        if (seeded_rate) {
          seed.fps_num = rate.first;
          seed.fps_den = rate.second;
        }
        const auto output = derive_output_spec(nodes, seed);
        ContractCompileInput compile_input;
        compile_input.ingress.ingress_spec = seed;
        ps::ManifestBuildDiagnostics diagnostics;
        auto compiled = compile_node_contracts(nodes, compile_input, &diagnostics);
        require(compiled.fully_renderable && diagnostics.errors.empty(),
                "temporal image contract compilation failed");
        auto manifest =
            render_manifest_from_compiled_contracts(compiled, compile_input, &diagnostics);
        require(manifest && manifest->stages.size() == 2 && diagnostics.errors.empty(),
                "temporal image render failed");
        auto failure = prepare_stages(&*manifest);
        require(!failure, failure ? failure->field + ": " + failure->message : "");
        for (std::size_t i = 0; i < manifest->stages.size(); ++i) {
          const auto* caps =
              gst_caps_get_structure(manifest->stages[i].tensor_publication->output_caps.get(), 0);
          const auto expected = !i || !boundary ? rate
                                : boundary == 2 ? std::pair{24, 1}
                                                : std::pair{0, 1};
          int numerator = -1, denominator = -1;
          const bool present =
              gst_structure_get_fraction(caps, "framerate", &numerator, &denominator);
          require(expected.first > 0
                      ? present && numerator == expected.first && denominator == expected.second
                      : !present,
                  "image stage lost or invented its upstream framerate");
        }
        const auto expected = !boundary ? rate : boundary == 2 ? std::pair{24, 1} : std::pair{0, 1};
        // The public best-effort projector still skips opaque nodes for legacy
        // compatibility. Only strict image publication invalidates that rate.
        if (boundary == 0 || boundary == 2)
          require(output.fps_num == expected.first && output.fps_den == expected.second,
                  "image OutputSpec did not preserve its upstream rate");
      }
    }
  }
  for (const int denominator : {0, -1}) {
    InputOptions input;
    input.payload_type = PayloadType::Image;
    input.format = FormatTag::RGB;
    input.width = 36;
    input.height = 20;
    ResizeOptions resize;
    resize.width = 18;
    resize.height = 10;
    std::vector<std::shared_ptr<Node>> nodes{nodes::Input("image", input), nodes::Resize(resize)};
    OutputSpec seed;
    seed.fps_num = 30;
    seed.fps_den = denominator;
    derive_output_spec(nodes, seed);
    ContractCompileInput compile_input;
    compile_input.ingress.ingress_spec = seed;
    ps::ManifestBuildDiagnostics diagnostics;
    auto compiled = compile_node_contracts(nodes, compile_input, &diagnostics);
    require(!compiled.fully_renderable && !diagnostics.errors.empty() &&
                diagnostics.errors.front().find("framerate") != std::string::npos,
            "invalid known frame rate was silently replaced");
  }
}
void require_image_query(const ps::SimaPluginStaticManifest& manifest) {
  namespace terminal = pipeline_internal::terminal_output_contract;
  const auto& stage = manifest.stages.back();
  require(bool(stage.image_publication), "renderer lost checked semantic image authority");
  for (const int selection : {0, 1, 2, 3}) {
    terminal::PublicOutputEndpointSelector endpoint;
    if (selection) {
      endpoint.terminal_stage_key = stage.logical_stage_id;
      endpoint.terminal_stage_key_required = true;
      endpoint.allow_unresolved_terminal_stage_fallback = true;
    }
    if (selection == 2)
      endpoint.output_segment_name = stage.logical_outputs.back().logical_name;
    if (selection == 3)
      endpoint.output_segment_name = "not_an_authored_output";
    auto reason = terminal::OutputOverrideFailureReason::InvalidPublicationContract;
    std::string diagnostic = "stale diagnostic";
    const auto override =
        terminal::build_output_override_from_manifest(manifest, endpoint, &diagnostic, &reason);
    if (selection == 3) {
      require(!override && !diagnostic.empty() &&
                  reason == terminal::OutputOverrideFailureReason::EndpointSelectorNoMatch,
              "raw image suppressed an invalid endpoint selector");
    } else {
      require(!override && diagnostic.empty() &&
                  reason == terminal::OutputOverrideFailureReason::None,
              "raw image planes became a tensor override or upstream fallback");
    }
  }
  std::string error;
  const auto roundtrip = ps::parse_manifest_json(ps::serialize_manifest_json(manifest), &error);
  require(roundtrip && error.empty() && roundtrip->stages.back().image_publication &&
              ps::to_json(roundtrip->stages.back())["image_publication"] ==
                  ps::to_json(stage)["image_publication"],
          "metadata roundtrip lost relative image publication fields");
  require(!terminal::build_output_override_from_manifest(*roundtrip, {}, &error) && error.empty(),
          "metadata roundtrip restored a tensor-plane override");
}

void require_tensor_query(const ps::SimaPluginStaticManifest& manifest) {
  std::string error;
  const auto override =
      pipeline_internal::terminal_output_contract::build_output_override_from_manifest(manifest, {},
                                                                                       &error);
  require(!manifest.stages.back().image_publication && override && override->outputs.size() == 1 &&
              override->outputs.front().dtype == TensorDType::BFloat16,
          "tensor-valued normalization lost its public tensor override");
}

void render_images() {
  for (const auto format : {ImageFormat::RGB, ImageFormat::BGR, ImageFormat::GRAY8,
                            ImageFormat::NV12, ImageFormat::I420, ImageFormat::NV16}) {
    ColorConvertOptions options;
    options.output_format = format;
    auto node = nodes::ColorConvert(options);
    auto* p = dynamic_cast<Preproc*>(node.get());
    OutputSpec in;
    in.media_type = "video/x-raw";
    in.format = format == ImageFormat::GRAY8 ? "GRAY8" : "RGB";
    in.dtype = "UInt8";
    in.width = 18;
    in.height = 10;
    in.depth = format == ImageFormat::GRAY8 ? 1 : 3;
    p->output_spec(in);
    CompiledNodeContract compiled_node;
    std::string error;
    require(p->compile_node_contract({}, &compiled_node, &error), "compile: " + error);
    require(compiled_node.processcvu &&
                compiled_node.processcvu->runtime_contract.image_publication,
            "node compilation lost checked image metadata");
    const auto projected = project_processcvu_stage_contract(*compiled_node.processcvu);
    require(projected.image_publication ==
                compiled_node.processcvu->runtime_contract.image_publication,
            "private contract projection dropped image authority");
    namespace semantics = ps::stagesemantics;
    semantics::TransportCanonicalFacts facts;
    facts.plugin_kind = "processcvu";
    facts.kernel_kind = "image_transport";
    facts.payload_kind = ps::StagePayloadKind::ProcessCvu;
    facts.processcvu_payload = compiled_node.processcvu->payload;
    facts.runtime_contract = semantics::build_transport_runtime_contract_from_processcvu_compiled(
        *compiled_node.processcvu);
    CompiledNodeContract transport;
    transport.node_kind = "Transport";
    transport.element_name = transport.logical_stage_id = "image_transport";
    transport.transport = semantics::build_transport_compiled_contract_from_facts(facts);
    CompiledPipelineContracts transported;
    transported.fully_renderable = true;
    transported.stages.push_back(std::move(transport));
    ps::ManifestBuildDiagnostics transport_diagnostics;
    auto transport_manifest =
        render_manifest_from_compiled_contracts(transported, {}, &transport_diagnostics);
    require(transport_manifest && transport_diagnostics.errors.empty(),
            "image transport rendering failed");
    require_image_query(*transport_manifest);
    CompiledPipelineContracts compiled;
    compiled.fully_renderable = true;
    compiled.stages.push_back(std::move(compiled_node));
    ps::ManifestBuildDiagnostics diagnostics;
    auto manifest = render_manifest_from_compiled_contracts(compiled, {}, &diagnostics);
    require(manifest.has_value() && diagnostics.errors.empty(),
            diagnostics.errors.empty() ? "image render failed" : diagnostics.errors.front());
    const auto expected_planes =
        format == ImageFormat::I420
            ? 3U
            : (format == ImageFormat::NV12 || format == ImageFormat::NV16 ? 2U : 1U);
    require(manifest->stages[0].logical_outputs.size() == expected_planes &&
                manifest->stages[0].physical_outputs.size() == 1 &&
                !manifest->stages[0].frame_layout,
            "render lost one-image plane ownership");
    // GraphBuildInput queries public output semantics immediately after render,
    // before GraphBuildPipeline prepares recipes. Test that actual phase order.
    require(!manifest->stages[0].native_kernel, "test must query before recipe preparation");
    require_image_query(*manifest);
    const auto authored_stage = manifest->stages[0];
    const auto authored_image = authored_stage.image_publication;
    const auto authored_json = ps::to_json(authored_stage)["image_publication"];
    // Recheck every descriptor field and presence before recipe attachment.
    for (int change = 0; change != 10; ++change) {
      auto invalid = *manifest;
      auto changed =
          std::make_shared<simaai::scheduler::client::TensorStageImagePublication>(*authored_image);
      switch (change) {
      case 0:
        changed->format = GST_VIDEO_FORMAT_UNKNOWN;
        break;
      case 1:
        ++changed->width;
        break;
      case 2:
        ++changed->height;
        break;
      case 3:
        changed->planes.push_back(changed->planes.front());
        break;
      case 4:
        ++changed->planes.front().backend_output_index;
        break;
      case 5:
        ++changed->planes.front().byte_offset;
        break;
      case 6:
        ++changed->planes.front().stride;
        break;
      case 7:
        ++changed->planes.front().visible_rows;
        break;
      case 8:
        ++changed->planes.front().visible_row_bytes;
        break;
      case 9:
        changed.reset();
        break;
      }
      invalid.stages[0].image_publication = std::move(changed);
      const auto rejected = prepare_stages(&invalid);
      require(rejected && rejected->field == "stage.image_publication" &&
                  !invalid.stages[0].native_kernel,
              "inconsistent authored image authority reached recipe attachment");
    }
    // A typed policy is not an immutable proof of the later stage. A changed
    // canvas must be checked against the authored publication at cold admission.
    auto changed_canvas = *manifest;
    changed_canvas.stages[0].processcvu.output_shapes[0][1] += 2;
    const auto changed_failure = prepare_stages(&changed_canvas);
    require(changed_failure && !changed_canvas.stages[0].native_kernel,
            "changed raw image canvas reused stale numerical policy");
    auto failure = prepare_stages(&*manifest);
    require(!failure, failure ? failure->field + ": " + failure->message : "");
    const auto& stage = manifest->stages[0];
    const auto& recipe = *stage.native_kernel;
    auto first = recipe.frame_adapter_factory();
    auto second = recipe.frame_adapter_factory();
    using Adapter = simaai::scheduler::client::TensorStageFrameAdapter;
    require(std::holds_alternative<std::shared_ptr<Adapter>>(first) &&
                std::holds_alternative<std::shared_ptr<Adapter>>(second) &&
                std::get<std::shared_ptr<Adapter>>(first) !=
                    std::get<std::shared_ptr<Adapter>>(second),
            "independent admissions shared a mutable image adapter");
    require(recipe.image_publication && recipe.image_publication->width == 18 &&
                recipe.image_publication->height == 10 &&
                recipe.members[0].outputs.size() == expected_planes,
            "lowering confused an image canvas with numerical plane shape");
    const auto* caps = gst_caps_get_structure(stage.tensor_publication->output_caps.get(), 0);
    require(std::string_view(gst_structure_get_name(caps)) == "video/x-raw",
            "raw image route was lowered as tensor caps");
    require_image_query(*manifest);
    auto recipe_stage = authored_stage;
    recipe_stage.image_publication = recipe.image_publication;
    require(recipe.image_publication &&
                ps::to_json(recipe_stage)["image_publication"] == authored_json &&
                ps::to_json(authored_stage)["image_publication"] == authored_json,
            "final image publication must preserve the immutable relative source descriptor");
    require(stage.frame_layout && stage.frame_layout->outputs.size() == expected_planes,
            "final image publication lost its plane bindings");
    for (std::size_t i = 0; i < expected_planes; ++i)
      require(ps::dtype_source_is_public_logical_contract_authoritative(
                  stage.logical_outputs[i].dtype_source) &&
                  stage.frame_layout->outputs[i].tensor ==
                      std::size_t(stage.image_publication->planes[i].backend_output_index) &&
                  stage.image_publication->planes[i].byte_offset ==
                      stage.frame_layout->outputs[i].span.byte_offset &&
                  stage.tensor_publication->contract->logical_outputs[i].byte_offset ==
                      std::int64_t(stage.frame_layout->outputs[i].span.byte_offset) &&
                  stage.tensor_publication->contract->logical_outputs[i].memory_index ==
                      int(stage.frame_layout->outputs[i].span.allocation_index) &&
                  stage.frame_layout->outputs[i].span.byte_offset ==
                      std::uint64_t(stage.physical_outputs[0].source_byte_offset) +
                          std::uint64_t(stage.logical_outputs[i].byte_offset) &&
                  stage.frame_layout->outputs[i].span.size_bytes ==
                      stage.logical_outputs[i].size_bytes,
              "final image metadata differs from its allocation-local plane binding");
  }
  auto normalization = nodes::Normalization();
  auto* p = dynamic_cast<Preproc*>(normalization.get());
  OutputSpec in;
  in.media_type = "video/x-raw";
  in.format = "BGR";
  in.dtype = "UInt8";
  in.width = 18;
  in.height = 10;
  in.depth = 3;
  in.fps_num = 30000;
  in.fps_den = 1001;
  const auto normalized = p->output_spec(in);
  require(normalized.fps_num == 30000 && normalized.fps_den == 1001,
          "normalization OutputSpec lost its upstream rate");
  CompiledNodeContract node;
  std::string error;
  require(p->compile_node_contract({}, &node, &error), "normalization compile: " + error);
  CompiledPipelineContracts compiled;
  compiled.fully_renderable = true;
  compiled.stages.push_back(std::move(node));
  ps::ManifestBuildDiagnostics diagnostics;
  auto manifest = render_manifest_from_compiled_contracts(compiled, {}, &diagnostics);
  require(manifest && diagnostics.errors.empty(), "normalization render failed");
  require(!manifest->stages[0].native_kernel, "normalization queried after preparation");
  require_tensor_query(*manifest);
  auto invalid = *manifest;
  invalid.stages[0].image_publication =
      std::make_shared<simaai::scheduler::client::TensorStageImagePublication>();
  const auto rejected = prepare_stages(&invalid);
  require(rejected && rejected->field == "stage.image_publication" &&
              !invalid.stages[0].native_kernel,
          "tensor normalization accepted forged image authority");
  auto failure = prepare_stages(&*manifest);
  require(!failure, failure ? failure->field + ": " + failure->message : "");
  const auto& recipe = *manifest->stages[0].native_kernel;
  require(!recipe.image_publication && manifest->stages[0].tensor_publication &&
              std::string_view(gst_structure_get_name(gst_caps_get_structure(
                  manifest->stages[0].tensor_publication->output_caps.get(), 0))) ==
                  "application/vnd.simaai.tensor",
          "normalization lowered as raw video");
  require_tensor_query(*manifest);
}
} // namespace
RUN_TEST("image transform cold render", [] {
  gst_init(nullptr, nullptr);
  render_images();
  temporal_caps();
})
