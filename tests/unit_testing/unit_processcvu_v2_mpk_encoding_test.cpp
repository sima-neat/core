#include "pipeline/internal/sima/MpkContract.h"
#include "pipeline/internal/sima/PluginContractSubsets.h"
#include "pipeline/internal/sima/PreparedRuntimeBuild.h"
#include "pipeline/internal/sima/stagesemantics/ProcessCvuRuntimeConfigAdapterInternal.h"
#include "test_main.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gst/gst.h>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

namespace {

namespace ps = simaai::neat::pipeline_internal::sima;
namespace pcs = simaai::neat::pipeline_internal::sima::plugin_contracts;
namespace pss = simaai::neat::pipeline_internal::sima::stagesemantics;
using json = nlohmann::json;

void ensure_gst_ready() {
  static bool ready = false;
  if (!ready) {
    int argc = 0;
    char** argv = nullptr;
    gst_init(&argc, &argv);
    ready = true;
  }
}

std::filesystem::path write_mpk_fixture(const std::string& name, const bool align_c16,
                                        const bool cblock, const std::size_t transport_bytes) {
  const auto root =
      std::filesystem::temp_directory_path() / ("sima_processcvu_v2_encoding_" + name);
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  require(!ec, "failed to create ProcessCVU V2 fixture root");

  json plugins = json::array();
  plugins.push_back({{"name", "MLA_0"},
                     {"sequence", 1},
                     {"processor", "MLA"},
                     {"config_params", {{"desired_batch_size", 1}, {"actual_batch_size", 1}}},
                     {"resources", {{"executable", "fixture.elf"}}},
                     {"input_nodes", {{{"name", "model_input"}, {"type", "buffer"}, {"size", 1}}}},
                     {"output_nodes",
                      {{{"name", "descriptor"},
                        {"type", "buffer"},
                        {"size", transport_bytes},
                        {"dtype", "bfloat16"},
                        {"shape", json::array({1, transport_bytes})}}}}});
  plugins.push_back(
      {{"name", "detess_descriptor"},
       {"sequence", 2},
       {"processor", "EV74"},
       {"config_params",
        {{"desired_batch_size", 1},
         {"actual_batch_size", 1},
         {"kernel", "detessellation_transform"},
         {"params",
          {{"slice_shape", json::array({2, 3, 33})},
           {"align_c16", align_c16},
           {"cblock", cblock},
           {"frame_type", "bfloat16"},
           {"frame_shape", json::array({2, 3, 33})},
           {"input_shapes", json::array({json::array({1, transport_bytes})})},
           {"output_shapes", json::array({json::array({2, 3, 33})})}}}}},
       {"input_nodes", {{{"name", "descriptor"}, {"type", "buffer"}, {"size", transport_bytes}}}},
       {"output_nodes",
        {{{"name", "detess_output"},
          {"type", "buffer"},
          {"size", 396},
          {"dtype", "bfloat16"},
          {"shape", json::array({2, 3, 33})}}}}});

  std::ofstream out(root / "mpk.json");
  require(out.is_open(), "failed to open ProcessCVU V2 fixture manifest");
  out << json{{"name", "processcvu_v2_encoding_fixture"},
              {"model_path", "fixture.elf"},
              {"input_nodes", {{{"name", "model_input"}, {"type", "buffer"}, {"size", 1}}}},
              {"plugins", std::move(plugins)}}
             .dump(2);
  out.close();
  require(out.good(), "failed to finalize ProcessCVU V2 fixture manifest");
  return root;
}

ps::SimaPluginStaticManifest make_manifest(const ps::MpkContract& contract) {
  require(contract.plugins.size() == 2U, "fixture should contain MLA and detess stages");
  const auto& detess = contract.plugins[1];
  const auto subsets = pcs::extract_detessellate_contract_subsets_from_mpk(contract);
  require(subsets.size() == 1U, "fixture should produce one detess subset");
  const std::string output_name = detess.output_tensors.front().name;
  const auto runtime =
      pcs::build_detessellate_runtime_config_from_subsets(subsets, {output_name}, {output_name});
  const auto compiled = pss::build_processcvu_compiled_contract_from_runtime_config(runtime);

  ps::StageStaticSpec stage;
  stage.element_name = detess.name;
  stage.logical_stage_id = detess.name;
  stage.model_managed_stage = true;
  stage.plugin_kind = compiled.runtime_contract.plugin_kind;
  stage.kernel_kind = detess.kernel;
  stage.payload_kind = ps::StagePayloadKind::ProcessCvu;
  stage.processcvu = compiled.payload;
  stage.processcvu.exact_stage_name_or_id = detess.name;
  stage.logical_inputs = compiled.runtime_contract.logical_inputs;
  stage.input_bindings = compiled.runtime_contract.input_bindings;
  stage.physical_inputs = compiled.runtime_contract.physical_inputs;
  stage.physical_outputs = compiled.runtime_contract.physical_outputs;
  stage.logical_outputs = compiled.runtime_contract.logical_outputs;
  stage.output_order = compiled.runtime_contract.output_order;
  stage.output_quant = compiled.runtime_contract.output_quant;

  ps::SimaPluginStaticManifest manifest;
  manifest.model_id = contract.model_name;
  manifest.stages.push_back(std::move(stage));
  return manifest;
}

std::uint32_t prepared_input_flags(const std::filesystem::path& root) {
  ensure_gst_ready();
  std::string error;
  const auto contract = ps::load_mpk_contract_from_pack_root(root.string(), &error);
  require(contract.has_value(), "ProcessCVU V2 fixture should load: " + error);
  const auto manifest = make_manifest(*contract);

  ps::PipelineElementSpec mla_element;
  mla_element.plugin = "neatprocessmla";
  mla_element.model_path_property = root.string();
  const auto prepared = ps::build_prepared_runtime_context(
      nullptr, manifest, std::nullopt, {mla_element}, {}, simaai::neat::NameTransform{}, &error);
  require(prepared.has_value(), "ProcessCVU V2 prepared runtime should build: " + error);
  require(prepared->stages.size() == 1U && prepared->stages.front().processcvu.has_value(),
          "fixture should produce one prepared ProcessCVU stage");
  const auto& tensors = prepared->stages.front().processcvu->typed_config.input_tensors;
  require(tensors.size() == 1U, "prepared ProcessCVU stage should have one input descriptor");
  require(tensors.front().layout_kind == SIMA_EV_LAYOUT_TILED,
          "prepared ProcessCVU input should remain tiled");
  return tensors.front().layout.tiled.flags;
}

simaai::neat::GraphProcessCvuPreparedBridgeAbiV2 matching_probe() {
  simaai::neat::GraphProcessCvuPreparedBridgeAbiV2 probe;
  probe.abi_version = simaai::neat::GRAPH_PROCESSCVU_PREPARED_BRIDGE_ABI_VERSION_V2;
  probe.struct_size = sizeof(simaai::neat::GraphProcessCvuPreparedBridgeAbiV2);
  probe.reserved = 0U;
  probe.tiled_channel_encoding_size = sizeof(simaai::neat::GraphProcessCvuTiledChannelEncoding);
  probe.graph_processcvu_stage_request_size = sizeof(simaai::neat::GraphProcessCvuStageRequest);
  probe.graph_processcvu_stage_request_v2_size =
      sizeof(simaai::neat::GraphProcessCvuStageRequestV2);
  probe.processcvu_prepared_stage_size = sizeof(simaai::gst::ProcessCvuPreparedStage);
  probe.prepared_processcvu_typed_config_size = sizeof(simaai::gst::PreparedProcessCvuTypedConfig);
  return probe;
}

} // namespace

RUN_TEST(
    "unit_processcvu_v2_mpk_encoding_test", ([] {
      require(ps::validate_graph_processcvu_prepared_bridge_v2().empty(),
              "loaded prepared-runtime bridge should expose the matching V2 probe and builder");
      const auto cblock_flags = prepared_input_flags(write_mpk_fixture("cblock", true, true, 576U));
      require(cblock_flags == SIMA_EV_TILED_FLAG_CBLOCK16,
              "MPK cblock must produce exactly the CBlock16 descriptor flag; got " +
                  std::to_string(cblock_flags));
      const auto padded_flags =
          prepared_input_flags(write_mpk_fixture("padded_hwc", true, false, 576U));
      require(padded_flags == SIMA_EV_TILED_FLAG_PADDED_HWC_C16,
              "MPK align_c16 without cblock must produce exactly the padded-HWC descriptor flag; "
              "got " +
                  std::to_string(padded_flags));
      const auto compact_flags =
          prepared_input_flags(write_mpk_fixture("compact", false, false, 396U));
      require(compact_flags == SIMA_EV_TILED_FLAG_COMPACT_CHANNELS,
              "MPK compact channels must produce exactly the compact descriptor flag; got " +
                  std::to_string(compact_flags));

      auto probe = matching_probe();
      require(ps::validate_graph_processcvu_prepared_bridge_v2_abi(&probe, true).empty(),
              "matching V2 bridge ABI should validate");
      require(!ps::validate_graph_processcvu_prepared_bridge_v2_abi(nullptr, true).empty(),
              "missing V2 bridge probe should be rejected before request dispatch");
      require(!ps::validate_graph_processcvu_prepared_bridge_v2_abi(&probe, false).empty(),
              "missing V2 builder should be rejected before request dispatch");
      --probe.struct_size;
      require(!ps::validate_graph_processcvu_prepared_bridge_v2_abi(&probe, true).empty(),
              "a mismatched V2 probe size must be rejected from its fixed prefix");
      probe = matching_probe();
      --probe.abi_version;
      require(!ps::validate_graph_processcvu_prepared_bridge_v2_abi(&probe, true).empty(),
              "a legacy V2 ABI version must be rejected before request dispatch");
    }));
