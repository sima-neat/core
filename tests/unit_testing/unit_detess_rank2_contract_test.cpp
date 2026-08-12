#include "pipeline/internal/sima/MpkContract.h"
#include "pipeline/internal/sima/PluginContractSubsets.h"
#include "pipeline/internal/sima/PreparedRuntimeBuild.h"
#include "pipeline/internal/sima/stagesemantics/ProcessCvuRuntimeConfigAdapterInternal.h"
#include "test_main.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gst/gst.h>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using simaai::neat::pipeline_internal::sima::detess_runtime_frame_shape;
using simaai::neat::pipeline_internal::sima::load_mpk_contract_from_pack_root;
using simaai::neat::pipeline_internal::sima::MpkContract;
using json = nlohmann::json;

struct DetessFixtureHead {
  std::string name;
  std::vector<std::int64_t> frame_shape;
  std::vector<std::int64_t> slice_shape;
  std::size_t transport_bytes = 0U;
  std::size_t output_bytes = 0U;
  bool align_c16 = true;
  bool cblock = true;
  int actual_batch_size = 1;
};

std::filesystem::path write_detess_fixture(const std::string& name,
                                           const std::vector<DetessFixtureHead>& heads,
                                           std::vector<std::size_t> mla_output_order = {}) {
  const auto root = std::filesystem::temp_directory_path() / ("sima_detess_rank2_" + name);
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  require(!ec, "failed to create detess fixture root");

  if (mla_output_order.empty()) {
    mla_output_order.resize(heads.size());
    for (std::size_t i = 0; i < heads.size(); ++i) {
      mla_output_order[i] = i;
    }
  }

  json plugins = json::array();
  json mla_outputs = json::array();
  for (const auto index : mla_output_order) {
    require(index < heads.size(), "invalid MLA output order in detess fixture");
    mla_outputs.push_back(
        {{"name", heads[index].name}, {"type", "buffer"}, {"size", heads[index].transport_bytes}});
  }
  plugins.push_back({{"name", "MLA_0"},
                     {"sequence", 1},
                     {"processor", "MLA"},
                     {"config_params", {{"desired_batch_size", 1}, {"actual_batch_size", 1}}},
                     {"resources", {{"executable", "fixture.elf"}}},
                     {"input_nodes", {{{"name", "model_input"}, {"type", "buffer"}, {"size", 1}}}},
                     {"output_nodes", std::move(mla_outputs)}});

  for (std::size_t i = 0; i < heads.size(); ++i) {
    const auto& head = heads[i];
    plugins.push_back(
        {{"name", "detess_" + head.name},
         {"sequence", static_cast<int>(i + 2U)},
         {"processor", "EV74"},
         {"config_params",
          {{"desired_batch_size", 1},
           {"actual_batch_size", head.actual_batch_size},
           {"kernel", "detessellation_transform"},
           {"params",
            {{"slice_shape", head.slice_shape},
             {"align_c16", head.align_c16},
             {"cblock", head.cblock},
             {"frame_type", "bfloat16"},
             {"frame_shape", head.frame_shape},
             {"input_shapes", json::array({json::array({1, head.transport_bytes})})},
             {"output_shapes", json::array({head.frame_shape})}}}}},
         {"input_nodes",
          {{{"name", head.name}, {"type", "buffer"}, {"size", head.transport_bytes}}}},
         {"output_nodes",
          {{{"name", head.name + "_output"}, {"type", "buffer"}, {"size", head.output_bytes}}}}});
  }

  std::ofstream out(root / "mpk.json");
  require(out.is_open(), "failed to open detess fixture manifest");
  out << json{{"name", "detess_rank2_fixture"},
              {"model_path", "fixture.elf"},
              {"input_nodes", {{{"name", "model_input"}, {"type", "buffer"}, {"size", 1}}}},
              {"plugins", std::move(plugins)}}
             .dump(2);
  out.close();
  require(out.good(), "failed to finalize detess fixture manifest");
  return root;
}

std::filesystem::path write_detess_fixture(const std::string& name,
                                           const std::vector<std::int64_t>& frame_shape,
                                           const std::size_t transport_bytes,
                                           const std::size_t output_bytes,
                                           const bool align_c16 = true, const bool cblock = true,
                                           const std::vector<std::int64_t>& slice_shape = {64}) {
  return write_detess_fixture(name, {{"MLA_0", frame_shape, slice_shape, transport_bytes,
                                      output_bytes, align_c16, cblock}});
}

const MpkContract load_fixture(const std::filesystem::path& root) {
  std::string error;
  const auto contract = load_mpk_contract_from_pack_root(root.string(), &error);
  require(contract.has_value(), "detess fixture should load: " + error);
  return *contract;
}

std::string reject_fixture(const std::filesystem::path& root) {
  std::string error;
  const auto contract = load_mpk_contract_from_pack_root(root.string(), &error);
  require(!contract.has_value(), "invalid detess fixture should be rejected");
  require(!error.empty(), "invalid detess fixture should report an actionable error");
  return error;
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

std::vector<std::int64_t> tensor_desc_shape(const sima_ev_tensor_desc& desc) {
  std::vector<std::int64_t> shape;
  const auto rank = std::min<std::uint32_t>(desc.shape.rank, SIMA_EV_MAX_RANK);
  shape.reserve(rank);
  for (std::uint32_t i = 0; i < rank; ++i) {
    shape.push_back(desc.shape.sizes[i]);
  }
  return shape;
}

simaai::neat::pipeline_internal::sima::SimaPluginStaticManifest
make_detess_manifest(const MpkContract& contract) {
  namespace pcs = simaai::neat::pipeline_internal::sima::plugin_contracts;
  namespace pss = simaai::neat::pipeline_internal::sima::stagesemantics;
  using simaai::neat::pipeline_internal::sima::StagePayloadKind;
  using simaai::neat::pipeline_internal::sima::StageStaticSpec;

  require(contract.plugins.size() == 2U, "prepared-runtime fixture should contain MLA and detess");
  const auto& detess = contract.plugins[1];
  require(detess.output_tensors.size() == 1U,
          "prepared-runtime fixture should contain one detess output");
  const std::string output_name = detess.output_tensors.front().name;
  const auto subsets = pcs::extract_detessellate_contract_subsets_from_mpk(contract);
  require(subsets.size() == 1U, "prepared-runtime fixture should produce one detess subset");
  const auto runtime =
      pcs::build_detessellate_runtime_config_from_subsets(subsets, {output_name}, {output_name});
  const auto compiled = pss::build_processcvu_compiled_contract_from_runtime_config(runtime);

  StageStaticSpec stage;
  stage.element_name = detess.name;
  stage.logical_stage_id = detess.name;
  stage.model_managed_stage = true;
  stage.plugin_kind = compiled.runtime_contract.plugin_kind;
  stage.kernel_kind = detess.kernel;
  stage.payload_kind = StagePayloadKind::ProcessCvu;
  stage.processcvu = compiled.payload;
  stage.processcvu.exact_stage_name_or_id = detess.name;
  stage.logical_inputs = compiled.runtime_contract.logical_inputs;
  stage.input_bindings = compiled.runtime_contract.input_bindings;
  stage.physical_inputs = compiled.runtime_contract.physical_inputs;
  stage.physical_outputs = compiled.runtime_contract.physical_outputs;
  stage.logical_outputs = compiled.runtime_contract.logical_outputs;
  stage.output_order = compiled.runtime_contract.output_order;
  stage.output_quant = compiled.runtime_contract.output_quant;

  simaai::neat::pipeline_internal::sima::SimaPluginStaticManifest manifest;
  manifest.model_id = contract.model_name;
  manifest.stages.push_back(std::move(stage));
  return manifest;
}

void require_prepared_runtime_geometry(const std::filesystem::path& root,
                                       const std::vector<std::int64_t>& runtime_shape,
                                       const std::vector<std::int64_t>& logical_shape,
                                       const std::uint64_t transport_bytes) {
  using simaai::neat::pipeline_internal::sima::build_prepared_runtime_context;
  using simaai::neat::pipeline_internal::sima::PipelineElementSpec;

  ensure_gst_ready();
  const auto contract = load_fixture(root);
  const auto manifest = make_detess_manifest(contract);
  PipelineElementSpec mla_element;
  mla_element.plugin = "neatprocessmla";
  mla_element.model_path_property = root.string();

  std::string error;
  const auto prepared = build_prepared_runtime_context(
      nullptr, manifest, std::nullopt, {mla_element}, {}, simaai::neat::NameTransform{}, &error);
  require(prepared.has_value(), "detess prepared runtime should build: " + error);
  require(prepared->stages.size() == 1U && prepared->stages.front().processcvu.has_value(),
          "detess prepared runtime should contain one processcvu stage");
  const auto& detess = *prepared->stages.front().processcvu;
  require(detess.typed_config.input_tensors.size() == 1U &&
              tensor_desc_shape(detess.typed_config.input_tensors.front()) == runtime_shape,
          "prepared detess input descriptor should use resolved runtime geometry");
  require(detess.typed_config.output_tensors.size() == 1U &&
              tensor_desc_shape(detess.typed_config.output_tensors.front()) == runtime_shape,
          "prepared detess output descriptor should use resolved runtime geometry");
  require(detess.typed_config.input_tensors.front().storage.nbytes == transport_bytes,
          "prepared detess input descriptor should preserve the packed MLA byte span");
  require(detess.output_publish_contract.logical_outputs.size() == 1U &&
              detess.output_publish_contract.logical_outputs.front().shape == logical_shape,
          "prepared detess output should preserve the authored logical rank");
}

} // namespace

RUN_TEST(
    "unit_detess_rank2_contract_test", ([] {
      const auto contract = load_fixture(write_detess_fixture("customer_nc", {1, 213}, 448U, 426U));
      require(contract.plugins.size() == 2U, "expected MLA and detess stages");
      const auto& detess = contract.plugins[1];
      require(detess.frame_shape == std::vector<std::int64_t>({1, 213}),
              "loader must preserve the authored logical frame_shape");
      require(detess.runtime_frame_shape == std::vector<std::int64_t>({1, 1, 1, 213}),
              "customer rank-2 shape should resolve uniquely as NC geometry");
      const auto physical_outputs =
          simaai::neat::pipeline_internal::sima::get_mla_published_outputs_contract(contract);
      require(physical_outputs.size() == 1U, "expected one MLA boundary output");
      require(physical_outputs.front().size_bytes == 448U,
              "rank-2 NC output should preserve the packed MLA transport span");
      require(physical_outputs.front().mpk_shape == std::vector<std::int64_t>({1, 448}) &&
                  physical_outputs.front().shape_semantics ==
                      simaai::neat::pipeline_internal::sima::MpkShapeSemantics::PackedExtent,
              "published MLA boundary should remain the authored packed byte-carrier view");

      const auto logical_outputs =
          simaai::neat::pipeline_internal::sima::get_mla_logical_outputs_contract(contract);
      require(logical_outputs.size() == 1U, "expected one logical output");
      require(logical_outputs.front().mpk_shape == std::vector<std::int64_t>({1, 213}),
              "logical output must retain the authored rank-2 shape");
      require(logical_outputs.front().size_bytes == 426U,
              "logical BF16 output should contain 213 two-byte scalars");

      const auto hw_contract = load_fixture(write_detess_fixture("hw", {2, 3}, 192U, 12U));
      require(hw_contract.plugins[1].runtime_frame_shape == std::vector<std::int64_t>({1, 2, 3, 1}),
              "rank-2 HW shape should resolve from its C16 transport span");

      const auto batched_error = reject_fixture(write_detess_fixture(
          "batched",
          std::vector<DetessFixtureHead>{{"MLA_0", {2, 3}, {1, 1, 1}, 12U, 12U, false, false, 2}}));
      require(batched_error.find("requires batch=1") != std::string::npos,
              "batched rank-2 contracts must fail before NC/HW inference");

      const auto rank3_contract =
          load_fixture(write_detess_fixture("rank3", {1, 10, 7}, 320U, 140U));
      require(detess_runtime_frame_shape(rank3_contract.plugins[1]) ==
                  std::vector<std::int64_t>({1, 10, 7}),
              "canonical rank-3 detess geometry must remain unchanged");

      const auto rank4_contract =
          load_fixture(write_detess_fixture("rank4", {1, 2, 3, 7}, 192U, 84U));
      require(detess_runtime_frame_shape(rank4_contract.plugins[1]) ==
                  std::vector<std::int64_t>({1, 2, 3, 7}),
              "canonical rank-4 detess geometry must remain unchanged");

      const auto two_head_contract =
          load_fixture(write_detess_fixture("two_head_by_name",
                                            {{"carrier_hw", {2, 3}, {1, 1, 1}, 192U, 12U},
                                             {"carrier_nc", {1, 213}, {64}, 448U, 426U}},
                                            {1U, 0U}));
      require(two_head_contract.plugins[1].runtime_frame_shape ==
                      std::vector<std::int64_t>({1, 2, 3, 1}) &&
                  two_head_contract.plugins[2].runtime_frame_shape ==
                      std::vector<std::int64_t>({1, 1, 1, 213}),
              "each rank-2 detess head should resolve from its own named transport tensor");
      const auto two_head_outputs =
          simaai::neat::pipeline_internal::sima::get_mla_published_outputs_contract(
              two_head_contract);
      require(two_head_outputs.size() == 2U && two_head_outputs[0].size_bytes == 448U &&
                  two_head_outputs[1].size_bytes == 192U,
              "multi-output MLA boundaries should retain physical output order and byte spans");

      require_prepared_runtime_geometry(write_detess_fixture("prepared_nc", {1, 213}, 448U, 426U),
                                        {1, 1, 1, 213}, {1, 213}, 448U);
      require_prepared_runtime_geometry(
          write_detess_fixture("prepared_hw", {2, 3}, 192U, 12U, true, true, {1, 1, 1}),
          {1, 2, 3, 1}, {2, 3}, 192U);
      require_prepared_runtime_geometry(
          write_detess_fixture("prepared_rank4", {1, 2, 3, 7}, 192U, 84U, true, true, {1, 1, 1}),
          {1, 2, 3, 7}, {1, 2, 3, 7}, 192U);

      const auto ambiguous_error =
          reject_fixture(write_detess_fixture("ambiguous", {1, 16}, 32U, 32U, false, false));
      require(ambiguous_error.find("ambiguous") != std::string::npos &&
                  ambiguous_error.find("NC") != std::string::npos &&
                  ambiguous_error.find("HW") != std::string::npos,
              "equally valid NC/HW interpretations must fail without guessing");

      const auto transport_error =
          reject_fixture(write_detess_fixture("transport_mismatch", {1, 213}, 450U, 426U));
      require(transport_error.find("no matching interpretation") != std::string::npos &&
                  transport_error.find("transport_bytes=450") != std::string::npos,
              "transport mismatch should identify the observed byte span");

      const auto output_error =
          reject_fixture(write_detess_fixture("output_mismatch", {1, 213}, 448U, 424U));
      require(output_error.find("no matching interpretation") != std::string::npos &&
                  output_error.find("output_bytes=424") != std::string::npos,
              "logical output mismatch should identify the observed byte span");

      const auto invalid_dim_error =
          reject_fixture(write_detess_fixture("invalid_dim", {1, 0}, 32U, 2U));
      require(invalid_dim_error.find("non-positive dimensions") != std::string::npos &&
                  invalid_dim_error.find("frame_shape=[1,0]") != std::string::npos,
              "non-positive rank-2 dimensions should fail before runtime configuration");
    }));
