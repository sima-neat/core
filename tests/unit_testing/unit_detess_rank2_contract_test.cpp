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
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using simaai::neat::pipeline_internal::sima::load_mpk_contract_from_pack_root;
using simaai::neat::pipeline_internal::sima::MpkContract;

std::filesystem::path
write_detess_fixture(const std::string& name, const std::vector<std::int64_t>& frame_shape,
                     const std::size_t transport_bytes, const std::size_t output_bytes,
                     const int actual_batch = 1, const bool align_c16 = true,
                     const bool cblock = true, const bool include_detess_batch_metadata = true,
                     const std::vector<std::int64_t>& slice_shape = {64}) {
  const auto root = std::filesystem::temp_directory_path() / ("sima_detess_rank2_" + name);
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  require(!ec, "failed to create detess fixture root");

  auto shape_json = [](const std::vector<std::int64_t>& shape) {
    std::ostringstream out;
    out << '[';
    for (std::size_t i = 0; i < shape.size(); ++i) {
      if (i != 0U) {
        out << ',';
      }
      out << shape[i];
    }
    out << ']';
    return out.str();
  };

  std::ostringstream detess_batch_metadata;
  if (include_detess_batch_metadata) {
    detess_batch_metadata << "        \"desired_batch_size\": " << actual_batch << ",\n"
                          << "        \"actual_batch_size\": " << actual_batch << ",\n";
  }

  std::ofstream out(root / "mpk.json");
  require(out.is_open(), "failed to open detess fixture manifest");
  out << R"JSON({
  "name": "detess_rank2_fixture",
  "model_path": "fixture.elf",
  "input_nodes": [{ "name": "model_input", "type": "buffer", "size": 1 }],
  "plugins": [
    {
      "name": "MLA_0",
      "sequence": 1,
      "processor": "MLA",
      "config_params": {
        "desired_batch_size": )JSON"
      << actual_batch << R"JSON(,
        "actual_batch_size": )JSON"
      << actual_batch << R"JSON(
      },
      "resources": { "executable": "fixture.elf" },
      "input_nodes": [{ "name": "model_input", "type": "buffer", "size": 1 }],
      "output_nodes": [
        { "name": "MLA_0", "type": "buffer", "size": )JSON"
      << transport_bytes << R"JSON( }
      ]
    },
    {
      "name": "detessellate_MLA_0_detessellation_transform",
      "sequence": 2,
      "processor": "EV74",
      "config_params": {
 )JSON"
      << detess_batch_metadata.str() << R"JSON(        "kernel": "detessellation_transform",
        "params": {
          "slice_shape": )JSON"
      << shape_json(slice_shape) << R"JSON(,
          "align_c16": )JSON"
      << (align_c16 ? "true" : "false") << R"JSON(,
          "cblock": )JSON"
      << (cblock ? "true" : "false") << R"JSON(,
          "frame_type": "bfloat16",
          "frame_shape": )JSON"
      << shape_json(frame_shape) << R"JSON(,
          "input_shapes": [[1,)JSON"
      << transport_bytes << R"JSON(]],
          "output_shapes": [)JSON"
      << shape_json(frame_shape) << R"JSON(]
        }
      },
      "input_nodes": [
        { "name": "MLA_0", "type": "buffer", "size": )JSON"
      << transport_bytes << R"JSON( }
      ],
      "output_nodes": [
        { "name": "detess_output", "type": "buffer", "size": )JSON"
      << output_bytes << R"JSON( }
      ]
    }
  ]
})JSON";
  out.close();
  require(out.good(), "failed to finalize detess fixture manifest");
  return root;
}

std::filesystem::path write_two_head_fixture() {
  const auto root = std::filesystem::temp_directory_path() / "sima_detess_rank2_two_head_by_name";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root, ec);
  require(!ec, "failed to create two-head detess fixture root");

  std::ofstream out(root / "two_head_mpk.json");
  require(out.is_open(), "failed to open two-head detess fixture manifest");
  out << R"JSON({
  "name": "two_head_detess_fixture",
  "model_path": "fixture.elf",
  "input_nodes": [{ "name": "model_input", "type": "buffer", "size": 1 }],
  "plugins": [
    {
      "name": "MLA_0",
      "sequence": 1,
      "processor": "MLA",
      "resources": { "executable": "fixture.elf" },
      "input_nodes": [{ "name": "model_input", "type": "buffer", "size": 1 }],
      "output_nodes": [
        { "name": "carrier_nc", "type": "buffer", "size": 448 },
        { "name": "carrier_hw", "type": "buffer", "size": 192 }
      ]
    },
    {
      "name": "detess_hw",
      "sequence": 2,
      "processor": "EV74",
      "config_params": {
        "desired_batch_size": 1,
        "actual_batch_size": 1,
        "kernel": "detessellation_transform",
        "params": {
          "slice_shape": [1, 1, 1],
          "align_c16": true,
          "cblock": true,
          "frame_type": "bfloat16",
          "frame_shape": [2, 3],
          "input_shapes": [[1, 192]],
          "output_shapes": [[2, 3]]
        }
      },
      "input_nodes": [{ "name": "carrier_hw", "type": "buffer", "size": 192 }],
      "output_nodes": [{ "name": "hw_output", "type": "buffer", "size": 12 }]
    },
    {
      "name": "detess_nc",
      "sequence": 3,
      "processor": "EV74",
      "config_params": {
        "desired_batch_size": 1,
        "actual_batch_size": 1,
        "kernel": "detessellation_transform",
        "params": {
          "slice_shape": [64],
          "align_c16": true,
          "cblock": true,
          "frame_type": "bfloat16",
          "frame_shape": [1, 213],
          "input_shapes": [[1, 448]],
          "output_shapes": [[1, 213]]
        }
      },
      "input_nodes": [{ "name": "carrier_nc", "type": "buffer", "size": 448 }],
      "output_nodes": [{ "name": "nc_output", "type": "buffer", "size": 426 }]
    }
  ]
})JSON";
  out.close();
  require(out.good(), "failed to finalize two-head detess fixture manifest");
  return root;
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
      require(detess.batch_size == 1 && detess.batch_sz_model == 1,
              "detess resolver should retain explicit top-level batch metadata");

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

      const auto batched_nc_error =
          reject_fixture(write_detess_fixture("batched_nc", {4, 17}, 256U, 136U, 4));
      require(batched_nc_error.find("requires batch=1") != std::string::npos,
              "batched rank-2 NC contracts must fail at model loading");

      const auto batched_hw_error =
          reject_fixture(write_detess_fixture("batched_hw", {2, 3}, 768U, 48U, 4));
      require(batched_hw_error.find("requires batch=1") != std::string::npos,
              "batched rank-2 HW contracts must fail at model loading");

      const auto missing_batch_error = reject_fixture(
          write_detess_fixture("missing_batch", {1, 213}, 448U, 426U, 1, true, true, false));
      require(missing_batch_error.find("explicit batch metadata") != std::string::npos,
              "rank-2 contracts must not silently assume batch one");

      const auto rank3_contract =
          load_fixture(write_detess_fixture("rank3", {1, 10, 7}, 320U, 140U));
      require(rank3_contract.plugins[1].runtime_frame_shape ==
                  std::vector<std::int64_t>({1, 10, 7}),
              "canonical rank-3 detess geometry must remain unchanged");

      const auto rank4_contract =
          load_fixture(write_detess_fixture("rank4", {1, 2, 3, 7}, 192U, 84U));
      require(rank4_contract.plugins[1].runtime_frame_shape ==
                  std::vector<std::int64_t>({1, 2, 3, 7}),
              "canonical rank-4 detess geometry must remain unchanged");

      const auto batched_rank4_contract =
          load_fixture(write_detess_fixture("batched_rank4", {4, 2, 3, 7}, 768U, 336U, 4));
      require(batched_rank4_contract.plugins[1].runtime_frame_shape ==
                  std::vector<std::int64_t>({4, 2, 3, 7}),
              "canonical batched rank-4 detess geometry must remain unchanged");
      require(batched_rank4_contract.plugins[1].batch_size == 0 &&
                  batched_rank4_contract.plugins[1].batch_sz_model == 0,
              "rank-2 batch parsing must not alter canonical detess contracts");

      const auto two_head_contract = load_fixture(write_two_head_fixture());
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
          write_detess_fixture("prepared_hw", {2, 3}, 192U, 12U, 1, true, true, true, {1, 1, 1}),
          {1, 2, 3, 1}, {2, 3}, 192U);
      require_prepared_runtime_geometry(write_detess_fixture("prepared_rank4", {1, 2, 3, 7}, 192U,
                                                             84U, 1, true, true, true, {1, 1, 1}),
                                        {1, 2, 3, 7}, {1, 2, 3, 7}, 192U);

      const auto ambiguous_error =
          reject_fixture(write_detess_fixture("ambiguous", {1, 16}, 32U, 32U, 1, false, false));
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
