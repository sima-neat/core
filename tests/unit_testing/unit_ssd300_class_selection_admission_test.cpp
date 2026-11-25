#include "model/Model.h"
#include "model/internal/ModelInternal.h"
#include "model_archive_fixture_utils.h"
#include "pipeline/BoxDecodeType.h"
#include "pipeline/internal/sima/stagesemantics/SsdRecipeId.h"
#include "support/cold_contract_gst_init.h"
#include "test_main.h"

#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

// SSD300 encodes 81 classes; requesting 8 must select background plus the first seven tool classes.

namespace {

constexpr int kEncodedClasses = 81;
constexpr int kSelectedClasses = 8;

// SSD300 heads, grouped by role: six localization heads, then six confidence heads.
sima_test::ModelArchiveFixture make_ssd300_heads_fixture() {
  using nlohmann::json;
  const std::vector<int> feats = {38, 19, 10, 5, 3, 1};
  const std::vector<int> priors = {4, 6, 6, 6, 4, 4};
  json shapes = json::array(), bf16_heads = json::array(), fp32_heads = json::array();
  std::size_t total_bytes = 0;
  for (const bool confidence : {false, true}) {
    for (std::size_t level = 0; level < feats.size(); ++level) {
      const int side = feats[level];
      const int channels = (confidence ? kEncodedClasses : 4) * priors[level];
      const auto bytes = static_cast<std::size_t>(side) * side * channels * 2;
      const auto name = std::string(confidence ? "conf_" : "loc_") + std::to_string(level);
      shapes.push_back({1, side, side, channels});
      bf16_heads.push_back({{"name", name + "_bf16"}, {"type", "buffer"}, {"size", bytes}});
      fp32_heads.push_back({{"name", name}, {"type", "buffer"}, {"size", bytes * 2}});
      total_bytes += bytes;
    }
  }
  auto mpk = json::parse(R"json({
    "name": "ssd300_heads", "model_sdk_version": "3.0.0",
    "input_nodes": [{"name": "input", "type": "buffer", "size": 1080000}],
    "plugins": [
      {"name": "cast_input", "processor": "EV74", "sequence": 1, "type": "sgpProcess",
       "config_params": {"kernel": "cast_transform", "params": {
         "in_dtype": "float32", "out_dtype": "bfloat16",
         "input_shapes": [[1,300,300,3]], "output_shapes": [[1,300,300,3]]}},
       "input_nodes": [{"name": "input", "size": 1080000}],
       "output_nodes": [{"name": "cast_input", "type": "buffer", "size": 540000}]},
      {"name": "MLA_0", "processor": "MLA", "sequence": 2, "type": "sgpProcess",
       "config_params": {"number_of_quads_to_user": 4},
       "input_nodes": [{"name": "cast_input", "size": 540000}],
       "output_nodes": [{"name": "MLA_0", "type": "buffer", "size": 0}],
       "resources": {"executable": "model.elf"}},
      {"name": "unpack", "processor": "EV74", "sequence": 3, "type": "sgpProcess",
       "config_params": {"kernel": "unpack_transform", "params": {}},
       "input_nodes": [{"name": "MLA_0", "size": 0}]}
    ]
  })json");
  auto& plugins = mpk["plugins"];
  plugins[1]["output_nodes"][0]["size"] = total_bytes;
  plugins[2]["input_nodes"][0]["size"] = total_bytes;
  plugins[2]["output_nodes"] = bf16_heads;
  plugins[2]["config_params"]["params"] = {
      {"tensor_types", std::vector<std::string>(shapes.size(), "bfloat16")},
      {"tensor_shapes", shapes},
      {"input_shapes", {{1, total_bytes}}},
      {"output_shapes", shapes}};
  for (std::size_t i = 0; i < shapes.size(); ++i) {
    auto cast = plugins.front();
    cast["name"] = "cast_" + std::to_string(i);
    cast["sequence"] = 4 + i;
    cast["input_nodes"] = {bf16_heads[i]};
    cast["output_nodes"] = {fp32_heads[i]};
    cast["config_params"]["params"] = {{"in_dtype", "bfloat16"},
                                       {"out_dtype", "float32"},
                                       {"input_shapes", {shapes[i]}},
                                       {"output_shapes", {shapes[i]}}};
    plugins.push_back(std::move(cast));
  }
  auto published = fp32_heads;
  for (std::size_t i = 0; i < published.size(); ++i)
    published[i]["name"] = "pass_through_out_" + std::to_string(i);
  plugins.push_back({{"name", "PassThrough"},
                     {"processor", "EV74"},
                     {"sequence", 4 + shapes.size()},
                     {"type", "sgpProcess"},
                     {"config_params", {{"kernel", "pass_through"}, {"params", json::object()}}},
                     {"input_nodes", fp32_heads},
                     {"output_nodes", published}});
  for (auto& plugin : plugins) {
    plugin["config_params"]["desired_batch_size"] = 1;
    plugin["config_params"]["actual_batch_size"] = 1;
  }
  const auto fixture = sima_test::make_model_archive_fixture(
      "ssd300_class_selection", {{"etc/ssd300_heads_mpk.json", mpk.dump()}}, false);
  sima_test::write_mpk_topology_elfs(fixture.root_dir, mpk.dump());
  const auto archive = "tar -czf " + sima_test::model_archive_shell_quote(fixture.tar_path) +
                       " -C " + sima_test::model_archive_shell_quote(fixture.root_dir) + " .";
  require(std::system(archive.c_str()) == 0, "failed to archive the SSD300 heads fixture");
  return fixture;
}

simaai::neat::Model::Options ssd300_options(int num_classes) {
  using namespace simaai::neat;
  Model::Options options;
  options.preprocess.kind = InputKind::Tensor;
  options.preprocess.enable = AutoFlag::Off;
  options.decode_type = BoxDecodeType::Ssd;
  options.num_classes = num_classes;
  options.score_threshold = 0.001f;
  options.nms_iou_threshold = 0.45f;
  options.top_k = 24;
  return options;
}

} // namespace

RUN_TEST("unit_ssd300_class_selection_admission_test", ([] {
           using namespace simaai::neat;
           using pipeline_internal::sima::stagesemantics::SsdClassSelectionKind;
           sima_test::initialize_cold_contract_gstreamer();
           const auto fixture = make_ssd300_heads_fixture();

           Model narrowed(fixture.tar_path, ssd300_options(kSelectedClasses));
           for (const bool sync : {false, true}) {
             const auto contract =
                 internal::ModelAccess::build_boxdecode_stage_contract(narrowed, sync);
             const auto& selection = contract.payload.ssd_class_selection;
             require(contract.payload.decode_type == BoxDecodeType::Ssd,
                     "the SSD300 heads must resolve to an SSD BoxDecode contract");
             require(selection.encoded_count == kEncodedClasses,
                     "num_classes = 8 must keep the 81 classes the SSD300 heads encode");
             require(selection.selected_count == kSelectedClasses,
                     "num_classes = 8 must select 8 classes; 81 means the option was ignored");
             require(selection.kind == SsdClassSelectionKind::PrefixFromZero,
                     "SSD300 must narrow to a prefix of the encoded classes");
           }

           // Without the option the contract keeps every encoded class. This is the value the
           // narrowed case must not report.
           Model full(fixture.tar_path, ssd300_options(0));
           const auto full_contract =
               internal::ModelAccess::build_boxdecode_stage_contract(full, false);
           require(full_contract.payload.ssd_class_selection.encoded_count == kEncodedClasses &&
                       full_contract.payload.ssd_class_selection.selected_count == kEncodedClasses,
                   "without num_classes the SSD300 contract must select all 81 encoded classes");

           std::string over_error;
           try {
             Model over(fixture.tar_path, ssd300_options(kEncodedClasses + 10));
             (void)internal::ModelAccess::build_boxdecode_stage_contract(over, false);
           } catch (const std::exception& error) {
             over_error = error.what();
           }
           require(over_error.find("num_classes=91 exceeds the 81 classes encoded") !=
                       std::string::npos,
                   "num_classes = 91 must be rejected because the heads encode 81 classes, got: " +
                       (over_error.empty() ? std::string("no error") : over_error));
         }));
