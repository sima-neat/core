#include "model/Model.h"
#include "mpk_fixture_utils.h"
#include "test_main.h"
#include "test_utils.h"

#include <unordered_map>

namespace {

sima_test::MpkFixture make_metadata_fixture(const std::string& tag, bool include_metadata) {
  std::vector<std::pair<std::string, std::string>> files = {
      {"etc/pipeline_sequence.json",
       R"json({
  "pipelines": [{
    "sequence": [
      {
        "sequence_id": 1,
        "name": "preproc_0",
        "pluginId": "processcvu",
        "configPath": "0_preproc.json",
        "processor": "CVU",
        "kernel": "preproc",
        "input": "decoder"
      },
      {
        "sequence_id": 2,
        "name": "mla_0",
        "pluginId": "processmla",
        "configPath": "0_process_mla.json",
        "processor": "MLA",
        "kernel": "infer",
        "input": "preproc_0"
      }
    ]
  }]
})json"},
      {"etc/0_preproc.json",
       R"json({
  "node_name": "preproc_0",
  "input_width": 1280,
  "input_height": 720,
  "input_img_type": "RGB",
  "output_width": 640,
  "output_height": 640,
  "output_img_type": "RGB"
})json"},
      {"etc/0_process_mla.json",
       R"json({
  "node_name": "mla_0",
  "input_buffers": [{"name": "preproc_0"}],
  "data_type": ["INT8"],
  "output_width": [80],
  "output_height": [80],
  "output_depth": [6]
})json"},
  };

  if (include_metadata) {
    files.push_back({"etc/metadata.json",
                     R"json({
  "model_name": "unit-meta",
  "version": 3,
  "calibration": true,
  "labels": ["cat", "dog"],
  "nested": {"a": 1}
})json"});
  }

  return sima_test::make_strict_mpk_tar_fixture(tag, files, true);
}

} // namespace

RUN_TEST("unit_model_metadata_test", ([] {
           using namespace simaai::neat;

           {
             const auto fixture = make_metadata_fixture("model_metadata_present", true);
             Model model(fixture.tar_path);
             const std::unordered_map<std::string, std::string> meta = model.metadata();

             require(!meta.empty(),
                     "Model::metadata should return entries when metadata.json exists");
             require(meta.at("model_name") == "unit-meta",
                     "Model::metadata should preserve string values verbatim");
             require(meta.at("version") == "3", "Model::metadata should stringify integer values");
             require(meta.at("calibration") == "true",
                     "Model::metadata should stringify boolean values");
             require(meta.at("labels") == "[\"cat\",\"dog\"]",
                     "Model::metadata should dump array values");
             require(meta.at("nested") == "{\"a\":1}", "Model::metadata should dump object values");
           }

           {
             const auto fixture = make_metadata_fixture("model_metadata_missing", false);
             Model model(fixture.tar_path);
             const auto meta = model.metadata();
             require(meta.empty(),
                     "Model::metadata should return empty map when metadata.json is missing");
           }

           {
             const auto legacy = sima_test::make_mpk_tar_fixture(
                 "model_metadata_legacy_missing_mpk", {
                                                          {"etc/pipeline_sequence.json",
                                                           R"json({
  "pipelines": [{
    "sequence": [
      {
        "sequence_id": 1,
        "name": "mla_0",
        "pluginId": "processmla",
        "configPath": "0_process_mla.json",
        "processor": "MLA",
        "kernel": "infer",
        "input": "decoder"
      }
    ]
  }]
})json"},
                                                          {"etc/0_process_mla.json",
                                                           R"json({
  "node_name": "mla_0",
  "input_buffers": [{"name": "decoder"}]
})json"},
                                                      });
             bool threw = false;
             try {
               Model legacy_model(legacy.tar_path);
               (void)legacy_model.metadata();
             } catch (const std::exception& e) {
               threw = true;
               require_contains(
                   std::string(e.what()), "strict MPK contract required",
                   "legacy missing-mpk fixture should fail with strict contract error");
             }
             require(threw, "legacy missing-mpk fixture must fail under strict contract");
           }
         }));
