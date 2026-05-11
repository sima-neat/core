#include "model/Model.h"
#include "mpk_fixture_utils.h"
#include "test_main.h"
#include "test_utils.h"

namespace {

sima_test::MpkFixture make_stage_fixture(const std::string& tag) {
  return sima_test::make_strict_mpk_tar_fixture(tag,
                                                {
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
      },
      {
        "sequence_id": 3,
        "name": "detessdequant_0",
        "pluginId": "processcvu",
        "configPath": "0_postproc.json",
        "processor": "CVU",
        "kernel": "detessdequant",
        "input": "mla_0"
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
                                             {"etc/0_postproc.json",
                                              R"json({
  "node_name": "detessdequant_0",
  "num_in_tensor": 1,
  "out_data_type": "FP32",
  "input_width": [80],
  "input_height": [80],
  "input_depth": [6]
})json"},
                                                },
                                                true);
}

} // namespace

RUN_TEST(
    "unit_model_stage_fragments_test", ([] {
      using namespace simaai::neat;

      const auto fixture = make_stage_fixture("model_stage_fragments");
      Model model(fixture.tar_path);

      const NodeGroup pre = model.preprocess();
      const NodeGroup infer = model.inference();
      const NodeGroup post = model.postprocess();
      const NodeGroup sess = model.session();

      require(!pre.nodes().empty(), "Model::preprocess should produce a non-empty stage group");
      require(!infer.nodes().empty(), "Model::inference should produce a non-empty stage group");
      require(!post.nodes().empty(),
              "Model::postprocess should expose Model boundary post stage");
      require(sess.nodes().size() >= 3,
              "Model::session should include appsrc + model stages + appsink by default");

      const std::string infer_fragment = model.backend_fragment(Model::Stage::Inference);
      require_contains(infer_fragment, "neatprocessmla",
                       "Model::backend_fragment(inference) should include MLA plugin");
      require_contains(infer_fragment, "stage-id=",
                       "Model::backend_fragment(inference) should include stage metadata");

      const std::string full_fragment = model.backend_fragment(Model::Stage::Full);
      require_contains(full_fragment, "neatprocesscvu",
                       "Model::backend_fragment(full) should include CVU preproc plugin");
      require_contains(full_fragment, "neatprocessmla",
                       "Model::backend_fragment(full) should include MLA plugin");
      require_contains(full_fragment, "stage-id=",
                       "Model::backend_fragment(full) should include stage metadata");

      const NodeGroup infer_only = model.fragment(Model::Stage::Inference);
      require(!infer_only.nodes().empty(),
              "Model::fragment(inference) should produce non-empty NodeGroup");

      const auto legacy = sima_test::make_mpk_tar_fixture(
          "model_stage_fragments_legacy_missing_mpk",
          {
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
        (void)legacy_model.session();
      } catch (const std::exception& e) {
        threw = true;
        require_contains(std::string(e.what()), "strict MPK contract required",
                         "legacy missing-mpk fixture should fail with strict contract error");
      }
      require(threw, "legacy missing-mpk fixture must fail under strict contract");
    }));
