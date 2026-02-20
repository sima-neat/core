#include "model/Model.h"
#include "mpk_fixture_utils.h"
#include "test_main.h"
#include "test_utils.h"

namespace {

sima_test::MpkFixture make_stage_fixture(const std::string& tag) {
  return sima_test::make_mpk_tar_fixture(tag,
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
      require(post.nodes().empty(),
              "Model::postprocess should be empty when no post kernel exists in pipeline_sequence");
      require(sess.nodes().size() >= 3,
              "Model::session should include appsrc + model stages + appsink by default");

      const std::string infer_fragment = model.backend_fragment(Model::Stage::Inference);
      require_contains(infer_fragment, "neatprocessmla",
                       "Model::backend_fragment(inference) should include MLA plugin");
      require_contains(infer_fragment, "stage-id=mla_0",
                       "Model::backend_fragment(inference) should include MLA stage-id");

      const std::string full_fragment = model.backend_fragment(Model::Stage::Full);
      require_contains(full_fragment, "neatprocesscvu",
                       "Model::backend_fragment(full) should include CVU preproc plugin");
      require_contains(full_fragment, "neatprocessmla",
                       "Model::backend_fragment(full) should include MLA plugin");
      require_contains(full_fragment, "stage-id=preproc_0",
                       "Model::backend_fragment(full) should include preproc stage-id");

      const NodeGroup infer_only = model.fragment(Model::Stage::Inference);
      require(!infer_only.nodes().empty(),
              "Model::fragment(inference) should produce non-empty NodeGroup");
    }));
