#include "asset_utils.h"
#include "mpk_fixture_utils.h"
#include "model/Model.h"
#include "test_main.h"
#include "test_utils.h"

#include <filesystem>
#include <functional>
#include <string>

namespace {

bool runtime_unavailable_for_model(const std::string& msg) {
  if (is_dispatcher_unavailable(msg))
    return true;
  return msg.find("No such element") != std::string::npos ||
         msg.find("could not load") != std::string::npos ||
         msg.find("dispatcher unavailable") != std::string::npos;
}

bool throws_with(const std::function<void()>& fn, const std::string& needle) {
  try {
    fn();
  } catch (const std::exception& e) {
    if (needle.empty())
      return true;
    return std::string(e.what()).find(needle) != std::string::npos;
  }
  return false;
}

} // namespace

RUN_TEST(
    "unit_model_infer_output_name_test", ([] {
      namespace fs = std::filesystem;

      const std::string tar = sima_test::resolve_resnet50_tar_local_only(fs::current_path());
      if (!tar.empty()) {
        try {
          simaai::neat::Model model(tar);
          const std::string output_name = model.infer_output_name();

          require(!output_name.empty(), "Model::infer_output_name returned empty output name");
          require(output_name == model.infer_output_name(),
                  "Model::infer_output_name should be deterministic across calls");

          const std::string infer_fragment =
              model.backend_fragment(simaai::neat::Model::Stage::Inference);
          require_contains(infer_fragment, output_name,
                           "inference backend fragment does not include inferred output name");
        } catch (const std::exception& e) {
          if (!runtime_unavailable_for_model(e.what())) {
            throw;
          }
        }
      }

      require(throws_with(
                  []() {
                    simaai::neat::Model missing("/tmp/this_model_pack_does_not_exist.tar.gz");
                    (void)missing.infer_output_name();
                  },
                  "ModelPack"),
              "constructing Model from missing path should fail");

      // Malformed model metadata should fail deterministically.
      {
        const auto fixture =
            sima_test::make_malformed_mpk_tar_fixture("infer_output_name_malformed");
        bool threw = false;
        std::string msg;
        try {
          simaai::neat::Model bad(fixture.tar_path);
          (void)bad.infer_output_name();
        } catch (const std::exception& e) {
          threw = true;
          msg = e.what();
        }
        require(threw, "malformed MPK fixture should throw");
        require(!msg.empty(), "malformed MPK fixture should report non-empty error text");
        require(msg.find("parse error") != std::string::npos ||
                    msg.find("ModelPack") != std::string::npos,
                "malformed MPK fixture error should be actionable");
      }

      // Multi-stage inference sequence should pick the final MLA stage name deterministically.
      {
        const auto fixture =
            sima_test::make_strict_mpk_tar_fixture("infer_output_name_multi_stage",
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
        "name": "mla_stage_a",
        "pluginId": "processmla",
        "configPath": "0_process_mla_a.json",
        "processor": "MLA",
        "kernel": "infer",
        "input": "preproc_0"
      },
      {
        "sequence_id": 3,
        "name": "mla_stage_b",
        "pluginId": "processmla",
        "configPath": "0_process_mla_b.json",
        "processor": "MLA",
        "kernel": "infer",
        "input": "mla_stage_a"
      }
    ]
  }]
})json"},
                                                                 {"etc/0_preproc.json",
                                                                  R"json({
  "node_name": "preproc_0",
  "input_width": 64,
  "input_height": 48,
  "input_img_type": "RGB",
  "output_width": 64,
  "output_height": 48,
  "output_img_type": "RGB"
})json"},
                                                                 {"etc/0_process_mla_a.json",
                                                                  R"json({
  "node_name": "mla_stage_a",
  "input_buffers": [{"name": "preproc_0"}]
})json"},
                                                                 {"etc/0_process_mla_b.json",
                                                                  R"json({
  "node_name": "mla_stage_b",
  "input_buffers": [{"name": "mla_stage_a"}]
})json"},
                                                   });

        simaai::neat::Model::Options opt;
        opt.preprocess.kind = simaai::neat::InputKind::Image;
        opt.preprocess.enable = simaai::neat::AutoFlag::On;
        opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::RGB;

        simaai::neat::Model model(fixture.tar_path, opt);
        const std::string inferred = model.infer_output_name();
        require(!inferred.empty(),
                "Model::infer_output_name should remain non-empty for strict multi-stage fixtures");
        require_contains(model.backend_fragment(simaai::neat::Model::Stage::Inference), inferred,
                         "multi-stage inference backend fragment should include inferred output "
                         "name");
        require(inferred == model.infer_output_name(),
                "Model::infer_output_name should remain deterministic on repeated calls");
      }

      {
        const auto legacy = sima_test::make_mpk_tar_fixture("infer_output_name_legacy_missing_mpk",
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
        require(throws_with(
                    [&]() {
                      simaai::neat::Model legacy_model(legacy.tar_path);
                      (void)legacy_model.infer_output_name();
                    },
                    "strict MPK contract required"),
                "legacy fixture without *_mpk.json should fail with strict contract error");
      }
    }));
