#include "model/Model.h"
#include "mpk_fixture_utils.h"
#include "test_main.h"
#include "test_utils.h"

#include <string>
#include <vector>

namespace {

sima_test::MpkFixture make_preproc_mla_fixture(const std::string& tag,
                                               const std::string& input_format = "RGB") {
  return sima_test::make_mpk_tar_fixture(
      tag,
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
          {"etc/0_preproc.json", std::string("{\n") +
                                     "  \"node_name\": \"preproc_0\",\n"
                                     "  \"input_width\": 1280,\n"
                                     "  \"input_height\": 720,\n"
                                     "  \"input_img_type\": \"" +
                                     input_format +
                                     "\",\n"
                                     "  \"output_width\": 640,\n"
                                     "  \"output_height\": 640,\n"
                                     "  \"output_img_type\": \"RGB\",\n"
                                     "  \"dynamic_input_dims\": true\n"
                                     "}\n"},
          {"etc/0_process_mla.json",
           R"json({
  "node_name": "mla_0",
  "input_buffers": [{"name": "preproc_0"}],
  "data_type": ["INT8"],
  "input_width": [640],
  "input_height": [640],
  "input_depth": [3],
  "output_width": [80],
  "output_height": [80],
  "output_depth": [6]
})json"},
      },
      true);
}

sima_test::MpkFixture make_quanttess_fixture(const std::string& tag) {
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
  "input_width": 1920,
  "input_height": 1080,
  "input_img_type": "RGB",
  "output_width": 640,
  "output_height": 640,
  "output_img_type": "RGB"
})json"},
                                             {"etc/0_quanttess.json",
                                              R"json({
  "node_name": "quanttess_0",
  "input_width": 640,
  "input_height": 640,
  "input_depth": 9,
  "caps": {"sink_pads": [], "src_pads": []}
})json"},
                                             {"etc/0_process_mla.json",
                                              R"json({
  "node_name": "mla_0",
  "input_buffers": [{"name": "preproc_0"}],
  "data_type": ["INT8"],
  "input_width": [640],
  "input_height": [640],
  "input_depth": [4],
  "output_width": [80],
  "output_height": [80],
  "output_depth": [6]
})json"},
                                         },
                                         true);
}

} // namespace

RUN_TEST("unit_model_input_spec_contract_test", ([] {
           using namespace simaai::neat;

           {
             const auto fixture = make_preproc_mla_fixture("model_input_spec_dynamic_rgb", "RGB");
             Model model(fixture.tar_path);
             const TensorSpec spec = model.input_spec();

             require(spec.rank == 3, "Model::input_spec RGB rank should be 3");
             require(
                 spec.shape == std::vector<int64_t>({-1, -1, 3}),
                 "Model::input_spec RGB should be dynamic spatial dims with fixed channel depth");
             require(!spec.dtypes.empty() && spec.dtypes[0] == TensorDType::UInt8,
                     "Model::input_spec RGB dtype should be UInt8");
             require(spec.image_format.has_value() &&
                         spec.image_format.value() == ImageSpec::PixelFormat::RGB,
                     "Model::input_spec RGB image_format mismatch");
           }

           {
             const auto fixture = make_preproc_mla_fixture("model_input_spec_dynamic_gray", "GRAY");
             Model::Options opt;
             opt.format = "GRAY";
             Model model(fixture.tar_path, opt);
             const TensorSpec spec = model.input_spec();

             require(spec.rank == 2, "Model::input_spec GRAY rank should be 2");
             require(spec.shape == std::vector<int64_t>({-1, -1}),
                     "Model::input_spec GRAY should expose dynamic 2D shape");
             require(spec.image_format.has_value() &&
                         spec.image_format.value() == ImageSpec::PixelFormat::GRAY8,
                     "Model::input_spec GRAY image_format mismatch");
           }

           {
             const auto fixture = make_quanttess_fixture("model_input_spec_dynamic_quanttess");
             Model::Options opt;
             opt.media_type = "application/vnd.simaai.tensor";
             opt.format.clear();
             Model model(fixture.tar_path, opt);

             const TensorSpec spec = model.input_spec();
             require(spec.rank == 3, "Model::input_spec tensor rank should be 3");
             require(spec.shape == std::vector<int64_t>({-1, -1, 4}),
                     "Model::input_spec tensor should be dynamic H/W with fixed MLA channel depth");
             require(!spec.dtypes.empty() && spec.dtypes[0] == TensorDType::Float32,
                     "Model::input_spec tensor dtype should be Float32");
           }
         }));
