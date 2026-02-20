#include "model/Model.h"
#include "mpk_fixture_utils.h"
#include "test_main.h"
#include "test_utils.h"

#include <vector>

namespace {

sima_test::MpkFixture make_mla_params_output_fixture(const std::string& tag,
                                                     const std::string& output_format) {
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
          {"etc/0_preproc.json",
           R"json({
  "node_name": "preproc_0",
  "input_width": 1280,
  "input_height": 720,
  "input_img_type": "RGB",
  "output_width": 640,
  "output_height": 640,
  "output_img_type": "RGB",
  "dynamic_input_dims": true
})json"},
          {"etc/0_process_mla.json", std::string("{\n") +
                                         "  \"node_name\": \"mla_0\",\n"
                                         "  \"input_buffers\": [{\"name\": \"preproc_0\"}],\n"
                                         "  \"data_type\": [\"INT16\"],\n"
                                         "  \"simaai__params\": {\n"
                                         "    \"output_width\": [64],\n"
                                         "    \"output_height\": [48],\n"
                                         "    \"output_depth\": [7],\n"
                                         "    \"output_format\": \"" +
                                         output_format +
                                         "\"\n"
                                         "  }\n"
                                         "}\n"},
      },
      true);
}

sima_test::MpkFixture make_boxdecode_fixture(const std::string& tag) {
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
      },
      {
        "sequence_id": 3,
        "name": "boxdecode_0",
        "pluginId": "processcvu",
        "configPath": "0_boxdecode.json",
        "processor": "CVU",
        "kernel": "boxdecode",
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
                                             {"etc/0_boxdecode.json",
                                              R"json({
  "node_name": "boxdecode_0",
  "input_buffers": [{"name": "mla_0"}],
  "decode_type": "yolo"
})json"},
                                         },
                                         true);
}

} // namespace

RUN_TEST("unit_model_output_spec_contract_test", ([] {
           using namespace simaai::neat;

           {
             const auto fixture = make_mla_params_output_fixture("model_output_spec_nchw", "NCHW");
             Model model(fixture.tar_path);
             const TensorSpec spec = model.output_spec();

             require(!spec.dtypes.empty() && spec.dtypes[0] == TensorDType::Int16,
                     "Model::output_spec dtype should map from MLA data_type");
             require(spec.rank == 3, "Model::output_spec NCHW rank should be 3");
             require(spec.shape == std::vector<int64_t>({7, 48, 64}),
                     "Model::output_spec should respect NCHW layout ordering");
           }

           {
             const auto fixture = make_mla_params_output_fixture("model_output_spec_hwc", "HWC");
             Model model(fixture.tar_path);
             const TensorSpec spec = model.output_spec();

             require(spec.rank == 3, "Model::output_spec HWC rank should be 3");
             require(spec.shape == std::vector<int64_t>({48, 64, 7}),
                     "Model::output_spec should respect HWC layout ordering");
           }

           {
             const auto fixture = make_boxdecode_fixture("model_output_spec_boxdecode");
             Model model(fixture.tar_path);
             const TensorSpec spec = model.output_spec();

             require(spec.rank == -1,
                     "Model::output_spec boxdecode path should return unknown rank");
             require(!spec.dtypes.empty() && spec.dtypes[0] == TensorDType::UInt8,
                     "Model::output_spec boxdecode path should expose UInt8 dtype");
           }
         }));
