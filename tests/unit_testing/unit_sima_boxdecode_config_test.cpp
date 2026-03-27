#include "model/Model.h"
#include "mpk_fixture_utils.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "test_main.h"
#include "test_utils.h"

#include <nlohmann/json.hpp>
#include <sstream>

namespace {

sima_test::MpkFixture make_boxdecode_fixture(const std::string& tag,
                                             double detection_threshold = 0.52) {
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
  "memory": {
    "cpu": 0,
    "next_cpu": 0
  },
  "system": {
    "out_buf_queue": 1,
    "debug": 0
  },
  "buffers": {
    "output": {
      "size": 580
    }
  },
  "decode_type": "yolov8",
  "original_width": 320,
  "original_height": 240,
  "detection_threshold": )json" + std::to_string(detection_threshold) +
                                                  R"json(,
  "nms_iou_threshold": 0.45,
  "topk": 24
})json"},
                                         },
                                         true);
}

} // namespace

RUN_TEST("unit_sima_boxdecode_config_test", ([] {
           using namespace simaai::neat;

           const auto fixture = make_boxdecode_fixture("boxdecode_async_queue_override");
           Model model(fixture.tar_path);

           auto node = nodes::SimaBoxDecode(model);
           auto* box = dynamic_cast<simaai::neat::SimaBoxDecode*>(node.get());
           require(box != nullptr, "SimaBoxDecode cast failed");

           const nlohmann::json* cfg = box->config_json();
           require(cfg != nullptr, "SimaBoxDecode config_json missing");
           require(cfg->contains("system") && (*cfg)["system"].is_object(),
                   "SimaBoxDecode config should expose system object");
           require((*cfg)["system"]["out_buf_queue"].get<int>() == 4,
                   "SimaBoxDecode model-backed async config should override system.out_buf_queue");
           require((*cfg)["original_width"].get<int>() == 320,
                   "SimaBoxDecode should preserve model-pack original_width by default");
           require((*cfg)["original_height"].get<int>() == 240,
                   "SimaBoxDecode should preserve model-pack original_height by default");

           const std::string fragment = box->backend_fragment(3);
           require_contains(fragment, "num-buffers=4",
                            "SimaBoxDecode fragment should expose async num-buffers");

           const auto warning_fixture =
               make_boxdecode_fixture("boxdecode_warning_threshold_cliff", 0.5);
           std::ostringstream captured;
           auto* old_cerr = std::cerr.rdbuf(captured.rdbuf());
           try {
             auto warning_node = nodes::SimaBoxDecode(Model(warning_fixture.tar_path));
             require(warning_node != nullptr, "warning fixture should build SimaBoxDecode");
           } catch (...) {
             std::cerr.rdbuf(old_cerr);
             throw;
           }
           std::cerr.rdbuf(old_cerr);
           require_contains(
               captured.str(), "detection-threshold=0.500",
               "SimaBoxDecode should warn when effective YOLOv8 threshold resolves to <= 0.5");
         }));
