#include "pipeline/internal/sima/SimaPluginStaticManifestResolver.h"
#include "test_main.h"
#include "test_utils.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <fstream>
#include <string>
#include <unistd.h>

namespace {

struct TempFile {
  std::string path;
  explicit TempFile(std::string p) : path(std::move(p)) {}
  ~TempFile() {
    if (!path.empty()) {
      std::remove(path.c_str());
    }
  }
};

TempFile write_temp_json(const std::string& tag, const nlohmann::json& j) {
  const std::string path = "/tmp/" + tag + "_" + std::to_string(::getpid()) + ".json";
  std::ofstream out(path);
  require(out.is_open(), "failed to create temp json " + path);
  out << j.dump(2);
  return TempFile(path);
}

} // namespace

RUN_TEST("unit_boxdecode_required_runtime_fields_test", ([] {
           using namespace simaai::neat::pipeline_internal::sima;

           const nlohmann::json mla_cfg = {
               {"node_name", "mla_0"},
               {"simaai__params",
                {{"model_path", "/opt/models/yolo_v8.elf"},
                 {"input_width", nlohmann::json::array({640})},
                 {"input_height", nlohmann::json::array({640})},
                 {"input_depth", nlohmann::json::array({3})},
                 {"input_format", "HWC"},
                 {"output_width", nlohmann::json::array({80})},
                 {"output_height", nlohmann::json::array({80})},
                 {"output_depth", nlohmann::json::array({84})},
                 {"output_format", "CHW"},
                 {"data_type", nlohmann::json::array({"INT8"})},
                 {"q_scale", nlohmann::json::array({0.125})},
                 {"q_zp", nlohmann::json::array({-7})}}},
           };

           TempFile mla = write_temp_json("required_runtime_mla", mla_cfg);

           const std::string pipeline =
               "fakesrc ! neatprocessmla name=mla stage-id=stage_mla config=\"" + mla.path +
               "\" ! neatboxdecode name=box stage-id=stage_box decode-type=yolov8 ! fakesink";

           ManifestBuildDiagnostics diag;
           (void)resolve_manifest_from_pipeline(pipeline, "sess-required-runtime", &diag);

           require(!diag.errors.empty(),
                   "model-managed boxdecode stage with missing runtime knobs should emit diagnostics");
           bool missing_topk = false;
           bool missing_det = false;
           bool missing_nms = false;
           for (const auto& err : diag.errors) {
             if (err.find("runtime field 'topk'") != std::string::npos) {
               missing_topk = true;
             }
             if (err.find("runtime field 'detection_threshold'") != std::string::npos) {
               missing_det = true;
             }
             if (err.find("runtime field 'nms_iou_threshold'") != std::string::npos) {
               missing_nms = true;
             }
           }

           require(missing_topk, "missing topk runtime field should be reported");
           require(missing_det, "missing detection_threshold runtime field should be reported");
           require(missing_nms, "missing nms_iou_threshold runtime field should be reported");
         }));
