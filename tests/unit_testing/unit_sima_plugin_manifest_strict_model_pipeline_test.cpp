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

const simaai::neat::pipeline_internal::sima::StageStaticSpec* find_stage(
    const simaai::neat::pipeline_internal::sima::SimaPluginStaticManifest& manifest,
    const std::string& element_name) {
  for (const auto& stage : manifest.stages) {
    if (stage.element_name == element_name) {
      return &stage;
    }
  }
  return nullptr;
}

} // namespace

RUN_TEST("unit_sima_plugin_manifest_strict_model_pipeline_test", ([] {
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

           TempFile mla = write_temp_json("strict_model_mla", mla_cfg);

           const std::string pipeline =
               "fakesrc ! neatprocessmla name=mla stage-id=stage_mla config=\"" + mla.path +
               "\" ! neatboxdecode name=box stage-id=stage_box decode-type=yolov8 "
               "detection-threshold=0.25 nms-iou-threshold=0.55 topk=120 ! fakesink";

           ManifestBuildDiagnostics strict_diag;
           const SimaPluginStaticManifest manifest =
               resolve_manifest_from_pipeline(pipeline, "sess-strict-model", &strict_diag);

           require(strict_diag.errors.empty(),
                   "model pipeline with MLA contract + properties should resolve without fallback errors");

           const StageStaticSpec* box_stage = find_stage(manifest, "box");
           require(box_stage != nullptr, "box stage should exist");
           require(!box_stage->inputs.empty(),
                   "box stage inputs should be inferred from MLA contract in strict mode");
           require(box_stage->runtime_defaults.contains("decode_type"),
                   "box stage decode_type should come from properties in strict mode");
         }));
