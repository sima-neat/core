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

const simaai::neat::pipeline_internal::sima::StageStaticSpec*
find_stage(const simaai::neat::pipeline_internal::sima::SimaPluginStaticManifest& manifest,
           const std::string& element_name) {
  for (const auto& stage : manifest.stages) {
    if (stage.element_name == element_name) {
      return &stage;
    }
  }
  return nullptr;
}

} // namespace

RUN_TEST("unit_sima_plugin_manifest_strict_fallback_test", ([] {
           using namespace simaai::neat::pipeline_internal::sima;

           const nlohmann::json box_cfg = {
               {"node_name", "box_only_0"},
               {"decode_type", "yolov8"},
               {"topk", 200},
               {"detection_threshold", 0.15},
               {"nms_iou_threshold", 0.45},
               {"input_width", nlohmann::json::array({20})},
               {"input_height", nlohmann::json::array({10})},
               {"input_depth", nlohmann::json::array({91})},
               {"input_format", "CHW"},
           };
           TempFile box = write_temp_json("strict_manifest_box", box_cfg);

           const std::string pipeline =
               "fakesrc ! neatboxdecode name=box_only stage-id=stage_box config=\"" + box.path +
               "\" ! fakesink";

           ManifestBuildDiagnostics strict_diag;
           const SimaPluginStaticManifest manifest =
               resolve_manifest_from_pipeline(pipeline, "sess-strict", &strict_diag);

           require(
               strict_diag.errors.empty(),
               "model-managed boxdecode should resolve packaged runtime defaults from stage JSON");
           const auto* stage = find_stage(manifest, "box_only");
           require(stage != nullptr, "manifest should include the named boxdecode stage");
           require(stage->runtime_defaults.contains("decode_type"),
                   "runtime defaults should include decode_type from stage JSON");
           require(stage->runtime_defaults["decode_type"] == "yolov8",
                   "decode_type should match the packaged boxdecode JSON");
           require(stage->runtime_defaults.contains("detection_threshold"),
                   "runtime defaults should include detection_threshold from stage JSON");
           require(stage->runtime_defaults["detection_threshold"] == 0.15,
                   "detection_threshold should match the packaged boxdecode JSON");
           require(stage->runtime_defaults.contains("nms_iou_threshold"),
                   "runtime defaults should include nms_iou_threshold from stage JSON");
           require(stage->runtime_defaults["nms_iou_threshold"] == 0.45,
                   "nms_iou_threshold should match the packaged boxdecode JSON");
           require(stage->runtime_defaults.contains("topk"),
                   "runtime defaults should include topk from stage JSON");
           require(stage->runtime_defaults["topk"] == 200,
                   "topk should match the packaged boxdecode JSON");
         }));
