#include "pipeline/internal/sima/SimaPluginStaticManifestResolver.h"
#include "test_main.h"
#include "test_utils.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <fstream>
#include <optional>
#include <string>
#include <unistd.h>
#include <vector>

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

std::optional<std::string> find_trace_source(
    const simaai::neat::pipeline_internal::sima::StageStaticSpec& stage,
    const std::string& field) {
  for (const auto& trace : stage.resolution_trace) {
    if (trace.field == field) {
      return trace.source_used;
    }
  }
  return std::nullopt;
}

} // namespace

RUN_TEST("unit_boxdecode_resolution_precedence_test", ([] {
           using namespace simaai::neat::pipeline_internal::sima;

           const nlohmann::json pre_cfg = {{"node_name", "pre_0"}};
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
           const nlohmann::json box_cfg = {
               {"node_name", "box_0"},
               {"decode_type", "yolov8"},
               {"topk", 200},
               {"detection_threshold", 0.15},
               {"nms_iou_threshold", 0.45},
               {"input_width", nlohmann::json::array({13})},
               {"input_height", nlohmann::json::array({17})},
               {"input_depth", nlohmann::json::array({99})},
               {"input_format", "CHW"},
           };

           TempFile pre = write_temp_json("box_pre", pre_cfg);
           TempFile mla = write_temp_json("box_mla", mla_cfg);
           TempFile box = write_temp_json("box_cfg", box_cfg);

           const std::string pipeline =
               "fakesrc ! neatprocesscvu name=pre config=\"" + pre.path +
               "\" ! neatprocessmla name=mla config=\"" + mla.path +
               "\" ! neatboxdecode name=box config=\"" + box.path +
               "\" decode-type=yolov8p detection-threshold=0.25 nms-iou-threshold=0.55 topk=120 "
               "! fakesink";

           ManifestBuildDiagnostics diag;
           const SimaPluginStaticManifest manifest =
               resolve_manifest_from_pipeline(pipeline, "sess-box", &diag);
           const StageStaticSpec* box_stage = find_stage(manifest, "box");
           require(box_stage != nullptr, "box stage should exist");
           require(!box_stage->inputs.empty(), "box stage should have inferred inputs");
           require(box_stage->inputs[0].shape == std::vector<int64_t>({84, 80, 80}),
                   "box inputs should come from MLA contract, not box JSON fallback");
           require(box_stage->runtime_defaults.contains("decode_type"),
                   "box runtime defaults should include decode_type");
           require(box_stage->runtime_defaults["decode_type"] == "yolov8p",
                   "box decode_type should prefer property over JSON");
           require(box_stage->runtime_defaults["topk"] == 120,
                   "box topk should prefer property over JSON");
           require(box_stage->runtime_defaults["detection_threshold"] == 0.25,
                   "box detection threshold should prefer property over JSON");
           require(box_stage->runtime_defaults["nms_iou_threshold"] == 0.55,
                   "box nms threshold should prefer property over JSON");

           const auto inputs_trace = find_trace_source(*box_stage, "inputs");
           require(inputs_trace.has_value(), "box inputs trace should exist");
           require(inputs_trace->find("infer:mla_transform_registry") != std::string::npos,
                   "box inputs should prefer inferred MLA contract over JSON fallback");

           const nlohmann::json box_only_cfg = {
               {"node_name", "box_only_0"},
               {"decode_type", "yolov8"},
               {"input_width", nlohmann::json::array({20})},
               {"input_height", nlohmann::json::array({10})},
               {"input_depth", nlohmann::json::array({91})},
               {"input_format", "CHW"},
           };
           TempFile box_only = write_temp_json("box_only_cfg", box_only_cfg);

           const std::string box_only_pipeline =
               "fakesrc ! neatboxdecode name=box_only config=\"" + box_only.path + "\" ! fakesink";
           ManifestBuildDiagnostics diag_box_only;
           const SimaPluginStaticManifest box_only_manifest =
               resolve_manifest_from_pipeline(box_only_pipeline, "sess-box-only", &diag_box_only);
           const StageStaticSpec* box_only_stage = find_stage(box_only_manifest, "box_only");
           require(box_only_stage != nullptr, "box_only stage should exist");
           require(!box_only_stage->inputs.empty(),
                   "box_only stage should fallback to JSON tensor fields when MLA contract is absent");
           require(box_only_stage->inputs[0].shape == std::vector<int64_t>({91, 10, 20}),
                   "box_only fallback input shape mismatch");

           const auto fallback_trace = find_trace_source(*box_only_stage, "inputs");
           require(fallback_trace.has_value(), "box_only inputs trace should exist");
           require(fallback_trace->find("json_fallback") != std::string::npos,
                   "box_only inputs should use JSON fallback when inference is unavailable");
         }));
