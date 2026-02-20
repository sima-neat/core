#include "pipeline/internal/sima/SimaPluginStaticManifest.h"
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

RUN_TEST("unit_wiring_tensor_index_test", ([] {
           using namespace simaai::neat::pipeline_internal::sima;

           const nlohmann::json mla_cfg = {
               {"node_name", "mla_0"},
               {"simaai__params",
                {{"model_path", "/opt/models/yolo_v8.elf"},
                 {"input_width", nlohmann::json::array({640})},
                 {"input_height", nlohmann::json::array({640})},
                 {"input_depth", nlohmann::json::array({3})},
                 {"input_format", "HWC"},
                 {"output_width", nlohmann::json::array({80, 40})},
                 {"output_height", nlohmann::json::array({80, 40})},
                 {"output_depth", nlohmann::json::array({84, 84})},
                 {"output_format", "CHW"},
                 {"data_type", nlohmann::json::array({"INT8", "INT8"})},
                 {"q_scale", nlohmann::json::array({0.125, 0.25})},
                 {"q_zp", nlohmann::json::array({-7, -9})}}},
           };
           const nlohmann::json box_cfg = {
               {"node_name", "box_0"},
               {"decode_type", "yolov8"},
           };
           TempFile mla = write_temp_json("wiring_mla", mla_cfg);
           TempFile box = write_temp_json("wiring_box", box_cfg);

           const std::string pipeline =
               "fakesrc ! neatprocessmla name=mla config=\"" + mla.path +
               "\" ! neatboxdecode name=box config=\"" + box.path + "\" ! fakesink";

           ManifestBuildDiagnostics diag;
           const SimaPluginStaticManifest manifest =
               resolve_manifest_from_pipeline(pipeline, "sess-wiring", &diag);
           const StageStaticSpec* box_stage = find_stage(manifest, "box");
           require(box_stage != nullptr, "box stage should exist");
           require(box_stage->inputs.size() == 2,
                   "box stage should have two inferred inputs from MLA outputs");
           require(box_stage->sink_pad_tensor_index_map == std::vector<int>({0, 1}),
                   "sink pad -> tensor index map should be deterministic [0,1]");

           const auto trace = find_trace_source(*box_stage, "sink_pad_tensor_index_map");
           require(trace.has_value(), "sink pad tensor index map trace should exist");
           require(trace->find("infer:stage_input_order") != std::string::npos,
                   "sink pad mapping trace source should be infer:stage_input_order");

           const std::string payload = serialize_manifest_json(manifest);
           const auto roundtrip = parse_manifest_json(payload);
           require(roundtrip.has_value(), "manifest should parse after serialize");
           const StageStaticSpec* box_roundtrip = find_stage(*roundtrip, "box");
           require(box_roundtrip != nullptr, "roundtrip manifest should include box stage");
           require(box_roundtrip->sink_pad_tensor_index_map == std::vector<int>({0, 1}),
                   "roundtrip sink_pad_tensor_index_map mismatch");
         }));

