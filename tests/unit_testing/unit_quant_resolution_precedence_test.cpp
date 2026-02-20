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

RUN_TEST("unit_quant_resolution_precedence_test", ([] {
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
           const nlohmann::json detess_cfg = {
               {"node_name", "post_0"},
               {"q_scale", nlohmann::json::array({0.99})},
               {"q_zp", nlohmann::json::array({0})},
               {"input_width", nlohmann::json::array({20})},
               {"input_height", nlohmann::json::array({10})},
               {"input_depth", nlohmann::json::array({91})},
           };

           TempFile mla = write_temp_json("quant_mla", mla_cfg);
           TempFile post = write_temp_json("quant_post", detess_cfg);
           const std::string pipeline =
               "fakesrc ! neatprocessmla name=mla config=\"" + mla.path +
               "\" ! simaaidetessdequant name=post config=\"" + post.path + "\" ! fakesink";

           ManifestBuildDiagnostics diag;
           const SimaPluginStaticManifest manifest =
               resolve_manifest_from_pipeline(pipeline, "sess-quant", &diag);
           const StageStaticSpec* post_stage = find_stage(manifest, "post");
           require(post_stage != nullptr, "post stage should exist");
           require(!post_stage->inputs.empty(),
                   "post stage inputs should be inferred from MLA outputs");
           require(post_stage->inputs[0].shape == std::vector<int64_t>({84, 80, 80}),
                   "post stage inferred input shape mismatch");
           require(!post_stage->output_quant.empty(),
                   "post stage quant should be inferred from MLA output quant");
           require(!post_stage->output_quant[0].scales.empty(),
                   "post stage inferred quant scale missing");
           require(post_stage->output_quant[0].scales[0] == 0.125,
                   "post stage should prefer MLA quant over JSON fallback");
           require(!post_stage->output_quant[0].zero_points.empty(),
                   "post stage inferred quant zero point missing");
           require(post_stage->output_quant[0].zero_points[0] == -7,
                   "post stage zero point should come from MLA quant");

           const auto quant_trace = find_trace_source(*post_stage, "output_quant");
           require(quant_trace.has_value(), "post output_quant trace should exist");
           require(quant_trace->find("infer:mla_transform_registry") != std::string::npos,
                   "post quant should trace MLA neighbor inference source");

           const nlohmann::json detess_only_cfg = {
               {"node_name", "post_only_0"},
               {"q_scale", nlohmann::json::array({0.75})},
               {"q_zp", nlohmann::json::array({3})},
               {"input_width", nlohmann::json::array({20})},
               {"input_height", nlohmann::json::array({10})},
               {"input_depth", nlohmann::json::array({91})},
           };
           TempFile post_only = write_temp_json("quant_post_only", detess_only_cfg);
           const std::string post_only_pipeline =
               "fakesrc ! simaaidetessdequant name=post_only config=\"" + post_only.path +
               "\" ! fakesink";
           ManifestBuildDiagnostics diag_post_only;
           const SimaPluginStaticManifest post_only_manifest = resolve_manifest_from_pipeline(
               post_only_pipeline, "sess-quant-only", &diag_post_only);
           const StageStaticSpec* post_only_stage = find_stage(post_only_manifest, "post_only");
           require(post_only_stage != nullptr, "post_only stage should exist");
           require(!post_only_stage->output_quant.empty(),
                   "post_only should fallback to JSON quant when MLA contract is absent");
           require(post_only_stage->output_quant[0].scales == std::vector<double>({0.75}),
                   "post_only JSON fallback scale mismatch");
           require(post_only_stage->output_quant[0].zero_points == std::vector<int64_t>({3}),
                   "post_only JSON fallback zero point mismatch");

           const auto post_only_trace = find_trace_source(*post_only_stage, "output_quant");
           require(post_only_trace.has_value(), "post_only output_quant trace should exist");
           require(post_only_trace->find("json_fallback") != std::string::npos,
                   "post_only quant trace should indicate JSON fallback");
         }));
