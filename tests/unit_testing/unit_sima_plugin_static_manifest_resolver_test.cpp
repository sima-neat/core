#include "pipeline/internal/sima/SimaPluginStaticManifestResolver.h"
#include "test_main.h"
#include "test_utils.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <fstream>
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

RUN_TEST("unit_sima_plugin_static_manifest_resolver_test", ([] {
           using namespace simaai::neat::pipeline_internal::sima;

           const nlohmann::json pre_cfg = {
               {"node_name", "pre_0"}, {"output_width", 640},      {"output_height", 640},
               {"output_channels", 3}, {"output_img_type", "RGB"},
               {"q_scale", nlohmann::json::array({0.03125})},
               {"q_zp", nlohmann::json::array({0})},
           };
           const nlohmann::json mla_cfg = {
               {"node_name", "mla_0"},
               {"simaai__params",
                {
                    {"model_path", "/opt/models/yolo_v8.elf"},
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
                    {"q_zp", nlohmann::json::array({-7})},
                }},
           };
           const nlohmann::json box_cfg = {
               {"node_name", "box_0"},
               {"decode_type", "yolov8"},
               {"topk", 100},
               {"detection_threshold", 0.35},
           };

           TempFile pre = write_temp_json("sima_manifest_pre", pre_cfg);
           TempFile mla = write_temp_json("sima_manifest_mla", mla_cfg);
           TempFile box = write_temp_json("sima_manifest_box", box_cfg);

           const std::string pipeline =
               "fakesrc ! neatprocesscvu name=pre config=\"" + pre.path +
               "\" ! neatprocessmla name=mla config=\"" + mla.path +
               "\" ! neatboxdecode name=box stage-id=logical_box config=\"" + box.path +
               "\" decode-type=yolov8p topk=120 detection-threshold=0.25 nms-iou-threshold=0.55"
               " ! fakesink";

           ManifestBuildDiagnostics diag;
           const SimaPluginStaticManifest manifest =
               resolve_manifest_from_pipeline(pipeline, "sess-1", &diag);

           require(manifest.manifest_version == 2, "manifest version should be 2");
           require(manifest.session_id == "sess-1", "manifest session id mismatch");
           require(manifest.model_id == "/opt/models/yolo_v8.elf",
                   "manifest model id should come from MLA model_path");

           const StageStaticSpec* pre_stage = find_stage(manifest, "pre");
           const StageStaticSpec* mla_stage = find_stage(manifest, "mla");
           const StageStaticSpec* box_stage = find_stage(manifest, "box");
           require(pre_stage != nullptr, "pre stage should be present in manifest");
           require(mla_stage != nullptr, "mla stage should be present in manifest");
           require(box_stage != nullptr, "box stage should be present in manifest");

           require(!pre_stage->outputs.empty(),
                   "pre stage outputs should be inferred from MLA input contract");
           require(pre_stage->outputs[0].shape == std::vector<int64_t>({640, 640, 3}),
                   "pre stage inferred output shape mismatch");
           require(pre_stage->runtime_defaults.contains("config_path"),
                   "pre stage runtime defaults should include config_path");
           require(pre_stage->runtime_defaults["config_path"] == pre.path,
                   "pre stage config_path default mismatch");
           require(!pre_stage->output_quant.empty(),
                   "pre stage output quant should be sourced from stage JSON");
           require(!pre_stage->output_quant[0].scales.empty(),
                   "pre stage output quant scale missing");
           require(pre_stage->output_quant[0].scales[0] == 0.03125,
                   "pre stage output quant scale mismatch");

           require(!mla_stage->outputs.empty(), "MLA stage outputs should be populated");
           require(mla_stage->outputs[0].shape == std::vector<int64_t>({84, 80, 80}),
                   "MLA stage output shape mismatch");
           require(!mla_stage->output_quant.empty(),
                   "MLA stage output quant should come from MLA config");
           require(mla_stage->runtime_defaults.contains("config_path"),
                   "MLA stage runtime defaults should include config_path");
           require(mla_stage->runtime_defaults["config_path"] == mla.path,
                   "MLA stage config_path default mismatch");
           require(mla_stage->runtime_defaults.contains("input_quant"),
                   "MLA stage should receive input quant contract from upstream stage JSON");
           require(mla_stage->runtime_defaults["input_quant"].is_array() &&
                       !mla_stage->runtime_defaults["input_quant"].empty(),
                   "MLA stage input quant contract payload missing");

           require(!box_stage->inputs.empty(),
                   "box stage inputs should be inferred from MLA outputs");
           require(box_stage->inputs[0].shape == std::vector<int64_t>({84, 80, 80}),
                   "box stage inferred input shape mismatch");
           require(box_stage->logical_stage_id == "logical_box",
                   "box stage logical_stage_id should honor stage-id property");
           require(box_stage->runtime_defaults.contains("decode_type"),
                   "box stage runtime defaults should include decode_type");
           require(box_stage->runtime_defaults["decode_type"] == "yolov8p",
                   "box stage decode_type should prefer property over JSON");
           require(box_stage->runtime_defaults["topk"] == 120,
                   "box stage topk should come from property");
           require(box_stage->runtime_defaults["detection_threshold"] == 0.25,
                   "box stage detection threshold should come from property");
           require(box_stage->runtime_defaults["nms_iou_threshold"] == 0.55,
                   "box stage nms threshold should come from property");
           require(box_stage->runtime_defaults.contains("config_path"),
                   "box stage runtime defaults should include config_path");
           require(box_stage->runtime_defaults["config_path"] == box.path,
                   "box stage config_path default mismatch");

           const nlohmann::json tess_cfg = {
               {"node_name", "tess_0"},
           };
           const nlohmann::json deq_cfg = {
               {"node_name", "deq_0"},
           };
           TempFile tess = write_temp_json("sima_manifest_tess", tess_cfg);
           TempFile deq = write_temp_json("sima_manifest_deq", deq_cfg);

           const std::string transform_pipeline =
               "fakesrc ! simaaitessellate name=tess stage-id=logical_tess config=\"" + tess.path +
               "\" ! neatprocessmla name=mla2 config=\"" + mla.path +
               "\" ! simaaidequantize name=deq stage-id=logical_deq config=\"" + deq.path +
               "\" ! fakesink";

           ManifestBuildDiagnostics transform_diag;
           const SimaPluginStaticManifest transform_manifest =
               resolve_manifest_from_pipeline(transform_pipeline, "sess-transform", &transform_diag);
           const StageStaticSpec* tess_stage = find_stage(transform_manifest, "tess");
           const StageStaticSpec* deq_stage = find_stage(transform_manifest, "deq");
           require(tess_stage != nullptr, "tessellate stage should be present");
           require(deq_stage != nullptr, "dequantize stage should be present");
           require(tess_stage->kernel_kind == "tessellate",
                   "tess stage kernel kind should be tessellate");
           require(deq_stage->kernel_kind == "dequantize",
                   "deq stage kernel kind should be dequantize");
           require(!tess_stage->outputs.empty(),
                   "tess stage outputs should be inferred from MLA input contract");
           require(tess_stage->outputs[0].shape == std::vector<int64_t>({640, 640, 3}),
                   "tess stage inferred output shape mismatch");
           require(!deq_stage->inputs.empty(),
                   "deq stage inputs should be inferred from MLA output contract");
           require(deq_stage->inputs[0].shape == std::vector<int64_t>({84, 80, 80}),
                   "deq stage inferred input shape mismatch");
           require(!deq_stage->output_quant.empty(),
                   "deq stage should inherit output quant contract from MLA outputs");
           require(!deq_stage->output_quant[0].scales.empty(),
                   "deq stage inherited quant scale missing");
           require(deq_stage->output_quant[0].scales[0] == 0.125,
                   "deq stage inherited quant scale mismatch");

           const nlohmann::json mla_cfg_no_quant = {
               {"node_name", "mla_no_quant_0"},
               {"simaai__params",
                {
                    {"model_path", "/opt/models/resnet50.elf"},
                    {"input_width", nlohmann::json::array({224})},
                    {"input_height", nlohmann::json::array({224})},
                    {"input_depth", nlohmann::json::array({3})},
                    {"input_format", "HWC"},
                    {"output_width", nlohmann::json::array({1})},
                    {"output_height", nlohmann::json::array({1})},
                    {"output_depth", nlohmann::json::array({1000})},
                    {"output_format", "HWC"},
                    {"data_type", nlohmann::json::array({"INT8"})},
                }},
           };
           const nlohmann::json detess_cfg = {
               {"node_name", "detessdequant_0"},
               {"dq_scale", nlohmann::json::array({7.2624930847688303})},
               {"dq_zp", nlohmann::json::array({-67})},
           };

           TempFile mla_no_quant = write_temp_json("sima_manifest_mla_no_quant", mla_cfg_no_quant);
           TempFile detess = write_temp_json("sima_manifest_detess", detess_cfg);

           const std::string detess_pipeline =
               "fakesrc ! neatprocesscvu name=pre_detess stage-id=pre_detess config=\"" + pre.path +
               "\" ! neatprocessmla name=mla_detess stage-id=mla_detess config=\"" +
               mla_no_quant.path +
               "\" ! neatprocesscvu name=detessdequant stage-id=detessdequant config=\"" +
               detess.path + "\" ! fakesink";

           ManifestBuildDiagnostics detess_diag;
           const SimaPluginStaticManifest detess_manifest =
               resolve_manifest_from_pipeline(detess_pipeline, "sess-detess", &detess_diag);
           const StageStaticSpec* mla_detess_stage = find_stage(detess_manifest, "mla_detess");
           const StageStaticSpec* detess_stage = find_stage(detess_manifest, "detessdequant");
           require(mla_detess_stage != nullptr, "mla stage should be present in detess manifest");
           require(detess_stage != nullptr, "detess stage should be present");
           require(detess_diag.errors.empty(),
                   "detess stage should satisfy strict quant contract via JSON fallback");
           require(!mla_detess_stage->output_quant.empty(),
                   "MLA stage should derive output quant from downstream detess JSON fallback");
           require(mla_detess_stage->runtime_defaults.contains("input_quant"),
                   "MLA detess stage should carry upstream input quant payload");
           require(mla_detess_stage->runtime_defaults["input_quant"].is_array() &&
                       !mla_detess_stage->runtime_defaults["input_quant"].empty(),
                   "MLA detess stage input quant payload missing");
           require(!mla_detess_stage->output_quant[0].scales.empty(),
                   "MLA stage downstream-derived quant scale missing");
           require(mla_detess_stage->output_quant[0].scales[0] == 7.2624930847688303,
                   "MLA stage downstream-derived quant scale mismatch");
           require(!detess_stage->output_quant.empty(),
                   "detess stage should derive quant from dq_scale/dq_zp JSON fallback");
           require(!detess_stage->output_quant[0].scales.empty(),
                   "detess stage dq_scale fallback missing");
           require(!detess_stage->output_quant[0].zero_points.empty(),
                   "detess stage dq_zp fallback missing");
           require(detess_stage->output_quant[0].scales[0] == 7.2624930847688303,
                   "detess stage dq_scale fallback mismatch");
           require(detess_stage->output_quant[0].zero_points[0] == -67,
                   "detess stage dq_zp fallback mismatch");
         }));
