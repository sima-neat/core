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

RUN_TEST("unit_sima_plugin_manifest_strict_fallback_test", ([] {
           using namespace simaai::neat::pipeline_internal::sima;

           const nlohmann::json box_cfg = {
               {"node_name", "box_only_0"},
               {"decode_type", "yolov8"},
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

           require(!strict_diag.errors.empty(),
                   "resolver should fail when required runtime/static fields are unresolved");
           bool found_missing_decode_type = false;
           for (const auto& err : strict_diag.errors) {
             if (err.find("Missing required runtime field 'decode_type'") != std::string::npos) {
               found_missing_decode_type = true;
               break;
             }
           }
           require(found_missing_decode_type,
                   "resolver should not resolve decode_type from stage JSON fallback");
           require(!manifest.stages.empty(), "manifest should still include parsed stages");
           const auto& stage = manifest.stages.front();
           require(!stage.runtime_defaults.contains("decode_type"),
                   "runtime defaults must not be populated from stage JSON");
         }));
