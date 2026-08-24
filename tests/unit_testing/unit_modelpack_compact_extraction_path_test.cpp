#include "model/internal/ModelArchiveLoader.h"
#include "model/internal/ModelPack.h"
#include "model_archive_fixture_utils.h"
#include "model_archive_test_utils.h"
#include "test_main.h"
#include "test_utils.h"

#include <filesystem>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

RUN_TEST("unit_modelpack_compact_extraction_path_test", ([] {
           namespace fs = std::filesystem;
           using nlohmann::json;
           using simaai::neat::internal::ModelArchiveLoader;
           using simaai::neat::internal::ModelArchiveLoaderOptions;
           using simaai::neat::internal::ModelPack;

           const auto fixture = sima_test::make_model_archive_fixture(
               "modelpack_compact_extraction_source",
               {{"etc/pipeline_sequence.json",
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
  "input_width": 64,
  "input_height": 48,
  "input_img_type": "RGB",
  "output_width": 64,
  "output_height": 48,
  "output_img_type": "RGB"
})json"},
                {"etc/0_process_mla.json",
                 R"json({
  "node_name": "mla_0",
  "input_buffers": [{"name": "preproc_0"}],
  "data_type": ["INT8"],
  "output_width": [64],
  "output_height": [48],
  "output_depth": [3],
  "simaai__params": {"model_path": "placeholder.elf"}
})json"},
                {"etc/model_mpk.json", R"json({"plugins": []})json"}});

           const fs::path scratch =
               fs::path(sima_test::make_temp_dir("modelpack_compact_extraction"));
           const std::string logical_name(120U, 'a');
           const fs::path archive = scratch / (logical_name + ".tar.gz");
           fs::copy_file(fixture.tar_path, archive, fs::copy_options::overwrite_existing);

           const auto inspected = ModelArchiveLoader::inspect(archive.string());
           require(inspected.package_name == logical_name,
                   "archive inspection must retain the logical package name");

           // The generic loader keeps its archive-derived physical directory by default.
           const fs::path direct_root = scratch / "direct";
           const auto direct = ModelArchiveLoader::extract(archive.string(), direct_root.string());
           require(fs::path(direct.package_root).filename() == logical_name,
                   "the generic loader default must retain its public directory behavior");

           // ModelPack owns a unique pkg_<archive-identity> parent, so its redundant archive-name
           // leaf is compact even when both the extraction base and logical package name are long.
           const fs::path runtime_base = scratch / std::string(80U, 'r');
           sima_test::ScopedEnvVar extract_root("SIMA_MPK_EXTRACT_ROOT", runtime_base.string());
           ModelPack pack(archive.string());
           const fs::path package_root = fs::path(pack.etc_dir()).parent_path();
           require(pack.logical_package_name() == logical_name,
                   "runtime ModelPack must retain the logical package name");
           require(package_root.filename() == "p",
                   "runtime ModelPack must use the compact physical package leaf");
           require(package_root.parent_path().filename().string().rfind("pkg_", 0U) == 0U,
                   "compact package leaf must remain inside the unique archive-identity directory");
           require(package_root.string().find(logical_name) == std::string::npos,
                   "physical runtime paths must not repeat the logical archive name");

           std::ifstream config(pack.etc_dir() + "/0_process_mla.json");
           require(config.is_open(),
                   "rewritten MLA config should exist under compact package root");
           json rewritten;
           config >> rewritten;
           const std::string model_path = rewritten["simaai__params"]["model_path"];
           require(model_path == (package_root / "share" / "placeholder.elf").string(),
                   "runtime config model path must be rewritten against compact package root");
           require(fs::exists(model_path), "rewritten compact MLA model path should exist");

           ModelArchiveLoaderOptions invalid;
           invalid.physical_package_leaf = "../escape";
           bool rejected = false;
           try {
             (void)ModelArchiveLoader::extract(archive.string(), (scratch / "invalid").string(),
                                               invalid);
           } catch (const simaai::neat::internal::ModelArchiveError&) {
             rejected = true;
           }
           require(rejected, "physical package leaf override must reject multiple path components");
         }));
