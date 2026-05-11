#include "asset_utils.h"
#include "model/internal/ModelPack.h"
#include "mpk_fixture_utils.h"
#include "mpk_test_utils.h"
#include "test_main.h"
#include "test_utils.h"

#include <filesystem>
#include <fstream>
#include <functional>
#include <string>

namespace {

using simaai::neat::internal::ModelPack;

bool throws_with(const std::function<void()>& fn, const std::string& needle) {
  try {
    fn();
  } catch (const std::exception& e) {
    if (needle.empty())
      return true;
    return std::string(e.what()).find(needle) != std::string::npos;
  }
  return false;
}

bool has_ext(const std::filesystem::path& dir, const char* ext) {
  for (const auto& it : std::filesystem::directory_iterator(dir)) {
    if (!it.is_regular_file())
      continue;
    if (it.path().extension() == ext)
      return true;
  }
  return false;
}

} // namespace

RUN_TEST(
    "unit_modelpack_extract_test", ([] {
      namespace fs = std::filesystem;

      const std::string strict_tar =
          sima_test::resolve_modelzoo_tar("yolo_v9c_seg", sima_test::repo_root_for_modelzoo());
      require(!strict_tar.empty(), "missing strict fixture tar; run sima-cli modelzoo get yolo_v9c_seg");
      require(fs::exists(strict_tar), "strict fixture tar path does not exist: " + strict_tar);

      const std::string extract_root = sima_test::make_temp_dir("modelpack_extract_root");
      sima_test::ScopedEnvVar env_root("SIMA_MPK_EXTRACT_ROOT", extract_root);
      sima_test::ScopedEnvVar env_gc("SIMA_MPK_EXTRACT_GC_STALE_PROC", "1");

      const fs::path stale_dead = fs::path(extract_root) / "proc_99999991";
      const fs::path stale_keep = fs::path(extract_root) / "proc_99999992";
      {
        std::error_code ec;
        fs::create_directories(stale_dead, ec);
        fs::create_directories(stale_keep, ec);
        std::ofstream dead_out(stale_dead / "dummy.txt", std::ios::out | std::ios::trunc);
        dead_out << "orphan\n";
        std::ofstream keep_out(stale_keep / ".sima_modelpack_keep", std::ios::out | std::ios::trunc);
        keep_out << "keep\n";
      }

      ModelPack first(strict_tar);
      const fs::path etc_dir(first.etc_dir());
      const fs::path package_root = etc_dir.parent_path();
      const fs::path lib_dir = package_root / "lib";
      const fs::path share_dir = package_root / "share";

      require(fs::exists(package_root), "ModelPack extraction package root must exist");
      require(fs::exists(etc_dir), "ModelPack extraction etc dir must exist");
      require(fs::exists(lib_dir), "ModelPack extraction lib dir must exist");
      require(fs::exists(share_dir), "ModelPack extraction share dir must exist");

      require(has_ext(etc_dir, ".json"),
              "ModelPack extraction etc dir should contain JSON configs");
      require(has_ext(lib_dir, ".so") || has_ext(share_dir, ".elf"),
              "ModelPack extraction should expose runtime artifacts (.so or .elf)");
      require(has_ext(share_dir, ".elf"),
              "ModelPack extraction share dir should contain .elf artifacts");

      const fs::path canonical_root = fs::weakly_canonical(fs::path(extract_root));
      const fs::path canonical_package = fs::weakly_canonical(package_root);
      require(canonical_package.string().find(canonical_root.string()) == 0,
              "ModelPack extraction must stay inside SIMA_MPK_EXTRACT_ROOT");
      require(!fs::exists(stale_dead), "stale dead proc_* extraction root should be auto-removed");
      require(fs::exists(stale_keep),
              "stale proc_* extraction root with keep marker should be preserved");

      const fs::path strict_copy = fs::path(extract_root) / "strict_copy_for_keep.mpk";
      std::error_code copy_ec;
      fs::copy_file(strict_tar, strict_copy, fs::copy_options::overwrite_existing, copy_ec);
      require(!copy_ec && fs::exists(strict_copy),
              "failed to prepare strict MPK copy for cleanup-disabled extraction check");

      {
        sima_test::ScopedEnvVar env_cleanup("SIMA_MPK_CLEANUP_EXTRACTED", "0");
        ModelPack keep_pack(strict_copy.string());
        const fs::path keep_proc_root = fs::path(keep_pack.etc_dir()).parent_path().parent_path();
        require(fs::exists(keep_proc_root / ".sima_modelpack_keep"),
                "cleanup-disabled extraction should emit keep marker in proc root");
      }

      ModelPack second(strict_tar);
      require(second.etc_dir() == first.etc_dir(),
              "ModelPack extraction should be deterministic for same archive and root");

      require(throws_with(
                  [&]() {
                    ModelPack bad(
                        sima_test::mpk_fixture_path("invalid/symlink_escape.mpk").string());
                    (void)bad.etc_dir();
                  },
                  "path_traversal"),
              "symlink escape fixture should be rejected with path_traversal");

      require(throws_with(
                  [&]() {
                    ModelPack bad(
                        sima_test::mpk_fixture_path("invalid/hardlink_escape.mpk").string());
                    (void)bad.etc_dir();
                  },
                  "path_traversal"),
              "hardlink escape fixture should be rejected with path_traversal");

      require(throws_with(
                  [&]() {
                    ModelPack bad(
                        sima_test::mpk_fixture_path("invalid/truncated_archive.mpk").string());
                    (void)bad.etc_dir();
                  },
                  "invalid_archive"),
              "truncated archive should be rejected with invalid_archive");

      require(throws_with(
                  [&]() {
                    const auto legacy = sima_test::make_mpk_tar_fixture(
                        "modelpack_extract_legacy_missing_mpk",
                        {
                            {"etc/pipeline_sequence.json",
                             R"json({
  "pipelines": [{
    "sequence": [
      {
        "sequence_id": 1,
        "name": "mla_0",
        "pluginId": "processmla",
        "configPath": "0_process_mla.json",
        "processor": "MLA",
        "kernel": "infer",
        "input": "decoder"
      }
    ]
  }]
})json"},
                            {"etc/0_process_mla.json",
                             R"json({
  "node_name": "mla_0",
  "input_buffers": [{"name": "decoder"}]
})json"},
                        });
                    ModelPack bad(legacy.tar_path);
                    (void)bad.etc_dir();
                  },
                  "strict MPK contract required"),
              "legacy fixture without *_mpk.json should fail strict contract check");
    }));
