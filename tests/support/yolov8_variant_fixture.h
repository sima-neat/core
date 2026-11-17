#pragma once

#include "asset_utils.h"
#include "test_main.h"

#include <cstdlib>
#include <filesystem>
#include <string>

namespace sima_test {

struct Yolov8VariantFixture {
  std::string tar_path;
  std::string unpack_root;
};

// Both model-contract tests must provision their own required BF16 fixture;
// neither may depend on test execution order or silently skip a missing model.
inline Yolov8VariantFixture resolve_yolov8_variant_fixture(const std::string& stem) {
  namespace fs = std::filesystem;
  const fs::path root = test_source_root();
  const fs::path drive_dir = root / "tmp" / "yolov8n_drive";
  const fs::path unpack_dir = root / "tmp" / "yolov8n_unpack" / stem;
  const fs::path tar_path = drive_dir / (stem + ".tar.gz");
  std::error_code ec;
  fs::create_directories(drive_dir, ec);
  ec.clear();
  fs::create_directories(unpack_dir.parent_path(), ec);
  const ScopedFileLock lock(drive_dir / ".download.lock");
  if (!is_listable_tar_gz(tar_path)) {
    purge_unlistable_tar_gz(tar_path);
    std::string base = "https://docs.sima.ai/pkg_downloads/SDK2.0.0/models/modalix";
    if (const char* env = std::getenv("SIMA_YOLOV8N_VARIANTS_BASE_URL"); env && *env)
      base = env;
    else if (const char* env = std::getenv("SIMA_YOLOV8_VARIANTS_BASE_URL"); env && *env)
      base = env;
    while (!base.empty() && base.back() == '/')
      base.pop_back();
    const auto url = base + "/" + stem + ".tar.gz";
    require(download_file(url, tar_path),
            "failed to download YOLOv8n fixture from " + url +
                " (run `sima-cli login` or set SIMA_YOLOV8N_VARIANTS_BASE_URL to a mirror)");
    require(is_listable_tar_gz(tar_path),
            "YOLOv8n fixture is not a readable tar.gz: " + tar_path.string());
  }
  const auto manifest = unpack_dir / "yolov8n_modified_mpk.json";
  if (!is_usable_regular_file(manifest)) {
    fs::remove_all(unpack_dir, ec);
    ec.clear();
    fs::create_directories(unpack_dir, ec);
    const auto command =
        "tar -xzf " + shell_quote(tar_path.string()) + " -C " + shell_quote(unpack_dir.string());
    require(std::system(command.c_str()) == 0,
            "failed to unpack YOLOv8n fixture " + tar_path.string());
    require(is_usable_regular_file(manifest),
            "YOLOv8n fixture is missing its authoritative MPK manifest");
  }
  return {tar_path.string(), unpack_dir.string()};
}

} // namespace sima_test
