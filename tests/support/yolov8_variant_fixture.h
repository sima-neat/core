#pragma once

#include "asset_utils.h"
#include "model_archive_fixture_utils.h"
#include "test_main.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

namespace sima_test {

struct Yolov8VariantFixture {
  std::string tar_path;
  std::string unpack_root;
};

// Legacy YOLOv8 packages rebuilt from their committed MPK manifests. Each manifest is the
// one the legacy archive shipped; each MLA executable is replaced by an ELF that carries
// only the compiler-named I/O sections and their sizes, which is all the contract tests
// read from it. Nothing is downloaded.
inline Yolov8VariantFixture resolve_yolov8_variant_fixture(const std::string& stem) {
  namespace fs = std::filesystem;
  const fs::path seed = test_model_archive_fixture_root_path() / "strict-seeds" / "legacy" / stem;
  std::error_code ec;
  require(fs::is_directory(seed, ec) && !ec,
          "legacy YOLOv8 fixture metadata is missing: " + seed.string());

  // Each call gets its own directory, which holds the package root and the archive. Tests
  // run in parallel, and a shared archive path could be read while another test writes it.
  const fs::path base(make_fixture_temp_dir("legacy_" + stem));
  const fs::path root_path = base / "package";
  fs::create_directories(root_path);
  std::string manifest_text;
  for (const auto& entry : fs::recursive_directory_iterator(seed)) {
    if (!entry.is_regular_file())
      continue;
    const fs::path relative = entry.path().lexically_relative(seed);
    const fs::path target = root_path / relative;
    fs::create_directories(target.parent_path());
    fs::copy_file(entry.path(), target, fs::copy_options::overwrite_existing);
    if (entry.path().filename().string().ends_with("_mpk.json")) {
      require(manifest_text.empty(), "legacy YOLOv8 fixture has more than one MPK manifest");
      std::ifstream input(entry.path(), std::ios::binary);
      manifest_text.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    }
  }
  require(!manifest_text.empty(), "legacy YOLOv8 fixture has no MPK manifest: " + seed.string());
  // The legacy archives are flat, with the executable beside the manifest. Keep that
  // layout for both the archive and the unpacked package root.
  write_mpk_topology_elfs(root_path, manifest_text);
  const fs::path share = root_path / "share";
  for (const auto& entry : fs::directory_iterator(share))
    fs::rename(entry.path(), root_path / entry.path().filename());
  fs::remove(share);

  const fs::path tar_path = base / ("legacy_" + stem + ".tar.gz");
  const std::string command = "tar -czf " + shell_quote(tar_path.string()) + " -C " +
                              shell_quote(root_path.string()) + " .";
  require(std::system(command.c_str()) == 0,
          "failed to archive legacy YOLOv8 fixture " + tar_path.string());
  return {tar_path.string(), root_path.string()};
}

} // namespace sima_test
