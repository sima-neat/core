#include "model/internal/ModelPack.h"
#include "mpk_test_utils.h"
#include "test_main.h"
#include "test_utils.h"

#include <filesystem>
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

      const fs::path basic = sima_test::mpk_fixture_path("valid/basic_valid.mpk");
      require(
          fs::exists(basic),
          "missing MPK fixtures; expected tests/assets/mpk (run tests/tools/make_mpk_fixtures.py)");

      const std::string extract_root = sima_test::make_temp_dir("modelpack_extract_root");
      sima_test::ScopedEnvVar env_root("SIMA_MPK_EXTRACT_ROOT", extract_root);

      ModelPack first(basic.string());
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
      require(has_ext(lib_dir, ".so"), "ModelPack extraction lib dir should contain .so artifacts");
      require(has_ext(share_dir, ".elf"),
              "ModelPack extraction share dir should contain .elf artifacts");

      const fs::path canonical_root = fs::weakly_canonical(fs::path(extract_root));
      const fs::path canonical_package = fs::weakly_canonical(package_root);
      require(canonical_package.string().find(canonical_root.string()) == 0,
              "ModelPack extraction must stay inside SIMA_MPK_EXTRACT_ROOT");

      ModelPack second(basic.string());
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
    }));
