#include "asset_utils.h"
#include "test_main.h"
#include "test_utils.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

namespace {

namespace fs = std::filesystem;

class TempDirGuard {
public:
  explicit TempDirGuard(const fs::path& path) : path_(path) {}

  ~TempDirGuard() {
    std::error_code ec;
    fs::remove_all(path_, ec);
  }

private:
  fs::path path_;
};

void write_file(const fs::path& path, const std::string& data) {
  std::error_code ec;
  fs::create_directories(path.parent_path(), ec);
  std::ofstream out(path, std::ios::binary);
  require(static_cast<bool>(out), std::string("failed to open file for write: ") + path.string());
  out << data;
  require(static_cast<bool>(out), std::string("failed to write file: ") + path.string());
}

void set_or_unset(const char* name, const fs::path& value) {
  if (value.empty()) {
    require(::unsetenv(name) == 0, std::string("failed to unset ") + name);
  } else {
    require(::setenv(name, value.c_str(), 1) == 0, std::string("failed to set ") + name);
  }
}

} // namespace

// The YOLOv8s resolver reads SIMA_YOLO_TAR, then SIMA_MODEL_TAR, and nothing else:
// no folder search and no download.
RUN_TEST("unit_asset_utils_readable_model_tar_test", ([] {
           const fs::path root =
               fs::temp_directory_path() / "sima_unit_asset_utils_readable_model_tar_test";
           TempDirGuard cleanup(root);
           std::error_code ec;
           fs::remove_all(root, ec);

           const fs::path yolo = root / "yolov8s_w8a8_mpk.tar.gz";
           const fs::path generic = root / "generic_mpk.tar.gz";
           const fs::path empty = root / "empty_mpk.tar.gz";
           write_file(yolo, "yolo");
           write_file(generic, "generic");
           write_file(empty, "");
           // A file the old resolver would have found by searching the asset root.
           write_file(root / "tmp" / "yolo_v8s_mpk.tar.gz", "stale");

           set_or_unset("SIMA_YOLO_TAR", yolo);
           set_or_unset("SIMA_MODEL_TAR", generic);
           require(sima_test::resolve_yolov8s_tar(root) == yolo.string(),
                   "SIMA_YOLO_TAR should win over SIMA_MODEL_TAR");

           set_or_unset("SIMA_YOLO_TAR", {});
           require(sima_test::resolve_yolov8s_tar(root) == generic.string(),
                   "SIMA_MODEL_TAR should be used when SIMA_YOLO_TAR is unset");

           set_or_unset("SIMA_YOLO_TAR", empty);
           require(sima_test::resolve_yolov8s_tar(root).empty(),
                   "an empty SIMA_YOLO_TAR file should be refused, not replaced");

           set_or_unset("SIMA_YOLO_TAR", root / "missing_mpk.tar.gz");
           require(sima_test::resolve_yolov8s_tar(root).empty(),
                   "a missing SIMA_YOLO_TAR file should be refused, not replaced");

           set_or_unset("SIMA_YOLO_TAR", yolo);
           require(sima_test::resolve_yolov8s_strict_mpk_tar(root).empty(),
                   "a SIMA_YOLO_TAR that is not an MPK should be refused, not replaced");

           set_or_unset("SIMA_YOLO_TAR", {});
           set_or_unset("SIMA_MODEL_TAR", {});
           require(sima_test::resolve_yolov8s_tar(root).empty(),
                   "nothing should be found when both variables are unset");
           require(sima_test::resolve_yolov8s_strict_mpk_tar(root).empty(),
                   "the strict resolver should find nothing when both variables are unset");
           require(sima_test::resolve_yolov8s_tar_local_only(root).empty(),
                   "an unlistable <root>/tmp/yolo_v8s_mpk.tar.gz should not be returned");
         }));
