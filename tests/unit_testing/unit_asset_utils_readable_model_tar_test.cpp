#include "asset_utils.h"
#include "test_main.h"
#include "test_utils.h"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>

namespace {

namespace fs = std::filesystem;

class TempDirGuard {
public:
  explicit TempDirGuard(const fs::path& path) : path_(path) {}

  ~TempDirGuard() {
    std::error_code ec;
    const fs::path tmp_tar = path_ / "tmp" / "yolo_v8s_mpk.tar.gz";
    fs::permissions(tmp_tar, fs::perms::owner_all | fs::perms::group_all | fs::perms::others_all,
                    fs::perm_options::replace, ec);
    ec.clear();
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

} // namespace

RUN_TEST("unit_asset_utils_readable_model_tar_test", ([] {
           const fs::path root =
               fs::temp_directory_path() / "sima_unit_asset_utils_readable_model_tar_test";
           TempDirGuard cleanup(root);

           std::error_code ec;
           fs::remove_all(root, ec);
           ec.clear();
           fs::create_directories(root / "tmp", ec);
           require(!ec, std::string("failed to create temp dirs: ") + ec.message());

           const fs::path unreadable_tmp = root / "tmp" / "yolo_v8s_mpk.tar.gz";
           const fs::path alternate = root / "yolov8s_mpk.tar.gz";

           write_file(unreadable_tmp, "stale");
           fs::permissions(unreadable_tmp, fs::perms::none, fs::perm_options::replace, ec);
           require(!ec, std::string("failed to chmod unreadable fixture: ") + ec.message());

           const std::string expected_payload = "fresh";
           write_file(alternate, expected_payload);

           const std::string resolved = sima_test::resolve_yolov8s_tar_local_first(root, true);
           require(resolved == unreadable_tmp.string(),
                   std::string("expected resolver to repopulate tmp tar, got: ") + resolved);
           require(sima_test::is_usable_regular_file(unreadable_tmp),
                   "resolved tmp tar should be readable and non-empty");

           std::ifstream in(unreadable_tmp, std::ios::binary);
           require(static_cast<bool>(in), "expected to reopen resolved tmp tar");
           std::string actual_payload((std::istreambuf_iterator<char>(in)),
                                      std::istreambuf_iterator<char>());
           require(actual_payload == expected_payload,
                   std::string("resolved tmp tar payload mismatch: ") + actual_payload);
         }));
