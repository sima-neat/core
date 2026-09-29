#include "genai/ImageStage.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace fs = std::filesystem;
using simaai::neat::pcie::genai::internal::StagedImage;

namespace {
void require(bool c, const std::string& m) {
  if (!c) throw std::runtime_error(m);
}

fs::path make_temp_dir() {
  fs::path d = fs::temp_directory_path() /
               ("image_stage_test_" + std::to_string(::getpid()) + "_" +
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::create_directories(d);
  return d;
}

void write_file(const fs::path& p, const std::string& text) {
  fs::create_directories(p.parent_path());
  std::ofstream(p) << text;
}
}  // namespace

int main() {
  try {
    const fs::path root = make_temp_dir();
    const fs::path stage = root / "pcie-genai";
    const fs::path src = root / "photo.jpg";
    write_file(src, "JPEGDATA");

    // copy made, extension kept, name relative to the serve root
    {
      StagedImage s(stage, "h9-1", src);
      require(s.relative_name() == "pcie-genai/h9-1.jpg", "relative name");
      require(fs::exists(s.staged_path()), "staged copy exists");
      require(fs::exists(stage / "h9-1.jpg"), "copy lands in the stage dir");
    }
    require(!fs::exists(stage / "h9-1.jpg"), "copy deleted on scope exit");

    // missing source throws
    {
      bool threw = false;
      try {
        StagedImage s(stage, "h9-2", root / "does-not-exist.png");
      } catch (const std::runtime_error&) {
        threw = true;
      }
      require(threw, "missing source throws");
    }

    // 1-hour sweep: an old file goes, a fresh one stays
    {
      write_file(stage / "old.jpg", "x");
      const auto two_hours_ago = fs::file_time_type::clock::now() - std::chrono::hours(2);
      fs::last_write_time(stage / "old.jpg", two_hours_ago);
      write_file(stage / "young.jpg", "y");
      StagedImage s(stage, "h9-3", src);
      require(!fs::exists(stage / "old.jpg"), "old file swept");
      require(fs::exists(stage / "young.jpg"), "young file kept");
    }

    fs::remove_all(root);
    std::cout << "[PASS] pcie-genai image stage\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
