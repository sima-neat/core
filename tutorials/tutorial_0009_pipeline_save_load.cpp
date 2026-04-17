// tutorial_0009_pipeline_save_load.cpp
// Story: save a pipeline to JSON and load it back.
// What you learn:
// - Session::save() writes a reproducible node list.
// - Session::load() rehydrates a session.
// - describe_backend() parity is a quick sanity check.

#include "neat/session.h"
#include "neat/nodes.h"

#include "tutorial_common.h"

#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

namespace {

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [--image <path>] [--out <path>]\n";
  sima_tutorial::print_common_flags(std::cout);
  std::cout
      << "  --image <path>       Image path used in FileInput (default: shipped tutorial sample)\n";
  std::cout
      << "  --out <path>         Output JSON path (default: tmp/tutorial_0009_pipeline.json)\n";
}

fs::path default_image_path() {
  return sima_tutorial::find_asset_root() / "ilena_488.jpg";
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (sima_tutorial::wants_help(argc, argv)) {
      print_help(argv[0]);
      return 0;
    }

    const fs::path root = sima_tutorial::find_repo_root();
    std::string image_arg;
    fs::path image_path = sima_tutorial::get_arg(argc, argv, "--image", image_arg)
                              ? fs::path(image_arg)
                              : default_image_path();

    std::string out_arg;
    const fs::path out_path = sima_tutorial::get_arg(argc, argv, "--out", out_arg)
                                  ? fs::path(out_arg)
                                  : (root / "tmp" / "tutorial_0009_pipeline.json");
    if (!out_path.parent_path().empty()) {
      std::error_code mkdir_ec;
      fs::create_directories(out_path.parent_path(), mkdir_ec);
      sima_tutorial::require(!mkdir_ec, "failed to create output directory: " +
                                            out_path.parent_path().string());
    }

    simaai::neat::Session p;
    p.add(simaai::neat::nodes::FileInput(image_path.string()));
    p.add(simaai::neat::nodes::ImageDecode());
    p.add(simaai::neat::nodes::ImageFreeze(1));
    p.add(simaai::neat::nodes::Output());

    if (sima_tutorial::wants_print_gst(argc, argv)) {
      std::cout << p.describe_backend() << "\n";
      return 0;
    }

    p.save(out_path.string());
    simaai::neat::Session loaded = simaai::neat::Session::load(out_path.string());

    const std::string original = p.describe_backend(false);
    const std::string restored = loaded.describe_backend(false);
    sima_tutorial::require(original == restored, "gst string mismatch after load");

    std::cout << "Saved pipeline to: " << out_path.string() << "\n";
    std::cout << "[OK] tutorial_0009 complete\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
