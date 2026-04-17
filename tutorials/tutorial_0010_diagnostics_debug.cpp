// tutorial_0010_diagnostics_debug.cpp
// Story: diagnostics are structured and reproducible.
// What you learn:
// - validate() returns a SessionReport with repro helpers.
// - bad pipelines still produce useful reports.
// - env vars like SIMA_GST_DOT_DIR can emit DOT graphs.

#include "neat/session.h"
#include "neat/nodes.h"

#include "tutorial_common.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

namespace {

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [--image <path>]\n";
  sima_tutorial::print_common_flags(std::cout);
  std::cout << "  --image <path>       Image path used for the good validation example\n";
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

    // Keep validation fast even if a source stalls.
    if (!std::getenv("SIMA_GST_VALIDATE_TIMEOUT_MS")) {
      setenv("SIMA_GST_VALIDATE_TIMEOUT_MS", "1000", 1);
    }

    std::string image_arg;
    const fs::path image_path = sima_tutorial::get_arg(argc, argv, "--image", image_arg)
                                    ? fs::path(image_arg)
                                    : default_image_path();

    // 1) A "good" pipeline: validate and print the repro note.
    if (fs::exists(image_path)) {
      simaai::neat::Session good;
      good.add(simaai::neat::nodes::FileInput(image_path.string()));
      good.add(simaai::neat::nodes::ImageDecode());
      good.add(simaai::neat::nodes::ImageFreeze(1));
      good.add(simaai::neat::nodes::Output());

      simaai::neat::SessionReport rep = good.validate();
      std::cout << "Good pipeline validate():\n";
      std::cout << rep.repro_note << "\n";
    } else {
      std::cout << "Skipping good pipeline validation (image not found).\n";
    }

    // 2) A "bad" pipeline: raw fragment with a missing element.
    simaai::neat::Session bad;
    bad.add(simaai::neat::nodes::Custom("no_such_element"));
    bad.add(simaai::neat::nodes::Output());

    simaai::neat::SessionReport bad_rep = bad.validate();
    std::cout << "Bad pipeline validate():\n";
    std::cout << bad_rep.repro_note << "\n";
    if (!bad_rep.repro_gst_launch.empty()) {
      std::cout << "Repro: " << bad_rep.repro_gst_launch << "\n";
    }

    std::cout << "[OK] tutorial_0010 complete\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
