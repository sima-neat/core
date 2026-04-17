// tutorial_0010_diagnostics_debug.cpp
// Story: diagnostics are structured and reproducible.
// What you learn:
// - validate() returns a SessionReport with repro helpers.
// - bad pipelines still produce useful reports.
// - env vars like SIMA_GST_DOT_DIR can emit DOT graphs.

#include "neat/session.h"
#include "neat/nodes.h"


#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <array>
#include <exception>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

bool has_flag(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; ++i) {
    if (key == argv[i]) return true;
  }
  return false;
}

bool get_arg(int argc, char** argv, const std::string& key, std::string& out) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (key == argv[i]) {
      out = argv[i + 1];
      return true;
    }
  }
  return false;
}

bool wants_help(int argc, char** argv) {
  return has_flag(argc, argv, "--help") || has_flag(argc, argv, "-h");
}

void print_common_flags(std::ostream& os) {
  os << "  --help               Show this help message\n";
  os << "  --print-gst          Print the gst-launch string and exit\n";
}

std::filesystem::path find_repo_root() {
  namespace fs = std::filesystem;
  fs::path cur = fs::current_path();
  for (int i = 0; i < 6; ++i) {
    if (fs::exists(cur / "CMakeLists.txt") && fs::exists(cur / "include") &&
        fs::exists(cur / "tests")) {
      return cur;
    }
    if (!cur.has_parent_path()) break;
    cur = cur.parent_path();
  }
  return fs::current_path();
}

std::filesystem::path find_asset_root() {
  namespace fs = std::filesystem;
  if (const char* env = std::getenv("SIMA_NEAT_TUTORIAL_ASSETS")) {
    fs::path p{env};
    if (fs::exists(p)) return p;
  }
  for (const fs::path& p : {
           fs::path{"/usr/share/sima-neat/tutorials/assets"},
           fs::path{"/neat-resources/core-src/tutorials/assets"},
       }) {
    if (fs::exists(p)) return p;
  }
  return find_repo_root() / "tutorials" / "assets";
}

} // namespace

namespace fs = std::filesystem;

namespace {

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [--image <path>]\n";
  print_common_flags(std::cout);
  std::cout << "  --image <path>       Image path used for the good validation example\n";
}

fs::path default_image_path() {
  return find_asset_root() / "ilena_488.jpg";
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (wants_help(argc, argv)) {
      print_help(argv[0]);
      return 0;
    }

    // Keep validation fast even if a source stalls.
    if (!std::getenv("SIMA_GST_VALIDATE_TIMEOUT_MS")) {
      setenv("SIMA_GST_VALIDATE_TIMEOUT_MS", "1000", 1);
    }

    std::string image_arg;
    const fs::path image_path = get_arg(argc, argv, "--image", image_arg)
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
