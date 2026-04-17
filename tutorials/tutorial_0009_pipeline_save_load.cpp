// tutorial_0009_pipeline_save_load.cpp
// Story: save a pipeline to JSON and load it back.
// What you learn:
// - Session::save() writes a reproducible node list.
// - Session::load() rehydrates a session.
// - describe_backend() parity is a quick sanity check.

#include "neat/session.h"
#include "neat/nodes.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <array>
#include <cstdlib>
#include <exception>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

bool has_flag(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; ++i) {
    if (key == argv[i])
      return true;
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

bool wants_print_gst(int argc, char** argv) {
  return has_flag(argc, argv, "--print-gst");
}

void print_common_flags(std::ostream& os) {
  os << "  --help               Show this help message\n";
  os << "  --print-gst          Print the gst-launch string and exit\n";
}

void require(bool ok, const std::string& msg) {
  if (!ok)
    throw std::runtime_error(msg);
}

std::filesystem::path find_repo_root() {
  namespace fs = std::filesystem;
  fs::path cur = fs::current_path();
  for (int i = 0; i < 6; ++i) {
    if (fs::exists(cur / "CMakeLists.txt") && fs::exists(cur / "include") &&
        fs::exists(cur / "tests")) {
      return cur;
    }
    if (!cur.has_parent_path())
      break;
    cur = cur.parent_path();
  }
  return fs::current_path();
}

std::filesystem::path find_asset_root() {
  namespace fs = std::filesystem;
  if (const char* env = std::getenv("SIMA_NEAT_TUTORIAL_ASSETS")) {
    fs::path p{env};
    if (fs::exists(p))
      return p;
  }
  for (const fs::path& p : {
           fs::path{"/usr/share/sima-neat/tutorials/assets"},
           fs::path{"/neat-resources/core-src/tutorials/assets"},
       }) {
    if (fs::exists(p))
      return p;
  }
  return find_repo_root() / "tutorials" / "assets";
}

} // namespace

namespace fs = std::filesystem;

namespace {

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [--image <path>] [--out <path>]\n";
  print_common_flags(std::cout);
  std::cout
      << "  --image <path>       Image path used in FileInput (default: shipped tutorial sample)\n";
  std::cout
      << "  --out <path>         Output JSON path (default: tmp/tutorial_0009_pipeline.json)\n";
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

    const fs::path root = find_repo_root();
    std::string image_arg;
    fs::path image_path =
        get_arg(argc, argv, "--image", image_arg) ? fs::path(image_arg) : default_image_path();

    std::string out_arg;
    const fs::path out_path = get_arg(argc, argv, "--out", out_arg)
                                  ? fs::path(out_arg)
                                  : (root / "tmp" / "tutorial_0009_pipeline.json");
    if (!out_path.parent_path().empty()) {
      std::error_code mkdir_ec;
      fs::create_directories(out_path.parent_path(), mkdir_ec);
      require(!mkdir_ec, "failed to create output directory: " + out_path.parent_path().string());
    }

    simaai::neat::Session p;
    p.add(simaai::neat::nodes::FileInput(image_path.string()));
    p.add(simaai::neat::nodes::ImageDecode());
    p.add(simaai::neat::nodes::ImageFreeze(1));
    p.add(simaai::neat::nodes::Output());

    if (wants_print_gst(argc, argv)) {
      std::cout << p.describe_backend() << "\n";
      return 0;
    }

    p.save(out_path.string());
    simaai::neat::Session loaded = simaai::neat::Session::load(out_path.string());

    const std::string original = p.describe_backend(false);
    const std::string restored = loaded.describe_backend(false);
    require(original == restored, "gst string mismatch after load");

    std::cout << "Saved pipeline to: " << out_path.string() << "\n";
    std::cout << "[OK] tutorial_0009 complete\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
