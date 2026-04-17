// tutorial_0003_nodes_and_groups.cpp
// Story: Nodes are atomic building blocks; NodeGroups package common recipes.
// What you learn:
// - A Node contributes deterministic gst fragments and element names.
// - NodeGroups are just ordered lists of Nodes.
// - describe() and describe_backend() are stable and reproducible.
//
// This tutorial only prints pipeline descriptions and does not run them.

#include "neat/session.h"
#include "neat/nodes.h"
#include "neat/node_groups.h"


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

bool wants_print_gst(int argc, char** argv) {
  return has_flag(argc, argv, "--print-gst");
}

void print_common_flags(std::ostream& os) {
  os << "  --help               Show this help message\n";
  os << "  --print-gst          Print the gst-launch string and exit\n";
}

void require(bool ok, const std::string& msg) {
  if (!ok) throw std::runtime_error(msg);
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
  std::cout << "Usage: " << argv0 << " [--image <path>] [--width <w>] [--height <h>]\n";
  print_common_flags(std::cout);
  std::cout << "  --image <path>       Image file used in the pipeline description\n";
  std::cout << "  --width <w>          Target width (default 256)\n";
  std::cout << "  --height <h>         Target height (default 256)\n";
}

int parse_int_arg(int argc, char** argv, const std::string& key, int def) {
  std::string val;
  if (!get_arg(argc, argv, key, val))
    return def;
  try {
    return std::stoi(val);
  } catch (...) {
    return def;
  }
}

fs::path default_image_path() {
  return find_asset_root() / "ilena_488.jpg";
}

simaai::neat::Session make_group_session(const std::string& image_path, int w, int h) {
  simaai::neat::Session p;
  simaai::neat::nodes::groups::ImageInputGroupOptions in;
  in.path = image_path;
  in.output_caps.width = w;
  in.output_caps.height = h;
  in.output_caps.format = "RGB";
  p.add(simaai::neat::nodes::groups::ImageInputGroup(in));
  p.add(simaai::neat::nodes::Output());
  return p;
}

simaai::neat::Session make_manual_session(const std::string& image_path, int w, int h) {
  simaai::neat::Session p;
  p.add(simaai::neat::nodes::FileInput(image_path));
  p.add(simaai::neat::nodes::ImageDecode());
  p.add(simaai::neat::nodes::ImageFreeze(1));
  p.add(simaai::neat::nodes::VideoConvert());
  p.add(simaai::neat::nodes::VideoScale());
  p.add(simaai::neat::nodes::CapsRaw("RGB", w, h));
  p.add(simaai::neat::nodes::Output());
  return p;
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
    const fs::path image_path = get_arg(argc, argv, "--image", image_arg)
                                    ? fs::path(image_arg)
                                    : default_image_path();

    const int w = parse_int_arg(argc, argv, "--width", 256);
    const int h = parse_int_arg(argc, argv, "--height", 256);

    // 1) Group-based pipeline (higher-level recipe).
    simaai::neat::Session group_session = make_group_session(image_path.string(), w, h);

    // 2) Manual pipeline (explicit node-by-node path).
    simaai::neat::Session manual_session = make_manual_session(image_path.string(), w, h);

    std::cout << "--- NodeGroup pipeline ---\n";
    std::cout << group_session.describe() << "\n";
    std::cout << "--- Manual pipeline ---\n";
    std::cout << manual_session.describe() << "\n";

    if (wants_print_gst(argc, argv)) {
      std::cout << "--- NodeGroup gst ---\n";
      std::cout << group_session.describe_backend() << "\n";
      std::cout << "--- Manual gst ---\n";
      std::cout << manual_session.describe_backend() << "\n";
      return 0;
    }

    // Deterministic naming: describe_backend() should be stable across calls.
    require(group_session.describe_backend() == group_session.describe_backend(),
                           "NodeGroup gst string should be deterministic");
    require(manual_session.describe_backend() == manual_session.describe_backend(),
                           "Manual gst string should be deterministic");

    if (!fs::exists(image_path)) {
      std::cout << "Note: image path does not exist; this tutorial only prints pipelines.\n";
    }

    std::cout << "[OK] tutorial_0003 complete\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
