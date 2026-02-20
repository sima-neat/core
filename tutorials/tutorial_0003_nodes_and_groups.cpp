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

#include "tutorial_common.h"

#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

namespace {

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [--image <path>] [--width <w>] [--height <h>]\n";
  sima_tutorial::print_common_flags(std::cout);
  std::cout << "  --image <path>       Image file used in the pipeline description\n";
  std::cout << "  --width <w>          Target width (default 256)\n";
  std::cout << "  --height <h>         Target height (default 256)\n";
}

int parse_int_arg(int argc, char** argv, const std::string& key, int def) {
  std::string val;
  if (!sima_tutorial::get_arg(argc, argv, key, val))
    return def;
  try {
    return std::stoi(val);
  } catch (...) {
    return def;
  }
}

fs::path default_image_path(const fs::path& root) {
  const fs::path candidate1 = root / "test.jpg";
  const fs::path candidate2 = root / "tests" / "assets" / "preproc_dynamic" / "ilena_488.jpg";
  if (fs::exists(candidate1))
    return candidate1;
  if (fs::exists(candidate2))
    return candidate2;
  return candidate1;
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
    if (sima_tutorial::wants_help(argc, argv)) {
      print_help(argv[0]);
      return 0;
    }

    const fs::path root = sima_tutorial::find_repo_root();
    std::string image_arg;
    const fs::path image_path = sima_tutorial::get_arg(argc, argv, "--image", image_arg)
                                    ? fs::path(image_arg)
                                    : default_image_path(root);

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

    if (sima_tutorial::wants_print_gst(argc, argv)) {
      std::cout << "--- NodeGroup gst ---\n";
      std::cout << group_session.describe_backend() << "\n";
      std::cout << "--- Manual gst ---\n";
      std::cout << manual_session.describe_backend() << "\n";
      return 0;
    }

    // Deterministic naming: describe_backend() should be stable across calls.
    sima_tutorial::require(group_session.describe_backend() == group_session.describe_backend(),
                           "NodeGroup gst string should be deterministic");
    sima_tutorial::require(manual_session.describe_backend() == manual_session.describe_backend(),
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
