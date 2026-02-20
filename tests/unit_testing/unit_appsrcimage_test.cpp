#include "nodes/io/StillImageInput.h"

#include "test_utils.h"

#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
  try {
    require(argc >= 2, "unit_appsrcimage_test requires an image path");

    std::string path = argv[1];
    require(file_exists(path), "unit_appsrcimage_test image not found");
    simaai::neat::StillImageInput node(path, 64, 64, 64, 64, 30);

    require(node.nv12_enc() && !node.nv12_enc()->empty(), "nv12_enc missing");
    auto frag = node.backend_fragment(0);
    require_contains(frag, "appsrc name=mysrc", "StillImageInput fragment missing appsrc");
    require_contains(frag, "width=64", "StillImageInput fragment missing width");
    require_contains(frag, "height=64", "StillImageInput fragment missing height");

    std::cout << "[OK] unit_appsrcimage_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
