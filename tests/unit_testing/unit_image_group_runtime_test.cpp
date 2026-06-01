#include "pipeline/Graph.h"
#include "nodes/groups/ImageInputGroup.h"
#include "nodes/common/Output.h"

#include "test_utils.h"

#include <gst/gst.h>

#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
  try {
    require(argc >= 2, "unit_image_group_runtime_test requires an image path");
    require(file_exists(argv[1]), "unit_image_group_runtime_test image not found");

    simaai::neat::nodes::groups::ImageInputGroupOptions opt;
    opt.path = argv[1];
    opt.imagefreeze_num_buffers = 5;
    opt.fps = 30;
    opt.use_videoscale = true;
    opt.output_caps.width = 64;
    opt.output_caps.height = 64;

    simaai::neat::Graph p;
    p.add(simaai::neat::nodes::groups::ImageInputGroup(opt));
    p.add(simaai::neat::nodes::Output());

    bool got = false;
    p.set_tensor_callback([&](const simaai::neat::Tensor&) {
      got = true;
      return false;
    });
    p.run();
    require(got, "no frame from image group runtime test");
    std::cout << "[OK] unit_image_group_runtime_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
