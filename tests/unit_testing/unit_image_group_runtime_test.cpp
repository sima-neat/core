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
    const bool hardware = argc > 2 && std::string(argv[2]) == "--encoder";
    if (hardware) {
      opt.output_caps.width = 256;
      opt.output_caps.height = 256;
      opt.output_caps.format = "NV12";
      opt.sima_decoder.enable = true;
      opt.sima_decoder.raw_output = true;
    }

    simaai::neat::Graph p;
    p.add(simaai::neat::nodes::groups::ImageInputGroup(opt));
    p.add(simaai::neat::nodes::Output());

    if (hardware) {
      auto run = p.build();
      int count = 0;
      for (;;) {
        simaai::neat::Sample sample;
        simaai::neat::PullError error;
        const auto status = run.pull(5000, sample, &error);
        if (status == simaai::neat::PullStatus::Closed)
          break;
        require(status == simaai::neat::PullStatus::Ok,
                "image encode/decode failed: " + error.message);
        const auto frames = simaai::neat::tensors_from_sample(sample, true);
        require(frames.size() == 1 && frames.front().is_nv12(),
                "image group returned wrong format");
        require(frames.front().width() == 256 && frames.front().height() == 256,
                "image group returned wrong geometry");
        ++count;
      }
      run.stop();
      require(count == 5, "image encode/decode lost finite-source frames");
      std::cout << "[OK] ImageInputGroup hardware encode/decode frames=" << count << "\n";
      return 0;
    }

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
