#include "pipeline/Graph.h"

#include "test_utils.h"

#include <gst/gst.h>

#include <iostream>
#include <stdexcept>

using namespace simaai::neat::nodes;

int main() {
  try {
    simaai::neat::Graph p;
    p.custom("videotestsrc num-buffers=1", simaai::neat::InputRole::Source);
    p.add(VideoConvert());
    p.add(CapsNV12SysMem(64, 64, 30));
    p.add(Output());

    simaai::neat::Tensor ref{};
    bool got = false;
    p.set_tensor_callback([&](const simaai::neat::Tensor& f) {
      ref = f;
      got = true;
      return false;
    });
    p.run();

    require(got, "no frame received");
    require(ref.shape.size() >= 2, "shape missing");
    require(ref.shape[0] == 64, "height mismatch");
    require(ref.shape[1] == 64, "width mismatch");
    require(ref.semantic.image.has_value(), "image semantic missing");

    std::cout << "[OK] unit_framestream_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
