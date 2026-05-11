#include "pipeline/Session.h"

#include "test_utils.h"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>

using namespace simaai::neat::nodes;

int main() {
  try {
    // Explicit one-buffer policy: latest frame, no clock sync.
    {
      simaai::neat::Session p;
      p.custom("videotestsrc num-buffers=1", simaai::neat::InputRole::Source);
      p.add(VideoConvert());
      p.add(CapsNV12SysMem(64, 64, 30));
      simaai::neat::OutputOptions opt;
      opt.max_buffers = 1;
      p.add(Output(opt));

      const std::string gst = p.describe_backend();
      require_contains(gst, "appsink name=mysink", "default Output name mismatch");
      require_contains(gst, "emit-signals=false sync=false max-buffers=1 drop=false",
                       "explicit one-buffer Output settings mismatch");
    }

    // Latest/drop: only the most recent buffer should remain.
    {
      simaai::neat::Session p;
      p.custom("videotestsrc num-buffers=5", simaai::neat::InputRole::Source);
      p.add(VideoConvert());
      p.add(CapsNV12SysMem(64, 64, 30));
      simaai::neat::OutputOptions opt;
      opt.max_buffers = 1;
      opt.drop = true;
      p.add(Output(opt));

      int got = 0;
      p.set_tensor_callback([&](const simaai::neat::Tensor&) {
        ++got;
        return got < 1;
      });
      p.run();
      require(got == 1, "latest/drop: expected exactly one frame");
    }

    // Every-frame: keep all buffers (bounded).
    {
      simaai::neat::Session p;
      p.custom("videotestsrc num-buffers=5", simaai::neat::InputRole::Source);
      p.add(VideoConvert());
      p.add(CapsNV12SysMem(64, 64, 30));
      p.add(Output(simaai::neat::OutputOptions::EveryFrame(5)));

      int got = 0;
      p.set_tensor_callback([&](const simaai::neat::Tensor&) {
        ++got;
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        return true;
      });
      p.run();
      require(got == 5, "every-frame: expected exactly 5 frames");
    }

    // add_output_tensor uses default Output() sink policy.
    {
      simaai::neat::Session p;
      p.custom("videotestsrc num-buffers=1", simaai::neat::InputRole::Source);

      simaai::neat::OutputTensorOptions out;
      out.target_width = 64;
      out.target_height = 64;
      p.add_output_tensor(out);

      const std::string gst = p.describe_backend();
      require_contains(gst, "max-buffers=4", "add_output_tensor should use default sink options");
      require_contains(gst, "drop=false", "add_output_tensor should keep default drop=false");
    }

    // Clocked: verify sync=true in the pipeline string.
    {
      simaai::neat::Session p;
      p.custom("videotestsrc num-buffers=1", simaai::neat::InputRole::Source);
      p.add(VideoConvert());
      p.add(CapsNV12SysMem(64, 64, 30));
      p.add(Output(simaai::neat::OutputOptions::Clocked()));

      const std::string gst = p.describe_backend();
      require_contains(gst, "sync=true", "clocked Output should set sync=true");
    }

    std::cout << "[OK] output_appsink_policy_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
