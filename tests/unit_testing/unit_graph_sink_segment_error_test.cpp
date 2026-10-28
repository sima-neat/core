// A graph branch that ends in its own sink (as VideoSender ends in udpsink) has no Output to
// pull. When an element in that branch fails, the graph must stop with that element's error
// instead of running on with the branch silently dead.
#include "gst/GstInit.h"
#include "nodes/common/Caps.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/EncodedSampleUtil.h"
#include "pipeline/Graph.h"
#include "test_utils.h"

#include <iostream>
#include <optional>
#include <string>
#include <vector>

int main() {
  try {
    using namespace simaai::neat;
    gst_init_once();
    const std::string caps = "image/jpeg,width=(int)16,height=(int)16,framerate=(fraction)30/1";

    InputOptions src_opt;
    src_opt.payload_type = PayloadType::Encoded;
    src_opt.format = FormatTag::ENCODED;
    src_opt.caps_override = caps;
    src_opt.memory_policy = InputMemoryPolicy::SystemMemory;
    Graph source("sink_error_source");
    source.add(nodes::Input(src_opt));
    Graph pulled("sink_error_output");
    pulled.add(nodes::Output());
    Graph failing("sink_error_branch");
    failing.add(nodes::Custom("identity error-after=3 ! fakesink sync=false async=false"));
    Graph split("sink_error_split");
    split.connect(source, pulled);
    split.connect(source, failing);
    Run run = split.build();

    bool stopped = false;
    for (int frame = 0; frame < 200 && !stopped; ++frame) {
      try {
        const Sample sample = make_encoded_sample(std::vector<uint8_t>{0xff, 0xd8, 0xff, 0xd9},
                                                  caps, frame * 33333333LL, -1, 33333333LL);
        if (!run.push(sample) || !run.pull(1000).has_value()) {
          stopped = true;
        }
      } catch (const std::exception&) {
        stopped = true;
      }
    }
    const std::string error = run.last_error();
    std::cout << "[INFO] last_error: " << error << "\n";
    require(stopped, "the graph kept running after its sink-only branch failed");
    require(error.find("identity") != std::string::npos,
            "the graph error should name the failing sink-only branch element");
    run.stop();
  } catch (const std::exception& e) {
    return fail_test(e.what());
  }
  std::cout << "[OK] unit_graph_sink_segment_error_test\n";
  return 0;
}
