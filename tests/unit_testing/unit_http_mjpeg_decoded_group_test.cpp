#include "nodes/groups/HttpMjpegDecodedInput.h"
#include "nodes/groups/GroupOutputSpec.h"

#include "pipeline/Graph.h"

#include "test_utils.h"

#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {

using simaai::neat::Graph;

static_assert(
    std::is_same_v<
        decltype(simaai::neat::nodes::groups::HttpMjpegDecodedInput(
            std::declval<const simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions&>())),
        Graph>);

} // namespace

int main() {
  try {
    simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions opt;
    opt.url = "http://example.local/mjpeg";
    opt.timeout_seconds = 9;
    opt.retries = -1;
    opt.user_agent = "NeatTest";
    opt.multipart_boundary = "frame";
    opt.multipart_single_stream = true;
    opt.decoder_name = "mjpeg_decoder";
    opt.dec_width = 640;
    opt.dec_height = 480;
    opt.dec_fps = 30;
    opt.num_buffers = 8;

    const Graph group = simaai::neat::nodes::groups::HttpMjpegDecodedInput(opt);
    (void)group;

    const simaai::neat::OutputSpec default_spec =
        simaai::neat::nodes::groups::HttpMjpegDecodedInputOutputSpec(opt);
    require(default_spec.media_type == "video/x-raw",
            "HTTP MJPEG group default output media type mismatch");
    require(default_spec.format == "NV12", "HTTP MJPEG group default output format mismatch");
    require(default_spec.width == 640 && default_spec.height == 480,
            "HTTP MJPEG group default output shape mismatch");
    require(default_spec.fps_num == 30 && default_spec.fps_den == 1,
            "HTTP MJPEG group default output fps mismatch");
    require(default_spec.memory == "SimaAI",
            "HTTP MJPEG group default raw decoder output should advertise SimaAI memory");

    opt.insert_queue = false;
    const Graph no_queue_group = simaai::neat::nodes::groups::HttpMjpegDecodedInput(opt);
    (void)no_queue_group;

    opt.insert_queue = true;
    opt.decoder_raw_output = false;
    opt.use_videoconvert = true;
    opt.use_videoscale = true;
    opt.output_caps.enable = true;
    opt.output_caps.width = 320;
    opt.output_caps.height = 240;
    opt.output_caps.fps = 15;
    const Graph tail_group = simaai::neat::nodes::groups::HttpMjpegDecodedInput(opt);
    (void)tail_group;

    const simaai::neat::OutputSpec tail_spec =
        simaai::neat::nodes::groups::HttpMjpegDecodedInputOutputSpec(opt);
    require(tail_spec.width == 320 && tail_spec.height == 240,
            "HTTP MJPEG group output caps shape mismatch");
    require(tail_spec.fps_num == 15 && tail_spec.fps_den == 1,
            "HTTP MJPEG group output caps fps mismatch");
    require(tail_spec.memory == "SystemMemory",
            "HTTP MJPEG group output caps should advertise requested memory");

    std::cout << "[OK] unit_http_mjpeg_decoded_group_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
