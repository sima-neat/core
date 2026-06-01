#include "nodes/groups/ImageInputGroup.h"
#include "nodes/groups/VideoInputGroup.h"
#include "nodes/groups/RtspDecodedInput.h"

#include "nodes/common/Caps.h"
#include "nodes/common/FileInput.h"
#include "nodes/common/ImageDecode.h"
#include "nodes/common/ImageFreeze.h"
#include "nodes/common/VideoTrackSelect.h"
#include "nodes/common/Queue.h"
#include "nodes/common/VideoConvert.h"
#include "nodes/common/VideoRate.h"
#include "nodes/common/VideoScale.h"
#include "nodes/io/RTSPInput.h"
#include "nodes/rtp/H264Depacketize.h"
#include "nodes/sima/H264DecodeSima.h"
#include "nodes/sima/H264Parse.h"
#include "pipeline/Graph.h"

#include "test_utils.h"

#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using simaai::neat::Graph;

static_assert(
    std::is_same_v<decltype(simaai::neat::nodes::groups::ImageInputGroup(
                       std::declval<const simaai::neat::nodes::groups::ImageInputGroupOptions&>())),
                   Graph>);
static_assert(
    std::is_same_v<decltype(simaai::neat::nodes::groups::VideoInputGroup(
                       std::declval<const simaai::neat::nodes::groups::VideoInputGroupOptions&>())),
                   Graph>);
static_assert(std::is_same_v<
              decltype(simaai::neat::nodes::groups::RtspDecodedInput(
                  std::declval<const simaai::neat::nodes::groups::RtspDecodedInputOptions&>())),
              Graph>);

Graph graph_from_nodes(std::vector<std::shared_ptr<simaai::neat::Node>> nodes) {
  Graph graph;
  for (auto& node : nodes) {
    graph.add(std::move(node));
  }
  return graph;
}

void compare_graph_fragments(const Graph& actual, const Graph& expected) {
  const std::string actual_text = actual.describe();
  const std::string expected_text = expected.describe();
  require(actual_text == expected_text,
          "Graph fragment mismatch\nactual:\n" + actual_text + "\nexpected:\n" + expected_text);
}

} // namespace

int main(int argc, char** argv) {
  try {
    std::error_code ec;
    const auto cwd = std::filesystem::current_path(ec);
    const std::string argv0 = (argc > 0 && argv && argv[0]) ? argv[0] : "<unknown>";
    std::error_code abs_ec;
    const auto abs_path = std::filesystem::absolute(argv0, abs_ec);
    std::cout << "[INFO] unit_groups_test argv0=" << argv0 << "\n";
    std::cout << "[INFO] unit_groups_test exe_abs="
              << (abs_ec ? std::string("<error>") : abs_path.string()) << "\n";
    std::cout << "[INFO] unit_groups_test cwd=" << (ec ? std::string("<error>") : cwd.string())
              << "\n";
    // ----------------------------
    // Image group (auto decode)
    // ----------------------------
    simaai::neat::nodes::groups::ImageInputGroupOptions io;
    io.path = "test.jpg";
    io.imagefreeze_num_buffers = 5;
    io.fps = 30;
    io.use_videorate = true;
    io.use_videoscale = true;
    io.output_caps.width = 64;
    io.output_caps.height = 64;

    auto group_img = simaai::neat::nodes::groups::ImageInputGroup(io);

    std::vector<std::shared_ptr<simaai::neat::Node>> manual_img;
    manual_img.push_back(simaai::neat::nodes::FileInput(io.path));
    manual_img.push_back(simaai::neat::nodes::ImageDecode());
    manual_img.push_back(simaai::neat::nodes::ImageFreeze(io.imagefreeze_num_buffers));
    if (io.use_videorate)
      manual_img.push_back(simaai::neat::nodes::VideoRate());
    if (io.use_videoconvert)
      manual_img.push_back(simaai::neat::nodes::VideoConvert());
    if (io.use_videoscale)
      manual_img.push_back(simaai::neat::nodes::VideoScale());
    int img_fps = io.output_caps.fps > 0 ? io.output_caps.fps : io.fps;
    manual_img.push_back(simaai::neat::nodes::CapsRaw(io.output_caps.format, io.output_caps.width,
                                                      io.output_caps.height, img_fps,
                                                      io.output_caps.memory));

    compare_graph_fragments(group_img, graph_from_nodes(std::move(manual_img)));

    // ----------------------------
    // Video group
    // ----------------------------
    simaai::neat::nodes::groups::VideoInputGroupOptions vo;
    vo.path = "video.mp4";
    vo.demux_video_pad_index = 0;
    vo.insert_queue = true;
    vo.parse_config_interval = 1;
    vo.parse_enforce_au = true;
    vo.output_caps.enable = false;

    auto group_vid = simaai::neat::nodes::groups::VideoInputGroup(vo);

    std::vector<std::shared_ptr<simaai::neat::Node>> manual_vid;
    manual_vid.push_back(simaai::neat::nodes::FileInput(vo.path));
    manual_vid.push_back(simaai::neat::nodes::VideoTrackSelect(vo.demux_video_pad_index));
    manual_vid.push_back(simaai::neat::nodes::Queue());
    manual_vid.push_back(simaai::neat::nodes::H264ParseAu(vo.parse_config_interval));
    manual_vid.push_back(simaai::neat::nodes::Queue());
    manual_vid.push_back(simaai::neat::nodes::H264Decode(vo.sima_allocator_type, vo.out_format));

    compare_graph_fragments(group_vid, graph_from_nodes(std::move(manual_vid)));

    // ----------------------------
    // RTSP group
    // ----------------------------
    simaai::neat::nodes::groups::RtspDecodedInputOptions ro;
    ro.url = "rtsp://example";
    ro.latency_ms = 123;
    ro.tcp = true;
    ro.payload_type = 97;
    ro.h264_fps = 30;
    ro.h264_width = 640;
    ro.h264_height = 480;
    ro.insert_queue = true;

    auto group_rtsp = simaai::neat::nodes::groups::RtspDecodedInput(ro);

    std::vector<std::shared_ptr<simaai::neat::Node>> manual_rtsp;
    manual_rtsp.push_back(simaai::neat::nodes::RTSPInput(ro.url, ro.latency_ms, ro.tcp));
    manual_rtsp.push_back(simaai::neat::nodes::Queue());
    manual_rtsp.push_back(
        simaai::neat::nodes::H264Depacketize(ro.payload_type, ro.h264_parse_config_interval,
                                             ro.h264_fps, ro.h264_width, ro.h264_height));
    manual_rtsp.push_back(simaai::neat::nodes::Queue());
    manual_rtsp.push_back(simaai::neat::nodes::H264Decode(
        ro.sima_allocator_type, ro.out_format, ro.decoder_name, ro.decoder_raw_output,
        ro.decoder_next_element, ro.h264_width, ro.h264_height, ro.h264_fps));

    compare_graph_fragments(group_rtsp, graph_from_nodes(std::move(manual_rtsp)));

    std::cout << "[OK] unit_groups_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
