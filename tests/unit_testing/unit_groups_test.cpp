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
#include "nodes/sima/SimaDecode.h"
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
    simaai::neat::SimaDecodeOptions vid_dec;
    vid_dec.type = simaai::neat::SimaDecodeType::H264;
    vid_dec.sima_allocator_type = vo.sima_allocator_type;
    vid_dec.out_format = vo.out_format;
    vid_dec.raw_output = true;
    manual_vid.push_back(simaai::neat::nodes::SimaDecode(vid_dec));

    compare_graph_fragments(group_vid, graph_from_nodes(manual_vid));
    require(vo.output_caps.memory == simaai::neat::CapsMemory::Any,
            "native video tail caps must preserve producer memory by default");

    auto converted_vo = vo;
    converted_vo.use_videoconvert = true;
    converted_vo.use_videoscale = true;
    auto converted_nodes = manual_vid;
    converted_nodes.push_back(simaai::neat::nodes::VideoConvert());
    converted_nodes.push_back(simaai::neat::nodes::VideoScale());
    compare_graph_fragments(simaai::neat::nodes::groups::VideoInputGroup(converted_vo),
                            graph_from_nodes(std::move(converted_nodes)));

    auto rgb_vo = vo;
    rgb_vo.out_format = simaai::neat::FormatTag::RGB;
    auto rgb_nodes = manual_vid;
    vid_dec.out_format = rgb_vo.out_format;
    vid_dec.raw_output = false;
    rgb_nodes.back() = simaai::neat::nodes::SimaDecode(vid_dec);
    compare_graph_fragments(simaai::neat::nodes::groups::VideoInputGroup(rgb_vo),
                            graph_from_nodes(std::move(rgb_nodes)));

    auto system_memory_vo = vo;
    system_memory_vo.output_caps.enable = true;
    system_memory_vo.output_caps.memory = simaai::neat::CapsMemory::SystemMemory;
    auto system_memory_nodes = manual_vid;
    simaai::neat::SimaDecodeOptions system_memory_dec;
    system_memory_dec.type = simaai::neat::SimaDecodeType::H264;
    system_memory_dec.sima_allocator_type = system_memory_vo.sima_allocator_type;
    system_memory_dec.out_format = system_memory_vo.out_format;
    system_memory_dec.raw_output = false;
    system_memory_nodes.back() = simaai::neat::nodes::SimaDecode(system_memory_dec);
    const auto& system_caps = system_memory_vo.output_caps;
    system_memory_nodes.push_back(
        simaai::neat::nodes::CapsRaw(system_caps.format, system_caps.width, system_caps.height,
                                     system_caps.fps, system_caps.memory));
    compare_graph_fragments(simaai::neat::nodes::groups::VideoInputGroup(system_memory_vo),
                            graph_from_nodes(std::move(system_memory_nodes)));

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
    simaai::neat::SimaDecodeOptions rtsp_dec;
    rtsp_dec.type = simaai::neat::SimaDecodeType::H264;
    rtsp_dec.sima_allocator_type = ro.sima_allocator_type;
    rtsp_dec.out_format = ro.out_format;
    rtsp_dec.decoder_name = ro.decoder_name;
    rtsp_dec.raw_output = ro.decoder_raw_output;
    rtsp_dec.next_element = ro.decoder_next_element;
    rtsp_dec.dec_width = ro.h264_width;
    rtsp_dec.dec_height = ro.h264_height;
    rtsp_dec.dec_fps = ro.h264_fps;
    manual_rtsp.push_back(simaai::neat::nodes::SimaDecode(rtsp_dec));

    compare_graph_fragments(group_rtsp, graph_from_nodes(std::move(manual_rtsp)));

    std::cout << "[OK] unit_groups_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
