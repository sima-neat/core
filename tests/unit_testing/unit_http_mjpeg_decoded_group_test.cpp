#include "nodes/groups/HttpMjpegDecodedInput.h"
#include "nodes/groups/GroupOutputSpec.h"

#include "gst/GstHelpers.h"
#include "nodes/common/Caps.h"
#include "nodes/common/JpegParse.h"
#include "nodes/common/MultipartJpegDemux.h"
#include "nodes/common/Queue.h"
#include "nodes/common/VideoConvert.h"
#include "nodes/common/VideoScale.h"
#include "nodes/io/HttpSource.h"
#include "nodes/sima/SimaDecode.h"
#include "pipeline/Graph.h"
#include "pipeline/GraphReport.h"

#include "test_utils.h"

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
    std::is_same_v<
        decltype(simaai::neat::nodes::groups::HttpMjpegDecodedInput(
            std::declval<const simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions&>())),
        Graph>);

Graph graph_from_nodes(std::vector<std::shared_ptr<simaai::neat::Node>> nodes) {
  Graph graph;
  for (auto& node : nodes) {
    graph.add(std::move(node));
  }
  return graph;
}

void compare_graph_fragments(const Graph& actual, const Graph& expected, const std::string& label) {
  const std::string actual_text = actual.describe();
  const std::string expected_text = expected.describe();
  require(actual_text == expected_text,
          label + " mismatch\nactual:\n" + actual_text + "\nexpected:\n" + expected_text);
}

std::size_t count_substrings(const std::string& text, const std::string& needle) {
  std::size_t count = 0;
  std::size_t pos = 0;
  while ((pos = text.find(needle, pos)) != std::string::npos) {
    ++count;
    pos += needle.size();
  }
  return count;
}

void require_not_contains(const std::string& haystack, const std::string& needle,
                          const std::string& msg) {
  if (haystack.find(needle) != std::string::npos) {
    throw std::runtime_error(msg + " (unexpected: " + needle + ")");
  }
}

void require_backend_shape(const Graph& graph, const std::string& label) {
  if (!simaai::neat::element_exists("neatdecoder")) {
    std::cout << "[INFO] " << label << ": neatdecoder not available; skipping backend checks\n";
    return;
  }

  const std::string backend = graph.describe_backend(false);
  require_contains(backend, "souphttpsrc", label + " should contain souphttpsrc");
  require_contains(backend, "multipartdemux", label + " should contain multipartdemux");
  require_contains(backend, "jpegparse", label + " should contain jpegparse");
  require_contains(backend, "neatdecoder", label + " should contain neatdecoder");
  require_contains(backend, "dec-type=mjpeg", label + " should configure MJPEG decode");
}

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
    std::vector<std::shared_ptr<simaai::neat::Node>> manual;
    simaai::neat::HttpSourceOptions source;
    source.location = opt.url;
    source.timeout_seconds = opt.timeout_seconds;
    source.retries = opt.retries;
    source.is_live = opt.is_live;
    source.do_timestamp = opt.do_timestamp;
    source.user_agent = opt.user_agent;
    manual.push_back(simaai::neat::nodes::HttpSource(std::move(source)));
    manual.push_back(simaai::neat::nodes::Queue());
    simaai::neat::MultipartJpegDemuxOptions demux;
    demux.boundary = opt.multipart_boundary;
    demux.single_stream = opt.multipart_single_stream;
    manual.push_back(simaai::neat::nodes::MultipartJpegDemux(std::move(demux)));
    manual.push_back(simaai::neat::nodes::JpegParse());
    manual.push_back(simaai::neat::nodes::Queue());
    simaai::neat::SimaDecodeOptions dec;
    dec.type = simaai::neat::SimaDecodeType::MJPEG;
    dec.sima_allocator_type = opt.sima_allocator_type;
    dec.out_format = opt.out_format;
    dec.decoder_name = opt.decoder_name;
    dec.raw_output = opt.decoder_raw_output;
    dec.next_element = opt.decoder_next_element;
    dec.dec_width = opt.dec_width;
    dec.dec_height = opt.dec_height;
    dec.dec_fps = opt.dec_fps;
    dec.num_buffers = opt.num_buffers;
    manual.push_back(simaai::neat::nodes::SimaDecode(dec));
    compare_graph_fragments(group, graph_from_nodes(std::move(manual)), "default topology");
    require_backend_shape(group, "default backend");
    require(count_substrings(group.describe(), "Queue") == 2,
            "default topology should include two queues");

    simaai::neat::ValidateOptions validate_opt;
    validate_opt.parse_launch = false;
    const simaai::neat::GraphReport report = group.validate(validate_opt);
    require(report.error_code.empty(), "default topology structural validation failed");

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

    simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions disabled_caps_opt = opt;
    disabled_caps_opt.output_caps.width = 320;
    disabled_caps_opt.output_caps.height = 240;
    disabled_caps_opt.output_caps.fps = 15;
    const Graph disabled_caps_group =
        simaai::neat::nodes::groups::HttpMjpegDecodedInput(disabled_caps_opt);
    require_not_contains(disabled_caps_group.describe(), "CapsRaw",
                         "disabled output_caps should not insert CapsRaw");
    const simaai::neat::OutputSpec disabled_caps_spec =
        simaai::neat::nodes::groups::HttpMjpegDecodedInputOutputSpec(disabled_caps_opt);
    require(disabled_caps_spec.width == 640 && disabled_caps_spec.height == 480,
            "disabled output_caps should leave decoder output shape unchanged");
    require(disabled_caps_spec.fps_num == 30 && disabled_caps_spec.memory == "SimaAI",
            "disabled output_caps should leave decoder output fps and memory unchanged");

    opt.insert_queue = false;
    const Graph no_queue_group = simaai::neat::nodes::groups::HttpMjpegDecodedInput(opt);
    require(count_substrings(no_queue_group.describe(), "Queue") == 0,
            "insert_queue=false topology should not include queues");
    if (simaai::neat::element_exists("neatdecoder")) {
      const std::string no_queue_backend = no_queue_group.describe_backend(false);
      require_not_contains(no_queue_backend,
                           "queue name=", "insert_queue=false backend should not contain queues");
    }

    opt.insert_queue = true;
    opt.decoder_raw_output = false;
    opt.use_videoconvert = true;
    opt.use_videoscale = true;
    opt.output_caps.enable = true;
    opt.output_caps.width = 320;
    opt.output_caps.height = 240;
    opt.output_caps.fps = 15;
    const Graph tail_group = simaai::neat::nodes::groups::HttpMjpegDecodedInput(opt);
    if (simaai::neat::element_exists("neatdecoder")) {
      const std::string tail_backend = tail_group.describe_backend(false);
      require_contains(tail_backend, "videoconvert", "tail backend should include videoconvert");
      require_contains(tail_backend, "videoscale", "tail backend should include videoscale");
      require_contains(tail_backend, "capsfilter", "tail backend should include capsfilter");
      require_contains(tail_backend, "width=320", "tail backend should include output width");
      require_contains(tail_backend, "height=240", "tail backend should include output height");
      require_contains(tail_backend, "framerate=15/1", "tail backend should include output fps");
    }

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
