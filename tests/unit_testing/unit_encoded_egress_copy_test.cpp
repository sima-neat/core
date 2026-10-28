// A USB camera graph splits its loaned JPEG capture buffers to a decoder and to a VideoSender
// passthrough. Only the sender edge copies: its segment can keep what it is handed, so taking the
// capture buffer and loan there exhausts the camera pool. The decode edge stays zero-copy.
#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif

#include "gst/GstInit.h"
#include "nodes/common/Output.h"
#include "nodes/groups/VideoSender.h"
#include "nodes/io/Input.h"
#include "nodes/io/UsbCameraInput.h"
#include "nodes/io/internal/UvcCamera.h"
#include "nodes/sima/SimaDecode.h"
#include "pipeline/Graph.h"
#include "pipeline/EncodedSampleUtil.h"
#include "pipeline/GraphOptions.h"
#include "pipeline/TensorAdapters.h"
#include "pipeline/internal/HolderLoanGate.h"
#include "pipeline/internal/InputStream.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/internal/SampleUtil.h"
#include "pipeline/runtime/ExecutionGraphPlan.h"
#include "pipeline/runtime/RunCore.h"
#include "test_utils.h"

#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace simaai::neat;

const std::string kJpegCaps = "image/jpeg,width=(int)16,height=(int)16";

bool has_node(const runtime::PipelineSegmentPlan& segment, const std::string& kind) {
  return std::any_of(segment.nodes.begin(), segment.nodes.end(),
                     [&](const auto& node) { return node && node->kind() == kind; });
}

// The camera's decode edge and its VideoSender edge land in separate segments; only the
// sender's segment copies encoded input, and building it hands that to its InputStream.
void sender_segment_is_the_only_copying_segment(GraphLinkPolicy policy) {
  UsbCameraInputOptions camera;
  camera.camera_name = "usb-camera-egress";
  camera.buffer_name = "egress_usb";
  SimaDecodeOptions decode;
  decode.type = SimaDecodeType::MJPEG;
  decode.raw_output = true;
  decode.out_format = FormatTag::NV12;
  decode.dec_width = static_cast<int>(camera.width);
  decode.dec_height = static_cast<int>(camera.height);
  decode.dec_fps = static_cast<int>(camera.framerate_num);

  Graph source("usb_source");
  source.add(internal::uvc::camera_node(camera, {"/dev/video0", FormatTag::NV12}));
  Graph decoder("usb_decode");
  decoder.add(nodes::SimaDecode(decode));
  decoder.add(nodes::Output());
  auto sender = nodes::groups::VideoSenderOptions::Passthrough(nodes::groups::RtspCodec::MJPEG);
  sender.host = "127.0.0.1";
  GraphLinkOptions video_link;
  video_link.policy = policy;
  Graph app("usb_split");
  app.connect(source, decoder);
  app.connect(source, nodes::groups::VideoSender(sender), video_link);

  const auto plan = runtime::compile_public_graph(app, RunOptions{});
  int copying = 0;
  bool saw_decoder = false;
  const runtime::PipelineSegmentPlan* sender_segment = nullptr;
  for (const auto& segment : plan.pipeline_segments) {
    const bool copies = runtime::segment_copies_encoded_input(segment);
    if (has_node(segment, "SimaDecode")) {
      saw_decoder = true;
      require(!copies, "the decode segment must keep zero-copy holder transport");
    }
    if (copies) {
      ++copying;
      sender_segment = &segment;
      require(has_node(segment, "JpegPacketize") && !has_node(segment, "SimaDecode"),
              "only the VideoSender segment may copy encoded input");
    }
  }
  require(saw_decoder, "plan must contain the decode segment");
  require(copying == 1, "exactly the VideoSender segment must copy encoded input (got " +
                            std::to_string(copying) + ")");

  std::vector<uint8_t> jpeg;
  require(cv::imencode(".jpg", cv::Mat(camera.height, camera.width, CV_8UC3, cv::Scalar(0)), jpeg),
          "failed to encode the seed JPEG");
  runtime::RunCoreStartOptions start;
  start.mode = RunMode::Async;
  start.seed = make_encoded_sample(jpeg, "image/jpeg,width=(int)" + std::to_string(camera.width) +
                                             ",height=(int)" + std::to_string(camera.height));
  auto core = runtime::RunCore::start_pipeline_segment(*sender_segment, std::move(start));
  const bool stream_copies = core->pipeline.stream_opt.copy_encoded_input;
  core->close();
  require(stream_copies, "the built VideoSender segment must copy encoded input");
}

// A device-tagged JPEG holder that takes one loan credit, as a camera capture buffer does.
Sample loaned_jpeg(const pipeline_internal::HolderLoanGatePtr& gate, GstMemory** memory) {
  const guint8 bytes[] = {0xFF, 0xD8, 0xFF, 0xD9};
  GstBuffer* buffer = gst_buffer_new_allocate(nullptr, sizeof(bytes), nullptr);
  require(buffer != nullptr, "failed to allocate fixture buffer");
  gst_buffer_fill(buffer, 0, bytes, sizeof(bytes));
  *memory = gst_buffer_peek_memory(buffer, 0);
  GstCaps* caps = gst_caps_from_string(kJpegCaps.c_str());
  GstSample* gst_sample = gst_sample_new(buffer, caps, nullptr, nullptr);
  gst_caps_unref(caps);
  gst_buffer_unref(buffer);
  require(gst_sample != nullptr, "failed to create fixture GstSample");

  Sample sample;
  sample.kind = SampleKind::TensorSet;
  sample.payload_type = PayloadType::Encoded;
  sample.media_type = "image/jpeg";
  sample.caps_string = kJpegCaps;
  sample.tensors = TensorList{from_gst_sample(gst_sample)};
  gst_sample_unref(gst_sample);
  sample.tensors.front().device.type = DeviceType::SIMA_CVU;
  sample.tensors.front().storage->device.type = DeviceType::SIMA_CVU;
  std::string err;
  require(pipeline_internal::attach_zero_copy_loan_to_sample(sample, gate, &err),
          "fixture holder did not take a loan: " + err);
  require(gate->inflight() == 1, "fixture holder must hold one loan");
  return sample;
}

// Push one loaned holder through an InputStream configured like a graph segment edge and report
// whether the downstream buffer is the holder's memory and whether it still holds the loan.
void push_through_segment_edge(bool copy_encoded_input) {
  const char* where = copy_encoded_input ? "egress edge" : "decode edge";
  auto gate = std::make_shared<pipeline_internal::HolderLoanGate>(1);
  GstMemory* holder_memory = nullptr;
  Sample sample = loaned_jpeg(gate, &holder_memory);

  GstElement* pipeline = gst_pipeline_new(nullptr);
  GstElement* appsrc = gst_element_factory_make("appsrc", "mysrc");
  GstElement* appsink = gst_element_factory_make("appsink", "mysink");
  require(pipeline && appsrc && appsink, std::string(where) + ": failed to create pipeline");
  g_object_set(appsrc, "is-live", TRUE, "format", GST_FORMAT_TIME, "block", TRUE, nullptr);
  g_object_set(appsink, "sync", FALSE, "max-buffers", 1U, "drop", FALSE, "enable-last-sample",
               FALSE, nullptr);
  GstCaps* caps = gst_caps_from_string(kJpegCaps.c_str());
  gst_app_src_set_caps(GST_APP_SRC(appsrc), caps);
  gst_caps_unref(caps);
  gst_bin_add_many(GST_BIN(pipeline), appsrc, appsink, nullptr);
  require(gst_element_link(appsrc, appsink), std::string(where) + ": failed to link pipeline");
  appsrc = gst_bin_get_by_name(GST_BIN(pipeline), "mysrc");
  appsink = gst_bin_get_by_name(GST_BIN(pipeline), "mysink");
  require(gst_element_set_state(pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE,
          std::string(where) + ": failed to start pipeline");

  InputOptions src_opt;
  src_opt.payload_type = PayloadType::Encoded;
  src_opt.format = FormatTag::ENCODED;
  src_opt.caps_override = kJpegCaps;
  src_opt.block = true;
  InputStreamOptions stream_opt;
  stream_opt.allow_graph_internal_zero_copy_input = true;
  stream_opt.copy_encoded_input = copy_encoded_input;
  InputStream stream =
      InputStream::create(pipeline, appsrc, appsink, derive_sample_spec_or_throw(sample), src_opt,
                          stream_opt, {}, nullptr);
  require(stream.try_push_message(sample), std::string(where) + ": push failed");
  sample = Sample{};

  GstSample* pulled = gst_app_sink_try_pull_sample(GST_APP_SINK(appsink), GST_SECOND);
  require(pulled != nullptr, std::string(where) + ": pull timed out");
  const bool same_memory =
      gst_buffer_peek_memory(gst_sample_get_buffer(pulled), 0) == holder_memory;
  const int inflight_while_held = gate->inflight();
  gst_sample_unref(pulled);
  stream.close();

  if (copy_encoded_input) {
    require(!same_memory, "egress edge must push a copy, not the holder's buffer");
    require(inflight_while_held == 0, "egress edge must not carry the producer's loan");
  } else {
    require(same_memory, "decode edge must forward the holder's buffer zero-copy");
    require(inflight_while_held == 1, "decode edge buffer must carry the producer's loan");
  }
  // The pipeline drops its last buffer reference on its own thread.
  for (int i = 0; i < 100 && gate->inflight() != 0; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  require(gate->inflight() == 0, std::string(where) + ": loan must release with the buffer");
}

} // namespace

int main() {
  try {
    gst_init_once();
    sender_segment_is_the_only_copying_segment(GraphLinkPolicy::Default);
    sender_segment_is_the_only_copying_segment(GraphLinkPolicy::RealtimeLatestByStream);
    push_through_segment_edge(/*copy_encoded_input=*/true);
    push_through_segment_edge(/*copy_encoded_input=*/false);
  } catch (const std::exception& e) {
    return fail_test(e.what());
  }
  std::cout << "[OK] unit_encoded_egress_copy_test\n";
  return 0;
}
