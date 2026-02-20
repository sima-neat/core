#include "graph/Graph.h"
#include "graph/GraphSession.h"
#include "graph/StageExecutor.h"
#include "graph/nodes/PipelineNode.h"
#include "graph/nodes/StageNode.h"
#include "nodes/common/Output.h"
#include "nodes/common/VideoConvert.h"
#include "pipeline/internal/TensorUtil.h"
#include "test_main.h"
#include "test_utils.h"

#include <gst/gst.h>
#include <gst/video/video.h>

#include <limits>

namespace {

GstVideoFormat gst_format_from_pixel(simaai::neat::ImageSpec::PixelFormat fmt) {
  switch (fmt) {
  case simaai::neat::ImageSpec::PixelFormat::RGB:
    return GST_VIDEO_FORMAT_RGB;
  case simaai::neat::ImageSpec::PixelFormat::BGR:
    return GST_VIDEO_FORMAT_BGR;
  case simaai::neat::ImageSpec::PixelFormat::GRAY8:
    return GST_VIDEO_FORMAT_GRAY8;
  case simaai::neat::ImageSpec::PixelFormat::NV12:
    return GST_VIDEO_FORMAT_NV12;
  case simaai::neat::ImageSpec::PixelFormat::I420:
    return GST_VIDEO_FORMAT_I420;
  case simaai::neat::ImageSpec::PixelFormat::UNKNOWN:
    return GST_VIDEO_FORMAT_UNKNOWN;
  }
  return GST_VIDEO_FORMAT_UNKNOWN;
}

void tensor_dims_from_shape(const simaai::neat::Tensor& tensor, int* w, int* h) {
  if (w)
    *w = -1;
  if (h)
    *h = -1;
  if (tensor.shape.size() < 2)
    return;
  const int64_t height = tensor.shape[0];
  const int64_t width = tensor.shape[1];
  if (w && width > 0 && width <= std::numeric_limits<int>::max()) {
    *w = static_cast<int>(width);
  }
  if (h && height > 0 && height <= std::numeric_limits<int>::max()) {
    *h = static_cast<int>(height);
  }
}

void verify_cpu_and_video_meta(const simaai::neat::Sample& sample) {
  if (sample.kind != simaai::neat::SampleKind::Tensor || !sample.tensor.has_value()) {
    throw std::runtime_error("PassThroughStage: expected tensor sample");
  }
  const simaai::neat::Tensor& tensor = *sample.tensor;
  if (tensor.device.type != simaai::neat::DeviceType::CPU) {
    throw std::runtime_error("PassThroughStage: incoming tensor not on CPU");
  }
  if (!tensor.storage) {
    throw std::runtime_error("PassThroughStage: tensor missing storage");
  }
  if (tensor.storage->kind != simaai::neat::StorageKind::GstSample || !tensor.storage->holder) {
    return;
  }

  if (!tensor.semantic.image.has_value()) {
    return;
  }

  const GstVideoFormat expected = gst_format_from_pixel(tensor.semantic.image->format);
  if (expected == GST_VIDEO_FORMAT_UNKNOWN) {
    return;
  }

  int w = -1;
  int h = -1;
  tensor_dims_from_shape(tensor, &w, &h);

  GstBuffer* buf =
      simaai::neat::pipeline_internal::buffer_from_tensor_holder(tensor.storage->holder);
  if (!buf) {
    throw std::runtime_error("PassThroughStage: missing GstBuffer from holder");
  }

  GstVideoMeta* meta = gst_buffer_get_video_meta(buf);
  if (!meta) {
    gst_buffer_unref(buf);
    throw std::runtime_error("PassThroughStage: missing GstVideoMeta on holder buffer");
  }
  if (w > 0 && meta->width != static_cast<guint>(w)) {
    gst_buffer_unref(buf);
    throw std::runtime_error("PassThroughStage: GstVideoMeta width mismatch");
  }
  if (h > 0 && meta->height != static_cast<guint>(h)) {
    gst_buffer_unref(buf);
    throw std::runtime_error("PassThroughStage: GstVideoMeta height mismatch");
  }
  if (meta->format != expected) {
    gst_buffer_unref(buf);
    throw std::runtime_error("PassThroughStage: GstVideoMeta format mismatch");
  }

  gst_buffer_unref(buf);
}

class PassThroughStage final : public simaai::neat::graph::StageExecutor {
public:
  void set_ports(const simaai::neat::graph::StagePorts& ports) override {
    const simaai::neat::graph::PortId only = ports.only_output();
    if (only != simaai::neat::graph::kInvalidPort) {
      out_port_ = only;
    }
  }

  void on_input(simaai::neat::graph::StageMsg&& msg,
                std::vector<simaai::neat::graph::StageOutMsg>& out) override {
    verify_cpu_and_video_meta(msg.sample);
    const simaai::neat::graph::PortId out_port = (out_port_ == simaai::neat::graph::kInvalidPort)
                                                     ? simaai::neat::graph::kInvalidPort
                                                     : out_port_;
    out.push_back(
        simaai::neat::graph::StageOutMsg{.out_port = out_port, .sample = std::move(msg.sample)});
  }

private:
  simaai::neat::graph::PortId out_port_ = simaai::neat::graph::kInvalidPort;
};

std::shared_ptr<simaai::neat::graph::Node> make_pass_node(const std::string& label) {
  using simaai::neat::graph::PortDesc;
  using simaai::neat::graph::nodes::StageNode;
  StageNode::StageExecutorFactory factory = []() { return std::make_unique<PassThroughStage>(); };
  std::vector<PortDesc> inputs = {PortDesc{.name = "in", .spec = simaai::neat::OutputSpec{}}};
  std::vector<PortDesc> outputs = {PortDesc{.name = "out", .spec = simaai::neat::OutputSpec{}}};
  return std::make_shared<StageNode>("PassThrough", std::move(factory), std::move(inputs),
                                     std::move(outputs), label);
}

} // namespace

RUN_TEST("hybrid_graph_basic_test", [] {
  simaai::neat::graph::Graph g;

  auto pipe_in = g.add(std::make_shared<simaai::neat::graph::nodes::PipelineNode>(
      simaai::neat::nodes::VideoConvert(), "convert"));
  auto stage = g.add(make_pass_node("stage"));
  auto pipe_out = g.add(std::make_shared<simaai::neat::graph::nodes::PipelineNode>(
      simaai::neat::nodes::Output(), "sink"));

  g.connect(pipe_in, stage);
  g.connect(stage, pipe_out);

  simaai::neat::graph::GraphSession session(std::move(g));
  simaai::neat::graph::GraphRunOptions run_opt;
  simaai::neat::graph::GraphRun run = session.build(run_opt);
  struct RunStopGuard {
    simaai::neat::graph::GraphRun* run_ptr = nullptr;
    ~RunStopGuard() {
      if (!run_ptr)
        return;
      try {
        run_ptr->stop();
      } catch (...) {
      }
    }
  } run_guard{&run};
  std::fprintf(stderr, "%s", run.describe().c_str());

  simaai::neat::Sample sample;
  sample.kind = simaai::neat::SampleKind::Tensor;
  sample.tensor = make_color_tensor(64, 48, simaai::neat::ImageSpec::PixelFormat::RGB);
  sample.frame_id = 42;
  sample.stream_id = "stream0";

  require(run.push(pipe_in, sample), "GraphRun::push failed");

  auto out = run.pull(pipe_out, 2000);
  if (!out.has_value()) {
    const std::string err = run.last_error();
    if (!err.empty()) {
      throw std::runtime_error(err);
    }
    throw std::runtime_error("GraphRun::pull timed out");
  }
  require(out->frame_id == sample.frame_id, "frame_id mismatch");
  require(out->stream_id == sample.stream_id, "stream_id mismatch");

  run.stop();
  run_guard.run_ptr = nullptr;
});
