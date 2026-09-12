#include "dmabuf_test_utils.h"
#include "gst/GstInit.h"
#include "graph/Graph.h"
#include "graph/GraphBuild.h"
#include "graph/StageExecutor.h"
#include "graph/nodes/StageNode.h"
#include "pipeline/TensorAdapters.h"
#include "test_main.h"
#include "test_utils.h"

#include <chrono>
#include <memory>
#include <optional>
#include <thread>

namespace {

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
    const simaai::neat::graph::PortId out_port = (out_port_ == simaai::neat::graph::kInvalidPort)
                                                     ? simaai::neat::graph::kInvalidPort
                                                     : out_port_;
    out.push_back(
        simaai::neat::graph::StageOutMsg{.out_port = out_port, .sample = std::move(msg.sample)});
  }

private:
  simaai::neat::graph::PortId out_port_ = simaai::neat::graph::kInvalidPort;
};

class FanOutStage final : public simaai::neat::graph::StageExecutor {
public:
  void set_ports(const simaai::neat::graph::StagePorts& ports) override {
    left_ = ports.out_port("left");
    right_ = ports.out_port("right");
  }

  void on_input(simaai::neat::graph::StageMsg&& msg,
                std::vector<simaai::neat::graph::StageOutMsg>& out) override {
    simaai::neat::Sample sample = std::move(msg.sample);
    if (left_ != simaai::neat::graph::kInvalidPort) {
      out.push_back(simaai::neat::graph::StageOutMsg{.out_port = left_, .sample = sample});
    }
    if (right_ != simaai::neat::graph::kInvalidPort) {
      out.push_back(
          simaai::neat::graph::StageOutMsg{.out_port = right_, .sample = std::move(sample)});
    }
  }

private:
  simaai::neat::graph::PortId left_ = simaai::neat::graph::kInvalidPort;
  simaai::neat::graph::PortId right_ = simaai::neat::graph::kInvalidPort;
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

std::shared_ptr<simaai::neat::graph::Node> make_fanout_node() {
  using simaai::neat::graph::PortDesc;
  using simaai::neat::graph::nodes::StageNode;
  StageNode::StageExecutorFactory factory = []() { return std::make_unique<FanOutStage>(); };
  std::vector<PortDesc> inputs = {PortDesc{.name = "in", .spec = simaai::neat::OutputSpec{}}};
  std::vector<PortDesc> outputs = {PortDesc{.name = "left", .spec = simaai::neat::OutputSpec{}},
                                   PortDesc{.name = "right", .spec = simaai::neat::OutputSpec{}}};
  return std::make_shared<StageNode>("FanOut", std::move(factory), std::move(inputs),
                                     std::move(outputs), "fanout");
}

sima_test::DmaBufSpan sample_dma_span(const simaai::neat::Sample& sample) {
  const auto tensors = simaai::neat::tensors_from_sample(sample, true);
  require(tensors.size() == 1 && tensors.front().storage && tensors.front().storage->holder,
          "fan-out output must retain one tensor holder");
  require(tensors.front().storage->kind == simaai::neat::StorageKind::GstSample,
          "fan-out copied DMA-BUF payload to CPU storage");
  auto* retained = static_cast<GstSample*>(tensors.front().storage->holder.get());
  require(GST_IS_SAMPLE(retained), "fan-out output holder is not a GstSample");
  return sima_test::dmabuf_span(gst_sample_get_buffer(retained));
}

void require_pool_held(GstBufferPool* pool) {
  GstBufferPoolAcquireParams params{};
  params.flags = GST_BUFFER_POOL_ACQUIRE_FLAG_DONTWAIT;
  GstBuffer* buffer = nullptr;
  const auto result = gst_buffer_pool_acquire_buffer(pool, &buffer, &params);
  if (buffer) {
    gst_buffer_unref(buffer);
  }
  require(result != GST_FLOW_OK, "one-slot DMA-BUF pool recycled a still-held branch allocation");
}

} // namespace

RUN_TEST("hybrid_graph_fanout_test", [] {
  simaai::neat::gst_init_once();
  namespace dma = simaai::neat::internal::dmabuf;
  dma::Error error;
  GstBufferPool* raw_pool =
      dma::createDmaBufPool(dma::HeapKind::Cma, 16 * 12 * 3, 1, 1, {}, &error);
  require(raw_pool != nullptr, "real one-slot CMA pool creation failed: " + error.message());
  std::shared_ptr<GstBufferPool> pool(raw_pool, [](GstBufferPool* p) {
    gst_buffer_pool_set_active(p, FALSE);
    gst_object_unref(p);
  });
  std::shared_ptr<GstCaps> caps(
      gst_caps_from_string("video/x-raw,format=RGB,width=16,height=12,framerate=30/1"),
      gst_caps_unref);
  require(caps != nullptr, "fan-out test caps creation failed");
  simaai::neat::graph::Graph g;

  auto fan = g.add(make_fanout_node());
  auto sink_a = g.add(make_pass_node("sink_a"));
  auto sink_b = g.add(make_pass_node("sink_b"));

  g.connect(fan, sink_a, "left", "in");
  g.connect(fan, sink_b, "right", "in");
  simaai::neat::graph::GraphRun run = simaai::neat::graph::build(std::move(g));

  const int total = 5;
  std::optional<sima_test::DmaBufSpan> pool_identity;
  for (int i = 0; i < total; ++i) {
    GstBuffer* buffer = nullptr;
    GstBufferPoolAcquireParams params{};
    params.flags = GST_BUFFER_POOL_ACQUIRE_FLAG_DONTWAIT;
    require(gst_buffer_pool_acquire_buffer(pool.get(), &buffer, &params) == GST_FLOW_OK && buffer,
            "fan-out pool acquisition failed");
    const auto original = sima_test::dmabuf_span(buffer);
    if (pool_identity) {
      require(original == *pool_identity, "one-slot fan-out pool replaced its DMA allocation");
    } else {
      pool_identity = original;
    }
    GstSample* gst_sample = gst_sample_new(buffer, caps.get(), nullptr, nullptr);
    gst_buffer_unref(buffer);
    require(gst_sample != nullptr, "fan-out test sample creation failed");
    simaai::neat::Sample sample = simaai::neat::sample_from_tensors(
        simaai::neat::TensorList{simaai::neat::from_gst_sample(gst_sample)});
    gst_sample_unref(gst_sample);
    sample.frame_id = i;
    sample.stream_id = "fanout";
    require(run.push(fan, std::move(sample)), "GraphRun::push failed");
    sample = {};

    auto out_a = run.pull(sink_a, 2000);
    auto out_b = run.pull(sink_b, 2000);
    require(out_a.has_value(), "sink_a pull timed out");
    require(out_b.has_value(), "sink_b pull timed out");
    require(out_a->frame_id == i, "sink_a frame_id mismatch");
    require(out_b->frame_id == i, "sink_b frame_id mismatch");
    require(sample_dma_span(*out_a) == original && sample_dma_span(*out_b) == original,
            "fan-out branches must retain the original DMA-BUF identity and span");
    require_pool_held(pool.get());
    out_a.reset();
    require_pool_held(pool.get());
    out_b.reset();

    GstBuffer* recycled = nullptr;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (gst_buffer_pool_acquire_buffer(pool.get(), &recycled, &params) != GST_FLOW_OK &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    require(recycled != nullptr, "DMA-BUF did not return after both branches released their views");
    const auto recycled_span = sima_test::dmabuf_span(recycled);
    gst_buffer_unref(recycled);
    require(recycled_span == original, "fan-out release must recycle, not replace, pixel storage");
  }

  run.stop();
});
