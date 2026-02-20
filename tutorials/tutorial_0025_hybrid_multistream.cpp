// tutorial_0025_hybrid_multistream.cpp
// Story: multi-stream hybrid graph (scheduler + pooled stage + bundle join + pipeline sink).
// What you learn:
// - Stage nodes can accept many streams via stream_id.
// - StreamScheduler provides fairness across active streams.
// - StageNodeOptions allow pooling (instances + key_by routing).
// - Bundle output lands at a graph sink for inspection.

#include "neat/graph.h"
#include "neat/session.h"
#include "tutorial_common.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

std::vector<int64_t> contiguous_strides_bytes(const std::vector<int64_t>& shape,
                                              int64_t elem_bytes) {
  std::vector<int64_t> strides(shape.size(), 0);
  int64_t stride = elem_bytes;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides[static_cast<size_t>(i)] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return strides;
}

simaai::neat::Sample make_rgb_sample(const std::string& stream_id, int frame_id) {
  const int w = 8;
  const int h = 6;
  const int c = 3;
  const std::size_t bytes = static_cast<std::size_t>(w) * h * c;

  simaai::neat::Tensor t;
  t.device = {simaai::neat::DeviceType::CPU, 0};
  t.dtype = simaai::neat::TensorDType::UInt8;
  t.layout = simaai::neat::TensorLayout::HWC;
  t.shape = {h, w, c};
  t.semantic.image = simaai::neat::ImageSpec{simaai::neat::ImageSpec::PixelFormat::RGB, ""};
  t.storage = simaai::neat::make_cpu_owned_storage(bytes);
  t.strides_bytes = contiguous_strides_bytes(t.shape, 1);
  t.read_only = false;

  {
    auto mapping = t.map(simaai::neat::MapMode::Write);
    auto* ptr = static_cast<std::uint8_t*>(mapping.data);
    for (std::size_t i = 0; i < bytes; ++i) {
      ptr[i] = static_cast<std::uint8_t>(i % 255);
    }
  }
  t.read_only = true;

  simaai::neat::Sample sample;
  sample.kind = simaai::neat::SampleKind::Tensor;
  sample.tensor = std::move(t);
  sample.frame_id = frame_id;
  sample.stream_id = stream_id;
  return sample;
}

simaai::neat::Sample make_bbox_sample(const simaai::neat::Sample& in) {
  std::vector<float> data = {0.1f, 0.1f, 0.5f, 0.5f};
  auto holder = std::make_shared<std::vector<float>>(std::move(data));

  simaai::neat::Tensor t;
  t.storage = simaai::neat::make_cpu_external_storage(holder->data(),
                                                      holder->size() * sizeof(float), holder, true);
  t.device = {simaai::neat::DeviceType::CPU, 0};
  t.read_only = true;
  t.dtype = simaai::neat::TensorDType::Float32;
  t.layout = simaai::neat::TensorLayout::HW;
  t.shape = {1, static_cast<int64_t>(holder->size())};
  t.strides_bytes = {static_cast<int64_t>(holder->size() * sizeof(float)),
                     static_cast<int64_t>(sizeof(float))};

  simaai::neat::Sample out;
  out.kind = simaai::neat::SampleKind::Tensor;
  out.media_type = "application/vnd.simaai.tensor";
  out.format = "FP32";
  out.payload_tag = "BBOX";
  out.tensor = std::move(t);
  out.frame_id = in.frame_id;
  out.stream_id = in.stream_id;
  out.pts_ns = in.pts_ns;
  out.dts_ns = in.dts_ns;
  out.duration_ns = in.duration_ns;
  out.input_seq = in.input_seq;
  return out;
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (sima_tutorial::wants_help(argc, argv)) {
      std::cout << "Usage: " << argv[0] << "\n";
      sima_tutorial::print_common_flags(std::cout);
      return 0;
    }

    const int num_streams = 20;
    const int frames_per_stream = 5;

    simaai::neat::graph::Graph g;

    using namespace simaai::neat::graph;
    using namespace simaai::neat::graph::dsl;

    auto stamp = add(g, nodes::StampFrameIdNode("stamp"));

    nodes::StreamSchedulerOptions sched_opt;
    sched_opt.per_stream_queue = 2;
    sched_opt.drop_policy = nodes::StreamDropPolicy::DropOldest;
    auto sched = add(g, nodes::StreamSchedulerNode(sched_opt, "sched"));

    auto fan = add(g, nodes::FanOutNode({"image", "infer"}, "fan"));

    nodes::StageNodeOptions pool_opt;
    pool_opt.instances = 4;
    pool_opt.key_by = nodes::StageKeyBy::StreamId;
    pool_opt.max_inflight = 64;

    auto model =
        add(g, nodes::LambdaStageNode(
                   "FakeModel", {"in"}, {"out"},
                   [](StageMsg&& msg, std::vector<StageOutMsg>& out, const StagePorts& ports) {
                     auto sample = make_bbox_sample(msg.sample);
                     out.push_back(StageOutMsg{.out_port = ports.out_port("out"),
                                               .sample = std::move(sample)});
                   },
                   "model_pool", pool_opt));

    auto join = add(g, nodes::JoinBundleNode({"image", "bbox"}, "join"));

    stamp >> sched >> fan;
    fan["image"] >> join["image"];
    fan["infer"] >> model >> join["bbox"];
    // Graph sink: join is terminal.

    std::cout << "[GraphPrinter] text\n";
    std::cout << GraphPrinter::to_text(g) << "\n\n";

    GraphRun run = GraphSession(std::move(g)).build();
    auto in = run.input(stamp);
    auto out = run.output(join);

    for (int frame = 0; frame < frames_per_stream; ++frame) {
      for (int sid = 0; sid < num_streams; ++sid) {
        in.push(make_rgb_sample(std::to_string(sid), frame));
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    const int expected = num_streams * frames_per_stream;
    for (int i = 0; i < expected; ++i) {
      auto bundle = out.pull_or_throw(2000);
      if (i < 5) {
        std::cout << "[bundle] stream=" << bundle.stream_id << " frame=" << bundle.frame_id
                  << " fields=" << bundle.fields.size() << "\n";
      }
    }

    run.stop();
    std::cout << "[OK] tutorial_0025 complete\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
