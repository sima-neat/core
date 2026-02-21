#include "neat/graph.h"
#include "common/cpp_utils.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

// Why: this chapter contrasts model-backed graph stages with deterministic fallback flow.
// Why: learners should see identical checkpoint output even when model assets are unavailable.

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
    auto map = t.map(simaai::neat::MapMode::Write);
    auto* p = static_cast<std::uint8_t*>(map.data);
    for (std::size_t i = 0; i < bytes; ++i) {
      p[i] = static_cast<std::uint8_t>(i % 255);
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

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << "\n";
  tutorial_v2::print_common_flags(std::cout);
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (tutorial_v2::wants_help(argc, argv)) {
      print_help(argv[0]);
      return 0;
    }

    const int streams = tutorial_v2::parse_int_arg(argc, argv, "--streams", 8);
    const int frames = tutorial_v2::parse_int_arg(argc, argv, "--frames", 4);

    tutorial_v2::step("input_contract", "stream/frame tags must survive scheduler and join stages");
    tutorial_v2::step("run_mode_choice",
                      "build multistream graph with fair scheduling and bundle join");
    tutorial_v2::why("understand the contract first: inputs, run mode, and outputs");
    tutorial_v2::tradeoff(
        "prefer deterministic samples and stable contracts over production realism");
    tutorial_v2::failure_mode(
        "runtime/plugin issues should degrade to runtime_fallback without losing observability");
    tutorial_v2::interpret_output(
        "use CHECK markers plus SIGNATURE fields to validate behavior and parity");

    using namespace simaai::neat::graph;
    using namespace simaai::neat::graph::dsl;

    Graph g;

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
                     out.push_back(
                         StageOutMsg{.out_port = ports.out_port("out"), .sample = msg.sample});
                   },
                   "model_pool", pool_opt));

    auto join = add(g, nodes::JoinBundleNode({"image", "bbox"}, "join"));

    stamp >> sched >> fan;
    fan["image"] >> join["image"];
    fan["infer"] >> model >> join["bbox"];

    std::cout << GraphPrinter::to_text(g) << "\n";

    GraphRun run = GraphSession(std::move(g)).build();
    auto in = run.input(stamp);
    auto out = run.output(join);

    for (int frame = 0; frame < frames; ++frame) {
      for (int sid = 0; sid < streams; ++sid) {
        tutorial_v2::check("graph_push", in.push(make_rgb_sample(std::to_string(sid), frame)),
                           "sample accepted by input node");
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    const int expected = streams * frames;
    int received = 0;
    int first_fields = -1;
    for (int i = 0; i < expected; ++i) {
      auto bundle = out.pull_or_throw(2000);
      if (first_fields < 0) {
        first_fields = static_cast<int>(bundle.fields.size());
      }
      ++received;
      if (i < 4) {
        std::cout << "bundle stream=" << bundle.stream_id << " fields=" << bundle.fields.size()
                  << "\n";
      }
    }

    run.stop();

    tutorial_v2::step("output_interpretation", "joined bundle cardinality validates graph wiring");
    tutorial_v2::check("all_outputs_received", received == expected,
                       "expected=" + std::to_string(expected) +
                           ", received=" + std::to_string(received));
    tutorial_v2::check("bundle_has_two_fields", first_fields == 2,
                       "join should emit image+bbox bundle");

    tutorial_v2::print_signature({
        {"tutorial", "016"},
        {"lang", "cpp"},
        {"flow", "multistream_stage_graph"},
        {"run_mode", "graph_sync_pull"},
        {"output_kind", std::to_string(static_cast<int>(simaai::neat::SampleKind::Bundle))},
        {"tensor_rank", "-1"},
        {"field_count", std::to_string(first_fields)},
        {"streams", std::to_string(streams)},
        {"frames", std::to_string(frames)},
    });

    std::cout << "[OK] 016_graph_multistream\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
