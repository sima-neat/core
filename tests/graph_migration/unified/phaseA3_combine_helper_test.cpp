#include "graph_migration/common/phase3_graph_test_utils.h"
#include "graphs/Fragments.h"
#include "nodes/common/Caps.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "test_main.h"
#include "test_utils.h"

#include <chrono>
#include <functional>
#include <string>
#include <thread>

namespace {

void require_throws_contains(const std::function<void()>& fn, const std::string& needle,
                             const std::string& label) {
  try {
    fn();
  } catch (const std::exception& e) {
    const std::string what = e.what();
    if (what.find(needle) == std::string::npos) {
      throw std::runtime_error(label + ": expected exception containing '" + needle +
                               "', got: " + what);
    }
    return;
  }
  throw std::runtime_error(label + ": expected exception containing '" + needle + "'");
}

simaai::neat::Sample sample_with_frame(int64_t frame_id, std::string stream_id, uint8_t value) {
  (void)value;
  simaai::neat::Sample sample = graph_phase3_test::make_tensor_sample(frame_id, stream_id);
  return sample;
}

simaai::neat::Sample sample_with_pts(int64_t pts_ns, std::string stream_id, uint8_t value) {
  simaai::neat::Sample sample = sample_with_frame(-1, std::move(stream_id), value);
  sample.pts_ns = pts_ns;
  return sample;
}

void require_bundle_fields(const simaai::neat::Sample& sample, const std::string& label) {
  require(sample.kind == simaai::neat::SampleKind::Bundle, label + ": output is not a bundle");
  require(sample.fields.size() == 2U, label + ": expected two bundle fields");
  require(sample.fields[0].stream_label == "left", label + ": field[0] should be left");
  require(sample.fields[1].stream_label == "right", label + ": field[1] should be right");
}

void require_strict_combine_error(simaai::neat::Run& run, const std::string& input,
                                  simaai::neat::Sample sample, const std::string& needle,
                                  const std::string& label) {
  const bool pushed = run.push(input, simaai::neat::Sample{std::move(sample)});
  if (!pushed) {
    require_contains(run.last_error(), needle, label + ": push error mismatch");
    return;
  }

  for (int attempt = 0; attempt < 20; ++attempt) {
    if (run.last_error().find(needle) != std::string::npos) {
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
  }

  auto out = run.pull("combined", 100);
  require(!out.has_value(), label + ": strict combine should not emit an output");
  require_contains(run.last_error(), needle, label + ": runtime error mismatch");
}

} // namespace

RUN_TEST("graph_migration_phaseA3_combine_helper_test", [] {
  {
    require_throws_contains(
        [] { (void)simaai::neat::graphs::Combine({"left", "left"}, "combined"); },
        "duplicate endpoint name", "Combine helper should reject duplicate input names");
  }

  {
    simaai::neat::Graph pass;
    pass.add(simaai::neat::nodes::Input("image"));
    pass.add(simaai::neat::nodes::Output("out"));
    pass.connect("image", "out");
    simaai::neat::Run run = pass.build();
    simaai::neat::Sample sample = sample_with_pts(987654321, "cam-pass", 0x55);
    sample.frame_id = 777;
    require(run.push("image", sample), "Input->Output push failed: " + run.last_error());
    auto out = run.pull("out", 5000);
    require(out.has_value(), "Input->Output pull timed out: " + run.last_error());
    require(out->pts_ns == 987654321, "Input->Output should preserve pts_ns");
    require(out->frame_id == 777, "Input->Output should preserve frame_id");
    require(out->stream_id == "cam-pass", "Input->Output should preserve stream_id");
    run.close();
  }

  {
    simaai::neat::Graph join = simaai::neat::graphs::Combine({"left", "right"}, "combined",
                                                             simaai::neat::CombinePolicy::ByFrame);
    simaai::neat::Run run = join.build();
    require(run.push("left", simaai::neat::Sample{sample_with_frame(101, "cam-left", 0x11)}),
            "Combine ByFrame left push failed: " + run.last_error());
    require(run.push("right", simaai::neat::Sample{sample_with_frame(101, "cam-right", 0x22)}),
            "Combine ByFrame right push failed: " + run.last_error());
    auto out = run.pull("combined", 5000);
    require(out.has_value(), "Combine ByFrame pull timed out");
    require_bundle_fields(*out, "Combine ByFrame");
    require(out->frame_id == 101, "Combine ByFrame should preserve frame_id");
    run.close();
  }

  {
    simaai::neat::Graph join = simaai::neat::graphs::Combine({"left", "right"}, "combined",
                                                             simaai::neat::CombinePolicy::ByPts);
    simaai::neat::Run run = join.build();
    require(run.push("left", simaai::neat::Sample{sample_with_pts(123456, "cam-left", 0x33)}),
            "Combine ByPts left push failed: " + run.last_error());
    require(run.push("right", simaai::neat::Sample{sample_with_pts(123456, "cam-right", 0x44)}),
            "Combine ByPts right push failed: " + run.last_error());
    auto out = run.pull("combined", 5000);
    require(out.has_value(), "Combine ByPts pull timed out: " + run.last_error());
    require_bundle_fields(*out, "Combine ByPts");
    require(out->pts_ns == 123456, "Combine ByPts should preserve pts_ns");
    run.close();
  }

  {
    simaai::neat::Graph rr = simaai::neat::graphs::Combine({"left", "right"}, "combined",
                                                           simaai::neat::CombinePolicy::RoundRobin);
    simaai::neat::Run run = rr.build();
    simaai::neat::Sample left = sample_with_frame(201, "", 0x71);
    simaai::neat::Sample right = sample_with_frame(202, "", 0x72);
    require(run.push("left", simaai::neat::Sample{std::move(left)}),
            "Combine RoundRobin left push failed: " + run.last_error());
    require(run.push("right", simaai::neat::Sample{std::move(right)}),
            "Combine RoundRobin right push failed: " + run.last_error());

    auto first = run.pull("combined", 5000);
    require(first.has_value(), "Combine RoundRobin first pull timed out: " + run.last_error());
    auto second = run.pull("combined", 5000);
    require(second.has_value(), "Combine RoundRobin second pull timed out: " + run.last_error());
    require(first->kind != simaai::neat::SampleKind::Bundle,
            "Combine RoundRobin should forward original samples, not bundles");
    require(second->kind != simaai::neat::SampleKind::Bundle,
            "Combine RoundRobin should forward original samples, not bundles");
    require(first->stream_id == "left", "Combine RoundRobin should stamp the left stream id");
    require(second->stream_id == "right", "Combine RoundRobin should stamp the right stream id");
    require(first->frame_id == 201, "Combine RoundRobin should preserve first frame_id");
    require(second->frame_id == 202, "Combine RoundRobin should preserve second frame_id");
    run.close();
  }

  {
    simaai::neat::Graph rr = simaai::neat::graphs::Combine({"left", "right"}, "combined",
                                                           simaai::neat::CombinePolicy::RoundRobin);
    simaai::neat::Run run = rr.build();
    require(run.push("left", simaai::neat::Sample{sample_with_frame(221, "cam-left", 0x77)}),
            "Combine RoundRobin left original stream push failed: " + run.last_error());
    require(run.push("right", simaai::neat::Sample{sample_with_frame(222, "cam-right", 0x78)}),
            "Combine RoundRobin right original stream push failed: " + run.last_error());

    auto first = run.pull("combined", 5000);
    require(first.has_value(),
            "Combine RoundRobin original-stream first pull timed out: " + run.last_error());
    auto second = run.pull("combined", 5000);
    require(second.has_value(),
            "Combine RoundRobin original-stream second pull timed out: " + run.last_error());
    require(first->stream_id == "cam-left",
            "Combine RoundRobin should preserve a non-empty original left stream_id");
    require(second->stream_id == "cam-right",
            "Combine RoundRobin should preserve a non-empty original right stream_id");
    run.close();
  }

  {
    simaai::neat::Graph rr = simaai::neat::graphs::Combine({"left", "right"}, "combined",
                                                           simaai::neat::CombinePolicy::RoundRobin);
    simaai::neat::Run run = rr.build();
    simaai::neat::Sample left = sample_with_frame(-1, "", 0x75);
    simaai::neat::Sample right = sample_with_frame(-1, "", 0x76);
    left.pts_ns = -1;
    right.pts_ns = -1;
    require(run.push("left", simaai::neat::Sample{std::move(left)}),
            "Combine RoundRobin left no-key push failed: " + run.last_error());
    require(run.push("right", simaai::neat::Sample{std::move(right)}),
            "Combine RoundRobin right no-key push failed: " + run.last_error());

    auto first = run.pull("combined", 5000);
    require(first.has_value(),
            "Combine RoundRobin no-key first pull timed out: " + run.last_error());
    auto second = run.pull("combined", 5000);
    require(second.has_value(),
            "Combine RoundRobin no-key second pull timed out: " + run.last_error());
    require(first->kind != simaai::neat::SampleKind::Bundle,
            "Combine RoundRobin no-key should not materialize bundles");
    require(second->kind != simaai::neat::SampleKind::Bundle,
            "Combine RoundRobin no-key should not materialize bundles");
    require(first->stream_id == "left", "Combine RoundRobin no-key should stamp left stream id");
    require(second->stream_id == "right", "Combine RoundRobin no-key should stamp right stream id");
    require(first->frame_id < 0 && first->pts_ns < 0,
            "Combine RoundRobin no-key should not require frame_id/pts on first sample");
    require(second->frame_id < 0 && second->pts_ns < 0,
            "Combine RoundRobin no-key should not require frame_id/pts on second sample");
    run.close();
  }

  {
    simaai::neat::Graph source_left("left");
    source_left.add(simaai::neat::nodes::Input("left"));
    simaai::neat::Graph source_right("right");
    source_right.add(simaai::neat::nodes::Input("right"));
    simaai::neat::Graph rr = simaai::neat::graphs::Combine({"left", "right"}, "combined",
                                                           simaai::neat::CombinePolicy::RoundRobin);
    simaai::neat::Graph sink("combined");
    sink.add(simaai::neat::nodes::Input("combined"));
    sink.add(simaai::neat::nodes::Output("out"));

    simaai::neat::Graph app("connected_round_robin");
    app.connect(source_left, rr);
    app.connect(source_right, rr);
    app.connect(rr, sink);

    const std::string backend = app.describe_backend(false);
    require_contains(backend, "StreamScheduler",
                     "Connected RoundRobin should lower through StreamScheduler");
    require(backend.find("JoinBundle") == std::string::npos,
            "Connected RoundRobin must not lower through JoinBundle");

    simaai::neat::Run run = app.build();
    require(run.push("left", simaai::neat::Sample{sample_with_frame(211, "", 0x73)}),
            "Connected RoundRobin left push failed: " + run.last_error());
    require(run.push("right", simaai::neat::Sample{sample_with_frame(212, "", 0x74)}),
            "Connected RoundRobin right push failed: " + run.last_error());
    auto first = run.pull("out", 5000);
    require(first.has_value(), "Connected RoundRobin first pull timed out: " + run.last_error());
    auto second = run.pull("out", 5000);
    require(second.has_value(), "Connected RoundRobin second pull timed out: " + run.last_error());
    require(first->stream_id == "left", "Connected RoundRobin should preserve left routing");
    require(second->stream_id == "right", "Connected RoundRobin should preserve right routing");
    require(first->kind != simaai::neat::SampleKind::Bundle,
            "Connected RoundRobin should not materialize bundles");
    require(second->kind != simaai::neat::SampleKind::Bundle,
            "Connected RoundRobin should not materialize bundles");
    run.close();
  }

  {
    // Regression: a source/branch/round-robin chain must use the logical Branch/Combine endpoint
    // names after public boundary elision.  Otherwise physically identical source tails such as
    // CapsRaw collapse to duplicate fan-in names ("capsraw", "capsraw", ...).
    simaai::neat::Graph rr = simaai::neat::graphs::Combine({"stream0", "stream1"}, "detector",
                                                           simaai::neat::CombinePolicy::RoundRobin);
    simaai::neat::Graph detector_sink("detector");
    detector_sink.add(simaai::neat::nodes::Input("detector"));
    detector_sink.add(simaai::neat::nodes::Output("detections"));

    simaai::neat::Graph app("branched_source_round_robin");
    for (int i = 0; i < 2; ++i) {
      const std::string stream = "stream" + std::to_string(i);
      const std::string preview = "preview" + std::to_string(i);
      simaai::neat::Graph source("source_graph_" + std::to_string(i));
      source.add(
          simaai::neat::nodes::Custom("videotestsrc is-live=true num-buffers=1 ! "
                                      "video/x-raw,format=NV12,width=16,height=16,framerate=1/1",
                                      simaai::neat::InputRole::Source));
      source.add(simaai::neat::nodes::CapsRaw("NV12", 16, 16, 1, simaai::neat::CapsMemory::Any));

      simaai::neat::Graph branch =
          simaai::neat::graphs::Branch("source" + std::to_string(i), {stream, preview});

      simaai::neat::Graph preview_sink(preview);
      preview_sink.add(simaai::neat::nodes::Input(preview));
      preview_sink.add(simaai::neat::nodes::Output(preview + "_out"));

      simaai::neat::GraphLinkOptions link;
      link.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
      link.queue_depth = 1;
      link.stream_id = stream;

      app.connect(source, branch, link);
      app.connect(branch, rr, link);
      app.connect(branch, preview_sink, link);
    }
    app.connect(rr, detector_sink);

    require_throws_contains([&] { (void)app.describe_backend(false); },
                            "appsrc fallback is disabled",
                            "realtime branched RoundRobin must not use segmented appsrc transport");
  }

  {
    // Logical endpoint names are part of edge identity even when both paths share the same
    // FanOut source node, target node, physical ports, and link options.
    simaai::neat::Graph source("shared_source");
    source.add(simaai::neat::nodes::Custom(
        "videotestsrc num-buffers=1 ! video/x-raw,format=NV12,width=16,height=16,framerate=1/1",
        simaai::neat::InputRole::Source));
    source.add(simaai::neat::nodes::CapsRaw("NV12", 16, 16, 1, simaai::neat::CapsMemory::Any));
    simaai::neat::Graph branch = simaai::neat::graphs::Branch("source", {"left_out", "right_out"});
    simaai::neat::Graph rr = simaai::neat::graphs::Combine({"left_in", "right_in"}, "combined",
                                                           simaai::neat::CombinePolicy::RoundRobin);
    simaai::neat::Graph sink("combined");
    sink.add(simaai::neat::nodes::Input("combined"));
    sink.add(simaai::neat::nodes::Output("out"));

    simaai::neat::Graph app("same_runtime_node_endpoint_round_robin");
    app.connect(rr, sink);
    app.connect(source, branch);
    app.connect("left_out", "left_in");
    app.connect("right_out", "right_in");

    const std::string backend = app.describe_backend(false);
    require_contains(backend, "StreamScheduler",
                     "Endpoint-distinct edges should retain RoundRobin fan-in");
    require_contains(backend, ":left_in", "RoundRobin should retain the left logical endpoint");
    require_contains(backend, ":right_in", "RoundRobin should retain the right logical endpoint");
  }

  {
    simaai::neat::Graph join = simaai::neat::graphs::Combine({"left", "right"}, "combined",
                                                             simaai::neat::CombinePolicy::ByFrame);
    simaai::neat::Run run = join.build();
    simaai::neat::Sample sample = sample_with_pts(777777, "cam-left", 0x50);
    sample.frame_id = -1;
    require_strict_combine_error(run, "left", std::move(sample),
                                 "missing frame_id for strict ByFrame combine",
                                 "Combine ByFrame must not fall back to PTS");
    run.close();
  }

  {
    simaai::neat::Graph join = simaai::neat::graphs::Combine({"left", "right"}, "combined",
                                                             simaai::neat::CombinePolicy::ByPts);
    simaai::neat::Run run = join.build();
    simaai::neat::Sample sample = sample_with_frame(303, "cam-left", 0x60);
    sample.pts_ns = -1;
    require_strict_combine_error(run, "left", std::move(sample),
                                 "missing pts_ns for strict ByPts combine",
                                 "Combine ByPts must not fall back to frame_id");
    run.close();
  }

  {
    simaai::neat::Graph join;
    join.add(simaai::neat::nodes::Input("left"));
    join.add(simaai::neat::nodes::Input("right"));
    join.add(simaai::neat::nodes::Output("combined"));
    join.connect("left", "combined");
    join.connect("right", "combined");
    require_throws_contains([&] { (void)join.build(); }, "CombinePolicy::ByFrame",
                            "CombinePolicy::None should fail for multiple producers");
  }
});
