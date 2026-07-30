#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif

#include "graphs/Fragments.h"
#include "nodes/common/Output.h"
#include "nodes/common/Queue.h"
#include "nodes/io/Input.h"
#include "pipeline/Graph.h"
#include "pipeline/internal/Diagnostics.h"
#include "pipeline/runtime/RunCore.h"
#include "test_main.h"
#include "test_utils.h"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

// What the boundary identity probes saw at one materialized pipeline segment. The PTS fields are
// raw GstBuffer header values, which the appsink-side Sample reconstruction would otherwise mask by
// restoring PTS from Neat meta. The buffer counts make the PTS observation non-vacuous: a stale
// initial value cannot be mistaken for an observed timeline.
struct SegmentBoundaryPts {
  std::uint64_t in_buffers = 0;   ///< Buffers into the first boundary, i.e. out of the appsrc.
  std::uint64_t out_buffers = 0;  ///< Buffers out of the last boundary of the segment.
  std::int64_t first_in_pts = -1; ///< Last PTS the segment's own appsrc emitted.
  std::int64_t last_out_pts = -1; ///< Last PTS leaving the segment.
};

SegmentBoundaryPts segment_boundary_pts(const simaai::neat::runtime::PipelineSegmentRuntime& pipe,
                                        const std::string& where) {
  const auto diag = pipe.run_core ? pipe.run_core->pipeline.stream.diag_ctx() : nullptr;
  require(diag != nullptr, where + ": segment has no diagnostics context");
  require(!diag->boundaries.empty(),
          where + ": no boundary probes were inserted; the PTS observation is vacuous");

  SegmentBoundaryPts out;
  out.in_buffers = diag->boundaries.front()->in_buffers.load();
  out.out_buffers = diag->boundaries.back()->out_buffers.load();
  out.first_in_pts = diag->boundaries.front()->last_in_pts_ns.load();
  out.last_out_pts = diag->boundaries.back()->last_out_pts_ns.load();
  require(out.in_buffers > 0U, where + ": no buffer reached the first boundary");
  require(out.out_buffers > 0U, where + ": no buffer left the last boundary");
  return out;
}

bool has_injected_input(const simaai::neat::runtime::PipelineSegmentPlan& seg) {
  return !seg.materialized_node_attribution.empty() &&
         seg.materialized_node_attribution.front().role ==
             simaai::neat::runtime::MaterializedNodeAttribution::Role::InjectedInput;
}

// `pts_ns < 0` produces a frame with no presentation timestamp of its own, which is the state
// decoded video frames are in when they reach an internal boundary.
simaai::neat::Sample make_frame(int frame_id, std::int64_t pts_ns) {
  simaai::neat::Sample s;
  s.kind = simaai::neat::SampleKind::Tensor;
  s.tensor = make_color_tensor(64, 48, simaai::neat::ImageSpec::PixelFormat::RGB,
                               static_cast<std::uint8_t>(0x30 + frame_id));
  s.frame_id = frame_id;
  s.pts_ns = pts_ns;
  s.stream_id = "pts-probe";
  return s;
}

simaai::neat::Graph leg_graph(const std::string& input, const std::string& output) {
  simaai::neat::Graph g(input);
  g.add(simaai::neat::nodes::Input(input));
  g.add(simaai::neat::nodes::Queue());
  g.add(simaai::neat::nodes::Output(output));
  return g;
}

// A source fan-out lowered into several pipeline segments. Each leg is fed by a Core-injected
// boundary appsrc, so the legs of one logical stream share a timeline only as long as no boundary
// invents one of its own.
simaai::neat::Run build_fanout_run(bool public_do_timestamp = false) {
  using namespace simaai::neat;

  InputOptions src_opt;
  src_opt.do_timestamp = public_do_timestamp;

  Graph source("image");
  source.add(nodes::Input("image", src_opt));
  source.add(nodes::Queue());

  Graph branch = graphs::Branch("image", {"leg_a", "leg_b"});
  Graph leg_a = leg_graph("leg_a", "out_a");
  Graph leg_b = leg_graph("leg_b", "out_b");

  Graph app("internal_boundary_pts");
  app.connect(source, branch);
  app.connect(branch, leg_a);
  app.connect(branch, leg_b);
  return app.build(RunOptions{});
}

struct FanOutBoundaries {
  SegmentBoundaryPts source;
  std::vector<SegmentBoundaryPts> legs;
};

FanOutBoundaries fanout_boundaries(const simaai::neat::Run& run, const std::string& where) {
  using namespace simaai::neat;
  const auto core = run_internal::core(run);
  const auto& exec = core->graph_execution();

  FanOutBoundaries out;
  bool have_source = false;
  for (const auto& pipe : exec.pipelines) {
    require(pipe != nullptr, where + ": missing segment runtime");
    const auto pts =
        segment_boundary_pts(*pipe, where + " segment " + std::to_string(pipe->seg.id));
    if (has_injected_input(pipe->seg)) {
      out.legs.push_back(pts);
      continue;
    }
    require(!have_source, where + ": expected exactly one source segment");
    out.source = pts;
    have_source = true;
  }
  require(have_source, where + ": no source segment was materialized");
  require(out.legs.size() == 2U,
          where + ": expected the Branch fan-out to lower into two injected-input segments, got " +
              std::to_string(out.legs.size()));
  return out;
}

// A Core-injected boundary transports the timeline it was handed; it never authors one. Pushes
// `frames` frames starting at `base_pts` (`-1` pushes frames with no PTS at all), then requires
// every injected boundary to emit exactly what the source segment emitted.
// `expected_source_pts` pins the premise, so a leg assertion cannot pass because the source
// timeline silently changed underneath it.
void internal_boundary_carries_source_timeline(std::int64_t base_pts,
                                               std::int64_t expected_source_pts,
                                               const std::string& where) {
  using namespace simaai::neat;

  const int frames = 5;
  Run run = build_fanout_run();
  for (int i = 0; i < frames; ++i) {
    const std::int64_t pts = base_pts < 0 ? -1 : base_pts + i * 33000000LL;
    require(run.push("image", make_frame(i, pts)), where + ": push failed: " + run.last_error());
    require(run.pull("out_a", 5000).has_value(),
            where + ": out_a pull timed out: " + run.last_error());
    require(run.pull("out_b", 5000).has_value(),
            where + ": out_b pull timed out: " + run.last_error());
  }

  const FanOutBoundaries boundaries = fanout_boundaries(run, where);
  require(boundaries.source.last_out_pts == expected_source_pts,
          where + ": source segment should emit PTS " + std::to_string(expected_source_pts) +
              " got " + std::to_string(boundaries.source.last_out_pts));
  for (const auto& leg : boundaries.legs) {
    require(leg.first_in_pts == boundaries.source.last_out_pts,
            where + ": injected boundary re-stamped the buffer, expected " +
                std::to_string(boundaries.source.last_out_pts) + " got " +
                std::to_string(leg.first_in_pts));
  }
}

// The public contract is unchanged: an application-owned Input still stamps a pushed frame that
// carries no PTS of its own.
void public_input_timestamps_untimed_frame() {
  using namespace simaai::neat;

  Graph app("public_input_pts");
  app.add(nodes::Input("image"));
  app.add(nodes::Queue());
  app.add(nodes::Output("out"));

  Run run = app.build(RunOptions{});
  for (int i = 0; i < 3; ++i) {
    require(run.push("image", make_frame(i, -1)),
            "public input PTS: push failed: " + run.last_error());
    require(run.pull("out", 5000).has_value(),
            "public input PTS: pull timed out: " + run.last_error());
  }

  const auto core = run_internal::core(run);
  const auto& exec = core->graph_execution();
  require(exec.pipelines.size() == 1U, "public input PTS: expected a single pipeline segment");
  const auto pts = segment_boundary_pts(*exec.pipelines.front(), "public input PTS segment");

  // Without do_timestamp the untimed buffer would leave appsrc at PTS 0.
  require(pts.first_in_pts > 0,
          "public input PTS: a public Input must stamp a pushed frame that carries no PTS, got " +
              std::to_string(pts.first_in_pts));
}

// A timeline authored by a public Input is still a source timeline once it exists, so the
// boundaries must carry it rather than replace it. The buffer crosses zero-copy, so the PTS the
// public appsrc stamped is the one every leg has to see.
void public_input_timeline_survives_internal_boundary() {
  using namespace simaai::neat;

  const std::string where = "public input across boundary";
  Run run = build_fanout_run(true);
  for (int i = 0; i < 5; ++i) {
    require(run.push("image", make_frame(i, -1)), where + ": push failed: " + run.last_error());
    require(run.pull("out_a", 5000).has_value(),
            where + ": out_a pull timed out: " + run.last_error());
    require(run.pull("out_b", 5000).has_value(),
            where + ": out_b pull timed out: " + run.last_error());
  }

  const FanOutBoundaries boundaries = fanout_boundaries(run, where);
  require(boundaries.source.last_out_pts > 0,
          where + ": a public Input must stamp an untimed frame, got " +
              std::to_string(boundaries.source.last_out_pts));
  for (const auto& leg : boundaries.legs) {
    require(leg.first_in_pts == boundaries.source.last_out_pts,
            where + ": injected boundary lost the generated timeline, expected " +
                std::to_string(boundaries.source.last_out_pts) + " got " +
                std::to_string(leg.first_in_pts));
  }
}

} // namespace

RUN_TEST("unit_graph_internal_boundary_pts_test", ([] {
           setenv("SIMA_GST_RUN_INSERT_BOUNDARIES", "1", 1);
           setenv("SIMA_GST_BOUNDARY_PROBES", "1", 1);

           // An untimed frame must stay untimed. appsrc emits PTS 0 for a buffer that carries no
           // timestamp, so a boundary left at do-timestamp=true replaces that with its own
           // per-segment pipeline running time and the legs of one logical stream end up on
           // different clocks. This is the state decoded video frames cross the boundary in.
           internal_boundary_carries_source_timeline(-1, 0, "internal boundary untimed");

           // One hour of source time, so a buffer re-stamped with running time is unmistakable.
           const std::int64_t base_pts = 3600LL * 1000000000LL;
           internal_boundary_carries_source_timeline(base_pts, base_pts + 4 * 33000000LL,
                                                     "internal boundary timed");

           public_input_timestamps_untimed_frame();
           public_input_timeline_survives_internal_boundary();
         }));
