#include "graph_migration/common/phase3_graph_test_utils.h"
#include "graphs/Fragments.h"
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
        [] {
          (void)simaai::neat::graphs::Combine({"left", "left"}, "combined");
        },
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
