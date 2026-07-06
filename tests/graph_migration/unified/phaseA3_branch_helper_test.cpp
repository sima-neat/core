#include "graph_migration/common/phase3_graph_test_utils.h"
#include "graphs/Fragments.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "test_main.h"
#include "test_utils.h"

#include <functional>
#include <string>

namespace {

simaai::neat::Graph input_graph(const std::string& name) {
  simaai::neat::Graph g(name);
  g.add(simaai::neat::nodes::Input(name));
  return g;
}

simaai::neat::Graph output_graph(const std::string& name) {
  simaai::neat::Graph g(name);
  g.add(simaai::neat::nodes::Output(name));
  return g;
}

simaai::neat::Graph passthrough_graph(const std::string& input, const std::string& output) {
  simaai::neat::Graph g(input);
  g.add(simaai::neat::nodes::Input(input));
  g.add(simaai::neat::nodes::Output(output));
  return g;
}

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

} // namespace

RUN_TEST("graph_migration_phaseA3_branch_helper_test", [] {
  {
    require_throws_contains(
        [] { (void)simaai::neat::graphs::Branch("image", {"preview", "preview"}); },
        "duplicate endpoint name", "Branch helper should reject duplicate output names");
  }

  {
    auto source = input_graph("image");
    auto branch = simaai::neat::graphs::Branch("image", {"preview", "model"});
    auto preview = output_graph("preview");
    auto model = output_graph("model");

    simaai::neat::Graph app;
    app.connect(source, branch);
    app.connect(branch, preview);
    app.connect(branch, model);

    simaai::neat::Run run = app.build();
    require(run.push("image", simaai::neat::Sample{graph_phase3_test::make_tensor_sample(
                                  11, "branch-helper")}),
            "Branch helper push failed: " + run.last_error());

    auto preview_sample = run.pull("preview", 5000);
    require(preview_sample.has_value(), "Branch helper preview pull timed out");
    require(preview_sample->stream_id == "branch-helper",
            "Branch helper preview stream_id mismatch");

    auto model_sample = run.pull("model", 5000);
    require(model_sample.has_value(), "Branch helper model pull timed out");
    require(model_sample->stream_id == "branch-helper", "Branch helper model stream_id mismatch");
    run.close();
  }

  {
    auto source = input_graph("source");
    auto branch = simaai::neat::graphs::Branch("source", {"detector_h264", "video_h264"});
    auto detector = passthrough_graph("detector_h264", "detections");
    auto video = passthrough_graph("video_h264", "video");

    simaai::neat::Graph app;
    app.connect(source, branch);

    simaai::neat::GraphLinkOptions realtime;
    realtime.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
    realtime.queue_depth = 1;
    realtime.stream_id = "stream0";
    app.connect(branch, detector, realtime);
    app.connect(branch, video, realtime);

    const auto report = app.validate();
    require(report.error_code.empty(),
            "Branch boundary elision with realtime fanout should validate, got " +
                report.error_code + ": " + report.repro_note);
  }
});
