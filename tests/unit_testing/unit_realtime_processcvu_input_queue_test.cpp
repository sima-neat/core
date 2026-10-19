#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif

#include "pipeline/Run.h"
#include "nodes/common/Queue.h"
#include "pipeline/graph/internal/GraphBuildInternal.h"
#include "test_main.h"
#include "test_utils.h"

#include <string>

namespace {

const std::string kPipeline = "rtspsrc latency=100 ! neatdecoder ! queue leaky=downstream "
                              "max-size-buffers=2 ! neatkernelrunner name=preproc async=true ! "
                              "neatprocessmla name=mla ! queue max-size-time=5000000 ! "
                              "neatkernelrunner name=post ! appsink";

std::string apply_preset(simaai::neat::RunPreset preset, const std::string& pipeline = kPipeline) {
  simaai::neat::RunOptions opt;
  opt.preset = preset;
  return simaai::neat::session_build_apply_run_preset_to_pipeline(pipeline, opt);
}

} // namespace

RUN_TEST(
    "unit_realtime_processcvu_input_queue_test", ([] {
      using simaai::neat::RunPreset;

      const std::string realtime = apply_preset(RunPreset::Realtime);
      require_contains(
          realtime, "queue leaky=downstream max-size-buffers=2 max-size-time=20000000 ! ",
          "Realtime bounds only the existing unsent queue and preserves its count policy");
      require_contains(realtime, "queue max-size-time=5000000 ! neatkernelrunner name=post",
                       "an explicitly rendered value must win over the preset");
      require(realtime.find("max-input-queue-time") == std::string::npos,
              "canonical stages must not acquire a retired aggregator property");
      require_contains(realtime,
                       "neatkernelrunner name=preproc async=true ! neatprocessmla name=mla !",
                       "accepted backend jobs keep their existing capacity and retirement policy");

      for (const std::string explicit_value : {"0", "5000000"}) {
        const auto spaced =
            "appsrc ! queue max-size-time = " + explicit_value + " ! neatkernelrunner ! appsink";
        require(apply_preset(RunPreset::Realtime, spaced) == spaced,
                "an explicit value including zero wins even with spaces around '='");
      }
      const std::string quoted =
          "filesrc location=\"/tmp/! queue !fixture\" ! queue name=user max-size-time=0 ! appsink";
      require(apply_preset(RunPreset::Realtime, quoted) == quoted,
              "quoted link characters are data, never queue topology");
      // Gst's assignment lexer is maximal-munch, not a shell quote toggle.
      for (const std::string path : {"/tmp/!queue", "https://host/!queue", "\\!queue"}) {
        const std::string source = "filesrc location=" + path + " ! appsink";
        require(apply_preset(RunPreset::Realtime, source) == source,
                "an unquoted assignment link character must not create a queue");
      }
      for (const char quote : {'\"', '\''}) {
        const std::string source =
            "filesrc location=" + std::string(1, quote) + "/tmp/unmatched ! queue ! appsink";
        require(apply_preset(RunPreset::Realtime, source) ==
                    "filesrc location=" + std::string(1, quote) +
                        "/tmp/unmatched ! queue max-size-time=20000000 ! appsink",
                "an unmatched literal quote must not hide a real queue link");
      }
      const std::string capped =
          "appsrc ! application/x-test,field=(string)\"a\\!queue\" ! queue ! appsink";
      require(apply_preset(RunPreset::Realtime, capped) ==
                  "appsrc ! application/x-test,field=(string)\"a\\!queue\" ! "
                  "queue max-size-time=20000000 ! appsink",
              "caps link tokens are opaque and must retain their exact bytes");

      const std::vector<simaai::neat::LaunchNameOrigin> origins{
          {.kind = simaai::neat::LaunchNameOrigin::Kind::Queue, .name = "generated"}};
      simaai::neat::RunOptions realtime_options;
      realtime_options.preset = RunPreset::Realtime;
      const auto spaced_default = simaai::neat::session_build_apply_run_preset_to_pipeline(
          "queue name=generated max-size-time = \"0\" ! appsink", realtime_options, origins);
      require(spaced_default == "queue name=generated max-size-time = \"20000000\" ! appsink",
              "replace the generated value span without duplicate properties or quoting changes");

      const auto lane_window = simaai::neat::session_build_propagate_terminal_consumer_lane_window(
          "neatprocessmla name=p num-buffers = \"4\" ! application/x-test,field=opaque ! "
          "identity note=\"num-buffers=99 !queue\" ! genericterminal num-buffers = \"2\"");
      require(lane_window ==
                  "neatprocessmla name=p num-buffers = \"4\" ! application/x-test,field=opaque ! "
                  "identity note=\"num-buffers=99 !queue\" ! genericterminal num-buffers = \"4\"",
              "terminal lane policy uses the same opaque token/value-span authority");
      for (const std::string topology :
           {"neatprocessmla num-buffers=7 : genericterminal num-buffers=2",
            "( neatprocessmla num-buffers=7 ! genericterminal num-buffers=2 )",
            "neatprocessmla num-buffers=7 ; genericterminal num-buffers=2"}) {
        require(simaai::neat::session_build_propagate_terminal_consumer_lane_window(topology) ==
                    topology,
                "nontrivial topology cannot grant a lexical lane-adjacency proof");
      }
      const std::string undeclared =
          "neatprocessmla num-buffers=4 ! genericterminal note=\"num-buffers=2\"";
      require(simaai::neat::session_build_propagate_terminal_consumer_lane_window(undeclared) ==
                  undeclared,
              "a property-like substring is not a declared terminal lane-window contract");

      const std::string serial = "appsrc ! neatkernelrunner ! neatprocessmla ! appsink";
      require(apply_preset(RunPreset::Realtime, serial) == serial,
              "Realtime must not invent an extra queue or rewrite bounded execution slots");
      const std::string reliable = "appsrc ! queue leaky=no max-size-buffers=3 ! appsink";
      require_contains(apply_preset(RunPreset::Realtime, reliable),
                       "queue leaky=no max-size-buffers=3 max-size-time=20000000",
                       "Realtime does not convert an explicitly reliable branch to lossy");

      // Exercise the actual graph queue producer and its provenance, not a
      // hand-written string that accidentally omits the generated time default.
      simaai::neat::GraphOptions graph_options;
      graph_options.async_queue_depth = 3;
      simaai::neat::QueueOptions explicit_options;
      explicit_options.max_buffers = 7;
      explicit_options.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
      const std::vector<std::shared_ptr<simaai::neat::Node>> nodes{
          simaai::neat::nodes::Queue(explicit_options), simaai::neat::nodes::Queue()};
      const auto built =
          simaai::neat::build_pipeline_full(nodes, false, "mysink", true, {}, &graph_options);
      simaai::neat::RunOptions options;
      options.preset = RunPreset::Realtime;
      const auto generated = simaai::neat::session_build_apply_run_preset_to_pipeline(
          built.pipeline_string, options, built.framework_name_origins);
      require_contains(generated,
                       "queue name=queue_neat_async_0 max-size-buffers=3 max-size-bytes=0 "
                       "max-size-time=20000000",
                       "actual framework queue receives Realtime budget and preserves its count");
      require_contains(generated,
                       "queue name=n0_queue max-size-buffers=7 max-size-bytes=0 "
                       "max-size-time=0 leaky=downstream",
                       "public count-only QueueOptions remains an explicit override");
      const auto no_authority =
          simaai::neat::session_build_apply_run_preset_to_pipeline(built.pipeline_string, options);
      require_contains(no_authority, "max-size-buffers=3 max-size-bytes=0 max-size-time=0",
                       "framework-like names alone cannot override customer limits");
      require(simaai::neat::session_build_apply_run_preset_to_pipeline(
                  generated, options, built.framework_name_origins) == generated,
              "cold queue policy is idempotent");
      for (const auto preset : {RunPreset::Balanced, RunPreset::Reliable}) {
        options.preset = preset;
        require(simaai::neat::session_build_apply_run_preset_to_pipeline(
                    built.pipeline_string, options, built.framework_name_origins) ==
                    built.pipeline_string,
                "non-Realtime presets leave actual generated queues unchanged");
      }

      require(apply_preset(RunPreset::Balanced) == kPipeline,
              "Balanced must leave queues unchanged");
      require(apply_preset(RunPreset::Reliable) == kPipeline,
              "Reliable must leave queues unchanged");
    }));
