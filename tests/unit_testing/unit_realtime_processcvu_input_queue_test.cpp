#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif

#include "pipeline/Run.h"
#include "pipeline/graph/internal/GraphBuildInternal.h"
#include "test_main.h"
#include "test_utils.h"

#include <string>

namespace {

const std::string kPipeline = "rtspsrc latency=100 ! neatdecoder ! queue leaky=downstream ! "
                              "neatprocesscvu name=preproc async=true ! neatprocessmla name=mla ! "
                              "neatprocesscvu name=post max-input-queue-time=5000000 ! appsink";

std::string apply_preset(simaai::neat::RunPreset preset, const std::string& pipeline = kPipeline) {
  simaai::neat::RunOptions opt;
  opt.preset = preset;
  return simaai::neat::session_build_apply_run_preset_to_pipeline(pipeline, opt);
}

} // namespace

RUN_TEST("unit_realtime_processcvu_input_queue_test", [] {
  using simaai::neat::RunPreset;

  const std::string realtime = apply_preset(RunPreset::Realtime);
  require_contains(realtime,
                   "neatprocesscvu name=preproc async=true max-input-queue-time=20000000 ! "
                   "neatprocessmla name=mla ! ",
                   "Realtime must bound the ProcessCVU input queue");
  require_contains(realtime, "neatprocesscvu name=post max-input-queue-time=5000000 ! appsink",
                   "an explicitly rendered value must win over the preset");
  require(realtime.find("neatprocessmla name=mla max-input-queue-time") == std::string::npos,
          "only ProcessCVU stages receive the limit");

  const std::string spaced = "appsrc ! neatprocesscvu name=post max-input-queue-time = 5000000 ! "
                             "appsink";
  require(apply_preset(RunPreset::Realtime, spaced) == spaced,
          "an explicit value with spaces around '=' must win over the preset");

  require(apply_preset(RunPreset::Balanced) == kPipeline, "Balanced must leave queues unchanged");
  require(apply_preset(RunPreset::Reliable) == kPipeline, "Reliable must leave queues unchanged");
});
