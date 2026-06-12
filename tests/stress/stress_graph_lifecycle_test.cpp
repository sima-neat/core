#include "nodes/common/Caps.h"
#include "nodes/common/Output.h"
#include "nodes/common/VideoConvert.h"
#include "pipeline/Graph.h"
#include "test_main.h"
#include "test_utils.h"

#include <algorithm>
#include <cstdlib>
#include <string>

namespace {

int env_int(const char* key, int fallback) {
  const char* raw = std::getenv(key);
  if (!raw || !*raw) {
    return fallback;
  }
  return std::atoi(raw);
}

int clamp_iters(int value) {
  return std::max(10, std::min(value, 4000));
}

} // namespace

RUN_TEST("stress_graph_lifecycle_test", [] {
  const int iters = clamp_iters(env_int("SIMA_STRESS_ITERS", 100));

  for (int i = 0; i < iters; ++i) {
    simaai::neat::Graph graph;
    graph.custom("videotestsrc num-buffers=4 pattern=black", simaai::neat::InputRole::Source);
    graph.add(simaai::neat::nodes::VideoConvert());
    graph.add(simaai::neat::nodes::CapsNV12SysMem(64, 48, 30));
    graph.add(simaai::neat::nodes::Output(simaai::neat::OutputOptions::Latest()));

    simaai::neat::Run run = graph.build();
    int pulled = 0;
    for (int j = 0; j < 4; ++j) {
      auto out = run.pull(1000);
      if (!out.has_value()) {
        break;
      }
      ++pulled;
    }
    require(pulled > 0, "graph lifecycle stress: expected at least one output");
    run.stop();
  }
});
