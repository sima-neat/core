// Graph.h under SIMA_NEAT_GRAPH_NO_GSTREAMER (H0 C2, decision D10): the public Graph header, and
// everything it includes, compiles without GStreamer's headers. The macro is defined before the
// first include of this translation unit, and the checks below fail the compile if any GStreamer
// header was reached anyway. The gate also compiles this file with GStreamer's include directories
// removed from the command line (c2.md, gates).
#define SIMA_NEAT_GRAPH_NO_GSTREAMER 1
#include "pipeline/Graph.h"

#if defined(__GST_H__) || defined(GST_API) || defined(GST_VERSION_MAJOR)
#error "pipeline/Graph.h reached GStreamer's headers under SIMA_NEAT_GRAPH_NO_GSTREAMER"
#endif

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

int main() {
  // The header is usable as declared: a Graph built from it, without a Run (no backend involved).
  simaai::neat::Graph g;
  const auto in = g.input("image");
  g.output("out", in);
  const bool ok = g.inputs() == std::vector<std::string>{"image"} &&
                  g.outputs() == std::vector<std::string>{"out"};
  std::cout << (ok ? "[OK] unit_graph_header_no_gstreamer_test passed\n"
                   : "[FAIL] Graph::input/output endpoints\n");
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
