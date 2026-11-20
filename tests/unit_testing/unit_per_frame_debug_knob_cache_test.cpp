// Debug knobs read on per-frame paths are resolved once per process: the value present at the
// first read stays in effect, so later environment changes do not reach a running graph.
#include "graph/internal/GraphRunState.h"
#include "test_main.h"
#include "test_utils.h"

#include <cstdlib>

RUN_TEST("unit_per_frame_debug_knob_cache_test", ([] {
           using simaai::neat::graph::graph_debug_enabled;
           using simaai::neat::graph::graph_sched_debug_enabled;

           ::setenv("SIMA_GRAPH_DEBUG", "1", 1);
           ::unsetenv("SIMA_GRAPH_SCHED_DEBUG");
           require(graph_debug_enabled(), "SIMA_GRAPH_DEBUG=1 at first read enables it");
           require(!graph_sched_debug_enabled(), "unset SIMA_GRAPH_SCHED_DEBUG reads as off");

           ::setenv("SIMA_GRAPH_DEBUG", "0", 1);
           ::setenv("SIMA_GRAPH_SCHED_DEBUG", "1", 1);
           require(graph_debug_enabled(), "SIMA_GRAPH_DEBUG keeps its first-read value");
           require(!graph_sched_debug_enabled(),
                   "SIMA_GRAPH_SCHED_DEBUG keeps its first-read value");
           ::unsetenv("SIMA_GRAPH_DEBUG");
           ::unsetenv("SIMA_GRAPH_SCHED_DEBUG");
         }));
