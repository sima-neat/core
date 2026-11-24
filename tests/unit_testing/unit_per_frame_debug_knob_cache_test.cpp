// Debug knobs read on per-frame paths are cached: a raw environment change after the first read
// does not reach a running graph, but a runtime verbosity change (VerboseOptions) does.
#include "graph/internal/GraphRunState.h"
#include "pipeline/GraphOptions.h"
#include "pipeline/internal/UxLogging.h"
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

           auto verbose = simaai::neat::VerboseOptions::production();
           verbose.graph = true;
           auto lease = simaai::neat::pipeline_internal::ux::acquire_runtime_verbosity(verbose);
           require(graph_sched_debug_enabled(), "a graph verbosity lease enables graph knobs");
           lease.reset();
           require(!graph_sched_debug_enabled(), "releasing the lease disables graph knobs");
         }));
