#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif

#include "pipeline/GraphOptions.h"
#include "pipeline/graph/internal/GraphBuildInternal.h"
#include "test_main.h"

#include <stdexcept>
#include <string>

namespace {

void require_contains(const std::string& value, const std::string& token, const char* message) {
  if (value.find(token) == std::string::npos) {
    throw std::runtime_error(std::string(message) + " (missing: " + token + ")");
  }
}

} // namespace

RUN_TEST("unit_mla_workload_priority_options_test", ([] {
           using simaai::neat::AdvancedExecutionOptions;
           using simaai::neat::GraphOptions;
           using simaai::neat::WorkloadPriority;

           /*
            * Normal is explicit in the rendered graph.  Relying on the plugin's
            * ambient default would make a saved/inspected graph incomplete and
            * could silently change scheduling if that default ever moved.
            */
           GraphOptions normal;
           const std::string normal_fragment =
               simaai::neat::session_build_apply_fast_path_options_to_fragment(
                   "neatprocessmla name=mla", &normal);
           require_contains(normal_fragment, "workload-priority=normal",
                            "default MLA workload priority must render explicitly");

           GraphOptions background;
           background.processmla.workload_priority = WorkloadPriority::Background;
           const std::string background_fragment =
               simaai::neat::session_build_apply_fast_path_options_to_fragment(
                   "identity name=before ! neatprocessmla name=mla ! fakesink", &background);
           require_contains(background_fragment,
                            "neatprocessmla name=mla async=true defer-output-invalidate=true "
                            "workload-priority=background",
                            "background intent must project only onto ProcessMLA");
           if (background_fragment.find("identity name=before workload-priority") !=
               std::string::npos) {
             throw std::runtime_error("MLA priority leaked onto an unrelated graph segment");
           }

           /*
            * New applications use the intent-named advanced surface.  It must win
            * over the raw compatibility field, and route overlay must win over the
            * model value, without inventing another scheduler policy.
            */
           GraphOptions resolved;
           resolved.processmla.workload_priority = WorkloadPriority::Background;
           resolved.processmla.async = false;
           resolved.advanced_execution.inference_async = false;
           resolved.advanced_execution.workload_priority = WorkloadPriority::Foreground;
           resolved.resolve_advanced_execution();
           if (resolved.processmla.async) {
             throw std::runtime_error(
                 "deprecated inference_async must not mutate compatibility storage");
           }
           const std::string foreground_fragment =
               simaai::neat::session_build_apply_fast_path_options_to_fragment(
                   "neatprocessmla name=mla workload-priority=normal", &resolved);
           require_contains(foreground_fragment, "async=true",
                            "deprecated async=false must still render the sole direct path");
           require_contains(foreground_fragment, "workload-priority=foreground",
                            "advanced MLA priority must override the compatibility value");
           if (foreground_fragment.find("workload-priority=normal") != std::string::npos) {
             throw std::runtime_error("priority projection left a stale duplicate property");
           }

           AdvancedExecutionOptions model;
           model.workload_priority = WorkloadPriority::Background;
           AdvancedExecutionOptions route;
           route.workload_priority = WorkloadPriority::Foreground;
           model.overlay(route);
           if (model.workload_priority != WorkloadPriority::Foreground) {
             throw std::runtime_error("route MLA priority did not override model priority");
           }
         }));
