#include "graph/GraphRun.h"
#include "model/Model.h"
#include "pipeline/GraphOptions.h"
#include "test_main.h"

RUN_TEST(
    "unit_verbose_options_surface_test", ([] {
      using simaai::neat::GraphOptions;
      using simaai::neat::VerbosityLevel;

      GraphOptions route_opt;
      if (route_opt.verbose.level != VerbosityLevel::Production) {
        throw std::runtime_error("GraphOptions.verbose.level default mismatch");
      }
      if (!route_opt.verbose.progress) {
        throw std::runtime_error("GraphOptions.verbose.progress should default to true");
      }
      if (route_opt.verbose.progress_force) {
        throw std::runtime_error("GraphOptions.verbose.progress_force should default to false");
      }
      if (route_opt.verbose.gstreamer) {
        throw std::runtime_error("GraphOptions.verbose.gstreamer should default to false");
      }
      if (route_opt.verbose.planner || route_opt.verbose.graph || route_opt.verbose.pipeline ||
          route_opt.verbose.inputstream || route_opt.verbose.tensor || route_opt.verbose.plugins) {
        throw std::runtime_error("GraphOptions.verbose detail topics should default to false");
      }

      route_opt.verbose.level = VerbosityLevel::Verbose;
      route_opt.verbose.progress_force = true;
      route_opt.verbose.gstreamer = true;
      route_opt.verbose.planner = true;
      route_opt.verbose.graph = true;
      route_opt.verbose.pipeline = true;
      route_opt.verbose.inputstream = true;
      route_opt.verbose.tensor = true;
      route_opt.verbose.plugins = true;
      if (route_opt.verbose.level != VerbosityLevel::Verbose || !route_opt.verbose.progress_force ||
          !route_opt.verbose.gstreamer || !route_opt.verbose.planner || !route_opt.verbose.graph ||
          !route_opt.verbose.pipeline || !route_opt.verbose.inputstream ||
          !route_opt.verbose.tensor || !route_opt.verbose.plugins) {
        throw std::runtime_error("GraphOptions.verbose mutation failed");
      }

      const auto quiet = simaai::neat::VerboseOptions::quiet();
      if (quiet.level != VerbosityLevel::Quiet || quiet.progress || quiet.progress_force ||
          quiet.gstreamer || quiet.planner || quiet.graph || quiet.pipeline || quiet.inputstream ||
          quiet.tensor || quiet.plugins) {
        throw std::runtime_error("VerboseOptions::quiet preset mismatch");
      }

      const auto production = simaai::neat::VerboseOptions::production();
      if (production.level != VerbosityLevel::Production || !production.progress ||
          production.progress_force || production.gstreamer || production.planner ||
          production.graph || production.pipeline || production.inputstream || production.tensor ||
          production.plugins) {
        throw std::runtime_error("VerboseOptions::production preset mismatch");
      }

      const auto debug_plugins = simaai::neat::VerboseOptions::debug_plugins();
      if (debug_plugins.level != VerbosityLevel::Production || !debug_plugins.progress ||
          debug_plugins.progress_force || !debug_plugins.gstreamer || debug_plugins.planner ||
          debug_plugins.graph || debug_plugins.pipeline || debug_plugins.inputstream ||
          debug_plugins.tensor || !debug_plugins.plugins) {
        throw std::runtime_error("VerboseOptions::debug_plugins preset mismatch");
      }

      const auto debug_all = simaai::neat::VerboseOptions::debug_all();
      if (debug_all.level != VerbosityLevel::Verbose || !debug_all.progress ||
          !debug_all.progress_force || !debug_all.gstreamer || !debug_all.planner ||
          !debug_all.graph || !debug_all.pipeline || !debug_all.inputstream || !debug_all.tensor ||
          !debug_all.plugins) {
        throw std::runtime_error("VerboseOptions::debug_all preset mismatch");
      }

      simaai::neat::graph::GraphRunOptions graph_opt;
      graph_opt.verbose.level = VerbosityLevel::Quiet;
      if (graph_opt.verbose.level != VerbosityLevel::Quiet) {
        throw std::runtime_error("GraphRunOptions.verbose mutation failed");
      }

      simaai::neat::Model::RouteOptions model_opt;
      model_opt.verbose.level = VerbosityLevel::Verbose;
      if (model_opt.verbose.level != VerbosityLevel::Verbose) {
        throw std::runtime_error("Model::RouteOptions.verbose mutation failed");
      }

      simaai::neat::Model::Options model_ctor_opt;
      model_ctor_opt.verbose.planner = true;
      if (!model_ctor_opt.verbose.planner) {
        throw std::runtime_error("Model::Options.verbose mutation failed");
      }
    }));
