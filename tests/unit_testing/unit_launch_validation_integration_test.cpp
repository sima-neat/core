#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "nodes/io/StillImageInput.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/Graph.h"
#include "pipeline/NeatError.h"
#include "test_main.h"
#include "test_utils.h"

#include <opencv2/imgcodecs.hpp>

#include <filesystem>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

simaai::neat::Graph make_push_collision_graph() {
  simaai::neat::Graph graph;
  graph.add(simaai::neat::nodes::Input());
  graph.custom("identity name=mysrc");
  graph.add(simaai::neat::nodes::Output());
  return graph;
}

simaai::neat::Graph make_connected_collision_graph() {
  using namespace simaai::neat;
  Graph ingress("image");
  ingress.add(nodes::Input("raw"));
  Graph consumer("image");
  consumer.add(nodes::Input("image"));
  consumer.custom("identity name=mysrc");
  consumer.add(nodes::Output("classes"));

  Graph app;
  app.connect(ingress, consumer);
  return app;
}

void require_pipeline_shape(const simaai::neat::NeatError& error, const std::string& context) {
  require(error.report().error_code == simaai::neat::error_codes::kPipelineShape,
          context + " should preserve misconfig.pipeline_shape");
  require(!error.report().pipeline_string.empty(), context + " should preserve the final launch");
}

} // namespace

RUN_TEST(
    "unit_launch_validation_integration_test", ([] {
      using namespace simaai::neat;
      const cv::Mat input(8, 8, CV_8UC3, cv::Scalar(17, 31, 47));

      {
        Graph graph = make_push_collision_graph();
        bool threw = false;
        try {
          Run run = graph.build(std::vector<cv::Mat>{input});
          run.close();
        } catch (const NeatError& error) {
          threw = true;
          require_pipeline_shape(error, "seeded build");
        }
        require(threw, "seeded Graph::build must validate names automatically");
      }

      {
        Graph graph = make_push_collision_graph();
        bool threw = false;
        try {
          (void)graph.run(std::vector<cv::Mat>{input});
        } catch (const NeatError& error) {
          threw = true;
          require_pipeline_shape(error, "one-shot run");
        }
        require(threw, "one-shot Graph::run must validate names automatically");
      }

      {
        Graph graph;
        graph.custom("fakesrc name=source_duplicate ! identity name=source_duplicate ! fakesink",
                     InputRole::Source);
        bool threw = false;
        try {
          Run run = graph.build();
          run.close();
        } catch (const NeatError& error) {
          threw = true;
          require_pipeline_shape(error, "source build");
        }
        require(threw, "source Graph::build must validate names automatically");
      }

      {
        Graph app = make_connected_collision_graph();
        bool threw = false;
        try {
          Run run = app.build(std::vector<cv::Mat>{input});
          run.close();
        } catch (const NeatError& error) {
          threw = true;
          require_pipeline_shape(error, "seeded connected build");
        }
        require(threw, "seeded connected Graph::build must validate each eager segment");
      }

      {
        Graph app = make_connected_collision_graph();
        Run run = app.build();
        const bool pushed = run.push("raw", std::vector<cv::Mat>{input});
        require(!pushed, "lazy connected materialization should reject the colliding segment");

        Sample output;
        PullError pull_error;
        const PullStatus status = run.pull("classes", 250, output, &pull_error);
        require(status == PullStatus::Error,
                "lazy connected failure should be observable through structured pull");
        require(pull_error.code == error_codes::kPipelineShape,
                "lazy connected PullError should retain the build error code");
        require(pull_error.report.has_value() &&
                    pull_error.report->error_code == error_codes::kPipelineShape &&
                    !pull_error.report->pipeline_string.empty(),
                "lazy connected PullError should retain its GraphReport");
        run.close();
      }

      {
        const std::filesystem::path image_path =
            std::filesystem::temp_directory_path() /
            ("neat_rtsp_name_collision_" + std::to_string(::getpid()) + ".jpg");
        require(cv::imwrite(image_path.string(), input), "failed to create RTSP image fixture");

        Graph graph;
        graph.add(nodes::StillImageInput(image_path.string(), 8, 8, 8, 8, 5));
        graph.custom("identity name=rtsp_h264_capsfix ! rtph264pay name=pay0");
        bool threw = false;
        try {
          RtspServerOptions options;
          options.mount = "name-collision";
          options.port = 18554;
          RtspServerHandle handle = graph.run_rtsp(options);
          handle.stop();
        } catch (const NeatError& error) {
          threw = true;
          require_pipeline_shape(error, "RTSP precheck");
          require_contains(error.what(), "RTSP helper",
                           "RTSP collision diagnostics should identify framework provenance");
          require_contains(error.what(), "Node fragment",
                           "RTSP collision diagnostics should identify Node provenance");
        }
        std::filesystem::remove(image_path);
        require(threw,
                "RTSP must reject explicit collisions before creating its server thread; launch=" +
                    graph.last_pipeline());
        require_contains(graph.last_pipeline(), "rtsp_h264_capsfix",
                         "failed RTSP validation should preserve the final launch string");
      }
    }));
