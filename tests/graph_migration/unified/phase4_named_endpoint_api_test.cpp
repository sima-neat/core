#include "graph_migration/common/phase3_graph_test_utils.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "test_main.h"
#include "test_utils.h"

#include <algorithm>
#include <string>
#include <vector>

#if defined(SIMA_WITH_OPENCV)
#include <opencv2/core.hpp>
#endif

namespace {

bool contains_name(const std::vector<std::string>& names, const std::string& name) {
  return std::find(names.begin(), names.end(), name) != names.end();
}

simaai::neat::Graph named_input_graph(const std::string& name) {
  simaai::neat::Graph g;
  g.add(simaai::neat::nodes::Input(name));
  return g;
}

simaai::neat::Graph named_output_graph(const std::string& name) {
  simaai::neat::Graph g;
  g.add(simaai::neat::nodes::Output(name));
  return g;
}

void require_names(const simaai::neat::Run& run, const std::string& input,
                   const std::string& output, const std::string& where) {
  const std::vector<std::string> inputs = run.input_names();
  const std::vector<std::string> outputs = run.output_names();
  require(contains_name(inputs, input), where + ": missing input endpoint name " + input);
  require(contains_name(outputs, output), where + ": missing output endpoint name " + output);
}

} // namespace

RUN_TEST("graph_migration_phase4_named_endpoint_api_test", [] {
  {
    simaai::neat::Graph app("linear_app");
    app.add(simaai::neat::nodes::Input("image"));
    app.add(simaai::neat::nodes::Output("classes"));

    simaai::neat::Run run = app.build();
    require_names(run, "image", "classes", "linear named endpoints");
    require(run.push("image",
                     simaai::neat::TensorList{graph_phase3_test::make_rgb_tensor(32, 24, 0x41)}),
            "linear named TensorList push failed");
    simaai::neat::TensorList tensors = run.pull_tensors("classes", 5000);
    graph_phase3_test::require_nonempty_tensor_output(tensors, "linear named pull_tensors");
    run.close();
  }

  {
    auto input = named_input_graph("image");
    auto output = named_output_graph("classes");
    simaai::neat::Graph app;
    app.connect(input, output);

    simaai::neat::Run run = app.build();
    require_names(run, "image", "classes", "connected named endpoints");
    require(run.try_push("image", simaai::neat::TensorList{graph_phase3_test::make_rgb_tensor(
                                      32, 24, 0x42)}),
            "connected named try_push(TensorList) failed");
    simaai::neat::TensorList tensors = run.pull_tensors("classes", 5000);
    graph_phase3_test::require_nonempty_tensor_output(
        tensors, "connected named pull_tensors after try_push");

    require(run.push("image", simaai::neat::Sample{graph_phase3_test::make_tensor_sample(
                                  7, "named-sample")}),
            "connected named Sample push failed");
    simaai::neat::Sample samples = run.pull_samples("classes", 5000);
    require(samples.size() == 1U, "connected named pull_samples expected one sample");
    require(samples.front().stream_id == "named-sample",
            "connected named pull_samples stream_id mismatch");
    run.close();
  }

  {
    auto input = named_input_graph("image");
    simaai::neat::Graph outputs("classes");
    outputs.add(simaai::neat::nodes::Output());
    outputs.add(simaai::neat::nodes::Output());
    outputs.add(simaai::neat::nodes::Output());

    simaai::neat::Graph app;
    app.connect(input, outputs);

    simaai::neat::Run run = app.build();
    require_names(run, "image", "classes_0", "multi-output named fragment");
    require(contains_name(run.output_names(), "classes_1"),
            "multi-output named fragment missing classes_1");
    require(contains_name(run.output_names(), "classes_2"),
            "multi-output named fragment missing classes_2");
    require(!contains_name(run.output_names(), "classes"),
            "multi-output named fragment should expose suffixed endpoint names only");

    require(run.push("image",
                     simaai::neat::Sample{graph_phase3_test::make_tensor_sample(9, "auto-suffix")}),
            "multi-output named fragment push failed");
    const std::vector<std::string> suffixed_outputs({"classes_0", "classes_1", "classes_2"});
    for (const std::string& name : suffixed_outputs) {
      auto sample = run.pull(name, 5000);
      require(sample.has_value(), "multi-output named fragment pull timed out for " + name);
      require(sample->stream_id == "auto-suffix",
              "multi-output named fragment stream_id mismatch for " + name);
    }
    run.close();
  }

#if defined(SIMA_WITH_OPENCV)
  {
    auto input = named_input_graph("frame");
    auto output = named_output_graph("pixels");
    simaai::neat::Graph app;
    app.connect(input, output);

    simaai::neat::Run run = app.build();
    const cv::Mat mat(24, 32, CV_8UC3, cv::Scalar(3, 5, 7));
    require(run.push("frame", std::vector<cv::Mat>{mat}), "connected named cv::Mat push failed");
    simaai::neat::TensorList tensors = run.pull_tensors("pixels", 5000);
    graph_phase3_test::require_nonempty_tensor_output(tensors, "connected named cv::Mat output");
    run.close();
  }
#endif
});
