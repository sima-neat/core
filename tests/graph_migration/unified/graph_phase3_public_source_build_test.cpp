#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/Graph.h"
#include "test_main.h"
#include "test_utils.h"

#if defined(SIMA_WITH_OPENCV)
#include <opencv2/core.hpp>
#endif

RUN_TEST("graph_migration_phase3_public_source_build_test", [] {
  simaai::neat::Graph graph;
  graph.custom("videotestsrc num-buffers=3 is-live=false pattern=black ! "
               "video/x-raw,format=RGB,width=32,height=24,framerate=1/1",
               simaai::neat::InputRole::Source);
  graph.add(simaai::neat::nodes::Output());

  simaai::neat::Run run = graph.build();
  auto out = run.pull(5000);
  require(out.has_value(), "public Graph source build produced no sample");
  require(out->kind != simaai::neat::SampleKind::Unknown,
          "public Graph source build returned unknown sample");

  run.stop();

  simaai::neat::Graph source;
  source.custom("videotestsrc num-buffers=3 is-live=false pattern=smpte ! "
                "video/x-raw,format=RGB,width=32,height=24,framerate=1/1",
                simaai::neat::InputRole::Source);

  simaai::neat::Graph sink;
  sink.add(simaai::neat::nodes::Output());

  simaai::neat::Graph app;
  app.connect(source, sink);

  simaai::neat::Run connected = app.build();
  auto connected_out = connected.pull(5000);
  require(connected_out.has_value(), "public Graph connect build produced no sample");
  require(connected_out->kind != simaai::neat::SampleKind::Unknown,
          "public Graph connect build returned unknown sample");

  connected.stop();

  simaai::neat::Graph push_input;
  push_input.add(simaai::neat::nodes::Input());
  simaai::neat::Graph push_sink;
  push_sink.add(simaai::neat::nodes::Output());
  simaai::neat::Graph push_app;
  push_app.connect(push_input, push_sink);

  simaai::neat::Run push_run = push_app.build();
  const simaai::neat::Tensor push_tensor =
      make_color_tensor(32, 24, simaai::neat::ImageSpec::PixelFormat::RGB, 0x5A);
  require(push_run.push(simaai::neat::TensorList{push_tensor}),
          "connected Graph default TensorList push should succeed");
  auto pushed_out = push_run.pull(5000);
  require(pushed_out.has_value(), "connected Graph default TensorList push produced no output");
  push_run.stop();

  simaai::neat::Run seeded_tensor_build = push_app.build(simaai::neat::TensorList{push_tensor});
  const simaai::neat::Tensor seeded_push_tensor =
      make_color_tensor(32, 24, simaai::neat::ImageSpec::PixelFormat::RGB, 0x61);
  require(seeded_tensor_build.push(simaai::neat::TensorList{seeded_push_tensor}),
          "connected Graph seeded TensorList build should accept a later push");
  auto seeded_tensor_out = seeded_tensor_build.pull(5000);
  require(seeded_tensor_out.has_value(),
          "connected Graph seeded TensorList build produced no later output");
  seeded_tensor_build.stop();

  const simaai::neat::Sample seed_sample =
      simaai::neat::sample_from_tensors(simaai::neat::TensorList{push_tensor});
  simaai::neat::Run seeded_sample_build = push_app.build(simaai::neat::Sample{seed_sample});
  simaai::neat::Sample later_sample =
      simaai::neat::sample_from_tensors(simaai::neat::TensorList{seeded_push_tensor});
  require(seeded_sample_build.push(simaai::neat::Sample{later_sample}),
          "connected Graph seeded Sample build should accept a later push");
  auto seeded_sample_out = seeded_sample_build.pull(5000);
  require(seeded_sample_out.has_value(),
          "connected Graph seeded Sample build produced no later output");
  seeded_sample_build.stop();

  simaai::neat::Run run_once = push_app.build();
  const simaai::neat::Tensor run_tensor =
      make_color_tensor(32, 24, simaai::neat::ImageSpec::PixelFormat::RGB, 0x33);
  const simaai::neat::TensorList run_tensors =
      run_once.run(simaai::neat::TensorList{run_tensor}, 5000);
  require(!run_tensors.empty(), "connected Graph default TensorList run() produced no tensors");
  run_once.stop();

  const simaai::neat::Tensor graph_run_tensor =
      make_color_tensor(32, 24, simaai::neat::ImageSpec::PixelFormat::RGB, 0x44);
  const simaai::neat::TensorList graph_run_tensors =
      push_app.run(simaai::neat::TensorList{graph_run_tensor});
  require(!graph_run_tensors.empty(),
          "connected Graph::run(TensorList) default path produced no tensors");

#if defined(SIMA_WITH_OPENCV)
  simaai::neat::Graph cv_input;
  cv_input.add(simaai::neat::nodes::Input());
  simaai::neat::Graph cv_sink;
  cv_sink.add(simaai::neat::nodes::Output());
  simaai::neat::Graph cv_app;
  cv_app.connect(cv_input, cv_sink);

  simaai::neat::Run cv_run = cv_app.build();
  const cv::Mat cv_frame(24, 32, CV_8UC3, cv::Scalar(7, 31, 59));
  require(cv_run.push(std::vector<cv::Mat>{cv_frame}),
          "connected Graph default cv::Mat push should succeed");
  auto cv_out = cv_run.pull(5000);
  require(cv_out.has_value(), "connected Graph default cv::Mat push produced no output");
  cv_run.stop();

  simaai::neat::Run seeded_cv_build = cv_app.build(std::vector<cv::Mat>{cv_frame});
  const cv::Mat later_cv_frame(24, 32, CV_8UC3, cv::Scalar(19, 73, 127));
  require(seeded_cv_build.push(std::vector<cv::Mat>{later_cv_frame}),
          "connected Graph seeded cv::Mat build should accept a later push");
  auto seeded_cv_out = seeded_cv_build.pull(5000);
  require(seeded_cv_out.has_value(), "connected Graph seeded cv::Mat build produced no output");
  seeded_cv_build.stop();

  simaai::neat::Run cv_once = cv_app.build();
  const cv::Mat cv_once_frame(24, 32, CV_8UC3, cv::Scalar(11, 47, 83));
  const simaai::neat::TensorList cv_once_tensors =
      cv_once.run(std::vector<cv::Mat>{cv_once_frame}, 5000);
  require(!cv_once_tensors.empty(), "connected Graph default cv::Mat run() produced no tensors");
  cv_once.stop();

  const cv::Mat cv_graph_frame(24, 32, CV_8UC3, cv::Scalar(17, 23, 101));
  const simaai::neat::TensorList cv_graph_tensors =
      cv_app.run(std::vector<cv::Mat>{cv_graph_frame});
  require(!cv_graph_tensors.empty(),
          "connected Graph::run(cv::Mat) default path produced no tensors");
#endif

  simaai::neat::Graph fragment_source;
  fragment_source.custom("videotestsrc num-buffers=3 is-live=false pattern=white ! "
                         "video/x-raw,format=RGB,width=32,height=24,framerate=1/1",
                         simaai::neat::InputRole::Source);

  simaai::neat::Graph fragment_sink;
  fragment_sink.add(simaai::neat::nodes::Output());

  simaai::neat::Graph spliced;
  spliced.add(fragment_source);
  spliced.add(std::move(fragment_sink));

  simaai::neat::Run spliced_run = spliced.build();
  auto spliced_out = spliced_run.pull(5000);
  require(spliced_out.has_value(), "public Graph add(Graph) build produced no sample");
  require(spliced_out->kind != simaai::neat::SampleKind::Unknown,
          "public Graph add(Graph) build returned unknown sample");

  spliced_run.stop();

  simaai::neat::Graph source_a;
  source_a.custom("videotestsrc num-buffers=1 is-live=false pattern=black ! "
                  "video/x-raw,format=RGB,width=16,height=16,framerate=1/1",
                  simaai::neat::InputRole::Source);
  simaai::neat::Graph source_b;
  source_b.custom("videotestsrc num-buffers=1 is-live=false pattern=white ! "
                  "video/x-raw,format=RGB,width=16,height=16,framerate=1/1",
                  simaai::neat::InputRole::Source);
  simaai::neat::Graph sink_a;
  sink_a.add(simaai::neat::nodes::Output());
  simaai::neat::Graph sink_b;
  sink_b.add(simaai::neat::nodes::Output());

  simaai::neat::Graph ambiguous_outputs;
  ambiguous_outputs.connect(source_a, sink_a);
  ambiguous_outputs.connect(source_b, sink_b);
  simaai::neat::Run ambiguous_output_run = ambiguous_outputs.build();
  simaai::neat::Sample ignored;
  simaai::neat::PullError pull_error;
  const simaai::neat::PullStatus pull_status = ambiguous_output_run.pull(1, ignored, &pull_error);
  require(pull_status == simaai::neat::PullStatus::Error,
          "ambiguous connected Graph output should fail closed");
  require_contains(pull_error.message, "no unambiguous default output",
                   "ambiguous output diagnostic should name default output ambiguity");
  ambiguous_output_run.stop();

  simaai::neat::Graph input_a;
  input_a.add(simaai::neat::nodes::Input());
  simaai::neat::Graph input_b;
  input_b.add(simaai::neat::nodes::Input());
  simaai::neat::Graph ambiguous_inputs;
  ambiguous_inputs.connect(input_a, sink_a);
  ambiguous_inputs.connect(input_b, sink_b);
  simaai::neat::Run ambiguous_input_run = ambiguous_inputs.build();
  bool push_threw = false;
  try {
    const simaai::neat::Sample ambiguous_input_sample =
        simaai::neat::sample_from_tensors(simaai::neat::TensorList{push_tensor});
    (void)ambiguous_input_run.push(ambiguous_input_sample);
  } catch (const std::exception& e) {
    push_threw = true;
    require_contains(std::string(e.what()), "no unambiguous default input",
                     "ambiguous input diagnostic should name default input ambiguity");
  }
  require(push_threw, "ambiguous connected Graph input should fail closed");
  ambiguous_input_run.stop();
});
